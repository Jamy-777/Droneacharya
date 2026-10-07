"""Acoustic datasets -> artifact / capture / group index.

  dataset      artifact                       capture                         group
  esc50        5 s clip                       clip                            Freesound source recording (src_file)
  svanstrom    10 s clip                      clip                            clip; byte-identical clips share one
  uavirbase    8-channel recording            recording                       recording
  dronenoise   one microphone's WAV           event (all its microphones)     event
  miesikowska  one mono WAV                   file                            drone + position (its takes are repeats)
  ddl          0.1 s 8-channel clip           gap-free run of a segment's clips  flight day + recording segment

Every WAV header is read from its archive (first bytes only). Acoustic labels:
label_uas_present = a drone is the sound source; label_emitter = aircraft or none.
"""
import csv
import io
import json
import re
import struct
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath

from ..paths import MANIFESTS, RAW

FORMATS = {1: "pcm", 3: "float", 0xFFFE: "extensible"}


def wav_header(raw):
    """RIFF/WAVE header fields from the first bytes of a file; {"valid": False} if it is not one."""
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return {"valid": False}
    pos, info = 12, {"valid": True}
    while pos + 8 <= len(raw):
        chunk, size = raw[pos:pos + 4], struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = raw[pos + 8:pos + 8 + min(size, 40)]
        if chunk == b"fmt ":
            fmt, channels, rate = struct.unpack("<HHI", body[:8])
            bits = struct.unpack("<H", body[14:16])[0]
            if fmt == 0xFFFE and len(body) >= 26:
                fmt = struct.unpack("<H", body[24:26])[0]
            info.update(format=FORMATS.get(fmt, str(fmt)), channels=channels, sample_rate=rate, bits=bits)
        elif chunk == b"data":
            info["data_bytes"] = size
            break
        pos += 8 + size + (size & 1)
    if info.get("data_bytes") is not None and info.get("channels") and info.get("bits"):
        info["duration_s"] = info["data_bytes"] / (info["channels"] * info["bits"] // 8) / info["sample_rate"]
    return info


def _zip_header(zf, info, n=4096):
    with zf.open(info) as fh:
        return wav_header(fh.read(n))


def _artifact(dataset, artifact_id, storage, chain, size, crc, mtime, capture_id, channel=None, part=None):
    return {"artifact_id": f"{dataset}/{artifact_id}", "dataset_id": dataset, "storage_relpath": storage,
            "member_chain": chain, "size_bytes": size, "crc32": crc, "member_mtime": mtime, "media_type": "audio/wav",
            "capture_id": capture_id, "channel": channel, "part_index": part}


def _capture(dataset, capture_id, group_id, n, header, duration, present, cls, condition, source, *,
             channels=(), recorded_at=None, split=None, start=None):
    return {"capture_id": capture_id, "dataset_id": dataset, "group_id": group_id, "n_artifacts": n,
            "channels": list(channels), "recorded_at": recorded_at, "start_s": start, "duration_s": duration,
            "center_frequency_hz": None, "sample_rate_hz": float(header["sample_rate"]), "reference_snr_db": None,
            "official_split": split, "label_uas_present": present,
            "label_emitter": None if present is None else ("aircraft" if present else "none"),
            "label_class": cls, "label_condition": condition, "label_source": source, "label_status": "AUTHOR_DOC"}


def _group(dataset, group_id, kind, basis, status, n):
    return {"group_id": group_id, "dataset_id": dataset, "kind": kind, "basis": basis, "status": status, "n_captures": n}


def _ev(entity, field, value, status, source, n=None):
    return {"entity_id": entity, "field": field, "value": str(value), "status": status, "source": source, "n": n}


def _mtime(info):
    return "{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}".format(*info.date_time)


def _groups_from(captures, dataset, kind, basis, status):
    counts = defaultdict(int)
    for c in captures:
        counts[c["group_id"]] += 1
    return [_group(dataset, g, kind, basis, status, n) for g, n in sorted(counts.items())]


def _formats(headers):
    return "; ".join(sorted({f"{h['sample_rate']} Hz {h['bits']}-bit {h['format']} {h['channels']} ch" for h in headers}))


# ---------------------------------------------------------------- ESC-50
def build_esc50(raw_root=RAW):
    ds, rel = "esc50", "esc50/original/ESC-50-master.zip"
    artifacts, captures, headers = [], [], []
    with zipfile.ZipFile(raw_root / rel) as zf:
        meta = {r["filename"]: r for r in csv.DictReader(io.StringIO(zf.read("ESC-50-master/meta/esc50.csv").decode()))}
        for info in zf.infolist():
            if not info.filename.startswith("ESC-50-master/audio/") or not info.filename.endswith(".wav"):
                continue
            name = PurePosixPath(info.filename).name
            h = _zip_header(zf, info)
            headers.append(h)
            m = meta[name]
            capture_id = f"{ds}/audio/{name.removesuffix('.wav')}"
            artifacts.append(_artifact(ds, info.filename, rel, [info.filename], info.file_size, f"{info.CRC:08x}",
                                       _mtime(info), capture_id))
            captures.append(_capture(ds, capture_id, f"{ds}/src_file/{m['src_file']}", 1, h, h["duration_s"], False,
                                     m["category"], None, "meta/esc50.csv", split=f"fold{m['fold']}"))
    groups = _groups_from(captures, ds, "src_file", "clips cut from one Freesound recording", "AUTHOR_DOC")
    evidence = [_ev(ds, "clips", len(captures), "VERIFIED", "zip listing + meta/esc50.csv", len(captures)),
                _ev(ds, "formats", _formats(headers), "VERIFIED", "WAV headers", len(headers))]
    return {"artifacts": artifacts, "captures": captures, "groups": groups, "lineage": [], "evidence": evidence}


# ---------------------------------------------------------------- Svanström
def build_svanstrom(raw_root=RAW, manifests=MANIFESTS):
    ds, folder = "svanstrom", "svanstrom/original/Audio"
    with open(manifests / "raw_sha256.csv", newline="", encoding="utf-8") as fh:
        sha = {r["relpath"]: r["sha256"] for r in csv.DictReader(fh)}
    first_with_digest, artifacts, captures, headers, lineage = {}, [], [], [], []
    for path in sorted((raw_root / folder).glob("*.wav")):
        rel = f"{folder}/{path.name}"
        h = wav_header(path.read_bytes()[:4096])
        headers.append(h)
        kind = path.stem.split("_")[0]
        digest = sha[rel]
        group_stem = first_with_digest.setdefault(digest, path.stem)
        if group_stem != path.stem:
            lineage.append({"subject": rel, "relation": "identical_content_to", "object": f"{folder}/{group_stem}.wav",
                            "method": "SHA-256 equal; also one blob in the upstream GitHub repository", "status": "VERIFIED"})
        capture_id = f"{ds}/Data/Audio/{path.stem}"
        artifacts.append(_artifact(ds, f"Data/Audio/{path.name}", rel, [], path.stat().st_size, None, None, capture_id))
        captures.append(_capture(ds, capture_id, f"{ds}/clip/{group_stem}", 1, h, h["duration_s"], kind == "DRONE",
                                 kind.lower(), None, "file name prefix (repository README)"))
    groups = _groups_from(captures, ds, "clip", "one clip; byte-identical clips share a group; event grouping unknown",
                          "VERIFIED")
    evidence = [_ev(ds, "clips", len(captures), "VERIFIED", "local files", len(captures)),
                _ev(ds, "formats", _formats(headers), "VERIFIED", "WAV headers", len(headers))]
    return {"artifacts": artifacts, "captures": captures, "groups": groups, "lineage": lineage, "evidence": evidence}


# ---------------------------------------------------------------- UaVirBASE
def build_uavirbase(raw_root=RAW):
    ds, rel = "uavirbase", "uavirbase/original/Microphone_array.zip"
    artifacts, captures, headers, evidence = [], [], [], []
    with zipfile.ZipFile(raw_root / rel) as zf:
        for info in zf.infolist():
            if not info.filename.endswith("/output.wav"):
                continue
            folder = info.filename.rsplit("/", 1)[0]
            label = json.loads(zf.read(f"{folder}/label.json"))
            drone = label["drone"]
            h = _zip_header(zf, info)
            headers.append(h)
            is_drone = drone["sound_source"] == "Drone"
            capture_id = f"{ds}/{folder}"
            artifacts.append(_artifact(ds, info.filename, rel, [info.filename], info.file_size, f"{info.CRC:08x}",
                                       _mtime(info), capture_id))
            condition = (f"distance {drone['distance']} m, height {drone['height']} m, azimuth {drone['azimuth']}, "
                         f"{drone['rotation']}, {drone['movement']}") if is_drone else "ambient"
            captures.append(_capture(ds, capture_id, capture_id, 1, h, h["duration_s"], is_drone,
                                     drone["type"] if is_drone else None, condition, "label.json",
                                     channels=[str(c) for c in range(1, h["channels"] + 1)],
                                     recorded_at=label["start_recording_time"].replace(" ", "T")))
            wind = label.get("weather_data", {}).get("measurements", {}).get("wind speed")
            evidence.append(_ev(capture_id, "label.wind_speed", wind, "AUTHOR_DOC", "label.json weather_data"))
    groups = _groups_from(captures, ds, "recording", "one recording; only 4 ambient recordings exist", "AUTHOR_DOC")
    evidence += [_ev(ds, "recordings", len(captures), "VERIFIED", "zip listing", len(captures)),
                 _ev(ds, "ambient_recordings", sum(not c["label_uas_present"] for c in captures), "VERIFIED", "label.json", len(captures)),
                 _ev(ds, "formats", _formats(headers), "VERIFIED", "WAV headers", len(headers))]
    return {"artifacts": artifacts, "captures": captures, "groups": groups, "lineage": [], "evidence": evidence}


# ---------------------------------------------------------------- DroneNoise
DRONENOISE = re.compile(r"^(?:(?P<fileid>\d+)_)?Ed_(?P<drone>\w+?)_(?P<height>\d+)_(?P<op>[FH]\d\d)_(?P<rest>.+)_ev(?P<ev>\d)_M(?P<mic>\d)\.wav$")
CALIB = re.compile(r"^Calib_Scotland_ch(?P<ch>\d)\.wav$")


def build_dronenoise(raw_root=RAW):
    ds, rel = "dronenoise", "dronenoise/original/22133411.zip"
    parts, headers, artifacts, captures, evidence, lineage = defaultdict(list), [], [], [], [], []
    with zipfile.ZipFile(raw_root / rel) as zf:
        for info in zf.infolist():
            name = info.filename
            if not name.endswith(".wav"):
                continue
            h = _zip_header(zf, info)
            headers.append(h)
            if m := CALIB.match(name):
                key = ("calibration", f"Calib_Scotland_ch{m['ch']}", None)
            elif m := DRONENOISE.match(name):
                stem = name.rsplit("_M", 1)[0]
                key = ("event", stem, m)
            else:
                raise ValueError(f"DroneNoise: unexpected file {name}")
            parts[key[:2]].append((name, info, h, m))
        crcs = defaultdict(list)
        for (kind, stem), files in sorted(parts.items()):
            capture_id = f"{ds}/{stem}"
            if kind == "calibration":
                (name, info, h, _), = files
                artifacts.append(_artifact(ds, name, rel, [name], info.file_size, f"{info.CRC:08x}", _mtime(info), capture_id))
                captures.append(_capture(ds, capture_id, f"{ds}/calibration", 1, h, h["duration_s"], None,
                                         "calibration_tone", f"channel {stem[-1]}", "file name"))
                continue
            m = files[0][3]
            event = f"{ds}/Ed_{m['drone']}_{m['height']}_{m['op']}_{m['rest']}_ev{m['ev']}"
            mics = sorted(files, key=lambda f: int(f[3]["mic"]))
            for name, info, h, mm in mics:
                crcs[(info.file_size, info.CRC)].append(name)
                artifacts.append(_artifact(ds, name, rel, [name], info.file_size, f"{info.CRC:08x}", _mtime(info),
                                           capture_id, channel=f"M{mm['mic']}"))
            h = mics[0][2]
            captures.append(_capture(ds, capture_id, event, len(mics), h, min(f[2]["duration_s"] for f in mics), True,
                                     m["drone"], f"{'flyover 15 m/s' if m['op'] == 'F15' else 'hover'}, {m['height']} m",
                                     "file name (drone code, height, operation, event)",
                                     channels=[f"M{f[3]['mic']}" for f in mics]))
            if len(mics) != 9:
                evidence.append(_ev(capture_id, "microphones_present", ",".join(f"M{f[3]['mic']}" for f in mics),
                                    "VERIFIED", "zip listing"))
        for group in crcs.values():
            for copy in group[1:]:
                lineage.append({"subject": f"{rel}::{copy}", "relation": "identical_content_to", "object": f"{rel}::{group[0]}",
                                "method": "size + CRC32 in the zip central directory", "status": "VERIFIED"})
    groups = _groups_from(captures, ds, "event", "one flyover or hover event recorded by up to 9 microphones; "
                          "calibration tones form their own group", "AUTHOR_DOC")
    evidence += [_ev(ds, "events", sum(c["label_uas_present"] is True for c in captures), "VERIFIED", "zip listing"),
                 _ev(ds, "formats", _formats(headers), "VERIFIED", "WAV headers", len(headers))]
    return {"artifacts": artifacts, "captures": captures, "groups": groups, "lineage": lineage, "evidence": evidence}


# ---------------------------------------------------------------- Mięsikowska
MIES = re.compile(r"^x4_(?P<drone>d\d+)_(?P<model>[a-z0-9]+)_(?P<alt>\d+)m_(?P<off>\d+)m_(?P<date>\d{6})_(?P<take>\d+)(?P<rest>.*)\.wav$", re.I)


def build_miesikowska(raw_root=RAW):
    ds, folder = "miesikowska_uav", "miesikowska_uav/original"
    artifacts, captures, headers, other = [], [], [], 0
    for zip_path in sorted((raw_root / folder).glob("*.zip")):
        rel = f"{folder}/{zip_path.name}"
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if not info.filename.lower().endswith(".wav"):
                    other += 1
                    continue
                name = PurePosixPath(info.filename).name
                m = MIES.match(name)
                if not m:
                    raise ValueError(f"Mięsikowska: unexpected file {info.filename}")
                h = _zip_header(zf, info)
                headers.append(h)
                speech = "sekwencja" in m["rest"].lower()
                capture_id = f"{ds}/{info.filename.rsplit('.', 1)[0]}"
                artifacts.append(_artifact(ds, f"{zip_path.name}/{info.filename}", rel, [info.filename], info.file_size,
                                           f"{info.CRC:08x}", _mtime(info), capture_id))
                date = datetime.strptime(m["date"], "%y%m%d").date().isoformat()
                captures.append(_capture(ds, capture_id, f"{ds}/{m['drone']}/{m['alt']}m_{m['off']}m", 1, h, h["duration_s"],
                                         True, m["model"], f"altitude {m['alt']} m, offset {m['off']} m"
                                         + (", speech commands" if speech else ""), "file name", recorded_at=date))
    groups = _groups_from(captures, ds, "drone_position", "one drone at one position: its takes are repeats in one session",
                          "INFERRED")
    evidence = [_ev(ds, "recordings", len(captures), "VERIFIED", "zip listings (4 of 17 drone archives)", len(captures)),
                _ev(ds, "non_audio_members", other, "VERIFIED", "zip listings (SLM .NBF/.xlsx, photos, video)"),
                _ev(ds, "formats", _formats(headers), "VERIFIED", "WAV headers", len(headers))]
    return {"artifacts": artifacts, "captures": captures, "groups": groups, "lineage": [], "evidence": evidence}


# ---------------------------------------------------------------- DDL
DDL = re.compile(r"^(?P<ts>\d{14})(?P<cls>[A-Z0-9]{4})(?P<bearing>\d{3})(?P<range>\d{3})(?P<alt>\d{3})(?P<temp>\d{4})"
                 r"(?P<rs>[RS])(?P<session>\d{6})-(?P<segment>T\d{3})-(?P<seq>\d{6})\.wav$")
DDL_CLASSES = {"MINI": "DJI Mini 2", "PRO4": "DJI Phantom 4 Pro"}


def build_ddl(raw_root=RAW):
    ds, rel = "ddl", "ddl/original/MLSP_2022_Real_Data.zip"
    artifacts, captures, evidence, headers = [], [], [], []
    clips = defaultdict(list)  # segment folder -> [(seq, name, info, header, match)]
    unusable = {"empty": 0, "name": 0}
    with zipfile.ZipFile(raw_root / rel) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            folder, name = info.filename.rsplit("/", 1)
            m = DDL.match(name)
            h = _zip_header(zf, info)
            empty = info.file_size <= 44  # header only
            if not empty and not h.get("data_bytes"):
                raise ValueError(f"DDL: {info.filename} has data but no readable data chunk")
            if m is None or empty:
                unusable["name" if m is None else "empty"] += 1
                artifacts.append(_artifact(ds, info.filename, rel, [info.filename], info.file_size, f"{info.CRC:08x}",
                                           _mtime(info), None))
                continue
            headers.append(h)
            clips[folder].append((int(m["seq"]), info, h, m))
    for folder, items in sorted(clips.items()):
        items.sort(key=lambda item: item[0])
        runs = []
        for item in items:
            (runs[-1].append(item) if runs and item[0] == runs[-1][-1][0] + 1 else runs.append([item]))
        for run in runs:
            classes = {DDL_CLASSES.get(i[3]["cls"], i[3]["cls"]) for i in run}
            capture_id = f"{ds}/{folder}/{run[0][0]:06d}-{run[-1][0]:06d}"
            for part, (_, info, _, _) in enumerate(run):
                artifacts.append(_artifact(ds, info.filename, rel, [info.filename], info.file_size, f"{info.CRC:08x}",
                                           _mtime(info), capture_id, part=part))
            first = run[0][3]
            captures.append(_capture(ds, capture_id, f"{ds}/{folder}", len(run), run[0][2],
                                     sum(i[2]["duration_s"] for i in run), True, " / ".join(sorted(classes)), None,
                                     "file name class code (per clip: bearing, range, altitude in interim/ddl/ddl_filename_index.csv)",
                                     channels=[str(c) for c in range(1, run[0][2]["channels"] + 1)],
                                     recorded_at=datetime.strptime(first["ts"], "%Y%m%d%H%M%S").isoformat()))
    groups = _groups_from(captures, ds, "segment", "one recording segment on one flight day", "AUTHOR_DOC")
    evidence += [_ev(ds, "clips_in_captures", len(headers), "VERIFIED", "WAV headers", len(headers)),
                 _ev(ds, "empty_clips_without_capture", unusable["empty"], "VERIFIED", "WAV headers (no data)"),
                 _ev(ds, "unparseable_names_without_capture", unusable["name"], "VERIFIED", "file name grammar"),
                 _ev(ds, "formats", _formats(headers), "VERIFIED", "WAV headers", len(headers))]
    return {"artifacts": artifacts, "captures": captures, "groups": groups, "lineage": [], "evidence": evidence}
