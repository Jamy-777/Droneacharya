"""CardRF (raw CARDRF.zip, 9,600 MAT v7.3 captures) -> artifact / capture / group index.

Release layout: CARDRF/<LOS|NLOS>/[<Train|Test>/]<category>/<device>[/<flight mode>]/<file>.mat
with category UAV, UAV_Controller, WIFI or BLUETOOTH; 500 zero-byte __MACOSX
entries are skipped.

  artifact = one .mat member (one triggered 250 us oscilloscope capture)
  capture  = the artifact
  group    = UAS system (aircraft and its controller together) or non-UAS
             device, LOS and NLOS together: one physical unit per model
             (INFERRED from the authors' Table I), so "unseen UAS" must hold out
             the aircraft and controller of a system at once; Train/Test are
             interleaved inside the same recording block, so they are not
             independent either

Per-capture Frame.Date, NumPoints and XInc are read from every file (HDF5 inside
the zip, small datasets only) and cached in interim/cardrf/frame_fields.json.
"""
import json
import re
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import PurePosixPath

import h5py

from ..paths import INTERIM, RAW

DATASET = "cardrf"
ZIP_RELPATH = "cardrf/original/CARDRF.zip"
PROCESSED_RELPATH = "cardrf/original/Processed_CardRF.zip"
EMITTER = {"UAV": "aircraft", "UAV_Controller": "controller", "WIFI": "wifi", "BLUETOOTH": "bluetooth"}
BLOCK_GAP = timedelta(hours=1)  # a longer pause between consecutive captures of a device starts a new block


def _text(dataset):
    return "".join(map(chr, dataset[()].flatten()))


def _frame_fields(zip_path, cache_path):
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    with zipfile.ZipFile(zip_path) as zf:
        todo = [i for i in zf.infolist() if _is_capture(i) and cache.get(i.filename, {}).get("crc") != i.CRC]
        for count, info in enumerate(todo, 1):
            with zf.open(info) as fh, h5py.File(fh, "r") as f:
                channel = f["Channel_1"]
                cache[info.filename] = {
                    "crc": info.CRC,
                    "date": _text(f["Frame"]["Date"]),
                    "model": _text(f["Frame"]["Model"]),
                    "serial": _text(f["Frame"]["Serial"]),
                    "num_points": int(channel["NumPoints"][()].flatten()[0]),
                    "x_inc": float(channel["XInc"][()].flatten()[0]),
                }
            if count % 500 == 0 or count == len(todo):
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
                print(f"  Frame fields read: {count}/{len(todo)}", flush=True)
    return cache


def _is_capture(info):
    return not info.is_dir() and info.filename.endswith(".mat") and "__MACOSX" not in info.filename


def _parse_date(text):
    return datetime.strptime(re.sub(r"\s+", " ", text.strip()), "%d-%b-%Y %H:%M:%S")


def _describe(path):
    parts = PurePosixPath(path).parts  # CARDRF, LOS|NLOS, [Train|Test], category, device, [mode], file
    los = parts[1]
    rest = list(parts[2:-1])
    split = rest.pop(0).lower() if rest and rest[0] in ("Train", "Test") else None
    category, device = rest[0], rest[1]
    mode = rest[2] if len(rest) > 2 else None
    if category not in EMITTER or len(rest) > 3:
        raise ValueError(f"unexpected CardRF path {path}")
    return los, split, category, device, mode


def build(raw_root=RAW, interim_root=INTERIM):
    zip_path = raw_root / ZIP_RELPATH
    frames = _frame_fields(zip_path, interim_root / DATASET / "frame_fields.json")
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if _is_capture(i)]
        junk = sum(1 for i in zf.infolist() if "__MACOSX" in i.filename)

    artifacts, captures, times = [], [], defaultdict(list)
    for info in infos:
        los, split, category, device, mode = _describe(info.filename)
        frame = frames[info.filename]
        recorded = _parse_date(frame["date"])
        capture_id = f"{DATASET}/{info.filename.removesuffix('.mat')}"
        group_id = f"{DATASET}/UAS/{device}" if category in ("UAV", "UAV_Controller") else f"{DATASET}/{category}/{device}"
        artifacts.append({
            "artifact_id": f"{DATASET}/{info.filename}",
            "dataset_id": DATASET,
            "storage_relpath": ZIP_RELPATH,
            "member_chain": [info.filename],
            "size_bytes": info.file_size,
            "crc32": f"{info.CRC:08x}",
            "member_mtime": "{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}".format(*info.date_time),
            "media_type": "application/x-matlab-data",
            "capture_id": capture_id,
            "channel": None,
            "part_index": None,
        })
        captures.append({
            "capture_id": capture_id,
            "dataset_id": DATASET,
            "group_id": group_id,
            "n_artifacts": 1,
            "channels": [],
            "recorded_at": recorded.isoformat(),
            "start_s": None,
            "duration_s": frame["num_points"] * frame["x_inc"],
            "center_frequency_hz": None,  # direct RF sampling
            "sample_rate_hz": 1.0 / frame["x_inc"],
            "reference_snr_db": None,
            "official_split": split,
            "label_uas_present": category in ("UAV", "UAV_Controller"),
            "label_emitter": EMITTER[category],
            "label_class": device,
            "label_condition": f"{los}/{mode}" if mode else los,
            "label_source": "release folders: LOS/NLOS, Train/Test, category, device, flight mode",
            "label_status": "AUTHOR_DOC",
        })
        times[group_id].append(recorded)

    groups, evidence = [], []
    for group_id, stamps in sorted(times.items()):
        stamps.sort()
        blocks = [[stamps[0]]]
        for previous, current in zip(stamps, stamps[1:]):
            (blocks.append([current]) if current - previous > BLOCK_GAP else blocks[-1].append(current))
        groups.append({
            "group_id": group_id,
            "dataset_id": DATASET,
            "kind": "uas_system" if "/UAS/" in group_id else "device",
            "basis": "one physical unit per model (authors' Table I); a UAS system's aircraft and controller together; "
                     "Train/Test captures interleaved within the recording blocks (Frame.Date of every capture)",
            "status": "INFERRED",
            "n_captures": len(stamps),
        })
        evidence.append(_ev(group_id, "recording_blocks",
                            "; ".join(f"{b[0].isoformat(sep=' ')} → {b[-1].isoformat(sep=' ')} ({len(b)})" for b in blocks),
                            "VERIFIED", "Frame.Date of every capture", len(stamps)))

    scopes = {(f["model"], f["serial"]) for f in frames.values()}
    rates = {round(1 / f["x_inc"]) for f in frames.values()}
    points = {f["num_points"] for f in frames.values()}
    hours = sorted({_parse_date(f["date"]).hour for f in frames.values()})
    evidence += [
        _ev(DATASET, "captures", len(captures), "VERIFIED", "zip central directory", len(captures)),
        _ev(DATASET, "macosx_junk_entries_skipped", junk, "VERIFIED", "zip central directory", None),
        _ev(DATASET, "instrument", "; ".join(f"{m} {s}" for m, s in sorted(scopes)), "VERIFIED",
            "Frame.Model / Frame.Serial of every capture", len(captures)),
        _ev(DATASET, "sample_rate_hz_values", ", ".join(str(r) for r in sorted(rates)), "VERIFIED",
            "1 / Channel_1.XInc of every capture", len(captures)),
        _ev(DATASET, "num_points_values", ", ".join(str(p) for p in sorted(points)), "VERIFIED",
            "Channel_1.NumPoints of every capture", len(captures)),
        _ev(DATASET, "recorded_at_meaning",
            f"scope clock as stored; hours present {hours}; 12/24-hour convention and time zone not established",
            "UNKNOWN", "Frame.Date of every capture", len(captures)),
    ]
    lineage = [{
        "subject": PROCESSED_RELPATH,
        "relation": "derived_from",
        "object": ZIP_RELPATH,
        "method": "authors' code + our check: 100 chained 1024-sample slices per LOS UAV/controller capture, "
                  "from sample 2,500,000; Wi-Fi, Bluetooth and NLOS omitted",
        "status": "EMPIRICAL",
    }]
    return {"artifacts": artifacts, "captures": captures, "groups": groups, "lineage": lineage, "evidence": evidence}


def _ev(entity, field, value, status, source, n):
    return {"entity_id": entity, "field": field, "value": str(value), "status": status, "source": source, "n": n}
