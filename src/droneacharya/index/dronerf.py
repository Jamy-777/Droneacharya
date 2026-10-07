"""DroneRF (Mendeley 10.17632/f4c2b4n755.1) -> artifact / capture / group index.

Release layout: one zip; inside it, one folder per class holding RAR archives;
inside each RAR, single-row CSVs named <BUI><L|H>_<segment>.csv.

  artifact = one CSV (zip member -> RAR member)
  capture  = the L and H CSVs of one BUI and segment (two simultaneous bands)
  group    = BUI: each flight mode is ~5.25 s cut into ~21 segments, likely one
             continuous recording (INFERRED), so its segments are not independent

The listing comes from RAR headers (name, size, CRC32, time); no CSV is
unpacked. Each RAR is copied out of the zip to a temporary file for UnRAR and
deleted afterwards; listings are cached in interim/dronerf/rar_listings.json.
"""
import json
import re
import shutil
import subprocess
import time
import zipfile
import zlib
from pathlib import Path, PurePosixPath

from ..paths import INTERIM, RAW
from .rar import UNRAR, parse_unrar_lt

DATASET = "dronerf"
ZIP_RELPATH = "dronerf/original/f4c2b4n755-1.zip"

CSV_NAME = re.compile(r"^(?P<bui>[01]{5})(?P<band>[LH])_(?P<segment>\d+)\.csv$")
# BUI = [drone present][drone type x2][flight mode x2]
MODELS = {"00": "Parrot Bebop", "01": "Parrot AR Drone", "10": "DJI Phantom 3"}
MODES = {"00": "on_connected", "01": "hovering", "10": "flying", "11": "flying_video"}
FOLDER_MODEL = {"Bepop drone": "Parrot Bebop", "AR drone": "Parrot AR Drone", "Phantom drone": "DJI Phantom 3",
                "Background RF activites": None}
PAPER_SEGMENTS = {None: 41, "Parrot Bebop": 84, "Parrot AR Drone": 81, "DJI Phantom 3": 21}


def crc32_of(path):
    crc = 0
    with open(path, "rb") as fh:
        while block := fh.read(16 << 20):
            crc = zlib.crc32(block, crc)
    return crc


def _unlink_when_released(path, attempts=24, wait_s=5):
    """Windows antivirus may hold a freshly written file for a few seconds."""
    for _ in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(wait_s)
    raise PermissionError(f"{path} is still locked; delete it by hand")


def list_rars(zip_path, cache_path, tmp_dir):
    """{zip member name: {"zip_crc32": int, "entries": [...]}} for every RAR in the zip."""
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if not info.filename.lower().endswith(".rar"):
                continue
            if cache.get(info.filename, {}).get("zip_crc32") == info.CRC:
                continue
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp = tmp_dir / PurePosixPath(info.filename).name
            try:
                with zf.open(info) as src, open(tmp, "wb") as dst:  # zipfile checks the CRC on read
                    shutil.copyfileobj(src, dst, 16 << 20)
                listing = subprocess.run([UNRAR, "lt", "-v", str(tmp)], capture_output=True, text=True, check=True)
            finally:
                _unlink_when_released(tmp)
            cache[info.filename] = {"zip_crc32": info.CRC, "entries": parse_unrar_lt(listing.stdout)}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache


def build(raw_root=RAW, interim_root=INTERIM):
    zip_path = raw_root / ZIP_RELPATH
    listings = list_rars(zip_path, interim_root / DATASET / "rar_listings.json", interim_root / DATASET / "_tmp")

    artifacts, problems = [], []
    pairs = {}  # (bui, segment) -> {band: artifact}
    for rar_member, listing in sorted(listings.items()):
        folder = PurePosixPath(rar_member).parts[1]
        if folder not in FOLDER_MODEL:
            problems.append(f"unexpected folder {folder}")
            continue
        for entry in listing["entries"]:
            inner = entry["name"].replace("\\", "/")
            match = CSV_NAME.match(PurePosixPath(inner).name)
            if not match:
                problems.append(f"unexpected member {rar_member} :: {inner}")
                continue
            bui, band, segment = match["bui"], match["band"], int(match["segment"])
            model = MODELS[bui[1:3]] if bui[0] == "1" else None
            if model != FOLDER_MODEL[folder]:
                problems.append(f"{inner}: BUI says {model}, folder says {FOLDER_MODEL[folder]}")
            row = {
                "artifact_id": f"{DATASET}/{rar_member}/{inner}",
                "dataset_id": DATASET,
                "storage_relpath": ZIP_RELPATH,
                "member_chain": [rar_member, entry["name"]],
                "size_bytes": int(entry["size"]),
                "crc32": entry["crc32"].lower(),
                "member_mtime": entry["modified"],
                "media_type": "text/csv",
                "capture_id": f"{DATASET}/{bui}/segment_{segment}",
                "channel": band,
                "part_index": None,
            }
            if band in pairs.setdefault((bui, segment), {}):
                problems.append(f"duplicate {bui}{band}_{segment}")
            pairs[(bui, segment)][band] = row
            artifacts.append(row)

    captures, groups = [], {}
    for (bui, segment), bands in sorted(pairs.items()):
        if set(bands) != {"L", "H"}:
            problems.append(f"{bui} segment {segment}: bands {sorted(bands)} (expected L and H)")
        present = bui[0] == "1"
        captures.append({
            "capture_id": f"{DATASET}/{bui}/segment_{segment}",
            "dataset_id": DATASET,
            "group_id": f"{DATASET}/{bui}",
            "n_artifacts": len(bands),
            "channels": sorted(bands),
            "recorded_at": None,
            "start_s": None,
            "duration_s": None,  # 10 M samples at an unconfirmed 40 MS/s: not asserted here
            "center_frequency_hz": None,
            "sample_rate_hz": None,
            "reference_snr_db": None,
            "official_split": None,
            "label_uas_present": present,
            "label_emitter": "uas_link" if present else "none",
            "label_class": MODELS[bui[1:3]] if present else None,
            "label_condition": MODES[bui[3:]] if present else None,
            "label_source": "BUI digits in the file name; digit meaning from the DroneRF paper; "
                            "model digits agree with the release folder of every file",
            "label_status": "AUTHOR_DOC",
        })
        groups.setdefault(bui, 0)
        groups[bui] += 1

    group_rows = [{
        "group_id": f"{DATASET}/{bui}",
        "dataset_id": DATASET,
        "kind": "bui_recording",
        "basis": "segments of one BUI are consecutive 0.25 s cuts of one ~5.25 s recording (~10.25 s background)",
        "status": "INFERRED",
        "n_captures": count,
    } for bui, count in sorted(groups.items())]

    lineage = []
    with zipfile.ZipFile(zip_path) as zf:
        members = {(i.file_size, i.CRC): i.filename for i in zf.infolist() if not i.is_dir()}
    for loose in sorted((raw_root / DATASET).rglob("*.rar")):
        relpath = loose.relative_to(raw_root).as_posix()
        target = members.get((loose.stat().st_size, crc32_of(loose)))
        lineage.append({
            "subject": relpath,
            "relation": "exact_copy_of" if target else "unmatched",
            "object": f"{ZIP_RELPATH}::{target}" if target else "",
            "method": "size + CRC32 against the zip central directory",
            "status": "VERIFIED" if target else "UNKNOWN",
        })

    segments_by_model = {}
    for c in captures:
        segments_by_model[c["label_class"]] = segments_by_model.get(c["label_class"], 0) + 1
    evidence = [
        _ev("rar_archives", len(listings), "VERIFIED", "zip central directory", len(listings)),
        _ev("csv_artifacts", len(artifacts), "VERIFIED", "RAR headers (UnRAR lt)", len(listings)),
        _ev("captures_l_h_pairs", len(captures), "VERIFIED", "RAR headers", len(artifacts)),
        _ev("groups_bui", len(group_rows), "VERIFIED", "RAR headers", len(artifacts)),
        _ev("csv_size_bytes_min", min(a["size_bytes"] for a in artifacts), "VERIFIED", "RAR headers", len(artifacts)),
        _ev("csv_size_bytes_max", max(a["size_bytes"] for a in artifacts), "VERIFIED", "RAR headers", len(artifacts)),
    ]
    for model, expected in PAPER_SEGMENTS.items():
        found = segments_by_model.get(model, 0)
        evidence.append(_ev(f"segments[{model or 'background'}]", found,
                            "VERIFIED" if found == expected else "CONFLICTING",
                            f"RAR headers vs paper ({expected})", len(artifacts)))
    for bui in sorted(groups):
        segments = sorted(s for b, s in pairs if b == bui)
        if segments != list(range(len(segments))):
            problems.append(f"BUI {bui}: segment numbers are not 0..{len(segments) - 1}")
        evidence.append({**_ev(f"segments[{bui}]", len(segments), "VERIFIED", "RAR headers", 2 * len(segments)),
                         "entity_id": f"{DATASET}/{bui}"})
    names = sorted(listings)
    odd = [n for n in names if not PurePosixPath(n).name.startswith("RF Data_")]
    if odd:
        evidence.append(_ev("official_name_typos", "; ".join(odd), "VERIFIED", "zip central directory", None))

    if problems:
        raise ValueError("DroneRF index problems:\n" + "\n".join(problems))
    return {"artifacts": artifacts, "captures": captures, "groups": group_rows,
            "lineage": lineage, "evidence": evidence}


def _ev(field, value, status, source, n):
    return {"entity_id": DATASET, "field": field, "value": str(value), "status": status, "source": source, "n": n}
