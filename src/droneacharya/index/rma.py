"""RMA Drone RF Dataset (KU Leuven RDR 10.48804/HZRVNZ v1.0, 16 zips) -> artifact / capture / group index.

Release layout: <archive>.zip -> <archive>/<name>_<i>.mat, MAT v7.3 with one
variable `uhd_samps` (complex, int16 codes / 32768). One archive per system and
link: RC uplink or video downlink ("Vid"); hobby controllers have RC only.

  artifact = one .mat member
  capture  = the artifact (files are separate captures seconds apart, not one stream)
  group    = archive (one recording session; INFERRED from MAT creation times)

Per member: the MAT header creation time (first 128 bytes) and the length of
uhd_samps (HDF5 metadata; the member is decompressed), cached in
interim/rma/mat_fields.json. File names repeat across archives (Mavic_1.mat in
RC and Vid archives), so every ID carries the archive.
"""
import csv
import json
import re
import zipfile
import zlib
from datetime import datetime

import h5py

from ..paths import INTERIM, MANIFESTS, RAW

DATASET = "rma"
LOCAL_DIR = "rma/original"
SAMPLE_RATE_HZ = 100e6          # README.txt and authors' code
CENTER_FREQUENCY_HZ = 2.44e9    # README.txt and authors' code
# archive -> (system class, link condition, emitter)
ARCHIVES = {
    "Spektrum_DX4e": ("Spektrum DX4e", None, "controller"),
    "Frysky": ("Frysky", None, "controller"),
    "NineEagles": ("NineEagles", None, "controller"),
    "wltoys": ("wltoys", None, "controller"),
    "Q205": ("Q205", None, "controller"),
    "SJRC_pro": ("SJRC F11 Pro", None, None),  # README: "drone + remote controller", one archive: link unknown
    "mini2RC": ("DJI Mini 2", "RC", "controller"), "mini2vid": ("DJI Mini 2", "Vid", "aircraft"),
    "inspire2RC": ("DJI Inspire 2", "RC", "controller"), "inspire2Vid": ("DJI Inspire 2", "Vid", "aircraft"),
    "matriceRC": ("DJI Matrice", "RC", "controller"), "matricevid": ("DJI Matrice", "Vid", "aircraft"),
    "MavicRC1": ("Mavic", "RC", "controller"), "MavicRC2": ("Mavic", "RC", "controller"),
    "MavicVid1": ("Mavic", "Vid", "aircraft"), "MavicVid2": ("Mavic", "Vid", "aircraft"),
}
CREATED = re.compile(rb"Created on: (\w{3} \w{3} +\d+ \d\d:\d\d:\d\d \d{4})")


def _mat_fields(zips, cache_path):
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    for zip_path in zips:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                key = f"{zip_path.name}::{info.filename}"
                if info.is_dir() or cache.get(key, {}).get("crc") == info.CRC:
                    continue
                with zf.open(info) as fh:
                    header = fh.read(128)
                    fh.seek(0)
                    with h5py.File(fh, "r") as f:
                        shape = f["uhd_samps"].shape
                created = CREATED.search(header)
                cache[key] = {"crc": info.CRC, "header": header[:116].decode("latin1").strip(),
                              "created": created.group(1).decode() if created else None, "shape": list(shape)}
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
        print(f"  MAT fields read: {zip_path.name}", flush=True)
    return cache


def _crc32_of(path):
    crc = 0
    with open(path, "rb") as fh:
        while block := fh.read(16 << 20):
            crc = zlib.crc32(block, crc)
    return crc


def build(raw_root=RAW, interim_root=INTERIM, manifests=MANIFESTS):
    local = raw_root / LOCAL_DIR
    zips = sorted(local.glob("*.zip"))
    fields = _mat_fields(zips, interim_root / DATASET / "mat_fields.json")
    listing = json.loads((manifests / "rma_file_listing.json").read_text(encoding="utf-8"))
    official = {f["label"]: f["dataFile"] for f in listing}

    artifacts, captures, groups, lineage, evidence, problems = [], [], [], [], [], []
    members = {}
    for zip_path in zips:
        archive = zip_path.stem
        if archive not in ARCHIVES:
            problems.append(f"unexpected archive {zip_path.name}")
            continue
        system, link, emitter = ARCHIVES[archive]
        relpath = f"{LOCAL_DIR}/{zip_path.name}"
        record = official.get(zip_path.name)
        same = record is not None and record["filesize"] == zip_path.stat().st_size
        lineage.append({"subject": relpath,
                        "relation": "local_copy_of" if same else "unmatched",
                        "object": f"doi:10.48804/HZRVNZ/{zip_path.name}",
                        "method": "size equals the repository record; MD5 checked against it at download "
                                  "(scripts/download_dataverse.py)",
                        "status": "VERIFIED" if same else "UNKNOWN"})
        group_id = f"{DATASET}/{archive}"
        created_times, lengths = [], set()
        with zipfile.ZipFile(zip_path) as zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
        for info in infos:
            members[(info.file_size, info.CRC)] = f"{relpath}::{info.filename}"
            meta = fields[f"{zip_path.name}::{info.filename}"]
            if not info.filename.endswith(".mat") or meta["shape"][0] != 1:
                problems.append(f"{archive}: unexpected member or shape {info.filename} {meta['shape']}")
                continue
            created = datetime.strptime(" ".join(meta["created"].split()), "%a %b %d %H:%M:%S %Y")
            n_samples = meta["shape"][1]
            created_times.append(created)
            lengths.add(n_samples)
            capture_id = f"{DATASET}/{archive}/{info.filename.rsplit('/', 1)[-1].removesuffix('.mat')}"
            artifacts.append({
                "artifact_id": f"{DATASET}/{zip_path.name}/{info.filename}",
                "dataset_id": DATASET,
                "storage_relpath": relpath,
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
                "recorded_at": created.isoformat(),
                "start_s": None,
                "duration_s": n_samples / SAMPLE_RATE_HZ,
                "center_frequency_hz": CENTER_FREQUENCY_HZ,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "reference_snr_db": None,
                "official_split": None,
                "label_uas_present": True,
                "label_emitter": emitter,
                "label_class": system,
                "label_condition": link,
                "label_source": "archive name; system names from README.txt where it lists them",
                "label_status": "AUTHOR_DOC",
            })
        groups.append({
            "group_id": group_id,
            "dataset_id": DATASET,
            "kind": "archive",
            "basis": "one archive = one recording session (MAT creation times of its files span minutes)",
            "status": "INFERRED",
            "n_captures": len(created_times),
        })
        created_times.sort()
        evidence += [
            _ev(group_id, "created_span", f"{created_times[0]} → {created_times[-1]}", "VERIFIED",
                "MAT header of every file", len(created_times)),
            _ev(group_id, "samples_per_file", ", ".join(str(n) for n in sorted(lengths)), "VERIFIED",
                "uhd_samps shape of every file", len(created_times)),
        ]

    for loose in sorted(p for p in local.rglob("*.mat")):
        relpath = loose.relative_to(raw_root).as_posix()
        target = members.get((loose.stat().st_size, _crc32_of(loose)))
        lineage.append({"subject": relpath, "relation": "exact_copy_of" if target else "unmatched",
                        "object": target or "", "method": "size + CRC32 against the zip central directory",
                        "status": "VERIFIED" if target else "UNKNOWN"})

    evidence += [
        _ev(DATASET, "archives", len(zips), "VERIFIED", "local zips", None),
        _ev(DATASET, "captures", len(captures), "VERIFIED", "zip central directories", len(captures)),
        _ev(DATASET, "capture_metadata_columns", "sample_rate_hz and center_frequency_hz from README.txt; "
            "duration_s = uhd_samps length / 100 MS/s; recorded_at = MAT header creation time", "AUTHOR_DOC",
            "README.txt + MAT headers", len(captures)),
    ]
    if problems:
        raise ValueError("RMA index problems:\n" + "\n".join(problems))
    return {"artifacts": artifacts, "captures": captures, "groups": groups, "lineage": lineage, "evidence": evidence}


def _ev(entity, field, value, status, source, n):
    return {"entity_id": entity, "field": field, "value": str(value), "status": status, "source": source, "n": n}
