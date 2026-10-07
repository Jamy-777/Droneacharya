"""Noisy Drone RF Signal Classification (Kaggle, ZHAW) -> artifact / capture / group index.

archive.zip holds dataset.pt (PyTorch zip format, extracted CRC-verified to
interim/noisy_rf/extracted/ by scripts/extract_member.py). Its pickle is
disassembled with pickletools, never executed; the small tensors (y, snr,
duty_cycle) are read straight from their uncompressed storages.

  artifact = dataset.pt (one stored file holding every vector)
  capture  = one 16,384-sample vector: a derived observation (a clean RC-link
             recording or noise snippet mixed with noise at a synthetic SNR)
  group    = the whole dataset: no recording or noise-snippet ID exists, so no
             split inside it is leakage-safe; evaluation only
"""
import csv
import io
import pickletools
import zipfile
from pathlib import PurePath

import numpy as np

from ..paths import INTERIM, RAW

DATASET = "noisy_rf"
ZIP_RELPATH = "noisy_rf/original/archive.zip"
EXTRACTED = "noisy_rf/extracted/dataset.pt"
SAMPLE_RATE_HZ = 14e6           # Kaggle description
CENTER_FREQUENCY_HZ = 2.44175e9  # authors' 2024 paper
VECTOR_SAMPLES = 16384
# pickle layout read with pickletools: key -> (storage member, dtype, shape)
EXPECTED = {"x_iq": ("0", "f4", (98705, 2, 16384)), "x_spec": ("1", "f4", (98705, 2, 128, 128)),
            "y": ("2", "<i8", (98705,)), "snr": ("3", "<i4", (98705,)), "duty_cycle": ("4", "<f4", (98705,))}


def _pickle_keys(pkl):
    out = io.StringIO()
    pickletools.dis(pkl, out=out)
    return [line.split("'")[1] for line in out.getvalue().splitlines()
            if "BINUNICODE" in line and line.split("'")[1] in EXPECTED]


def build(raw_root=RAW, interim_root=INTERIM):
    with zipfile.ZipFile(raw_root / ZIP_RELPATH) as raw:
        info = raw.getinfo("dataset.pt")
        classes = {int(r["class_int"]): r["class"] for r in csv.DictReader(io.StringIO(raw.read("class_stats.csv").decode()))}
        class_counts = {r["class"]: int(r["count"]) for r in csv.DictReader(io.StringIO(raw.read("class_stats.csv").decode()))}
        snr_counts = {int(r["SNR"]): int(r["count"]) for r in csv.DictReader(io.StringIO(raw.read("SNR_stats.csv").decode()))}
    extracted = interim_root / EXTRACTED
    with zipfile.ZipFile(extracted) as pt:
        keys = _pickle_keys(pt.read("archive/data.pkl"))
        sizes = {PurePath(i.filename).name: i.file_size for i in pt.infolist()}
        y = np.frombuffer(pt.read("archive/data/2"), "<i8")
        snr = np.frombuffer(pt.read("archive/data/3"), "<i4")
        duty = np.frombuffer(pt.read("archive/data/4"), "<f4")

    problems = []
    if keys != list(EXPECTED):
        problems.append(f"pickle keys {keys} differ from {list(EXPECTED)}")
    for key, (member, dtype, shape) in EXPECTED.items():
        if sizes.get(member) != int(np.prod(shape)) * np.dtype(dtype).itemsize:
            problems.append(f"{key}: storage {member} has {sizes.get(member)} bytes, expected {shape} {dtype}")
    found_classes = {classes[k]: int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    if found_classes != class_counts:
        problems.append(f"y counts {found_classes} differ from class_stats.csv {class_counts}")
    found_snr = {int(k): int(v) for k, v in zip(*np.unique(snr, return_counts=True))}
    if found_snr != snr_counts:
        problems.append("snr counts differ from SNR_stats.csv")
    if problems:
        raise ValueError("Noisy RF index problems:\n" + "\n".join(problems))

    artifact_id = f"{DATASET}/archive.zip/dataset.pt"
    group_id = f"{DATASET}/all"
    artifacts = [{
        "artifact_id": artifact_id, "dataset_id": DATASET, "storage_relpath": ZIP_RELPATH, "member_chain": ["dataset.pt"],
        "size_bytes": info.file_size, "crc32": f"{info.CRC:08x}",
        "member_mtime": "{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}".format(*info.date_time),
        "media_type": "application/x-pytorch", "capture_id": None, "channel": None, "part_index": None,
    }]
    captures = [{
        "capture_id": f"{DATASET}/dataset.pt/{i}", "dataset_id": DATASET, "group_id": group_id, "n_artifacts": 0,
        "stored_in": artifact_id, "stored_index": i, "channels": ["I", "Q"], "recorded_at": None, "start_s": None,
        "duration_s": VECTOR_SAMPLES / SAMPLE_RATE_HZ, "center_frequency_hz": CENTER_FREQUENCY_HZ,
        "sample_rate_hz": SAMPLE_RATE_HZ, "reference_snr_db": float(snr[i]), "official_split": None,
        "label_uas_present": classes[int(y[i])] != "Noise",
        "label_emitter": "none" if classes[int(y[i])] == "Noise" else "controller",
        "label_class": classes[int(y[i])], "label_condition": None,
        "label_source": "y tensor + class_stats.csv (Kaggle release)", "label_status": "REPO_METADATA",
    } for i in range(len(y))]
    groups = [{"group_id": group_id, "dataset_id": DATASET, "kind": "no_recording_ids",
               "basis": "vectors carry no recording or noise-snippet ID: no within-dataset split is leakage-safe; "
                        "evaluation only", "status": "UNKNOWN", "n_captures": len(captures)}]
    lineage = [{"subject": f"interim/{EXTRACTED}", "relation": "exact_copy_of", "object": f"{ZIP_RELPATH}::dataset.pt",
                "method": "CRC-32 verified on extraction (scripts/extract_member.py)", "status": "VERIFIED"}]
    evidence = [
        _ev("tensors", "x_iq float32 (98705, 2, 16384) — I and Q as two real channels; x_spec float32 (98705, 2, 128, 128); "
            "y int64; snr int32; duty_cycle float32", "pickle disassembly + storage sizes"),
        _ev("id_fields", "none (only x_iq, x_spec, y, snr, duty_cycle)", "pickle disassembly"),
        _ev("class_counts_match_class_stats", True, "y tensor vs class_stats.csv", len(y)),
        _ev("snr_counts_match_snr_stats", True, "snr tensor vs SNR_stats.csv", len(snr)),
        _ev("duty_cycle_range", f"{duty.min():.4f} – {duty.max():.4f}", "duty_cycle tensor", len(duty)),
    ]
    return {"artifacts": artifacts, "captures": captures, "groups": groups, "lineage": lineage, "evidence": evidence}


def _ev(field, value, source, n=None):
    return {"entity_id": DATASET, "field": field, "value": str(value), "status": "VERIFIED", "source": source, "n": n}

