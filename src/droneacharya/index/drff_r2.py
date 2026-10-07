"""DRFF-R2 (731 MAT v7.3 files in 7 subsets) -> artifact / capture / group index.

Every file is one 1.4 s capture: RF0_I and RF0_Q (float32, 140 M samples at
100 MS/s) plus scalar metadata (TD unit, State, C config, U receiver, D day,
Height, V, Distance, Direction, NTD1/NTD2 co-flying units, Near).

  artifact = one .mat file at its documented subset location
  capture  = the artifact
  group    = unit + day across subsets and receivers (two receivers may record
             one flight, and a unit's same-day recordings share conditions);
             Dataset-2 mixtures: the mixture folder; Dataset-6: the unit;
             Dataset-7: each environment file on its own

Seven stored files are byte-identical copies (five loose files next to the
release folder, Dataset-5's nested copy of Dataset-7): they are recorded in
lineage, not indexed twice. Scalar fields are cached in interim/drff_r2/mat_fields.json.
"""
import csv
import json
from collections import defaultdict
from pathlib import PurePosixPath

import h5py

from ..paths import INTERIM, MANIFESTS, RAW

DATASET = "drff_r2"
LOCAL_DIR = "drff_r2/original"
RELEASE = "DRFF-R2"
COPY_FOLDERS = ("dataset5-single_drone_inside_absorbent_cotton/dataset7-environment",)
SUBSET_TAG = {
    "dataset1-single_drone_states": "single_drone_states", "dataset2-drone_mixed": "drone_mixed",
    "dataset3-single_drone_hover": "single_drone_hover", "dataset4-single_drone_dual_frequency": "dual_frequency",
    "dataset5-single_drone_inside_absorbent_cotton": "absorbent_cotton", "dataset6-wifi_mixed": "wifi_mixed",
    "dataset7-environment": "environment",
}
EVIDENCE_FIELDS = ("U", "D", "C", "Height", "V", "Distance", "Direction", "Near", "State")


def _scalar(dataset):
    value = dataset[()]
    if dataset.attrs.get("MATLAB_class", b"") == b"char":
        return "".join(map(chr, value.flatten()))
    flat = value.flatten().tolist()
    return flat[0] if len(flat) == 1 else flat


def _fields(root, cache_path):
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    rows = {r["path"]: r for r in cache} if isinstance(cache, list) else cache
    changed = False
    for path in sorted(root.rglob("*.mat")):
        key = path.relative_to(root).as_posix()
        if key in rows and rows[key].get("_size") == path.stat().st_size:
            continue
        with h5py.File(path, "r") as f:
            row = {"path": key, "_size": path.stat().st_size}
            for name in f.keys():
                row[name] = _scalar(f[name]) if f[name].size <= 64 else f"{f[name].dtype}{f[name].shape}"
        rows[key] = row
        changed = True
    if changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    return rows


def build(raw_root=RAW, interim_root=INTERIM, manifests=MANIFESTS):
    local = raw_root / LOCAL_DIR
    fields = _fields(local, interim_root / DATASET / "mat_fields.json")
    with open(manifests / "raw_sha256.csv", newline="", encoding="utf-8") as fh:
        sha = {r["relpath"]: r["sha256"] for r in csv.DictReader(fh)}

    canonical, copies = {}, []
    for key in fields:
        in_release = key.startswith(f"{RELEASE}/")
        if in_release and not any(key.startswith(f"{RELEASE}/{c}/") for c in COPY_FOLDERS):
            canonical[sha[f"{LOCAL_DIR}/{key}"]] = key
        else:
            copies.append(key)

    artifacts, captures, groups, evidence, problems = [], [], defaultdict(int), [], []
    for key in sorted(canonical.values()):
        f = fields[key]
        parts = PurePosixPath(key).parts  # DRFF-R2, subset, [mixture], file
        subset = parts[1]
        tag = SUBSET_TAG.get(subset)
        if tag is None:
            problems.append(f"unexpected subset {subset}")
            continue
        if f.get("RF0_I") != "float32(140000000, 1)" or f.get("RF0_Q") != "float32(140000000, 1)":
            problems.append(f"{key}: unexpected RF0 shape {f.get('RF0_I')}")
        fs = float(f["Fs"])
        fc = float(f.get("CenterFrequence") or f.get("CenterFreq"))
        units = [u for u in (f.get("TD"), f.get("NTD1"), f.get("NTD2")) if u and u != "none"]
        environment = tag == "environment"
        if environment:
            group_id = f"{DATASET}/environment/{PurePosixPath(key).stem}"
        elif tag == "drone_mixed":
            group_id = f"{DATASET}/drone_mixed/{parts[2]}"
        elif tag == "wifi_mixed":
            group_id = f"{DATASET}/wifi_mixed/{f['TD']}"
        else:
            group_id = f"{DATASET}/{f['TD']}/{f['D']}"
        capture_id = f"{DATASET}/{key.removeprefix(RELEASE + '/').removesuffix('.mat')}"
        artifacts.append({
            "artifact_id": f"{DATASET}/{key}",
            "dataset_id": DATASET,
            "storage_relpath": f"{LOCAL_DIR}/{key}",
            "member_chain": [],
            "size_bytes": f["_size"],
            "crc32": None,
            "member_mtime": None,
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
            "recorded_at": None,
            "start_s": None,
            "duration_s": 140_000_000 / fs,
            "center_frequency_hz": fc,
            "sample_rate_hz": fs,
            "reference_snr_db": None,
            "official_split": None,
            "label_uas_present": not environment,
            "label_emitter": "none" if environment else "uas_link",
            "label_class": None if environment else "+".join(units),
            "label_condition": f"{tag}/{f['State']}" if f.get("State") and not environment else tag,
            "label_source": "release subset folder; units and state from the file's TD/NTD/State fields",
            "label_status": "AUTHOR_DOC",
        })
        groups[group_id] += 1
        for name in EVIDENCE_FIELDS:
            if name in f:
                evidence.append(_ev(capture_id, f"mat.{name}", f[name], "VERIFIED", "MAT scalar field"))
        if environment and f.get("TD"):
            evidence.append(_ev(capture_id, "environment_file_metadata",
                                f"folder says environment (negative); MAT fields say TD={f['TD']}, State={f.get('State')}, "
                                f"Height={f.get('Height')}, Near={f.get('Near')} — the Dataset-6 drone layout",
                                "CONFLICTING", "MAT scalar fields vs release folder"))

    lineage = []
    for key in sorted(copies):
        digest = sha[f"{LOCAL_DIR}/{key}"]
        target = canonical.get(digest)
        lineage.append({"subject": f"{LOCAL_DIR}/{key}", "relation": "exact_copy_of" if target else "unmatched",
                        "object": f"{LOCAL_DIR}/{target}" if target else "",
                        "method": "SHA-256 equal (manifests/raw_sha256.csv)", "status": "VERIFIED" if target else "UNKNOWN"})

    group_rows = []
    for group_id, n in sorted(groups.items()):
        kind = group_id.split("/")[1] if group_id.split("/")[1] in ("environment", "drone_mixed", "wifi_mixed") else "unit_day"
        group_rows.append({
            "group_id": group_id, "dataset_id": DATASET, "kind": kind,
            "basis": {"unit_day": "one physical unit on one day, all subsets and receivers together",
                      "drone_mixed": "one mixture folder (same co-flying units)",
                      "wifi_mixed": "one unit in the Wi-Fi-mixed subset (no day field); the unit also appears elsewhere",
                      "environment": "one environment recording"}[kind],
            "status": "INFERRED", "n_captures": n,
        })

    all_fields = [fields[k] for k in canonical.values()]
    evidence += [
        _ev(DATASET, "files_indexed", len(artifacts), "VERIFIED", "local release folder minus copies", len(artifacts)),
        _ev(DATASET, "copies_not_indexed", len(copies), "VERIFIED", "SHA-256 equal to an indexed file", len(copies)),
        _ev(DATASET, "gain_field_present", sum("Gain" in r for r in all_fields), "VERIFIED", "MAT keys of every file",
            len(all_fields)),
        _ev(DATASET, "center_frequency_by_subset",
            "; ".join(f"{t}: {sorted({float(r.get('CenterFrequence') or r.get('CenterFreq')) / 1e9 for r in all_fields if r['path'].split('/')[1] == s})} GHz"
                      for s, t in SUBSET_TAG.items()), "VERIFIED", "MAT fields of every file", len(all_fields)),
        _ev(DATASET, "metadata_layouts",
            "; ".join(sorted({",".join(sorted(k for k in r if not k.startswith(("_", "RF0", "path")))) for r in all_fields})),
            "VERIFIED", "MAT keys of every file", len(all_fields)),
    ]
    if problems:
        raise ValueError("DRFF-R2 index problems:\n" + "\n".join(problems))
    return {"artifacts": artifacts, "captures": captures, "groups": group_rows, "lineage": lineage, "evidence": evidence}


def _ev(entity, field, value, status, source, n=None):
    return {"entity_id": entity, "field": field, "value": str(value), "status": status, "source": source, "n": n}
