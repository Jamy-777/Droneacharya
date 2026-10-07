"""UAVSig v4.0 (UCLA Dataverse 10.25346/S6/LVRRAE, local subset) -> artifact / capture / group index.

Release layout: <category>/2024-05-09/<name>.mat with category one_drone,
two_drone or controllers. File name grammar (authors' README, matches our decoding):
  drone_ABCD_NN.mat           digit k = channel of drone k (0 = off), NN = capture 00–05
  controller_pP_ABCD_NN.mat   P = physical arrangement, digit k = controller k on (1) / off (0);
                              controller_pP_0000 = no experimental transmitter (baseline)

  artifact = one .mat file (MAT v5: `data` complex IQ + per-transmission labels)
  capture  = the artifact (one 1 s capture)
  group    = scenario (file name without the capture index): its 6 captures are consecutive

Per file, the variable directory is read with scipy.io.whosmat (no data loaded)
and num_transmission with a one-variable load. MAT header times are conversion
times (Sep 2024), not acquisition times, so recorded_at stays null.
"""
import json
import re
from datetime import datetime
from pathlib import Path

import scipy.io

from ..paths import MANIFESTS, RAW

DATASET = "uavsig"
LOCAL_DIR = "uavsig/original"
SAMPLE_RATE_HZ = 50e6           # authors' README
CENTER_FREQUENCY_HZ = 2.4435e9  # authors' README
DRONE = re.compile(r"^drone_(?P<set>[0-4]{4})_(?P<capture>\d{2})\.mat$")
CONTROLLER = re.compile(r"^controller_p(?P<arrangement>\d)_(?P<set>[01]{4})_(?P<capture>\d{2})\.mat$")
LABEL_VARIABLES = ("start", "end", "fc", "bw", "id")


def _labels(name):
    if m := DRONE.match(name):
        active = [(k + 1, int(c)) for k, c in enumerate(m["set"]) if c != "0"]
        return (m["set"], m["capture"], {
            "label_uas_present": bool(active),
            "label_emitter": "aircraft" if active else "none",
            "label_class": "+".join(f"drone_{k}" for k, _ in active) or None,
            "label_condition": ", ".join(f"drone_{k}:ch{c}" for k, c in active) or None,
        })
    if m := CONTROLLER.match(name):
        active = [k + 1 for k, c in enumerate(m["set"]) if c == "1"]
        return (f"p{m['arrangement']}_{m['set']}", m["capture"], {
            "label_uas_present": bool(active),
            "label_emitter": "controller" if active else "none",
            "label_class": "+".join(f"controller_{k}" for k in active) or None,
            "label_condition": f"arrangement p{m['arrangement']}",
        })
    return None


def build(raw_root=RAW, manifests=MANIFESTS):
    listing = json.loads((manifests / "uavsig_v4_file_listing.json").read_text(encoding="utf-8"))
    files = listing  # list of Dataverse file records
    official = {f"{f.get('directoryLabel', '')}/{f['label']}".lstrip("/"): f["dataFile"]["filesize"] for f in files}

    artifacts, captures, groups, lineage, evidence, problems = [], [], {}, [], [], []
    headers = set()
    for path in sorted((raw_root / LOCAL_DIR).glob("*/*/*.mat")):
        official_path = path.relative_to(raw_root / LOCAL_DIR).as_posix()
        category = official_path.split("/")[0]
        decoded = _labels(path.name)
        if decoded is None:
            problems.append(f"{official_path}: name does not follow the grammar")
            continue
        scenario, capture_index, labels = decoded
        size = path.stat().st_size
        if official.get(official_path) != size:
            problems.append(f"{official_path}: not in the v4.0 listing or size differs")
        directory = {name: shape for name, shape, _ in scipy.io.whosmat(path)}
        n_transmissions = int(scipy.io.loadmat(path, variable_names=["num_transmission"])["num_transmission"].item())
        if directory.get("data", (0, 0))[0] != 1 or any(directory.get(v, (0, 0))[-1] != n_transmissions
                                                        for v in LABEL_VARIABLES if n_transmissions):
            problems.append(f"{official_path}: unexpected variables {directory}")
        with open(path, "rb") as fh:
            headers.add(fh.read(116).decode("latin1").strip())
        relpath = f"{LOCAL_DIR}/{official_path}"
        capture_id = f"{DATASET}/{official_path.removesuffix('.mat')}"
        group_id = f"{DATASET}/{category}/{path.name.rsplit('_', 1)[0]}"
        artifacts.append({
            "artifact_id": f"{DATASET}/{official_path}",
            "dataset_id": DATASET,
            "storage_relpath": relpath,
            "member_chain": [],
            "size_bytes": size,
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
            "duration_s": directory["data"][1] / SAMPLE_RATE_HZ,
            "center_frequency_hz": CENTER_FREQUENCY_HZ,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "reference_snr_db": None,
            "official_split": None,  # published: 5 train / 1 test per scenario; which index is test is not stated
            **labels,
            "label_source": "file name grammar (authors' README)",
            "label_status": "AUTHOR_DOC",
        })
        evidence.append(_ev(capture_id, "labelled_transmissions", n_transmissions, "VERIFIED", "num_transmission"))
        lineage.append({"subject": relpath, "relation": "local_copy_of" if official.get(official_path) == size else "unmatched",
                        "object": f"doi:10.25346/S6/LVRRAE/{official_path}",
                        "method": "size equals the v4.0 listing; MD5 checked against it at download",
                        "status": "VERIFIED" if official.get(official_path) == size else "UNKNOWN"})
        groups.setdefault(group_id, []).append(capture_index)

    group_rows = [{
        "group_id": group_id,
        "dataset_id": DATASET,
        "kind": "scenario",
        "basis": "one transmitter set and channel/arrangement, 6 consecutive 1 s captures",
        "status": "AUTHOR_DOC",
        "n_captures": len(indices),
    } for group_id, indices in sorted(groups.items())]

    per_category = {}
    for c in captures:
        category = c["capture_id"].split("/")[1]
        per_category[category] = per_category.get(category, 0) + 1
    released = {}
    for path in official:
        if path.endswith(".mat"):
            released[path.split("/")[0]] = released.get(path.split("/")[0], 0) + 1
    created = sorted(datetime.strptime(" ".join(re.search(r"Created on: (.*)$", h).group(1).split()), "%a %b %d %H:%M:%S %Y")
                     for h in headers if "Created on" in h)
    evidence += [
        _ev(DATASET, "local_subset", "; ".join(f"{k}: {per_category.get(k, 0)} of {v}" for k, v in sorted(released.items())),
            "VERIFIED", "local files vs v4.0 listing", len(captures)),
        _ev(DATASET, "mat_header_created", f"{created[0]} … {created[-1]} (conversion time, not acquisition)",
            "VERIFIED", "MAT headers", len(captures)),
        _ev(DATASET, "capture_metadata_columns", "sample_rate_hz, center_frequency_hz from the authors' README; "
            "duration_s = data length / 50 MS/s", "AUTHOR_DOC", "UAVSig Readme.pdf", len(captures)),
        _ev(DATASET, "baseline_captures", sum(not c["label_uas_present"] for c in captures), "VERIFIED",
            "controller_p*_0000 files", len(captures)),
    ]
    if problems:
        raise ValueError("UAVSig index problems:\n" + "\n".join(problems))
    return {"artifacts": artifacts, "captures": captures, "groups": group_rows, "lineage": lineage, "evidence": evidence}


def _ev(entity, field, value, status, source, n=None):
    return {"entity_id": entity, "field": field, "value": str(value), "status": status, "source": source, "n": n}
