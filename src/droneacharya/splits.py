"""Split registry: official splits as the releases define them, and leakage-safe experiment splits.

A split assigns every capture of its datasets to one partition (train/test,
fold0..foldK-1, or test only). Experiment splits move whole groups, so no
independence unit appears in two partitions. Each split is stored as
splits/<dataset>/<name>.parquet (capture_id, partition) and described in
splits/registry.json: kind, strategy, seed, counts per partition and label, and
the SHA-256 of its assignment, so a split ID always means the same partition.
"""
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from .index import read
from .paths import DATA_ROOT

SPLITS = DATA_ROOT / "splits"
REGISTRY = SPLITS / "registry.json"


def captures_of(dataset):
    return read(dataset, "captures").to_pylist()


def official(captures):
    """{capture_id: partition} from the release's own split; captures without one are left out."""
    return {c["capture_id"]: c["official_split"] for c in captures if c["official_split"]}


def group_labels(captures, label="label_uas_present"):
    labels = defaultdict(set)
    for c in captures:
        labels[c["group_id"]].add(c[label])
    return labels


def feasibility(captures, label="label_uas_present"):
    """Groups per label value; a value with fewer than 2 groups cannot appear in both train and test."""
    per_value = defaultdict(int)
    for values in group_labels(captures, label).values():
        if values == {None}:  # not a detection class (e.g. calibration tones)
            continue
        per_value[tuple(sorted(values, key=str)) if len(values) > 1 else next(iter(values))] += 1
    return dict(per_value)


def group_kfold(captures, k, seed=0, label="label_uas_present"):
    """{capture_id: "fold<i>"}: whole groups to folds, balancing each label value's captures across folds."""
    sizes, labels = defaultdict(int), group_labels(captures, label)
    for c in captures:
        sizes[c["group_id"]] += 1
    strata = defaultdict(list)
    for group, values in labels.items():
        strata[tuple(sorted(values, key=str))].append(group)
    rng = random.Random(seed)
    fold_of = {}
    for stratum in sorted(strata, key=str):
        groups = sorted(strata[stratum])
        rng.shuffle(groups)
        groups.sort(key=lambda g: -sizes[g])  # largest first, ties in shuffled order
        load = [0] * k
        for g in groups:
            fold = min(range(k), key=lambda i: (load[i], i))
            fold_of[g] = fold
            load[fold] += sizes[g]
    return {c["capture_id"]: f"fold{fold_of[c['group_id']]}" for c in captures}


def leakage_problems(captures, assignment):
    problems = []
    partitions = defaultdict(set)
    known = {c["capture_id"]: c for c in captures}
    for capture_id, partition in assignment.items():
        if capture_id not in known:
            problems.append(f"unknown capture {capture_id}")
            continue
        partitions[known[capture_id]["group_id"]].add(partition)
    problems += [f"group {g} spans partitions {sorted(p)}" for g, p in partitions.items() if len(p) > 1]
    return problems


def save(split_id, kind, strategy, captures, assignment, *, seed=None, notes=None, check_groups=True):
    """Write the split and its registry entry; refuses group leakage for experiment splits."""
    if check_groups and (problems := leakage_problems(captures, assignment)):
        raise ValueError(f"{split_id}: " + "; ".join(problems[:5]))
    rows = sorted(assignment.items())
    digest = hashlib.sha256("\n".join(f"{c}\t{p}" for c, p in rows).encode()).hexdigest()
    path = SPLITS / f"{split_id}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"capture_id": [c for c, _ in rows], "partition": [p for _, p in rows]}), path)

    label_of = {c["capture_id"]: c["label_uas_present"] for c in captures}
    counts = defaultdict(lambda: defaultdict(int))
    for capture_id, partition in rows:
        counts[partition][str(label_of[capture_id])] += 1
    registry = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else {}
    registry[split_id] = {
        "kind": kind, "strategy": strategy, "seed": seed, "datasets": sorted({c.split("/")[0] for c, _ in rows}),
        "captures": len(rows), "assignment_sha256": digest,
        "counts_by_partition_and_uas_present": {p: dict(v) for p, v in sorted(counts.items())},
        "group_leakage_checked": check_groups, "notes": notes,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    REGISTRY.write_text(json.dumps(dict(sorted(registry.items())), indent=1), encoding="utf-8")
    return registry[split_id]


def load(split_id):
    """{capture_id: partition}, after checking the file still matches its registry hash."""
    table = pq.read_table(SPLITS / f"{split_id}.parquet").to_pylist()
    entry = json.loads(REGISTRY.read_text(encoding="utf-8"))[split_id]
    digest = hashlib.sha256("\n".join(f"{r['capture_id']}\t{r['partition']}" for r in table).encode()).hexdigest()
    if digest != entry["assignment_sha256"]:
        raise ValueError(f"{split_id}: assignment no longer matches the registry")
    return {r["capture_id"]: r["partition"] for r in table}
