"""Create the default splits for every indexed dataset and report detection feasibility.

  <dataset>/official          the release's own split, when it has one (kind: official)
  <dataset>/group-kfold-s0    whole groups to k folds, UAS-present balanced (kind: experiment),
                              k = min(5, smallest number of groups of a label value)
  noisy_rf/evaluation-only    every vector in "test": no leakage-safe split exists

Usage:
    python scripts/build_splits.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya import splits  # noqa: E402
from droneacharya.index import BUILDERS  # noqa: E402
from droneacharya.paths import INDEX  # noqa: E402

DEFAULT_K = 5
# Experiment split names. A name changes when the grouping behind it changes (the old split is retired, never
# overwritten): CardRF moved from device groups to UAS-system groups on 2026-10-08.
EXPERIMENT_SPLIT = {"cardrf": "system-kfold-s0"}


def main():
    for dataset in sorted(BUILDERS):
        if not (INDEX / dataset / "captures.parquet").exists():
            print(f"{dataset}: not indexed")
            continue
        captures = splits.captures_of(dataset)
        groups = splits.read(dataset, "groups").to_pylist()
        if assignment := splits.official(captures):
            entry = splits.save(f"{dataset}/official", "official", "release's own split", captures, assignment,
                                check_groups=False, notes="same-session leakage is the release's, not ours")
            print(f"{dataset}/official: {entry['counts_by_partition_and_uas_present']}")
        if any(g["status"] == "UNKNOWN" for g in groups):
            entry = splits.save(f"{dataset}/evaluation-only", "experiment", "all captures in test",
                                captures, {c["capture_id"]: "test" for c in captures},
                                notes="no recording IDs: no within-dataset training split is leakage-safe")
            print(f"{dataset}/evaluation-only: {entry['captures']} captures")
            continue
        per_value = splits.feasibility(captures)
        k = min(DEFAULT_K, min(per_value.values()))
        note = f"groups per UAS-present value: {per_value}"
        if k < 2:
            k = min(DEFAULT_K, max(per_value.values()))
            note += ("; a label value has a single group, so within-dataset detection cannot keep it out of "
                     "one partition — use this split for positives-only tasks or test detection across datasets")
        assignment = splits.group_kfold(captures, k, seed=0)
        entry = splits.save(f"{dataset}/{EXPERIMENT_SPLIT.get(dataset, 'group-kfold-s0')}", "experiment", f"group k-fold, k={k}, UAS-present balanced",
                            captures, assignment, seed=0, notes=note)
        print(f"{dataset}/{EXPERIMENT_SPLIT.get(dataset, 'group-kfold-s0')}: k={k}; {note}")


if __name__ == "__main__":
    main()
