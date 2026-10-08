"""Acoustic detection on an unseen recording chain (claims.md A1): the gate items still missing.

Design committed before running (2026-10-08).

  scores    the external detectors of scripts/run_detectors.py, refitted identically (gbm on the cached features of
            the same 1 s windows) and checked against the saved AUC: UaVirBASE → Svanström, Svanström → UaVirBASE
  gate 1    Svanström: drone vs background and drone vs helicopter, separately.
            UaVirBASE (one drone model): drone vs each of the 4 ambient recordings, separately. Does one recording
            carry the 0.85?
  gate 2    inside each test set, with no training: loudness (dBFS; direction fixed in advance: louder = drone),
            and the strongest single feature among the 41 detector features + loudness (feature and direction
            picked on the test set: optimistic for the baseline). Paired difference detector − each, intervals
            resampling groups.
  decision  A1's direction "beats the in-test single feature" only if its paired interval against the optimistic
            strongest feature is above zero; loudness is reported beside it. If a non-drone class or a single
            ambient recording gives an interval reaching 0.5, the claim is restricted to the classes that hold.

CPU only; cached features, no audio is read.
Output: DroneacharyaData/results/checks/acoustic_gate.json
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya import evaluation as E  # noqa: E402
from droneacharya import splits  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

DETECTORS = DATA_ROOT / "results" / "detectors"
OUT = DATA_ROOT / "results" / "checks"
PREFIX = ("band_", "occupied", "spectral", "envelope", "peakiness", "modulation")


def load(dataset):
    rows = json.loads(sorted(DETECTORS.glob(f"{dataset}-*.features.json"))[-1].read_text(encoding="utf-8"))
    captures = {c["capture_id"]: c for c in splits.captures_of(dataset) if c["label_uas_present"] is not None}
    names = sorted(k for k in rows[0] if k.startswith(PREFIX))
    return {"X": np.array([[r[k] for k in names] for r in rows]), "names": names,
            "y": np.array([captures[r["capture_id"]]["label_uas_present"] for r in rows]),
            "groups": np.array([captures[r["capture_id"]]["group_id"] for r in rows]),
            "cls": np.array([str(captures[r["capture_id"]]["label_class"]) for r in rows]),
            "dbfs": np.array([r["dbfs"] for r in rows])}


def gate(train, test, saved_auc):
    scores = E.positive_scores(E.model("gbm").fit(train["X"], train["y"]), test["X"])
    y, g = test["y"], test["groups"]
    columns = {**{n: test["X"][:, j] for j, n in enumerate(test["names"])}, "dbfs": test["dbfs"]}
    aucs = {n: E.metric(y, v) for n, v in columns.items()}
    best = max(aucs, key=lambda n: abs(aucs[n] - 0.5))
    direction = 1 if aucs[best] >= 0.5 else -1
    negatives = sorted(set(test["cls"][~y])) if len(set(test["cls"][~y])) > 1 else sorted(set(g[~y]))
    split_by = "class" if len(set(test["cls"][~y])) > 1 else "recording"
    breakdown = {n: E.subset_auc(scores, y, g, y | ((test["cls"] if split_by == "class" else g) == n)) for n in negatives}
    return {"detector_auc": E.metric(y, scores), "saved_auc": saved_auc,
            "breakdown_by": f"non-drone {split_by}", "breakdown": breakdown,
            "loudness": E.subset_auc(test["dbfs"], y, g, np.ones(len(y), bool)),
            "vs_loudness": E.paired_difference(scores, test["dbfs"], y, g),
            "best_single_feature": {"feature": best, "direction": direction, "auc": abs(aucs[best] - 0.5) + 0.5},
            "vs_best_single_feature": E.paired_difference(scores, direction * columns[best], y, g)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {d: load(d) for d in ("svanstrom", "uavirbase")}
    result = {}
    for train, test in (("uavirbase", "svanstrom"), ("svanstrom", "uavirbase")):
        saved = json.loads((DETECTORS / f"external_{train}_to_{test}.json").read_text(encoding="utf-8"))["roc_auc"]
        r = result[f"{train} -> {test}"] = gate(data[train], data[test], saved)
        holds = [n for n, b in r["breakdown"].items() if b["ci95"][0] > 0.5]
        r["decision"] = {"beats_best_single_feature": bool(r["vs_best_single_feature"]["ci95"][0] > 0),
                         "non_drone_parts_that_hold": holds,
                         "restricted": len(holds) < len(r["breakdown"])}
        print(f"{train} -> {test}: detector {r['detector_auc']:.3f} (saved {saved:.3f}) | "
              + ", ".join(f"vs {n} {b['roc_auc']:.2f} {[round(v, 2) for v in b['ci95']]}" for n, b in r["breakdown"].items())
              + f" | loudness {r['loudness']['roc_auc']:.2f}, vs loudness {r['vs_loudness']['difference']:+.2f} "
              f"{[round(v, 2) for v in r['vs_loudness']['ci95']]} | best {r['best_single_feature']['feature']} "
              f"{r['best_single_feature']['auc']:.2f}, vs best {r['vs_best_single_feature']['difference']:+.2f} "
              f"{[round(v, 2) for v in r['vs_best_single_feature']['ci95']]} | {r['decision']}", flush=True)
    result["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (OUT / "acoustic_gate.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
