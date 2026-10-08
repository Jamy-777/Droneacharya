"""Received power inside each RF test set: the single-feature baseline the audit found missing (claims.md R1, R3, R4).

Design committed before running (2026-10-08). Audit point 4.

  power      log10 RMS of each tile before unit-RMS normalisation: the `log_rms` field run_rf_cross.py cached, and
             the same computation on the cached stage-2 14 MHz tiles. No training. Direction fixed in advance: more
             power = drone; an AUC below 0.5 means the quieter tiles are the drones, reported as is.
  rows       the cached rows the reported numbers came from (rf_cross v1.1 25 MHz, v1.2 40 MHz, stage-2 14 MHz)
  detectors  refitted exactly as their scripts did, and checked against the saved AUC:
               R1 T1 UAVSig → CardRF and T2 CardRF → UAVSig, gbm, v1.1 and v1.2         (run_rf_cross.py)
               R4 within CardRF (cardrf/system-kfold-s0) and within UAVSig (uavsig/group-kfold-s0), gbm pooled
                  out of fold, v1.1 and v1.2                                              (run_rf_cross.py)
               R3 merged drone-only pool → CardRF, kNN and GMM, stage-2 tiles             (run_merged_one_class.py)
  paired     AUC(detector) − AUC(power) on identical rows, interval resampling groups; the same against the
             strongest single feature inside the test set (feature and direction picked on the test set itself:
             optimistic for the baseline)
  also       power AUC inside Noisy RF per SNR (vectors were unit-power normalised before mixing), for R2
  decision   a detector number "beats power inside its test set" only if its paired interval is above zero; every
             result goes into claims.md either way

CPU only; no raw data is read.
Output: DroneacharyaData/results/checks/power_inside_tests.json
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_merged_one_class as M  # noqa: E402
from droneacharya import evaluation as E  # noqa: E402
from droneacharya import splits  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

RESULTS = DATA_ROOT / "results"
OUT = RESULTS / "checks"
VERSIONS = {"v1.1": RESULTS / "rf_cross", "v1.2": RESULTS / "rf_cross_v1.2"}
PREFIX = ("shape_", "occupied", "active", "spectral", "duty", "transitions", "frame_", "centroid", "amplitude", "papr")
WITHIN_SPLIT = {"cardrf": "cardrf/system-kfold-s0", "uavsig": "uavsig/group-kfold-s0"}


def log_rms(tiles):
    return np.array([np.log10(np.sqrt(np.mean(np.abs(t.astype(np.complex128)) ** 2)) + 1e-12) for t in tiles])


def rf_cross_rows(folder, dataset):
    rows = json.loads(sorted(folder.glob(f"{dataset}-*.features.json"))[-1].read_text(encoding="utf-8"))
    captures = {c["capture_id"]: c for c in splits.captures_of(dataset)}
    names = sorted(k for k in rows[0] if k.startswith(PREFIX))
    return {"ids": [r["capture_id"] for r in rows], "X": np.array([[r[k] for k in names] for r in rows]),
            "y": np.array([captures[r["capture_id"]]["label_uas_present"] for r in rows]),
            "groups": np.array([captures[r["capture_id"]]["group_id"] for r in rows]),
            "power": np.array([r["log_rms"] for r in rows]), "names": names}


def against_baselines(scores, test, saved_auc):
    y, g = test["y"], test["groups"]
    columns = {**{n: test["X"][:, j] for j, n in enumerate(test["names"])}, "log_rms": test["power"]}
    aucs = {n: E.metric(y, v) for n, v in columns.items()}
    best = max(aucs, key=lambda n: abs(aucs[n] - 0.5))
    direction = 1 if aucs[best] >= 0.5 else -1
    return {"detector_auc": E.metric(y, scores), "saved_auc": saved_auc,
            "power": E.subset_auc(test["power"], y, g, np.ones(len(y), bool)),
            "vs_power": E.paired_difference(scores, test["power"], y, g),
            "best_single_feature": {"feature": best, "direction": direction, "auc": abs(aucs[best] - 0.5) + 0.5},
            "vs_best_single_feature": E.paired_difference(scores, direction * columns[best], y, g)}


def saved(path):
    return json.loads(path.read_text(encoding="utf-8"))["roc_auc"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for version, folder in VERSIONS.items():
        data = {d: rf_cross_rows(folder, d) for d in ("cardrf", "uavsig")}
        for train, test in (("uavsig", "cardrf"), ("cardrf", "uavsig")):
            fitted = E.model("gbm").fit(data[train]["X"], data[train]["y"])
            scores = E.positive_scores(fitted, data[test]["X"])
            result[f"R1 {train} -> {test} ({version})"] = against_baselines(
                scores, data[test], saved(folder / f"cross_{train}_to_{test}.json"))
        for dataset, d in data.items():
            assignment = splits.load(WITHIN_SPLIT[dataset])
            parts = np.array([assignment[i] for i in d["ids"]])
            scores = E.out_of_fold(d["X"], d["y"], parts, "gbm")
            result[f"R4 within {dataset} ({version})"] = against_baselines(scores, d, saved(folder / f"within_{dataset}.json"))

    pool = json.loads(sorted(M.OUT.glob("rf_pool-*.features.json"))[-1].read_text(encoding="utf-8"))
    names = M.names_of(pool)
    ids, feats, tiles, meta = M.stage2_set("cardrf")
    stage2 = {"X": np.array([[f[k] for k in names] for f in feats]), "names": names,
              "y": np.array([meta[i]["label_uas_present"] for i in ids]),
              "groups": np.array([meta[i]["group_id"] for i in ids]), "power": log_rms(np.asarray(tiles))}
    scores = M.one_class_scores(np.array([[r[k] for k in names] for r in pool]), stage2["X"])
    for model in ("knn", "gmm"):
        result[f"R3 merged pool -> cardrf {model} (stage 2)"] = against_baselines(
            scores[model], stage2, saved(M.OUT / f"rf_{model}_to_cardrf.json"))

    n_meta = {c["capture_id"]: c for c in splits.captures_of("noisy_rf")}
    n_ids = sorted(n_meta)
    y = np.array([n_meta[i]["label_uas_present"] for i in n_ids])
    snr = np.array([n_meta[i]["reference_snr_db"] for i in n_ids])
    n_tiles = np.load(sorted((RESULTS / "rf_stage2").glob("noisy_rf-*.tiles.npy"))[-1], mmap_mode="r")
    power = np.concatenate([log_rms(np.asarray(n_tiles[i:i + 8192])) for i in range(0, len(n_tiles), 8192)])
    result["R2 power inside noisy_rf by snr"] = {
        int(s): E.metric(np.r_[np.ones(np.sum(y & (snr == s)), bool), np.zeros(np.sum(~y), bool)],
                         np.r_[power[y & (snr == s)], power[~y]]) for s in sorted(set(snr[y]))}
    result["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (OUT / "power_inside_tests.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    for name, r in result.items():
        if isinstance(r, dict) and "detector_auc" in r:
            print(f"{name:42} detector {r['detector_auc']:.3f} (saved {r['saved_auc']:.3f})  power {r['power']['roc_auc']:.3f}  "
                  f"vs power {r['vs_power']['difference']:+.3f} {[round(v, 2) for v in r['vs_power']['ci95']]}  "
                  f"best {r['best_single_feature']['feature']} {r['best_single_feature']['auc']:.3f}  "
                  f"vs best {r['vs_best_single_feature']['difference']:+.3f} "
                  f"{[round(v, 2) for v in r['vs_best_single_feature']['ci95']]}")
    print("noisy_rf power by snr:", {k: round(v, 2) for k, v in result["R2 power inside noisy_rf by snr"].items()})


if __name__ == "__main__":
    main()
