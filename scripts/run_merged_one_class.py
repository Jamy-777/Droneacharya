"""The merged drone-only model: learn what drones look like from every positive-only dataset, then test it
against datasets that hold both classes. RF and acoustic. Plus RF recall of the CardRF-trained CNN.

Design fixed before any result was seen (2026-10-09):

  RF one-class
    pool       drone tiles only, 14 MHz stage-2 format (rf_common.iq_steered, 250 us):
               RFUAV every pack, 2.4 and 5.8 GHz (4 consecutive tiles at the start of every 1 s chunk);
               RMA every capture (2 tiles); DRFF-R2 Datasets 1–5, 5.745 GHz (2 tiles). Excluded: DRFF-R2 Dataset 6
               (its drone captures contain Wi-Fi), Dataset 7 (environment), DroneRF (I only, rate unconfirmed),
               every test set
    models     primary: mean distance to the 10 nearest pool tiles (standardised rf_features); secondary:
               8-component diagonal Gaussian mixture log-likelihood. No negatives are seen in training.
    tests      CardRF (9,600 captures, UAS vs Wi-Fi/BT; 13 groups), UAVSig (360 captures x 4 tiles, UAS vs
               no-transmitter; 120 groups, 2 negative), Noisy RF (98,705 vectors; transmitter-only interval, no p).
               CardRF and Noisy RF tiles are the stage-2 tiles (same format, same code).
  RF recall
    model      the stage-2 CNN recipe trained on all CardRF tiles (GPU)
    threshold  rejects 90% of CardRF's negative tiles (fixed on CardRF before scoring the pool)
    reported   share of pool tiles above it, per dataset and per band (2.4 vs 5.8 GHz)
  acoustic one-class
    pool       drone windows only from DDL, Mięsikowska, DroneNoise (level-normalised features.py features)
    models     as RF; tests UaVirBASE and Svanström (both classes, one chain each)
  decision     a one-class test "works" if its interval is above 0.5 (and p < 0.05 where groups allow)

Parallel CPU (12 workers, RAM-capped) for tiles and features; GPU for the CNN.
Outputs: DroneacharyaData/results/merged_one_class/*.json, docs/results/merged_one_class.md.
"""
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from droneacharya import cnn  # noqa: E402
from droneacharya import evaluation as E  # noqa: E402
from droneacharya import splits  # noqa: E402
from droneacharya.cache import key  # noqa: E402
from droneacharya.index import read  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

OUT = DATA_ROOT / "results" / "merged_one_class"
STAGE2 = DATA_ROOT / "results" / "rf_stage2"
REPORT = ROOT / "docs" / "results" / "merged_one_class.md"
FS = 14e6
BLOCK100 = 25_000          # 250 us at 100 MS/s
WORKERS = 12


# ---------------------------------------------------------------- RF pool
def pool_jobs():
    jobs = []
    for c in read("rfuav", "captures").to_pylist():
        parts = c["n_artifacts"]
        base = int(round(c["start_s"] * 1e8)) if c["start_s"] else 0
        jobs.append(("rfuav", c["capture_id"], c["group_id"],
                     [p * 100_000_000 + k * BLOCK100 for p in range(parts) for k in range(4)], base))
    for c in read("rma", "captures").to_pylist():
        jobs.append(("rma", c["capture_id"], c["group_id"], [0, BLOCK100], 0))
    for c in read("drff_r2", "captures").to_pylist():
        subset = c["capture_id"].split("/")[1]
        if subset.startswith(("dataset1", "dataset2", "dataset3", "dataset4", "dataset5")):
            jobs.append(("drff_r2", c["capture_id"], c["group_id"], [0, BLOCK100], 0))
    return jobs


def _pool_rows(dataset, capture_id, group, starts, _):
    with threadpool_limits(1):
        from droneacharya import rf_common as R
        from droneacharya.rf_features import rf_tile_features
        tiles, fc = R.iq_steered(capture_id, starts, width=R.STAGE2_WIDTH)
        return [{"dataset": dataset, "capture_id": capture_id, "group": group,
                 "band": "5.8 GHz" if fc > 5e9 else "2.4 GHz", **rf_tile_features(t, FS)} for t, _ in tiles], \
               [t for t, _ in tiles]


def rf_pool():
    jobs = pool_jobs()
    tag = key(_pool_rows, {"jobs": len(jobs), "format": "stage2 14 MHz"}, ["rfuav", "rma", "drff_r2"])
    rows_path, tiles_path = OUT / f"rf_pool-{tag}.features.json", OUT / f"rf_pool-{tag}.tiles.npy"
    if not rows_path.exists():
        out = Parallel(n_jobs=WORKERS)(delayed(_pool_rows)(*job) for job in jobs)
        rows_path.write_text(json.dumps([r for rows, _ in out for r in rows]), encoding="utf-8")
        np.save(tiles_path, np.stack([t for _, tiles in out for t in tiles]))
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    print(f"  RF pool: {len(rows)} drone tiles", flush=True)
    return rows, np.load(tiles_path)


def _uavsig_rows(capture_id):
    with threadpool_limits(1):
        from droneacharya import rf_common as R
        from droneacharya.rf_features import rf_tile_features
        tiles, _ = R.uavsig_steered(capture_id, n_tiles=4, width=R.STAGE2_WIDTH)
        return [{"capture_id": capture_id, **rf_tile_features(t, FS)} for t in tiles]


def uavsig_test():
    captures = {c["capture_id"]: c for c in splits.captures_of("uavsig")}
    tag = key(_uavsig_rows, {"tiles": 4, "format": "stage2 14 MHz"}, ["uavsig"])
    path = OUT / f"uavsig-{tag}.features.json"
    if not path.exists():
        out = Parallel(n_jobs=4)(delayed(_uavsig_rows)(c) for c in sorted(captures))   # ~800 MB per capture
        path.write_text(json.dumps([r for rows in out for r in rows]), encoding="utf-8")
    rows = json.loads(path.read_text(encoding="utf-8"))
    for r in rows:
        r["y"], r["group"] = captures[r["capture_id"]]["label_uas_present"], captures[r["capture_id"]]["group_id"]
    return rows


def stage2_set(dataset):
    """CardRF / Noisy RF tiles and features from the stage-2 run (identical format and code)."""
    captures = sorted(c["capture_id"] for c in splits.captures_of(dataset))
    feats = json.loads(sorted(STAGE2.glob(f"{dataset}-*.features.json"))[-1].read_text(encoding="utf-8"))
    tiles = np.load(sorted(STAGE2.glob(f"{dataset}-*.tiles.npy"))[-1], mmap_mode="r")
    meta = {c["capture_id"]: c for c in splits.captures_of(dataset)}
    return captures, feats, tiles, meta


def names_of(rows):
    return sorted(k for k in rows[0] if k.startswith(("shape_", "occupied", "active", "spectral", "duty", "transitions",
                                                      "frame_", "centroid", "amplitude", "papr")))


# ---------------------------------------------------------------- one-class models
def one_class_scores(pool_X, test_X):
    scaler = StandardScaler().fit(pool_X)
    p, t = scaler.transform(pool_X), scaler.transform(test_X)
    knn = NearestNeighbors(n_neighbors=10).fit(p)
    distance = knn.kneighbors(t)[0].mean(axis=1)
    gmm = GaussianMixture(n_components=8, covariance_type="diag", random_state=0).fit(p)
    return {"knn": -distance, "gmm": gmm.score_samples(t)}


def scored(name, scores, y, groups, model, transmitter_only=False, extra=None):
    result = E.evaluate_external(scores, y, groups, model, n_permutations=1 if transmitter_only else 2000)
    if transmitter_only:
        result.pop("permutation")
        result["interval_note"] = "resamples the 6 RC transmitters only (no recording IDs); no p"
    result.update(extra or {})
    result["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    p = result.get("permutation", {}).get("p_value")
    print(f"{name}: AUC {result['roc_auc']:.3f} CI {[round(v, 3) for v in result['ci95']]}" +
          (f" p={p:.3f}" if p is not None else ""), flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    results = {}
    acoustic_only = "--acoustic-only" in sys.argv     # the RF half is saved; rerun only the acoustic half
    if acoustic_only:
        return acoustic(results, start)

    pool, pool_tiles = rf_pool()
    names = names_of(pool)
    pool_X = np.array([[r[k] for k in names] for r in pool])

    c_ids, c_feats, c_tiles, c_meta = stage2_set("cardrf")
    n_ids, n_feats, n_tiles, n_meta = stage2_set("noisy_rf")
    u_rows = uavsig_test()
    c_X = np.array([[f[k] for k in names] for f in c_feats])
    n_X = np.array([[f[k] for k in names] for f in n_feats])
    u_X = np.array([[r[k] for k in names] for r in u_rows])
    c_y = np.array([c_meta[i]["label_uas_present"] for i in c_ids])
    c_g = np.array([c_meta[i]["group_id"] for i in c_ids])
    n_y = np.array([n_meta[i]["label_uas_present"] for i in n_ids])
    n_cls = np.array([n_meta[i]["label_class"] for i in n_ids])
    u_y, u_g = np.array([r["y"] for r in u_rows]), np.array([r["group"] for r in u_rows])

    for test, X, y, g, only in (("cardrf", c_X, c_y, c_g, False), ("uavsig", u_X, u_y, u_g, False),
                                ("noisy_rf", n_X, n_y, np.where(n_y, n_cls, "noise"), True)):
        scores = one_class_scores(pool_X, X)
        for model in ("knn", "gmm"):
            results[f"rf {model} → {test}"] = scored(f"rf_{model}_to_{test}", scores[model], y, g, model, only, {
                "task": f"merged drone-only RF pool → {test}", "pool_tiles": len(pool),
                "pool_by_dataset_band": {f"{d} {b}": sum(1 for r in pool if r["dataset"] == d and r["band"] == b)
                                         for d in ("rfuav", "rma", "drff_r2") for b in ("2.4 GHz", "5.8 GHz")}})

    # RF recall of the CardRF-trained CNN on every drone in the pool
    c_spec = cnn.spectrograms(np.asarray(c_tiles))
    net = cnn.train(c_spec, c_y)
    c_scores = cnn.predict(net, c_spec)
    threshold = float(np.quantile(c_scores[~c_y], 0.90))
    p_scores = np.concatenate([cnn.predict(net, cnn.spectrograms(pool_tiles[i:i + 8192]))
                               for i in range(0, len(pool_tiles), 8192)])
    recall = {}
    for d in ("rfuav", "rma", "drff_r2"):
        for b in ("2.4 GHz", "5.8 GHz"):
            sel = np.array([r["dataset"] == d and r["band"] == b for r in pool])
            if sel.sum():
                recall[f"{d} {b}"] = {"tiles": int(sel.sum()), "recall": float(np.mean(p_scores[sel] > threshold))}
    results["rf cnn recall"] = {"threshold_rejects_cardrf_negatives": 0.90, "threshold": threshold, "recall": recall}
    (OUT / "rf_cnn_recall.json").write_text(json.dumps(results["rf cnn recall"], indent=1), encoding="utf-8")
    print("rf cnn recall:", {k: round(v["recall"], 3) for k, v in recall.items()}, flush=True)

    acoustic(results, start)


def acoustic(results, start):
    """Acoustic one-class: drone-only pool from DDL, Mięsikowska, DroneNoise; tests UaVirBASE and Svanström."""
    spec = importlib.util.spec_from_file_location("widening", ROOT / "scripts" / "run_acoustic_widening.py")
    widening = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(widening)
    widening.OUT.mkdir(parents=True, exist_ok=True)
    a = {d: widening.dataset_rows(d) for d in ("ddl", "miesikowska_uav", "dronenoise", "uavirbase", "svanstrom")}
    a_names = sorted(k for k in a["ddl"][0] if k.startswith(("band_", "peakiness", "spectral", "modulation")))
    a_pool = np.array([[r[k] for k in a_names] for d in ("ddl", "miesikowska_uav", "dronenoise") for r in a[d] if r["y"]])
    for test in ("uavirbase", "svanstrom"):
        X = np.array([[r[k] for k in a_names] for r in a[test]])
        y, g = np.array([r["y"] for r in a[test]]), np.array([r["group"] for r in a[test]])
        scores = one_class_scores(a_pool, X)
        for model in ("knn", "gmm"):
            results[f"acoustic {model} → {test}"] = scored(f"acoustic_{model}_to_{test}", scores[model], y, g, model,
                                                           extra={"task": f"merged drone-only acoustic pool → {test}",
                                                                  "pool_windows": len(a_pool)})

    for path in sorted(OUT.glob("rf_*_to_*.json")):
        name = path.stem.replace("rf_", "rf ").replace("_to_", " → ")
        results.setdefault(name, json.loads(path.read_text(encoding="utf-8")))
    if "rf cnn recall" not in results and (OUT / "rf_cnn_recall.json").exists():
        results["rf cnn recall"] = json.loads((OUT / "rf_cnn_recall.json").read_text(encoding="utf-8"))
    lines = ["<!-- Generated by scripts/run_merged_one_class.py; results in DroneacharyaData/results/merged_one_class/. -->",
             "", "# The merged drone-only model (one-class), RF and acoustic", "",
             "Design fixed before results (script docstring). Training sees drones only; every test set holds both",
             "classes from one receiver/chain and was never in training. Noisy RF intervals cover transmitter variability only.", "",
             "| Pool → test | Model | AUC [95% CI] | p | Groups by label |", "| --- | --- | --- | --- | --- |"]
    for name, r in results.items():
        if "roc_auc" in r:
            p = r.get("permutation", {}).get("p_value")
            lines.append(f"| {name.rsplit(' ', 2)[0].split(' ', 1)[0]} pool → {name.rsplit(' ', 1)[-1]} | {r['model']} | "
                         f"{r['roc_auc']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}] | " +
                         (f"{p:.3f}" if p is not None else "—") + f" | {r['groups_per_label']} |")
    rec = results["rf cnn recall"]
    lines += ["", f"**RF recall** of the CardRF-trained CNN at the threshold rejecting 90% of CardRF negatives "
              f"(share of drone tiles flagged):", ""]
    lines += [f"- {k}: {v['recall']:.2f} ({v['tiles']} tiles)" for k, v in rec["recall"].items()]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done in {(time.time() - start) / 60:.1f} min; report {REPORT}")


if __name__ == "__main__":
    main()
