"""RF detection across receivers: the two datasets that hold both classes, each tested on the other.

Design fixed before any result was seen (2026-10-08):

  representation  steered 25 MHz sub-band tiles (rf_common v1.1: complex baseband, 25 MS/s, 250 us,
                  steered to the strongest activity the receiver sees in 2.39–2.4935 GHz, DC notched),
                  position-invariant unit-RMS features (rf_features) — no absolute frequency, no power
  data            CardRF: every capture (9,600), one tile each (oscilloscope; UAS vs Wi-Fi/Bluetooth)
                  UAVSig: 360 local captures x 20 tiles (USRP B205mini; UAS vs no-transmitter ambient)
  model           gbm (evaluation.model), fixed in advance
  primary tests   T1 train UAVSig -> test CardRF (13 groups: 6 UAS systems, 7 Wi-Fi/BT devices)
                  T2 train CardRF -> test UAVSig (120 groups, only 2 no-transmitter scenarios: weak)
                  T1 carries the claim; T2 is reported but cannot be strong with 2 negative groups
  baseline        the tile's received power (log RMS before normalisation, depth-3 tree), trained the same
                  way, paired on identical test rows
  references      within-chain cross-validation on each dataset's saved split; dataset identification
                  (CardRF vs UAVSig) on a saved split, to show how distinguishable the receivers are
  decision rule   a direction is "established" if its 95% interval is above 0.5 and p < 0.05; it "beats the
                  power shortcut" if the paired difference's interval is above zero

Tile extraction runs in parallel: CardRF on every core, UAVSig on 4 workers (~800 MB per capture).
Outputs: DroneacharyaData/results/rf_cross/*.json, docs/results/rf_cross.md.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya import evaluation as E  # noqa: E402
from droneacharya import splits  # noqa: E402
from droneacharya.cache import cached  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

# v1.1 (20 MHz tiles) is the pre-registered run. v1.2 (40 MHz tiles) is the single revision allowed on these test
# sets, fixed before running from a physical diagnosis of v1.1: DJI video links (36 MHz measured) are wider than a
# 25 MHz tile, so they fill it edge to edge and look like noise. Nothing else changes. No further revisions on T1.
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v1.1"
WIDTH = {"v1.1": 25e6, "v1.2": 40e6}[VERSION]
TILE_FS = {"v1.1": 25e6, "v1.2": 40e6}[VERSION]
OUT = DATA_ROOT / "results" / ("rf_cross" if VERSION == "v1.1" else f"rf_cross_{VERSION}")
REPORT = Path(__file__).resolve().parents[1] / "docs" / "results" / ("rf_cross.md" if VERSION == "v1.1" else f"rf_cross_{VERSION}.md")
UAVSIG_TILES = 20
DATASET_ID_SPLIT = "rf/dataset-id-s0"   # same split for both versions (identical captures)


def _tile_rows(dataset, capture_id):
    with threadpool_limits(1):
        from droneacharya import rf_common as R
        from droneacharya.rf_features import rf_tile_features
        if dataset == "cardrf":
            tile, centre = R.cardrf_steered(capture_id, width=WIDTH if VERSION != "v1.1" else R.SUB_FS)
            tiles, centres = [tile], [centre]
        else:
            tiles, centres = R.uavsig_steered(capture_id, UAVSIG_TILES, width=WIDTH if VERSION != "v1.1" else R.SUB_FS)
        return [{"capture_id": capture_id, "tile": i, "centre_hz": c,
                 "log_rms": float(np.log10(np.sqrt(np.mean(np.abs(t.astype(np.complex128)) ** 2)) + 1e-12)),
                 **rf_tile_features(t, TILE_FS if VERSION != "v1.1" else 25e6)} for i, (t, c) in enumerate(zip(tiles, centres))]


def dataset_rows(dataset, n_jobs):
    captures = {c["capture_id"]: c for c in splits.captures_of(dataset)}

    def compute():
        chunks = Parallel(n_jobs=n_jobs)(delayed(_tile_rows)(dataset, cid) for cid in sorted(captures))
        return [r for chunk in chunks for r in chunk]

    rows = cached(OUT, dataset, compute, {"uavsig_tiles": UAVSIG_TILES, "format": f"rf_common {VERSION}"}, [dataset])
    for r in rows:
        c = captures[r["capture_id"]]
        r["y"], r["group"], r["dataset"] = c["label_uas_present"], c["group_id"], dataset
    print(f"  {dataset}: {len(rows)} tiles", flush=True)
    return rows


def matrix(rows):
    names = sorted(k for k in rows[0] if k.startswith(("shape_", "occupied", "active", "spectral", "duty", "transitions",
                                                        "frame_", "centroid", "amplitude", "papr")))
    return (np.array([[r[k] for k in names] for r in rows]), np.array([r["y"] for r in rows]),
            np.array([r["group"] for r in rows]), np.array([[r["log_rms"]] for r in rows]), names)


def save(name, result):
    result["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    m = result["metric"]
    d = result.get("vs_baseline")
    print(f"{name}: {m} {result[m]:.3f} CI {[round(v, 3) for v in result['ci95']]} p={result['permutation']['p_value']:.3f}"
          + (f"; vs power {d['difference']:+.3f} CI {[round(v, 3) for v in d['ci95']]}" if d else ""), flush=True)
    return result


def cross(train_name, train, test_name, test):
    X_train, y_train, _, p_train, names = matrix(train)
    X_test, y_test, g_test, p_test, _ = matrix(test)
    scores = E.positive_scores(E.model("gbm").fit(X_train, y_train), X_test)
    baseline = E.positive_scores(E.model("tree").fit(p_train, y_train), p_test)
    result = E.evaluate_external(scores, y_test, g_test, "gbm")
    result.update({"task": f"UAS present, trained on {train_name}, tested on {test_name}", "features": names,
                   "baseline_roc_auc": E.metric(y_test, baseline),
                   "vs_baseline": E.paired_difference(scores, baseline, y_test, g_test)})
    if test_name == "cardrf":
        captures = {c["capture_id"]: c for c in splits.captures_of("cardrf")}
        emitter = np.array([captures[r["capture_id"]]["label_emitter"] for r in test])
        result["per_emitter"] = {k: E.subset_auc(scores, y_test, g_test, (emitter == k) | ~y_test)
                                 for k in ("aircraft", "controller")}
        result["per_negative"] = {k: E.subset_auc(scores, y_test, g_test, (emitter == k) | y_test)
                                  for k in ("wifi", "bluetooth")}
    return save(f"cross_{train_name}_to_{test_name}", result)


def within(dataset, rows, split_id):
    assignment = splits.load(split_id)
    X, y, g, _, names = matrix(rows)
    parts = np.array([assignment[r["capture_id"]] for r in rows])
    result = E.evaluate(X, y, parts, g, "gbm", n_permutations=200)
    result.update({"task": f"UAS present within {dataset} (reference, same receiver)", "split": split_id})
    return save(f"within_{dataset}", result)


def dataset_id(cardrf, uavsig):
    captures = splits.captures_of("cardrf") + splits.captures_of("uavsig")
    assignment = splits.group_kfold(captures, 5, seed=0, label="dataset_id")
    splits.save(DATASET_ID_SPLIT, "experiment", "group k-fold over CardRF + UAVSig, k=5, dataset-balanced",
                captures, assignment, seed=0, notes="for RF dataset-identification reference")
    assignment = splits.load(DATASET_ID_SPLIT)
    rows = cardrf + uavsig
    X, _, g, _, _ = matrix(rows)
    labels = np.array([r["dataset"] == "cardrf" for r in rows])   # True = CardRF tile
    result = E.evaluate(X, labels, np.array([assignment[r["capture_id"]] for r in rows]), g, "gbm", n_permutations=50)
    result.update({"task": "which receiver/dataset a tile comes from (CardRF vs UAVSig, AUC), unseen groups",
                   "split": DATASET_ID_SPLIT, "chance_balanced_accuracy": 0.5})
    return save("dataset_id", result)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    cardrf = dataset_rows("cardrf", n_jobs=-1)
    uavsig = dataset_rows("uavsig", n_jobs=4)
    results = {
        "T1 UAVSig → CardRF": cross("uavsig", uavsig, "cardrf", cardrf),
        "T2 CardRF → UAVSig": cross("cardrf", cardrf, "uavsig", uavsig),
        "within CardRF (reference)": within("cardrf", cardrf, "cardrf/system-kfold-s0"),
        "within UAVSig (reference)": within("uavsig", uavsig, "uavsig/group-kfold-s0"),
        "dataset identification": dataset_id(cardrf, uavsig),
    }
    lines = [f"<!-- Generated by scripts/run_rf_cross.py; results in DroneacharyaData/results/{OUT.name}/. -->", "",
             "> Raw output, rewritten on every run. Which numbers may be quoted, and with what limits: [claims.md](claims.md).", "",
             f"# RF detection across receivers ({VERSION})", "",
             f"Design and decision rules fixed before results (script docstring). Steered {WIDTH / 1e6:.0f} MHz tiles, position-invariant",
             "unit-RMS features, gradient boosting. T1 carries the claim; T2 has only 2 negative groups.", "",
             "| Test | Metric [95% CI] | p | Groups by label | Power baseline | Detector − power [95% CI] |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, r in results.items():
        m, d = r["metric"], r.get("vs_baseline")
        lines.append(f"| {name} | {r[m]:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}] ({m}) | {r['permutation']['p_value']:.3f} | "
                     f"{r['groups_per_label']} | " + (f"{r['baseline_roc_auc']:.2f} | {d['difference']:+.2f} "
                     f"[{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]" if d else "— | —") + " |")
    t1 = results["T1 UAVSig → CardRF"]
    lines += ["", "T1 breakdown (all negatives kept for emitter rows, all positives kept for negative rows):", ""]
    for k, sub in {**t1["per_emitter"], **t1["per_negative"]}.items():
        lines.append(f"- {k}: AUC {sub['roc_auc']:.2f} [{sub['ci95'][0]:.2f}, {sub['ci95'][1]:.2f}] ({sub['groups_per_label']})")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done in {(time.time() - start) / 60:.1f} min; report {REPORT}")


if __name__ == "__main__":
    main()
