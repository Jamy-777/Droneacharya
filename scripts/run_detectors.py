"""Classical detectors (gradient boosting on level-normalised features), scored like the baselines.

  RF        CardRF, UAS vs Wi-Fi/Bluetooth, unseen UAS systems and devices (cardrf/system-kfold-s0);
            per emitter type and on low-clipping captures; paired against the clipping+energy baseline
  acoustic  Svanström and UaVirBASE within-chain (saved splits), paired against the loudness baseline;
            external: train on one dataset, test on the other (both test classes share one chain)

Same rows and folds as the baselines, so every "beats the baseline" is a paired, group-resampled
difference. Outputs: DroneacharyaData/results/detectors/<name>.json and docs/results/detectors.md.

Usage:
    python scripts/run_detectors.py
"""
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya import baselines as B  # noqa: E402
from droneacharya import evaluation as E  # noqa: E402
from droneacharya import features as F  # noqa: E402
from droneacharya import splits  # noqa: E402
from droneacharya.cache import cached  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

OUT = DATA_ROOT / "results" / "detectors"
REPORT = Path(__file__).resolve().parents[1] / "docs" / "results" / "detectors.md"
SEED = 0
CARDRF_PER_GROUP = 60          # identical selection to the baselines
N_PERMUTATIONS = 200


def save(name, result):
    result["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{name}: AUC {result['roc_auc']:.3f} CI {[round(v, 3) for v in result['ci95']]} "
          f"p={result['permutation']['p_value']:.3f}" + (f"; vs baseline {result['vs_baseline']['difference']:+.3f} "
          f"CI {[round(v, 3) for v in result['vs_baseline']['ci95']]}" if "vs_baseline" in result else ""), flush=True)
    return result


def matrix(rows, prefix=("band_", "occupied", "spectral", "envelope", "peakiness", "modulation")):
    names = sorted(k for k in rows[0] if k.startswith(prefix))
    return np.array([[r[k] for k in names] for r in rows]), names


# ---------------------------------------------------------------- RF
def cardrf():
    split_id = "cardrf/system-kfold-s0"
    assignment = splits.load(split_id)
    captures = {c["capture_id"]: c for c in splits.captures_of("cardrf")}
    by_group = defaultdict(list)
    for c in captures.values():
        by_group[c["group_id"]].append(c)

    def compute():
        rng = random.Random(SEED)
        chosen = [c for g in sorted(by_group) for c in rng.sample(sorted(by_group[g], key=lambda c: c["capture_id"]),
                                                                  min(CARDRF_PER_GROUP, len(by_group[g])))]
        rows = []
        for i, c in enumerate(chosen, 1):
            rows.append({"capture_id": c["capture_id"], **F.cardrf_detector_features(c["capture_id"]),
                         **{f"baseline_{k}": v for k, v in B.cardrf_features(c["capture_id"]).items()}})
            if i % 100 == 0:
                print(f"  cardrf features {i}/{len(chosen)}", flush=True)
        return rows

    rows = cached(OUT, "cardrf", compute, {"per_group": CARDRF_PER_GROUP, "seed": SEED}, ["cardrf"])
    X, names = matrix(rows)
    y = np.array([captures[r["capture_id"]]["label_uas_present"] for r in rows])
    parts = np.array([assignment[r["capture_id"]] for r in rows])
    groups = np.array([captures[r["capture_id"]]["group_id"] for r in rows])
    emitter = np.array([captures[r["capture_id"]]["label_emitter"] for r in rows])
    result, scores = E.evaluate(X, y, parts, groups, "gbm", n_permutations=N_PERMUTATIONS, return_scores=True)
    bx = np.array([[r[f"baseline_{k}"] for k in ("clip_fraction", "log_rms_pre_trigger", "log_rms_post_trigger")]
                   for r in rows])
    baseline_scores = E.out_of_fold(bx, y, parts, "tree")
    clip = np.array([r["baseline_clip_fraction"] for r in rows])
    negative = ~y
    result.update({
        "task": "UAS vs Wi-Fi/Bluetooth from post-trigger, unit-RMS spectral shape (unseen UAS systems and devices)",
        "split": split_id, "features": names,
        "baseline": "clipping fraction + pre/post-trigger RMS (depth-3 tree), same rows and folds",
        "baseline_roc_auc": E.metric(y, baseline_scores),
        "vs_baseline": E.paired_difference(scores, baseline_scores, y, groups),
        "per_emitter": {kind: E.subset_auc(scores, y, groups, (emitter == kind) | negative)
                        for kind in ("aircraft", "controller")},
        "low_clipping": E.subset_auc(scores, y, groups, clip < 0.01) if len(set(y[clip < 0.01])) == 2 else None,
        "low_clipping_rule": "captures with < 1% of samples at the ADC rails",
    })
    return {"cardrf_detector": save("cardrf_detector", result)}


# ---------------------------------------------------------------- acoustic
def audio_rows(dataset, max_samples=None):
    captures = {c["capture_id"]: c for c in splits.captures_of(dataset) if c["label_uas_present"] is not None}

    def compute():
        rows = []
        for c in captures.values():
            windows, rate = B.audio_windows(c["capture_id"], max_samples=max_samples)
            rows += [{"capture_id": c["capture_id"], "dbfs": B.level_dbfs(w), **F.audio_detector_features(w, rate)}
                     for w in windows]
        return rows

    rows = cached(OUT, dataset, compute, {"max_samples": max_samples, "window_s": 1.0, "max_windows": 10}, [dataset])
    X, names = matrix(rows)
    y = np.array([captures[r["capture_id"]]["label_uas_present"] for r in rows])
    groups = np.array([captures[r["capture_id"]]["group_id"] for r in rows])
    level = np.array([[r["dbfs"]] for r in rows])
    return rows, X, names, y, groups, level


def within(dataset, data):
    rows, X, names, y, groups, level = data
    split_id = f"{dataset}/group-kfold-s0"
    assignment = splits.load(split_id)
    parts = np.array([assignment[r["capture_id"]] for r in rows])
    result, scores = E.evaluate(X, y, parts, groups, "gbm", n_permutations=N_PERMUTATIONS, return_scores=True)
    baseline_scores = E.out_of_fold(level, y, parts, "tree")
    result.update({"task": "drone vs non-drone, 1 s windows, level-normalised features, unseen groups (within chain)",
                   "split": split_id, "features": names, "baseline": "loudness (depth-3 tree), same rows and folds",
                   "baseline_roc_auc": E.metric(y, baseline_scores),
                   "vs_baseline": E.paired_difference(scores, baseline_scores, y, groups)})
    return save(f"{dataset}_detector", result)


def external(train_name, train, test_name, test):
    _, X_train, names, y_train, _, level_train = train
    _, X_test, names_test, y_test, groups_test, level_test = test
    assert names == names_test
    scores = E.positive_scores(E.model("gbm").fit(X_train, y_train), X_test)
    baseline_scores = E.positive_scores(E.model("tree").fit(level_train, y_train), level_test)
    result = E.evaluate_external(scores, y_test, groups_test, "gbm")
    result.update({"task": f"trained on all of {train_name}, tested on {test_name} (both test classes from one chain)",
                   "split": f"external: {train_name} → {test_name}", "features": names,
                   "baseline": "loudness (depth-3 tree) trained the same way",
                   "baseline_roc_auc": E.metric(y_test, baseline_scores),
                   "vs_baseline": E.paired_difference(scores, baseline_scores, y_test, groups_test)})
    return save(f"external_{train_name}_to_{test_name}", result)


def write_report(results):
    lines = ["<!-- Generated by scripts/run_detectors.py; results in DroneacharyaData/results/detectors/. -->", "",
             "> Raw output, rewritten on every run. Which numbers may be quoted, and with what limits: [claims.md](claims.md).", "",
             "# Classical detectors", "",
             "Gradient boosting (depth 3, fixed in advance) on level-normalised features: RF uses the post-trigger half",
             "of each CardRF capture at unit RMS (spectral shape, occupied bandwidth, flatness, envelope); acoustic uses",
             "1 s windows at 16 kHz and unit RMS (band shape, tonal peakiness per octave, flatness, centroid, modulation).",
             "Same rows and folds as the shortcut baselines. AUC is pooled out of fold; **p** from 200 group-level label",
             "permutations (2,000 label permutations for external tests); intervals resample whole groups. **vs baseline**",
             "is the paired AUC difference on identical rows with its group-bootstrap interval: the detector beats the",
             "baseline only if that interval is above zero.", "",
             "| Detector | Task | Groups by label | AUC [95% CI] | p | Baseline AUC | Detector − baseline [95% CI] |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in results.items():
        groups = ", ".join(f"{k}: {v}" for k, v in sorted(r["groups_per_label"].items()))
        d = r["vs_baseline"]
        lines.append(f"| {name} | {r['task']} | {groups} | {r['roc_auc']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}] | "
                     f"{r['permutation']['p_value']:.3f} | {r['baseline_roc_auc']:.2f} | "
                     f"{d['difference']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}] |")
    rf = results["cardrf_detector"]
    lines += ["", "CardRF breakdown (pooled out-of-fold scores, all negatives kept):", ""]
    for kind, sub in rf["per_emitter"].items():
        lines.append(f"- {kind} vs Wi-Fi/Bluetooth: AUC {sub['roc_auc']:.2f} [{sub['ci95'][0]:.2f}, {sub['ci95'][1]:.2f}] "
                     f"({sub['groups_per_label']})")
    low = rf["low_clipping"]
    lines.append(f"- captures with < 1% clipped samples: " + (f"AUC {low['roc_auc']:.2f} [{low['ci95'][0]:.2f}, "
                 f"{low['ci95'][1]:.2f}] ({low['n']} captures, {low['groups_per_label']})" if low else "too few to score"))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    results = {}
    results.update(cardrf())
    svanstrom = audio_rows("svanstrom")
    uavirbase = audio_rows("uavirbase", max_samples=10 * 96000)
    results["svanstrom_detector"] = within("svanstrom", svanstrom)
    results["uavirbase_detector"] = within("uavirbase", uavirbase)
    results["external_uavirbase_to_svanstrom"] = external("uavirbase", uavirbase, "svanstrom", svanstrom)
    results["external_svanstrom_to_uavirbase"] = external("svanstrom", svanstrom, "uavirbase", uavirbase)
    write_report(results)
    print(f"done in {(time.time() - start) / 60:.1f} min; report {REPORT}")


if __name__ == "__main__":
    main()
