"""Acoustic: does a single feature chosen on the training chain transfer, does the detector add anything beyond it,
and does a threshold set on the training chain hold on the other chain? (claims.md A1, open question 1)

Design committed before running (2026-10-08). Motivation: inside each test set, a 250 Hz band near 7 kHz chosen with
hindsight matched (0.89 vs 0.85) or beat (0.85 vs 0.68) the detector (results/checks/acoustic_gate.json). This test
removes the hindsight: every choice is made on the training chain only.

  rows       the cached 1 s windows of scripts/run_detectors.py (same windows as A1); directions UaVirBASE → Svanström
             and Svanström → UaVirBASE
  rule       the single feature among the 41 detector features + loudness (dBFS) with the largest |AUC − 0.5| on all
             training-chain windows; its direction is taken from the training chain. Applied unchanged to the test
             chain. Whether it lands at 6.75–7.25 kHz is reported, not required.
  detector   the external gbm of run_detectors.py, refitted identically (checked against the saved AUC)
  threshold  the 95th percentile of training-chain non-drone scores: for the detector, its out-of-fold scores on the
             training chain's saved within-chain split (<train>/group-kfold-s0); for the rule, the feature values
             themselves. Applied unchanged to the test chain.
  alarms     an alarm starts when 2 of 3 consecutive windows of one recording exceed the threshold. False alarms per
             hour = alarm onsets in test-chain non-drone recordings / their total duration, with the one-sided 95%
             Poisson upper bound. Indicative only: the cached windows cover the first ≤ 10 s of each recording.
  decisions  the rule "transfers" if its test AUC interval is above 0.5 and p < 0.05 (test labels permuted between
             groups, 2,000).
             The detector "adds beyond the rule" if the paired AUC difference detector − rule has an interval above
             zero (groups resampled).
             A threshold "transfers" if, on the test chain, the window false-positive rate is ≤ 10% (twice the 5%
             set on training) and drone-window recall is ≥ 50%.
  deferred   a frequency ablation with features recomputed from audio low-passed at 6 kHz, and false alarms per hour
             over the full ambient recordings: both re-read the audio and get their own design.

CPU only; cached features, no audio is read.
Output: DroneacharyaData/results/checks/acoustic_single_feature_transfer.json
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya import evaluation as E  # noqa: E402
from droneacharya import splits  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

DETECTORS = DATA_ROOT / "results" / "detectors"
OUT = DATA_ROOT / "results" / "checks"
PREFIX = ("band_", "occupied", "spectral", "envelope", "peakiness", "modulation")
TARGET_FPR, MAX_FPR, MIN_RECALL = 0.05, 0.10, 0.50
K, N = 2, 3                      # alarm: K of N consecutive windows above threshold


def load(dataset):
    rows = json.loads(sorted(DETECTORS.glob(f"{dataset}-*.features.json"))[-1].read_text(encoding="utf-8"))
    captures = {c["capture_id"]: c for c in splits.captures_of(dataset) if c["label_uas_present"] is not None}
    names = sorted(k for k in rows[0] if k.startswith(PREFIX))
    return {"X": np.array([[r[k] for k in names] for r in rows]), "names": names,
            "ids": np.array([r["capture_id"] for r in rows]),
            "y": np.array([captures[r["capture_id"]]["label_uas_present"] for r in rows]),
            "groups": np.array([captures[r["capture_id"]]["group_id"] for r in rows]),
            "dbfs": np.array([r["dbfs"] for r in rows])}


def columns(d):
    return {**{n: d["X"][:, j] for j, n in enumerate(d["names"])}, "dbfs": d["dbfs"]}


def rate_ci(flags, y_keep, groups, n=2000, seed=0):
    """Share of flagged rows among the kept rows, with an interval resampling groups."""
    _, _, members = E._groups(groups[y_keep], np.zeros(y_keep.sum(), bool))
    f = flags[y_keep]
    rng = np.random.default_rng(seed)
    values = [f[np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])].mean() for _ in range(n)]
    return {"rate": float(f.mean()), "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]}


def alarms(flags, ids, y):
    """Alarm onsets per recording (K of N consecutive windows), rows of one recording in time order."""
    onsets, seconds, recordings_alarmed = {False: 0, True: 0}, {False: 0, True: 0}, {False: [], True: []}
    for cid in dict.fromkeys(ids):
        sel = ids == cid
        f, label = flags[sel], bool(y[sel][0])
        active = np.array([f[max(0, i - N + 1):i + 1].sum() >= K for i in range(len(f))])
        starts = int(np.sum(active & ~np.r_[False, active[:-1]]))
        onsets[label] += starts
        seconds[label] += len(f)
        recordings_alarmed[label].append(starts > 0)
    hours = seconds[False] / 3600
    return {"false_alarm_onsets": onsets[False], "non_drone_seconds": seconds[False],
            "false_alarms_per_hour": onsets[False] / hours,
            "false_alarms_per_hour_upper95": float(chi2.ppf(0.95, 2 * (onsets[False] + 1)) / 2 / hours),
            "drone_recordings_with_alarm": float(np.mean(recordings_alarmed[True])),
            "rule": f"{K} of {N} consecutive 1 s windows"}


def operating_point(test_scores, threshold, test):
    flags = test_scores > threshold
    return {"threshold": float(threshold),
            "window_false_positive_rate": rate_ci(flags, ~test["y"], test["groups"]),
            "window_recall": rate_ci(flags, test["y"], test["groups"]),
            "alarms": alarms(flags, test["ids"], test["y"])}


def transfer(train_name, train, test_name, test, saved_auc):
    # the single-feature rule, chosen on the training chain only
    train_cols, test_cols = columns(train), columns(test)
    train_auc = {n: E.metric(train["y"], v) for n, v in train_cols.items()}
    feature = max(train_auc, key=lambda n: abs(train_auc[n] - 0.5))
    direction = 1 if train_auc[feature] >= 0.5 else -1
    rule_test = direction * test_cols[feature]
    rule = E.evaluate_external(rule_test, test["y"], test["groups"], "single feature")
    rule.update({"feature": feature, "direction": direction, "train_auc": train_auc[feature],
                 "near_7khz": feature in ("band_27", "band_28")})
    # the detector, refitted as run_detectors.py did
    fitted = E.model("gbm").fit(train["X"], train["y"])
    detector_test = E.positive_scores(fitted, test["X"])
    assignment = splits.load(f"{train_name}/group-kfold-s0")
    parts = np.array([assignment[i] for i in train["ids"]])
    detector_oof = E.out_of_fold(train["X"], train["y"], parts, "gbm")
    rule_train = direction * train_cols[feature]
    result = {
        "rule": rule,
        "detector_auc": E.metric(test["y"], detector_test), "saved_detector_auc": saved_auc,
        "detector_minus_rule": E.paired_difference(detector_test, rule_test, test["y"], test["groups"]),
        "detector_threshold": operating_point(detector_test, np.quantile(detector_oof[~train["y"]], 1 - TARGET_FPR), test),
        "rule_threshold": operating_point(rule_test, np.quantile(rule_train[~train["y"]], 1 - TARGET_FPR), test)}
    result["decisions"] = {
        "rule_transfers": bool(rule["ci95"][0] > 0.5 and rule["permutation"]["p_value"] < 0.05),
        "detector_adds_beyond_rule": bool(result["detector_minus_rule"]["ci95"][0] > 0),
        **{f"{k}_transfers": bool(result[f"{k}_threshold"]["window_false_positive_rate"]["rate"] <= MAX_FPR and
                                  result[f"{k}_threshold"]["window_recall"]["rate"] >= MIN_RECALL)
           for k in ("detector", "rule")}}
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {d: load(d) for d in ("svanstrom", "uavirbase")}
    result = {}
    for train, test in (("uavirbase", "svanstrom"), ("svanstrom", "uavirbase")):
        saved = json.loads((DETECTORS / f"external_{train}_to_{test}.json").read_text(encoding="utf-8"))["roc_auc"]
        r = result[f"{train} -> {test}"] = transfer(train, data[train], test, data[test], saved)
        rule, d = r["rule"], r["detector_minus_rule"]
        print(f"{train} -> {test}: rule {rule['feature']} (dir {rule['direction']:+d}, train {rule['train_auc']:.2f}) "
              f"test {rule['roc_auc']:.3f} {[round(v, 2) for v in rule['ci95']]} p={rule['permutation']['p_value']:.4f} | "
              f"detector {r['detector_auc']:.3f} (saved {saved:.3f}), detector − rule {d['difference']:+.3f} "
              f"{[round(v, 2) for v in d['ci95']]}", flush=True)
        for k in ("detector", "rule"):
            op = r[f"{k}_threshold"]
            a = op["alarms"]
            print(f"   {k:8} threshold: window FPR {op['window_false_positive_rate']['rate']:.3f} "
                  f"{[round(v, 2) for v in op['window_false_positive_rate']['ci95']]}, recall {op['window_recall']['rate']:.3f} "
                  f"{[round(v, 2) for v in op['window_recall']['ci95']]} | false alarms {a['false_alarm_onsets']} in "
                  f"{a['non_drone_seconds']} s = {a['false_alarms_per_hour']:.0f}/h (≤ {a['false_alarms_per_hour_upper95']:.0f}/h), "
                  f"drone recordings alarmed {a['drone_recordings_with_alarm']:.2f}", flush=True)
        print("   decisions:", r["decisions"], flush=True)
    result["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (OUT / "acoustic_single_feature_transfer.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
