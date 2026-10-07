"""Shortcut baselines: deliberately weak models given one confound each.

A detector is only credible if it beats these on the same split. Each baseline
is evaluated on a saved split: the model is fitted on the other partitions and
scored on the held-out one, so whole groups (devices, recordings, packs) are
always unseen at test time.
"""
from collections import defaultdict

import numpy as np
from scipy.signal import resample_poly, welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import signal

CARDRF_RAILS = (-32736, 30720)  # processed-data rail codes; EMPIRICAL n=135 raw captures
TRIGGER = 2_500_000             # trigger at the capture midpoint (XOrg = -125 us)


def model(kind="linear"):
    """linear: one threshold per feature direction; tree: depth-3 intervals (catches 'in between' shortcuts)."""
    if kind == "tree":
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, class_weight="balanced", random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))


def evaluate(X, y, partitions, binary=True):
    """Both model kinds; the stronger one (by mean balanced accuracy) is the shortcut's bar."""
    results = {kind: evaluate_folds(X, y, partitions, binary, kind) for kind in ("linear", "tree")}
    best = max(results, key=lambda k: results[k]["summary"].get("balanced_accuracy", {}).get("mean", 0))
    return {**results[best], "model": best,
            "other_model": {k: v["summary"] for k, v in results.items() if k != best}}


def evaluate_folds(X, y, partitions, binary=True, kind="linear"):
    """Per held-out partition: fit on the rest, score on it. Returns per-fold metrics and their summary."""
    X, y, partitions = np.asarray(X, float), np.asarray(y), np.asarray(partitions)
    folds = []
    for held_out in sorted(set(partitions)):
        test, train = partitions == held_out, partitions != held_out
        if len(set(y[train])) < 2 or len(set(y[test])) < (2 if binary else 1):
            folds.append({"partition": held_out, "skipped": "a class is missing on one side", "n_test": int(test.sum())})
            continue
        fitted = model(kind).fit(X[train], y[train])
        fold = {"partition": held_out, "n_test": int(test.sum()),
                "balanced_accuracy": float(balanced_accuracy_score(y[test], fitted.predict(X[test])))}
        if binary:
            fold["roc_auc"] = float(roc_auc_score(y[test], fitted.predict_proba(X[test])[:, 1]))
        folds.append(fold)
    scored = [f for f in folds if "skipped" not in f]
    summary = {m: {"mean": float(np.mean([f[m] for f in scored])), "min": float(np.min([f[m] for f in scored])),
                   "max": float(np.max([f[m] for f in scored]))}
               for m in ("balanced_accuracy", "roc_auc") if scored and m in scored[0]}
    return {"folds": folds, "summary": summary, "n": int(len(y))}


# ---------------------------------------------------------------- features
def cardrf_features(capture_id):
    """Clipping fraction over the whole capture; RMS before and after the trigger (ADC codes)."""
    x = signal.read(capture_id).samples[0]
    clipped = np.mean((x <= CARDRF_RAILS[0]) | (x >= CARDRF_RAILS[1]))
    return {"clip_fraction": float(clipped),
            "log_rms_pre_trigger": float(np.log10(np.sqrt(np.mean(x[:TRIGGER] ** 2)) + 1e-9)),
            "log_rms_post_trigger": float(np.log10(np.sqrt(np.mean(x[TRIGGER:] ** 2)) + 1e-9))}


def audio_windows(capture_id, window_s=1.0, max_windows=10, max_samples=None):
    """Up to max_windows consecutive non-silent windows (channels averaged) and the sample rate."""
    s = signal.read(capture_id, count=max_samples)
    x = s.samples.mean(axis=0)
    n = int(window_s * s.sample_rate_hz)
    windows = [x[i:i + n] for i in range(0, len(x) - n + 1, n)]
    windows = [w for w in windows if np.sqrt(np.mean(w ** 2)) > 1e-6][:max_windows]
    return windows, s.sample_rate_hz


def level_dbfs(window):
    return float(20 * np.log10(np.sqrt(np.mean(window ** 2)) + 1e-12))


def spectral_signature(window, sample_rate, target=16000, bands=32, normalise=False):
    """Log energy in equal-width bands from 0 to 8 kHz after resampling to 16 kHz; optional unit-RMS first."""
    x = resample_poly(window, target, int(sample_rate))
    if normalise:
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-12)
    f, p = welch(x, fs=target, nperseg=1024)
    edges = np.linspace(0, target / 2, bands + 1)
    energy = [p[(f >= lo) & (f < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])]
    return np.log10(np.asarray(energy) + 1e-20)


def group_partitions(groups, k=5):
    """Deterministic group-to-partition assignment for cross-dataset experiments."""
    order = sorted(set(groups))
    index = {g: i % k for i, g in enumerate(order)}
    return [f"fold{index[g]}" for g in groups]


def per_dataset_counts(labels):
    counts = defaultdict(int)
    for label in labels:
        counts[label] += 1
    return dict(counts)
