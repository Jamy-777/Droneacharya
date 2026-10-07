"""Shortcut baselines: deliberately weak models given one confound each.

A detector is only credible if it beats these on the same split. Each baseline
is evaluated on a saved split: the model is fitted on the other partitions and
predicts the held-out one, so whole groups (devices, recordings, packs) are
unseen at test time. Scoring follows three rules, because these datasets have
few independent groups:

- one metric over the pooled out-of-fold predictions (averaging per-fold AUC
  over folds of 1–3 groups is biased upward);
- the model kind is fixed per baseline in advance, never chosen on test folds;
- significance by permuting labels between groups (the folds stay fixed), and
  a 95% interval by resampling whole groups.
"""
import warnings
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


def model(kind):
    """tree: depth-3 intervals, for 1–3 scalar confounds (catches 'in between' levels); linear: many features."""
    if kind == "tree":
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, class_weight="balanced", random_state=0)
    if kind == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    raise ValueError(kind)


def out_of_fold(X, y, partitions, kind):
    """Scores for every row from a model fitted without its partition: P(class 1) if binary, else labels."""
    binary = len(set(y)) == 2
    scores = np.empty(len(y), dtype=float if binary else object)
    for held_out in np.unique(partitions):
        test = partitions == held_out
        fitted = model(kind).fit(X[~test], y[~test])
        scores[test] = fitted.predict_proba(X[test])[:, list(fitted.classes_).index(True)] if binary             else fitted.predict(X[test])
    return scores


def metric(y, scores):
    """ROC AUC for binary scores, balanced accuracy for predicted labels."""
    if scores.dtype == object:
        with warnings.catch_warnings():  # a bootstrap resample may lack a class that was predicted
            warnings.simplefilter("ignore", UserWarning)
            return float(balanced_accuracy_score(y, scores.astype(y.dtype)))
    return float(roc_auc_score(y, scores))


def evaluate(X, y, partitions, groups, kind, n_permutations=1000, n_bootstrap=2000, seed=0):
    """Pooled out-of-fold metric, group-permutation p-value and group-bootstrap 95% interval."""
    X, y = np.asarray(X, float), np.asarray(y)
    partitions, groups = np.asarray(partitions), np.asarray(groups)
    group_ids, group_index = np.unique(groups, return_inverse=True)
    group_label = {}
    for g, label in zip(group_index, y):
        if group_label.setdefault(g, label) != label:
            raise ValueError("permutation needs label-pure groups")
    labels = np.array([group_label[g] for g in range(len(group_ids))])
    binary = len(set(y)) == 2

    scores = out_of_fold(X, y, partitions, kind)
    observed = metric(y, scores)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(labels)[group_index]
        if all(len(set(y_perm[partitions != p])) > 1 for p in np.unique(partitions)):
            null.append(metric(y_perm, out_of_fold(X, y_perm, partitions, kind)))
    null = np.asarray(null)
    boot = []
    members = [np.flatnonzero(group_index == g) for g in range(len(group_ids))]
    for _ in range(n_bootstrap):
        rows = np.concatenate([members[g] for g in rng.integers(0, len(group_ids), len(group_ids))])
        if len(set(y[rows])) > 1:
            boot.append(metric(y[rows], scores[rows]))
    name = "roc_auc" if binary else "balanced_accuracy"
    return {
        "model": kind, "metric": name, "n": int(len(y)),
        "groups_per_label": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        name: observed,
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "permutation": {"n": int(len(null)), "p_value": float((1 + np.sum(null >= observed)) / (1 + len(null))),
                        "null_median": float(np.median(null)), "null_95th": float(np.percentile(null, 95))},
    }


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
