"""Evaluation shared by baselines and detectors.

These datasets have few independent groups, so every score follows the same rules:

- one metric over pooled out-of-fold predictions (per-fold AUC averaged over
  folds of 1–3 groups is biased upward);
- the model kind is fixed in advance, never chosen on test data;
- significance by permuting labels between groups, and 95% intervals by
  resampling whole groups;
- "beats the baseline" is a paired comparison on identical rows: the interval
  is for the difference, resampled by group.
"""
import warnings

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def model(kind):
    """tree: depth-3 intervals for 1–3 scalar confounds; linear: many features, few rows;
    gbm: the detectors' classical model (shallow boosted trees, class-balanced)."""
    if kind == "tree":
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, class_weight="balanced", random_state=0)
    if kind == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
                                              class_weight="balanced", random_state=0)
    raise ValueError(kind)


def positive_scores(fitted, X):
    return fitted.predict_proba(X)[:, list(fitted.classes_).index(True)]


def out_of_fold(X, y, partitions, kind):
    """Scores for every row from a model fitted without its partition: P(class 1) if binary, else labels."""
    binary = len(set(y)) == 2
    scores = np.empty(len(y), dtype=float if binary else object)
    for held_out in np.unique(partitions):
        test = partitions == held_out
        fitted = model(kind).fit(X[~test], y[~test])
        scores[test] = positive_scores(fitted, X[test]) if binary else fitted.predict(X[test])
    return scores


def metric(y, scores):
    """ROC AUC for binary scores, balanced accuracy for predicted labels."""
    if scores.dtype == object:
        with warnings.catch_warnings():  # a bootstrap resample may lack a class that was predicted
            warnings.simplefilter("ignore", UserWarning)
            return float(balanced_accuracy_score(y, scores.astype(y.dtype)))
    return float(roc_auc_score(y, scores))


def _groups(groups, y):
    group_ids, group_index = np.unique(groups, return_inverse=True)
    group_label = {}
    for g, label in zip(group_index, y):
        if group_label.setdefault(g, label) != label:
            raise ValueError("permutation needs label-pure groups")
    labels = np.array([group_label[g] for g in range(len(group_ids))])
    members = [np.flatnonzero(group_index == g) for g in range(len(group_ids))]
    return group_index, labels, members


def _bootstrap(y, scores, members, rng, n):
    values = []
    for _ in range(n):
        rows = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        if len(set(y[rows])) > 1:
            values.append(metric(y[rows], scores[rows]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _summary(kind, y, labels, observed, ci, null):
    name = "roc_auc" if len(set(y)) == 2 else "balanced_accuracy"
    return {
        "model": kind, "metric": name, "n": int(len(y)),
        "groups_per_label": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        name: observed, "ci95": ci,
        "permutation": {"n": int(len(null)), "p_value": float((1 + np.sum(null >= observed)) / (1 + len(null))),
                        "null_median": float(np.median(null)), "null_95th": float(np.percentile(null, 95))},
    }


def _null_metric(X, y_perm, partitions, kind):
    """One permutation's refit, single-threaded: many small fits run faster side by side than threaded."""
    with threadpool_limits(1):
        return metric(y_perm, out_of_fold(X, y_perm, partitions, kind))


def evaluate(X, y, partitions, groups, kind, n_permutations=1000, n_bootstrap=2000, seed=0, return_scores=False,
             n_jobs=-1):
    """Cross-validated: pooled out-of-fold metric, group-permutation p (refits), group-bootstrap 95% interval."""
    X, y, partitions = np.asarray(X, float), np.asarray(y), np.asarray(partitions)
    group_index, labels, members = _groups(np.asarray(groups), y)
    scores = out_of_fold(X, y, partitions, kind)
    observed = metric(y, scores)
    rng = np.random.default_rng(seed)
    permuted = [rng.permutation(labels)[group_index] for _ in range(n_permutations)]
    permuted = [y_perm for y_perm in permuted
                if all(len(set(y_perm[partitions != p])) > 1 for p in np.unique(partitions))]
    null = Parallel(n_jobs=n_jobs)(delayed(_null_metric)(X, y_perm, partitions, kind) for y_perm in permuted)
    result = _summary(kind, y, labels, observed, _bootstrap(y, scores, members, rng, n_bootstrap), np.asarray(null))
    return (result, scores) if return_scores else result


def evaluate_external(scores, y, groups, kind, n_permutations=2000, n_bootstrap=2000, seed=0):
    """A fixed model's scores on an external test set: AUC, permutation of test labels between groups, bootstrap."""
    scores, y = np.asarray(scores, float), np.asarray(y)
    group_index, labels, members = _groups(np.asarray(groups), y)
    observed = metric(y, scores)
    rng = np.random.default_rng(seed)
    null = np.asarray([metric(rng.permutation(labels)[group_index], scores) for _ in range(n_permutations)])
    return _summary(kind, y, labels, observed, _bootstrap(y, scores, members, rng, n_bootstrap), null)


def paired_difference(scores_a, scores_b, y, groups, n_bootstrap=2000, seed=0):
    """AUC(a) − AUC(b) on identical rows, with a group-bootstrap 95% interval and the share of resamples ≤ 0."""
    scores_a, scores_b, y = np.asarray(scores_a, float), np.asarray(scores_b, float), np.asarray(y)
    _, _, members = _groups(np.asarray(groups), y)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_bootstrap):
        rows = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        if len(set(y[rows])) > 1:
            diffs.append(roc_auc_score(y[rows], scores_a[rows]) - roc_auc_score(y[rows], scores_b[rows]))
    return {"difference": float(roc_auc_score(y, scores_a) - roc_auc_score(y, scores_b)),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "share_of_resamples_at_or_below_zero": float(np.mean(np.asarray(diffs) <= 0))}


def subset_auc(scores, y, groups, keep, n_bootstrap=2000, seed=0):
    """AUC on a subset of rows (e.g. one emitter type vs all negatives), with a group-bootstrap interval."""
    keep = np.asarray(keep, bool)
    scores, y, groups = np.asarray(scores, float)[keep], np.asarray(y)[keep], np.asarray(groups)[keep]
    _, labels, members = _groups(groups, y)
    return {"roc_auc": metric(y, scores), "n": int(keep.sum()),
            "groups_per_label": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
            "ci95": _bootstrap(y, scores, members, np.random.default_rng(seed), n_bootstrap)}
