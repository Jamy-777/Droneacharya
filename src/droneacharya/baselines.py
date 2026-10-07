"""Shortcut baselines: deliberately weak models given one confound each.

A detector is only credible if it beats these on the same split and rows.
Evaluation rules (pooled out-of-fold metric, model fixed in advance, group
permutation, group bootstrap) live in evaluation.py and are shared with the
detectors.
"""
from collections import defaultdict

import numpy as np
from scipy.signal import resample_poly, welch

from . import signal
from .evaluation import evaluate, metric, model, out_of_fold  # noqa: F401  (re-exported for scripts and tests)

CARDRF_RAILS = (-32736, 30720)  # processed-data rail codes; EMPIRICAL n=135 raw captures
TRIGGER = 2_500_000             # trigger at the capture midpoint (XOrg = -125 us)


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
