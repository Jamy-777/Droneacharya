"""Detector features. Level and amplitude are normalised away by construction, so the
shortcuts measured by the baselines (energy, clipping fraction, loudness) cannot enter
directly; what is left is spectral shape, tonality and temporal structure.
"""
import numpy as np
from scipy.signal import resample_poly, welch

from . import signal

# ---------------------------------------------------------------- RF: CardRF (direct RF at 20 GSa/s)
CARDRF_FS = 20e9
CARDRF_TRIGGER = 2_500_000                 # capture midpoint
CARDRF_BAND = (2.38e9, 2.50e9)             # 2.4 GHz ISM plus margin
CARDRF_BINS = 24                           # 5 MHz each
CARDRF_FRAME = 10_000                      # 0.5 us envelope frames


def cardrf_detector_features(capture_id):
    """Post-trigger half only, unit RMS: in-band spectral shape, occupied bandwidth, flatness, envelope."""
    x = signal.read(capture_id, start=CARDRF_TRIGGER).samples[0].astype(np.float64)
    x -= x.mean()
    x /= np.sqrt(np.mean(x ** 2)) + 1e-12
    f, p = welch(x, fs=CARDRF_FS, nperseg=65536, noverlap=32768)
    band = (f >= CARDRF_BAND[0]) & (f < CARDRF_BAND[1])
    fb, pb = f[band], p[band]
    edges = np.linspace(*CARDRF_BAND, CARDRF_BINS + 1)
    shape = np.array([pb[(fb >= lo) & (fb < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])
    shape = np.log10(shape / shape.sum() + 1e-12)
    cumulative = np.cumsum(pb) / pb.sum()
    occupied = fb[np.searchsorted(cumulative, 0.995)] - fb[np.searchsorted(cumulative, 0.005)]
    flatness = np.exp(np.mean(np.log(pb + 1e-30))) / (pb.mean() + 1e-30)
    frames = np.sqrt(np.mean(x[: len(x) // CARDRF_FRAME * CARDRF_FRAME].reshape(-1, CARDRF_FRAME) ** 2, axis=1))
    active = frames > 0.5 * frames.max()
    features = {f"band_{i:02d}": float(v) for i, v in enumerate(shape)}
    features.update({
        "occupied_bw_mhz": float(occupied / 1e6),
        "spectral_flatness": float(flatness),
        "envelope_active_fraction": float(active.mean()),
        "envelope_transitions": float(np.sum(active[1:] != active[:-1])),
        "envelope_cv": float(frames.std() / (frames.mean() + 1e-12)),
    })
    return features


# ---------------------------------------------------------------- acoustic (1 s windows)
AUDIO_FS = 16000
OCTAVES = ((62, 125), (125, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000))


def audio_detector_features(window, sample_rate):
    """Resampled to 16 kHz, unit RMS: band shape, tonal peakiness per octave, flatness, centroid, modulation."""
    x = resample_poly(window, AUDIO_FS, int(sample_rate)).astype(np.float64)
    x -= x.mean()
    x /= np.sqrt(np.mean(x ** 2)) + 1e-12
    f, p = welch(x, fs=AUDIO_FS, nperseg=4096)          # 3.9 Hz resolution for rotor harmonics
    edges = np.linspace(0, AUDIO_FS / 2, 33)
    shape = np.array([p[(f >= lo) & (f < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])
    shape = np.log10(shape / shape.sum() + 1e-12)
    features = {f"band_{i:02d}": float(v) for i, v in enumerate(shape)}
    for lo, hi in OCTAVES:
        sel = p[(f >= lo) & (f < hi)]
        features[f"peakiness_{lo}_{hi}"] = float(10 * np.log10(sel.max() / (np.median(sel) + 1e-30) + 1e-30))
    useful = (f >= 100) & (f <= 8000)
    pu, fu = p[useful], f[useful]
    features["spectral_flatness"] = float(np.exp(np.mean(np.log(pu + 1e-30))) / (pu.mean() + 1e-30))
    features["spectral_centroid_hz"] = float((fu * pu).sum() / (pu.sum() + 1e-30))
    frames = np.sqrt(np.mean(x[: len(x) // 512 * 512].reshape(-1, 512) ** 2, axis=1))  # 32 ms frames
    features["modulation_cv"] = float(frames.std() / (frames.mean() + 1e-12))
    return features
