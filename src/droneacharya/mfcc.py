"""MFCC features for 1 s audio windows: the standard representation in acoustic drone-detection papers.

Same preparation as features.audio_detector_features (16 kHz, mean removed, unit RMS), so loudness cannot enter.
Frames of 25 ms every 10 ms, Hann window, 512-point power spectrum, 40 HTK-mel triangles from 20 Hz to 8 kHz, log,
DCT-II; coefficients 0–12, summarised by their mean and standard deviation over the window's frames.

No cepstral mean normalisation per recording: it would subtract any sound present for the whole recording,
including a hovering drone.
"""
import numpy as np
from scipy.fft import dct
from scipy.signal import resample_poly

FS = 16000
FRAME, HOP, NFFT = 400, 160, 512
N_MELS, N_COEFFS = 40, 13


def _mel(f):
    return 2595 * np.log10(1 + f / 700)


def _mel_inverse(m):
    return 700 * (10 ** (m / 2595) - 1)


def filterbank(n_mels=N_MELS, nfft=NFFT, fs=FS, fmin=20.0, fmax=FS / 2):
    edges = _mel_inverse(np.linspace(_mel(fmin), _mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(nfft, 1 / fs)
    bank = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        lo, centre, hi = edges[i:i + 3]
        bank[i] = np.clip(np.minimum((bins - lo) / (centre - lo), (hi - bins) / (hi - centre)), 0, None)
    return bank


BANK = filterbank()


def mfcc_features(window, sample_rate):
    x = resample_poly(window, FS, int(sample_rate)).astype(np.float64)
    x -= x.mean()
    x /= np.sqrt(np.mean(x ** 2)) + 1e-12
    starts = np.arange(0, len(x) - FRAME + 1, HOP)
    frames = np.stack([x[s:s + FRAME] for s in starts]) * np.hanning(FRAME)
    power = np.abs(np.fft.rfft(frames, NFFT, axis=1)) ** 2
    coeffs = dct(np.log(power @ BANK.T + 1e-10), type=2, norm="ortho", axis=1)[:, :N_COEFFS]
    out = {f"mfcc_mean_{i:02d}": float(v) for i, v in enumerate(coeffs.mean(axis=0))}
    out.update({f"mfcc_std_{i:02d}": float(v) for i, v in enumerate(coeffs.std(axis=0))})
    return out
