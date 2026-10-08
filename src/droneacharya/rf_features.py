"""Position-invariant features of a steered 20 MHz RF tile (complex baseband, 25 MS/s, 250 us).

Nothing here depends on where in frequency a signal sits, and every tile is scaled to unit
RMS first, so absolute frequency, received power and receiver gain cannot enter. What is
left is the "driving style": how wide, how busy, how bursty, how hoppy, what modulation.
"""
import numpy as np

NFFT = 128                 # 195 kHz bins, 5.12 us frames
FLOOR_DB = 6.0             # "active" = this far above the tile's median bin power


def rf_tile_features(tile, sample_rate=25e6):
    x = tile.astype(np.complex128)
    x = x - x.mean()
    x /= np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12
    frames = x[: len(x) // NFFT * NFFT].reshape(-1, NFFT)
    spec = np.abs(np.fft.fftshift(np.fft.fft(frames * np.hanning(NFFT), axis=1), axes=1)) ** 2   # (frames, bins)
    floor = np.median(spec)
    active = spec > floor * 10 ** (FLOOR_DB / 10)
    mean_spec = spec.mean(axis=0)
    cumulative = np.cumsum(mean_spec) / mean_spec.sum()
    bin_hz = sample_rate / NFFT
    occupied = (np.searchsorted(cumulative, 0.995) - np.searchsorted(cumulative, 0.005)) * bin_hz
    shape = np.sort(mean_spec / mean_spec.sum())[::-1]
    quantiles = shape[np.linspace(0, NFFT - 1, 12).astype(int)]
    frame_power = spec.sum(axis=1)
    on = frame_power > np.median(frame_power) * 2
    centroid = (spec * np.arange(NFFT)).sum(axis=1) / (spec.sum(axis=1) + 1e-30)
    per_frame_bw = active.sum(axis=1)
    amplitude = np.abs(x)
    features = {f"shape_q{i:02d}": float(np.log10(q + 1e-12)) for i, q in enumerate(quantiles)}
    features.update({
        "occupied_bw_mhz": float(occupied / 1e6),
        "active_bins_fraction": float(active.any(axis=0).mean()),
        "spectral_flatness": float(np.exp(np.mean(np.log(mean_spec + 1e-30))) / (mean_spec.mean() + 1e-30)),
        "duty_cycle": float(on.mean()),
        "transitions": float(np.sum(on[1:] != on[:-1])),
        "frame_power_cv": float(frame_power.std() / (frame_power.mean() + 1e-30)),
        "centroid_jump_mean": float(np.mean(np.abs(np.diff(centroid))) / NFFT),
        "centroid_spread": float(np.std(centroid) / NFFT),
        "frame_bw_mean": float(per_frame_bw.mean() / NFFT),
        "frame_bw_std": float(per_frame_bw.std() / NFFT),
        "amplitude_kurtosis": float(np.mean(amplitude ** 4) / (np.mean(amplitude ** 2) ** 2 + 1e-30)),
        "papr_db": float(10 * np.log10(amplitude.max() ** 2 / (np.mean(amplitude ** 2) + 1e-30))),
    })
    return features
