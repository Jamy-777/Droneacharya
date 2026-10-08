"""RF common format v1: every dataset becomes the same "photograph" of the spectrum.

A tile is complex baseband, 50 MS/s, 250 us (12,500 samples), covering 50 MHz around a
centre frequency. For the 2.4 GHz band the centre is UAVSig's own (2.4435 GHz, so
2.4185–2.4685 GHz), which makes UAVSig native and lets every other dataset be cut to
the identical absolute band. Absolute frequency is a label on the tile, never part of
the samples' meaning: models see baseband only, so the same model can be pointed at
another band (5.8 GHz) without retraining.

Band selection is done in the frequency domain: FFT of the input, keep the bins of the
target band, inverse FFT at the output rate. For CardRF's real direct-RF samples
(20 GSa/s) this yields the analytic (I/Q) baseband of the band directly; for complex
I/Q at 100 MS/s it re-centres and decimates by two. Bins the input does not cover are
zero and reported, never invented.
"""
import numpy as np

from . import signal

FC_24 = 2.4435e9            # UAVSig's centre: 2.4185–2.4685 GHz
FS = 50e6
TILE = 12_500                # 250 us at 50 MS/s
DF = FS / TILE               # 4 kHz bin spacing of a tile


def band_tile(x, fs_in, fc_in, fc_out=FC_24, real_input=False):
    """One 250 us tile at fc_out from a block of input that spans exactly 250 us.

    Returns (tile complex64 (12500,), fraction of the target band the input covered)."""
    n = len(x)
    if abs(n / fs_in - TILE / FS) > 1e-9:
        raise ValueError(f"input block is {n / fs_in * 1e6:.1f} us, need 250 us")
    spectrum = np.fft.rfft(x) if real_input else np.fft.fft(x)
    freqs = (np.fft.rfftfreq(n, 1 / fs_in) if real_input else np.fft.fftfreq(n, 1 / fs_in)) + (0 if real_input else fc_in)
    wanted = fc_out + np.fft.fftfreq(TILE, 1 / FS)          # absolute frequency of each output bin
    index = np.round((wanted - freqs[0]) / (fs_in / n)).astype(int) if real_input else None
    out = np.zeros(TILE, dtype=np.complex128)
    if real_input:
        ok = (index >= 0) & (index < len(spectrum))
        out[ok] = 2 * spectrum[index[ok]]                    # analytic signal: positive side, doubled
    else:
        relative = np.round((wanted - fc_in) / (fs_in / n)).astype(int)
        ok = np.abs(wanted - fc_in) < fs_in / 2
        out[ok] = spectrum[relative[ok] % n]
    tile = np.fft.ifft(out) * (TILE / n)
    return tile.astype(np.complex64), float(ok.mean())


# ---------------------------------------------------------------- per dataset
def cardrf_tile(capture_id):
    """CardRF: the whole 250 us capture is exactly one tile."""
    s = signal.read(capture_id)
    return band_tile(s.samples[0].astype(np.float64), s.sample_rate_hz, 0.0, real_input=True)


def uavsig_tiles(capture_id, n_tiles=20):
    """UAVSig: native format; n_tiles evenly spaced 250 us tiles from the 1 s capture."""
    x = signal.read(capture_id).samples[0]
    starts = np.linspace(0, len(x) - TILE, n_tiles).astype(int)
    return np.stack([x[s:s + TILE] for s in starts]), 1.0


def iq100_tiles(capture_id, n_tiles, fc_out=FC_24):
    """RFUAV / RMA / DRFF-R2 (complex I/Q at 100 MS/s): n_tiles evenly spaced, re-centred to fc_out, decimated."""
    first = signal.read(capture_id, count=1)
    block = round(TILE / FS * first.sample_rate_hz)
    total = signal.capture_length(capture_id)
    starts = np.linspace(0, total - block, n_tiles).astype(int)
    tiles, coverage = [], []
    for start in starts:
        x = signal.read(capture_id, start=int(start), count=block).samples[0].astype(np.complex128)
        tile, covered = band_tile(x, first.sample_rate_hz, first.center_frequency_hz, fc_out)
        tiles.append(tile)
        coverage.append(covered)
    return np.stack(tiles), float(np.mean(coverage))


# ---------------------------------------------------------------- v1.1: steered 25 MHz sub-band tiles
# The fixed 50 MHz window misses CardRF's Wi-Fi (channel 1) and Bluetooth (advertising channels at the band
# edges) almost entirely. Real spectrum monitors channelise a wide band and classify each sub-band; v1.1 does
# the same: a 25 MHz tile (complex, 25 MS/s) steered to the strongest activity the receiver can see, by one rule
# for every capture. v1.2 widens it to 40 MHz (40 MS/s).
SUB_FS = 25e6
SUB_TILE = 6_250                      # 250 us at 25 MS/s
SEARCH_24 = (2.390e9, 2.4935e9)       # 2.4 GHz ISM (2.400–2.4835) plus margin
DC_NOTCH_HZ = 100e3                   # receivers' LO leakage, removed before steering


def _boxcar_same(values, width):
    """np.convolve(values, np.ones(width), mode="same") in O(n) with a running sum."""
    padded = np.concatenate([np.zeros(width // 2), values, np.zeros(width - width // 2)])
    cumulative = np.concatenate([[0.0], np.cumsum(padded)])
    return (cumulative[width:] - cumulative[:-width])[: len(values)]


def steered_tile(x, fs_in, fc_in, real_input=False, search=SEARCH_24, width=SUB_FS):
    """The `width` Hz of the visible band with the most energy above the noise floor, as complex baseband sampled
    at `width` (v1.1: 20 MHz at 25 MS/s; v1.2: 40 MHz at 40 MS/s).

    Returns (tile complex64 (250 us of samples), chosen centre Hz). x must span 250 us."""
    n = len(x)
    out_n = int(round(width * TILE / FS))                 # samples in 250 us at the output rate
    if abs(n / fs_in - TILE / FS) > 1e-9:
        raise ValueError(f"input block is {n / fs_in * 1e6:.1f} us, need 250 us")
    if real_input:
        spectrum, freqs = 2 * np.fft.rfft(x), np.fft.rfftfreq(n, 1 / fs_in)
    else:
        spectrum = np.fft.fft(x)
        offsets = np.fft.fftfreq(n, 1 / fs_in)
        spectrum[np.abs(offsets) < DC_NOTCH_HZ] = 0
        freqs = offsets + fc_in
    order = np.argsort(freqs)
    spectrum, freqs = spectrum[order], freqs[order]
    visible = (freqs >= search[0]) & (freqs < search[1])
    power = np.abs(spectrum) ** 2 * visible
    excess = np.clip(power - np.median(power[visible]), 0, None)
    df = fs_in / n
    half = int(round(width / 2 / df))
    window = _boxcar_same(excess, 2 * half)
    lo_ok = freqs >= max(search[0], freqs[visible].min()) + width / 2
    hi_ok = freqs <= min(search[1], freqs[visible].max()) - width / 2
    candidates = np.flatnonzero(lo_ok & hi_ok)
    centre = freqs[candidates[np.argmax(window[candidates])]]
    wanted = centre + np.fft.fftfreq(out_n, 1 / width)
    index = np.clip(np.round((wanted - freqs[0]) / df).astype(int), 0, n - 1)
    tile = np.fft.ifft(spectrum[index]) * (out_n / n)
    return tile.astype(np.complex64), float(centre)


def cardrf_steered(capture_id, width=SUB_FS):
    s = signal.read(capture_id)
    return steered_tile(s.samples[0].astype(np.float64), s.sample_rate_hz, 0.0, real_input=True, width=width)


def uavsig_steered(capture_id, n_tiles=20, width=SUB_FS):
    s = signal.read(capture_id)
    x = s.samples[0]
    block = TILE                                          # 250 us at 50 MS/s
    starts = np.linspace(0, len(x) - block, n_tiles).astype(int)
    out = [steered_tile(x[a:a + block].astype(np.complex128), s.sample_rate_hz, s.center_frequency_hz, width=width)
           for a in starts]
    return np.stack([t for t, _ in out]), [c for _, c in out]


# ---------------------------------------------------------------- stage 2: 14 MHz tiles (Noisy RF's full width)
STAGE2_WIDTH = 14e6                   # complex, 14 MS/s, 250 us = 3,500 samples


def notch_dc(tile, fs):
    """Zero the receiver's LO leakage (|f| < 100 kHz) of a complex baseband tile."""
    spectrum = np.fft.fft(tile.astype(np.complex128))
    spectrum[np.abs(np.fft.fftfreq(len(tile), 1 / fs)) < DC_NOTCH_HZ] = 0
    return np.fft.ifft(spectrum).astype(np.complex64)


def noisy_rf_tile(capture_id):
    """Noisy RF: already 14 MHz complex baseband at 14 MS/s; the first 250 us of the vector, DC notched."""
    s = signal.read(capture_id, count=int(round(STAGE2_WIDTH * TILE / FS)))
    return notch_dc(s.samples[0], s.sample_rate_hz)
