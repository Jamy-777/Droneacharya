"""RF activity and emitters: what is transmitting, and how it behaves over time (docs/architecture.md, layers 2–4).

No learning here. A capture becomes a spectrogram; cells clearly above the receiver's noise floor (CFAR) form
time-frequency blobs; blobs of one shape form an emitter whose behaviour (hop set, burst rhythm, bandwidth, duty
cycle) is what distinguishes a drone link from Wi-Fi or Bluetooth. Those are properties of the transmitter, not of
the receiver, which is why they come before any model.

Noise floor: in deployment it comes from a site calibration (pass `floor`). Without one it is estimated per
frequency bin as the median over time, capped at twice the band's quiet level so that a continuous emitter does not
become its own floor. This assumes every bin is free of signal for over half the time, or the emitter covers under
three quarters of the bins.

Wi-Fi is recognised from its own numerology, not from "OFDM": DJI's video links are OFDM too. An 802.11 burst
starts with the short training field, ten repetitions of 0.8 us, so at 20 MS/s it correlates with itself at a lag
of 16 samples for 8 us. A burst only counts if it is also 20 or 40 MHz wide.
"""
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import ndimage
from scipy.signal import resample_poly
from scipy.signal.windows import blackmanharris

BIN_HZ = 200e3                 # target spectrogram resolution: ~200 kHz bins, ~5 us frames


@dataclass
class Spectrogram:
    power: np.ndarray          # (frames, bins), fftshifted, float32
    fs: float
    nfft: int

    @property
    def frame_s(self):
        return self.nfft / self.fs

    @property
    def bin_hz(self):
        return self.fs / self.nfft

    def bin_freq(self, b):
        """Baseband frequency (Hz) of bin index b."""
        return (np.asarray(b) - self.nfft // 2) * self.bin_hz


def spectrogram(x, fs, bin_hz=BIN_HZ):
    nfft = int(2 ** np.round(np.log2(fs / bin_hz)))
    frames = np.asarray(x)[: len(x) // nfft * nfft].reshape(-1, nfft)
    # Blackman-Harris: sidelobes at -92 dB, so a signal 50 dB above the floor does not leak into neighbouring bins
    spec = np.fft.fftshift(np.fft.fft(frames * blackmanharris(nfft), axis=1), axes=1)
    return Spectrogram((np.abs(spec) ** 2).astype(np.float32), fs, nfft)


def noise_floor(spec, floor=None):
    """Median noise power per bin (calibrated if `floor` is given)."""
    if floor is not None:
        return np.broadcast_to(np.asarray(floor, np.float32), spec.power.shape[1:]).copy()
    per_bin = np.median(spec.power, axis=0)
    quiet = np.percentile(per_bin, 25)
    return np.minimum(per_bin, 2 * quiet)


def cfar_mask(spec, floor, pfa=1e-3, dc_bins=1):
    """Cells above the floor at a per-cell false-alarm probability `pfa` for noise alone.

    A single-frame periodogram of complex Gaussian noise is exponential, so P(cell > t * mean) = exp(-t); the floor
    is a median, which is ln 2 times the mean. Bins within `dc_bins` of DC are never detections (receiver LO spur)."""
    factor = -np.log(pfa) / np.log(2)
    mask = spec.power > factor * floor[None, :]
    centre = spec.nfft // 2
    if dc_bins:
        notch = slice(centre - dc_bins, centre + dc_bins + 1)
        # never a detection on its own, but an emitter detected on both sides of DC stays one emitter
        mask[:, notch] = (mask[:, centre - dc_bins - 1] & mask[:, centre + dc_bins + 1])[:, None]
    return mask


@dataclass
class Blob:
    t0: float                  # s
    t1: float
    f_lo: float                # Hz, baseband
    f_hi: float
    cells: int
    snr_db: float              # peak cell power over floor
    occupied: tuple = None     # (lo, hi) Hz holding 99% of the blob's energy above the floor: independent of SNR

    @property
    def duration(self):
        return self.t1 - self.t0

    @property
    def bandwidth(self):
        lo, hi = self.occupied or (self.f_lo, self.f_hi)
        return hi - lo

    @property
    def centre(self):
        lo, hi = self.occupied or (self.f_lo, self.f_hi)
        return (lo + hi) / 2


GAP_FRAMES = 3                 # a burst that fades below threshold for up to 3 frames (~15 us) stays one blob


def blobs(spec, mask, floor, min_cells=3):
    """Connected detections (8-connectivity, short gaps in time bridged) as time-frequency boxes."""
    bridged = ndimage.binary_closing(mask, structure=np.ones((GAP_FRAMES + 2, 1)), border_value=0) | mask
    labels, n = ndimage.label(bridged, structure=np.ones((3, 3)))
    if n == 0:
        return []
    cells = np.bincount(labels.ravel(), minlength=n + 1)
    snr = ndimage.maximum(spec.power / floor[None, :], labels, index=np.arange(1, n + 1))
    out = []
    for i, box in enumerate(ndimage.find_objects(labels)):
        if box is None or cells[i + 1] < min_cells:
            continue
        t, f = box
        excess = np.where(labels[box] == i + 1, np.clip(spec.power[box] - floor[None, f], 0, None), 0).sum(axis=0)
        cumulative = np.cumsum(excess) / (excess.sum() + 1e-30)
        lo, hi = np.searchsorted(cumulative, 0.005), np.searchsorted(cumulative, 0.995)
        out.append(Blob(t.start * spec.frame_s, t.stop * spec.frame_s,
                        float(spec.bin_freq(f.start) - spec.bin_hz / 2), float(spec.bin_freq(f.stop - 1) + spec.bin_hz / 2),
                        int(cells[i + 1]), float(10 * np.log10(snr[i])),
                        (float(spec.bin_freq(f.start + lo) - spec.bin_hz / 2), float(spec.bin_freq(f.start + hi) + spec.bin_hz / 2))))
    return out


SHAPE_TOLERANCE = 0.5          # octaves: blobs of one emitter share bandwidth and burst length within this


def _shape(b, spec):
    return np.log2(max(b.bandwidth, spec.bin_hz)), np.log2(max(b.duration, spec.frame_s))


def emitters(found, spec, capture_s):
    """Group blobs of one shape (bandwidth and burst length within half an octave of the group's first, largest
    member) and describe each group's behaviour over time."""
    groups = []
    for b in sorted(found, key=lambda b: -b.cells):
        bw, dur = _shape(b, spec)
        for g in groups:
            if abs(bw - g["bw"]) <= SHAPE_TOLERANCE and abs(dur - g["dur"]) <= SHAPE_TOLERANCE:
                g["members"].append(b)
                break
        else:
            groups.append({"bw": bw, "dur": dur, "members": [b]})
    out = []
    for members in (g["members"] for g in groups):
        members.sort(key=lambda b: b.t0)
        starts = np.array([b.t0 for b in members])
        intervals = np.diff(starts)
        bandwidth = float(np.median([b.bandwidth for b in members]))
        centres = np.sort([b.centre for b in members])            # channels: centres more than half a width apart
        out.append({
            "blobs": len(members),
            "bandwidth_hz": bandwidth,
            "burst_s": float(np.median([b.duration for b in members])),
            "hop_set": int(1 + np.sum(np.diff(centres) > 0.5 * bandwidth)),
            "span_hz": float(max(b.f_hi for b in members) - min(b.f_lo for b in members)),
            "interval_s": float(np.median(intervals)) if len(intervals) else float("nan"),
            "interval_cv": float(intervals.std() / intervals.mean()) if len(intervals) > 1 and intervals.mean() > 0 else float("nan"),
            "duty": float(min(1.0, sum(b.duration for b in members) / capture_s)),
            "snr_db": float(np.median([b.snr_db for b in members])),
            "continuous": bool(max(b.duration for b in members) > 0.5 * capture_s),
            "members": members})
    return sorted(out, key=lambda e: -e["blobs"] * e["burst_s"])


def analyse(x, fs, floor=None, pfa=1e-3):
    """Capture → (spectrogram, floor, blobs, emitters)."""
    spec = spectrogram(x, fs)
    fl = noise_floor(spec, floor)
    found = blobs(spec, cfar_mask(spec, fl, pfa), fl)
    return spec, fl, found, emitters(found, spec, len(x) / fs)


# ---------------------------------------------------------------- recognisers for known standards
WIFI_FS = 20e6
STF_LAG = 16                   # 0.8 us at 20 MS/s
STF_MIN_RUN = 96               # 4.8 us of the 8 us short training field
WIFI_WIDTHS = ((14e6, 22e6), (30e6, 44e6))


def stf_correlation(x, fs, centre_hz):
    """Normalised lag-0.8 us autocorrelation of x shifted to `centre_hz` and resampled to 20 MS/s."""
    n = np.arange(len(x))
    y = np.asarray(x) * np.exp(-2j * np.pi * centre_hz * n / fs)
    ratio = Fraction(int(WIFI_FS), int(round(fs))).limit_denominator(1000)
    y = resample_poly(y, ratio.numerator, ratio.denominator)
    prod = y[STF_LAG:] * np.conj(y[:-STF_LAG])
    energy = np.abs(y) ** 2
    window = np.ones(STF_LAG * 4)
    num = np.abs(np.convolve(prod, window, mode="valid"))
    den = np.sqrt(np.convolve(energy[STF_LAG:], window, mode="valid") * np.convolve(energy[:-STF_LAG], window, mode="valid"))
    return num / (den + 1e-30)


def is_wifi(x, fs, blob, threshold=0.75):
    """A blob is 802.11 if it is 20/40 MHz wide and its first 20 us hold a sustained short training field."""
    if not any(lo <= blob.bandwidth <= hi for lo, hi in WIFI_WIDTHS):
        return False
    a = max(0, int((blob.t0 - 5e-6) * fs))
    corr = stf_correlation(np.asarray(x)[a:a + int(25e-6 * fs)], fs, blob.centre)
    run = np.convolve(corr > threshold, np.ones(STF_MIN_RUN - 4 * STF_LAG + 1), mode="valid")
    return bool(len(run) and run.max() >= STF_MIN_RUN - 4 * STF_LAG + 1)
