"""Activity, emitters and the Wi-Fi recogniser on synthetic signals whose truth is known."""
import numpy as np
from scipy.signal import resample_poly

from droneacharya import rf_emitters as R

FS = 50e6


def noise(rng, n):
    return (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)


def band_noise(rng, n, fs, f_lo, f_hi, power=1.0):
    spectrum = np.fft.fft(noise(rng, n))
    f = np.fft.fftfreq(n, 1 / fs)
    spectrum[(f < f_lo) | (f > f_hi)] = 0
    x = np.fft.ifft(spectrum)
    return x * np.sqrt(power / np.mean(np.abs(x) ** 2))


def test_cfar_false_alarm_rate_matches_pfa_on_noise():
    rng = np.random.default_rng(0)
    spec = R.spectrogram(noise(rng, 2_000_000), FS)
    floor = R.noise_floor(spec)
    rate = R.cfar_mask(spec, floor, pfa=1e-3).mean()
    assert 0.5e-3 < rate < 2e-3


def test_hopper_becomes_one_emitter_with_its_hop_set_and_rhythm():
    rng = np.random.default_rng(1)
    n, period, burst = 1_000_000, 50_000, 20_000              # 20 ms; a 400 us burst every 1 ms
    x = noise(rng, n)
    hops = np.arange(-17.5e6, 18e6, 5e6)                      # 8 channels
    order = rng.permutation(np.tile(np.arange(8), 3))[: n // period]
    for i, h in enumerate(order):
        x[i * period:i * period + burst] += band_noise(rng, burst, FS, hops[h] - 0.5e6, hops[h] + 0.5e6)
    _, _, _, found = R.analyse(x, FS)
    top = found[0]
    assert top["hop_set"] == len(set(order.tolist()))
    assert top["blobs"] == len(order)
    assert 0.95e-3 < top["interval_s"] < 1.05e-3 and top["interval_cv"] < 0.05
    assert 0.6e6 < top["bandwidth_hz"] < 2e6 and 350e-6 < top["burst_s"] < 450e-6


def test_continuous_wide_link_is_one_continuous_emitter():
    rng = np.random.default_rng(2)
    n = 250_000                                               # 5 ms
    x = noise(rng, n) + band_noise(rng, n, FS, -9e6, 9e6, power=0.36 * 31.6)
    _, _, _, found = R.analyse(x, FS)
    assert found[0]["continuous"] and 15e6 < found[0]["bandwidth_hz"] < 22e6


# ---------------------------------------------------------------- Wi-Fi
STF = {-24: 1, -20: -1, -16: 1, -12: -1, -8: -1, -4: 1, 4: -1, 8: -1, 12: 1, 16: 1, 20: 1, 24: 1}


def ofdm_symbols(rng, nfft, used, cp, count):
    out = []
    for _ in range(count):
        bins = np.zeros(nfft, complex)
        bins[used % nfft] = (rng.choice([-1, 1], len(used)) + 1j * rng.choice([-1, 1], len(used))) / np.sqrt(2)
        s = np.fft.ifft(bins)
        out.append(np.r_[s[-cp:], s])
    return np.concatenate(out)


def wifi_burst(rng):
    """802.11a/g at 20 MS/s: short training field (10 x 0.8 us), long training field, 20 data symbols."""
    bins = np.zeros(64, complex)
    for k, v in STF.items():
        bins[k % 64] = np.sqrt(13 / 6) * v * (1 + 1j)
    stf = np.tile(np.fft.ifft(bins), 3)[:160]
    used = np.r_[-26:0, 1:27]
    ltf = ofdm_symbols(rng, 64, used, 32, 1)[:160]
    return np.r_[stf, ltf, ofdm_symbols(rng, 64, used, 16, 20)]


def place(rng, burst_20msps, offset_hz, power, n=150_000, at=50_000):
    """A 20 MS/s burst moved to 50 MS/s at a baseband offset, inside receiver noise."""
    y = resample_poly(burst_20msps, 5, 2)
    y *= np.sqrt(power / np.mean(np.abs(y) ** 2))
    y *= np.exp(2j * np.pi * offset_hz * np.arange(len(y)) / FS)
    x = noise(rng, n)
    x[at:at + len(y)] += y
    return x, R.Blob(at / FS, (at + len(y)) / FS, offset_hz - 9e6, offset_hz + 9e6, 0, 20.0)


def test_wifi_burst_is_found_and_recognised():
    rng = np.random.default_rng(3)
    x, _ = place(rng, wifi_burst(rng), 10e6, power=30)
    _, _, found, _ = R.analyse(x, FS)
    wide = [b for b in found if 14e6 <= b.bandwidth <= 22e6]
    assert wide and any(R.is_wifi(x, FS, b) for b in wide)


def test_other_ofdm_and_noise_bursts_are_not_wifi():
    rng = np.random.default_rng(4)
    lte_like = ofdm_symbols(rng, 1280, np.r_[-550:0, 1:551], 90, 3)          # 15.6 kHz spacing, ~17 MHz
    for burst in (lte_like, band_noise(rng, 4000, 20e6, -8.5e6, 8.5e6)):
        x, blob = place(rng, burst, 10e6, power=30)
        assert not R.is_wifi(x, FS, blob)


def test_narrow_burst_never_counts_as_wifi():
    rng = np.random.default_rng(5)
    x = noise(rng, 100_000)
    blob = R.Blob(0.0, 1e-3, 1e6, 2e6, 0, 20.0)
    assert not R.is_wifi(x, FS, blob)
