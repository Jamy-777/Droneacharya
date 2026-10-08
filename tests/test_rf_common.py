import numpy as np

from droneacharya import rf_common as R
from droneacharya.rf_features import rf_tile_features


def peak_offset_hz(tile, fs):
    spectrum = np.abs(np.fft.fft(tile))
    return np.fft.fftfreq(len(tile), 1 / fs)[np.argmax(spectrum)]


def test_real_direct_rf_tone_lands_at_its_baseband_offset():
    fs = 20e9
    t = np.arange(5_000_000) / fs
    x = np.cos(2 * np.pi * (R.FC_24 + 7e6) * t)
    tile, covered = R.band_tile(x, fs, 0.0, real_input=True)
    assert covered == 1.0
    assert abs(peak_offset_hz(tile, R.FS) - 7e6) < R.DF


def test_complex_iq_is_recentred_and_uncovered_bins_stay_zero():
    fs, fc = 100e6, 2.47e9                      # covers 2.42–2.52 GHz: the tile's lowest 1.5 MHz is missing
    t = np.arange(25_000) / fs
    x = np.exp(2j * np.pi * (2.4435e9 + 3e6 - fc) * t)
    tile, covered = R.band_tile(x, fs, fc)
    assert abs(peak_offset_hz(tile, R.FS) - 3e6) < R.DF
    assert 0.9 < covered < 1.0


def test_steering_finds_a_lone_transmitter_and_ignores_dc():
    fs, fc = 50e6, 2.4435e9
    t = np.arange(12_500) / fs
    rng = np.random.default_rng(0)
    x = 0.01 * (rng.normal(size=t.size) + 1j * rng.normal(size=t.size))
    x += 5.0                                                     # LO leakage at DC
    x += np.exp(2j * np.pi * 15e6 * t)                           # transmitter at fc + 15 MHz
    tile, centre = R.steered_tile(x, fs, fc)
    assert abs(centre + peak_offset_hz(tile, R.SUB_FS) - (fc + 15e6)) < 0.5e6
    assert len(tile) == R.SUB_TILE


def test_tile_features_ignore_power_and_frequency_position():
    rng = np.random.default_rng(1)
    t = np.arange(R.SUB_TILE) / R.SUB_FS
    x = np.exp(2j * np.pi * 2e6 * t) + 0.05 * (rng.normal(size=t.size) + 1j * rng.normal(size=t.size))
    shifted = x * np.exp(2j * np.pi * 5e6 * t)                    # same signal, moved 5 MHz
    a, b, c = rf_tile_features(x), rf_tile_features(30 * x), rf_tile_features(shifted)
    for k in a:
        assert abs(a[k] - b[k]) < 1e-6 * max(1, abs(a[k])), k
    assert abs(a["occupied_bw_mhz"] - c["occupied_bw_mhz"]) < 0.5
