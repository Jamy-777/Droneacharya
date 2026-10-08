import numpy as np

from droneacharya import evaluation as E
from droneacharya import features as F


def grouped_scores(rng, effect, n_groups=20, per_group=30):
    labels = np.repeat([True, False], n_groups // 2)
    rows = [(f"g{g}", labels[g]) for g in range(n_groups) for _ in range(per_group)]
    y = np.array([r[1] for r in rows])
    groups = np.array([r[0] for r in rows])
    session = {f"g{g}": rng.normal() for g in range(n_groups)}  # group-level noise unrelated to class
    scores = np.array([session[g] for g in groups]) + effect * y + rng.normal(0, 0.2, len(y))
    return scores, y, groups


def test_external_p_value_separates_signal_from_group_noise():
    rng = np.random.default_rng(0)
    real = E.evaluate_external(*grouped_scores(rng, effect=3.0), "gbm", n_permutations=500, n_bootstrap=200)
    noise = E.evaluate_external(*grouped_scores(rng, effect=0.0), "gbm", n_permutations=500, n_bootstrap=200)
    assert real["roc_auc"] > 0.9 and real["permutation"]["p_value"] < 0.01
    assert noise["permutation"]["p_value"] > 0.01


def test_paired_difference_has_the_right_sign_and_interval():
    rng = np.random.default_rng(1)
    good, y, groups = grouped_scores(rng, effect=3.0)
    bad = rng.normal(size=len(y))
    d = E.paired_difference(good, bad, y, groups, n_bootstrap=300)
    assert d["difference"] > 0.3 and d["ci95"][0] > 0
    assert E.paired_difference(bad, good, y, groups, n_bootstrap=300)["ci95"][1] < 0


def test_subset_auc_keeps_only_the_selected_rows():
    rng = np.random.default_rng(2)
    scores, y, groups = grouped_scores(rng, effect=3.0)
    keep = (groups != "g0") & (groups != "g19")
    sub = E.subset_auc(scores, y, groups, keep, n_bootstrap=100)
    assert sub["n"] == int(keep.sum()) and sub["groups_per_label"] == {"False": 9, "True": 9}


def test_paired_label_difference_sign_and_zero():
    rng = np.random.default_rng(4)
    groups = np.repeat([f"u{i}" for i in range(12)], 10)
    y = np.repeat(np.array(["a", "b", "c"] * 4, dtype=object), 10)
    guess = rng.choice(["a", "b", "c"], size=y.size).astype(object)
    same = E.paired_label_difference(y, y, y, groups, n_bootstrap=100)
    assert same["difference"] == 0 and same["ci95"] == [0.0, 0.0]
    better = E.paired_label_difference(y, guess, y, groups, n_bootstrap=100)
    assert better["difference"] > 0.4 and better["ci95"][0] > 0


def test_audio_features_ignore_level():
    rng = np.random.default_rng(3)
    t = np.arange(48000) / 48000
    x = np.sin(2 * np.pi * 180 * t) + 0.3 * rng.normal(size=t.size)
    a, b = F.audio_detector_features(x, 48000), F.audio_detector_features(25 * x, 48000)
    for k in a:
        assert abs(a[k] - b[k]) < 1e-6 * max(1, abs(a[k])), k
    assert a["peakiness_125_250"] > 20  # a 180 Hz tone stands far above its octave's median


def test_mfcc_ignores_level_and_sees_spectral_shape():
    from droneacharya.mfcc import mfcc_features
    rng = np.random.default_rng(6)
    t = np.arange(44100) / 44100
    tone = np.sin(2 * np.pi * 300 * t) + 0.05 * rng.normal(size=t.size)
    a, b = mfcc_features(tone, 44100), mfcc_features(40 * tone, 44100)
    assert len(a) == 26 and all(abs(a[k] - b[k]) < 1e-6 * max(1, abs(a[k])) for k in a)
    hiss = mfcc_features(rng.normal(size=t.size), 44100)
    assert abs(hiss["mfcc_mean_01"] - a["mfcc_mean_01"]) > 1   # a low tone and white noise differ in spectral tilt


def test_detector_model_is_fixed_and_known():
    assert type(E.model("gbm")).__name__ == "HistGradientBoostingClassifier"
