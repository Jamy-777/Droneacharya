import numpy as np

from droneacharya import baselines as B


def test_an_informative_feature_scores_high_and_noise_scores_near_chance():
    rng = np.random.default_rng(0)
    y = np.repeat([True, False], 200)
    partitions = np.tile([f"fold{i}" for i in range(4)], 100)
    informative = B.evaluate_folds((y + rng.normal(0, 0.3, y.size))[:, None], y, partitions)
    noise = B.evaluate_folds(rng.normal(size=(y.size, 1)), y, partitions)
    assert informative["summary"]["roc_auc"]["mean"] > 0.95
    assert 0.35 < noise["summary"]["roc_auc"]["mean"] < 0.65


def test_folds_missing_a_class_are_skipped_not_scored():
    y = [True, True, False, False]
    result = B.evaluate_folds([[1], [2], [3], [4]], y, ["a", "a", "b", "b"])
    assert all("skipped" in f for f in result["folds"]) and result["summary"] == {}


def test_spectral_signature_ignores_level_when_normalised():
    rng = np.random.default_rng(1)
    x = rng.normal(size=48000)
    a = B.spectral_signature(x, 48000, normalise=True)
    b = B.spectral_signature(10 * x, 48000, normalise=True)
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert a.shape == (32,)


def test_level_dbfs_of_a_full_scale_square_wave_is_zero():
    assert abs(B.level_dbfs(np.array([1.0, -1.0] * 100))) < 1e-9
