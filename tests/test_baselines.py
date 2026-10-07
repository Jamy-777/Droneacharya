import numpy as np

from droneacharya import baselines as B
from droneacharya import splits


def grouped_data(rng, n_groups=18, n_pos=11, per_group=60, effect=0.0):
    """Each group has its own random level (a session effect unrelated to class) plus an optional class effect."""
    labels = np.array([True] * n_pos + [False] * (n_groups - n_pos))
    rows = [(f"g{g}", labels[g], rng.normal() + effect * labels[g] + rng.normal(0, 0.1))
            for g in range(n_groups) for _ in range(per_group)]
    groups = np.array([r[0] for r in rows])
    captures = [{"capture_id": f"{g}_{i}", "group_id": g, "label_uas_present": bool(y)} for i, (g, y, _) in enumerate(rows)]
    folds = splits.group_kfold(captures, 5, seed=0)
    partitions = np.array([folds[c["capture_id"]] for c in captures])
    return np.array([[r[2]] for r in rows]), np.array([r[1] for r in rows]), partitions, groups


def test_group_level_noise_scores_near_chance_when_pooled():
    rng = np.random.default_rng(0)
    aucs = []
    for _ in range(40):
        X, y, partitions, _ = grouped_data(rng)
        aucs.append(B.metric(y, B.out_of_fold(X, y, partitions, "tree")))
    assert 0.40 < np.median(aucs) < 0.58


def test_permutation_flags_a_real_group_effect_and_not_noise():
    rng = np.random.default_rng(1)
    real = B.evaluate(*grouped_data(rng, effect=3.0), "tree", n_permutations=200, n_bootstrap=200)
    noise = B.evaluate(*grouped_data(rng), "tree", n_permutations=200, n_bootstrap=200)
    assert real["roc_auc"] > 0.9 and real["permutation"]["p_value"] < 0.01
    assert noise["permutation"]["p_value"] > 0.01
    assert real["groups_per_label"] == {"False": 7, "True": 11}


def test_permutation_requires_label_pure_groups():
    try:
        B.evaluate([[0], [1], [2], [3]], [True, False, True, False], ["a", "a", "b", "b"], ["g", "g", "h", "h"], "tree",
                   n_permutations=1, n_bootstrap=1)
    except ValueError as error:
        assert "label-pure" in str(error)
    else:
        raise AssertionError("mixed groups were accepted")


def test_spectral_signature_ignores_level_when_normalised():
    rng = np.random.default_rng(1)
    x = rng.normal(size=48000)
    a = B.spectral_signature(x, 48000, normalise=True)
    np.testing.assert_allclose(a, B.spectral_signature(10 * x, 48000, normalise=True), atol=1e-6)
    assert a.shape == (32,)


def test_level_dbfs_of_a_full_scale_square_wave_is_zero():
    assert abs(B.level_dbfs(np.array([1.0, -1.0] * 100))) < 1e-9


def test_split_ids_never_change_meaning(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "SPLITS", tmp_path)
    monkeypatch.setattr(splits, "REGISTRY", tmp_path / "registry.json")
    monkeypatch.setattr(splits, "RETIRED", tmp_path / "retired.json")
    captures = [{"capture_id": f"x/c{i}", "group_id": f"x/g{i}", "label_uas_present": i % 2 == 0} for i in range(4)]
    first = {c["capture_id"]: f"fold{i % 2}" for i, c in enumerate(captures)}
    splits.save("x/s", "experiment", "test", captures, first)
    splits.save("x/s", "experiment", "test", captures, first)  # identical: no-op
    other = {c["capture_id"]: f"fold{(i + 1) % 2}" for i, c in enumerate(captures)}
    for attempt in (lambda: splits.save("x/s", "experiment", "test", captures, other),):
        try:
            attempt()
        except ValueError:
            pass
        else:
            raise AssertionError("split was overwritten")
    splits.retire("x/s", "test")
    try:
        splits.save("x/s", "experiment", "test", captures, first)
    except ValueError:
        pass
    else:
        raise AssertionError("retired ID was reused")
