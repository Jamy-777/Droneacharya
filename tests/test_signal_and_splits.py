import numpy as np
import pytest

from droneacharya import signal, splits
from droneacharya.index import read
from droneacharya.paths import INDEX, RAW

needs_data = pytest.mark.skipif(not RAW.exists() or not (INDEX / "cardrf").exists(), reason="data drive or index missing")


def first_capture(dataset, **match):
    for c in read(dataset, "captures").to_pylist():
        if all(c[k] == v for k, v in match.items()):
            return c
    raise LookupError(dataset)


# ---------------------------------------------------------------- splits
def test_group_kfold_keeps_groups_whole_and_balances_labels():
    captures = [{"capture_id": f"c{g}_{i}", "group_id": f"g{g}", "label_uas_present": g < 6} for g in range(10) for i in range(3)]
    assignment = splits.group_kfold(captures, k=2, seed=1)
    assert splits.leakage_problems(captures, assignment) == []
    per_fold = {f: sum(1 for c in captures if assignment[c["capture_id"]] == f and c["label_uas_present"]) for f in ("fold0", "fold1")}
    assert per_fold == {"fold0": 9, "fold1": 9}


def test_leakage_is_detected():
    captures = [{"capture_id": "a", "group_id": "g"}, {"capture_id": "b", "group_id": "g"}]
    assert splits.leakage_problems(captures, {"a": "train", "b": "test"})


@needs_data
def test_saved_splits_match_their_registry_and_never_leak():
    registry = __import__("json").loads(splits.REGISTRY.read_text(encoding="utf-8"))
    for split_id, entry in registry.items():
        assignment = splits.load(split_id)
        if entry["kind"] == "experiment":
            captures = [c for d in entry["datasets"] for c in splits.captures_of(d)]
            assert splits.leakage_problems(captures, assignment) == [], split_id


# ---------------------------------------------------------------- signal reader
@needs_data
@pytest.mark.parametrize("dataset,dtype,channels", [
    ("dronerf", np.float32, 2), ("cardrf", np.float32, 1), ("rma", np.complex64, 1), ("drff_r2", np.complex64, 1), ("rfuav", np.complex64, 1),
    ("noisy_rf", np.complex64, 1), ("esc50", np.float32, 1), ("svanstrom", np.float32, 2), ("uavirbase", np.float32, 8),
    ("miesikowska_uav", np.float32, 1), ("ddl", np.float32, 8),
])
def test_every_dataset_reads_a_window(dataset, dtype, channels):
    capture = first_capture(dataset)
    s = signal.read(capture["capture_id"], start=100, count=2048)
    assert s.samples.shape == (channels, 2048)
    assert s.samples.dtype == dtype and np.isfinite(s.samples).all()


@needs_data
@pytest.mark.parametrize("dataset,boundary", [("rfuav", 100_000_000), ("ddl", 9600)])
def test_windows_across_stored_parts_equal_their_halves(dataset, boundary):
    capture = next(c for c in read(dataset, "captures").to_pylist() if c["n_artifacts"] >= 2)
    whole = signal.read(capture["capture_id"], start=boundary - 500, count=1000).samples
    left = signal.read(capture["capture_id"], start=boundary - 500, count=500).samples
    right = signal.read(capture["capture_id"], start=boundary, count=500).samples
    np.testing.assert_array_equal(whole, np.concatenate([left, right], axis=1))


@needs_data
def test_windows_are_clipped_at_the_end_and_rejected_outside():
    capture = first_capture("drff_r2")
    s = signal.read(capture["capture_id"], start=140_000_000 - 10, count=100)
    assert s.samples.shape == (1, 10)
    with pytest.raises(ValueError):
        signal.read(capture["capture_id"], start=140_000_000)


@needs_data
def test_dronenoise_event_stacks_its_microphones():
    capture = next(c for c in read("dronenoise", "captures").to_pylist() if c["n_artifacts"] == 9)
    s = signal.read(capture["capture_id"], count=1000)
    assert s.samples.shape == (9, 1000) and s.channels == [f"M{i}" for i in range(1, 10)]
