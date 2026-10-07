import pytest

from droneacharya.index import read, validate
from droneacharya.index.uavsig import _labels
from droneacharya.paths import INDEX


def test_drone_names_decode_unit_and_channel():
    scenario, capture, labels = _labels("drone_1200_03.mat")
    assert (scenario, capture) == ("1200", "03")
    assert labels["label_class"] == "drone_1+drone_2"
    assert labels["label_condition"] == "drone_1:ch1, drone_2:ch2"
    assert labels["label_emitter"] == "aircraft"


def test_controller_baseline_is_not_a_uas_capture():
    _, _, labels = _labels("controller_p2_0000_05.mat")
    assert labels["label_uas_present"] is False
    assert labels["label_emitter"] == "none"
    assert labels["label_condition"] == "arrangement p2"
    assert _labels("controller_p1_1010_00.mat")[2]["label_class"] == "controller_1+controller_3"
    assert _labels("readme.mat") is None


built = pytest.mark.skipif(not (INDEX / "uavsig" / "artifacts.parquet").exists(), reason="index not built")


@pytest.fixture(scope="module")
def tables():
    return {name: read("uavsig", name).to_pylist() for name in ("artifacts", "captures", "groups", "lineage", "evidence")}


@built
def test_index_is_internally_consistent(tables):
    validate("uavsig", tables)


@built
def test_local_subset_counts(tables):
    assert len(tables["captures"]) == 360
    assert len(tables["groups"]) == 120
    for g in tables["groups"]:  # the local two-drone subset holds one capture from each of the 72 scenarios
        assert g["n_captures"] == (1 if "/two_drone/" in g["group_id"] else 6), g["group_id"]
    assert {r["relation"] for r in tables["lineage"]} == {"local_copy_of"}


@built
def test_labels_agree_with_the_stored_transmission_counts(tables):
    n_tx = {r["entity_id"]: int(r["value"]) for r in tables["evidence"] if r["field"] == "labelled_transmissions"}
    for c in tables["captures"]:
        assert (n_tx[c["capture_id"]] > 0) == c["label_uas_present"], c["capture_id"]
        assert c["duration_s"] == 1.0
