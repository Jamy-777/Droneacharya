import pytest

from droneacharya.index import read, validate
from droneacharya.index.cardrf import _describe, _parse_date
from droneacharya.paths import INDEX


def test_paths_decode_into_condition_split_category_device_mode():
    assert _describe("CARDRF/LOS/Train/UAV/DJI_INSPIRE/VIDEOING/x.mat") == ("LOS", "train", "UAV", "DJI_INSPIRE", "VIDEOING")
    assert _describe("CARDRF/NLOS/UAV/DJI_M600/FLYING/x.mat") == ("NLOS", None, "UAV", "DJI_M600", "FLYING")
    assert _describe("CARDRF/LOS/Test/WIFI/CISCO_LINKSYS_E3200/x.mat") == ("LOS", "test", "WIFI", "CISCO_LINKSYS_E3200", None)
    with pytest.raises(ValueError):
        _describe("CARDRF/LOS/Train/RADAR/X/x.mat")


def test_scope_dates_parse_with_padded_hours():
    assert _parse_date("25-Aug-2020  1:50:40").isoformat() == "2020-08-25T01:50:40"


built = pytest.mark.skipif(not (INDEX / "cardrf" / "artifacts.parquet").exists(), reason="index not built")


@pytest.fixture(scope="module")
def tables():
    return {name: read("cardrf", name).to_pylist() for name in ("artifacts", "captures", "groups", "lineage", "evidence")}


@built
def test_index_is_internally_consistent(tables):
    validate("cardrf", tables)


@built
def test_counts_match_the_release(tables):
    captures = tables["captures"]
    assert len(captures) == 9600
    assert sum(c["official_split"] == "train" for c in captures) == 6090
    assert sum(c["official_split"] == "test" for c in captures) == 2610
    assert sum(c["label_condition"].startswith("NLOS") for c in captures) == 900


@built
def test_hard_negatives_are_labelled_as_non_uas_emitters(tables):
    by_emitter = {}
    for c in tables["captures"]:
        by_emitter.setdefault(c["label_emitter"], set()).add(c["label_uas_present"])
    assert by_emitter == {"aircraft": {True}, "controller": {True}, "wifi": {False}, "bluetooth": {False}}


@built
def test_train_and_test_of_a_device_share_one_group(tables):
    splits = {}
    for c in tables["captures"]:
        splits.setdefault(c["group_id"], set()).add(c["official_split"])
    for group_id, seen in splits.items():
        if "NLOS" not in group_id:
            assert {"train", "test"} <= seen, group_id


@built
def test_every_capture_has_its_own_timestamp_and_duration(tables):
    for c in tables["captures"]:
        assert c["recorded_at"].startswith("2020-")
        assert c["duration_s"] == pytest.approx(250e-6)
        assert c["sample_rate_hz"] == pytest.approx(20e9)
