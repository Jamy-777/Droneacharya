import pytest

from droneacharya.index import read, validate
from droneacharya.index.dronerf import CSV_NAME
from droneacharya.index.rar import parse_unrar_lt
from droneacharya.paths import INDEX

LISTING = """
Archive: RF Data_10000_L.rar
Details: RAR 4

        Name: RF Data_10000_L\\10000L_0.csv
        Type: File
        Size: 96868077
       CRC32: 34E6FCD7
    Modified: 2019-01-03 04:32:00,000000000

        Name: RF Data_10000_L
        Type: Directory
"""


def test_parse_unrar_lt_keeps_files_only():
    entries = parse_unrar_lt(LISTING)
    assert entries == [{"name": "RF Data_10000_L\\10000L_0.csv", "type": "File", "size": "96868077",
                        "crc32": "34E6FCD7", "modified": "2019-01-03 04:32:00,000000000"}]


def test_csv_name_grammar():
    assert CSV_NAME.match("11000H_20.csv").groupdict() == {"bui": "11000", "band": "H", "segment": "20"}
    assert CSV_NAME.match("10000X_1.csv") is None


built = pytest.mark.skipif(not (INDEX / "dronerf" / "artifacts.parquet").exists(), reason="index not built")


@pytest.fixture(scope="module")
def tables():
    return {name: read("dronerf", name).to_pylist()
            for name in ("artifacts", "captures", "groups", "lineage", "evidence")}


@built
def test_index_is_internally_consistent(tables):
    validate("dronerf", tables)


@built
def test_counts_match_the_release(tables):
    assert len(tables["artifacts"]) == 454
    assert len(tables["captures"]) == 227
    assert len(tables["groups"]) == 10


@built
def test_every_capture_is_one_l_and_one_h_csv_of_the_same_segment(tables):
    by_capture = {}
    for a in tables["artifacts"]:
        by_capture.setdefault(a["capture_id"], []).append(a)
    for capture_id, parts in by_capture.items():
        assert sorted(p["channel"] for p in parts) == ["H", "L"]
        names = {CSV_NAME.match(p["member_chain"][-1].split("\\")[-1]).group("bui", "segment") for p in parts}
        assert len(names) == 1, capture_id


@built
def test_captures_of_a_bui_share_one_group_and_labels_follow_the_bui(tables):
    for c in tables["captures"]:
        bui = c["capture_id"].split("/")[1]
        assert c["group_id"] == f"dronerf/{bui}"
        assert c["label_uas_present"] == (bui[0] == "1")
        assert (c["label_class"] is None) == (bui == "00000")


@built
def test_ids_are_official_paths_not_disk_paths(tables):
    for a in tables["artifacts"]:
        assert a["artifact_id"].startswith("dronerf/DroneRF/")
        assert ":" not in a["artifact_id"] and "\\" not in a["artifact_id"]
        assert a["storage_relpath"] == "dronerf/original/f4c2b4n755-1.zip"


@built
def test_loose_rars_are_exact_copies(tables):
    assert {r["relation"] for r in tables["lineage"]} == {"exact_copy_of"}


def _minimal():
    artifact = {"artifact_id": "x/a", "dataset_id": "x", "storage_relpath": "x/original/a.zip", "capture_id": "x/c"}
    capture = {"capture_id": "x/c", "dataset_id": "x", "group_id": "x/g", "n_artifacts": 1, "label_status": "VERIFIED"}
    group = {"group_id": "x/g", "dataset_id": "x", "n_captures": 1, "status": "ASSESSMENT"}
    return {"artifacts": [artifact], "captures": [capture], "groups": [group], "lineage": [], "evidence": []}


def test_validate_accepts_a_minimal_index():
    validate("x", _minimal())


@pytest.mark.parametrize("breakage", [
    lambda t: t["artifacts"].append(dict(t["artifacts"][0])),                 # duplicate id
    lambda t: t["artifacts"][0].update(capture_id="x/missing"),              # dangling capture
    lambda t: t["artifacts"][0].update(storage_relpath=r"D:\data\a.zip"),    # drive path
    lambda t: t["captures"][0].update(n_artifacts=2),                        # wrong count
    lambda t: t["groups"][0].update(status="PROBABLY"),                      # unknown status
    lambda t: t["captures"][0].update(capture_id="y/c"),                     # wrong dataset prefix
])
def test_validate_rejects_broken_indexes(breakage):
    tables = _minimal()
    breakage(tables)
    with pytest.raises(ValueError):
        validate("x", tables)
