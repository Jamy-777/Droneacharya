import pytest

from droneacharya.index import read, validate
from droneacharya.paths import INDEX

built = pytest.mark.skipif(not (INDEX / "drff_r2" / "artifacts.parquet").exists(), reason="index not built")


@pytest.fixture(scope="module")
def tables():
    return {name: read("drff_r2", name).to_pylist() for name in ("artifacts", "captures", "groups", "lineage", "evidence")}


@built
def test_index_is_internally_consistent(tables):
    validate("drff_r2", tables)


@built
def test_copies_are_lineage_not_artifacts(tables):
    assert len(tables["artifacts"]) == 724
    assert len(tables["lineage"]) == 7
    assert {r["relation"] for r in tables["lineage"]} == {"exact_copy_of"}
    assert not any("dataset5-single_drone_inside_absorbent_cotton/dataset7-environment" in a["artifact_id"]
                   for a in tables["artifacts"])


@built
def test_same_file_name_in_different_subsets_gets_different_ids(tables):
    ids = [c["capture_id"] for c in tables["captures"] if c["capture_id"].endswith("/mavic3C_1_hover_c1_u2_d2")]
    assert len(ids) == len(set(ids)) >= 3


@built
def test_only_environment_files_are_negatives_and_the_odd_one_is_flagged(tables):
    negatives = [c for c in tables["captures"] if not c["label_uas_present"]]
    assert sorted(c["capture_id"].rsplit("/", 1)[-1] for c in negatives) == ["indoor_environment", "outdoor_environment"]
    flagged = {r["entity_id"] for r in tables["evidence"] if r["field"] == "environment_file_metadata"}
    assert flagged == {"drff_r2/dataset7-environment/indoor_environment"}


@built
def test_unit_day_groups_hold_one_unit_and_day(tables):
    for c in tables["captures"]:
        if c["group_id"].count("/") == 2 and c["group_id"].split("/")[1] not in ("environment", "drone_mixed", "wifi_mixed"):
            _, unit, day = c["group_id"].split("/")
            assert c["label_class"] == unit and c["capture_id"].endswith(f"_{day}"), c["capture_id"]
