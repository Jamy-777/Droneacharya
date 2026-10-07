import re

import pytest

from droneacharya.index import read, validate
from droneacharya.index.rfuav import IQ_NAME, _runs
from droneacharya.paths import INDEX


def test_runs_split_at_missing_chunks():
    assert _runs([0, 1, 2, 4, 5]) == [[0, 1, 2], [4, 5]]
    assert _runs([3]) == [[3]]


def test_iq_chunk_name_grammar():
    assert IQ_NAME.match("pack2_9-10s.iq").groupdict() == {"pack": "2", "start": "9", "end": "10"}
    assert IQ_NAME.match("pack1.xml") is None


built = pytest.mark.skipif(not (INDEX / "rfuav" / "artifacts.parquet").exists(), reason="index not built")


@pytest.fixture(scope="module")
def tables():
    return {name: read("rfuav", name).to_pylist() for name in ("artifacts", "captures", "groups", "lineage", "evidence")}


@built
def test_index_is_internally_consistent(tables):
    validate("rfuav", tables)


@built
def test_counts_match_the_release(tables):
    assert len(tables["artifacts"]) == 358
    assert len(tables["groups"]) == 53
    assert len(tables["captures"]) == 56  # 53 packs, three of them split by one missing chunk


@built
def test_every_local_archive_is_a_verified_copy_of_the_official_file(tables):
    assert len(tables["lineage"]) == 37
    assert {r["relation"] for r in tables["lineage"]} == {"local_copy_of"}


@built
def test_ids_use_official_names_and_storage_uses_local_names(tables):
    for a in tables["artifacts"]:
        assert "%20" not in a["artifact_id"] and ":" not in a["artifact_id"]
        assert a["storage_relpath"].startswith("rfuav/original/")
    assert any("%20" in a["storage_relpath"] for a in tables["artifacts"])


@built
def test_captures_are_gap_free_runs_in_time_order(tables):
    parts = {}
    for a in tables["artifacts"]:
        start = int(IQ_NAME.match(a["member_chain"][-1].split("\\")[-1])["start"])
        parts.setdefault(a["capture_id"], []).append((a["part_index"], start))
    for capture_id, items in parts.items():
        items.sort()
        assert [p for p, _ in items] == list(range(len(items))), capture_id
        starts = [s for _, s in items]
        assert starts == list(range(starts[0], starts[0] + len(starts))), capture_id
        first, last = re.search(r"_(\d+)-(\d+)s$", capture_id).groups()
        assert (int(first), int(last)) == (starts[0], starts[-1] + 1)


@built
def test_labels_come_from_the_class_folder(tables):
    for c in tables["captures"]:
        folder = c["capture_id"].split("/")[1]
        assert c["label_class"] == folder
        assert c["label_uas_present"] is True
        assert 0 < c["duration_s"] <= c["n_artifacts"]


@built
def test_xml_disagreements_are_recorded_not_silently_resolved(tables):
    conflicts = {(r["entity_id"], r["field"]) for r in tables["evidence"] if r["status"] == "CONFLICTING"}
    assert ("rfuav/RadioMaster BOXER/pack1", "label_disagreement") in conflicts
    assert ("rfuav/DJI MINI3/VTSBW=10/pack2", "xml_link") in conflicts
