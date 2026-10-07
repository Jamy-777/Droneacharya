import pytest

from droneacharya.index import read, validate
from droneacharya.index.acoustic import DDL, DRONENOISE, MIES, wav_header
from droneacharya.paths import INDEX

NAMES = ("artifacts", "captures", "groups", "lineage", "evidence")


def load(dataset):
    if not (INDEX / dataset / "artifacts.parquet").exists():
        pytest.skip(f"{dataset} index not built")
    return {name: read(dataset, name).to_pylist() for name in NAMES}


def test_wav_header_reads_pcm_and_rejects_other_bytes():
    header = (b"RIFF" + (36 + 8).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
              + (1).to_bytes(2, "little") + (2).to_bytes(2, "little") + (44100).to_bytes(4, "little")
              + (176400).to_bytes(4, "little") + (4).to_bytes(2, "little") + (16).to_bytes(2, "little")
              + b"data" + (176400).to_bytes(4, "little"))
    info = wav_header(header)
    assert (info["format"], info["channels"], info["sample_rate"], info["bits"]) == ("pcm", 2, 44100, 16)
    assert info["duration_s"] == pytest.approx(1.0)
    assert wav_header(b"not a wav") == {"valid": False}


def test_file_name_grammars():
    assert DDL.match("20210329141240MINI0030240312886R290321-T004-005236.wav")["segment"] == "T004"
    assert DRONENOISE.match("46212081_Ed_3p_10_H00_N_C_nw_ev1_M5.wav")["mic"] == "5"
    assert MIES.match("x4_d5_mavicmini2_10m_0m_230415_0061.WAV")["take"] == "0061"


@pytest.mark.parametrize("dataset", ["rma", "noisy_rf", "esc50", "svanstrom", "uavirbase", "dronenoise",
                                     "miesikowska_uav", "ddl"])
def test_index_is_internally_consistent(dataset):
    validate(dataset, load(dataset))


def test_rma_capture_length_is_constant_within_each_archive():
    lengths = {}
    for c in load("rma")["captures"]:
        lengths.setdefault(c["group_id"], set()).add(c["duration_s"])
    assert all(len(v) == 1 for v in lengths.values())
    assert lengths["rma/MavicRC1"] == {1.0} and lengths["rma/mini2RC"] == {0.1}


def test_noisy_rf_is_one_unsplittable_group():
    t = load("noisy_rf")
    assert len(t["captures"]) == 98705
    assert [g["status"] for g in t["groups"]] == ["UNKNOWN"]
    assert {c["stored_in"] for c in t["captures"]} == {t["artifacts"][0]["artifact_id"]}
    assert sum(not c["label_uas_present"] for c in t["captures"]) == 52552


def test_svanstrom_identical_clips_share_a_group():
    groups = {c["capture_id"].rsplit("/", 1)[-1]: c["group_id"] for c in load("svanstrom")["captures"]}
    assert groups["BACKGROUND_010"] == groups["BACKGROUND_019"]


def test_uavirbase_has_four_ambient_recordings():
    assert sum(not c["label_uas_present"] for c in load("uavirbase")["captures"]) == 4


def test_ddl_unusable_clips_are_kept_but_outside_captures():
    t = load("ddl")
    assert len(t["artifacts"]) == 62103
    assert sum(a["capture_id"] is None for a in t["artifacts"]) == 3945 + 11
    assert all(c["label_uas_present"] for c in t["captures"])  # the release has no no-drone clips


def test_esc50_keeps_the_official_folds():
    assert {c["official_split"] for c in load("esc50")["captures"]} == {f"fold{i}" for i in range(1, 6)}
