from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from droneacharya.cards import (
    Card, Fact, Measured, check_cards_against_spec, load_cards, load_spec, render_matrix,
)
from droneacharya.cards.io import REPO_ROOT

DATA_ROOT = Path(r"D:\Weeeeeeee\DroneacharyaData")


@pytest.fixture(scope="module")
def cards():
    return load_cards()


@pytest.mark.parametrize("modality", ["rf", "acoustic"])
def test_cards_fill_their_matrix(cards, modality):
    assert check_cards_against_spec(cards, load_spec(modality)) == []


@pytest.mark.parametrize("modality", ["rf", "acoustic"])
def test_committed_matrix_is_generated_from_cards(cards, modality):
    spec = load_spec(modality)
    on_disk = (REPO_ROOT / spec.output).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert on_disk == render_matrix(cards, spec), "run scripts/build_matrices.py"


def test_adopted_cards_store_raw_data_under_their_id(cards):
    for card in cards.values():
        if card.adoption == "adopted":
            assert card.storage.raw_dir == card.id


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="data drive not attached")
def test_raw_dirs_and_name_maps_exist(cards):
    for card in cards.values():
        if card.storage:
            assert (DATA_ROOT / "raw" / card.storage.raw_dir).is_dir(), card.id
            if card.storage.name_map:
                assert (DATA_ROOT / card.storage.name_map).is_file(), card.id


def test_empirical_needs_n():
    with pytest.raises(ValidationError):
        Fact(text="x", status="EMPIRICAL")
    assert Fact(text="x", status="EMPIRICAL", n=3).n == 3


def test_absent_status_carries_no_value_or_n():
    with pytest.raises(ValidationError):
        Measured[float](value=1.0, status="UNKNOWN")
    with pytest.raises(ValidationError):
        Fact(text="x", status="NOT_APPLICABLE", n=2)
    assert Measured[float](status="UNKNOWN").value is None


def test_measured_needs_value_or_varies_not_both():
    with pytest.raises(ValidationError):
        Measured[float](status="VERIFIED")
    with pytest.raises(ValidationError):
        Measured[float](value=1.0, varies=True, status="VERIFIED")
    assert Measured[float](varies=True, status="VERIFIED").varies


def test_fact_text_must_fit_a_table_cell():
    with pytest.raises(ValidationError):
        Fact(text="a | b", status="ASSESSMENT")
    with pytest.raises(ValidationError):
        Fact(text="a\nb", status="ASSESSMENT")


def test_unknown_keys_and_modalities_are_rejected(cards):
    adapter = TypeAdapter(Card)
    good = cards["uavirbase"].model_dump(exclude_defaults=False)
    adapter.validate_python(good)
    with pytest.raises(ValidationError):
        adapter.validate_python({**good, "surprise": 1})
    with pytest.raises(ValidationError):
        adapter.validate_python({**good, "modality": "video"})
    with pytest.raises(ValidationError):
        adapter.validate_python({**good, "signal": {**good["signal"], "sample_rate": 1}})
