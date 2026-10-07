from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from droneacharya.cards import (
    Card, Claim, Fact, Measured, check_cards_against_spec, iter_claims, load_cards, load_roles, load_spec,
    render_audit, render_matrix,
)
from droneacharya.cards.io import REPO_ROOT

DATA_ROOT = Path(r"D:\Weeeeeeee\DroneacharyaData")
MODALITIES = ["rf", "acoustic"]


@pytest.fixture(scope="module")
def cards():
    return load_cards()


@pytest.fixture(scope="module")
def roles():
    return load_roles()


def _on_disk(relpath):
    return (REPO_ROOT / relpath).read_text(encoding="utf-8").replace("\r\n", "\n")


@pytest.mark.parametrize("modality", MODALITIES)
def test_cards_and_roles_fill_their_matrix(cards, roles, modality):
    assert check_cards_against_spec(cards, roles, load_spec(modality)) == []


@pytest.mark.parametrize("modality", MODALITIES)
def test_committed_matrix_is_generated(cards, roles, modality):
    spec = load_spec(modality)
    assert _on_disk(spec.output) == render_matrix(cards, roles, spec), "run scripts/build_matrices.py"


def test_committed_audit_is_generated(cards):
    specs = [load_spec(m) for m in MODALITIES]
    assert _on_disk("docs/datasets/evidence_audit.md") == render_audit(cards, specs), "run scripts/build_matrices.py"


def test_reconnaissance_baseline_has_no_ungraded_claims(cards):
    specs = [load_spec(m) for m in MODALITIES]
    ungraded = [f"{d}.{s}.{k}" for d, s, k, _, claim in iter_claims(cards, specs) if claim.status == "UNGRADED"]
    assert ungraded == []


def test_policy_rows_live_only_in_roles(cards):
    policy_keys = {row.key for m in MODALITIES for row in load_spec(m).rows if row.policy}
    for card in cards.values():
        assert not policy_keys & set(card.facts), card.id


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


def test_empirical_needs_n_and_unit():
    with pytest.raises(ValidationError):
        Fact(text="x", kind="FACT", status="EMPIRICAL", source="s")
    with pytest.raises(ValidationError):
        Fact(text="x", kind="FACT", status="EMPIRICAL", n=3, source="s")
    assert Fact(text="x", kind="FACT", status="EMPIRICAL", n=3, n_unit="file", source="s").n_unit == "file"


def test_graded_claims_need_a_source():
    with pytest.raises(ValidationError):
        Fact(text="x", kind="FACT", status="AUTHOR_DOC")
    assert Fact(text="x", kind="FACT", status="UNKNOWN").source is None


def test_assessment_is_a_kind_not_a_status():
    with pytest.raises(ValidationError):
        Fact(text="x", kind="FACT", status="ASSESSMENT", source="s")
    assert Fact(text="HIGH", kind="ASSESSMENT", status="EMPIRICAL", n=2, n_unit="file", source="s").kind == "ASSESSMENT"


def test_conflicting_needs_two_claims():
    a = Claim(claim="float64", status="EMPIRICAL", source="s", n=8, n_unit="file")
    b = Claim(claim="float32", status="AUTHOR_DOC", source="paper")
    with pytest.raises(ValidationError):
        Fact(text="x", kind="FACT", status="CONFLICTING")
    with pytest.raises(ValidationError):
        Fact(text="x", kind="FACT", status="CONFLICTING", conflict=[a])
    with pytest.raises(ValidationError):
        Fact(text="x", kind="FACT", status="EMPIRICAL", n=8, n_unit="file", source="s", conflict=[a, b])
    assert len(Fact(text="x", kind="FACT", status="CONFLICTING", conflict=[a, b]).conflict) == 2


def test_conflicting_measured_keeps_the_stored_value():
    a = Claim(claim="float64", status="EMPIRICAL", source="s", n=8, n_unit="file")
    b = Claim(claim="float32", status="AUTHOR_DOC", source="paper")
    assert Measured[str](value="complex128", status="CONFLICTING", conflict=[a, b]).value == "complex128"


def test_absent_status_carries_no_value_or_n():
    with pytest.raises(ValidationError):
        Measured[float](value=1.0, status="UNKNOWN")
    with pytest.raises(ValidationError):
        Fact(text="x", kind="FACT", status="NOT_APPLICABLE", n=2, n_unit="file")


def test_measured_needs_value_or_varies_not_both():
    with pytest.raises(ValidationError):
        Measured[float](status="VERIFIED", source="s")
    with pytest.raises(ValidationError):
        Measured[float](value=1.0, varies=True, status="VERIFIED", source="s")
    assert Measured[float](varies=True, status="VERIFIED", source="s").varies


def test_fact_text_must_fit_a_table_cell():
    with pytest.raises(ValidationError):
        Fact(text="a | b", kind="ASSESSMENT", status="UNKNOWN")
    with pytest.raises(ValidationError):
        Fact(text="a\nb", kind="ASSESSMENT", status="UNKNOWN")


def test_unknown_keys_and_modalities_are_rejected(cards):
    adapter = TypeAdapter(Card)
    good = cards["uavirbase"].model_dump(exclude_none=True)
    adapter.validate_python(good)
    with pytest.raises(ValidationError):
        adapter.validate_python({**good, "surprise": 1})
    with pytest.raises(ValidationError):
        adapter.validate_python({**good, "modality": "video"})
    with pytest.raises(ValidationError):
        adapter.validate_python({**good, "signal": {**good["signal"], "sample_rate": 1}})
