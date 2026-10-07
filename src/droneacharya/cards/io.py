"""Load and cross-check dataset cards and matrix specs."""
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from .schema import Card, MatrixSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
CARDS_DIR = REPO_ROOT / "dataset_cards"
SPECS_DIR = REPO_ROOT / "configs" / "matrices"

_card_adapter = TypeAdapter(Card)


def _read_yaml(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_card(path):
    try:
        card = _card_adapter.validate_python(_read_yaml(path))
    except ValueError as error:
        raise ValueError(f"{Path(path).name}: {error}") from error
    if card.id != Path(path).stem:
        raise ValueError(f"{path}: id '{card.id}' does not match the file name")
    return card


def load_cards(cards_dir=CARDS_DIR):
    """All cards keyed by id, in file-name order."""
    return {card.id: card for card in (load_card(p) for p in sorted(Path(cards_dir).glob("*.yaml")))}


def load_spec(modality, specs_dir=SPECS_DIR):
    return MatrixSpec.model_validate(_read_yaml(Path(specs_dir) / f"{modality}.yaml"))


def check_cards_against_spec(cards, spec):
    """Problems that would make the generated matrix wrong or incomplete."""
    problems = []
    row_keys = [r.key for r in spec.rows]
    of_modality = {i for i, c in cards.items() if c.modality == spec.modality}
    columns = {c.id for c in spec.columns}
    if missing := of_modality - columns:
        problems.append(f"{spec.modality} cards without a matrix column: {sorted(missing)}")
    for column in spec.columns:
        card = cards.get(column.id)
        if card is None:
            problems.append(f"column '{column.id}' has no card")
            continue
        if card.modality != spec.modality:
            problems.append(f"column '{column.id}' is a {card.modality} card")
        if missing := [k for k in row_keys if k not in card.facts]:
            problems.append(f"{card.id}: facts missing for rows {missing}")
        if unknown := [k for k in card.facts if k not in row_keys]:
            problems.append(f"{card.id}: facts for rows not in the {spec.modality} matrix {unknown}")
    for supplement in spec.supplements:
        if supplement.id not in columns:
            problems.append(f"supplement '{supplement.id}' is not a matrix column")
        elif not cards[supplement.id].supplementary:
            problems.append(f"supplement '{supplement.id}' has no supplementary rows")
    return problems
