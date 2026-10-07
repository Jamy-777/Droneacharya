"""Render a compatibility matrix (markdown) from dataset cards."""

GENERATED_NOTE = (
    "<!-- Generated from dataset_cards/*.yaml and configs/matrices/{modality}.yaml "
    "by scripts/build_matrices.py. Edit the cards, not this file. -->"
)


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def render_matrix(cards, spec):
    columns = [cards[c.id] for c in spec.columns]
    lines = [GENERATED_NOTE.format(modality=spec.modality), "", spec.preamble.rstrip(), ""]
    lines += _table(
        ["Parameter"] + [c.label for c in spec.columns],
        [[f"**{row.label}**"] + [card.facts[row.key].text for card in columns] for row in spec.rows],
    )
    for supplement in spec.supplements:
        card = cards[supplement.id]
        lines += ["", f"## {supplement.heading}", ""]
        lines += _table(["Parameter", "Value"], [[label, fact.text] for label, fact in card.supplementary.items()])
    if spec.cross_dataset:
        lines += ["", f"## {spec.cross_dataset_heading or 'Cross-dataset facts'}", ""]
        lines += _table(["Parameter", "Value"], [[label, fact.text] for label, fact in spec.cross_dataset.items()])
    return "\n".join(lines) + "\n"


def status_counts(cards, spec):
    """{dataset id: {status: number of matrix cells}} for the evidence report."""
    counts = {}
    for column in spec.columns:
        tally = {}
        for row in spec.rows:
            status = cards[column.id].facts[row.key].status
            tally[status] = tally.get(status, 0) + 1
        counts[column.id] = tally
    return counts
