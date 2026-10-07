"""Generate the compatibility matrices and the evidence audit from cards and project roles.

Inputs:
    dataset_cards/*.yaml            dataset claims (facts and assessments, graded)
    configs/dataset_roles.yaml      Droneacharya decisions per dataset
    configs/matrices/*.yaml         matrix layouts
Outputs:
    docs/datasets/rf_compatibility_matrix.md
    docs/datasets/acoustic_compatibility_matrix.md
    docs/datasets/evidence_audit.md

Usage:
    python scripts/build_matrices.py           # validate, write all outputs
    python scripts/build_matrices.py --check   # validate and fail if a written output is stale
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya.cards import (  # noqa: E402
    check_cards_against_spec, load_cards, load_roles, load_spec, render_audit, render_matrix, status_counts,
)
from droneacharya.cards.io import REPO_ROOT  # noqa: E402

AUDIT_OUTPUT = "docs/datasets/evidence_audit.md"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    cards, roles = load_cards(), load_roles()
    specs = [load_spec(modality) for modality in ("rf", "acoustic")]
    for spec in specs:
        if problems := check_cards_against_spec(cards, roles, spec):
            sys.exit("\n".join(problems))

    outputs = [(spec.output, render_matrix(cards, roles, spec)) for spec in specs]
    outputs.append((AUDIT_OUTPUT, render_audit(cards, specs)))
    stale = []
    for relpath, text in outputs:
        path = REPO_ROOT / relpath
        if args.check:
            current = path.read_text(encoding="utf-8").replace("\r\n", "\n") if path.exists() else ""
            if current != text:
                stale.append(relpath)
        else:
            path.write_text(text, encoding="utf-8", newline="\n")

    for spec in specs:
        total = {}
        for tally in status_counts(cards, spec).values():
            for status, count in tally.items():
                total[status] = total.get(status, 0) + count
        print(f"{spec.output}: {len(spec.columns)} datasets x {len(spec.rows)} rows; "
              + ", ".join(f"{s} {c}" for s, c in sorted(total.items(), key=lambda kv: -kv[1])))
    if stale:
        sys.exit("stale, run scripts/build_matrices.py: " + ", ".join(stale))


if __name__ == "__main__":
    main()
