"""Generate docs/datasets/*_compatibility_matrix.md from the dataset cards.

Usage:
    python scripts/build_matrices.py           # validate cards, write both matrices
    python scripts/build_matrices.py --check   # validate and fail if a written matrix is stale
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya.cards import check_cards_against_spec, load_cards, load_spec, render_matrix, status_counts  # noqa: E402
from droneacharya.cards.io import REPO_ROOT  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    cards = load_cards()
    failed = False
    for modality in ("rf", "acoustic"):
        spec = load_spec(modality)
        if problems := check_cards_against_spec(cards, spec):
            print("\n".join(problems))
            sys.exit(1)
        output = REPO_ROOT / spec.output
        text = render_matrix(cards, spec)
        if args.check:
            current = output.read_text(encoding="utf-8") if output.exists() else ""
            if current != text:
                print(f"{spec.output} is stale; run scripts/build_matrices.py")
                failed = True
        else:
            output.write_text(text, encoding="utf-8", newline="\n")
        total = {}
        for tally in status_counts(cards, spec).values():
            for status, count in tally.items():
                total[status] = total.get(status, 0) + count
        print(f"{spec.output}: {len(spec.columns)} datasets x {len(spec.rows)} rows; "
              + ", ".join(f"{s} {c}" for s, c in sorted(total.items(), key=lambda kv: -kv[1])))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
