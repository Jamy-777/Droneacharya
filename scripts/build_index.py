"""Build the artifact / capture / group index of one dataset into DroneacharyaData/index/<dataset>/.

Usage:
    python scripts/build_index.py dronerf
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya.index import BUILDERS, write  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(BUILDERS))
    args = parser.parse_args()

    tables = BUILDERS[args.dataset]()
    out = write(args.dataset, tables)
    print(f"Index written to {out}")
    for name, rows in tables.items():
        print(f"  {name}: {len(rows)} rows")
    print("  lineage:", dict(Counter(r["relation"] for r in tables["lineage"])))
    for row in tables["evidence"]:
        print(f"  evidence {row['field']} = {row['value']} [{row['status']}]")


if __name__ == "__main__":
    main()
