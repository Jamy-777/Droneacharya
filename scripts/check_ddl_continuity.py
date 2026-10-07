"""Is DDL audio continuous across 0.1 s clip boundaries?

The index joins clips with consecutive sequence numbers into one capture. If
the recording is continuous, the jump between the last sample of a clip and
the first of the next looks like any other adjacent-sample jump. Here: for
each capture, every boundary jump is ranked against the within-clip jumps of
the same capture; continuity predicts ~1% of boundary jumps above the
within-clip 99th percentile.

Output: DroneacharyaData/results/checks/ddl_continuity.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya import signal  # noqa: E402
from droneacharya.index import read  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

CLIP = 9600          # 0.1 s at 96 kHz
BOUNDARIES = 200     # per capture


def main():
    captures = sorted((c for c in read("ddl", "captures").to_pylist() if c["n_artifacts"] > BOUNDARIES + 1),
                      key=lambda c: c["capture_id"])
    rows = []
    for c in captures:
        x = signal.read(c["capture_id"], count=CLIP * (BOUNDARIES + 1)).samples  # (8, n)
        jumps = np.abs(np.diff(x, axis=1))
        at_boundary = np.zeros(jumps.shape[1], bool)
        at_boundary[CLIP - 1::CLIP] = True
        within = jumps[:, ~at_boundary]
        boundary = jumps[:, at_boundary]
        p99 = np.percentile(within, 99, axis=1, keepdims=True)
        rows.append({"capture_id": c["capture_id"], "boundaries": int(boundary.shape[1]),
                     "share_above_within_p99": float(np.mean(boundary > p99)),
                     "median_ratio": float(np.median(boundary) / np.median(within))})
        print(f"{c['capture_id']}: {rows[-1]['share_above_within_p99']:.3f} above p99, "
              f"median ratio {rows[-1]['median_ratio']:.2f}", flush=True)
    summary = {"captures": len(rows), "channels": 8, "boundaries_per_capture": BOUNDARIES,
               "share_above_within_p99": {"median": float(np.median([r["share_above_within_p99"] for r in rows])),
                                          "max": float(np.max([r["share_above_within_p99"] for r in rows]))},
               "expected_if_continuous": 0.01, "per_capture": rows}
    out = DATA_ROOT / "results" / "checks" / "ddl_continuity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_capture"}))


if __name__ == "__main__":
    main()
