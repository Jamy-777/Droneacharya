"""DRFF-R2 identification: bandwidth alone, and an unseen unit on an unseen day (claims.md I; audit point 6).

Design committed before running (2026-10-08).

  rows       the cached Dataset 3 features of scripts/run_drff_identification.py (179 captures, 8 models, 26 units,
             20 steered 40 MHz tiles per capture): the rows the reported numbers came from. All 26 units are on
             day 1 (receiver u2); every unit on days 2–3 and on receiver u1 is also on day 1.
  model      gbm on all rf_features, as the identification run (its I1a, I1b, I2, I3 are refitted and checked
             against the saved results)
  bandwidth  gbm on occupied_bw_mhz alone (DJI links switch between 10, 20 and 40 MHz), on the same splits.
             Paired difference full − bandwidth-only, interval resampling units
  single     the strongest single feature per test, gbm on that one feature, picked on the test rows
             (optimistic for the baseline), and the paired difference against it
  test D     unseen physical unit on an unseen day: each unit of a model with ≥ 2 units on day 1 that also appears
             on day 2 or 3 at receiver u2 is held out in turn; train on day 1 without that unit, test on its day-2
             and day-3 captures at u2; predictions pooled over the held-out units
  test D-rx  the same on day 2 at receiver u1 (unseen unit + day + receiver): at most 4 units, descriptive only
  metric     balanced accuracy over the truth classes present; p by permuting model labels between units (2,000);
             95% interval resampling units
  decision   D "works" if its interval's lower bound is above the permutation null's 95th percentile and p < 0.05.
             This is stricter than I1–I3's "above 0.125": with fewer truth classes than predicted classes the null
             sits above 1/8. A test identifies "beyond bandwidth" only if full − bandwidth-only is above zero.

CPU: single-feature fits in parallel, one thread per worker (workers import no torch).
Output: DroneacharyaData/results/checks/drff_bandwidth_and_test_d.json
"""
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya import evaluation as E  # noqa: E402
from droneacharya.index import read  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

RESULTS = DATA_ROOT / "results"
OUT = RESULTS / "checks"
TILES = 20


def load():
    fields = defaultdict(dict)
    for r in read("drff_r2", "evidence").to_pylist():
        if r["field"].startswith("mat."):
            fields[r["entity_id"]][r["field"][4:]] = r["value"]
    captures = sorted((c for c in read("drff_r2", "captures").to_pylist() if "/dataset3-" in c["capture_id"]),
                      key=lambda c: c["capture_id"])
    feats = json.loads(sorted((RESULTS / "drff_identification").glob("ds3-*.features.json"))[-1].read_text(encoding="utf-8"))
    assert len(feats) == TILES * len(captures)
    names = sorted(feats[0])
    unit = np.array([c["label_class"] for c in captures for _ in range(TILES)])
    day = np.array([fields[c["capture_id"]]["D"] for c in captures for _ in range(TILES)])
    rx = np.array([fields[c["capture_id"]]["U"] for c in captures for _ in range(TILES)])
    return np.array([[f[k] for k in names] for f in feats]), names, unit, np.array([u.rsplit("_", 1)[0] for u in unit]), day, rx


def _predict(X_train, y_train, X_test):
    with threadpool_limits(1):
        return E.model("gbm").fit(X_train, y_train).predict(X_test).astype(object)


def splits_of(unit, model, day, rx):
    """Each test as a list of (train mask, test mask) folds whose predictions are pooled."""
    d1 = day == "d1"
    multi = [u for u in sorted(set(unit[d1])) if len(set(unit[d1 & (model == u.rsplit("_", 1)[0])])) > 1]
    tests = {"I1a day1 -> day2 (u2)": [(d1, (day == "d2") & (rx == "u2"))],
             "I1b day1 -> day3 (u2)": [(d1, (day == "d3") & (rx == "u2"))],
             "I2 u2 -> u1 (day 2)": [((day == "d2") & (rx == "u2"), (day == "d2") & (rx == "u1"))],
             "I3 unseen unit (day 1)": [(d1 & (unit != u), d1 & (unit == u)) for u in multi]}
    later = (day != "d1") & (rx == "u2")
    tests["D unseen unit + unseen day (u2)"] = [(d1 & (unit != u), later & (unit == u)) for u in multi
                                                 if (later & (unit == u)).any()]
    on_u1 = (day == "d2") & (rx == "u1")
    tests["D-rx unseen unit + day + receiver (d2 u1)"] = [(d1 & (unit != u), on_u1 & (unit == u)) for u in multi
                                                          if (on_u1 & (unit == u)).any()]
    return tests


def pooled(folds, X, model):
    jobs = [delayed(_predict)(X[train], model[train], X[test]) for train, test in folds]
    predicted = Parallel(n_jobs=-1)(jobs)
    held = np.concatenate([np.flatnonzero(test) for _, test in folds])
    return held, np.concatenate(predicted)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, names, unit, model, day, rx = load()
    bw = names.index("occupied_bw_mhz")
    result = {}
    for name, folds in splits_of(unit, model, day, rx).items():
        held, full = pooled(folds, X, model)
        _, bandwidth = pooled(folds, X[:, [bw]], model)
        singles = {n: pooled(folds, X[:, [j]], model)[1] for j, n in enumerate(names)}
        y, g = model[held], unit[held]
        best = max(singles, key=lambda n: E.metric(y, singles[n]))
        r = E.evaluate_external_labels(full, y, g)
        tag = name.split()[0]
        saved = RESULTS / "drff_identification" / f"{tag}_gbm.json"
        r.update({"test": name, "units_held_out": sorted(set(g)), "chance": 1 / 8,
                  "saved_balanced_accuracy": json.loads(saved.read_text(encoding="utf-8"))["balanced_accuracy"]
                  if saved.exists() else None,
                  "bandwidth_only": E.metric(y, bandwidth),
                  "full_minus_bandwidth": E.paired_label_difference(full, bandwidth, y, g),
                  "best_single_feature": {"feature": best, "balanced_accuracy": E.metric(y, singles[best])},
                  "full_minus_best_single": E.paired_label_difference(full, singles[best], y, g)})
        if tag.startswith("D"):
            r["works"] = bool(r["ci95"][0] > r["permutation"]["null_95th"] and r["permutation"]["p_value"] < 0.05)
        result[name] = r
        d = r["full_minus_bandwidth"]
        print(f"{name:42} full {r['balanced_accuracy']:.3f} {[round(v, 2) for v in r['ci95']]} "
              f"(saved {r['saved_balanced_accuracy']}) null95 {r['permutation']['null_95th']:.3f} p={r['permutation']['p_value']:.4f} | "
              f"bandwidth {r['bandwidth_only']:.3f}, full − bw {d['difference']:+.3f} {[round(v, 2) for v in d['ci95']]} | "
              f"best single {best} {r['best_single_feature']['balanced_accuracy']:.3f}", flush=True)
    result["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (OUT / "drff_bandwidth_and_test_d.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
