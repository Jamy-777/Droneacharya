"""Which drone model is this? DRFF-R2 Dataset 3: across days, across receivers, and on physical units never seen.

Design fixed before any result was seen (2026-10-08):

  data     DRFF-R2 Dataset 3 (single drone hovering, 5.745 GHz): 179 captures, 8 models, 26 units, days d1–d3,
           receivers u1/u2. 20 steered 40 MHz tiles per capture (rf_common.iq_steered: baseband only, unit RMS
           features — absolute frequency and power cannot enter), evenly spaced over the 1.4 s capture
  label    drone model (unit tag without its number: mavicAir2_3 -> mavicAir2)
  tests    I1a train day 1 -> test day 2, receiver u2      (unseen day)
           I1b train day 1 -> test day 3, receiver u2      (unseen day)
           I2  train day 2 u2 -> test day 2 u1             (unseen receiver)
           I3  day 1: each unit of the 4 multi-unit models held out in turn, trained on everything else of day 1
               (unseen physical unit; predictions pooled over the 22 held-out units)
  models   primary: gbm on rf_features; secondary: multi-class TileNet CNN on the GPU (cnn.py recipe)
  metric   balanced accuracy over tiles (chance 1/8 = 0.125); p by permuting model labels between whole units;
           95% interval by resampling units
  decision a test "works" if its interval is above 0.125 and p < 0.05

Parallel CPU (12 workers) for tiles and features; GPU for the CNN.
Outputs: DroneacharyaData/results/drff_identification/*.json, docs/results/drff_identification.md.
"""
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from droneacharya import cnn  # noqa: E402
from droneacharya import evaluation as E  # noqa: E402
from droneacharya.cache import key  # noqa: E402
from droneacharya.index import read  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

OUT = DATA_ROOT / "results" / "drff_identification"
REPORT = ROOT / "docs" / "results" / "drff_identification.md"
WIDTH, TILES, LENGTH, BLOCK = 40e6, 20, 140_000_000, 25_000


def _capture_rows(capture_id):
    with threadpool_limits(1):
        from droneacharya import rf_common as R
        from droneacharya.rf_features import rf_tile_features
        starts = np.linspace(0, LENGTH - BLOCK, TILES).astype(int)
        tiles, _ = R.iq_steered(capture_id, starts, width=WIDTH)
        return [rf_tile_features(t, WIDTH) for t, _ in tiles], np.stack([t for t, _ in tiles])


def load():
    fields = defaultdict(dict)
    for r in read("drff_r2", "evidence").to_pylist():
        if r["field"].startswith("mat."):
            fields[r["entity_id"]][r["field"][4:]] = r["value"]
    captures = sorted((c for c in read("drff_r2", "captures").to_pylist() if "/dataset3-" in c["capture_id"]),
                      key=lambda c: c["capture_id"])
    tag = key(_capture_rows, {"tiles": TILES, "width": WIDTH}, ["drff_r2"])
    feats_path, tiles_path = OUT / f"ds3-{tag}.features.json", OUT / f"ds3-{tag}.tiles.npy"
    if not feats_path.exists():
        out = Parallel(n_jobs=12)(delayed(_capture_rows)(c["capture_id"]) for c in captures)
        feats_path.write_text(json.dumps([f for feats, _ in out for f in feats]), encoding="utf-8")
        np.save(tiles_path, np.concatenate([tiles for _, tiles in out]))
    feats = json.loads(feats_path.read_text(encoding="utf-8"))
    names = sorted(feats[0])
    meta = [{"unit": c["label_class"], "model": c["label_class"].rsplit("_", 1)[0],
             "day": fields[c["capture_id"]]["D"], "rx": fields[c["capture_id"]]["U"]}
            for c in captures for _ in range(TILES)]
    print(f"  {len(feats)} tiles from {len(captures)} captures", flush=True)
    return np.array([[f[k] for k in names] for f in feats]), np.load(tiles_path), meta


def save(name, result):
    result["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"{name}: balanced accuracy {result['balanced_accuracy']:.3f} CI {[round(v, 3) for v in result['ci95']]} "
          f"p={result['permutation']['p_value']:.4f}", flush=True)
    return result


def confusion(y, predicted):
    table = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y, predicted):
        table[t][p] += 1
    return {t: dict(v) for t, v in table.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    X, tiles, meta = load()
    specs = cnn.spectrograms(tiles)
    model = np.array([m["model"] for m in meta])
    unit = np.array([m["unit"] for m in meta])
    day, rx = np.array([m["day"] for m in meta]), np.array([m["rx"] for m in meta])

    def fit_predict(train, test, kind):
        if kind == "gbm":
            return E.model("gbm").fit(X[train], model[train]).predict(X[test]).astype(object)
        net = cnn.train_multiclass(specs[np.flatnonzero(train)], model[train].tolist())
        return cnn.predict_multiclass(net, specs[np.flatnonzero(test)])

    tests = {
        "I1a day1 -> day2 (u2)": ((day == "d1"), (day == "d2") & (rx == "u2")),
        "I1b day1 -> day3 (u2)": ((day == "d1"), (day == "d3") & (rx == "u2")),
        "I2 u2 -> u1 (day 2)": ((day == "d2") & (rx == "u2"), (day == "d2") & (rx == "u1")),
    }
    results = {}
    for kind in ("gbm", "cnn"):
        for name, (train, test) in tests.items():
            predicted = fit_predict(train, test, kind)
            r = E.evaluate_external_labels(predicted, model[test], unit[test])
            r.update({"task": name, "model": kind, "chance": 1 / 8, "train_tiles": int(train.sum()),
                      "confusion": confusion(model[test], predicted)})
            results[f"{name} [{kind}]"] = save(f"{name.split()[0]}_{kind}", r)
        multi = [u for u in sorted(set(unit[day == "d1"]))
                 if len(set(unit[(day == "d1") & (model == u.rsplit('_', 1)[0])])) > 1]
        pred = np.empty(len(model), dtype=object)
        held = np.zeros(len(model), bool)
        for u in multi:
            test = (day == "d1") & (unit == u)
            pred[test] = fit_predict((day == "d1") & (unit != u), test, kind)
            held |= test
        r = E.evaluate_external_labels(pred[held], model[held], unit[held])
        r.update({"task": "I3 unseen physical unit (day 1)", "model": kind, "chance": 1 / 8, "held_out_units": multi,
                  "confusion": confusion(model[held], pred[held])})
        results[f"I3 unseen unit [{kind}]"] = save(f"I3_{kind}", r)

    lines = ["<!-- Generated by scripts/run_drff_identification.py; results in DroneacharyaData/results/drff_identification/. -->", "",
             "> Raw output, rewritten on every run. Which numbers may be quoted, and with what limits: [claims.md](claims.md).",
             "", "# DRFF-R2: which drone model is this?", "",
             "Design fixed before results (script docstring). 8 models, chance 0.125. p permutes model labels between",
             "whole physical units; intervals resample units.", "",
             "| Test | Model | Balanced accuracy [95% CI] | p | Test units by model |", "| --- | --- | --- | --- | --- |"]
    for name, r in results.items():
        lines.append(f"| {r['task']} | {r['model']} | {r['balanced_accuracy']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}] | "
                     f"{r['permutation']['p_value']:.4f} | {r['groups_per_label']} |")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done in {(time.time() - start) / 60:.1f} min; report {REPORT}")


if __name__ == "__main__":
    main()
