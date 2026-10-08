"""Does widening the acoustic training data improve detection on held-out recording chains?

Design fixed before any result was seen (2026-10-08):

  test sets      Svanström and UaVirBASE, each held out whole (both classes from one chain);
                 the same windows as scripts/run_detectors.py
  training       A  the other test dataset only (the current detector)
                 B  A + ESC-50 negatives
                 C  every other acoustic dataset (leave-one-dataset-out)
                 D  C without the drone-only datasets (DDL, Mięsikowska, DroneNoise)
  model          gbm on the level-normalised features of features.py, fixed in advance
  balance        each training source contributes at most 2,000 windows (seeded sample)
  decision rule  a variant helps only if its paired AUC difference against A, on identical
                 test rows, has a group-bootstrap 95% interval above zero

Feature extraction runs in parallel over captures on every CPU core (one thread per worker).
Outputs: DroneacharyaData/results/acoustic_widening/*.json, docs/results/acoustic_widening.md.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya import baselines as B  # noqa: E402
from droneacharya import evaluation as E  # noqa: E402
from droneacharya import features as F  # noqa: E402
from droneacharya import splits  # noqa: E402
from droneacharya.cache import cached  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

OUT = DATA_ROOT / "results" / "acoustic_widening"
REPORT = Path(__file__).resolve().parents[1] / "docs" / "results" / "acoustic_widening.md"
SEED = 0
SOURCE_CAP = 2000
# per dataset: (max windows per capture, seconds read per capture) — test sets match run_detectors.py
WINDOWS = {"svanstrom": (10, None), "uavirbase": (10, 10), "esc50": (5, None), "ddl": (60, 60),
           "miesikowska_uav": (10, 10), "dronenoise": (10, 10)}
DRONE_ONLY = ("ddl", "miesikowska_uav", "dronenoise")
TESTS = ("svanstrom", "uavirbase")
WORKERS = 8   # RAM cap: a DDL worker holds up to 60 s of 8-channel 96 kHz audio


def _capture_rows(capture_id, max_windows, max_seconds):
    with threadpool_limits(1):
        rate = None
        if max_seconds:
            from droneacharya import signal
            rate = signal.read(capture_id, count=1).sample_rate_hz
        windows, rate = B.audio_windows(capture_id, max_windows=max_windows,
                                        max_samples=int(max_seconds * rate) if max_seconds else None)
        return [{"capture_id": capture_id, **F.audio_detector_features(w, rate)} for w in windows]


def dataset_rows(dataset):
    max_windows, max_seconds = WINDOWS[dataset]
    captures = {c["capture_id"]: c for c in splits.captures_of(dataset) if c["label_uas_present"] is not None}

    def compute():
        chunks = Parallel(n_jobs=WORKERS)(delayed(_capture_rows)(cid, max_windows, max_seconds) for cid in sorted(captures))
        return [row for chunk in chunks for row in chunk]

    rows = cached(OUT, dataset, compute, {"max_windows": max_windows, "max_seconds": max_seconds}, [dataset])
    for r in rows:
        c = captures[r["capture_id"]]
        r["y"], r["group"], r["dataset"] = c["label_uas_present"], c["group_id"], dataset
    print(f"  {dataset}: {len(rows)} windows from {len(captures)} captures", flush=True)
    return rows


def capped(rows):
    if len(rows) <= SOURCE_CAP:
        return rows
    index = np.random.default_rng(SEED).choice(len(rows), SOURCE_CAP, replace=False)
    return [rows[i] for i in sorted(index)]


def matrix(rows):
    names = sorted(k for k in rows[0] if k.startswith(("band_", "peakiness", "spectral", "modulation")))
    return np.array([[r[k] for k in names] for r in rows]), np.array([r["y"] for r in rows])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    data = {d: dataset_rows(d) for d in WINDOWS}
    results = {}
    for test in TESTS:
        other = [d for d in TESTS if d != test][0]
        variants = {
            "A_single_chain": [other],
            "B_plus_esc50_negatives": [other, "esc50"],
            "C_leave_one_dataset_out": [d for d in WINDOWS if d != test],
            "D_without_drone_only_sets": [d for d in WINDOWS if d != test and d not in DRONE_ONLY],
        }
        X_test, y_test = matrix(data[test])
        groups_test = np.array([r["group"] for r in data[test]])
        scores = {}
        for name, sources in variants.items():
            train = [r for s in sources for r in capped(data[s])]
            X_train, y_train = matrix(train)
            scores[name] = E.positive_scores(E.model("gbm").fit(X_train, y_train), X_test)
            result = E.evaluate_external(scores[name], y_test, groups_test, "gbm")
            result.update({"test": test, "variant": name, "training_sources": sources,
                           "training_windows": {s: len(capped(data[s])) for s in sources},
                           "training_balance": {"drone": int(y_train.sum()), "not_drone": int((~y_train).sum())}})
            if name != "A_single_chain":
                result["vs_A"] = E.paired_difference(scores[name], scores["A_single_chain"], y_test, groups_test)
            result["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            (OUT / f"{test}__{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
            results[(test, name)] = result
            d = result.get("vs_A")
            print(f"test {test:10} {name:28} AUC {result['roc_auc']:.3f} [{result['ci95'][0]:.2f}, {result['ci95'][1]:.2f}] "
                  f"p={result['permutation']['p_value']:.3f}" + (f"  vs A {d['difference']:+.3f} "
                  f"[{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]" if d else ""), flush=True)
    lines = ["<!-- Generated by scripts/run_acoustic_widening.py; results in DroneacharyaData/results/acoustic_widening/. -->",
             "", "# Acoustic training-data widening", "",
             "Design fixed before results (see the script docstring). Each test set is held out whole; both of its classes",
             "come from one chain. AUC with a group-bootstrap interval and a test-label permutation p; **vs A** is the paired",
             "difference against the single-chain model on identical test windows. A variant helps only if that interval",
             "is above zero.", "",
             "| Test set | Training variant | Sources (windows) | AUC [95% CI] | p | vs A [95% CI] |",
             "| --- | --- | --- | --- | --- | --- |"]
    for (test, name), r in results.items():
        d = r.get("vs_A")
        sources = ", ".join(f"{s} ({n})" for s, n in r["training_windows"].items())
        lines.append(f"| {test} | {name} | {sources} | {r['roc_auc']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}] | "
                     f"{r['permutation']['p_value']:.3f} | " + (f"{d['difference']:+.2f} [{d['ci95'][0]:+.2f}, "
                     f"{d['ci95'][1]:+.2f}]" if d else "—") + " |")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done in {(time.time() - start) / 60:.1f} min; report {REPORT}")


if __name__ == "__main__":
    main()
