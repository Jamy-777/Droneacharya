"""RF stage 2: is the transmitter present a UAS, or Wi-Fi/Bluetooth?  CardRF -> Noisy RF.

Design fixed before any result was seen (2026-10-08):

  question     stage 2 of two-stage RF detection: given activity, is the emitter a UAS link?
  tiles        14 MHz complex baseband, 14 MS/s, 250 us (3,500 samples). CardRF: steered (rf_common, same rule
               for every capture). Noisy RF: first 250 us of each vector, DC notched. Noisy RF's whole band is
               14 MHz, so this is the widest common tile.
  train        all 9,600 CardRF captures (UAS aircraft + controllers vs Wi-Fi + Bluetooth)
  external     all 98,705 Noisy RF vectors (6 RC uplinks vs building Labnoise/Gaussian mixtures), never used before
  models       primary: gbm on position-invariant rf_features; secondary: TileNet CNN on the GPU (cnn.py recipe)
  reported     AUC on all of Noisy RF; AUC per synthetic SNR level (positives at that SNR vs all noise vectors);
               AUC per RC link; within-CardRF reference on cardrf/system-kfold-s0 (gbm with group permutation;
               CNN pooled out-of-fold with group bootstrap, no permutation: 1,000 CNN trainings is not feasible)
  limits       Noisy RF has no recording IDs. Its interval resamples the 6 RC transmitters with the noise vectors
               as one block, so it covers transmitter variability only; no permutation p is meaningful for it.
  decision     stage 2 "transfers" if the external AUC interval is above 0.5 for the primary model

Parallel CPU for tile extraction and features; GPU for spectrograms and the CNN.
Outputs: DroneacharyaData/results/rf_stage2/*.json, docs/results/rf_stage2.md.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya import cnn  # noqa: E402
from droneacharya import evaluation as E  # noqa: E402
from droneacharya import splits  # noqa: E402
from droneacharya.cache import key  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

OUT = DATA_ROOT / "results" / "rf_stage2"
REPORT = Path(__file__).resolve().parents[1] / "docs" / "results" / "rf_stage2.md"
FS = 14e6


def _tiles_and_features(dataset, capture_ids):
    with threadpool_limits(1):
        from droneacharya import rf_common as R
        from droneacharya.rf_features import rf_tile_features
        tiles = [R.cardrf_steered(c, width=R.STAGE2_WIDTH)[0] if dataset == "cardrf" else R.noisy_rf_tile(c)
                 for c in capture_ids]
        return np.stack(tiles), [rf_tile_features(t, FS) for t in tiles]


def load(dataset):
    captures = sorted(c["capture_id"] for c in splits.captures_of(dataset))
    tag = key(_tiles_and_features, {"format": "stage2 14 MHz"}, [dataset])
    tiles_path, feats_path = OUT / f"{dataset}-{tag}.tiles.npy", OUT / f"{dataset}-{tag}.features.json"
    if not tiles_path.exists():
        chunks = [captures[i:i + 500] for i in range(0, len(captures), 500)]
        parts = Parallel(n_jobs=-1)(delayed(_tiles_and_features)(dataset, chunk) for chunk in chunks)
        np.save(tiles_path, np.concatenate([p[0] for p in parts]))
        feats_path.write_text(json.dumps([f for p in parts for f in p[1]]), encoding="utf-8")
    tiles = np.load(tiles_path)
    feats = json.loads(feats_path.read_text(encoding="utf-8"))
    names = sorted(feats[0])
    print(f"  {dataset}: {len(tiles)} tiles", flush=True)
    return captures, tiles, np.array([[f[k] for k in names] for f in feats]), names


def save(name, result):
    result["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{name}: AUC {result['roc_auc']:.3f} CI {[round(v, 3) for v in result['ci95']]}", flush=True)
    return result


def external_report(scores, meta, model_name):
    y, cls, snr = meta["y"], meta["cls"], meta["snr"]
    groups = np.where(y, cls, "noise")                       # 6 transmitters + noise as one block
    result = E.evaluate_external(scores, y, groups, model_name, n_permutations=1)
    result.pop("permutation")
    result["interval_note"] = "resamples the 6 RC transmitters (noise vectors one block): transmitter variability only"
    result["by_snr"] = {int(s): E.metric(np.r_[np.ones(np.sum(y & (snr == s)), bool), np.zeros(np.sum(~y), bool)],
                                         np.r_[scores[y & (snr == s)], scores[~y]]) for s in sorted(set(snr[y]))}
    result["by_transmitter"] = {c: E.metric(np.r_[np.ones(np.sum(cls == c), bool), np.zeros(np.sum(~y), bool)],
                                            np.r_[scores[cls == c], scores[~y]]) for c in sorted(set(cls[y]))}
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    c_ids, c_tiles, c_X, names = load("cardrf")
    n_ids, n_tiles, n_X, _ = load("noisy_rf")
    n_tiles = np.load(next(OUT.glob("noisy_rf-*.tiles.npy")), mmap_mode="r")   # memory-mapped, read in chunks
    cardrf = {c["capture_id"]: c for c in splits.captures_of("cardrf")}
    noisy = {c["capture_id"]: c for c in splits.captures_of("noisy_rf")}
    c_y = np.array([cardrf[i]["label_uas_present"] for i in c_ids])
    c_groups = np.array([cardrf[i]["group_id"] for i in c_ids])
    meta = {"y": np.array([noisy[i]["label_uas_present"] for i in n_ids]),
            "cls": np.array([noisy[i]["label_class"] for i in n_ids]),
            "snr": np.array([noisy[i]["reference_snr_db"] for i in n_ids])}
    results = {}

    gbm = E.model("gbm").fit(c_X, c_y)
    results["external gbm"] = save("external_gbm", {
        **external_report(E.positive_scores(gbm, n_X), meta, "gbm"), "task": "CardRF -> Noisy RF (primary)"})

    # Spectrograms stream through the GPU in chunks: holding all 98,705 at once (2.8 GB) beside the tiles
    # exhausted the 16.8 GB of RAM in the first run.
    c_spec = cnn.spectrograms(c_tiles)
    net = cnn.train(c_spec, c_y)
    n_scores = np.concatenate([cnn.predict(net, cnn.spectrograms(n_tiles[i:i + 8192]))
                               for i in range(0, len(n_tiles), 8192)])
    del n_tiles
    results["external cnn"] = save("external_cnn", {
        **external_report(n_scores, meta, "cnn"), "task": "CardRF -> Noisy RF (secondary, GPU CNN)"})

    assignment = splits.load("cardrf/system-kfold-s0")
    parts = np.array([assignment[i] for i in c_ids])
    within_gbm, gbm_scores = E.evaluate(c_X, c_y, parts, c_groups, "gbm", n_permutations=200, return_scores=True)
    results["within CardRF gbm"] = save("within_cardrf_gbm", {**within_gbm, "task": "within CardRF, 14 MHz tiles (reference)"})
    cnn_scores = np.zeros(len(c_y))
    for p in np.unique(parts):
        test = parts == p
        cnn_scores[test] = cnn.predict(cnn.train(c_spec[np.flatnonzero(~test)], c_y[~test]), c_spec[np.flatnonzero(test)])
    _, _, members = E._groups(c_groups, c_y)
    results["within CardRF cnn"] = save("within_cardrf_cnn", {
        "model": "cnn", "metric": "roc_auc", "roc_auc": E.metric(c_y, cnn_scores), "n": int(len(c_y)),
        "ci95": E._bootstrap(c_y, cnn_scores, members, np.random.default_rng(0), 2000),
        "vs_gbm": E.paired_difference(cnn_scores, gbm_scores, c_y, c_groups),
        "task": "within CardRF, 14 MHz tiles, pooled out-of-fold (reference; no permutation)"})

    lines = ["<!-- Generated by scripts/run_rf_stage2.py; results in DroneacharyaData/results/rf_stage2/. -->", "",
             "> Raw output, rewritten on every run. Which numbers may be quoted, and with what limits: [claims.md](claims.md).", "",
             "# RF stage 2: UAS vs Wi-Fi/Bluetooth, CardRF → Noisy RF", "",
             "Design fixed before results (script docstring). External intervals resample the 6 RC transmitters only.", "",
             "| Test | AUC [95% CI] |", "| --- | --- |"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['roc_auc']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}] |")
    for model in ("external gbm", "external cnn"):
        r = results[model]
        lines += ["", f"**{model}, by synthetic SNR** (positives at that SNR vs all noise vectors):", "",
                  "| SNR dB | " + " | ".join(str(s) for s in r["by_snr"]) + " |",
                  "| --- | " + " | ".join("---" for _ in r["by_snr"]) + " |",
                  "| AUC | " + " | ".join(f"{v:.2f}" for v in r["by_snr"].values()) + " |", "",
                  "By RC link: " + ", ".join(f"{c} {v:.2f}" for c, v in r["by_transmitter"].items())]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done in {(time.time() - start) / 60:.1f} min; report {REPORT}")


if __name__ == "__main__":
    main()
