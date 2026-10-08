"""Noisy RF: does the separation at −20 dB come from the drone, or from how the two classes' backgrounds were built?

Design committed before running (2026-10-08). Audit point 2; docs/results/claims.md R2.

  facts       Noisy RF drone vectors are a chamber recording of an RC link, unit-power normalised, mixed with either
              building noise (Labnoise, 50%) or Gaussian noise (50%) at a synthetic SNR of −20…30 dB. The noise class
              is Labnoise + Gaussian mixtures (dataset card: AUTHOR_DOC / REPO_METADATA). The classes can therefore
              differ in their backgrounds before any drone is added.
  observed    CardRF-trained CNN AUC 0.60 at −20 dB, 0.54 at −4 dB, 0.74 at ≥ 20 dB; gbm 0.56 at −20 dB
              (docs/results/rf_stage2.md)
  scores      the stage-2 CNN and gbm, retrained with the identical recipe (cnn.train, seed 0; evaluation gbm) on the
              cached stage-2 tiles and features: the rows the reported numbers came from. Reproduction is reported
              per SNR against rf_stage2.
  background  four features of each tile's quiet half: spectrogram cells (128-point Hann frames, as rf_features; the
              3 bins around DC excluded) at or below the tile's median cell power
                bg_cell_cv           spread of quiet-cell power (std / mean)
                bg_freq_cv           spectral colour: spread across frequency bins of their mean quiet-cell power
                bg_time_cv           non-stationarity: spread across frames of their mean quiet-cell power
                bg_quiet_bin_spread  persistent narrowband occupancy: std across bins of each bin's quiet share
              A drone that fills under half the cells cannot enter them; they are read at −20 dB.
  model       gbm on the four background features, trained and scored inside Noisy RF by stratified 5-fold (strata =
              class × SNR, seed 0), the dataset authors' protocol, since there are no recording IDs. It shows what
              the backgrounds alone allow, not what the CNN used.
  reported    AUC per SNR (positives at that SNR vs all noise vectors) for the CNN, the gbm and the background-only
              model; Spearman correlation of the CNN score with the background-only score within −20 dB positives
              and within noise vectors (descriptive). Point estimates: ~1,800 positives per SNR against 52,552 noise
              vectors give row-level standard errors near 0.01; rows are not independent recordings.
  decision    background-only AUC at −20 dB ≥ 0.60 (the CNN's): the floor is explainable by background construction,
              and Noisy RF is retired as a detection test set for models not trained on it.
              < 0.55: not explained by these background features; the cause stays open. Between: inconclusive.
              Either way R2 stays NOT ESTABLISHED; no transfer claim is made from this check.

GPU: spectrograms and the CNN. CPU: background features vectorised in chunks, gbm.
Output: DroneacharyaData/results/checks/noisy_rf_floor.json
"""
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya import cnn  # noqa: E402
from droneacharya import evaluation as E  # noqa: E402
from droneacharya import splits  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

STAGE2 = DATA_ROOT / "results" / "rf_stage2"
OUT = DATA_ROOT / "results" / "checks"
NFFT, DC_BINS, CHUNK = 128, [63, 64, 65], 8192


def background_features(tiles):
    """Quiet-half statistics of complex tiles (n, samples); scale-free, so tile power cannot enter."""
    frames = tiles[:, : tiles.shape[1] // NFFT * NFFT].reshape(len(tiles), -1, NFFT)
    spec = np.abs(np.fft.fftshift(np.fft.fft(frames * np.hanning(NFFT).astype(np.float32), axis=2), axes=2)) ** 2
    spec = np.delete(spec, DC_BINS, axis=2)
    quiet = spec <= np.median(spec, axis=(1, 2))[:, None, None]
    q = np.where(quiet, spec, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # bins or frames with no quiet cell give NaN
        per_bin, per_frame = np.nanmean(q, axis=1), np.nanmean(q, axis=2)
        return np.column_stack([
            np.nanstd(q, axis=(1, 2)) / np.nanmean(q, axis=(1, 2)),
            np.nanstd(per_bin, axis=1) / np.nanmean(per_bin, axis=1),
            np.nanstd(per_frame, axis=1) / np.nanmean(per_frame, axis=1),
            quiet.mean(axis=1).std(axis=1)])


BACKGROUND = ["bg_cell_cv", "bg_freq_cv", "bg_time_cv", "bg_quiet_bin_spread"]


def by_snr(scores, y, snr):
    return {int(s): E.metric(np.r_[np.ones(np.sum(y & (snr == s)), bool), np.zeros(np.sum(~y), bool)],
                             np.r_[scores[y & (snr == s)], scores[~y]]) for s in sorted(set(snr[y]))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    c_meta = {c["capture_id"]: c for c in splits.captures_of("cardrf")}
    n_meta = {c["capture_id"]: c for c in splits.captures_of("noisy_rf")}
    c_ids, n_ids = sorted(c_meta), sorted(n_meta)
    c_y = np.array([c_meta[i]["label_uas_present"] for i in c_ids])
    y = np.array([n_meta[i]["label_uas_present"] for i in n_ids])
    snr = np.array([n_meta[i]["reference_snr_db"] for i in n_ids])
    cls = np.array([n_meta[i]["label_class"] for i in n_ids])
    c_tiles = np.load(sorted(STAGE2.glob("cardrf-*.tiles.npy"))[-1])
    n_tiles = np.load(sorted(STAGE2.glob("noisy_rf-*.tiles.npy"))[-1], mmap_mode="r")
    c_feats = json.loads(sorted(STAGE2.glob("cardrf-*.features.json"))[-1].read_text(encoding="utf-8"))
    n_feats = json.loads(sorted(STAGE2.glob("noisy_rf-*.features.json"))[-1].read_text(encoding="utf-8"))
    names = sorted(c_feats[0])
    assert len(c_tiles) == len(c_ids) and len(n_tiles) == len(n_ids)

    gbm = E.model("gbm").fit(np.array([[f[k] for k in names] for f in c_feats]), c_y)
    gbm_scores = E.positive_scores(gbm, np.array([[f[k] for k in names] for f in n_feats]))
    del c_feats, n_feats
    net = cnn.train(cnn.spectrograms(c_tiles), c_y)
    cnn_scores = np.concatenate([cnn.predict(net, cnn.spectrograms(np.asarray(n_tiles[i:i + CHUNK])))
                                 for i in range(0, len(n_tiles), CHUNK)])
    print(f"  scores ready ({(time.time() - start) / 60:.1f} min)", flush=True)

    bg = np.concatenate([background_features(np.asarray(n_tiles[i:i + CHUNK])) for i in range(0, len(n_tiles), CHUNK)])
    strata = np.array([f"{c}|{s}" for c, s in zip(np.where(y, "drone", "noise"), snr)])
    bg_scores = np.zeros(len(y))
    for train, test in StratifiedKFold(5, shuffle=True, random_state=0).split(bg, strata):
        bg_scores[test] = E.positive_scores(E.model("gbm").fit(bg[train], y[train]), bg[test])

    saved = {m: json.loads((STAGE2 / f"external_{m}.json").read_text(encoding="utf-8"))["by_snr"] for m in ("cnn", "gbm")}
    low = y & (snr == min(snr[y]))
    result = {
        "auc_by_snr": {"cnn": by_snr(cnn_scores, y, snr), "gbm": by_snr(gbm_scores, y, snr),
                       "background_only": by_snr(bg_scores, y, snr)},
        "auc_all": {"cnn": E.metric(y, cnn_scores), "gbm": E.metric(y, gbm_scores), "background_only": E.metric(y, bg_scores)},
        "reproduction_max_abs_diff": {m: max(abs(by_snr(s, y, snr)[int(k)] - v) for k, v in saved[m].items())
                                      for m, s in (("cnn", cnn_scores), ("gbm", gbm_scores))},
        "spearman_cnn_vs_background": {
            "positives_at_lowest_snr": float(spearmanr(cnn_scores[low], bg_scores[low])[0]),
            "noise_vectors": float(spearmanr(cnn_scores[~y], bg_scores[~y])[0])},
        "background_feature_medians": {
            name: {"noise": float(np.nanmedian(bg[~y, j])), "drone_lowest_snr": float(np.nanmedian(bg[low, j])),
                   "drone_highest_snr": float(np.nanmedian(bg[y & (snr == max(snr[y])), j]))}
            for j, name in enumerate(BACKGROUND)},
        "counts": {"noise": int((~y).sum()), "drone_per_snr": int(low.sum()), "rc_links": sorted(set(cls[y]))},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    a = result["auc_by_snr"]["background_only"][int(min(snr[y]))]
    result["decision"] = ("background construction explains the floor" if a >= 0.60 else
                          "not explained by these background features" if a < 0.55 else "inconclusive")
    (OUT / "noisy_rf_floor.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    for m, curve in result["auc_by_snr"].items():
        print(f"{m:16}", " ".join(f"{v:.2f}" for v in curve.values()))
    print("reproduction:", result["reproduction_max_abs_diff"], "| decision:", result["decision"],
          f"| {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
