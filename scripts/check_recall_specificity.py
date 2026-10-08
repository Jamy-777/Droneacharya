"""Is the CNN's recall on unseen drones real, or does it flag anything from an unfamiliar receiver?

Same model recipe and threshold as scripts/run_merged_one_class.py (CNN trained on all CardRF 14 MHz tiles,
threshold rejecting 90% of CardRF's training negatives). Applied to NON-drone recordings from other receivers:
  Noisy RF noise vectors (B210, building Wi-Fi/BT/Gaussian), UAVSig no-transmitter captures (B205mini),
  DRFF-R2 environment files (its own receiver, 5.745 and 2.437 GHz), and CardRF negatives scored out of fold.
The recall is meaningful only if these are flagged far less often than the drones were (83–95%).

Output: DroneacharyaData/results/merged_one_class/recall_specificity.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya import cnn, rf_common as R, splits  # noqa: E402
from droneacharya.index import read  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

STAGE2 = DATA_ROOT / "results" / "rf_stage2"
OUT = DATA_ROOT / "results" / "merged_one_class"


def main():
    c_ids = sorted(c["capture_id"] for c in splits.captures_of("cardrf"))
    meta = {c["capture_id"]: c for c in splits.captures_of("cardrf")}
    c_y = np.array([meta[i]["label_uas_present"] for i in c_ids])
    c_tiles = np.load(sorted(STAGE2.glob("cardrf-*.tiles.npy"))[-1])
    c_spec = cnn.spectrograms(c_tiles)
    net = cnn.train(c_spec, c_y)
    threshold = float(np.quantile(cnn.predict(net, c_spec)[~c_y], 0.90))

    flagged = {}
    # CardRF negatives out of fold (honest version of the "90% rejected")
    assignment = splits.load("cardrf/system-kfold-s0")
    parts = np.array([assignment[i] for i in c_ids])
    oof = np.zeros(len(c_y))
    for p in np.unique(parts):
        test = parts == p
        oof[test] = cnn.predict(cnn.train(c_spec[np.flatnonzero(~test)], c_y[~test]), c_spec[np.flatnonzero(test)])
    oof_threshold = [float(np.quantile(oof[(parts != p) & ~c_y], 0.90)) for p in np.unique(parts)]
    flagged["CardRF negatives, out of fold (fold-wise threshold)"] = float(np.mean(
        np.concatenate([oof[(parts == p) & ~c_y] > t for p, t in zip(np.unique(parts), oof_threshold)])))
    flagged["CardRF positives, out of fold (fold-wise threshold)"] = float(np.mean(
        np.concatenate([oof[(parts == p) & c_y] > t for p, t in zip(np.unique(parts), oof_threshold)])))

    # Noisy RF noise vectors
    n_ids = sorted(c["capture_id"] for c in splits.captures_of("noisy_rf"))
    n_meta = {c["capture_id"]: c for c in splits.captures_of("noisy_rf")}
    noise = np.array([not n_meta[i]["label_uas_present"] for i in n_ids])
    n_tiles = np.load(sorted(STAGE2.glob("noisy_rf-*.tiles.npy"))[-1], mmap_mode="r")
    idx = np.flatnonzero(noise)
    n_scores = np.concatenate([cnn.predict(net, cnn.spectrograms(np.asarray(n_tiles[idx[i:i + 8192]])))
                               for i in range(0, len(idx), 8192)])
    flagged["Noisy RF noise vectors (B210)"] = float(np.mean(n_scores > threshold))

    # UAVSig no-transmitter captures
    no_tx = [c["capture_id"] for c in splits.captures_of("uavsig") if not c["label_uas_present"]]
    u_tiles = np.concatenate([R.uavsig_steered(c, n_tiles=20, width=R.STAGE2_WIDTH)[0] for c in no_tx])
    flagged["UAVSig no-transmitter (B205mini)"] = float(np.mean(cnn.predict(net, cnn.spectrograms(u_tiles)) > threshold))

    # DRFF-R2 environment files
    for c in read("drff_r2", "captures").to_pylist():
        if "/dataset7-environment/" in c["capture_id"]:
            tiles, fc = R.iq_steered(c["capture_id"], np.linspace(0, 140_000_000 - 25_000, 40).astype(int),
                                     width=R.STAGE2_WIDTH)
            name = f"DRFF-R2 {c['capture_id'].rsplit('/', 1)[-1]} ({fc / 1e9:.3f} GHz)"
            flagged[name] = float(np.mean(cnn.predict(net, cnn.spectrograms(np.stack([t for t, _ in tiles]))) > threshold))

    result = {"threshold": threshold, "threshold_rule": "rejects 90% of CardRF training negatives",
              "share_flagged_as_drone": flagged,
              "drone_recall_for_comparison": json.loads((OUT / "rf_cnn_recall.json").read_text())["recall"]}
    (OUT / "recall_specificity.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    for k, v in flagged.items():
        print(f"{k:55} flagged as drone: {v:.1%}")


if __name__ == "__main__":
    main()
