"""RF identification within RFUAV and RMA: the per-dataset baseline, as honestly as each dataset allows.

Design committed before running (2026-10-08). Completes "identification replicated across the RF datasets"
(claims.md I; DRFF-R2 is done).

  RFUAV      37 classes in 53 packs (recording sessions); 12 classes have ≥ 2 packs (28 packs). Each of those 28
             packs is held out in turn, and the model trains on every other pack of all 37 classes. This is the only
             unseen-session test RFUAV allows.
  RMA        10 systems with one archive (= one session, 1–22 min) per system and link type, so for 9 systems an
             unseen-session test is impossible.
               RMA-a  within session: stratified 5-fold over files (seed 0). Every test file has same-session files
                      in training, and session = class, so this is an upper bound, never a generalisation claim. It is
                      the protocol most published results use.
               RMA-b  unseen session, Mavic only: each of the 4 Mavic archives (RC1, RC2, Vid1, Vid2) is held out in
                      turn; the share of its rows predicted "Mavic". One system, so descriptive only.
  tiles      primary: steered 40 MHz tiles (rf_common.iq_steered, as the DRFF-R2 identification), 4 consecutive
             250 us tiles at the start of every 1 s RFUAV chunk and of every RMA file; rf_features
  emitters   secondary: the behaviour of the 3 dominant emitters (rf_emitters.analyse on the first 100 ms of each
             RFUAV chunk, at most 3 chunks per capture, and of each RMA file): bandwidth, burst length, hop set, span,
             interval, interval CV, duty, continuity, plus the number of emitters with ≥ 5 bursts
  model      gbm (evaluation.model), fixed in advance, for both representations
  metric     balanced accuracy over the truth classes present; p by permuting class labels between held-out packs
             (2,000); interval resampling packs
  baselines  on identical rows: acquisition metadata only (centre frequency, reference SNR, ScaleFactor; linear, as
             the shortcut baseline that scored 0.53), occupied bandwidth only (gbm), and the best single feature
             (picked on the test rows: optimistic for the baseline)
  decisions  RFUAV identification "works" if its interval's lower bound is above the permutation null's 95th
             percentile and p < 0.05. It is "beyond metadata" / "beyond bandwidth" if the paired difference
             (intervals resampling packs) is above zero. RMA-a and RMA-b are reported, not ruled on.

CPU: tiles and emitters in parallel (8 workers; workers import no torch).
Output: DroneacharyaData/results/checks/rfuav_rma_identification.json
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya import evaluation as E  # noqa: E402
from droneacharya.cache import key  # noqa: E402
from droneacharya.index import read  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

OUT = DATA_ROOT / "results" / "checks"
WIDTH, TILES_PER_START, EMITTER_S, MAX_CHUNKS, CHUNK = 40e6, 4, 0.1, 3, 100_000_000
WORKERS = 8
EMITTER_KEYS = ("log_bw", "log_burst", "hop_set", "span_mhz", "log_interval", "interval_cv", "duty", "continuous")


def emitter_vector(found):
    v = {"n_emitters_5": float(sum(e["blobs"] >= 5 for e in found))}
    for r in range(3):
        e = found[r] if r < len(found) else None
        values = (np.log10(e["bandwidth_hz"]), np.log10(e["burst_s"]), e["hop_set"], e["span_hz"] / 1e6,
                  np.log10(e["interval_s"]) if e["interval_s"] > 0 else np.nan, e["interval_cv"], e["duty"],
                  float(e["continuous"])) if e else (np.nan,) * len(EMITTER_KEYS)
        v.update({f"e{r}_{k}": float(x) for k, x in zip(EMITTER_KEYS, values)})
    return v


def _capture_rows(capture_id, chunk_starts):
    with threadpool_limits(1):
        from droneacharya import rf_common as R
        from droneacharya import rf_emitters as RE
        from droneacharya import signal
        from droneacharya.rf_features import rf_tile_features
        fs = signal.read(capture_id, count=1).sample_rate_hz
        block = int(round(250e-6 * fs))
        tiles, _ = R.iq_steered(capture_id, [s + k * block for s in chunk_starts for k in range(TILES_PER_START)], width=WIDTH)
        emitters = []
        for s in chunk_starts[:MAX_CHUNKS]:
            x = signal.read(capture_id, start=s, count=int(EMITTER_S * fs)).samples[0]
            if len(x) == int(EMITTER_S * fs):
                emitters.append(emitter_vector(RE.analyse(x, fs)[3]))
        return [rf_tile_features(t, WIDTH) for t, _ in tiles], emitters


def load():
    scale = {r["entity_id"]: float(r["value"]) for r in read("rfuav", "evidence").to_pylist() if r["field"] == "xml.ScaleFactor"}
    jobs = [("rfuav", c, [p * CHUNK for p in range(c["n_artifacts"])]) for c in read("rfuav", "captures").to_pylist()]
    jobs += [("rma", c, [0]) for c in read("rma", "captures").to_pylist()]
    tag = key(_capture_rows, {"width": WIDTH, "tiles": TILES_PER_START, "emitter_s": EMITTER_S, "max_chunks": MAX_CHUNKS},
              ["rfuav", "rma"])
    path = OUT / f"rfuav_rma_id-{tag}.json"
    if not path.exists():
        out = Parallel(n_jobs=WORKERS)(delayed(_capture_rows)(c["capture_id"], starts) for _, c, starts in jobs)
        rows = {"tiles": [], "emitters": []}
        for (dataset, c, _), (tiles, emitters) in zip(jobs, out):
            meta = {"dataset": dataset, "capture_id": c["capture_id"], "group": c["group_id"], "label": c["label_class"],
                    "fc_ghz": c["center_frequency_hz"] / 1e9, "snr_db": c.get("reference_snr_db"),
                    "scale": scale.get(c["group_id"], np.nan)}
            rows["tiles"] += [{**meta, **f} for f in tiles]
            rows["emitters"] += [{**meta, **f} for f in emitters]
        path.write_text(json.dumps(rows), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def table(rows, dataset):
    rows = [r for r in rows if r["dataset"] == dataset]
    meta = {"dataset", "capture_id", "group", "label", "fc_ghz", "snr_db", "scale"}
    names = sorted(k for k in rows[0] if k not in meta)
    return {"X": np.array([[r[k] for k in names] for r in rows], float), "names": names,
            "y": np.array([r["label"] for r in rows], object), "groups": np.array([r["group"] for r in rows]),
            "meta": np.array([[r["fc_ghz"], r["snr_db"] if r["snr_db"] is not None else np.nan, r["scale"]] for r in rows], float)}


def _fit_predict(X_train, y_train, X_test, kind):
    with threadpool_limits(1):
        return E.model(kind).fit(X_train, y_train).predict(X_test).astype(object)


def held_out(X, y, folds, kind="gbm"):
    """Pooled predictions over (train mask, test mask) folds."""
    predicted = Parallel(n_jobs=-1)(delayed(_fit_predict)(X[tr], y[tr], X[te], kind) for tr, te in folds)
    rows = np.concatenate([np.flatnonzero(te) for _, te in folds])
    return rows, np.concatenate(predicted)


def rfuav_test(t):
    packs_of = {}
    for g, label in zip(t["groups"], t["y"]):
        packs_of.setdefault(label, set()).add(g)
    tested = sorted(g for label, packs in packs_of.items() if len(packs) >= 2 for g in packs)
    folds = [(t["groups"] != g, t["groups"] == g) for g in tested]
    rows, full = held_out(t["X"], t["y"], folds)
    y, g = t["y"][rows], t["groups"][rows]
    meta = np.nan_to_num(t["meta"], nan=0.0)
    _, metadata = held_out(meta, t["y"], folds, "linear")
    bw = [t["names"].index(n) for n in t["names"] if n.endswith(("occupied_bw_mhz", "_log_bw"))]
    _, bandwidth = held_out(t["X"][:, bw], t["y"], folds)
    singles = {n: held_out(t["X"][:, [j]], t["y"], folds)[1] for j, n in enumerate(t["names"])}
    best = max(singles, key=lambda n: E.metric(y, singles[n]))
    r = E.evaluate_external_labels(full, y, g)
    r.update({"packs_held_out": len(tested), "classes_tested": len(set(y)), "chance": 1 / len(set(t["y"])),
              "metadata_only": E.metric(y, metadata), "full_minus_metadata": E.paired_label_difference(full, metadata, y, g),
              "bandwidth_only": E.metric(y, bandwidth), "full_minus_bandwidth": E.paired_label_difference(full, bandwidth, y, g),
              "best_single_feature": {"feature": best, "balanced_accuracy": E.metric(y, singles[best])},
              "full_minus_best_single": E.paired_label_difference(full, singles[best], y, g),
              "per_class_recall": {c: float(np.mean(full[y == c] == c)) for c in sorted(set(y))}})
    r["decisions"] = {"works": bool(r["ci95"][0] > r["permutation"]["null_95th"] and r["permutation"]["p_value"] < 0.05),
                      "beyond_metadata": bool(r["full_minus_metadata"]["ci95"][0] > 0),
                      "beyond_bandwidth": bool(r["full_minus_bandwidth"]["ci95"][0] > 0)}
    return r


def rma_tests(t):
    archive, files = t["groups"], t["capture_ids"]
    unique_files = sorted(set(files))
    file_label = dict(zip(files, t["y"]))
    fold_of = {}
    split = StratifiedKFold(5, shuffle=True, random_state=0).split(unique_files, [file_label[f] for f in unique_files])
    for k, (_, test) in enumerate(split):
        fold_of.update({unique_files[i]: k for i in test})
    fold = np.array([fold_of[f] for f in files])
    rows, within = held_out(t["X"], t["y"], [(fold != k, fold == k) for k in range(5)])
    y = t["y"][rows]
    result = {"RMA-a within session (upper bound)": {
        "balanced_accuracy": E.metric(y, within), "classes": len(set(t["y"])),
        "per_class_recall": {c: float(np.mean(within[y == c] == c)) for c in sorted(set(y))}}}
    mavic = sorted(a for a in set(archive) if a.startswith("rma/Mavic"))
    rows, unseen = held_out(t["X"], t["y"], [(archive != a, archive == a) for a in mavic])
    result["RMA-b unseen Mavic session"] = {a: float(np.mean(unseen[archive[rows] == a] == "Mavic")) for a in mavic}
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    rows = load()
    if "--features-only" in sys.argv:          # extraction fits no model, so it may run before the design commit lands
        print(f"features cached ({(time.time() - start) / 60:.1f} min)")
        return
    print(f"  rows: {len(rows['tiles'])} tiles, {len(rows['emitters'])} emitter windows "
          f"({(time.time() - start) / 60:.1f} min)", flush=True)
    result = {}
    for rep in ("tiles", "emitters"):
        t = table(rows[rep], "rfuav")
        r = result[f"RFUAV unseen pack [{rep}]"] = rfuav_test(t)
        print(f"RFUAV [{rep}] {r['balanced_accuracy']:.3f} {[round(v, 2) for v in r['ci95']]} null95 "
              f"{r['permutation']['null_95th']:.3f} p={r['permutation']['p_value']:.4f} | metadata {r['metadata_only']:.3f} "
              f"(full − meta {r['full_minus_metadata']['difference']:+.3f} {[round(v, 2) for v in r['full_minus_metadata']['ci95']]}) | "
              f"bandwidth {r['bandwidth_only']:.3f} | best single {r['best_single_feature']['feature']} "
              f"{r['best_single_feature']['balanced_accuracy']:.3f} | {r['decisions']}", flush=True)
        m = table(rows[rep], "rma")
        m["capture_ids"] = np.array([r["capture_id"] for r in rows[rep] if r["dataset"] == "rma"])
        r = result[f"RMA [{rep}]"] = rma_tests(m)
        print(f"RMA [{rep}] within session {r['RMA-a within session (upper bound)']['balanced_accuracy']:.3f} | "
              f"unseen Mavic session {r['RMA-b unseen Mavic session']}", flush=True)
    result["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (OUT / "rfuav_rma_identification.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"done in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
