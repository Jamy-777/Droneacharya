"""DRFF-R2 identification under the SNR audit: is I-D carried by signal strength? (claims.md I-D)

Design committed before running (2026-10-08). Motivation, measured on synthetic signals the same day: the same burst
at 0 vs 30 dB SNR changes rf_features' occupied bandwidth (34 → 2.5 MHz), flatness and every shape quantile (about 1
per 10 dB), while rf_emitters' behaviour features stay the same from 10 to 30 dB. I-D rests on rf_features.

  rows      tiles     the cached Dataset 3 tiles and features (179 captures x 20 steered 40 MHz tiles), as I-D used
            tiles_eq  the same tiles with complex Gaussian noise added so that every tile's estimated SNR equals the
                      10th percentile of all tiles' SNR (tiles already below it are unchanged; their share is reported);
                      features recomputed with rf_features
            emitters  rf_emitters behaviour of the 3 dominant emitters (the RFUAV/RMA vector) in 4 windows of 100 ms per
                      capture, evenly spaced over the 1.4 s
  SNR       per tile: noise = 25th percentile over frequency of the per-bin median cell power (128-point Hann frames)
            / ln 2 / sum(w^2); SNR = (mean power - noise) / noise. Per emitter window: the dominant emitter's median peak
            SNR (rf_emitters).
  tests     I1a, I1b, I2, I3, D and D-rx exactly as scripts/check_drff_bandwidth_and_test_d.py, gbm, on each row set
  baselines SNR only (gbm on the SNR estimate), bandwidth only, and the best single feature (optimistic), paired by unit
  capture   mean class probabilities over a capture's rows give one prediction per capture; plus the within-capture
            correlation (ICC(1)) of the true-class probability
  receiver  day 2, the units recorded by both receivers: gbm predicting the receiver (u1 vs u2), leave one unit out; AUC
            with a permutation p (200, labels permuted between unit x receiver groups) and a unit-bootstrap interval;
            for each row set. Lower = the representation carries less of the receiver.
  curve     the D test for held-out mavicAir2 and mavicAir2s units, with only k in {1, 2, 4, 6} training units of their
            own model (5 seeded draws each); recall of the held-out unit vs k; tiles and emitters
  decisions I-D "survives the SNR audit" if (a) SNR alone does not meet the D rule (interval lower bound above the
            permutation null's 95th percentile with p < 0.05) and (b) tiles_eq still meets it.
            Emitter features "re-establish I-D" if they meet the D rule. Receiver check and curve: reported, not ruled on.

GPU not needed. CPU: emitter extraction on 4 workers (~0.8 GB each; workers import no torch); fits on every core.
Output: DroneacharyaData/results/checks/drff_snr_audit.json
"""
import hashlib
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
sys.path.insert(0, str(ROOT / "scripts"))

import check_drff_bandwidth_and_test_d as D  # noqa: E402
from droneacharya import evaluation as E  # noqa: E402
from droneacharya.cache import key  # noqa: E402
from droneacharya.index import read  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402
from droneacharya.rf_features import rf_tile_features  # noqa: E402

OUT = DATA_ROOT / "results" / "checks"
TILES, WINDOWS, WINDOW_S, LENGTH, WIDTH = 20, 4, 0.1, 140_000_000, 40e6
TARGET_PERCENTILE = 10
CODE_HASH = hashlib.sha256((ROOT / "src" / "droneacharya" / "rf_emitters.py").read_bytes()
                           + (ROOT / "scripts" / "check_rfuav_rma_identification.py").read_bytes()).hexdigest()[:12]


def tile_snr(t, nfft=128):
    x = t.astype(np.complex128)
    w = np.hanning(nfft)
    cells = np.abs(np.fft.fft(x[: len(x) // nfft * nfft].reshape(-1, nfft) * w, axis=1)) ** 2
    noise = np.percentile(np.median(cells, axis=0), 25) / np.log(2) / np.sum(w ** 2)
    total = np.mean(np.abs(x) ** 2)
    return max(total - noise, 1e-12 * total) / noise, noise, total


def equalise(t, target, rng):
    snr, noise, total = tile_snr(t)
    if snr <= target:
        return t.astype(np.complex128), False
    add = (total - noise) / target - noise
    return t + np.sqrt(add / 2) * (rng.normal(size=len(t)) + 1j * rng.normal(size=len(t))), True


def _emitter_rows(capture_id):
    with threadpool_limits(1):
        sys.path.insert(0, str(ROOT / "scripts"))
        from check_rfuav_rma_identification import emitter_vector
        from droneacharya import rf_emitters as RE
        from droneacharya import signal
        fs = signal.read(capture_id, count=1).sample_rate_hz
        n = int(WINDOW_S * fs)
        rows = []
        for s in np.linspace(0, LENGTH - n, WINDOWS).astype(int):
            found = RE.analyse(signal.read(capture_id, start=int(s), count=n).samples[0], fs)[3]
            rows.append({**emitter_vector(found), "snr_db": float(found[0]["snr_db"]) if found else float("nan")})
        return rows


def captures_and_meta():
    fields = defaultdict(dict)
    for r in read("drff_r2", "evidence").to_pylist():
        if r["field"].startswith("mat."):
            fields[r["entity_id"]][r["field"][4:]] = r["value"]
    captures = sorted((c for c in read("drff_r2", "captures").to_pylist() if "/dataset3-" in c["capture_id"]),
                      key=lambda c: c["capture_id"])
    meta = [{"capture": c["capture_id"], "unit": c["label_class"], "model": c["label_class"].rsplit("_", 1)[0],
             "day": fields[c["capture_id"]]["D"], "rx": fields[c["capture_id"]]["U"]} for c in captures]
    return captures, meta


def expand(meta, per):
    return {k: np.array([m[k] for m in meta for _ in range(per)]) for k in meta[0]}


def load():
    captures, meta = captures_and_meta()
    tiles = np.load(sorted((DATA_ROOT / "results" / "drff_identification").glob("ds3-*.tiles.npy"))[-1])
    X_tiles, names = D.load()[:2]
    snr = np.array([tile_snr(t)[0] for t in tiles])
    target = np.percentile(snr, TARGET_PERCENTILE)
    rng = np.random.default_rng(0)
    eq, changed = zip(*(equalise(t, target, rng) for t in tiles))
    X_eq = np.array([[f[k] for k in names] for f in (rf_tile_features(t, WIDTH) for t in eq)])
    snr_eq = np.array([tile_snr(t)[0] for t in eq])
    tag = key(_emitter_rows, {"windows": WINDOWS, "window_s": WINDOW_S, "code": CODE_HASH}, ["drff_r2"])
    path = OUT / f"drff_emitters-{tag}.json"
    if not path.exists():
        out = Parallel(n_jobs=4)(delayed(_emitter_rows)(c["capture_id"]) for c in captures)   # ~0.8 GB per worker
        path.write_text(json.dumps([r for rows in out for r in rows]), encoding="utf-8")
    em = json.loads(path.read_text(encoding="utf-8"))
    em_names = sorted(k for k in em[0] if k != "snr_db")
    return {
        "tiles": {"X": X_tiles, "names": names, "snr": 10 * np.log10(snr), **expand(meta, TILES)},
        "tiles_eq": {"X": X_eq, "names": names, "snr": 10 * np.log10(snr_eq), **expand(meta, TILES)},
        "emitters": {"X": np.array([[r[k] for k in em_names] for r in em], float), "names": em_names,
                     "snr": np.array([r["snr_db"] for r in em], float), **expand(meta, WINDOWS)},
    }, {"target_snr_db": float(10 * np.log10(target)), "tiles_changed": float(np.mean(changed)),
        "tile_snr_db_percentiles": {p: float(10 * np.log10(np.percentile(snr, p))) for p in (5, 10, 50, 90)}}


def _fit_proba(X_train, y_train, X_test):
    with threadpool_limits(1):
        fitted = E.model("gbm").fit(X_train, y_train)
        return list(fitted.classes_), fitted.predict_proba(X_test)


def run(X, y, folds):
    """Pooled held-out class probabilities and predictions over (train, test) folds."""
    out = Parallel(n_jobs=-1)(delayed(_fit_proba)(X[tr], y[tr], X[te]) for tr, te in folds)
    rows = np.concatenate([np.flatnonzero(te) for _, te in folds])
    classes = sorted(set(y))
    P = np.zeros((len(rows), len(classes)))
    at = 0
    for (fold_classes, proba), (_, te) in zip(out, folds):
        n = int(te.sum())
        for j, c in enumerate(fold_classes):
            P[at:at + n, classes.index(c)] = proba[:, j]
        at += n
    return rows, P, np.array(classes, object)[P.argmax(axis=1)], classes


def icc(values, groups):
    ids = np.unique(groups)
    parts = [values[groups == g] for g in ids]
    k = np.mean([len(p) for p in parts])
    grand = values.mean()
    msb = sum(len(p) * (p.mean() - grand) ** 2 for p in parts) / (len(parts) - 1)
    msw = sum(((p - p.mean()) ** 2).sum() for p in parts) / (len(values) - len(parts))
    return float((msb - msw) / (msb + (k - 1) * msw))


def identification(r):
    result = {}
    for name, folds in D.splits_of(r["unit"], r["model"], r["day"], r["rx"]).items():
        y_all = r["model"]
        rows, P, full, classes = run(r["X"], y_all, folds)
        y, g, cap = y_all[rows], r["unit"][rows], r["capture"][rows]
        snr_pred = run(r["snr"][:, None], y_all, folds)[2]
        bw = [j for j, n in enumerate(r["names"]) if n.endswith(("occupied_bw_mhz", "_log_bw"))]
        bw_pred = run(r["X"][:, bw], y_all, folds)[2]
        singles = {n: run(r["X"][:, [j]], y_all, folds)[2] for j, n in enumerate(r["names"])}
        best = max(singles, key=lambda n: E.metric(y, singles[n]))
        res = E.evaluate_external_labels(full, y, g)
        caps = np.unique(cap)
        cap_pred = np.array([np.array(classes, object)[P[cap == c].mean(axis=0).argmax()] for c in caps], object)
        cap_y = np.array([y[cap == c][0] for c in caps], object)
        cap_g = np.array([g[cap == c][0] for c in caps])
        snr_res = E.evaluate_external_labels(snr_pred, y, g)
        res.update({
            "snr_only": snr_res["balanced_accuracy"], "snr_only_null95": snr_res["permutation"]["null_95th"],
            "snr_only_p": snr_res["permutation"]["p_value"], "snr_only_ci95": snr_res["ci95"],
            "full_minus_snr": E.paired_label_difference(full, snr_pred, y, g),
            "bandwidth_only": E.metric(y, bw_pred), "full_minus_bandwidth": E.paired_label_difference(full, bw_pred, y, g),
            "best_single_feature": {"feature": best, "balanced_accuracy": E.metric(y, singles[best])},
            "per_capture": E.evaluate_external_labels(cap_pred, cap_y, cap_g),
            "icc_true_class_probability": icc(P[np.arange(len(y)), [classes.index(c) for c in y]], cap)})
        res["meets_D_rule"] = bool(res["ci95"][0] > res["permutation"]["null_95th"] and res["permutation"]["p_value"] < 0.05)
        res["snr_only_meets_D_rule"] = bool(snr_res["ci95"][0] > snr_res["permutation"]["null_95th"]
                                            and snr_res["permutation"]["p_value"] < 0.05)
        result[name] = res
    return result


def receiver_check(r):
    d2 = r["day"] == "d2"
    both = sorted(set(r["unit"][d2 & (r["rx"] == "u1")]) & set(r["unit"][d2 & (r["rx"] == "u2")]))
    sel = d2 & np.isin(r["unit"], both)
    y = r["rx"][sel] == "u1"
    res = E.evaluate(r["X"][sel], y, r["unit"][sel], np.char.add(r["unit"][sel].astype(str), r["rx"][sel].astype(str)),
                     "gbm", n_permutations=200)
    res["units"] = both
    return res


def _fit_predict(X_train, y_train, X_test):
    with threadpool_limits(1):
        return E.model("gbm").fit(X_train, y_train).predict(X_test)


def curve(r):
    d1, later = r["day"] == "d1", (r["day"] != "d1") & (r["rx"] == "u2")
    out = {}
    for m in ("mavicAir2", "mavicAir2s"):
        units = sorted(set(r["unit"][d1 & (r["model"] == m)]))
        held = [u for u in units if (later & (r["unit"] == u)).any()]
        for k in (1, 2, 4, 6):
            jobs = []
            for i, u in enumerate(held):
                others = [v for v in units if v != u]
                if k > len(others):
                    continue
                for draw in range(5):
                    chosen = np.random.default_rng([k, draw, i]).choice(others, k, replace=False)
                    jobs.append((d1 & ((r["model"] != m) | np.isin(r["unit"], chosen)), later & (r["unit"] == u)))
            preds = Parallel(n_jobs=-1)(delayed(_fit_predict)(r["X"][tr], r["model"][tr], r["X"][te]) for tr, te in jobs)
            out[f"{m} k={k}"] = {"recall": float(np.mean([np.mean(p == m) for p in preds])), "draws": len(jobs)}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    data, snr_info = load()
    print(f"  rows ready ({(time.time() - start) / 60:.1f} min); SNR equalisation {snr_info}", flush=True)
    if "--features-only" in sys.argv:          # extraction fits no model, so it may run before the design commit lands
        return
    result = {"snr_equalisation": snr_info}
    for rep, r in data.items():
        res = identification(r)
        rc = receiver_check(r)
        result[rep] = {"tests": res, "receiver_auc": {k: rc[k] for k in ("roc_auc", "ci95", "permutation", "units")}}
        if rep in ("tiles", "emitters"):
            result[rep]["learning_curve"] = curve(r)
        for name, t in res.items():
            print(f"[{rep}] {name:42} {t['balanced_accuracy']:.3f} {[round(v, 2) for v in t['ci95']]} null95 "
                  f"{t['permutation']['null_95th']:.3f} p={t['permutation']['p_value']:.4f} meets={t['meets_D_rule']} | "
                  f"SNR-only {t['snr_only']:.3f} (meets {t['snr_only_meets_D_rule']}) | bw {t['bandwidth_only']:.3f} | "
                  f"per capture {t['per_capture']['balanced_accuracy']:.3f} | ICC {t['icc_true_class_probability']:.2f}", flush=True)
        print(f"[{rep}] receiver AUC {rc['roc_auc']:.3f} {[round(v, 2) for v in rc['ci95']]} p={rc['permutation']['p_value']:.3f}"
              + (f" | curve {result[rep]['learning_curve']}" if "learning_curve" in result[rep] else ""), flush=True)
    d_tiles, d_eq, d_em = (result[k]["tests"]["D unseen unit + unseen day (u2)"] for k in ("tiles", "tiles_eq", "emitters"))
    result["decisions"] = {"I-D_survives_SNR_audit": bool(not d_tiles["snr_only_meets_D_rule"] and d_eq["meets_D_rule"]),
                           "emitters_reestablish_I-D": bool(d_em["meets_D_rule"])}
    result["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (OUT / "drff_snr_audit.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print("decisions:", result["decisions"], f"| {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
