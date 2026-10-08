"""Every main test, re-scored with the full metric set: AUC, accuracy, balanced accuracy, precision, recall,
specificity, F1 (binary) or accuracy, balanced accuracy, macro F1 (identification), training-set scores beside test
scores, and processing time per decision.

Reporting only (2026-10-08): the tests, splits, features and models are the committed ones, refitted identically
(every model here is deterministic). The one new choice, fixed before running: a binary decision is "drone" when the
class-balanced model's probability is >= 0.5. One-class and single-feature scores have no natural threshold, so they
get AUC only. No claim's status depends on these numbers; claims.md stays the record.

Why AUC and balanced accuracy lead in claims.md: accuracy and F1 depend on the threshold and on class balance.
UaVirBASE is 128 drone recordings vs 4 ambient, so "always drone" scores 97% accuracy and 0.98 F1. And thresholds do
not transfer between receivers or recording setups (claims.md W1, A0), so a threshold-free ranking metric is the
honest cross-dataset measure. Both are shown here side by side.

Output: DroneacharyaData/results/checks/all_metrics.json and docs/results/metrics_table.md
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import check_acoustic_gate as AG  # noqa: E402
import check_acoustic_single_mic_mfcc as SM  # noqa: E402
import check_drff_snr_audit as DS  # noqa: E402
import check_rfuav_rma_identification as RR  # noqa: E402
from droneacharya import evaluation as E  # noqa: E402
from droneacharya import splits  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

OUT = DATA_ROOT / "results" / "checks"
REPORT = ROOT / "docs" / "results" / "metrics_table.md"
RF_PREFIX = ("shape_", "occupied", "active", "spectral", "duty", "transitions", "frame_", "centroid", "amplitude", "papr")


# ---------------------------------------------------------------- metrics
def binary(y, score, threshold=0.5):
    y = np.asarray(y, bool)
    pred = np.asarray(score) >= threshold
    tp, fp = int(np.sum(pred & y)), int(np.sum(pred & ~y))
    tn, fn = int(np.sum(~pred & ~y)), int(np.sum(~pred & y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"n_drone": int(y.sum()), "n_not": int((~y).sum()), "auc": float(roc_auc_score(y, score)),
            "accuracy": (tp + tn) / len(y), "balanced_accuracy": (recall + specificity) / 2,
            "precision": precision, "recall": recall, "specificity": specificity,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "majority_class_accuracy": max(y.sum(), (~y).sum()) / len(y)}


def multiclass(y, pred):
    y, pred = np.asarray(y, object), np.asarray(pred, object)
    truth = sorted(set(y))
    _, counts = np.unique(y, return_counts=True)
    return {"n": int(len(y)), "classes": len(truth), "accuracy": float(np.mean(y == pred)),
            "balanced_accuracy": E.metric(y, pred),
            "macro_f1": float(f1_score(y.astype(str), pred.astype(str), labels=[str(t) for t in truth], average="macro",
                                       zero_division=0)),
            "majority_class_accuracy": float(counts.max() / len(y))}


# ---------------------------------------------------------------- fitting
def _fit(X_tr, y_tr, X_te, kind):
    with threadpool_limits(1):
        model = E.model(kind).fit(X_tr, y_tr)
        return list(model.classes_), model.predict_proba(X_te), model.predict_proba(X_tr)


def cross(X_tr, y_tr, X_te, y_te, kind="gbm"):
    classes, p_te, p_tr = _fit(X_tr, y_tr, X_te, kind)
    j = classes.index(True)
    return {"train": binary(y_tr, p_tr[:, j]), "test": binary(y_te, p_te[:, j])}


def within(X, y, parts, kind="gbm"):
    folds = sorted(set(parts))
    out = Parallel(n_jobs=-1)(delayed(_fit)(X[parts != f], y[parts != f], X[parts == f], kind) for f in folds)
    score = np.zeros(len(y))
    train = []
    for f, (classes, p_te, p_tr) in zip(folds, out):
        j = classes.index(True)
        score[parts == f] = p_te[:, j]
        train.append(binary(y[parts != f], p_tr[:, j]))
    return {"train": {k: float(np.mean([t[k] for t in train])) for k in train[0]}, "test": binary(y, score)}


def held_out_multiclass(X, y, folds, capture=None):
    out = Parallel(n_jobs=-1)(delayed(_fit)(X[tr], y[tr], X[te], "gbm") for tr, te in folds)
    classes = sorted(set(y))
    rows, P, train_acc = [], [], []
    for (fold_classes, p_te, p_tr), (tr, te) in zip(out, folds):
        full = np.zeros((int(te.sum()), len(classes)))
        for j, c in enumerate(fold_classes):
            full[:, classes.index(c)] = p_te[:, j]
        rows.append(np.flatnonzero(te))
        P.append(full)
        train_acc.append(float(np.mean(np.array(fold_classes, object)[p_tr.argmax(axis=1)] == y[tr])))
    rows, P = np.concatenate(rows), np.vstack(P)
    pred = np.array(classes, object)[P.argmax(axis=1)]
    result = {"train_accuracy": float(np.mean(train_acc)), "test": multiclass(y[rows], pred)}
    if capture is not None:
        cap = capture[rows]
        ids = np.unique(cap)
        cap_pred = [np.array(classes, object)[P[cap == c].mean(axis=0).argmax()] for c in ids]
        result["test_per_capture"] = multiclass([y[rows][cap == c][0] for c in ids], cap_pred)
    return result


# ---------------------------------------------------------------- the tests
def acoustic():
    out = {}
    data = {d: AG.load(d) for d in ("svanstrom", "uavirbase")}
    for d, t in data.items():
        assignment = splits.load(f"{d}/group-kfold-s0")
        rows = json.loads(sorted(AG.DETECTORS.glob(f"{d}-*.features.json"))[-1].read_text(encoding="utf-8"))
        parts = np.array([assignment[r["capture_id"]] for r in rows])
        out[f"Acoustic within {d} (unseen recordings, same setup)"] = within(t["X"], t["y"], parts)
    for a, b in (("uavirbase", "svanstrom"), ("svanstrom", "uavirbase")):
        out[f"Acoustic cross: train {a} -> test {b} (channel-averaged, detector)"] = cross(
            data[a]["X"], data[a]["y"], data[b]["X"], data[b]["y"])
    rows = SM.load_windows()
    for a, b in (("uavirbase", "svanstrom"), ("svanstrom", "uavirbase")):
        ta, tb = SM.table(rows, a, "single"), SM.table(rows, b, "single")
        for name, prefix, kind in (("detector, 41 features", SM.DETECTOR_PREFIX, "gbm"), ("MFCC, gbm", ("mfcc_",), "gbm"),
                                   ("MFCC, logistic", ("mfcc_",), "linear")):
            out[f"Acoustic cross: train {a} -> test {b} (single microphone, {name})"] = cross(
                SM.cols(ta, prefix)[0], ta["y"], SM.cols(tb, prefix)[0], tb["y"], kind)
    return out


def rf_rows(dataset):
    rows = json.loads(sorted((DATA_ROOT / "results" / "rf_cross_v1.2").glob(f"{dataset}-*.features.json"))[-1]
                      .read_text(encoding="utf-8"))
    captures = {c["capture_id"]: c for c in splits.captures_of(dataset)}
    names = sorted(k for k in rows[0] if k.startswith(RF_PREFIX))
    return {"X": np.array([[r[k] for k in names] for r in rows]), "ids": [r["capture_id"] for r in rows],
            "y": np.array([captures[r["capture_id"]]["label_uas_present"] for r in rows]),
            "power": np.array([r["log_rms"] for r in rows])}


def rf_detection():
    out = {}
    data = {d: rf_rows(d) for d in ("cardrf", "uavsig")}
    for d, split in (("cardrf", "cardrf/system-kfold-s0"), ("uavsig", "uavsig/group-kfold-s0")):
        assignment = splits.load(split)
        t = data[d]
        out[f"RF within {d} (unseen devices/scenarios, same receiver)"] = within(
            t["X"], t["y"], np.array([assignment[i] for i in t["ids"]]))
        out[f"RF within {d}: received power alone (AUC only)"] = {"test": {"auc": float(roc_auc_score(t["y"], t["power"]))}}
    for a, b in (("uavsig", "cardrf"), ("cardrf", "uavsig")):
        out[f"RF cross-receiver: train {a} -> test {b}"] = cross(data[a]["X"], data[a]["y"], data[b]["X"], data[b]["y"])
    stage2 = DATA_ROOT / "results" / "rf_stage2"
    feats = {d: json.loads(sorted(stage2.glob(f"{d}-*.features.json"))[-1].read_text(encoding="utf-8")) for d in ("cardrf", "noisy_rf")}
    names = sorted(feats["cardrf"][0])
    labels = {d: {c["capture_id"]: c["label_uas_present"] for c in splits.captures_of(d)} for d in feats}
    X = {d: np.array([[f[k] for k in names] for f in feats[d]]) for d in feats}
    y = {d: np.array([labels[d][i] for i in sorted(labels[d])]) for d in feats}
    out["RF cross-receiver: train CardRF -> test Noisy RF (stage 2, gbm)"] = cross(X["cardrf"], y["cardrf"], X["noisy_rf"], y["noisy_rf"])
    return out


def identification():
    out = {}
    data, _ = DS.load()
    for rep in ("tiles", "emitters"):
        r = data[rep]
        for name, folds in DS.D.splits_of(r["unit"], r["model"], r["day"], r["rx"]).items():
            out[f"DRFF-R2 [{rep}] {name}"] = held_out_multiclass(r["X"], r["model"], folds, r["capture"])
    rows = RR.load()
    for rep in ("tiles", "emitters"):
        t = RR.table(rows[rep], "rfuav")
        packs = {}
        for g, label in zip(t["groups"], t["y"]):
            packs.setdefault(label, set()).add(g)
        tested = sorted(g for label, p in packs.items() if len(p) >= 2 for g in p)
        out[f"RFUAV [{rep}] unseen recording pack (37-way)"] = held_out_multiclass(
            t["X"], t["y"], [(t["groups"] != g, t["groups"] == g) for g in tested])
        m = RR.table(rows[rep], "rma")
        files = np.array([r["capture_id"] for r in rows[rep] if r["dataset"] == "rma"])
        unique = sorted(set(files))
        label = dict(zip(files, m["y"]))
        fold_of = {}
        for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(unique, [label[f] for f in unique])):
            fold_of.update({unique[i]: k for i in te})
        fold = np.array([fold_of[f] for f in files])
        out[f"RMA [{rep}] within session (upper bound; session = class)"] = held_out_multiclass(
            m["X"], m["y"], [(fold != k, fold == k) for k in range(5)])
    return out


def timing():
    from droneacharya import features as F
    from droneacharya import rf_emitters as RE
    from droneacharya.mfcc import mfcc_features
    from droneacharya.rf_features import rf_tile_features
    rng = np.random.default_rng(0)
    audio = rng.normal(size=48000)
    tile = (rng.normal(size=10_000) + 1j * rng.normal(size=10_000)).astype(np.complex64)
    capture = (rng.normal(size=10_000_000) + 1j * rng.normal(size=10_000_000)).astype(np.complex64)

    def best(fn, repeat):
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    model = E.model("gbm").fit(rng.normal(size=(500, 41)), rng.integers(0, 2, 500).astype(bool))
    return {"acoustic 1 s window: 41 features (ms)": 1e3 * best(lambda: F.audio_detector_features(audio, 48000), 5),
            "acoustic 1 s window: 26 MFCC (ms)": 1e3 * best(lambda: mfcc_features(audio, 48000), 5),
            "gbm decision on one window (ms)": 1e3 * best(lambda: model.predict_proba(rng.normal(size=(1, 41))), 20),
            "RF 250 us tile: snapshot features (ms)": 1e3 * best(lambda: rf_tile_features(tile, 40e6), 10),
            "RF 100 ms at 100 MS/s: activity + emitter tracking (s)": best(lambda: RE.analyse(capture, 100e6), 2),
            "note": "one CPU thread of an i5-13500H laptop; time to compute, not wall-clock latency of a deployed system"}


def table_lines(results):
    lines = ["<!-- Generated by scripts/report_all_metrics.py; results in DroneacharyaData/results/checks/all_metrics.json. -->", "",
             "> Raw output, rewritten on every run. Which numbers may be quoted, and with what limits: [claims.md](claims.md).", "",
             "# Every main test with the full metric set", "",
             "Binary: decision = drone when the class-balanced model's probability ≥ 0.5 (fixed before running). "
             "Train columns score the model on its own training rows; the gap to test shows how much does not transfer.", "",
             "## Detection (drone vs not)", "",
             "| Test | Drone / not | Train acc | Test AUC | Test acc | Balanced acc | Precision | Recall | Specificity | F1 | Always-drone acc |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in results["detection"].items():
        t = r["test"]
        if "accuracy" not in t:
            lines.append(f"| {name} | | | {t['auc']:.2f} | | | | | | | |")
            continue
        lines.append(f"| {name} | {t['n_drone']} / {t['n_not']} | {r['train']['accuracy']:.2f} | {t['auc']:.2f} | {t['accuracy']:.2f} | "
                     f"{t['balanced_accuracy']:.2f} | {t['precision']:.2f} | {t['recall']:.2f} | {t['specificity']:.2f} | "
                     f"{t['f1']:.2f} | {t['majority_class_accuracy']:.2f} |")
    lines += ["", "## Identification (which drone model)", "",
              "| Test | Classes | Train acc | Test acc | Balanced acc | Macro F1 | Per-capture balanced acc | Majority-class acc |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in results["identification"].items():
        t = r["test"]
        cap = r.get("test_per_capture", {}).get("balanced_accuracy")
        lines.append(f"| {name} | {t['classes']} | {r['train_accuracy']:.2f} | {t['accuracy']:.2f} | {t['balanced_accuracy']:.2f} | "
                     f"{t['macro_f1']:.2f} | {'' if cap is None else f'{cap:.2f}'} | {t['majority_class_accuracy']:.2f} |")
    lines += ["", "## Processing time per decision", ""] + [f"- {k}: {v:.1f}" if isinstance(v, float) else f"- {k}: {v}"
                                                           for k, v in results["timing"].items()]
    return lines


def main():
    start = time.time()
    results = {"detection": {**acoustic(), **rf_detection()}, "identification": identification(), "timing": timing(),
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (OUT / "all_metrics.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    REPORT.write_text("\n".join(table_lines(results)) + "\n", encoding="utf-8")
    print("\n".join(table_lines(results)))
    print(f"done in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
