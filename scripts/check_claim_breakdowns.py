"""Breakdowns behind two claims the 2026-10-08 audit questioned. Diagnostics on saved features, not new tests.

1. Merged drone-only pool → CardRF (scripts/run_merged_one_class.py): AUC split by non-drone type (Wi-Fi: 2 routers,
   Bluetooth: 5 devices), next to single-feature references computed inside CardRF. Is the model only rejecting
   Bluetooth? The pool features are the cache written by that run; its cache key has since changed only because
   cnn.py (not used by the features) was edited, so a seeded sample of pool captures is recomputed with the current
   code and compared before the cache is used.
2. DRFF-R2 Dataset 3 (scripts/run_drff_identification.py): which physical units appear on which day and receiver.
   Were the units tested on days 2–3 and on receiver u1 already in training (all of day 1)?

The single-feature "best" reference picks the feature and its direction on CardRF itself, so it is optimistic.
Runs in one process (no workers: importing run_merged_one_class loads torch). Output:
DroneacharyaData/results/checks/claim_breakdowns.json
"""
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_merged_one_class as M  # noqa: E402
from droneacharya.index import read  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

OUT = DATA_ROOT / "results" / "checks"
SAMPLE = 6


def auc_by_negative(y, scores, kind):
    return {"all": float(roc_auc_score(y, scores)),
            **{k: float(roc_auc_score(y[y | (kind == k)], scores[y | (kind == k)])) for k in ("wifi", "bluetooth")}}


def cached_pool():
    path = sorted(M.OUT.glob("rf_pool-*.features.json"))[-1]
    pool = json.loads(path.read_text(encoding="utf-8"))
    jobs = M.pool_jobs()
    checked = []
    for j in np.random.default_rng(0).choice(len(jobs), SAMPLE, replace=False):
        fresh, _ = M._pool_rows(*jobs[j])
        old = [r for r in pool if r["capture_id"] == jobs[j][1]]
        same = len(old) == len(fresh) and all(abs(a[k] - b[k]) < 1e-6 for a, b in zip(old, fresh)
                                              for k in a if isinstance(a[k], float))
        checked.append({"capture_id": jobs[j][1], "matches_current_code": same})
    if not all(c["matches_current_code"] for c in checked):
        raise SystemExit(f"cached pool {path.name} no longer matches the feature code: {checked}")
    return pool, path.name, checked


def cardrf_by_negative_type():
    pool, cache_name, checked = cached_pool()
    names = M.names_of(pool)
    pool_X = np.array([[r[k] for k in names] for r in pool])
    ids, feats, _, meta = M.stage2_set("cardrf")
    X = np.array([[f[k] for k in names] for f in feats])
    y = np.array([meta[i]["label_uas_present"] for i in ids])
    kind = np.array([meta[i]["label_emitter"] for i in ids])
    devices = {k: sorted({meta[i]["label_class"] for i in ids if meta[i]["label_emitter"] == k}) for k in ("wifi", "bluetooth")}
    scores = M.one_class_scores(pool_X, X)
    single = {k: auc_by_negative(y, X[:, names.index(k)], kind) for k in ("occupied_bw_mhz", "duty_cycle", "papr_db")}
    best = max(names, key=lambda k: abs(roc_auc_score(y, X[:, names.index(k)]) - 0.5))
    v = X[:, names.index(best)]
    direction = 1 if roc_auc_score(y, v) >= 0.5 else -1
    return {"pool_cache": cache_name, "pool_cache_check": checked,
            "negatives": {k: {"tiles": int((kind == k).sum()), "devices": d} for k, d in devices.items()},
            "one_class": {m: auc_by_negative(y, s, kind) for m, s in scores.items()},
            "single_feature": single,
            "best_single_feature_optimistic": {"feature": best, "direction": direction,
                                               **auc_by_negative(y, direction * v, kind)}}


def drff_units_by_day():
    fields = defaultdict(dict)
    for r in read("drff_r2", "evidence").to_pylist():
        if r["field"].startswith("mat."):
            fields[r["entity_id"]][r["field"][4:]] = r["value"]
    units = defaultdict(set)
    for c in read("drff_r2", "captures").to_pylist():
        if "/dataset3-" in c["capture_id"]:
            f = fields[c["capture_id"]]
            units[f"{f['D']} {f['U']}"].add(c["label_class"])
    day1 = set().union(*(u for k, u in units.items() if k.startswith("d1")))
    later = set().union(*(u for k, u in units.items() if not k.startswith("d1")))
    return {"units_by_day_receiver": {k: sorted(v) for k, v in sorted(units.items())},
            "units_off_day1": sorted(later - day1), "units_per_model_on_day1": dict(Counter(u.rsplit("_", 1)[0] for u in day1))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"cardrf_merged_one_class": cardrf_by_negative_type(), "drff_r2_dataset3": drff_units_by_day(),
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (OUT / "claim_breakdowns.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    c = result["cardrf_merged_one_class"]
    for name, r in {**c["one_class"], **c["single_feature"], "best single (optimistic)": c["best_single_feature_optimistic"]}.items():
        print(f"{name:26} all {r['all']:.3f}  wifi {r['wifi']:.3f}  bluetooth {r['bluetooth']:.3f}")
    d = result["drff_r2_dataset3"]
    print({k: len(v) for k, v in d["units_by_day_receiver"].items()}, "units off day 1:", d["units_off_day1"])


if __name__ == "__main__":
    main()
