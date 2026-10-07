"""Run the shortcut baselines on the saved splits and write results + a summary page.

Outputs: DroneacharyaData/results/shortcut_baselines/<name>.json and
docs/results/shortcut_baselines.md. Feature tables are cached next to the
results, so re-runs only re-fit models.

Usage:
    python scripts/run_shortcut_baselines.py
"""
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya import baselines as B  # noqa: E402
from droneacharya import splits  # noqa: E402
from droneacharya.index import read  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

OUT = DATA_ROOT / "results" / "shortcut_baselines"
REPORT = Path(__file__).resolve().parents[1] / "docs" / "results" / "shortcut_baselines.md"
SEED = 0
CARDRF_PER_GROUP = 60


def cached(name, compute):
    path = OUT / f"{name}.features.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows = compute()
    path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def save(name, result):
    result["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{name}: {result.get('summary') or result.get('accuracy')}", flush=True)
    return result


def cardrf():
    split_id = "cardrf/group-kfold-s0"
    assignment = splits.load(split_id)
    by_group = defaultdict(list)
    for c in splits.captures_of("cardrf"):
        by_group[c["group_id"]].append(c)
    rng = random.Random(SEED)
    chosen = [c for g in sorted(by_group) for c in rng.sample(sorted(by_group[g], key=lambda c: c["capture_id"]),
                                                              min(CARDRF_PER_GROUP, len(by_group[g])))]

    def compute():
        rows = []
        for i, c in enumerate(chosen, 1):
            rows.append({"capture_id": c["capture_id"], "y": c["label_uas_present"], **B.cardrf_features(c["capture_id"])})
            if i % 200 == 0:
                print(f"  cardrf features {i}/{len(chosen)}", flush=True)
        return rows

    rows = cached("cardrf", compute)
    parts = [assignment[r["capture_id"]] for r in rows]
    y = [r["y"] for r in rows]
    results = {}
    for name, cols in (("clip_fraction", ["clip_fraction"]), ("rms_pre_trigger", ["log_rms_pre_trigger"]),
                       ("rms_post_trigger", ["log_rms_post_trigger"]),
                       ("clip_and_energy", ["clip_fraction", "log_rms_pre_trigger", "log_rms_post_trigger"])):
        results[name] = save(f"cardrf_{name}", {
            "task": "UAS (aircraft or controller) vs Wi-Fi/Bluetooth, unseen devices", "split": split_id,
            "features": cols, "sampling": f"up to {CARDRF_PER_GROUP} captures per device, seed {SEED}",
            **B.evaluate([[r[c] for c in cols] for r in rows], y, parts)})
    return results


def audio_level(dataset, max_samples=None):
    split_id = f"{dataset}/group-kfold-s0"
    assignment = splits.load(split_id)

    def compute():
        rows = []
        for c in splits.captures_of(dataset):
            if c["label_uas_present"] is None:
                continue
            windows, _ = B.audio_windows(c["capture_id"], max_samples=max_samples)
            rows += [{"capture_id": c["capture_id"], "y": c["label_uas_present"], "dbfs": B.level_dbfs(w)} for w in windows]
        return rows

    rows = cached(dataset, compute)
    return save(f"{dataset}_loudness", {
        "task": "drone vs non-drone from loudness only (1 s windows, unseen groups)", "split": split_id,
        "features": ["dbfs"], "windows_by_label": B.per_dataset_counts(str(r["y"]) for r in rows),
        **B.evaluate([[r["dbfs"]] for r in rows], [r["y"] for r in rows],
                           [assignment[r["capture_id"]] for r in rows])})


def acoustic_dataset_id():
    datasets = ["ddl", "uavirbase", "svanstrom", "dronenoise", "miesikowska_uav", "esc50"]

    def compute():
        rng = random.Random(SEED)
        rows = []
        for dataset in datasets:
            captures = [c for c in splits.captures_of(dataset) if c["label_uas_present"] is not None]
            rng.shuffle(captures)
            taken = 0
            for c in captures:
                windows, rate = B.audio_windows(c["capture_id"], max_windows=20 if dataset == "ddl" else 5,
                                                max_samples=int(25 * 96000))
                for w in windows:
                    rows.append({"dataset": dataset, "group": c["group_id"], "y_uas": c["label_uas_present"],
                                 "raw": B.spectral_signature(w, rate).tolist(),
                                 "norm": B.spectral_signature(w, rate, normalise=True).tolist()})
                taken += len(windows)
                if taken >= 200:
                    break
        return rows

    rows = cached("acoustic_dataset_id", compute)
    parts = B.group_partitions([r["group"] for r in rows])
    y = [r["dataset"] for r in rows]
    results = {}
    for variant in ("raw", "norm"):
        results[variant] = save(f"acoustic_dataset_id_{variant}", {
            "task": "which dataset a 1 s window comes from (6 datasets), unseen groups",
            "split": "groups dealt round-robin into 5 folds (deterministic)",
            "features": "32 log band energies 0–8 kHz after resampling to 16 kHz"
                        + (", each window scaled to unit RMS first" if variant == "norm" else ""),
            "chance_balanced_accuracy": 1 / len(datasets), "windows_by_dataset": B.per_dataset_counts(y),
            **B.evaluate([r[variant] for r in rows], y, parts, binary=False)})
    return results


def rfuav_metadata():
    captures = read("rfuav", "captures").to_pylist()
    evidence = read("rfuav", "evidence").to_pylist()
    scale = {r["entity_id"]: float(r["value"]) for r in evidence if r["field"] == "xml.ScaleFactor"}
    packs = defaultdict(set)
    for c in captures:
        packs[c["label_class"]].add(c["group_id"])
    kept = [c for c in captures if len(packs[c["label_class"]]) >= 2]
    X = [[c["center_frequency_hz"] / 1e9, c["reference_snr_db"], scale[c["group_id"]]] for c in kept]
    y = [c["label_class"] for c in kept]
    groups = [c["group_id"] for c in kept]
    predictions = []
    for g in sorted(set(groups)):
        test = [i for i, gg in enumerate(groups) if gg == g]
        train = [i for i, gg in enumerate(groups) if gg != g]
        fitted = B.model().fit([X[i] for i in train], [y[i] for i in train])
        predictions += [(y[i], fitted.predict([X[i]])[0]) for i in test]
    accuracy = float(np.mean([a == b for a, b in predictions]))
    classes = sorted(set(y))
    return save("rfuav_metadata_identification", {
        "task": "RFUAV class from acquisition metadata only (no signal), leave-one-pack-out",
        "split": "leave one pack out; only classes with >= 2 packs can be tested this way",
        "features": ["center_frequency_ghz", "reference_snr_db", "xml_scale_factor"],
        "classes": classes, "packs": len(set(groups)), "captures": len(kept),
        "accuracy": accuracy, "chance_accuracy": 1 / len(classes),
        "classes_untestable_with_one_pack": sorted(k for k, v in packs.items() if len(v) < 2)})


def write_report(results):
    def fmt(r):
        s = r["summary"]
        auc = s.get("roc_auc")
        return (f"{s['balanced_accuracy']['mean']:.2f} ({s['balanced_accuracy']['min']:.2f}–{s['balanced_accuracy']['max']:.2f})",
                f"{auc['mean']:.2f} ({auc['min']:.2f}–{auc['max']:.2f})" if auc else "—")

    lines = ["<!-- Generated by scripts/run_shortcut_baselines.py; results in DroneacharyaData/results/shortcut_baselines/. -->",
             "", "# Shortcut baselines", "",
             "Each model sees one confound only and is scored on groups it never saw (saved, leakage-checked splits).",
             "Chance is 0.50 balanced accuracy / 0.50 ROC AUC for the binary tasks. A real detector must clearly beat",
             "the baseline for its task before its result means anything.", "",
             "| Baseline | Task | Split | Model | Balanced accuracy, mean (fold min–max) | ROC AUC, mean (min–max) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, r in results.items():
        if "summary" in r and r["summary"]:
            ba, auc = fmt(r)
            lines.append(f"| {name} | {r['task']} | `{r['split']}` | {r['model']} | {ba} | {auc} |")
    rf = results["rfuav_metadata_identification"]
    lines += ["", f"RFUAV metadata-only identification (leave one pack out, {len(rf['classes'])} classes with ≥ 2 packs, "
              f"{rf['captures']} captures): accuracy {rf['accuracy']:.2f} vs chance {rf['chance_accuracy']:.2f}. "
              f"{len(rf['classes_untestable_with_one_pack'])} classes have a single pack, so unseen-pack identification "
              "cannot be tested for them at all.", ""]
    for variant in ("raw", "norm"):
        r = results[f"acoustic_dataset_id_{variant}"]
        lines.append(f"Acoustic dataset identification ({'level kept' if variant == 'raw' else 'each window scaled to unit RMS'}): "
                     f"balanced accuracy {r['summary']['balanced_accuracy']['mean']:.2f} vs chance {r['chance_balanced_accuracy']:.2f} "
                     f"(windows: {r['windows_by_dataset']}).")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    results = {}
    results.update({f"cardrf_{k}": v for k, v in cardrf().items()})
    results["svanstrom_loudness"] = audio_level("svanstrom")
    results["uavirbase_loudness"] = audio_level("uavirbase", max_samples=10 * 96000)
    results.update({f"acoustic_dataset_id_{k}": v for k, v in acoustic_dataset_id().items()})
    results["rfuav_metadata_identification"] = rfuav_metadata()
    write_report(results)
    print(f"done in {(time.time() - start) / 60:.1f} min; report {REPORT}")


if __name__ == "__main__":
    main()
