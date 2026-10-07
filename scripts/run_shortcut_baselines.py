"""Run the shortcut baselines on saved splits and write results + a summary page.

Every baseline: one model kind fixed in advance, pooled out-of-fold metric,
group-permutation p-value and group-bootstrap 95% interval (see
src/droneacharya/baselines.py). Feature caches are keyed by a hash of the
feature code and parameters, so changed code never reuses old features.

Outputs: DroneacharyaData/results/shortcut_baselines/<name>.json and
docs/results/shortcut_baselines.md.

Usage:
    python scripts/run_shortcut_baselines.py
"""
import hashlib
import inspect
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya import baselines as B  # noqa: E402
from droneacharya import splits  # noqa: E402
from droneacharya.index import read  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

OUT = DATA_ROOT / "results" / "shortcut_baselines"
REPORT = Path(__file__).resolve().parents[1] / "docs" / "results" / "shortcut_baselines.md"
SEED = 0
CARDRF_PER_GROUP = 60
ACOUSTIC = ["ddl", "uavirbase", "svanstrom", "dronenoise", "miesikowska_uav", "esc50"]
DATASET_ID_SPLIT = "acoustic/dataset-id-s0"


def cached(name, compute, params):
    key = hashlib.sha256(Path(B.__file__).read_bytes() + inspect.getsource(compute).encode()
                         + json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    path = OUT / f"{name}-{key}.features.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows = compute()
    path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def save(name, result):
    result["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{name}: {result['metric']} {result[result['metric']]:.3f} CI {[round(v, 3) for v in result['ci95']]} "
          f"p={result['permutation']['p_value']:.3f}", flush=True)
    return result


def cardrf():
    split_id = "cardrf/system-kfold-s0"
    assignment = splits.load(split_id)
    captures = {c["capture_id"]: c for c in splits.captures_of("cardrf")}
    by_group = defaultdict(list)
    for c in captures.values():
        by_group[c["group_id"]].append(c)

    def compute():
        rng = random.Random(SEED)
        chosen = [c for g in sorted(by_group) for c in rng.sample(sorted(by_group[g], key=lambda c: c["capture_id"]),
                                                                  min(CARDRF_PER_GROUP, len(by_group[g])))]
        rows = []
        for i, c in enumerate(chosen, 1):
            rows.append({"capture_id": c["capture_id"], **B.cardrf_features(c["capture_id"])})
            if i % 200 == 0:
                print(f"  cardrf features {i}/{len(chosen)}", flush=True)
        return rows

    rows = cached("cardrf", compute, {"per_group": CARDRF_PER_GROUP, "seed": SEED, "groups": sorted(by_group)})
    burst = lambda c: not (c["label_emitter"] == "aircraft" and c["label_class"].startswith("DJI"))  # noqa: E731
    task = "UAS vs Wi-Fi/Bluetooth, unseen UAS systems and devices"
    results = {}
    for name, cols, keep, description in (
            ("clip_fraction", ["clip_fraction"], None, task),
            ("rms_pre_trigger", ["log_rms_pre_trigger"], None, task),
            ("rms_post_trigger", ["log_rms_post_trigger"], None, task),
            ("clip_and_energy", ["clip_fraction", "log_rms_pre_trigger", "log_rms_post_trigger"], None, task),
            ("rms_pre_trigger_burst_only", ["log_rms_pre_trigger"], burst,
             "burst emitters only (pre-trigger = noise): skill here would mean session/device leakage")):
        sel = [r for r in rows if keep is None or keep(captures[r["capture_id"]])]
        results[f"cardrf_{name}"] = save(f"cardrf_{name}", {
            "task": description, "split": split_id, "features": cols,
            "sampling": f"up to {CARDRF_PER_GROUP} captures per group, seed {SEED}",
            **B.evaluate([[r[c] for c in cols] for r in sel], [captures[r["capture_id"]]["label_uas_present"] for r in sel],
                         [assignment[r["capture_id"]] for r in sel], [captures[r["capture_id"]]["group_id"] for r in sel],
                         "tree")})
    return results


def audio_level(dataset, max_samples=None):
    split_id = f"{dataset}/group-kfold-s0"
    assignment = splits.load(split_id)
    captures = {c["capture_id"]: c for c in splits.captures_of(dataset) if c["label_uas_present"] is not None}

    def compute():
        rows = []
        for c in captures.values():
            windows, _ = B.audio_windows(c["capture_id"], max_samples=max_samples)
            rows += [{"capture_id": c["capture_id"], "dbfs": B.level_dbfs(w)} for w in windows]
        return rows

    rows = cached(dataset, compute, {"max_samples": max_samples, "window_s": 1.0, "max_windows": 10})
    return save(f"{dataset}_loudness", {
        "task": "drone vs non-drone from loudness only (1 s windows, unseen groups)", "split": split_id,
        "features": ["dbfs"],
        **B.evaluate([[r["dbfs"]] for r in rows], [captures[r["capture_id"]]["label_uas_present"] for r in rows],
                     [assignment[r["capture_id"]] for r in rows], [captures[r["capture_id"]]["group_id"] for r in rows],
                     "tree")})


def dataset_id_split():
    captures = [c for d in ACOUSTIC for c in splits.captures_of(d) if c["label_uas_present"] is not None]
    assignment = splits.group_kfold(captures, 5, seed=SEED, label="dataset_id")
    splits.save(DATASET_ID_SPLIT, "experiment", "group k-fold over 6 acoustic datasets, k=5, dataset-balanced",
                captures, assignment, seed=SEED, notes="for dataset-identification baselines")
    return {c["capture_id"]: c for c in captures}, splits.load(DATASET_ID_SPLIT)


def acoustic_dataset_id():
    captures, assignment = dataset_id_split()

    def compute():
        rng = random.Random(SEED)
        rows = []
        for dataset in ACOUSTIC:
            pool = sorted((c for c in captures.values() if c["dataset_id"] == dataset), key=lambda c: c["capture_id"])
            rng.shuffle(pool)
            taken = 0
            for c in pool:
                windows, rate = B.audio_windows(c["capture_id"], max_windows=20 if dataset == "ddl" else 5,
                                                max_samples=int(25 * 96000))
                rows += [{"capture_id": c["capture_id"], "raw": B.spectral_signature(w, rate).tolist(),
                          "norm": B.spectral_signature(w, rate, normalise=True).tolist()} for w in windows]
                taken += len(windows)
                if taken >= 200:
                    break
        return rows

    rows = cached("acoustic_dataset_id", compute, {"seed": SEED, "datasets": ACOUSTIC, "per_dataset": 200})
    results = {}
    for subset, keep in (("all", lambda c: True), ("drones_only", lambda c: c["label_uas_present"])):
        sel = [r for r in rows if keep(captures[r["capture_id"]])]
        labels = [captures[r["capture_id"]]["dataset_id"] for r in sel]
        for variant in ("raw", "norm"):
            name = f"acoustic_dataset_id_{subset}_{variant}"
            results[name] = save(name, {
                "task": f"which dataset a 1 s window comes from ({len(set(labels))} datasets, "
                        f"{'drone windows only' if subset == 'drones_only' else 'all windows'}), unseen groups",
                "split": DATASET_ID_SPLIT,
                "features": "32 log band energies 0–8 kHz after resampling to 16 kHz"
                            + (", each window scaled to unit RMS first" if variant == "norm" else ""),
                "chance_balanced_accuracy": 1 / len(set(labels)), "windows_by_dataset": B.per_dataset_counts(labels),
                **B.evaluate([r[variant] for r in sel], labels, [assignment[r["capture_id"]] for r in sel],
                             [captures[r["capture_id"]]["group_id"] for r in sel], "linear", n_permutations=200)})
    return results


def rfuav_metadata():
    captures = read("rfuav", "captures").to_pylist()
    scale = {r["entity_id"]: float(r["value"]) for r in read("rfuav", "evidence").to_pylist() if r["field"] == "xml.ScaleFactor"}
    packs = defaultdict(set)
    for c in captures:
        packs[c["label_class"]].add(c["group_id"])
    kept = [c for c in captures if len(packs[c["label_class"]]) >= 2]
    labels = [c["label_class"] for c in kept]
    return save("rfuav_metadata_identification", {
        "task": "RFUAV class from acquisition metadata only (no signal), leave one pack out; "
                f"only the {len(set(labels))} classes with ≥ 2 packs can be tested this way",
        "split": "leave one pack out", "features": ["center_frequency_ghz", "reference_snr_db", "xml_scale_factor"],
        "chance_balanced_accuracy": 1 / len(set(labels)),
        "classes_untestable_with_one_pack": sorted(k for k, v in packs.items() if len(v) < 2),
        **B.evaluate([[c["center_frequency_hz"] / 1e9, c["reference_snr_db"], scale[c["group_id"]]] for c in kept],
                     labels, [c["group_id"] for c in kept], [c["group_id"] for c in kept], "linear"),
        "model_note": "linear: 31 rows cannot support the depth-3 tree (20 rows per leaf), which never splits"})


def write_report(results):
    lines = ["<!-- Generated by scripts/run_shortcut_baselines.py; results in DroneacharyaData/results/shortcut_baselines/. -->",
             "", "# Shortcut baselines", "",
             "Each model sees one confound only and predicts groups it never saw (saved, leakage-checked splits).",
             "Scored as one metric over the pooled out-of-fold predictions; the model kind is fixed per baseline in",
             "advance (tree for scalar confounds, linear for the 32-band signature); **p** is the share of group-level",
             "label permutations (1,000; 200 for dataset ID) scoring at least as high, and the interval resamples whole",
             "groups. With few groups the intervals are wide: CardRF has 13 groups, UaVirBASE 4 negative recordings.", "",
             "These bars are **minimums**: a stronger model can exploit the same confound better. Beating them is",
             "necessary, not sufficient — detectors are also scored with the confound removed.", "",
             "| Baseline | Task | Split | Groups by label | Metric | Value [95% CI] | p | Null median |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in results.items():
        groups = ", ".join(f"{k}: {v}" for k, v in sorted(r["groups_per_label"].items())) if r["metric"] == "roc_auc" \
            else f"{sum(r['groups_per_label'].values())} groups"
        value = f"{r[r['metric']]:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]"
        chance = f" (chance {r['chance_balanced_accuracy']:.2f})" if "chance_balanced_accuracy" in r else ""
        lines.append(f"| {name} | {r['task']}{chance} | `{r['split']}` | {groups} | {r['metric']} | {value} | "
                     f"{r['permutation']['p_value']:.3f} | {r['permutation']['null_median']:.2f} |")
    lines += ["", "Dataset identification uses 0–8 kHz only (every recording resampled to 16 kHz), so sample rate and "
              "anti-alias roll-off cannot drive it. The drones-only variant removes the drone/non-drone content "
              "difference, leaving recording chain, site and drone model. Both are lower bounds on identifiability."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    results = {}
    results.update(cardrf())
    results["svanstrom_loudness"] = audio_level("svanstrom")
    results["uavirbase_loudness"] = audio_level("uavirbase", max_samples=10 * 96000)
    results.update(acoustic_dataset_id())
    results["rfuav_metadata_identification"] = rfuav_metadata()
    write_report(results)
    print(f"done in {(time.time() - start) / 60:.1f} min; report {REPORT}")


if __name__ == "__main__":
    main()
