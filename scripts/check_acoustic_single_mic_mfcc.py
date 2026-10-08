"""Acoustic: does A0 survive without array averaging, does MFCC add anything, and how do the cues decay with range?

Design committed before running (2026-10-08). Answers two audit points: the window reader averaged all channels
(UaVirBASE is 8 directional shotgun microphones metres apart, so averaging comb-filters most at high frequency, where
the 7 kHz cue lives), and MFCC, the literature's standard feature, was never tried.

Svanström and UaVirBASE are development data now (the 7 kHz idea came from looking at them), so parts 1–2 are
diagnostics: they can weaken A0, never upgrade it. A claim needs an untouched third recording setup.

  windows   same recordings, positions and silence rule as run_detectors.py (1 s; Svanström first 10 windows,
            UaVirBASE first 10 s), but no channel averaging:
              single   channel 0 only (primary)
              featavg  features computed per channel, then averaged (secondary)
  features  the 41 detector features (features.py) + loudness, and 26 MFCCs (mfcc.py: 13 coefficients, mean and
            standard deviation; no cepstral mean normalisation)
  part 1    A0 again: the single feature with the largest |AUC − 0.5| on the training chain (41 + loudness), applied
            unchanged to the other chain; and the gbm detector on the 41 features. Both directions. Compared with the
            channel-averaged results (results/checks/acoustic_single_feature_transfer.json).
  part 2    MFCC: gbm and logistic regression on the 26 MFCCs, trained on one chain, tested on the other; paired
            difference against the A0 rule on identical windows (groups resampled)
  part 3    per recording: mean window score per recording, AUC over recordings; and the within-recording
            correlation (ICC(1)) of window scores, which says how much averaging can help
  part 4    DDL (drones only, range 0–249 m per 0.1 s clip): two cues per 1 s window of channel 0, one window every
            4 s, at most 50 per capture:
              hf_share_db     energy share in 6.75–7.25 kHz (the A0 cue), dB
              harmonic_db     harmonic-comb prominence: mean log power at f0..4·f0 for f0 in 120–400 Hz (above the
                              site's 58–105 Hz comb), best f0 minus the median over f0, dB
            Spearman correlation of each cue with range, per drone model and recording day; medians per 50 m bin.
            Descriptive: no detection claim (no negatives).
  decisions A0 "survives without array averaging" if the single-channel rule's interval is above 0.5 in both
            directions. MFCC "adds beyond the rule" in a direction if its paired interval is above zero.
            Part 4: the 7 kHz cue "decays with range" for a model and day if Spearman < 0 with p < 0.05.

CPU, 8 workers (workers import no torch).
Outputs: DroneacharyaData/results/checks/acoustic_single_mic_mfcc.json, ddl_cues_vs_range.json and .png
"""
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from droneacharya import evaluation as E  # noqa: E402
from droneacharya import splits  # noqa: E402
from droneacharya.cache import key  # noqa: E402
from droneacharya.index import read  # noqa: E402
from droneacharya.paths import DATA_ROOT  # noqa: E402

OUT = DATA_ROOT / "results" / "checks"
WORKERS = 8
PLAN = {"svanstrom": (10, None), "uavirbase": (10, 10.0)}       # max windows, seconds read (as run_detectors.py)
DETECTOR_PREFIX = ("band_", "spectral", "peakiness", "modulation")
DDL_STRIDE_S, DDL_MAX_WINDOWS, DDL_CLIP = 4, 50, 9600            # 0.1 s clips at 96 kHz
CODE_HASH = hashlib.sha256((ROOT / "src" / "droneacharya" / "mfcc.py").read_bytes()).hexdigest()[:12]


def harmonic_db(window, rate):
    from scipy.signal import resample_poly, welch
    x = resample_poly(window, 16000, int(rate)).astype(np.float64)
    f, p = welch(x - x.mean(), fs=16000, nperseg=8192)
    logp = 10 * np.log10(p + 1e-30)
    f0 = f[(f >= 120) & (f <= 400)]
    score = np.mean([np.interp(h * f0, f, logp) for h in range(1, 5)], axis=0)
    return float(score.max() - np.median(score))


def _features(window, rate):
    from droneacharya import features as F
    from droneacharya.mfcc import mfcc_features
    return {**F.audio_detector_features(window, rate), **mfcc_features(window, rate),
            "dbfs": float(20 * np.log10(np.sqrt(np.mean(window ** 2)) + 1e-12))}


def _capture_rows(capture_id, max_windows, seconds):
    with threadpool_limits(1):
        from droneacharya import signal
        rate = signal.read(capture_id, count=1).sample_rate_hz
        s = signal.read(capture_id, count=int(seconds * rate) if seconds else None)
        n = int(rate)
        starts = [i for i in range(0, s.samples.shape[1] - n + 1, n)
                  if np.sqrt(np.mean(s.samples.mean(axis=0)[i:i + n] ** 2)) > 1e-6][:max_windows]   # run_detectors' rule
        rows = []
        for k, i in enumerate(starts):
            single = _features(s.samples[0, i:i + n].astype(np.float64), rate)
            per_channel = [single] + [_features(s.samples[c, i:i + n].astype(np.float64), rate)
                                      for c in range(1, s.samples.shape[0])]
            featavg = {f: float(np.mean([p[f] for p in per_channel])) for f in single}
            rows.append({"capture_id": capture_id, "window": k, "single": single, "featavg": featavg})
        return rows


def _ddl_rows(capture_id, clips, label, day):
    with threadpool_limits(1):
        from droneacharya import features as F
        from droneacharya import signal
        rate = signal.read(capture_id, count=1).sample_rate_hz
        n = int(rate)
        rows = []
        for start in range(0, len(clips) * DDL_CLIP - n + 1, DDL_STRIDE_S * n)[:DDL_MAX_WINDOWS]:
            w = signal.read(capture_id, start=start, count=n).samples[0].astype(np.float64)
            if np.sqrt(np.mean(w ** 2)) <= 1e-6:
                continue
            f = F.audio_detector_features(w, rate)
            first = start // DDL_CLIP
            rows.append({"capture_id": capture_id, "model": label, "day": day,
                         "range_m": float(np.mean(clips[first:first + 10])),
                         "hf_share_db": float(10 * np.log10(10 ** f["band_27"] + 10 ** f["band_28"])),
                         "harmonic_db": harmonic_db(w, rate)})
        return rows


def load_windows():
    jobs = [(c, *PLAN[d]) for d in PLAN for c in splits.captures_of(d) if c["label_uas_present"] is not None]
    tag = key(_capture_rows, {"plan": PLAN, "mfcc": CODE_HASH}, list(PLAN))
    path = OUT / f"acoustic_single_mic-{tag}.json"
    if not path.exists():
        out = Parallel(n_jobs=WORKERS)(delayed(_capture_rows)(c["capture_id"], w, s) for c, w, s in jobs)
        path.write_text(json.dumps([r for rows in out for r in rows]), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def load_ddl():
    clips = {}
    for a in read("ddl", "artifacts").to_pylist():
        if a["capture_id"]:
            name = a["member_chain"][0].rsplit("/", 1)[-1]
            clips.setdefault(a["capture_id"], []).append((a["part_index"], int(name[21:24])))
    captures = {c["capture_id"]: c for c in read("ddl", "captures").to_pylist()}
    tag = key(_ddl_rows, {"stride": DDL_STRIDE_S, "max": DDL_MAX_WINDOWS}, ["ddl"])
    path = OUT / f"ddl_cues-{tag}.json"
    if not path.exists():
        out = Parallel(n_jobs=WORKERS)(delayed(_ddl_rows)(cid, [r for _, r in sorted(parts)], captures[cid]["label_class"],
                                                         captures[cid]["group_id"].rsplit("/", 1)[-1][:6])
                                       for cid, parts in sorted(clips.items()))
        path.write_text(json.dumps([r for rows in out for r in rows]), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def table(rows, dataset, variant):
    captures = {c["capture_id"]: c for c in splits.captures_of(dataset)}
    rows = [r for r in rows if r["capture_id"] in captures]
    names = sorted(rows[0][variant])
    return {"X": np.array([[r[variant][k] for k in names] for r in rows]), "names": names,
            "y": np.array([captures[r["capture_id"]]["label_uas_present"] for r in rows]),
            "groups": np.array([captures[r["capture_id"]]["group_id"] for r in rows]),
            "recording": np.array([r["capture_id"] for r in rows])}


def cols(t, prefix):
    idx = [i for i, n in enumerate(t["names"]) if n.startswith(prefix)]
    return t["X"][:, idx], [t["names"][i] for i in idx]


def icc(scores, recording):
    ids = np.unique(recording)
    groups = [scores[recording == r] for r in ids]
    k = np.mean([len(g) for g in groups])
    grand = scores.mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (len(groups) - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (len(scores) - len(groups))
    return float((msb - msw) / (msb + (k - 1) * msw))


def per_recording(scores, t):
    ids = np.unique(t["recording"])
    mean = np.array([scores[t["recording"] == r].mean() for r in ids])
    y = np.array([t["y"][t["recording"] == r][0] for r in ids])
    g = np.array([t["groups"][t["recording"] == r][0] for r in ids])
    r = E.evaluate_external(mean, y, g, "per recording", n_permutations=2000)
    return {"roc_auc": r["roc_auc"], "ci95": r["ci95"], "p": r["permutation"]["p_value"], "recordings": int(len(ids)),
            "icc_window_scores": icc(scores, t["recording"])}


def direction(train, test):
    X_tr, names = cols(train, DETECTOR_PREFIX + ("dbfs",))
    X_te, _ = cols(test, DETECTOR_PREFIX + ("dbfs",))
    aucs = [E.metric(train["y"], X_tr[:, j]) for j in range(len(names))]
    j = int(np.argmax(np.abs(np.array(aucs) - 0.5)))
    sign = 1 if aucs[j] >= 0.5 else -1
    rule = sign * X_te[:, j]
    D_tr, _ = cols(train, DETECTOR_PREFIX)
    D_te, _ = cols(test, DETECTOR_PREFIX)
    detector = E.positive_scores(E.model("gbm").fit(D_tr, train["y"]), D_te)
    M_tr, _ = cols(train, ("mfcc_",))
    M_te, _ = cols(test, ("mfcc_",))
    mfcc_gbm = E.positive_scores(E.model("gbm").fit(M_tr, train["y"]), M_te)
    mfcc_linear = E.positive_scores(E.model("linear").fit(M_tr, train["y"]), M_te)
    y, g = test["y"], test["groups"]
    out = {"rule": {"feature": names[j], "direction": sign, "train_auc": aucs[j],
                    **{k: v for k, v in E.evaluate_external(rule, y, g, "rule").items() if k in ("roc_auc", "ci95", "permutation")}}}
    for name, s in (("detector", detector), ("mfcc_gbm", mfcc_gbm), ("mfcc_linear", mfcc_linear)):
        out[name] = {k: v for k, v in E.evaluate_external(s, y, g, name).items() if k in ("roc_auc", "ci95", "permutation")}
        out[name]["minus_rule"] = E.paired_difference(s, rule, y, g)
    out["per_recording"] = {name: per_recording(s, test) for name, s in
                            (("rule", rule), ("detector", detector), ("mfcc_gbm", mfcc_gbm))}
    return out


def ddl_summary(rows):
    out = {}
    for model in sorted({r["model"] for r in rows}):
        for day in sorted({r["day"] for r in rows if r["model"] == model}):
            sel = [r for r in rows if r["model"] == model and r["day"] == day]
            rng = np.array([r["range_m"] for r in sel])
            entry = {"windows": len(sel), "range_m": [float(rng.min()), float(rng.max())]}
            for cue in ("hf_share_db", "harmonic_db"):
                v = np.array([r[cue] for r in sel])
                rho, p = spearmanr(rng, v)
                bins = {f"{lo}-{lo + 50}": float(np.median(v[(rng >= lo) & (rng < lo + 50)]))
                        for lo in range(0, 250, 50) if ((rng >= lo) & (rng < lo + 50)).sum() >= 5}
                entry[cue] = {"spearman": float(rho), "p": float(p), "median_by_50m": bins,
                              "decays": bool(rho < 0 and p < 0.05)}
            out[f"{model} {day}"] = entry
    return out


def plot_ddl(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cue, title in zip(axes, ("hf_share_db", "harmonic_db"),
                              ("7 kHz energy share (dB)", "Rotor-harmonic prominence (dB)")):
        for key_ in sorted({(r["model"], r["day"]) for r in rows}):
            sel = [r for r in rows if (r["model"], r["day"]) == key_]
            ax.scatter([r["range_m"] for r in sel], [r[cue] for r in sel], s=8, alpha=0.5, label=f"{key_[0]} {key_[1]}")
        ax.set_xlabel("range (m)")
        ax.set_title(title)
    axes[0].legend(fontsize=7)
    fig.suptitle("DDL: acoustic cues vs drone range (channel 0, drones only)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    rows = load_windows()
    ddl = load_ddl()
    print(f"  {len(rows)} windows, {len(ddl)} DDL windows ({(time.time() - start) / 60:.1f} min)", flush=True)
    if "--features-only" in sys.argv:          # extraction fits no model, so it may run before the design commit lands
        return
    reference = json.loads((OUT / "acoustic_single_feature_transfer.json").read_text(encoding="utf-8"))
    result = {}
    for variant in ("single", "featavg"):
        for train, test in (("uavirbase", "svanstrom"), ("svanstrom", "uavirbase")):
            r = direction(table(rows, train, variant), table(rows, test, variant))
            r["channel_averaged_reference"] = {"rule": reference[f"{train} -> {test}"]["rule"]["roc_auc"],
                                               "detector": reference[f"{train} -> {test}"]["detector_auc"]}
            result[f"{variant}: {train} -> {test}"] = r
            print(f"{variant:8} {train} -> {test}: rule {r['rule']['feature']} {r['rule']['roc_auc']:.3f} "
                  f"{[round(v, 2) for v in r['rule']['ci95']]} (averaged {r['channel_averaged_reference']['rule']:.3f}) | "
                  f"detector {r['detector']['roc_auc']:.3f} | mfcc gbm {r['mfcc_gbm']['roc_auc']:.3f} "
                  f"(− rule {r['mfcc_gbm']['minus_rule']['difference']:+.3f} "
                  f"{[round(v, 2) for v in r['mfcc_gbm']['minus_rule']['ci95']]}) linear {r['mfcc_linear']['roc_auc']:.3f} | "
                  f"per recording rule {r['per_recording']['rule']['roc_auc']:.3f} (ICC {r['per_recording']['rule']['icc_window_scores']:.2f})",
                  flush=True)
    single = [result[f"single: {a} -> {b}"]["rule"]["ci95"][0] > 0.5 for a, b in (("uavirbase", "svanstrom"), ("svanstrom", "uavirbase"))]
    result["decisions"] = {"A0_survives_without_array_averaging": bool(all(single)),
                           "mfcc_adds_beyond_rule": {k: bool(v["mfcc_gbm"]["minus_rule"]["ci95"][0] > 0)
                                                     for k, v in result.items() if k.startswith("single")}}
    result["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (OUT / "acoustic_single_mic_mfcc.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("decisions:", result["decisions"], flush=True)

    summary = ddl_summary(ddl)
    (OUT / "ddl_cues_vs_range.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    plot_ddl(ddl, OUT / "ddl_cues_vs_range.png")
    for k, v in summary.items():
        print(f"DDL {k}: {v['windows']} windows, range {v['range_m']} | 7 kHz rho {v['hf_share_db']['spearman']:+.2f} "
              f"(p {v['hf_share_db']['p']:.3g}) | harmonic rho {v['harmonic_db']['spearman']:+.2f} (p {v['harmonic_db']['p']:.3g})")
    print(f"done in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
