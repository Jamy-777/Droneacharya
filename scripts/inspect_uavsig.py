"""Inspect downloaded UAVSig MAT files without modifying them.

For each capture: MAT format, stored variables, how IQ is stored, sample
count, dropped-sample gap signature, the WHIRLS label table, and power
inside vs outside the labelled transmissions. Prints a per-file report
and writes a JSON summary to interim/uavsig/inspection/.

Usage:
    python scripts/inspect_uavsig.py                 # every .mat under DATA_ROOT
    python scripts/inspect_uavsig.py path/to/a.mat   # specific files
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io

DATA_ROOT = Path(r"D:\Weeeeeeee\DroneacharyaData\raw\uavsig\original")
OUT_DIR = Path(r"D:\Weeeeeeee\DroneacharyaData\interim\uavsig\inspection")

# From the dataset's metadata.json; the sample count below checks it.
FS_HZ = 50e6
EXPECTED_SAMPLES = 50_000_000

LABEL_FIELDS = ["start", "end_", "bw", "fc", "id"]

# A run of at least this many exact-zero complex samples is reported as a
# candidate dropped-sample gap. Real noise is never exactly zero for long.
MIN_ZERO_RUN = 16

N_WINDOWS = 1000


def mat_format(path):
    with open(path, "rb") as fh:
        header = fh.read(128)
    if b"MATLAB 7.3" in header:
        return "v7.3 (HDF5)"
    return header[:116].decode("latin-1").split(",")[0].strip()


def load_v5(path):
    variables = {name: {"shape": list(shape), "class": cls} for name, shape, cls in scipy.io.whosmat(path)}
    wanted = [name for name in ["data", *LABEL_FIELDS] if name in variables]
    loaded = scipy.io.loadmat(path, variable_names=wanted)
    return variables, loaded


def load_v73(path):
    import h5py

    variables, loaded = {}, {}
    with h5py.File(path, "r") as mat_file:
        for name, obj in mat_file.items():
            if not isinstance(obj, h5py.Dataset):
                variables[name] = {"type": type(obj).__name__}
                continue
            # MATLAB stores arrays transposed in HDF5.
            variables[name] = {"shape": list(obj.shape[::-1]), "dtype": str(obj.dtype)}
            if name in ["data", *LABEL_FIELDS]:
                array = obj[()]
                if array.dtype.names and {"real", "imag"} <= set(array.dtype.names):
                    array = array["real"] + 1j * array["imag"]
                loaded[name] = array.T
    return variables, loaded


def zero_runs(x, min_run):
    is_zero = x == 0
    if not is_zero.any():
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    edges = np.diff(np.concatenate(([0], is_zero.view(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    lengths = np.flatnonzero(edges == -1) - starts
    keep = lengths >= min_run
    return starts[keep], lengths[keep]


def label_table(loaded, n_samples):
    if not all(name in loaded for name in LABEL_FIELDS):
        return {"present": False, "missing_fields": [n for n in LABEL_FIELDS if n not in loaded]}

    start, end, bw, fc, ids = (np.asarray(loaded[name]).reshape(-1) for name in LABEL_FIELDS)
    table = {"present": True, "count": int(start.size)}
    if start.size == 0:
        return table

    durations_us = (end - start) / FS_HZ * 1e6
    table.update({
        "start_dtype": str(start.dtype),
        "start_min": float(start.min()),
        "end_max": float(end.max()),
        "end_le_n_samples": bool(end.max() <= n_samples),
        "all_end_gt_start": bool(np.all(end > start)),
        "duration_us_min_median_max": [float(np.min(durations_us)), float(np.median(durations_us)), float(np.max(durations_us))],
        "bw_mhz_min_max": [float(bw.min() / 1e6), float(bw.max() / 1e6)],
        # Baseband offsets would fall within +/-25 MHz; absolute RF would be ~2400 MHz.
        "fc_mhz_min_max": [float(fc.min() / 1e6), float(fc.max() / 1e6)],
        "id_values": {str(k): int(v) for k, v in zip(*np.unique(ids, return_counts=True))},
    })
    return table


def inspect(path):
    report = {"file": path.name, "category": path.parent.parent.name, "mat_format": mat_format(path)}
    variables, loaded = load_v73(path) if report["mat_format"].startswith("v7.3") else load_v5(path)
    report["variables"] = variables

    if "data" not in loaded:
        report["error"] = "no 'data' variable"
        return report

    data = np.asarray(loaded["data"])
    report["data_stored"] = {"dtype": str(data.dtype), "shape": list(data.shape), "is_complex": bool(np.iscomplexobj(data))}

    # visualize.m reads data(1, :): one capture per row.
    x = data[0] if data.ndim == 2 and data.shape[0] < data.shape[1] else data.reshape(-1)
    n = int(x.size)
    report["samples"] = {
        "count": n,
        "matches_expected": n == EXPECTED_SAMPLES,
        "duration_s_at_fs": n / FS_HZ,
        "twice_expected_maybe_interleaved": n == 2 * EXPECTED_SAMPLES,
    }

    starts, lengths = zero_runs(x, MIN_ZERO_RUN)
    report["gap_signature"] = {
        "nan_count": int(np.isnan(x).sum()),
        "exact_zero_samples": int((x == 0).sum()),
        f"zero_runs_ge_{MIN_ZERO_RUN}": int(starts.size),
        "zero_run_samples_total": int(lengths.sum()),
        "longest_zero_run": int(lengths.max()) if lengths.size else 0,
        "first_zero_run_starts": [int(s) for s in starts[:10]],
    }

    power = (np.abs(x) ** 2).astype(np.float64)
    usable = n - n % N_WINDOWS
    window_rms = np.sqrt(power[:usable].reshape(N_WINDOWS, -1).mean(axis=1))
    report["power"] = {
        "mean_power": float(power.mean()),
        "window_rms_min_median_p90_max": [float(np.min(window_rms)), float(np.median(window_rms)),
                                          float(np.percentile(window_rms, 90)), float(np.max(window_rms))],
    }

    labels = label_table(loaded, n)
    report["labels"] = labels
    if labels.get("count"):
        active = np.zeros(n, dtype=bool)
        for s, e in zip(np.asarray(loaded["start"]).reshape(-1), np.asarray(loaded["end_"]).reshape(-1)):
            active[max(int(s), 0):min(int(e), n)] = True
        inside, outside = power[active], power[~active]
        report["labels"]["time_occupancy_pct"] = float(100 * active.mean())
        if inside.size and outside.size and outside.mean() > 0:
            report["labels"]["inside_vs_outside_power_db"] = float(10 * np.log10(inside.mean() / outside.mean()))

    return report


def print_report(report):
    print("=" * 80)
    print(f"{report['category']}/{report['file']}  [{report['mat_format']}]")
    for key in ["variables", "data_stored", "samples", "gap_signature", "power", "labels", "error"]:
        if key in report:
            print(f"  {key}: {json.dumps(report[key], default=str)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    files = args.files or sorted(DATA_ROOT.rglob("*.mat"))
    if not files:
        raise SystemExit(f"No .mat files found under {DATA_ROOT}")

    reports = []
    for path in files:
        report = inspect(path)
        print_report(report)
        reports.append(report)

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / "uavsig_inspection_summary.json"
    out_file.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print(f"\nSummary written to {out_file}")


if __name__ == "__main__":
    main()
