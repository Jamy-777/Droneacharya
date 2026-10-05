"""Inspect downloaded RMA (Royal Military Academy / KU Leuven) MAT captures without modifying them.

For each capture: MAT header (creation date), stored variables, IQ storage,
sample count and duration at 100 MS/s, int16 quantization, burstiness
(1 ms window RMS), occupied bandwidth, and exact-zero samples. For consecutive
files of the same archive (e.g. mini2_0 / mini2_1) it also compares the end of
one file with the start of the next, to test whether they are one continuous
recording. Writes a JSON summary to interim/rma/inspection/.

Usage:
    python scripts/inspect_rma.py                 # every .mat under DATA_ROOT
    python scripts/inspect_rma.py path/to/a.mat   # specific files
"""
import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np

DATA_ROOT = Path(r"D:\Weeeeeeee\DroneacharyaData\raw\rma\original")
OUT_DIR = Path(r"D:\Weeeeeeee\DroneacharyaData\interim\rma\inspection")

FS_HZ = 100e6          # author README
CENTER_HZ = 2.44e9     # author README
WINDOW_S = 1e-3        # 1 ms windows for burstiness
NFFT = 1024
ACTIVE_DB_ABOVE_NOISE = 10  # a window counts as active if its power is this far above the quietest windows


def mat_header(path):
    with open(path, "rb") as fh:
        return fh.read(116).decode("latin-1").strip()


def load_iq(path):
    with h5py.File(path, "r") as mat_file:
        variables = {name: {"shape": list(obj.shape[::-1]), "dtype": str(obj.dtype)}
                     for name, obj in mat_file.items() if isinstance(obj, h5py.Dataset)}
        name = "uhd_samps" if "uhd_samps" in mat_file else next(iter(variables))
        array = mat_file[name][()]
    if array.dtype.names and {"real", "imag"} <= set(array.dtype.names):
        array = array["real"] + 1j * array["imag"]
    return variables, name, array.reshape(-1)


def occupied_band(x):
    """Frequency span (absolute GHz) where the average spectrum is >10 dB above its median."""
    n = (x.size // NFFT) * NFFT
    frames = x[:n].reshape(-1, NFFT)[:: max(1, n // NFFT // 2000)]  # up to ~2000 frames
    spectrum = np.mean(np.abs(np.fft.fftshift(np.fft.fft(frames * np.hanning(NFFT), axis=1), axes=1)) ** 2, axis=0)
    db = 10 * np.log10(spectrum + 1e-30)
    occupied = np.flatnonzero(db > np.median(db) + 10)
    freqs = CENTER_HZ + np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / FS_HZ))
    if occupied.size == 0:
        return None
    return {"low_ghz": float(freqs[occupied[0]] / 1e9), "high_ghz": float(freqs[occupied[-1]] / 1e9),
            "occupied_bins_pct": float(100 * occupied.size / NFFT)}


def inspect(path):
    variables, name, x = load_iq(path)
    n = x.size
    window = int(WINDOW_S * FS_HZ)
    usable = n - n % window
    power = np.abs(x[:usable]) ** 2
    window_power = power.reshape(-1, window).mean(axis=1)
    noise_floor = np.percentile(window_power, 5)
    active = window_power > noise_floor * 10 ** (ACTIVE_DB_ABOVE_NOISE / 10)

    components = np.concatenate([x[:200_000].real, x[:200_000].imag]) * 32768
    return {
        "archive": path.parent.name,
        "file": path.name,
        "mat_header": mat_header(path),
        "variables": variables,
        "variable_used": name,
        "samples": int(n),
        "duration_ms": n / FS_HZ * 1e3,
        "int16_scaled": bool(np.allclose(components, np.round(components), atol=1e-6)),
        "exact_zero_samples": int((x == 0).sum()),
        "window_rms_min_median_p90_max": [float(v) for v in np.sqrt(
            [window_power.min(), np.median(window_power), np.percentile(window_power, 90), window_power.max()])],
        "active_window_pct": float(100 * active.mean()),
        "active_vs_noise_db": float(10 * np.log10(window_power[active].mean() / noise_floor)) if active.any() else None,
        "occupied_band": occupied_band(x),
        "_head": x[:4096],
        "_tail": x[-4096:],
    }


def continuity(earlier, later):
    """Is `later` a direct continuation of `earlier`?

    MAT creation times are the main evidence. The boundary jump is weak evidence
    on its own (white noise has large sample-to-sample steps anyway).
    """
    tail, head = earlier["_tail"], later["_head"]
    overlap = any(np.array_equal(tail[-k:], head[:k]) for k in (1, 16, 256))
    jump = float(np.abs(head[0] - tail[-1]))
    typical_step = float(np.median(np.abs(np.diff(tail))))
    created = lambda r: r["mat_header"].split("Created on:")[-1].split("HDF5")[0].strip()
    return {"pair": f"{earlier['file']} -> {later['file']}",
            "created": [created(earlier), created(later)],
            "boundary_samples_overlap": overlap,
            "boundary_jump_vs_typical_step": jump / typical_step if typical_step else None}


def file_index(path):
    match = re.search(r"_(\d+)\.mat$", path.name)
    return int(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    files = args.files or sorted(DATA_ROOT.rglob("*.mat"))
    if not files:
        raise SystemExit(f"No .mat files found under {DATA_ROOT}")

    reports = {}
    for path in files:
        report = inspect(path)
        reports[path] = report
        shown = {k: v for k, v in report.items() if not k.startswith("_") and k != "variables"}
        print(json.dumps(shown, default=str))

    pairs = []
    for path, report in reports.items():
        nxt = path.with_name(re.sub(r"_(\d+)\.mat$", lambda m: f"_{int(m.group(1)) + 1}.mat", path.name))
        if nxt in reports:
            pairs.append(continuity(report, reports[nxt]))
    for pair in pairs:
        print("continuity:", json.dumps(pair))

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / "rma_inspection_summary.json"
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in reports.values()]
    out_file.write_text(json.dumps({"files": clean, "continuity": pairs}, indent=2, default=str), encoding="utf-8")
    print(f"\nSummary written to {out_file}")


if __name__ == "__main__":
    main()
