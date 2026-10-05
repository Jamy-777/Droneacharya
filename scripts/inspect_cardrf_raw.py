"""Inspect raw CardRF captures directly inside CARDRF.zip, without extracting it.

Step 1 (inventory): count .mat entries per directory so the real tree can be
compared with the author PDF (LOS/NLOS, Train/Test, category, device, mode).

Step 2 (sample): read a few captures per leaf directory into memory and report
Channel_1 metadata, int16 range, clipping, pre/post-trigger RMS and an RMS
profile after the trigger (where the transient ends, where clipping happens).

Usage:
    python scripts/inspect_cardrf_raw.py                  # inventory + 1 capture per leaf directory
    python scripts/inspect_cardrf_raw.py --per-dir 3      # 3 evenly spaced captures per leaf
    python scripts/inspect_cardrf_raw.py --inventory-only
"""
import argparse
import collections
import csv
import io
import math
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import scipy.io

ZIP_PATH = Path(r"D:\Weeeeeeee\DroneacharyaData\raw\cardrf\original\CARDRF.zip")
OUT_DIR = Path(r"D:\Weeeeeeee\DroneacharyaData\interim\cardrf\inspection")

# Rail codes observed in every Processed_CardRF CSV (2026-10-04).
PROCESSED_RAILS = (-32736, 30720)

# Time bands after the trigger, in microseconds.
PROFILE_US = [(0, 1), (1, 5), (5, 10), (10, 50), (50, 125)]


def decode_uint16_text(array):
    return "".join(chr(int(v)) for v in np.asarray(array).reshape(-1))


def h5_value(dataset):
    array = dataset[()]
    if array.dtype == np.uint16:
        return decode_uint16_text(array)
    return array.item() if array.size == 1 else array


def load_capture(raw_bytes):
    buffer = io.BytesIO(raw_bytes)
    if b"MATLAB 7.3" in raw_bytes[:128]:
        import h5py

        with h5py.File(buffer, "r") as mat_file:
            channel = mat_file["Channel_1"]
            fields = {k: h5_value(channel[k]) for k in channel.keys() if k not in ("Data", "XData")}
            data = channel["Data"][()].reshape(-1)
            frame = {k: h5_value(mat_file["Frame"][k]) for k in mat_file["Frame"].keys()} if "Frame" in mat_file else {}
        return "v7.3 (HDF5)", fields, frame, data

    mat = scipy.io.loadmat(buffer, squeeze_me=True, struct_as_record=False)
    channel = mat["Channel_1"]
    fields = {k: getattr(channel, k) for k in channel._fieldnames if k not in ("Data", "XData")}
    data = np.asarray(channel.Data).reshape(-1)
    frame = {k: getattr(mat["Frame"], k) for k in mat["Frame"]._fieldnames} if "Frame" in mat else {}
    return "v5", fields, frame, data


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else float("nan")


def analyse(fields, data):
    result = {
        "dtype": str(data.dtype),
        "n_samples": int(data.size),
        "code_min": int(data.min()),
        "code_max": int(data.max()),
        "pct_at_file_min_or_max": 100 * float(np.mean((data == data.min()) | (data == data.max()))),
        "pct_at_processed_rails": 100 * float(np.mean((data <= PROCESSED_RAILS[0]) | (data >= PROCESSED_RAILS[1]))),
        "quantization_step": math.gcd(*map(int, np.unique(data[:200_000]))),
    }

    x_inc, x_org, y_inc = fields.get("XInc"), fields.get("XOrg"), fields.get("YInc")
    if x_inc and x_org is not None:
        trigger = int(np.clip(round(-float(x_org) / float(x_inc)), 0, data.size))
        result.update({
            "sample_rate_gsps": 1e-9 / float(x_inc),
            "duration_us": data.size * float(x_inc) * 1e6,
            "trigger_index": trigger,
            "rms_pre_trigger_codes": rms(data[:trigger]),
            "rms_post_trigger_codes": rms(data[trigger:]),
        })
        for a_us, b_us in PROFILE_US:
            i0 = trigger + int(a_us * 1e-6 / float(x_inc))
            i1 = min(trigger + int(b_us * 1e-6 / float(x_inc)), data.size)
            segment = data[i0:i1]
            result[f"rms_{a_us}-{b_us}us"] = rms(segment)
            result[f"clip_pct_{a_us}-{b_us}us"] = (
                100 * float(np.mean((segment == data.min()) | (segment == data.max()))) if segment.size else float("nan")
            )
    if y_inc:
        result["volts_per_code"] = float(y_inc)
    return result


def describe_path(name):
    parts = PurePosixPath(name).parts
    upper = [p.upper() for p in parts]
    return {
        "los_nlos": next((p for p in parts if p.upper() in ("LOS", "NLOS")), ""),
        "split": next((p for p in parts if p.upper() in ("TRAIN", "TEST")), ""),
        "path_parts": "/".join(parts[:-1]),
        "has_wifi": any("WIFI" in p for p in upper),
        "has_bluetooth": any("BLUETOOTH" in p for p in upper),
    }


def pick_evenly(entries, n):
    if len(entries) <= n:
        return entries
    return [entries[i] for i in sorted({round(v) for v in np.linspace(0, len(entries) - 1, n)})]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=ZIP_PATH)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--per-dir", type=int, default=1)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()

    try:
        archive = zipfile.ZipFile(args.zip)
    except zipfile.BadZipFile:
        raise SystemExit("Not a readable zip yet — the download is probably incomplete (the file index sits at the end).")

    with archive:
        infos = archive.infolist()
        extensions = collections.Counter(PurePosixPath(i.filename).suffix.lower() for i in infos if not i.is_dir())
        # __MACOSX/ holds zero-byte AppleDouble files that end in .mat but are not captures.
        junk = [i for i in infos if i.filename.startswith("__MACOSX/")]
        mats = sorted((i for i in infos if i.filename.lower().endswith(".mat") and i not in junk),
                      key=lambda i: i.filename)
        print(f"Excluded {len(junk)} __MACOSX entries")
        by_dir = collections.defaultdict(list)
        for info in mats:
            by_dir[str(PurePosixPath(info.filename).parent)].append(info)

        print(f"Archive: {args.zip.name}  entries={len(infos)}  extensions={dict(extensions)}")
        print(f"{'directory':70s} {'files':>6s} {'MB each (median)':>17s}")
        for directory, entries in sorted(by_dir.items()):
            sizes = [e.file_size / 1e6 for e in entries]
            print(f"{directory:70s} {len(entries):6d} {np.median(sizes):17.2f}")
        print(f"Total .mat files: {len(mats)}")

        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "raw_inventory.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["directory", "files", "median_mb"])
            for directory, entries in sorted(by_dir.items()):
                writer.writerow([directory, len(entries), np.median([e.file_size / 1e6 for e in entries])])

        if args.inventory_only:
            return

        rows = []
        for directory, entries in sorted(by_dir.items()):
            for info in pick_evenly(entries, args.per_dir):
                mat_format, fields, frame, data = load_capture(archive.read(info))
                row = {"entry": info.filename, "mat_format": mat_format, **describe_path(info.filename)}
                row.update({f"Channel_1.{k}": v for k, v in fields.items()})
                row["Frame_fields"] = ";".join(f"{k}={v}" for k, v in frame.items())
                row.update(analyse(fields, data))
                rows.append(row)
                print(f"{info.filename}: {row.get('sample_rate_gsps', '?')} GSa/s, "
                      f"YInc={fields.get('YInc')}, YDispRange={fields.get('YDispRange')}, "
                      f"clip%={row['pct_at_file_min_or_max']:.3f}, "
                      f"rms pre/post={row.get('rms_pre_trigger_codes', float('nan')):.0f}/"
                      f"{row.get('rms_post_trigger_codes', float('nan')):.0f}")

    columns = sorted({k for row in rows for k in row}, key=lambda k: (k != "entry", k))
    out_file = args.out / "raw_capture_metadata.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} captures written to {out_file}")


if __name__ == "__main__":
    main()
