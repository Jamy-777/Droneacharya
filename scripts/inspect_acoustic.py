"""Audit the acoustic datasets directly inside their archives (no extraction).

For every WAV: header facts (sample rate, channels, sample format, bits, duration).
For a stratified sample: RMS level (dBFS), peak, clipping fraction, DC offset and
whether the channels are identical copies. Writes one CSV per dataset to
interim/<dataset>/ and prints a summary.

Usage:
    python scripts/inspect_acoustic.py                 # all datasets
    python scripts/inspect_acoustic.py ddl svanstrom   # selected
"""
import argparse
import io
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(r"D:\Weeeeeeee\DroneacharyaData\raw")
INTERIM = Path(r"D:\Weeeeeeee\DroneacharyaData\interim")
RNG = np.random.default_rng(0)

FORMATS = {1: "pcm", 3: "float", 0xFFFE: "extensible"}


def wav_header(raw):
    """Parse a RIFF/WAVE header from the first bytes of a file."""
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return {"valid": False}
    pos, info = 12, {"valid": True}
    while pos + 8 <= len(raw):
        chunk, size = raw[pos:pos + 4], struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = raw[pos + 8:pos + 8 + min(size, 40)]
        if chunk == b"fmt ":
            fmt, ch, sr = struct.unpack("<HHI", body[:8])
            bits = struct.unpack("<H", body[14:16])[0]
            if fmt == 0xFFFE and len(body) >= 26:
                fmt = struct.unpack("<H", body[24:26])[0]
            info.update(format=FORMATS.get(fmt, str(fmt)), channels=ch, sample_rate=sr, bits=bits)
        elif chunk == b"data":
            info["data_bytes"] = size
            break
        pos += 8 + size + (size & 1)
    if "data_bytes" in info and info.get("channels") and info.get("bits"):
        frame = info["channels"] * info["bits"] // 8
        info["duration_s"] = info["data_bytes"] / frame / info["sample_rate"]
    return info


def decode(raw, info):
    """Return float samples (frames x channels) scaled to [-1, 1]."""
    start = raw.find(b"data") + 8
    data = raw[start:start + info["data_bytes"]]
    bits, ch = info["bits"], info["channels"]
    if info["format"] == "float":
        x = np.frombuffer(data, dtype="<f4" if bits == 32 else "<f8")
    elif bits == 16:
        x = np.frombuffer(data, dtype="<i2") / 32768.0
    elif bits == 24:
        b = np.frombuffer(data[: len(data) // 3 * 3], dtype=np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        x = np.where(v >= 1 << 23, v - (1 << 24), v) / float(1 << 23)
    elif bits == 32:
        x = np.frombuffer(data, dtype="<i4") / float(1 << 31)
    else:
        raise ValueError(f"unsupported {bits}-bit {info['format']}")
    return x[: len(x) // ch * ch].reshape(-1, ch).astype(np.float64)


def level_stats(x):
    rms = np.sqrt(np.mean(x ** 2, axis=0))
    peak = np.max(np.abs(x), axis=0)
    out = {
        "rms_dbfs": float(20 * np.log10(np.mean(rms) + 1e-12)),
        "peak": float(peak.max()),
        "clip_pct": float(100 * np.mean(np.abs(x) >= 0.999)),
        "dc_offset": float(np.mean(x)),
        "channel_rms_spread_db": float(20 * np.log10((rms.max() + 1e-12) / (rms.min() + 1e-12))),
    }
    if x.shape[1] > 1:
        out["identical_channels"] = bool(np.allclose(x[:, 0], x[:, 1]))
    return out


def scan_zip(zip_path, select, sample_fn, head_bytes=4096):
    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        infos = [i for i in archive.infolist() if i.filename.lower().endswith(".wav") and select(i.filename)]
        sampled = set(sample_fn([i.filename for i in infos]))
        for info in infos:
            with archive.open(info) as fh:
                head = fh.read(head_bytes)
            row = {"file": info.filename, "bytes": info.file_size, **wav_header(head)}
            if info.filename in sampled and row.get("valid") and row.get("data_bytes"):
                raw = archive.read(info)
                row.update(level_stats(decode(raw, row)))
            rows.append(row)
    return pd.DataFrame(rows)


def scan_folder(folder):
    rows = []
    for path in sorted(folder.glob("*.wav")):
        raw = path.read_bytes()
        info = wav_header(raw[:4096])
        rows.append({"file": path.name, "bytes": len(raw), **info, **level_stats(decode(raw, info))})
    return pd.DataFrame(rows)


def sample_every(names, n):
    names = sorted(names)
    return names if len(names) <= n else [names[i] for i in RNG.choice(len(names), n, replace=False)]


def ddl():
    def per_folder(names, n=60):
        folders = {}
        for name in names:
            folders.setdefault(name.rsplit("/", 1)[0], []).append(name)
        return [f for group in folders.values() for f in sample_every(group, n)]

    # Up to 60 clips per recording session; empty 44-byte clips are recorded but not decoded.
    return scan_zip(RAW / "ddl/original/MLSP_2022_Real_Data.zip", lambda f: True, per_folder)


def uavirbase():
    return scan_zip(RAW / "uavirbase/original/Microphone_array.zip", lambda f: True,
                    lambda names: sample_every(names, 24), head_bytes=8192)


def miesikowska():
    frames = []
    for path in sorted((RAW / "miesikowska_uav/original").glob("*.zip")):
        frames.append(scan_zip(path, lambda f: "OLYMPUS" in f.upper(), lambda names: sample_every(names, 30)))
    return pd.concat(frames, ignore_index=True)


def dronenoise():
    return scan_zip(RAW / "dronenoise/original/22133411.zip", lambda f: True, lambda names: names)


def svanstrom():
    return scan_folder(RAW / "svanstrom/original/Audio")


def esc50():
    return scan_zip(RAW / "esc50/original/ESC-50-master.zip", lambda f: "/audio/" in f, lambda names: sample_every(names, 200))


DATASETS = {"ddl": ddl, "uavirbase": uavirbase, "miesikowska_uav": miesikowska,
            "dronenoise": dronenoise, "svanstrom": svanstrom, "esc50": esc50}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=list(DATASETS))
    args = parser.parse_args()
    for name in args.datasets:
        df = DATASETS[name]()
        out = INTERIM / name / f"{name}_audio_audit.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print("=" * 90, f"\n{name}: {len(df)} wav files -> {out}")
        cols = [c for c in ["valid", "format", "bits", "channels", "sample_rate"] if c in df]
        print(df.groupby(cols, dropna=False).size().to_string())
        if "duration_s" in df:
            print("duration s: min %.2f  median %.2f  max %.2f  total %.1f min" % (
                df.duration_s.min(), df.duration_s.median(), df.duration_s.max(), df.duration_s.sum() / 60))
        if "rms_dbfs" in df:
            measured = df.dropna(subset=["rms_dbfs"])
            print(f"level sample n={len(measured)}: RMS dBFS median {measured.rms_dbfs.median():.1f} "
                  f"[{measured.rms_dbfs.min():.1f}, {measured.rms_dbfs.max():.1f}], "
                  f"clip% max {measured.clip_pct.max():.3f}, files with any clipping {int((measured.clip_pct > 0).sum())}")
            if "identical_channels" in measured:
                print("identical channels:", measured.identical_channels.value_counts().to_dict())
        sys.stdout.flush()


if __name__ == "__main__":
    main()
