"""Read the samples of any indexed capture: read(capture_id, start, count) -> Signal.

Positions are in samples, not seconds: a capture's sample rate is reported
when the release establishes it (None for DroneRF, whose rate is only inferred).
Samples come back as (channels, n): complex64 for IQ, float32 otherwise, in the
units each release stores (`Signal.units`). Reads stay inside the archives;
large captures are read as windows (HDF5 slicing, RAR streaming, memory maps).
DroneRF reads the lossless int16 .npy conversion recorded in
interim/dronerf/npy/TRANSFORM.json.
"""
import json
import struct
import subprocess
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache

import h5py
import numpy as np
import scipy.io

from .index import read as read_table
from .index.acoustic import wav_header
from .index.rar import UNRAR
from .paths import INTERIM, RAW


@dataclass
class Signal:
    capture_id: str
    samples: np.ndarray            # (channels, n)
    sample_rate_hz: float | None
    center_frequency_hz: float | None
    channels: list[str]
    start: int                     # first sample of this window within the capture
    units: str
    meta: dict = field(default_factory=dict)


@lru_cache(maxsize=None)
def _tables(dataset):
    captures = {c["capture_id"]: c for c in read_table(dataset, "captures").to_pylist()}
    parts = {}
    for a in read_table(dataset, "artifacts").to_pylist():
        if a["capture_id"] is not None:
            parts.setdefault(a["capture_id"], []).append(a)
    for items in parts.values():
        items.sort(key=lambda a: (a["part_index"] is None, a["part_index"] or 0, a["channel"] or ""))
    return captures, parts


def capture_length(capture_id):
    """Samples per channel, when the index knows it without reading the data."""
    capture = _tables(capture_id.split("/", 1)[0])[0][capture_id]
    if capture["duration_s"] and capture["sample_rate_hz"]:
        return round(capture["duration_s"] * capture["sample_rate_hz"])
    return None


def read(capture_id, start=0, count=None):
    dataset = capture_id.split("/", 1)[0]
    captures, parts = _tables(dataset)
    if capture_id not in captures:
        raise KeyError(capture_id)
    capture = captures[capture_id]
    samples, channels, units, meta = READERS[dataset](capture, parts.get(capture_id, []), start, count)
    return Signal(capture_id, samples, capture["sample_rate_hz"], capture["center_frequency_hz"], channels, start,
                  units, meta)


def _window(n_total, start, count):
    stop = n_total if count is None else min(n_total, start + count)
    if start < 0 or start >= n_total:
        raise ValueError(f"start {start} outside 0..{n_total - 1}")
    return start, stop


# ---------------------------------------------------------------- RF readers
@lru_cache(maxsize=None)
def _dronerf_paths():
    record = json.loads((INTERIM / "dronerf" / "npy" / "TRANSFORM.json").read_text(encoding="utf-8"))
    return {artifact_id: INTERIM.parent / out["path"] for artifact_id, out in record["outputs"].items()}


def _dronerf(capture, parts, start, count):
    paths = _dronerf_paths()
    arrays = []
    for a in parts:
        if a["artifact_id"] not in paths:
            raise FileNotFoundError(f"{a['artifact_id']} not converted yet: run scripts/convert_dronerf_npy.py")
        arrays.append(np.load(paths[a["artifact_id"]], mmap_mode="r"))
    lo, hi = _window(min(len(x) for x in arrays), start, count)
    return (np.stack([x[lo:hi] for x in arrays]).astype(np.float32), [a["channel"] for a in parts],
            "integer sample values as released (int16 range)", {"transform": "dronerf/csv_to_int16_npy/v1"})


def _unrar_bytes(rar, member, offset, length):
    """`length` bytes of a RAR member starting at `offset`, streamed through UnRAR."""
    proc = subprocess.Popen([UNRAR, "p", "-inul", str(rar), member], stdout=subprocess.PIPE)
    try:
        skipped = 0
        while skipped < offset:
            block = proc.stdout.read(min(16 << 20, offset - skipped))
            if not block:
                raise EOFError(f"{member} ended before byte {offset}")
            skipped += len(block)
        data = proc.stdout.read(length)
    finally:
        proc.kill()
        proc.wait()
    return data


def _rfuav(capture, parts, start, count):
    chunk_samples = [a["size_bytes"] // 8 for a in parts]
    lo, hi = _window(sum(chunk_samples), start, count)
    out, position = [], 0
    for a, n in zip(parts, chunk_samples):
        a_lo, a_hi = max(lo, position), min(hi, position + n)
        if a_lo < a_hi:
            raw = _unrar_bytes(RAW / a["storage_relpath"], a["member_chain"][0], (a_lo - position) * 8, (a_hi - a_lo) * 8)
            out.append(np.frombuffer(raw, "<f4").view(np.complex64))
        position += n
    return np.concatenate(out)[None, :], ["IQ"], "float32 IQ as stored (ScaleFactor semantics unknown)", {}


def _cardrf(capture, parts, start, count):
    (a,) = parts
    with zipfile.ZipFile(RAW / a["storage_relpath"]) as zf, zf.open(a["member_chain"][0]) as fh, h5py.File(fh, "r") as f:
        data = f["Channel_1"]["Data"]
        flat = data.shape.index(max(data.shape))
        lo, hi = _window(data.shape[flat], start, count)
        x = data[(0, slice(lo, hi)) if flat == 1 else (slice(lo, hi), 0)] if data.ndim == 2 else data[lo:hi]
        y_inc = float(f["Channel_1"]["YInc"][()].flatten()[0])
        y_org = float(f["Channel_1"]["YOrg"][()].flatten()[0])
    return (np.asarray(x, np.float32)[None, :], ["Channel_1"], "int16 ADC codes (volts = codes * YInc + YOrg)",
            {"y_inc": y_inc, "y_org": y_org})


def _rma(capture, parts, start, count):
    (a,) = parts
    with zipfile.ZipFile(RAW / a["storage_relpath"]) as zf, zf.open(a["member_chain"][0]) as fh, h5py.File(fh, "r") as f:
        d = f["uhd_samps"]
        lo, hi = _window(d.shape[1], start, count)
        x = d[0, lo:hi]
    return (x["real"] + 1j * x["imag"]).astype(np.complex64)[None, :], ["IQ"], "int16 codes / 32768 (as stored)", {}


def _uavsig(capture, parts, start, count):
    (a,) = parts
    data = scipy.io.loadmat(RAW / a["storage_relpath"], variable_names=["data"])["data"].ravel()
    lo, hi = _window(data.size, start, count)
    return data[lo:hi].astype(np.complex64)[None, :], ["IQ"], "int16 codes / 32767 (as stored)", {}


def _drff_r2(capture, parts, start, count):
    (a,) = parts
    with h5py.File(RAW / a["storage_relpath"], "r") as f:
        lo, hi = _window(f["RF0_I"].shape[0], start, count)
        i, q = f["RF0_I"][lo:hi, 0], f["RF0_Q"][lo:hi, 0]
    return (i + 1j * q).astype(np.complex64)[None, :], ["IQ"], "float32 as stored (gain not recorded)", {}


@lru_cache(maxsize=None)
def _noisy_rf_x_iq():
    path = INTERIM / "noisy_rf" / "extracted" / "dataset.pt"
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo("archive/data/0")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("x_iq storage is compressed; cannot memory-map")
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        header = fh.read(30)
        name_len, extra_len = struct.unpack("<HH", header[26:30])
    offset = info.header_offset + 30 + name_len + extra_len
    return np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(98705, 2, 16384))


def _noisy_rf(capture, parts, start, count):
    row = _noisy_rf_x_iq()[capture["stored_index"]]
    lo, hi = _window(row.shape[1], start, count)
    return ((row[0, lo:hi] + 1j * row[1, lo:hi]).astype(np.complex64)[None, :], ["IQ"],
            "float32, unit-power normalised before mixing (authors)", {"row": capture["stored_index"]})


# ---------------------------------------------------------------- audio readers
def _decode_pcm(raw, info, lo=0, hi=None):
    """Frames lo..hi of a WAV as float32 (channels, n), PCM scaled to [-1, 1)."""
    data_start = raw.find(b"data") + 8
    frame = info["channels"] * info["bits"] // 8
    n_frames = info["data_bytes"] // frame
    hi = n_frames if hi is None else min(hi, n_frames)
    chunk = raw[data_start + lo * frame:data_start + hi * frame]
    bits = info["bits"]
    if info["format"] == "float":
        x = np.frombuffer(chunk, "<f4" if bits == 32 else "<f8").astype(np.float32)
    elif bits == 16:
        x = np.frombuffer(chunk, "<i2").astype(np.float32) / 32768
    elif bits == 32:
        x = np.frombuffer(chunk, "<i4").astype(np.float32) / 2 ** 31
    elif bits == 24:
        b = np.frombuffer(chunk, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = (np.where(v >= 1 << 23, v - (1 << 24), v) / 2 ** 23).astype(np.float32)
    else:
        raise ValueError(f"unsupported {bits}-bit {info['format']}")
    return x.reshape(-1, info["channels"]).T


def _wav_bytes(a):
    if a["member_chain"]:
        with zipfile.ZipFile(RAW / a["storage_relpath"]) as zf:
            return zf.read(a["member_chain"][-1])
    return (RAW / a["storage_relpath"]).read_bytes()


def _audio_single(capture, parts, start, count, units="full scale (uncalibrated)"):
    (a,) = parts
    raw = _wav_bytes(a)
    info = wav_header(raw[:4096])
    lo, hi = _window(info["data_bytes"] // (info["channels"] * info["bits"] // 8), start, count)
    x = _decode_pcm(raw, info, lo, hi)
    return x, capture["channels"] or [str(i + 1) for i in range(x.shape[0])], units, {}


def _audio_parts(capture, parts, start, count):
    """Time parts (DDL clips) concatenated; only the clips the window needs are read."""
    infos = []
    with zipfile.ZipFile(RAW / parts[0]["storage_relpath"]) as zf:
        heads = {a["artifact_id"]: zf.open(a["member_chain"][-1]).read(4096) for a in parts}
        for a in parts:
            info = wav_header(heads[a["artifact_id"]])
            infos.append(info["data_bytes"] // (info["channels"] * info["bits"] // 8))
        lo, hi = _window(sum(infos), start, count)
        out, position = [], 0
        for a, n in zip(parts, infos):
            a_lo, a_hi = max(lo, position), min(hi, position + n)
            if a_lo < a_hi:
                raw = zf.read(a["member_chain"][-1])
                out.append(_decode_pcm(raw, wav_header(raw[:4096]), a_lo - position, a_hi - position))
            position += n
    return np.concatenate(out, axis=1), capture["channels"], "full scale (uncalibrated)", {}


def _dronenoise(capture, parts, start, count):
    if len(parts) == 1:
        return _audio_single(capture, parts, start, count, units="pascal for M6–M9; M1–M5 calibration uncertain (±5 dB)")
    decoded = []
    for a in parts:
        raw = _wav_bytes(a)
        decoded.append(_decode_pcm(raw, wav_header(raw[:4096]))[0])
    n = min(len(x) for x in decoded)  # microphones of one event can differ by a few samples
    lo, hi = _window(n, start, count)
    return (np.stack([x[lo:hi] for x in decoded]), [a["channel"] for a in parts],
            "pascal for M6–M9; M1–M5 calibration uncertain (±5 dB)", {"trimmed_to_shortest_microphone": n})


READERS = {
    "dronerf": _dronerf, "rfuav": _rfuav, "cardrf": _cardrf, "rma": _rma, "uavsig": _uavsig, "drff_r2": _drff_r2,
    "noisy_rf": _noisy_rf, "esc50": _audio_single, "svanstrom": _audio_single, "uavirbase": _audio_single,
    "miesikowska_uav": _audio_single, "ddl": _audio_parts, "dronenoise": _dronenoise,
}
