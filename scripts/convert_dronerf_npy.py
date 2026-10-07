"""Convert DroneRF CSVs (text) to int16 .npy in interim/, losslessly, with a transformation record.

Each CSV member is streamed out of its RAR (the RAR is first copied out of the
release zip to a temporary file) and parsed. The conversion stops if any value
is not an integer or does not fit int16, so the .npy holds exactly the CSV's
numbers. Outputs: interim/dronerf/npy/<BUI>/<BUI><band>_<segment>.npy, and
interim/dronerf/npy/TRANSFORM.json recording the inputs, settings and checks.
Resumable: finished outputs listed in TRANSFORM.json are skipped.

Usage:
    python scripts/convert_dronerf_npy.py
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya.index import read  # noqa: E402
from droneacharya.index.dronerf import ZIP_RELPATH, _unlink_when_released  # noqa: E402
from droneacharya.index.rar import UNRAR  # noqa: E402
from droneacharya.paths import INTERIM, RAW  # noqa: E402

OUT = INTERIM / "dronerf" / "npy"
RECORD = OUT / "TRANSFORM.json"


def main():
    artifacts = read("dronerf", "artifacts").to_pylist()
    record = json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.exists() else {
        "transform_id": "dronerf/csv_to_int16_npy/v1",
        "script": "scripts/convert_dronerf_npy.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": ZIP_RELPATH,
        "settings": {"dtype": "int16", "layout": "1-D array, one CSV row", "lossless": True},
        "checks": "every value is an integer and within int16; length recorded",
        "outputs": {},
    }
    done = record["outputs"]
    by_rar = {}
    for a in artifacts:
        by_rar.setdefault(a["member_chain"][0], []).append(a)
    tmp_dir = INTERIM / "dronerf" / "_tmp"
    start = time.time()
    with zipfile.ZipFile(RAW / ZIP_RELPATH) as zf:
        for rar_member, items in sorted(by_rar.items()):
            todo = [a for a in items if a["artifact_id"] not in done]
            if not todo:
                continue
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp = tmp_dir / PurePosixPath(rar_member).name
            try:
                with zf.open(rar_member) as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst, 16 << 20)
                for a in todo:
                    raw = subprocess.run([UNRAR, "p", "-inul", str(tmp), a["member_chain"][1]],
                                         capture_output=True, check=True).stdout
                    values = np.array(raw.rstrip(b",\r\n ").split(b","), dtype=np.float64)
                    if not np.all(values == np.round(values)) or values.min() < -32768 or values.max() > 32767:
                        sys.exit(f"{a['artifact_id']}: values are not int16 integers; conversion would not be lossless")
                    data = values.astype(np.int16)
                    name = PurePosixPath(a["member_chain"][1].replace("\\", "/")).name.removesuffix(".csv") + ".npy"
                    bui = a["capture_id"].split("/")[1]
                    out = OUT / bui / name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    part = out.with_name(out.name + ".part")
                    with open(part, "wb") as fh:
                        np.save(fh, data)
                    part.replace(out)
                    done[a["artifact_id"]] = {
                        "path": out.relative_to(INTERIM.parent).as_posix(),
                        "source_crc32": a["crc32"], "samples": int(data.size), "dtype": "int16",
                        "min": int(data.min()), "max": int(data.max()),
                    }
                    record["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    RECORD.write_text(json.dumps(record, indent=1), encoding="utf-8")
            finally:
                _unlink_when_released(tmp)
            print(f"{rar_member}: {len(done)}/{len(artifacts)} done, {(time.time() - start) / 60:.1f} min", flush=True)
    print(f"Converted {len(done)} of {len(artifacts)} CSVs; record {RECORD}")


if __name__ == "__main__":
    main()
