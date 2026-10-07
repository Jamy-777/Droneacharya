"""Extract one member of a zip under raw/ into interim/, verified by the zip's CRC-32.

The member is streamed to <dest>.part; zipfile checks the CRC when the stream
ends, and only then is the file renamed into place. Raw data is never modified.

Usage:
    python scripts/extract_member.py raw/noisy_rf/original/archive.zip dataset.pt interim/noisy_rf/extracted
"""
import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from droneacharya.paths import DATA_ROOT  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", help="zip path relative to the data root")
    parser.add_argument("member")
    parser.add_argument("dest_dir", help="destination folder relative to the data root (must be under interim/)")
    args = parser.parse_args()

    dest_dir = DATA_ROOT / args.dest_dir
    if "interim" not in dest_dir.relative_to(DATA_ROOT).parts[:1]:
        sys.exit("destination must be under interim/")
    dest = dest_dir / Path(args.member).name
    if dest.exists():
        sys.exit(f"{dest} already exists")
    dest_dir.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    start = time.time()
    with zipfile.ZipFile(DATA_ROOT / args.zip) as zf:
        info = zf.getinfo(args.member)
        if shutil.disk_usage(dest_dir).free < info.file_size * 1.05:
            sys.exit(f"not enough free space for {info.file_size / 1e9:.1f} GB")
        with zf.open(info) as src, open(part, "wb") as dst:  # BadZipFile on CRC mismatch
            shutil.copyfileobj(src, dst, 16 << 20)
    if part.stat().st_size != info.file_size:
        sys.exit("size mismatch; .part kept")
    for _ in range(24):
        try:
            part.replace(dest)
            break
        except PermissionError:
            time.sleep(5)
    print(f"{dest.relative_to(DATA_ROOT).as_posix()}: {info.file_size:,} bytes, CRC-32 {info.CRC:08x} verified, "
          f"{(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
