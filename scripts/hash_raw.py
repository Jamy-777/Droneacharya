"""SHA-256 inventory of every stored file under DroneacharyaData/raw.

Writes one row per file to manifests/raw_sha256.csv:
    relpath, size_bytes, mtime_ns, sha256
relpath is relative to raw/ with forward slashes, so the manifest never holds
a drive path. Rows are appended as each file finishes, so an interrupted run
resumes: a file is skipped when its relpath, size and mtime match a row
already in the manifest.

At the end a summary goes to manifests/raw_sha256_summary.json: totals per
dataset folder, groups of files with identical content (duplicates), and
files that share a size but differ in content. Only that short summary is
printed.

Usage:
    python scripts/hash_raw.py [--data-root D:\\Weeeeeeee\\DroneacharyaData] [--workers 4]
"""
import argparse
import csv
import hashlib
import json
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CHUNK_BYTES = 16 * 1024 * 1024
FIELDS = ["relpath", "size_bytes", "mtime_ns", "sha256"]


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def load_existing(manifest):
    if not manifest.exists():
        return {}
    with open(manifest, newline="", encoding="utf-8") as fh:
        return {row["relpath"]: row for row in csv.DictReader(fh)}


def summarise(rows, summary_path):
    per_dataset = defaultdict(lambda: {"files": 0, "bytes": 0})
    by_hash = defaultdict(list)
    by_size = defaultdict(set)
    for row in rows:
        dataset = row["relpath"].split("/", 1)[0]
        size = int(row["size_bytes"])
        per_dataset[dataset]["files"] += 1
        per_dataset[dataset]["bytes"] += size
        by_hash[row["sha256"]].append(row["relpath"])
        by_size[size].add(row["sha256"])

    duplicates = [
        {"sha256": digest, "size_bytes": int(next(r["size_bytes"] for r in rows if r["sha256"] == digest)),
         "relpaths": sorted(paths)}
        for digest, paths in by_hash.items() if len(paths) > 1
    ]
    duplicates.sort(key=lambda group: -group["size_bytes"])
    same_size_different_content = sorted(size for size, digests in by_size.items() if len(digests) > 1)

    summary = {
        "files": len(rows),
        "bytes": sum(int(r["size_bytes"]) for r in rows),
        "per_dataset": dict(sorted(per_dataset.items())),
        "duplicate_groups": duplicates,
        "duplicate_bytes_reclaimable": sum(g["size_bytes"] * (len(g["relpaths"]) - 1) for g in duplicates),
        "sizes_shared_by_different_content": len(same_size_different_content),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\Weeeeeeee\DroneacharyaData"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    raw = args.data_root / "raw"
    manifest = args.data_root / "manifests" / "raw_sha256.csv"
    summary_path = args.data_root / "manifests" / "raw_sha256_summary.json"
    progress_path = args.data_root / "manifests" / "raw_sha256_progress.txt"

    existing = load_existing(manifest)
    files = sorted(p for p in raw.rglob("*") if p.is_file())
    todo = []
    for path in files:
        relpath = path.relative_to(raw).as_posix()
        stat = path.stat()
        row = existing.get(relpath)
        if row and int(row["size_bytes"]) == stat.st_size and int(row["mtime_ns"]) == stat.st_mtime_ns:
            continue
        todo.append((path, relpath, stat.st_size, stat.st_mtime_ns))

    total_bytes = sum(item[2] for item in todo)
    print(f"{len(files)} files under raw/; {len(todo)} to hash ({total_bytes / 1e9:.1f} GB); "
          f"{len(files) - len(todo)} already in manifest.", flush=True)

    # Rows for files that changed since an earlier run are rewritten; others are kept.
    stale = {item[1] for item in todo} & existing.keys()
    if stale:
        kept = [row for key, row in existing.items() if key not in stale]
        with open(manifest, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, FIELDS)
            writer.writeheader()
            writer.writerows(kept)

    new_file = not manifest.exists()
    lock = threading.Lock()
    done_bytes, start = 0, time.time()
    failures = []
    with open(manifest, "a", newline="", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
        writer = csv.DictWriter(fh, FIELDS)
        if new_file:
            writer.writeheader()
        futures = {pool.submit(sha256_of, item[0]): item for item in todo}
        for future in as_completed(futures):
            path, relpath, size, mtime_ns = futures[future]
            try:
                digest = future.result()
            except OSError as error:
                failures.append(f"{relpath}: {error}")
                continue
            with lock:
                writer.writerow({"relpath": relpath, "size_bytes": size, "mtime_ns": mtime_ns, "sha256": digest})
                fh.flush()
                done_bytes += size
                elapsed = time.time() - start
                rate = done_bytes / elapsed / 1e6 if elapsed else 0
                progress_path.write_text(
                    f"{done_bytes / 1e9:.1f} / {total_bytes / 1e9:.1f} GB, {rate:.0f} MB/s, "
                    f"elapsed {elapsed / 60:.1f} min, last {relpath}\n", encoding="utf-8")

    rows = list(load_existing(manifest).values())
    summary = summarise(rows, summary_path)
    print(f"Hashed {done_bytes / 1e9:.1f} GB in {(time.time() - start) / 60:.1f} min.")
    print(f"Manifest: {len(rows)} files, {summary['bytes'] / 1e9:.1f} GB. "
          f"Duplicate groups: {len(summary['duplicate_groups'])} "
          f"({summary['duplicate_bytes_reclaimable'] / 1e9:.1f} GB reclaimable). "
          f"Summary: {summary_path.name}")
    if failures:
        print(f"{len(failures)} files could not be read:")
        print("\n".join(failures[:20]))
        sys.exit(1)


if __name__ == "__main__":
    main()
