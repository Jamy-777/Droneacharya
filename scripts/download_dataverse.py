"""Download a selection of files from any Dataverse dataset (UCLA, KU Leuven, ...).

Resumable (HTTP Range), MD5-verified against Dataverse's own checksums, skips
files already complete, and saves the dataset's file listing as a manifest.
Files keep their Dataverse folder (directoryLabel) under --dest.

Selection is by regular expressions matched against "folder/filename".

Usage:
    # see what a selection would download (nothing is downloaded)
    python scripts/download_dataverse.py --server https://rdr.kuleuven.be --doi doi:10.48804/HZRVNZ \
        --dest D:/.../raw/rma/original --list

    # download it
    python scripts/download_dataverse.py ... (same arguments, without --list)

    # UCLA's certificate has expired: add --insecure (integrity is still checked by MD5)
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import requests
import urllib3

CHUNK_BYTES = 8 * 1024 * 1024
MAX_ATTEMPTS = 8


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def list_files(session, server, doi, version):
    response = session.get(
        f"{server}/api/datasets/:persistentId/versions/{version}/files",
        params={"persistentId": doi},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["data"]


def selected(listing, include, exclude):
    chosen = []
    for entry in listing:
        path = f"{entry.get('directoryLabel') or ''}/{entry['dataFile']['filename']}".lstrip("/")
        if include and not any(re.search(p, path) for p in include):
            continue
        if any(re.search(p, path) for p in exclude):
            continue
        chosen.append((path, entry["dataFile"]))
    return chosen


def finalize(part, target, size):
    """Rename a verified .part file; Windows antivirus may hold a fresh file briefly."""
    for wait in range(12):
        try:
            part.replace(target)
            print(f"  OK  {target.name}  ({size / 1e6:.1f} MB, MD5 verified)")
            return
        except PermissionError:
            time.sleep(5)
    raise SystemExit(f"{part.name} is verified but still locked by another process; "
                     f"rename it to {target.name} by hand or rerun later.")


def download(session, server, data_file, target):
    size, md5 = data_file["filesize"], data_file["checksum"]["value"]
    if target.exists() and target.stat().st_size == size and md5_of(target) == md5:
        print(f"  already complete: {target.name}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = part.stat().st_size if part.exists() else 0
        if have >= size:
            # Fully downloaded earlier (e.g. the rename failed): verify, never re-request.
            if have == size and md5_of(part) == md5:
                return finalize(part, target, size)
            part.unlink()
            have = 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session.get(f"{server}/api/access/datafile/{data_file['id']}", headers=headers,
                             stream=True, timeout=(30, 300)) as response:
                if have and response.status_code == 200:
                    have = 0  # server ignored Range: restart this file
                elif response.status_code not in (200, 206):
                    response.raise_for_status()
                with open(part, "ab" if have else "wb") as fh:
                    for block in response.iter_content(CHUNK_BYTES):
                        fh.write(block)
            got = part.stat().st_size
            if got < size:
                raise IOError(f"incomplete: {got:,}/{size:,} bytes")
            if got > size or md5_of(part) != md5:
                part.unlink()
                raise IOError("size or MD5 mismatch; restarting file")
            return finalize(part, target, size)
        except (requests.RequestException, IOError) as error:
            print(f"  attempt {attempt}/{MAX_ATTEMPTS} failed: {error}")
            time.sleep(min(60, 5 * attempt))
    raise SystemExit(f"Giving up on {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--doi", required=True)
    parser.add_argument("--version", default=":latest")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--include", action="append", default=[], help="regex on folder/filename (repeatable)")
    parser.add_argument("--exclude", action="append", default=[], help="regex on folder/filename (repeatable)")
    parser.add_argument("--manifest", type=Path, help="where to save the full file listing (JSON)")
    parser.add_argument("--list", action="store_true", help="show the selection and exit")
    parser.add_argument("--insecure", action="store_true", help="skip TLS verification (expired server certificate)")
    args = parser.parse_args()

    session = requests.Session()
    if args.insecure:
        session.verify = False
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    listing = list_files(session, args.server, args.doi, args.version)
    if args.manifest and args.manifest.exists():
        # The MD5s come from the same server as the files; comparing them with a
        # listing saved earlier catches a listing that changed since then.
        saved = {e["dataFile"]["id"]: e["dataFile"]["checksum"]["value"]
                 for e in json.loads(args.manifest.read_text(encoding="utf-8"))}
        changed = [e["dataFile"]["filename"] for e in listing
                   if e["dataFile"]["id"] in saved and saved[e["dataFile"]["id"]] != e["dataFile"]["checksum"]["value"]]
        if changed:
            raise SystemExit(f"Checksums differ from the saved manifest for {len(changed)} files (e.g. {changed[:3]}); stopping.")
        print(f"Checksums match the saved manifest ({len(saved)} files).")
    elif args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(listing, indent=2), encoding="utf-8")

    chosen = selected(listing, args.include, args.exclude)
    total = sum(f["filesize"] for _, f in chosen)
    present = sum(f["filesize"] for p, f in chosen if (args.dest / p).exists() and (args.dest / p).stat().st_size == f["filesize"])
    print(f"Selected {len(chosen)} of {len(listing)} files: {total / 1e9:.2f} GB "
          f"({present / 1e9:.2f} GB already on disk, {(total - present) / 1e9:.2f} GB to download)")
    if args.list:
        folders = {}
        for path, f in chosen:
            key = path.rsplit("/", 1)[0] if "/" in path else "(root)"
            folders.setdefault(key, [0, 0])
            folders[key][0] += 1
            folders[key][1] += f["filesize"]
        for key, (n, size) in sorted(folders.items()):
            print(f"  {n:4d} files  {size / 1e9:7.2f} GB  {key}")
        return

    for path, data_file in chosen:
        print(path)
        download(session, args.server, data_file, args.dest / path)
    print("Done.")


if __name__ == "__main__":
    main()
