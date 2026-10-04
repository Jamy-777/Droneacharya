"""Download a selected subset of UAVSig from UCLA Dataverse.

Resumable (HTTP Range), MD5-verified against Dataverse's own checksums,
skips files that are already complete, and saves the full file listing
as a manifest so we know exactly what version we sampled from.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import requests
import urllib3

SERVER = "https://dataverse.ucla.edu"
DOI = "doi:10.25346/S6/LVRRAE"
VERSION = "4.0"

DEST = Path(os.environ.get("UAVSIG_DEST", r"D:\Weeeeeeee\DroneacharyaData\raw\uavsig\original"))
MANIFEST = Path(os.environ.get(
    "UAVSIG_MANIFEST",
    r"D:\Weeeeeeee\DroneacharyaData\manifests\uavsig_v4_file_listing.json",
))

# dataverse.ucla.edu served an expired TLS certificate on 2026-10-04.
# Integrity is checked with Dataverse's MD5 for every file instead.
# Set to True once UCLA renews the certificate.
VERIFY_TLS = False

# (directory prefix, filename); "" = root-level documentation files
WANTED = [
    ("", "UAVSig Readme.pdf"),
    ("", "uavsig_dataset_info.pdf"),
    ("", "metadata.json"),
    ("", "visualize.m"),
    ("", "dataverse_download.py"),
    ("", "generate_spectrogram.ipynb"),
    ("one_drone", "drone_1000_00.mat"),
    ("one_drone", "drone_1000_01.mat"),
    ("one_drone", "drone_0001_00.mat"),
    ("one_drone", "drone_2000_00.mat"),
    ("two_drone", "drone_1200_00.mat"),
    ("controllers", "controller_p1_0000_00.mat"),
    ("controllers", "controller_p1_1000_00.mat"),
    ("controllers", "controller_p1_1111_00.mat"),
]
if os.environ.get("UAVSIG_ONLY"):
    WANTED = [w for w in WANTED if w[1] in os.environ["UAVSIG_ONLY"].split(";")]

CHUNK_BYTES = 8 * 1024 * 1024
MAX_ATTEMPTS = 8


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def list_files(session):
    response = session.get(
        f"{SERVER}/api/datasets/:persistentId/versions/{VERSION}/files",
        params={"persistentId": DOI},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["data"]


def download(session, file_id, target, expected_size, expected_md5):
    if target.exists() and target.stat().st_size == expected_size and md5_of(target) == expected_md5:
        print(f"  already complete: {target.name}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session.get(
                f"{SERVER}/api/access/datafile/{file_id}",
                headers=headers,
                stream=True,
                timeout=(30, 300),
            ) as response:
                if have and response.status_code == 200:
                    have = 0  # server ignored Range: restart this file
                elif response.status_code not in (200, 206):
                    response.raise_for_status()
                with open(part, "ab" if have else "wb") as fh:
                    for block in response.iter_content(CHUNK_BYTES):
                        fh.write(block)

            size = part.stat().st_size
            if size < expected_size:
                raise IOError(f"incomplete: {size:,}/{expected_size:,} bytes")
            if size > expected_size or md5_of(part) != expected_md5:
                part.unlink()
                raise IOError("size or MD5 mismatch; restarting file")

            part.replace(target)
            print(f"  OK  {target.name}  ({size / 1e6:.1f} MB, MD5 verified)")
            return
        except (requests.RequestException, IOError) as error:
            print(f"  attempt {attempt}/{MAX_ATTEMPTS} failed: {error}")
            time.sleep(min(60, 5 * attempt))

    raise SystemExit(f"Giving up on {target}")


def main():
    session = requests.Session()
    session.verify = VERIFY_TLS
    if not VERIFY_TLS:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    listing = list_files(session)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(listing, indent=2), encoding="utf-8")
    print(f"Listing: {len(listing)} files in v{VERSION}; saved to {MANIFEST}")

    for prefix, name in WANTED:
        matches = [
            f for f in listing
            if f["dataFile"]["filename"] == name
            and (f.get("directoryLabel") or "").startswith(prefix)
        ]
        if len(matches) != 1:
            raise SystemExit(f"Expected exactly one match for {prefix}/{name}, found {len(matches)}")

        data_file = matches[0]["dataFile"]
        if data_file["checksum"]["type"] != "MD5":
            raise SystemExit(f"Unexpected checksum type for {name}: {data_file['checksum']['type']}")

        folder = matches[0].get("directoryLabel") or "documentation"
        print(f"{folder}/{name}")
        download(
            session,
            data_file["id"],
            DEST / folder / name,
            data_file["filesize"],
            data_file["checksum"]["value"],
        )

    print("Done.")


if __name__ == "__main__":
    main()
