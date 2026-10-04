"""Download single files out of a remote zip on a Dataverse server, without
downloading the whole archive.

A zip keeps its index at the end, so HTTP Range requests let us read the
index and then only the bytes of the members we ask for. Every member is
CRC-checked by zipfile as it is read; a mismatch raises an error.

Usage:
    python scripts/fetch_zip_members.py --server https://rdr.kuleuven.be \
        --doi doi:10.48804/HZRVNZ --dest D:\\...\\raw\\rma\\original \
        mini2RC.zip:mini2RC/mini2_0.mat  mini2vid.zip:mini2vid/mini2_0.mat

    python scripts/fetch_zip_members.py ... --list mini2RC.zip   # show members only
"""
import argparse
import io
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath

import requests

MAX_ATTEMPTS = 5


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file using HTTP Range requests."""

    def __init__(self, url, session):
        self.session, self.pos = session, 0
        probe = session.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
        if probe.status_code != 206:
            raise IOError(f"Server does not support Range requests (HTTP {probe.status_code})")
        self.url = probe.url  # keep the post-redirect URL (often object storage)
        self.size = int(probe.headers["Content-Range"].split("/")[1])
        probe.close()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer):
        end = min(self.pos + len(buffer), self.size)
        if end <= self.pos:
            return 0
        response = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"}, timeout=300)
        response.raise_for_status()
        data = response.content
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def dataset_files(session, server, doi):
    response = session.get(
        f"{server}/api/datasets/:persistentId/versions/:latest/files",
        params={"persistentId": doi},
        timeout=120,
    )
    response.raise_for_status()
    return {f["dataFile"]["filename"]: f["dataFile"]["id"] for f in response.json()["data"]}


def open_remote_zip(session, server, file_id):
    raw = HttpRangeFile(f"{server}/api/access/datafile/{file_id}", session)
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=8 * 1024 * 1024))


def fetch_member(session, server, file_id, member, target):
    if target.exists():
        print(f"  already present: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with open_remote_zip(session, server, file_id) as archive:
                info = archive.getinfo(member)
                with archive.open(info) as src, open(part, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)  # CRC verified at end of read
            part.replace(target)
            print(f"  OK  {target.name}  ({info.file_size / 1e6:.1f} MB, CRC verified)")
            return
        except (requests.RequestException, zipfile.BadZipFile, IOError) as error:
            print(f"  attempt {attempt}/{MAX_ATTEMPTS} failed: {error}")
            part.unlink(missing_ok=True)
            time.sleep(min(60, 5 * attempt))
    raise SystemExit(f"Giving up on {member}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--doi", required=True)
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--list", metavar="ARCHIVE", help="list members of one archive and exit")
    parser.add_argument("members", nargs="*", help="ARCHIVE.zip:path/inside/zip")
    args = parser.parse_args()

    session = requests.Session()
    files = dataset_files(session, args.server, args.doi)

    if args.list:
        with open_remote_zip(session, args.server, files[args.list]) as archive:
            for info in archive.infolist():
                print(f"{info.file_size / 1e6:10.1f} MB  {info.filename}")
        return

    if not args.dest:
        raise SystemExit("--dest is required when fetching members")
    for spec in args.members:
        archive_name, member = spec.split(":", 1)
        print(f"{archive_name}:{member}")
        target = args.dest / PurePosixPath(archive_name).stem / PurePosixPath(member).name
        fetch_member(session, args.server, files[archive_name], member, target)


if __name__ == "__main__":
    main()
