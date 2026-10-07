"""Resumable download of one large file over HTTP (e.g. CARDRF.zip from SharePoint).

Keeps a <name>.part file and continues it with HTTP Range requests.

SharePoint guest shares: pass the public share link with --share-url. The
script opens it first to receive the same guest cookie a browser gets, and
re-opens it automatically whenever the cookie expires. Nothing is written to
disk except the file itself.

Without --share-url, the script asks for a download link and asks again
whenever the server rejects it.

Usage (CardRF):
    python scripts/resume_http_download.py "D:\\Weeeeeeee\\DroneacharyaData\\raw\\cardrf\\original\\CARDRF.zip" ^
        --share-url "https://cardmaillouisville-my.sharepoint.com/:f:/g/personal/aplauf01_louisville_edu/EhZkrkpdqbNHm7Xi-rEY67YBmv1dCHlQYpxLPX4hp5EFlg?e=y9WDAP" ^
        --url "https://cardmaillouisville-my.sharepoint.com/personal/aplauf01_louisville_edu/_layouts/15/download.aspx?UniqueId=e9f3de1a%2Df307%2D4d9d%2D831c%2D6dda679bbf20" ^
        --expected-bytes 71139787221
"""
import argparse
import sys
import time
from pathlib import Path

import requests

CHUNK_BYTES = 8 * 1024 * 1024
MAX_NETWORK_RETRIES = 50
MAX_COOKIE_REFRESHES = 20
REPORT_EVERY_S = 15


def ask_for_url(reason):
    print(f"\n{reason}")
    url = input("Paste a fresh download link (or press Enter to stop): ").strip()
    if not url:
        sys.exit("Stopped. Run the same command again later; the .part file is kept.")
    return url


def remote_size(response):
    content_range = response.headers.get("Content-Range", "")
    if "/" in content_range:
        return int(content_range.rsplit("/", 1)[1])
    if response.status_code == 200 and "Content-Length" in response.headers:
        return int(response.headers["Content-Length"])
    return None


def open_share(session, share_url):
    """Visit a SharePoint guest share link so the session holds its guest cookie."""
    response = session.get(share_url, timeout=60, allow_redirects=True)
    response.raise_for_status()
    if "FedAuth" not in session.cookies:
        sys.exit("Opening the share link did not grant a guest cookie; the share may have changed.")
    print("Guest access cookie obtained from the share link.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dest", type=Path)
    parser.add_argument("--url", help="direct download URL (otherwise you are asked for one)")
    parser.add_argument("--share-url", help="SharePoint guest share link to obtain and refresh the access cookie")
    parser.add_argument("--expected-bytes", type=int)
    args = parser.parse_args()

    dest = args.dest
    part = dest.with_name(dest.name + ".part")
    if dest.exists():
        sys.exit(f"{dest} already exists; nothing to do.")
    dest.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    if args.share_url:
        open_share(session, args.share_url)
    url = args.url or ask_for_url("A download link is needed.")
    total = args.expected_bytes
    failures = 0
    cookie_refreshes = 0

    while True:
        have = part.stat().st_size if part.exists() else 0
        if total and have >= total:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 120), allow_redirects=True) as response:
                content_type = response.headers.get("Content-Type", "")
                if response.status_code in (401, 403) or "text/html" in content_type:
                    reason = f"Link rejected or expired (HTTP {response.status_code}, {content_type or 'no type'})."
                    if args.share_url and cookie_refreshes < MAX_COOKIE_REFRESHES:
                        cookie_refreshes += 1
                        print(f"  {reason} Refreshing guest cookie ({cookie_refreshes}/{MAX_COOKIE_REFRESHES})...", flush=True)
                        session.cookies.clear()
                        open_share(session, args.share_url)
                    else:
                        url = ask_for_url(reason)
                    continue
                if have and response.status_code == 200:
                    sys.exit("Server ignored the Range request and would restart from byte 0; "
                             "stopping so the existing .part file is not overwritten.")
                response.raise_for_status()

                size = remote_size(response)
                if size and total and size != total:
                    sys.exit(f"Server reports {size:,} bytes but --expected-bytes is {total:,}; different file?")
                total = total or size
                print(f"Resuming at {have / 1e9:.2f} GB of {total / 1e9:.2f} GB" if total else f"Resuming at {have / 1e9:.2f} GB")

                start_bytes, start_time, last_report = have, time.time(), time.time()
                with open(part, "ab") as fh:
                    for block in response.iter_content(CHUNK_BYTES):
                        fh.write(block)
                        have += len(block)
                        if time.time() - last_report >= REPORT_EVERY_S:
                            rate = (have - start_bytes) / (time.time() - start_time) / 1e6
                            eta = (total - have) / (rate * 1e6) / 3600 if total and rate else float("nan")
                            print(f"  {have / 1e9:7.2f} GB  {rate:6.1f} MB/s  ETA {eta:5.1f} h", flush=True)
                            last_report = time.time()
            failures = 0
            cookie_refreshes = 0
            if total is None or have >= total:
                break
        except requests.RequestException as error:
            failures += 1
            if failures > MAX_NETWORK_RETRIES:
                sys.exit(f"Too many network failures ({error}); run again later to continue.")
            wait = min(120, 5 * failures)
            print(f"  network error: {error}; retrying in {wait}s", flush=True)
            time.sleep(wait)

    have = part.stat().st_size
    if total and have != total:
        sys.exit(f"Size mismatch: have {have:,}, expected {total:,}. The .part file is kept.")
    for _ in range(12):  # Windows antivirus may hold a freshly written file briefly
        try:
            part.replace(dest)
            break
        except PermissionError:
            time.sleep(5)
    else:
        sys.exit(f"{part.name} is complete but still locked; rename it to {dest.name} by hand.")
    print(f"Done: {dest} ({have / 1e9:.2f} GB). Verify the archive before extracting anything.")


if __name__ == "__main__":
    main()
