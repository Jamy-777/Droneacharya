"""Download one folder from a public GitHub repository without cloning it.

Used for datasets where only part of a large repository is needed
(e.g. the 90 audio clips of the Svanström drone-detection dataset).
Skips files that are already present with the right size.

Usage:
    python scripts/fetch_github_folder.py OWNER/REPO path/in/repo DEST_DIR [--branch master]
"""
import argparse
import time
from pathlib import Path

import requests

API = "https://api.github.com"


def list_folder(repo, folder, branch):
    response = requests.get(
        f"{API}/repos/{repo}/git/trees/{branch}",
        params={"recursive": 1},
        headers={"Accept": "application/vnd.github+json"},
        timeout=60,
    )
    response.raise_for_status()
    tree = response.json()
    if tree.get("truncated"):
        raise SystemExit("GitHub truncated the file tree; this repository is too large for this script.")
    prefix = folder.rstrip("/") + "/"
    return [item for item in tree["tree"] if item["type"] == "blob" and item["path"].startswith(prefix)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("folder")
    parser.add_argument("dest", type=Path)
    parser.add_argument("--branch", default="master")
    args = parser.parse_args()

    files = list_folder(args.repo, args.folder, args.branch)
    if not files:
        raise SystemExit(f"No files under {args.folder} on branch {args.branch}")
    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} files, {sum(f['size'] for f in files) / 1e6:.1f} MB")

    for item in files:
        target = args.dest / Path(item["path"]).relative_to(args.folder)
        if target.exists() and target.stat().st_size == item["size"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{args.repo}/{args.branch}/{item['path']}"
        for attempt in range(1, 6):
            try:
                data = requests.get(url, timeout=120).content
                if len(data) != item["size"]:
                    raise IOError(f"size {len(data)} != {item['size']}")
                target.write_bytes(data)
                break
            except (requests.RequestException, IOError) as error:
                print(f"  {item['path']}: attempt {attempt} failed ({error})")
                time.sleep(5 * attempt)
        else:
            raise SystemExit(f"Giving up on {item['path']}")
    print(f"Done: {args.dest}")


if __name__ == "__main__":
    main()
