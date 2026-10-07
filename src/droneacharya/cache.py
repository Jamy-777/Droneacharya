"""Feature caches keyed by everything that can change the features.

The key hashes the compute function's source, the parameters, the feature and
reader code (signal.py, index tables code) and the index tables of the datasets
involved, so a change to any of them recomputes instead of reusing stale numbers.
"""
import hashlib
import inspect
import json
from pathlib import Path

from .paths import INDEX

PACKAGE = Path(__file__).resolve().parent
CODE_FILES = ("signal.py", "baselines.py", "features.py", "index/tables.py", "index/acoustic.py")


def key(compute, params, datasets):
    digest = hashlib.sha256(inspect.getsource(compute).encode() + json.dumps(params, sort_keys=True).encode())
    for name in CODE_FILES:
        path = PACKAGE / name
        digest.update(path.read_bytes() if path.exists() else b"")
    for dataset in sorted(datasets):
        for table in ("captures", "artifacts"):
            digest.update((INDEX / dataset / f"{table}.parquet").read_bytes())
    return digest.hexdigest()[:12]


def cached(folder, name, compute, params, datasets):
    path = Path(folder) / f"{name}-{key(compute, params, datasets)}.features.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return rows
