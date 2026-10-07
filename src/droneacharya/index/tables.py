"""Index tables shared by every dataset: artifacts, captures, groups, lineage, evidence.

artifact  stored bytes: a file under raw/ or a member inside one (possibly nested)
capture   one physically continuous acquisition; may span artifacts (time parts or channels)
group     the independence unit for splits
lineage   relations between stored copies (exact copies, derived releases)
evidence  facts established while indexing: (entity, field, value, status, source, n)

IDs are readable paths built from official names, always '/'-separated and
starting with the dataset id. Locations on disk live only in storage_relpath
(relative to raw/); no table holds a drive path.
"""
import pyarrow as pa
import pyarrow.parquet as pq

from ..paths import INDEX

ARTIFACTS = pa.schema([
    ("artifact_id", pa.string()),
    ("dataset_id", pa.string()),
    ("storage_relpath", pa.string()),          # outermost stored file, relative to raw/ (matches raw_sha256.csv)
    ("member_chain", pa.list_(pa.string())),   # member names as stored, outermost first; empty for plain files
    ("size_bytes", pa.int64()),                # uncompressed size of the artifact itself
    ("crc32", pa.string()),                    # as recorded by the container, hex
    ("member_mtime", pa.string()),             # as recorded by the container
    ("media_type", pa.string()),
    ("capture_id", pa.string()),
    ("channel", pa.string()),                  # role inside the capture (band, microphone); null if single
    ("part_index", pa.int32()),                # time-order position inside the capture; null if single
])

CAPTURES = pa.schema([
    ("capture_id", pa.string()),
    ("dataset_id", pa.string()),
    ("group_id", pa.string()),
    ("n_artifacts", pa.int32()),
    ("channels", pa.list_(pa.string())),
    ("recorded_at", pa.string()),              # acquisition timestamp as stored (ISO 8601, no zone), when known
    ("start_s", pa.float64()),                 # offset inside the group's recording, when known
    ("duration_s", pa.float64()),              # null unless sample count and rate are both established
    ("center_frequency_hz", pa.float64()),     # per-capture acquisition metadata, when the release has it
    ("sample_rate_hz", pa.float64()),
    ("reference_snr_db", pa.float64()),
    ("official_split", pa.string()),           # the release's own split (train / test), if it has one
    ("label_uas_present", pa.bool_()),         # a UAS emitter (aircraft or its controller) is active
    ("label_emitter", pa.string()),            # aircraft | controller | uas_link | wifi | bluetooth | none; null if unlabelled
    ("label_class", pa.string()),              # the release's class (model / system), official spelling
    ("label_condition", pa.string()),          # operating mode or signal condition, if labelled
    ("label_source", pa.string()),
    ("label_status", pa.string()),
])

GROUPS = pa.schema([
    ("group_id", pa.string()),
    ("dataset_id", pa.string()),
    ("kind", pa.string()),
    ("basis", pa.string()),
    ("status", pa.string()),
    ("n_captures", pa.int32()),
])

LINEAGE = pa.schema([
    ("subject", pa.string()),      # raw relpath or artifact id
    ("relation", pa.string()),     # exact_copy_of | derived_from | contains_copy_of
    ("object", pa.string()),
    ("method", pa.string()),       # how the relation was established
    ("status", pa.string()),
])

EVIDENCE = pa.schema([
    ("entity_id", pa.string()),
    ("field", pa.string()),
    ("value", pa.string()),
    ("status", pa.string()),
    ("source", pa.string()),
    ("n", pa.int64()),
])

SCHEMAS = {"artifacts": ARTIFACTS, "captures": CAPTURES, "groups": GROUPS, "lineage": LINEAGE, "evidence": EVIDENCE}
STATUSES = {"VERIFIED", "EMPIRICAL", "AUTHOR_DOC", "AUTHOR_CODE", "REPO_METADATA", "INFERRED",
            "CONFLICTING", "UNKNOWN", "NOT_APPLICABLE", "ASSESSMENT"}


def _no_drive_path(value, where):
    if value and (":" in value or "\\" in value):
        raise ValueError(f"{where}: '{value}' looks like a drive path or uses backslashes")


def validate(dataset_id, tables):
    """Raise ValueError on broken keys, references, paths or statuses."""
    artifacts, captures, groups = tables["artifacts"], tables["captures"], tables["groups"]
    for name, rows, key in (("artifacts", artifacts, "artifact_id"), ("captures", captures, "capture_id"),
                            ("groups", groups, "group_id")):
        ids = [r[key] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name}: duplicate {key}")
        for r in rows:
            if r["dataset_id"] != dataset_id or not r[key].startswith(dataset_id + "/"):
                raise ValueError(f"{name}: {r[key]} is not under dataset '{dataset_id}'")
            _no_drive_path(r[key], name)
    for r in artifacts:
        _no_drive_path(r["storage_relpath"], "artifacts.storage_relpath")
    capture_ids = {r["capture_id"] for r in captures}
    group_ids = {r["group_id"] for r in groups}
    per_capture = {}
    for r in artifacts:
        if r["capture_id"] not in capture_ids:
            raise ValueError(f"artifact {r['artifact_id']} points at unknown capture {r['capture_id']}")
        per_capture[r["capture_id"]] = per_capture.get(r["capture_id"], 0) + 1
    per_group = {}
    for r in captures:
        if r["group_id"] not in group_ids:
            raise ValueError(f"capture {r['capture_id']} points at unknown group {r['group_id']}")
        if per_capture.get(r["capture_id"], 0) != r["n_artifacts"]:
            raise ValueError(f"capture {r['capture_id']}: n_artifacts {r['n_artifacts']} "
                             f"but {per_capture.get(r['capture_id'], 0)} artifacts point at it")
        per_group[r["group_id"]] = per_group.get(r["group_id"], 0) + 1
    for r in groups:
        if per_group.get(r["group_id"], 0) != r["n_captures"]:
            raise ValueError(f"group {r['group_id']}: n_captures does not match")
    for name in ("captures", "groups", "lineage", "evidence"):
        field = "label_status" if name == "captures" else "status"
        for r in tables[name]:
            if r[field] not in STATUSES:
                raise ValueError(f"{name}: unknown status {r[field]!r}")


def write(dataset_id, tables, index_root=INDEX):
    validate(dataset_id, tables)
    out = index_root / dataset_id
    out.mkdir(parents=True, exist_ok=True)
    for name, schema in SCHEMAS.items():
        pq.write_table(pa.Table.from_pylist(tables[name], schema=schema), out / f"{name}.parquet")
    return out


def read(dataset_id, name, index_root=INDEX):
    return pq.read_table(index_root / dataset_id / f"{name}.parquet")
