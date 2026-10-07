"""RFUAV (Hugging Face kitofrank/RFUAV, 37 class archives) -> artifact / capture / group index.

Release layout: one RAR per class; inside it a class folder, optionally one
folder per video-signal bandwidth condition (VTSBW=10/20/40/60), holding
packN.xml (acquisition metadata) and packN_<s>-<s+1>s.iq chunks (complex
float32, 1 s at 100 MS/s; a pack's last chunk may be shorter).

  artifact = one .iq chunk (RAR member)
  capture  = a gap-free run of a pack's chunks, concatenated in time order;
             a pack with a missing chunk yields one capture per run
  group    = pack (INFERRED; whether packs of one class share a session is not established)

IDs use the official archive and folder names; local archive names were
URL-mangled by the download and are resolved through
manifests/rfuav_local_to_official.json (SHA-256 checked against the manifest
of every raw file). Labels come from the official class folder, never from the
XML Drone field, which disagrees with the folder for several packs.
"""
import csv
import json
import re
from defusedxml import ElementTree as ET
from collections import defaultdict
from pathlib import PurePosixPath

from ..paths import INTERIM, MANIFESTS, RAW
from .rar import list_rar, read_member

DATASET = "rfuav"
LOCAL_DIR = "rfuav/original"
IQ_NAME = re.compile(r"^pack(?P<pack>\d+)_(?P<start>\d+)-(?P<end>\d+)s\.iq$")
XML_NAME = re.compile(r"^pack(?P<pack>\d+)\.xml$")
CONDITION = re.compile(r"^VTSBW=\d+$")
BYTES_PER_SAMPLE = 8  # complex float32 (XML DataType "Complex Float" in every pack)


def _listings(raw_root, cache_path):
    """{local archive name: {"size": int, "entries": [...], "xml": {member: {field: text}}}}"""
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    changed = False
    for rar in sorted((raw_root / LOCAL_DIR).glob("*.rar")):
        entry = cache.get(rar.name)
        if entry and entry.get("size") == rar.stat().st_size and "xml" in entry:
            continue
        entries = list_rar(rar)
        xml = {}
        for e in entries:
            if e["name"].lower().endswith(".xml"):
                root = ET.fromstring(read_member(rar, e["name"]))
                xml[e["name"]] = {el.tag: (el.text or "").strip() for el in root.iter() if len(el) == 0}
        cache[rar.name] = {"size": rar.stat().st_size, "entries": entries, "xml": xml}
        changed = True
    if changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return cache


def _runs(indices):
    """[0,1,2,4,5] -> [[0,1,2],[4,5]]"""
    runs = []
    for i in sorted(indices):
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def build(raw_root=RAW, interim_root=INTERIM, manifests=MANIFESTS):
    listings = _listings(raw_root, interim_root / DATASET / "rar_listings.json")
    name_map = json.loads((manifests / "rfuav_local_to_official.json").read_text(encoding="utf-8"))
    with open(manifests / "raw_sha256.csv", newline="", encoding="utf-8") as fh:
        sha = {r["relpath"]: r for r in csv.DictReader(fh)}

    artifacts, captures, groups, lineage, evidence, problems = [], [], [], [], [], []
    for local, listing in sorted(listings.items()):
        mapped = name_map.get(local)
        relpath = f"{LOCAL_DIR}/{local}"
        if mapped is None:
            problems.append(f"{local}: not in rfuav_local_to_official.json")
            continue
        official = mapped["official"]
        on_disk = sha.get(relpath)
        same = on_disk is not None and on_disk["sha256"] == mapped["sha256"]
        lineage.append({"subject": relpath, "relation": "local_copy_of" if same else "unmatched",
                        "object": f"huggingface:kitofrank/RFUAV/{official}",
                        "method": "SHA-256 of the local file equals the Hugging Face LFS oid",
                        "status": "VERIFIED" if same else "UNKNOWN"})

        folders = defaultdict(lambda: {"iq": defaultdict(dict), "xml": {}})
        for e in listing["entries"]:
            inner = e["name"].replace("\\", "/")
            folder, name = str(PurePosixPath(inner).parent), PurePosixPath(inner).name
            if m := IQ_NAME.match(name):
                if int(m["end"]) != int(m["start"]) + 1:
                    problems.append(f"{inner}: chunk is not one second by name")
                folders[folder]["iq"][int(m["pack"])][int(m["start"])] = (inner, e)
            elif m := XML_NAME.match(name):
                folders[folder]["xml"][int(m["pack"])] = listing["xml"][e["name"]] | {"_member": inner}
            else:
                problems.append(f"{official}: unexpected member {inner}")

        for folder, content in sorted(folders.items()):
            parts = folder.split("/")
            class_name = parts[0]
            if class_name != official.removesuffix(".rar"):
                problems.append(f"{official}: class folder '{class_name}' differs from the archive name")
            condition = parts[1] if len(parts) > 1 else None
            if condition and not CONDITION.match(condition) or len(parts) > 2:
                problems.append(f"{official}: unexpected folder {folder}")

            xmls = content["xml"]
            for pack, chunks in sorted(content["iq"].items()):
                group_id = f"{DATASET}/{folder}/pack{pack}"
                meta = xmls.get(pack)
                if meta is None and len(xmls) == 1 and len(content["iq"]) == 1:
                    (xml_pack, meta), = xmls.items()
                    evidence.append(_ev(group_id, "xml_link", f"pack{xml_pack}.xml describes pack{pack}_*.iq "
                                        "(names disagree; linked as the folder's only XML and only pack)",
                                        "CONFLICTING", meta["_member"]))
                if meta is None:
                    problems.append(f"{group_id}: no XML for this pack")
                    continue
                fs = float(meta["SampleRate"])
                for field in ("SerialNumber", "Drone", "ScaleFactor", "ReferenceSNRLevel", "CenterFrequency",
                              "SampleRate", "IFBandwidth", "DataType", "DeviceType", "SampleCount"):
                    evidence.append(_ev(group_id, f"xml.{field}", meta[field], "AUTHOR_DOC", meta["_member"]))
                if meta["Drone"].replace(" ", "").upper() != class_name.replace(" ", "").upper():
                    evidence.append(_ev(group_id, "label_disagreement",
                                        f"class folder '{class_name}' vs XML Drone '{meta['Drone']}'",
                                        "CONFLICTING", meta["_member"]))

                runs = _runs(chunks)
                if len(runs) > 1:
                    missing = sorted(set(range(max(chunks) + 1)) - set(chunks))
                    evidence.append(_ev(group_id, "missing_chunks_s", ", ".join(f"{i}-{i + 1}s" for i in missing),
                                        "VERIFIED", "RAR headers"))
                for run in runs:
                    capture_id = f"{DATASET}/{folder}/pack{pack}_{run[0]}-{run[-1] + 1}s"
                    total_bytes = 0
                    for position, start in enumerate(run):
                        inner, e = chunks[start]
                        size = int(e["size"])
                        if size % BYTES_PER_SAMPLE:
                            problems.append(f"{inner}: size is not a whole number of complex samples")
                        total_bytes += size
                        artifacts.append({
                            "artifact_id": f"{DATASET}/{official}/{inner}",
                            "dataset_id": DATASET,
                            "storage_relpath": relpath,
                            "member_chain": [e["name"]],
                            "size_bytes": size,
                            "crc32": e["crc32"].lower(),
                            "member_mtime": e["modified"],
                            "media_type": "application/octet-stream",
                            "capture_id": capture_id,
                            "channel": None,
                            "part_index": position,
                        })
                    captures.append({
                        "capture_id": capture_id,
                        "dataset_id": DATASET,
                        "group_id": group_id,
                        "n_artifacts": len(run),
                        "channels": [],
                        "recorded_at": None,
                        "start_s": float(run[0]),
                        "duration_s": total_bytes / BYTES_PER_SAMPLE / fs,
                        "center_frequency_hz": float(meta["CenterFrequency"]),
                        "sample_rate_hz": fs,
                        "reference_snr_db": float(meta["ReferenceSNRLevel"]),
                        "official_split": None,
                        "label_uas_present": True,
                        "label_emitter": None,  # class taxonomy mixes aircraft and transmitters
                        "label_class": class_name,
                        "label_condition": condition,
                        "label_source": "official class folder; condition from the VTSBW folder",
                        "label_status": "AUTHOR_DOC",
                    })
                groups.append({
                    "group_id": group_id,
                    "dataset_id": DATASET,
                    "kind": "pack",
                    "basis": "a pack's chunks are consecutive 1 s cuts of one recording (chunk names; every short "
                             "chunk is a pack's last); whether packs of one class share a session is not established",
                    "status": "INFERRED",
                    "n_captures": len(runs),
                })

    serials = defaultdict(list)
    for row in evidence:
        if row["field"] == "xml.SerialNumber":
            serials[row["value"]].append(row["entity_id"])
    for serial, packs in sorted(serials.items()):
        if len(packs) > 1:
            evidence.append(_ev(DATASET, "duplicate_xml_serial", f"{serial}: {'; '.join(packs)}", "VERIFIED",
                                "pack XMLs"))
    short = [a["artifact_id"] for a in artifacts if a["size_bytes"] != 100_000_000 * BYTES_PER_SAMPLE]
    evidence += [
        _ev(DATASET, "class_archives", len(listings), "VERIFIED", "local archives", len(listings)),
        _ev(DATASET, "iq_artifacts", len(artifacts), "VERIFIED", "RAR headers", len(listings)),
        _ev(DATASET, "packs", len(groups), "VERIFIED", "RAR headers", len(listings)),
        _ev(DATASET, "captures", len(captures), "VERIFIED", "RAR headers", len(artifacts)),
        _ev(DATASET, "short_chunks", "; ".join(short), "VERIFIED", "RAR headers", len(artifacts)),
        _ev(DATASET, "capture_metadata_columns", "center_frequency_hz, sample_rate_hz, reference_snr_db from the "
            "pack XML; duration_s = bytes / 8 / SampleRate", "AUTHOR_DOC", "pack XMLs", len(groups)),
        _ev(DATASET, "not_downloaded", "ValidationSet_5Drones (5 RARs, 162.1 GB) and the image set", "VERIFIED",
            "manifests/rfuav_hf_listing.json", None),
    ]
    if problems:
        raise ValueError("RFUAV index problems:\n" + "\n".join(problems))
    return {"artifacts": artifacts, "captures": captures, "groups": groups, "lineage": lineage, "evidence": evidence}


def _ev(entity, field, value, status, source, n=None):
    return {"entity_id": entity, "field": field, "value": str(value), "status": status, "source": source, "n": n}
