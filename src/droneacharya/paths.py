"""Locations of the data tree. Override the root with the DRONEACHARYA_DATA environment variable."""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DRONEACHARYA_DATA", r"D:\Weeeeeeee\DroneacharyaData"))
RAW = DATA_ROOT / "raw"
INTERIM = DATA_ROOT / "interim"
INDEX = DATA_ROOT / "index"
MANIFESTS = DATA_ROOT / "manifests"
