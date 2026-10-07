"""Dataset card schema (v1).

A card is one YAML file per dataset in dataset_cards/. Every section except
`details` is validated strictly (unknown keys are errors). `details` keeps the
pre-schema card body verbatim until its content is moved into typed fields.

Every claim has a kind and an evidence status, which are independent:

Kinds
-----
FACT        a property of the dataset (sample rate, class counts, site)
ASSESSMENT  our judgement about the dataset (shortcut risk, usefulness,
            claim ceiling); its status grades the observations it rests on
            and its source points at them
Project decisions (roles, download choices) are not dataset claims; they
live in configs/dataset_roles.yaml and are joined into the matrices.

Evidence statuses
-----------------
VERIFIED        read directly from the distributed files, covering the claim's
                whole scope, without sampling or statistics (a header field,
                a complete inventory, every member's CRC)
EMPIRICAL       measured by us: on a sample, or a statistic even over all
                files (clipping fractions, level medians); n and n_unit required
AUTHOR_DOC      stated by the dataset authors (paper, README, datasheet)
AUTHOR_CODE     read from the authors' own code
REPO_METADATA   repository record (Zenodo, Kaggle, Hugging Face, Dataverse)
INFERRED        reasoned by us from the above; needs confirmation
CONFLICTING     sources disagree about the same thing: list them in `conflict`
UNKNOWN         not established
NOT_APPLICABLE  the property does not exist for this dataset
UNGRADED        migration only: carried over from the hand-written matrix;
                a test requires none to remain

Grading rules
-------------
- A derived value is never graded above its weakest input.
- The status order is a trust heuristic, not a resolver: an author statement
  and a measurement about different stages (ADC bits vs stored dtype) do not
  conflict, and a higher status never silently overrides a lower one.
- "Frozen" means: schema-valid, no UNGRADED, every graded claim sourced.
  Cards still change afterwards, through reviewed commits.
"""
from typing import Annotated, Any, Generic, Literal, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

Status = Literal[
    "VERIFIED", "EMPIRICAL", "AUTHOR_DOC", "AUTHOR_CODE", "REPO_METADATA",
    "INFERRED", "CONFLICTING", "UNKNOWN", "NOT_APPLICABLE", "UNGRADED",
]
Kind = Literal["FACT", "ASSESSMENT"]
NUnit = Literal[
    "file", "archive", "archive_member", "capture", "clip", "recording", "pack",
    "class", "window", "physical_unit", "session",
]
ABSENT: frozenset[str] = frozenset({"UNKNOWN", "NOT_APPLICABLE"})
DATASET_ID = r"^[a-z][a-z0-9_]*$"

T = TypeVar("T")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Claim(Strict):
    """One side of a conflict."""

    claim: str
    status: Literal["VERIFIED", "EMPIRICAL", "AUTHOR_DOC", "AUTHOR_CODE", "REPO_METADATA", "INFERRED"]
    source: str
    n: int | None = Field(default=None, ge=1)
    n_unit: NUnit | None = None

    @model_validator(mode="after")
    def _valid(self):
        _check_n(self.status, self.n, self.n_unit)
        return self


def _check_n(status, n, n_unit):
    if status == "EMPIRICAL" and n is None:
        raise ValueError("EMPIRICAL needs n and n_unit")
    if (n is None) != (n_unit is None):
        raise ValueError("n and n_unit go together")
    if status in ABSENT and n is not None:
        raise ValueError(f"{status} cannot carry n")


def _check_cell_text(text):
    if "\n" in text or "|" in text.replace("\\|", ""):
        raise ValueError("text must be one line without unescaped '|' (it becomes a table cell)")


def _check_evidence(status, n, n_unit, source, conflict):
    _check_n(status, n, n_unit)
    if status not in ABSENT and status not in ("UNGRADED", "CONFLICTING") and not source:
        raise ValueError(f"{status} needs a source")  # CONFLICTING: each claim carries its own
    if (status == "CONFLICTING") != bool(conflict):
        raise ValueError("CONFLICTING needs a conflict list, and only CONFLICTING may have one")
    if conflict and len(conflict) < 2:
        raise ValueError("a conflict needs at least two claims")


class Fact(Strict):
    """One claim about a dataset, shown as a matrix cell."""

    text: str = Field(min_length=1)
    kind: Kind
    status: Status
    n: int | None = Field(default=None, ge=1)
    n_unit: NUnit | None = None
    source: str | None = None
    conflict: list[Claim] | None = None

    @model_validator(mode="after")
    def _valid(self):
        _check_cell_text(self.text)
        _check_evidence(self.status, self.n, self.n_unit, self.source, self.conflict)
        return self


class Measured(Strict, Generic[T]):
    """A typed value that code consumes. Null means absent; never a guess.

    For CONFLICTING, `value` is what the stored files contain (what code
    processes); `conflict` records the disagreeing claims.
    """

    value: T | None = None
    varies: bool = False  # differs per artifact/capture; read it from the file or metadata
    status: Status
    n: int | None = Field(default=None, ge=1)
    n_unit: NUnit | None = None
    source: str | None = None
    note: str | None = None
    conflict: list[Claim] | None = None

    @model_validator(mode="after")
    def _valid(self):
        _check_evidence(self.status, self.n, self.n_unit, self.source, self.conflict)
        if self.status in ABSENT:
            if self.value is not None or self.varies:
                raise ValueError(f"{self.status} must have value null and varies false")
        elif self.varies == (self.value is not None):
            raise ValueError("give either a value or varies: true, not both and not neither")
        return self


class Policy(Strict):
    """A Droneacharya decision about a dataset (configs/dataset_roles.yaml)."""

    text: str = Field(min_length=1)
    basis: str  # where the decision and its reasons are recorded

    @model_validator(mode="after")
    def _valid(self):
        _check_cell_text(self.text)
        return self


class DatasetPolicy(Strict):
    rows: dict[str, Policy]                 # keys = policy rows of the matrix
    supplementary: dict[str, Policy] = {}   # keys = supplementary row labels


class RolesFile(Strict):
    schema_version: Literal[1]
    datasets: dict[str, DatasetPolicy]


class Source(Strict):
    repository: str | None = None
    identifier: str | None = None  # DOI, repo slug or URL
    version: str | None = None


class Storage(Strict):
    raw_dir: str = Field(pattern=DATASET_ID)  # folder under DroneacharyaData/raw/
    source: Source
    name_map: str | None = None  # manifest mapping local names to official names, relative to the data root

    @model_validator(mode="after")
    def _relative(self):
        if self.name_map and (":" in self.name_map or "\\" in self.name_map or self.name_map.startswith("/")):
            raise ValueError("name_map must be a forward-slash path relative to the data root")
        return self


class Structure(Strict):
    """The three-level model: stored bytes -> continuous acquisition -> independence unit."""

    artifact: Fact
    capture: Fact
    group: Fact


class RFSignal(Strict):
    complex_iq: Measured[bool]
    scalar_dtype: Measured[str]
    container: Measured[str]
    sample_rate_hz: Measured[float]
    center_frequency_hz: Measured[float]
    capture_bandwidth_hz: Measured[float]
    samples_per_artifact: Measured[int]
    artifact_duration_s: Measured[float]


class AcousticSignal(Strict):
    container: Measured[str]
    sample_format: Measured[str]
    sample_rate_hz: Measured[float]
    channels_per_artifact: Measured[int]
    artifact_duration_s: Measured[float]


class CardBase(Strict):
    schema_version: Literal[1]
    id: str = Field(pattern=DATASET_ID)
    name: str
    full_name: str | None = None
    adoption: Literal["adopted", "backlog"]
    storage: Storage | None = None
    structure: Structure
    facts: dict[str, Fact]  # keys = matrix rows of the card's modality
    supplementary: dict[str, Fact] = {}  # dataset-specific rows; key = row label
    details: dict[str, Any] = {}

    @model_validator(mode="after")
    def _storage(self):
        if self.adoption == "adopted" and self.storage is None:
            raise ValueError("adopted datasets need a storage section")
        return self


class RFCard(CardBase):
    modality: Literal["rf"]
    signal: RFSignal


class AcousticCard(CardBase):
    modality: Literal["acoustic"]
    signal: AcousticSignal


Card = Annotated[Union[RFCard, AcousticCard], Field(discriminator="modality")]


class RowSpec(Strict):
    key: str = Field(pattern=DATASET_ID)
    label: str
    policy: bool = False  # cell comes from configs/dataset_roles.yaml, not the card


class ColumnSpec(Strict):
    id: str = Field(pattern=DATASET_ID)
    label: str


class SupplementSpec(Strict):
    id: str = Field(pattern=DATASET_ID)
    heading: str


class MatrixSpec(Strict):
    """Layout of one generated compatibility matrix (configs/matrices/<modality>.yaml)."""

    modality: Literal["rf", "acoustic"]
    output: str  # repo-relative path of the generated markdown
    preamble: str
    columns: list[ColumnSpec]
    rows: list[RowSpec]
    supplements: list[SupplementSpec] = []
    cross_dataset_heading: str | None = None
    cross_dataset: dict[str, Fact] = {}

    @model_validator(mode="after")
    def _unique(self):
        for name, keys in (("row", [r.key for r in self.rows]), ("column", [c.id for c in self.columns])):
            if len(keys) != len(set(keys)):
                raise ValueError(f"duplicate {name} keys")
        return self
