"""Dataset card schema (v1).

A card is one YAML file per dataset in dataset_cards/. Every section except
`details` is validated strictly (unknown keys are errors). `details` keeps the
pre-schema card body verbatim until its content is moved into typed fields.

Evidence statuses
-----------------
VERIFIED        measured or checked by us against the stored data
EMPIRICAL       measured by us on n inspected files (n required)
AUTHOR_DOC      stated by the dataset authors (paper, README, datasheet)
AUTHOR_CODE     read from the authors' own code
REPO_METADATA   repository record (Zenodo, Kaggle, Hugging Face, Dataverse)
INFERRED        derived by us from the above; needs confirmation
CONFLICTING     sources disagree
UNKNOWN         not established
NOT_APPLICABLE  the property does not exist for this dataset
ASSESSMENT      our judgement (roles, usefulness, claim ceilings, risks)
UNGRADED        carried over from the hand-written matrix without an
                evidence marker; to be graded, never cited as evidence
"""
from typing import Annotated, Any, Generic, Literal, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

Status = Literal[
    "VERIFIED", "EMPIRICAL", "AUTHOR_DOC", "AUTHOR_CODE", "REPO_METADATA",
    "INFERRED", "CONFLICTING", "UNKNOWN", "NOT_APPLICABLE", "ASSESSMENT", "UNGRADED",
]
ABSENT: frozenset[str] = frozenset({"UNKNOWN", "NOT_APPLICABLE"})
DATASET_ID = r"^[a-z][a-z0-9_]*$"

T = TypeVar("T")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_evidence(status, n):
    if status == "EMPIRICAL" and n is None:
        raise ValueError("EMPIRICAL needs n (number of files inspected)")
    if status in ABSENT and n is not None:
        raise ValueError(f"{status} cannot carry n")


class Fact(Strict):
    """One statement about a dataset, shown as a matrix cell."""

    text: str = Field(min_length=1)
    status: Status
    n: int | None = Field(default=None, ge=1)
    source: str | None = None

    @model_validator(mode="after")
    def _valid(self):
        if "\n" in self.text or "|" in self.text.replace("\\|", ""):
            raise ValueError("text must be one line without unescaped '|' (it becomes a table cell)")
        _check_evidence(self.status, self.n)
        return self


class Measured(Strict, Generic[T]):
    """A typed value that code consumes. Null means absent; never a guess."""

    value: T | None = None
    varies: bool = False  # differs per artifact/capture; read it from the file or metadata
    status: Status
    n: int | None = Field(default=None, ge=1)
    source: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _valid(self):
        _check_evidence(self.status, self.n)
        if self.status in ABSENT:
            if self.value is not None or self.varies:
                raise ValueError(f"{self.status} must have value null and varies false")
        elif self.varies == (self.value is not None):
            raise ValueError("give either a value or varies: true, not both and not neither")
        return self


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
