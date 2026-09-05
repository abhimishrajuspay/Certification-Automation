"""Immutable, LLM-ready knowledge derived from deterministic crawl evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from scraper.models import ActionKind, ActionRisk, ActionStatus, ScrapeRunStatus


KNOWLEDGE_SCHEMA_VERSION = "1.0"
SHA256_PATTERN = r"^[a-f0-9]{64}$"


class KnowledgeSourceKind(str, Enum):
    """Where the normalized testcase content originally came from."""

    CRAWL = "crawl"
    EXTERNAL_CSV = "external_csv"


class KnowledgeModel(BaseModel):
    """Strict immutable base model for normalized portal knowledge."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class EvidencePointer(KnowledgeModel):
    """Minimal citation back to immutable crawl evidence."""

    state_id: str = Field(min_length=1)
    state_sequence: int = Field(ge=0)
    url: str = Field(min_length=1)
    frame_id: str = Field(min_length=1)
    element_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_unique_references(self) -> "EvidencePointer":
        if len(self.element_ids) != len(set(self.element_ids)):
            raise ValueError("element_ids must be unique")
        if len(self.artifact_ids) != len(set(self.artifact_ids)):
            raise ValueError("artifact_ids must be unique")
        return self


class KnowledgeField(KnowledgeModel):
    """One normalized table field while retaining its rendered label."""

    key: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str = Field(min_length=1)
    value: str = ""


class NormalizedControl(KnowledgeModel):
    """One interactive control associated with a normalized table row."""

    element_id: str = Field(min_length=1)
    role: Optional[str] = None
    label: Optional[str] = None
    title: Optional[str] = None
    href: Optional[str] = None
    identifiers: tuple[str, ...] = ()
    action_kind: Optional[ActionKind] = None
    risk: Optional[ActionRisk] = None
    action_status: Optional[ActionStatus] = None
    policy_rule: Optional[str] = None

    @model_validator(mode="after")
    def validate_action_metadata(self) -> "NormalizedControl":
        values = (self.risk, self.action_status, self.policy_rule)
        if self.action_kind is None and any(value is not None for value in values):
            raise ValueError("action metadata requires action_kind")
        if len(self.identifiers) != len(set(self.identifiers)):
            raise ValueError("control identifiers must be unique")
        return self


class NormalizedTableRow(KnowledgeModel):
    """One deduplicated table row with its controls and source citations."""

    row_id: str = Field(pattern=SHA256_PATTERN)
    fields: tuple[KnowledgeField, ...]
    controls: tuple[NormalizedControl, ...] = ()
    evidence: tuple[EvidencePointer, ...]

    @model_validator(mode="after")
    def validate_row(self) -> "NormalizedTableRow":
        keys = [field.key for field in self.fields]
        if not keys:
            raise ValueError("a normalized row requires at least one field")
        if len(keys) != len(set(keys)):
            raise ValueError("normalized row field keys must be unique")
        control_ids = [control.element_id for control in self.controls]
        if len(control_ids) != len(set(control_ids)):
            raise ValueError("normalized row controls must be unique")
        if not self.evidence:
            raise ValueError("a normalized row requires evidence")
        return self


class NormalizedTable(KnowledgeModel):
    """Logical table merged across equivalent states and pagination aliases."""

    table_id: str = Field(pattern=SHA256_PATTERN)
    logical_url: str = Field(min_length=1)
    frame_path: str = Field(min_length=1)
    headers: tuple[str, ...]
    rows: tuple[NormalizedTableRow, ...]
    evidence: tuple[EvidencePointer, ...]

    @model_validator(mode="after")
    def validate_table(self) -> "NormalizedTable":
        if not self.headers:
            raise ValueError("a normalized table requires headers")
        row_ids = [row.row_id for row in self.rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("normalized table rows must be unique")
        if not self.evidence:
            raise ValueError("a normalized table requires evidence")
        return self


class TestCaseKnowledge(KnowledgeModel):
    """Compact testcase context grounded in table and modal evidence."""

    test_case_id: str = Field(min_length=1)
    fields: tuple[KnowledgeField, ...]
    dependency_case_ids: tuple[str, ...] = ()
    description: Optional[str] = None
    controls: tuple[NormalizedControl, ...] = ()
    evidence: tuple[EvidencePointer, ...]
    description_evidence: tuple[EvidencePointer, ...] = ()
    conflicts: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_testcase(self) -> "TestCaseKnowledge":
        keys = [field.key for field in self.fields]
        if len(keys) != len(set(keys)):
            raise ValueError("testcase fields must be unique")
        if self.test_case_id in self.dependency_case_ids:
            raise ValueError("a testcase cannot depend on itself")
        if len(self.dependency_case_ids) != len(set(self.dependency_case_ids)):
            raise ValueError("dependency_case_ids must be unique")
        if self.description is None and self.description_evidence:
            raise ValueError("description evidence requires a description")
        if self.description is not None and not self.description_evidence:
            raise ValueError("a description requires evidence")
        if not self.evidence:
            raise ValueError("a testcase requires row evidence")
        return self


class PortalRoute(KnowledgeModel):
    """A deduplicated route observed during the crawl."""

    url: str = Field(min_length=1)
    titles: tuple[str, ...] = ()
    state_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_route(self) -> "PortalRoute":
        if not self.state_ids:
            raise ValueError("a route requires at least one state")
        if len(self.state_ids) != len(set(self.state_ids)):
            raise ValueError("route state_ids must be unique")
        return self


class NetworkObservation(KnowledgeModel):
    """Deduplicated request/response behavior useful for later API synthesis."""

    method: str = Field(min_length=1)
    url: str = Field(min_length=1)
    resource_types: tuple[str, ...] = ()
    response_statuses: tuple[int, ...] = ()
    occurrences: int = Field(gt=0)
    request_body_artifact_ids: tuple[str, ...] = ()
    response_body_artifact_ids: tuple[str, ...] = ()
    sample_exchange_ids: tuple[str, ...] = ()


class KnowledgeCoverage(KnowledgeModel):
    """Completeness and conflict accounting for one normalized package."""

    source_status: ScrapeRunStatus
    source_bounded_complete: bool
    testcase_context_complete: bool
    states_examined: int = Field(ge=0)
    routes_normalized: int = Field(ge=0)
    tables_normalized: int = Field(ge=0)
    declared_test_cases: Optional[int] = Field(default=None, ge=0)
    test_cases_normalized: int = Field(ge=0)
    descriptions_captured: int = Field(ge=0)
    missing_description_ids: tuple[str, ...] = ()
    conflicting_test_case_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_counts(self) -> "KnowledgeCoverage":
        if self.descriptions_captured > self.test_cases_normalized:
            raise ValueError("descriptions_captured cannot exceed testcase count")
        if len(self.missing_description_ids) != (
            self.test_cases_normalized - self.descriptions_captured
        ):
            raise ValueError("missing description IDs do not match coverage counts")
        context_complete = bool(
            self.declared_test_cases is not None
            and self.declared_test_cases == self.test_cases_normalized
            and not self.missing_description_ids
            and not self.conflicting_test_case_ids
        )
        if self.testcase_context_complete != context_complete:
            raise ValueError("testcase_context_complete does not match coverage")
        return self


class PortalKnowledge(KnowledgeModel):
    """Deterministic handoff from browser evidence to retrieval and LLM phases."""

    schema_version: str = KNOWLEDGE_SCHEMA_VERSION
    source_kind: KnowledgeSourceKind = KnowledgeSourceKind.CRAWL
    source_run_id: str = Field(min_length=1)
    source_root_url: str = Field(min_length=1)
    source_started_at: datetime
    source_ended_at: Optional[datetime] = None
    normalized_at: datetime
    routes: tuple[PortalRoute, ...]
    tables: tuple[NormalizedTable, ...]
    test_cases: tuple[TestCaseKnowledge, ...]
    network: tuple[NetworkObservation, ...]
    coverage: KnowledgeCoverage

    @field_validator("source_started_at", "source_ended_at", "normalized_at")
    @classmethod
    def validate_timestamps(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("knowledge timestamps must include timezone information")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_collections(self) -> "PortalKnowledge":
        collections = (
            ("routes", [route.url for route in self.routes]),
            ("tables", [table.table_id for table in self.tables]),
            ("test_cases", [case.test_case_id for case in self.test_cases]),
            (
                "network",
                [(observation.method, observation.url) for observation in self.network],
            ),
        )
        for name, values in collections:
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        if self.coverage.routes_normalized != len(self.routes):
            raise ValueError("route coverage does not match normalized routes")
        if self.coverage.tables_normalized != len(self.tables):
            raise ValueError("table coverage does not match normalized tables")
        if self.coverage.test_cases_normalized != len(self.test_cases):
            raise ValueError("testcase coverage does not match normalized testcases")
        return self


class KnowledgeFile(KnowledgeModel):
    """One deterministic exported file and its digest."""

    name: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_size: int = Field(ge=0)


class KnowledgeExportManifest(KnowledgeModel):
    """Small manifest for an exported portal-knowledge package."""

    schema_version: str = KNOWLEDGE_SCHEMA_VERSION
    source_kind: KnowledgeSourceKind = KnowledgeSourceKind.CRAWL
    source_run_id: str = Field(min_length=1)
    normalized_at: datetime
    files: tuple[KnowledgeFile, ...]
    test_cases: int = Field(ge=0)
    descriptions: int = Field(ge=0)
    source_complete: bool
    testcase_context_complete: bool

    @field_validator("normalized_at")
    @classmethod
    def validate_normalized_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("normalized_at must include timezone information")
        return value.astimezone(timezone.utc)


__all__ = [
    "EvidencePointer",
    "KNOWLEDGE_SCHEMA_VERSION",
    "KnowledgeCoverage",
    "KnowledgeSourceKind",
    "KnowledgeExportManifest",
    "KnowledgeField",
    "KnowledgeFile",
    "NetworkObservation",
    "NormalizedControl",
    "NormalizedTable",
    "NormalizedTableRow",
    "PortalKnowledge",
    "PortalRoute",
    "TestCaseKnowledge",
]
