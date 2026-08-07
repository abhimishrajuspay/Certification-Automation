"""Strict contracts for repository and MCP-grounded testcase context."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from knowledge.models import KnowledgeField, SHA256_PATTERN


GROUNDING_SCHEMA_VERSION = "1.0"


class GroundingModel(BaseModel):
    """Immutable base model for all grounding artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class GroundingSourceKind(str, Enum):
    """Supported external context source types."""

    REPOSITORY = "repository"
    MCP = "mcp"


class RepositoryCitation(GroundingModel):
    """Line-level citation into one content-hashed repository file."""

    repository_id: str = Field(pattern=SHA256_PATTERN)
    path: str = Field(min_length=1)
    file_sha256: str = Field(pattern=SHA256_PATTERN)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    excerpt_sha256: str = Field(pattern=SHA256_PATTERN)
    redacted: bool

    @model_validator(mode="after")
    def validate_location(self) -> "RepositoryCitation":
        if self.line_end < self.line_start:
            raise ValueError("repository citation line_end precedes line_start")
        if self.path.startswith(("/", "\\")) or ".." in self.path.split("/"):
            raise ValueError("repository citation path must be relative and contained")
        return self


class MCPDocumentReference(GroundingModel):
    """Document identity parsed from an MCP search result when available."""

    source: Optional[str] = None
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_identity(self) -> "MCPDocumentReference":
        if not any((self.source, self.document_id, self.chunk_id)):
            raise ValueError("an MCP document reference requires an identifier")
        return self


class MCPCallCitation(GroundingModel):
    """Audit citation for one read-only MCP tool response."""

    endpoint: str = Field(min_length=1)
    server_name: str = Field(min_length=1)
    server_version: str = Field(min_length=1)
    protocol_version: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    query: str = Field(min_length=1)
    arguments_sha256: str = Field(pattern=SHA256_PATTERN)
    response_sha256: str = Field(pattern=SHA256_PATTERN)
    retrieved_at: datetime
    documents: tuple[MCPDocumentReference, ...] = ()

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("MCP endpoint must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "MCP endpoint must not contain credentials, query, or fragment"
            )
        return value

    @field_validator("retrieved_at")
    @classmethod
    def validate_retrieved_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at must include timezone information")
        return value.astimezone(timezone.utc)


class GroundingSnippet(GroundingModel):
    """A bounded external context excerpt with exactly one source citation."""

    snippet_id: str = Field(pattern=SHA256_PATTERN)
    source_kind: GroundingSourceKind
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    relevance_score: int = Field(ge=0)
    repository: Optional[RepositoryCitation] = None
    mcp: Optional[MCPCallCitation] = None

    @model_validator(mode="after")
    def validate_source(self) -> "GroundingSnippet":
        if self.source_kind == GroundingSourceKind.REPOSITORY:
            if self.repository is None or self.mcp is not None:
                raise ValueError("repository snippet requires only repository citation")
            if self.repository.excerpt_sha256 != _hash_text(self.content):
                raise ValueError("repository excerpt digest does not match content")
            expected_id = _hash_json(
                {
                    "source": GroundingSourceKind.REPOSITORY.value,
                    "citation": self.repository.model_dump(mode="json"),
                    "content": self.content,
                }
            )
        elif self.mcp is None or self.repository is not None:
            raise ValueError("MCP snippet requires only MCP citation")
        else:
            expected_id = _hash_json(
                {
                    "source": GroundingSourceKind.MCP.value,
                    "tool": self.mcp.tool_name,
                    "arguments_sha256": self.mcp.arguments_sha256,
                    "response_sha256": self.mcp.response_sha256,
                    "documents": [
                        item.model_dump(mode="json") for item in self.mcp.documents
                    ],
                    "content": self.content,
                }
            )
        if self.snippet_id != expected_id:
            raise ValueError("snippet_id does not match cited content")
        return self


class PortalTestCaseContext(GroundingModel):
    """The normalized portal facts carried into grounding unchanged."""

    test_case_id: str = Field(min_length=1)
    fields: tuple[KnowledgeField, ...]
    dependency_case_ids: tuple[str, ...] = ()
    description: Optional[str] = None
    evidence_state_ids: tuple[str, ...]
    description_state_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_context(self) -> "PortalTestCaseContext":
        if not self.evidence_state_ids:
            raise ValueError("portal testcase context requires evidence states")
        if len(self.evidence_state_ids) != len(set(self.evidence_state_ids)):
            raise ValueError("portal evidence state IDs must be unique")
        if len(self.description_state_ids) != len(set(self.description_state_ids)):
            raise ValueError("description state IDs must be unique")
        return self


class GroundedTestCase(GroundingModel):
    """One testcase and references to its relevant shared grounding snippets."""

    context: PortalTestCaseContext
    retrieval_query: str = Field(min_length=1)
    repository_snippet_ids: tuple[str, ...] = ()
    mcp_snippet_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_snippet_ids(self) -> "GroundedTestCase":
        for name, values in (
            ("repository_snippet_ids", self.repository_snippet_ids),
            ("mcp_snippet_ids", self.mcp_snippet_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        return self


class RepositoryIndexSummary(GroundingModel):
    """Auditable summary of the bounded repository scan."""

    repository_id: str = Field(pattern=SHA256_PATTERN)
    root_name: str = Field(min_length=1)
    files_examined: int = Field(ge=0)
    files_indexed: int = Field(ge=0)
    files_skipped: int = Field(ge=0)
    bytes_indexed: int = Field(ge=0)
    suffixes: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_counts(self) -> "RepositoryIndexSummary":
        if self.files_indexed + self.files_skipped != self.files_examined:
            raise ValueError("repository file counts do not reconcile")
        return self


class MCPServerSummary(GroundingModel):
    """Connection and capability evidence for the configured MCP server."""

    configured: bool
    available: bool
    retrieval_complete: bool = False
    endpoint: Optional[str] = None
    server_name: Optional[str] = None
    server_version: Optional[str] = None
    protocol_version: Optional[str] = None
    discovered_tools: tuple[str, ...] = ()
    invoked_tools: tuple[str, ...] = ()
    calls_attempted: int = Field(default=0, ge=0)
    calls_succeeded: int = Field(default=0, ge=0)
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_state(self) -> "MCPServerSummary":
        identity = (self.server_name, self.server_version, self.protocol_version)
        if self.available and (not self.configured or not all(identity)):
            raise ValueError("available MCP server requires complete identity")
        if self.retrieval_complete and not self.available:
            raise ValueError("complete MCP retrieval requires an available server")
        if not self.configured and any(
            (
                self.endpoint,
                *identity,
                self.discovered_tools,
                self.invoked_tools,
                self.calls_attempted,
                self.calls_succeeded,
                self.retrieval_complete,
            )
        ):
            raise ValueError("unconfigured MCP server cannot have connection metadata")
        if self.calls_succeeded > self.calls_attempted:
            raise ValueError("successful MCP calls cannot exceed attempted calls")
        if len(self.discovered_tools) != len(set(self.discovered_tools)):
            raise ValueError("discovered MCP tools must be unique")
        if len(self.invoked_tools) != len(set(self.invoked_tools)):
            raise ValueError("invoked MCP tools must be unique")
        return self


class GroundingCoverage(GroundingModel):
    """Coverage gates for the handoff to structured LLM synthesis."""

    source_testcase_context_complete: bool
    mcp_required: bool
    test_cases: int = Field(ge=0)
    repository_grounded: int = Field(ge=0)
    mcp_grounded: int = Field(ge=0)
    externally_grounded: int = Field(ge=0)
    missing_external_context_ids: tuple[str, ...] = ()
    grounding_complete: bool
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_coverage(self) -> "GroundingCoverage":
        for value in (
            self.repository_grounded,
            self.mcp_grounded,
            self.externally_grounded,
        ):
            if value > self.test_cases:
                raise ValueError("grounding coverage cannot exceed testcase count")
        if len(self.missing_external_context_ids) != (
            self.test_cases - self.externally_grounded
        ):
            raise ValueError("missing external context IDs do not match coverage")
        if self.grounding_complete and (
            not self.source_testcase_context_complete
            or self.missing_external_context_ids
        ):
            raise ValueError("complete grounding cannot have incomplete source context")
        return self


class GroundingPackage(GroundingModel):
    """Cited repository/MCP context ready for a later constrained LLM phase."""

    schema_version: str = GROUNDING_SCHEMA_VERSION
    source_run_id: str = Field(min_length=1)
    source_knowledge_sha256: str = Field(pattern=SHA256_PATTERN)
    grounded_at: datetime
    repository: RepositoryIndexSummary
    mcp: MCPServerSummary
    snippets: tuple[GroundingSnippet, ...]
    test_cases: tuple[GroundedTestCase, ...]
    coverage: GroundingCoverage

    @field_validator("grounded_at")
    @classmethod
    def validate_grounded_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("grounded_at must include timezone information")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_references(self) -> "GroundingPackage":
        snippet_ids = [snippet.snippet_id for snippet in self.snippets]
        if len(snippet_ids) != len(set(snippet_ids)):
            raise ValueError("grounding snippets must be unique")
        by_id = {snippet.snippet_id: snippet for snippet in self.snippets}
        known = set(by_id)
        case_ids = [case.context.test_case_id for case in self.test_cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("grounded testcases must be unique")
        for case in self.test_cases:
            referenced = set(case.repository_snippet_ids + case.mcp_snippet_ids)
            unknown = referenced - known
            if unknown:
                raise ValueError(
                    f"testcase references unknown snippets: {sorted(unknown)}"
                )
            if any(
                by_id[snippet_id].source_kind != GroundingSourceKind.REPOSITORY
                for snippet_id in case.repository_snippet_ids
            ):
                raise ValueError(
                    "repository snippet reference has the wrong source kind"
                )
            if any(
                by_id[snippet_id].source_kind != GroundingSourceKind.MCP
                for snippet_id in case.mcp_snippet_ids
            ):
                raise ValueError("MCP snippet reference has the wrong source kind")
        if self.coverage.test_cases != len(self.test_cases):
            raise ValueError("grounding coverage testcase count does not match package")
        repository_grounded = sum(
            bool(case.repository_snippet_ids) for case in self.test_cases
        )
        mcp_grounded = sum(bool(case.mcp_snippet_ids) for case in self.test_cases)
        externally_grounded = sum(
            bool(case.repository_snippet_ids or case.mcp_snippet_ids)
            for case in self.test_cases
        )
        missing_ids = tuple(
            case.context.test_case_id
            for case in self.test_cases
            if not (case.repository_snippet_ids or case.mcp_snippet_ids)
        )
        if self.coverage.repository_grounded != repository_grounded:
            raise ValueError("repository grounding coverage does not match package")
        if self.coverage.mcp_grounded != mcp_grounded:
            raise ValueError("MCP grounding coverage does not match package")
        if self.coverage.externally_grounded != externally_grounded:
            raise ValueError("external grounding coverage does not match package")
        if self.coverage.missing_external_context_ids != missing_ids:
            raise ValueError("missing external context IDs do not match package")
        expected_complete = bool(
            self.coverage.source_testcase_context_complete
            and not self.coverage.missing_external_context_ids
            and (
                not self.coverage.mcp_required
                or (
                    self.mcp.retrieval_complete
                    and (not self.test_cases or self.mcp.calls_succeeded > 0)
                )
            )
        )
        if self.coverage.grounding_complete != expected_complete:
            raise ValueError("grounding_complete does not match package evidence")
        return self


class GroundingFile(GroundingModel):
    """One exported grounding file and its content digest."""

    name: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_size: int = Field(ge=0)


class GroundingExportManifest(GroundingModel):
    """Small integrity manifest for one exported grounding package."""

    schema_version: str = GROUNDING_SCHEMA_VERSION
    source_run_id: str = Field(min_length=1)
    source_knowledge_sha256: str = Field(pattern=SHA256_PATTERN)
    grounded_at: datetime
    files: tuple[GroundingFile, ...]
    test_cases: int = Field(ge=0)
    snippets: int = Field(ge=0)
    grounding_complete: bool

    @field_validator("grounded_at")
    @classmethod
    def validate_manifest_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("grounded_at must include timezone information")
        return value.astimezone(timezone.utc)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_json(value: object) -> str:
    data = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "GROUNDING_SCHEMA_VERSION",
    "GroundedTestCase",
    "GroundingCoverage",
    "GroundingExportManifest",
    "GroundingFile",
    "GroundingPackage",
    "GroundingSnippet",
    "GroundingSourceKind",
    "MCPCallCitation",
    "MCPDocumentReference",
    "MCPServerSummary",
    "PortalTestCaseContext",
    "RepositoryCitation",
    "RepositoryIndexSummary",
]
