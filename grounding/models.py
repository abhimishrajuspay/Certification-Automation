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


GROUNDING_SCHEMA_VERSION = "1.1"


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


class GroundingStrategy(str, Enum):
    """Supported external-evidence retrieval strategies."""

    DETERMINISTIC = "deterministic"
    AGENTIC = "agentic"


class GroundingAgentAction(str, Enum):
    """One bounded action returned by the grounding model."""

    SEARCH = "search"
    FINAL = "final"


class GroundingAgentToolKind(str, Enum):
    """Read-only tool families exposed to the grounding model."""

    SEARCH_REPOSITORY = "search_repository"
    CALL_MCP = "call_mcp"


class GroundingAgentToolRequest(GroundingModel):
    """One model-selected, bounded repository or MCP request."""

    request_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    kind: GroundingAgentToolKind
    test_case_ids: tuple[str, ...] = Field(min_length=1)
    query: Optional[str] = Field(default=None, min_length=1, max_length=2_000)
    tool_name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    arguments: Optional[dict[str, object]] = None

    @model_validator(mode="after")
    def validate_request(self) -> "GroundingAgentToolRequest":
        if len(self.test_case_ids) != len(set(self.test_case_ids)):
            raise ValueError("grounding tool testcase IDs must be unique")
        if self.kind == GroundingAgentToolKind.SEARCH_REPOSITORY:
            if self.query is None or self.tool_name is not None or self.arguments:
                raise ValueError(
                    "repository search requires only a query and testcase IDs"
                )
        elif self.tool_name is None or self.arguments is None or self.query is not None:
            raise ValueError(
                "MCP call requires only tool_name, arguments, and testcase IDs"
            )
        if self.arguments is not None:
            try:
                encoded = json.dumps(
                    self.arguments,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("MCP arguments must be JSON serializable") from exc
            if len(encoded) > 10_000:
                raise ValueError("MCP arguments exceed the grounding size limit")
        return self


class GroundingAgentSelection(GroundingModel):
    """Evidence selected for one testcase after the model has read it."""

    test_case_id: str = Field(min_length=1)
    snippet_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_selection(self) -> "GroundingAgentSelection":
        if len(self.snippet_ids) != len(set(self.snippet_ids)):
            raise ValueError("selected grounding snippet IDs must be unique")
        if not self.snippet_ids and not self.limitations:
            raise ValueError("a selection without evidence requires a limitation")
        return self


class GroundingAgentDecision(GroundingModel):
    """Strict structured output for one agentic grounding turn."""

    action: GroundingAgentAction
    rationale: str = Field(min_length=1, max_length=1_000)
    requests: tuple[GroundingAgentToolRequest, ...] = ()
    selections: tuple[GroundingAgentSelection, ...] = ()

    @model_validator(mode="after")
    def validate_action_payload(self) -> "GroundingAgentDecision":
        if self.action == GroundingAgentAction.SEARCH:
            if not self.requests or self.selections:
                raise ValueError("search requires requests and no selections")
            request_ids = [item.request_id for item in self.requests]
            if len(request_ids) != len(set(request_ids)):
                raise ValueError("grounding request IDs must be unique per turn")
        elif not self.selections or self.requests:
            raise ValueError("final requires selections and no requests")
        selection_ids = [item.test_case_id for item in self.selections]
        if len(selection_ids) != len(set(selection_ids)):
            raise ValueError("grounding selections must have unique testcase IDs")
        return self


class GroundingTokenUsage(GroundingModel):
    """Provider token metrics retained without prompt or response content."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> "GroundingTokenUsage":
        if self.total_tokens < self.prompt_tokens + self.completion_tokens:
            raise ValueError("grounding token total is smaller than its components")
        return self


class GroundingAgentCallRecord(GroundingModel):
    """Secret-free audit record for one model grounding turn."""

    call_id: str = Field(pattern=SHA256_PATTERN)
    batch_id: str = Field(pattern=SHA256_PATTERN)
    turn: int = Field(ge=1)
    test_case_ids: tuple[str, ...] = Field(min_length=1)
    request_sha256: str = Field(pattern=SHA256_PATTERN)
    response_sha256: str = Field(pattern=SHA256_PATTERN)
    valid: bool
    validation_error: Optional[str] = Field(default=None, max_length=2_000)
    usage: GroundingTokenUsage = GroundingTokenUsage()

    @model_validator(mode="after")
    def validate_record(self) -> "GroundingAgentCallRecord":
        if len(self.test_case_ids) != len(set(self.test_case_ids)) or any(
            not value for value in self.test_case_ids
        ):
            raise ValueError("grounding call testcase IDs must be non-empty and unique")
        if self.valid == (self.validation_error is not None):
            raise ValueError(
                "valid grounding calls cannot have errors; invalid calls require one"
            )
        expected = _hash_json(
            {
                "batch_id": self.batch_id,
                "turn": self.turn,
                "request_sha256": self.request_sha256,
                "response_sha256": self.response_sha256,
            }
        )
        if self.call_id != expected:
            raise ValueError("grounding call ID does not match its evidence")
        return self


class GroundingAgentObservation(GroundingModel):
    """One model-selected read-only tool execution and its cited result IDs."""

    batch_id: str = Field(pattern=SHA256_PATTERN)
    turn: int = Field(ge=1)
    request_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    kind: GroundingAgentToolKind
    test_case_ids: tuple[str, ...] = Field(min_length=1)
    tool_name: Optional[str] = None
    request_sha256: str = Field(pattern=SHA256_PATTERN)
    snippet_ids: tuple[str, ...] = ()
    error: Optional[str] = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def validate_observation(self) -> "GroundingAgentObservation":
        if len(self.test_case_ids) != len(set(self.test_case_ids)) or any(
            not value for value in self.test_case_ids
        ):
            raise ValueError(
                "grounding observation testcase IDs must be non-empty and unique"
            )
        if len(self.snippet_ids) != len(set(self.snippet_ids)):
            raise ValueError("grounding observation snippet IDs must be unique")
        if self.kind == GroundingAgentToolKind.CALL_MCP:
            if self.tool_name is None:
                raise ValueError("MCP grounding observations require a tool name")
        elif self.tool_name is not None:
            raise ValueError(
                "repository grounding observations cannot name an MCP tool"
            )
        return self


class GroundingProviderSummary(GroundingModel):
    """Aggregate agentic-grounding model configuration and usage."""

    endpoint: str = Field(min_length=1)
    model: str = Field(min_length=1)
    response_format: str = Field(pattern=r"^(json_schema|json_object)$")
    prompt_version: str = "1.0"
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    calls_completed: int = Field(ge=0)
    valid_responses: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_counts(self) -> "GroundingProviderSummary":
        if self.valid_responses > self.calls_completed:
            raise ValueError("valid grounding responses cannot exceed completed calls")
        return self

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("grounding provider endpoint must be absolute HTTP(S)")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("grounding provider endpoint cannot contain secrets")
        return value


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
    strategy: GroundingStrategy = GroundingStrategy.DETERMINISTIC
    provider: Optional[GroundingProviderSummary] = None
    agent_calls: tuple[GroundingAgentCallRecord, ...] = ()
    agent_observations: tuple[GroundingAgentObservation, ...] = ()

    @field_validator("grounded_at")
    @classmethod
    def validate_grounded_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("grounded_at must include timezone information")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_references(self) -> "GroundingPackage":
        if self.strategy == GroundingStrategy.DETERMINISTIC:
            if self.provider is not None or self.agent_calls or self.agent_observations:
                raise ValueError(
                    "deterministic grounding cannot contain agent model records"
                )
        elif self.provider is None:
            raise ValueError("agentic grounding requires a provider summary")
        snippet_ids = [snippet.snippet_id for snippet in self.snippets]
        if len(snippet_ids) != len(set(snippet_ids)):
            raise ValueError("grounding snippets must be unique")
        by_id = {snippet.snippet_id: snippet for snippet in self.snippets}
        known = set(by_id)
        case_ids = [case.context.test_case_id for case in self.test_cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("grounded testcases must be unique")
        known_case_ids = set(case_ids)
        call_ids = [record.call_id for record in self.agent_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("grounding agent call records must be unique")
        valid_turns = {
            (record.batch_id, record.turn)
            for record in self.agent_calls
            if record.valid
        }
        for record in self.agent_calls:
            if not set(record.test_case_ids).issubset(known_case_ids):
                raise ValueError("grounding call references a non-package testcase")
        for observation in self.agent_observations:
            if not set(observation.test_case_ids).issubset(known_case_ids):
                raise ValueError(
                    "grounding observation references a non-package testcase"
                )
            if not set(observation.snippet_ids).issubset(known):
                raise ValueError("grounding observation references an unknown snippet")
            if (observation.batch_id, observation.turn) not in valid_turns:
                raise ValueError(
                    "grounding observation has no valid model decision turn"
                )
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
        if self.provider is not None:
            if self.provider.calls_completed != len(self.agent_calls):
                raise ValueError("provider call count does not match agent records")
            if self.provider.valid_responses != sum(
                record.valid for record in self.agent_calls
            ):
                raise ValueError("provider valid-response count does not match calls")
            if self.provider.prompt_tokens != sum(
                record.usage.prompt_tokens for record in self.agent_calls
            ):
                raise ValueError("provider prompt-token count does not match calls")
            if self.provider.completion_tokens != sum(
                record.usage.completion_tokens for record in self.agent_calls
            ):
                raise ValueError("provider completion-token count does not match calls")
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
    "GroundingAgentAction",
    "GroundingAgentCallRecord",
    "GroundingAgentDecision",
    "GroundingAgentObservation",
    "GroundingAgentSelection",
    "GroundingAgentToolKind",
    "GroundingAgentToolRequest",
    "GroundedTestCase",
    "GroundingCoverage",
    "GroundingExportManifest",
    "GroundingFile",
    "GroundingPackage",
    "GroundingProviderSummary",
    "GroundingSnippet",
    "GroundingSourceKind",
    "GroundingStrategy",
    "GroundingTokenUsage",
    "MCPCallCitation",
    "MCPDocumentReference",
    "MCPServerSummary",
    "PortalTestCaseContext",
    "RepositoryCitation",
    "RepositoryIndexSummary",
]
