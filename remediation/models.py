"""Strict contracts for agentic repository remediation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from knowledge.models import SHA256_PATTERN
from synthesis.models import TokenUsage


REMEDIATION_SCHEMA_VERSION = "1.0"
REMEDIATION_PROMPT_VERSION = "1.0"


class RemediationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class AgentAction(str, Enum):
    SEARCH_REPOSITORY = "search_repository"
    READ_REPOSITORY_FILE = "read_repository_file"
    SEARCH_MCP = "search_mcp"
    FINAL = "final"


class FileOperation(str, Enum):
    CREATE = "create"
    REPLACE = "replace"


class ApplyStatus(str, Enum):
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


class RepositoryFileChange(RemediationModel):
    """One complete-file change guarded by compare-and-swap metadata."""

    path: str = Field(min_length=1)
    operation: FileOperation
    expected_sha256: Optional[str] = Field(default=None, pattern=SHA256_PATTERN)
    content: str
    rationale: str = Field(min_length=1)
    test_case_ids: tuple[str, ...]
    evidence_snippet_ids: tuple[str, ...]

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("repository paths must use POSIX separators")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
            raise ValueError("repository path must be relative and contained")
        if path.parts[0] == ".git":
            raise ValueError("repository changes cannot target .git")
        return str(path)

    @model_validator(mode="after")
    def validate_operation(self) -> "RepositoryFileChange":
        if self.operation == FileOperation.CREATE and self.expected_sha256 is not None:
            raise ValueError("create operations cannot have an expected digest")
        if self.operation == FileOperation.REPLACE and self.expected_sha256 is None:
            raise ValueError("replace operations require an expected digest")
        for name, values in (
            ("test_case_ids", self.test_case_ids),
            ("evidence_snippet_ids", self.evidence_snippet_ids),
        ):
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be non-empty and unique")
        return self


class RemediationProposal(RemediationModel):
    summary: str = Field(min_length=1)
    changes: tuple[RepositoryFileChange, ...]
    risks: tuple[str, ...] = ()
    verification_notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_changes(self) -> "RemediationProposal":
        if not self.changes:
            raise ValueError("a remediation proposal requires at least one change")
        paths = [change.path for change in self.changes]
        if len(paths) != len(set(paths)):
            raise ValueError("a remediation proposal cannot change one path twice")
        return self


class CodeAgentResponse(RemediationModel):
    """One structured thought/action boundary in the repository tool loop."""

    action: AgentAction
    rationale: str = Field(min_length=1)
    query: Optional[str] = None
    path: Optional[str] = None
    tool_name: Optional[str] = None
    proposal: Optional[RemediationProposal] = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> "CodeAgentResponse":
        expected = {
            AgentAction.SEARCH_REPOSITORY: (self.query is not None),
            AgentAction.READ_REPOSITORY_FILE: (self.path is not None),
            AgentAction.SEARCH_MCP: (
                self.query is not None and self.tool_name is not None
            ),
            AgentAction.FINAL: (self.proposal is not None),
        }[self.action]
        if not expected:
            raise ValueError(f"{self.action.value} lacks its required payload")
        if self.action != AgentAction.SEARCH_MCP and self.tool_name is not None:
            raise ValueError("tool_name is valid only for search_mcp")
        if (
            self.action
            not in {
                AgentAction.SEARCH_REPOSITORY,
                AgentAction.SEARCH_MCP,
            }
            and self.query is not None
        ):
            raise ValueError("query is invalid for this action")
        if self.action != AgentAction.READ_REPOSITORY_FILE and self.path is not None:
            raise ValueError("path is valid only for read_repository_file")
        if self.action != AgentAction.FINAL and self.proposal is not None:
            raise ValueError("proposal is valid only for final")
        return self


class ToolObservation(RemediationModel):
    sequence: int = Field(ge=1)
    action: AgentAction
    request: str = Field(min_length=1)
    result_sha256: str = Field(pattern=SHA256_PATTERN)
    result_characters: int = Field(ge=0)
    evidence_snippet_ids: tuple[str, ...] = ()
    error: Optional[str] = None


class RemediationCallRecord(RemediationModel):
    turn: int = Field(ge=1)
    action: AgentAction
    request_sha256: str = Field(pattern=SHA256_PATTERN)
    response_sha256: str = Field(pattern=SHA256_PATTERN)
    valid: bool
    validation_error: Optional[str] = None
    usage: TokenUsage = Field(default_factory=TokenUsage)

    @model_validator(mode="after")
    def validate_call(self) -> "RemediationCallRecord":
        if self.valid == (self.validation_error is not None):
            raise ValueError(
                "valid calls cannot have errors; invalid calls require one"
            )
        return self


class RemediationPlan(RemediationModel):
    schema_version: str = REMEDIATION_SCHEMA_VERSION
    plan_id: str = Field(pattern=SHA256_PATTERN)
    source_run_id: str = Field(min_length=1)
    source_synthesis_sha256: str = Field(pattern=SHA256_PATTERN)
    source_grounding_sha256: str = Field(pattern=SHA256_PATTERN)
    repository_id: str = Field(pattern=SHA256_PATTERN)
    objective: str = Field(min_length=1)
    selected_test_case_ids: tuple[str, ...]
    generated_at: datetime
    model: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    proposal: RemediationProposal
    observations: tuple[ToolObservation, ...]
    calls: tuple[RemediationCallRecord, ...]
    approval_required: bool = True

    @field_validator("generated_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include timezone information")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_plan(self) -> "RemediationPlan":
        if not self.selected_test_case_ids or len(self.selected_test_case_ids) != len(
            set(self.selected_test_case_ids)
        ):
            raise ValueError("selected testcase IDs must be non-empty and unique")
        if not self.approval_required:
            raise ValueError("repository remediation always requires approval")
        expected = remediation_plan_id(
            source_run_id=self.source_run_id,
            source_synthesis_sha256=self.source_synthesis_sha256,
            source_grounding_sha256=self.source_grounding_sha256,
            repository_id=self.repository_id,
            objective=self.objective,
            selected_test_case_ids=self.selected_test_case_ids,
            proposal=self.proposal,
        )
        if self.plan_id != expected:
            raise ValueError("plan_id does not match the immutable change set")
        return self


class VerificationResult(RemediationModel):
    argv: tuple[str, ...]
    exit_code: Optional[int] = None
    timed_out: bool = False
    duration_ms: int = Field(ge=0)
    stdout_sha256: str = Field(pattern=SHA256_PATTERN)
    stderr_sha256: str = Field(pattern=SHA256_PATTERN)
    passed: bool

    @model_validator(mode="after")
    def validate_result(self) -> "VerificationResult":
        if not self.argv:
            raise ValueError("verification argv cannot be empty")
        if self.passed != (not self.timed_out and self.exit_code == 0):
            raise ValueError("verification result does not match exit state")
        return self


class AppliedFile(RemediationModel):
    path: str = Field(min_length=1)
    before_sha256: Optional[str] = Field(default=None, pattern=SHA256_PATTERN)
    after_sha256: str = Field(pattern=SHA256_PATTERN)


class RemediationApplyReport(RemediationModel):
    schema_version: str = REMEDIATION_SCHEMA_VERSION
    plan_id: str = Field(pattern=SHA256_PATTERN)
    repository_id_before: str = Field(pattern=SHA256_PATTERN)
    applied_at: datetime
    status: ApplyStatus
    files: tuple[AppliedFile, ...]
    verification: tuple[VerificationResult, ...]
    failure_reason: Optional[str] = None

    @field_validator("applied_at")
    @classmethod
    def validate_applied_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("applied_at must include timezone information")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_status(self) -> "RemediationApplyReport":
        if self.status == ApplyStatus.APPLIED and self.failure_reason is not None:
            raise ValueError("applied reports cannot have a failure reason")
        if self.status != ApplyStatus.APPLIED and not self.failure_reason:
            raise ValueError("non-applied reports require a failure reason")
        return self


def remediation_plan_id(
    *,
    source_run_id: str,
    source_synthesis_sha256: str,
    source_grounding_sha256: str,
    repository_id: str,
    objective: str,
    selected_test_case_ids: tuple[str, ...],
    proposal: RemediationProposal,
) -> str:
    return _hash_json(
        {
            "prompt_version": REMEDIATION_PROMPT_VERSION,
            "source_run_id": source_run_id,
            "source_synthesis_sha256": source_synthesis_sha256,
            "source_grounding_sha256": source_grounding_sha256,
            "repository_id": repository_id,
            "objective": objective,
            "selected_test_case_ids": selected_test_case_ids,
            "proposal": proposal.model_dump(mode="json"),
        }
    )


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
    "AgentAction",
    "AppliedFile",
    "ApplyStatus",
    "CodeAgentResponse",
    "FileOperation",
    "REMEDIATION_PROMPT_VERSION",
    "RemediationApplyReport",
    "RemediationCallRecord",
    "RemediationPlan",
    "RemediationProposal",
    "RepositoryFileChange",
    "ToolObservation",
    "VerificationResult",
    "remediation_plan_id",
]
