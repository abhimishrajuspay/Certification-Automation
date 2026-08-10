"""Strict contracts for repository-wide certification campaigns."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from grounding.models import GroundingSnippet
from knowledge.models import SHA256_PATTERN
from remediation.models import RemediationProposal, RepositoryFileChange
from synthesis.models import TokenUsage


CAMPAIGN_SCHEMA_VERSION = "1.0"
CAMPAIGN_PROMPT_VERSION = "1.0"


class CampaignModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class CaseSupportStatus(str, Enum):
    SUPPORTED_AS_IS = "supported_as_is"
    SUPPORTED_AFTER_CHANGE = "supported_after_change"
    UNSUPPORTED_MISSING_REQUIREMENT = "unsupported_missing_requirement"
    NEEDS_REVIEW = "needs_review"


class CampaignAction(str, Enum):
    LIST_TEST_CASES = "list_test_cases"
    READ_TEST_CASES = "read_test_cases"
    READ_EVIDENCE = "read_evidence"
    SEARCH_REPOSITORY = "search_repository"
    READ_REPOSITORY_FILE = "read_repository_file"
    SEARCH_MCP = "search_mcp"
    RECORD_ASSESSMENTS = "record_assessments"
    STAGE_FILE = "stage_file"
    FINAL = "final"


class CampaignGroup(CampaignModel):
    group_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    test_case_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_cases(self) -> "CampaignGroup":
        if not self.test_case_ids or len(self.test_case_ids) != len(
            set(self.test_case_ids)
        ):
            raise ValueError("campaign group testcase IDs must be non-empty and unique")
        return self


class CampaignAssessment(CampaignModel):
    test_case_id: str = Field(min_length=1)
    status: CaseSupportStatus
    rationale: str = Field(min_length=1)
    required_capabilities: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()
    repository_paths: tuple[str, ...] = ()
    portal_evidence_state_ids: tuple[str, ...] = ()
    evidence_snippet_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_assessment(self) -> "CampaignAssessment":
        for label, values in (
            ("required_capabilities", self.required_capabilities),
            ("missing_capabilities", self.missing_capabilities),
            ("repository_paths", self.repository_paths),
            ("portal_evidence_state_ids", self.portal_evidence_state_ids),
            ("evidence_snippet_ids", self.evidence_snippet_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        if (
            self.status == CaseSupportStatus.SUPPORTED_AS_IS
            and self.missing_capabilities
        ):
            raise ValueError("supported_as_is cannot retain missing capabilities")
        if self.status != CaseSupportStatus.SUPPORTED_AS_IS and not (
            self.missing_capabilities or self.required_capabilities
        ):
            raise ValueError(
                "non-supported assessments require an explicit capability reason"
            )
        return self


class CampaignConclusion(CampaignModel):
    summary: str = Field(min_length=1)
    risks: tuple[str, ...] = ()
    verification_notes: tuple[str, ...] = ()


class CampaignAgentResponse(CampaignModel):
    """One structured action from the repository-wide campaign agent."""

    action: CampaignAction
    rationale: str = Field(min_length=1)
    group_id: Optional[str] = None
    offset: Optional[int] = Field(default=None, ge=0)
    limit: Optional[int] = Field(default=None, ge=1, le=50)
    test_case_ids: Optional[tuple[str, ...]] = None
    snippet_id: Optional[str] = None
    query: Optional[str] = None
    path: Optional[str] = None
    tool_name: Optional[str] = None
    assessments: Optional[tuple[CampaignAssessment, ...]] = None
    change: Optional[RepositoryFileChange] = None
    conclusion: Optional[CampaignConclusion] = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> "CampaignAgentResponse":
        valid = {
            CampaignAction.LIST_TEST_CASES: self.group_id is not None,
            CampaignAction.READ_TEST_CASES: bool(self.test_case_ids),
            CampaignAction.READ_EVIDENCE: self.snippet_id is not None,
            CampaignAction.SEARCH_REPOSITORY: self.query is not None,
            CampaignAction.READ_REPOSITORY_FILE: self.path is not None,
            CampaignAction.SEARCH_MCP: (
                self.query is not None and self.tool_name is not None
            ),
            CampaignAction.RECORD_ASSESSMENTS: bool(self.assessments),
            CampaignAction.STAGE_FILE: self.change is not None,
            CampaignAction.FINAL: self.conclusion is not None,
        }[self.action]
        if not valid:
            raise ValueError(f"{self.action.value} lacks its required payload")

        allowed: dict[CampaignAction, set[str]] = {
            CampaignAction.LIST_TEST_CASES: {"group_id", "offset", "limit"},
            CampaignAction.READ_TEST_CASES: {"test_case_ids"},
            CampaignAction.READ_EVIDENCE: {"snippet_id"},
            CampaignAction.SEARCH_REPOSITORY: {"query"},
            CampaignAction.READ_REPOSITORY_FILE: {"path"},
            CampaignAction.SEARCH_MCP: {"query", "tool_name"},
            CampaignAction.RECORD_ASSESSMENTS: {"assessments"},
            CampaignAction.STAGE_FILE: {"change"},
            CampaignAction.FINAL: {"conclusion"},
        }
        present = {
            name
            for name in (
                "group_id",
                "offset",
                "limit",
                "test_case_ids",
                "snippet_id",
                "query",
                "path",
                "tool_name",
                "assessments",
                "change",
                "conclusion",
            )
            if getattr(self, name) is not None
        }
        unexpected = present - allowed[self.action]
        if unexpected:
            raise ValueError(
                f"{self.action.value} contains invalid fields: {sorted(unexpected)}"
            )
        return self


class CampaignCallRecord(CampaignModel):
    turn: int = Field(ge=1)
    action: CampaignAction
    request_sha256: str = Field(pattern=SHA256_PATTERN)
    response_sha256: str = Field(pattern=SHA256_PATTERN)
    valid: bool
    validation_error: Optional[str] = None
    usage: TokenUsage = Field(default_factory=TokenUsage)

    @model_validator(mode="after")
    def validate_call(self) -> "CampaignCallRecord":
        if self.valid == (self.validation_error is not None):
            raise ValueError(
                "valid calls cannot have errors; invalid calls require one"
            )
        return self


class CampaignToolObservation(CampaignModel):
    sequence: int = Field(ge=1)
    turn: int = Field(ge=1)
    action: CampaignAction
    request: str = Field(min_length=1)
    result_sha256: str = Field(pattern=SHA256_PATTERN)
    result_characters: int = Field(ge=0)
    evidence_snippet_ids: tuple[str, ...] = ()
    error: Optional[str] = None


class CampaignPlan(CampaignModel):
    schema_version: str = CAMPAIGN_SCHEMA_VERSION
    plan_id: str = Field(pattern=SHA256_PATTERN)
    source_run_id: str = Field(min_length=1)
    source_grounding_sha256: str = Field(pattern=SHA256_PATTERN)
    repository_id: str = Field(pattern=SHA256_PATTERN)
    objective: str = Field(min_length=1)
    selected_test_case_ids: tuple[str, ...]
    groups: tuple[CampaignGroup, ...]
    generated_at: datetime
    model: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    conclusion: CampaignConclusion
    assessments: tuple[CampaignAssessment, ...]
    retrieved_snippets: tuple[GroundingSnippet, ...] = ()
    proposal: Optional[RemediationProposal] = None
    observations: tuple[CampaignToolObservation, ...] = ()
    calls: tuple[CampaignCallRecord, ...] = ()
    approval_required: bool

    @field_validator("generated_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include timezone information")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_plan(self) -> "CampaignPlan":
        selected = self.selected_test_case_ids
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("selected testcase IDs must be non-empty and unique")
        assessed = tuple(item.test_case_id for item in self.assessments)
        if assessed != selected:
            raise ValueError(
                "campaign assessments must cover selected testcases in source order"
            )
        grouped = tuple(case for group in self.groups for case in group.test_case_ids)
        if set(grouped) != set(selected) or len(grouped) != len(set(grouped)):
            raise ValueError("campaign groups must partition selected testcases")
        if self.approval_required != (self.proposal is not None):
            raise ValueError("approval is required exactly when code changes exist")
        changed_cases: set[str] = set()
        if self.proposal is not None:
            for change in self.proposal.changes:
                if not set(change.test_case_ids).issubset(selected):
                    raise ValueError("campaign change cites an unselected testcase")
                changed_cases.update(change.test_case_ids)
        unsupported_changes = [
            item.test_case_id
            for item in self.assessments
            if item.status == CaseSupportStatus.SUPPORTED_AFTER_CHANGE
            and item.test_case_id not in changed_cases
        ]
        if unsupported_changes:
            raise ValueError(
                "supported_after_change assessments require a cited file change"
            )
        expected = campaign_plan_id(
            source_run_id=self.source_run_id,
            source_grounding_sha256=self.source_grounding_sha256,
            repository_id=self.repository_id,
            objective=self.objective,
            selected_test_case_ids=self.selected_test_case_ids,
            assessments=self.assessments,
            proposal=self.proposal,
        )
        if self.plan_id != expected:
            raise ValueError("plan_id does not match the immutable campaign result")
        return self


class CampaignCheckpoint(CampaignModel):
    source_run_id: str = Field(min_length=1)
    source_grounding_sha256: str = Field(pattern=SHA256_PATTERN)
    repository_id: str = Field(pattern=SHA256_PATTERN)
    configuration_sha256: str = Field(pattern=SHA256_PATTERN)
    model: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    updated_at: datetime
    selected_test_case_ids: tuple[str, ...]
    read_test_case_ids: tuple[str, ...] = ()
    read_evidence_snippet_ids: tuple[str, ...] = ()
    read_repository_paths: tuple[str, ...] = ()
    assessments: tuple[CampaignAssessment, ...] = ()
    retrieved_snippets: tuple[GroundingSnippet, ...] = ()
    staged_changes: tuple[RepositoryFileChange, ...] = ()
    calls: tuple[CampaignCallRecord, ...] = ()
    observations: tuple[CampaignToolObservation, ...] = ()
    errors: tuple[str, ...] = ()

    @field_validator("updated_at")
    @classmethod
    def validate_checkpoint_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("updated_at must include timezone information")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_progress(self) -> "CampaignCheckpoint":
        for label, values in (
            ("selected_test_case_ids", self.selected_test_case_ids),
            ("read_test_case_ids", self.read_test_case_ids),
            ("read_evidence_snippet_ids", self.read_evidence_snippet_ids),
            ("read_repository_paths", self.read_repository_paths),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        if not set(self.read_test_case_ids).issubset(self.selected_test_case_ids):
            raise ValueError("checkpoint read testcases must be selected")
        assessment_ids = [item.test_case_id for item in self.assessments]
        if len(assessment_ids) != len(set(assessment_ids)):
            raise ValueError("checkpoint assessments must be unique")
        if not set(assessment_ids).issubset(self.selected_test_case_ids):
            raise ValueError("checkpoint assessments must cite selected testcases")
        paths = [item.path for item in self.staged_changes]
        if len(paths) != len(set(paths)):
            raise ValueError("checkpoint staged file paths must be unique")
        if any(
            not set(item.test_case_ids).issubset(self.selected_test_case_ids)
            for item in self.staged_changes
        ):
            raise ValueError("checkpoint changes must cite selected testcases")
        sequences = [item.sequence for item in self.observations]
        if sequences != list(range(1, len(sequences) + 1)):
            raise ValueError("checkpoint observation sequence must be contiguous")
        return self


def campaign_plan_id(
    *,
    source_run_id: str,
    source_grounding_sha256: str,
    repository_id: str,
    objective: str,
    selected_test_case_ids: tuple[str, ...],
    assessments: tuple[CampaignAssessment, ...],
    proposal: Optional[RemediationProposal],
) -> str:
    return _hash_json(
        {
            "prompt_version": CAMPAIGN_PROMPT_VERSION,
            "source_run_id": source_run_id,
            "source_grounding_sha256": source_grounding_sha256,
            "repository_id": repository_id,
            "objective": objective,
            "selected_test_case_ids": selected_test_case_ids,
            "assessments": [item.model_dump(mode="json") for item in assessments],
            "proposal": proposal.model_dump(mode="json") if proposal else None,
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
    "CAMPAIGN_PROMPT_VERSION",
    "CAMPAIGN_SCHEMA_VERSION",
    "CampaignAction",
    "CampaignAgentResponse",
    "CampaignAssessment",
    "CampaignCallRecord",
    "CampaignCheckpoint",
    "CampaignConclusion",
    "CampaignGroup",
    "CampaignPlan",
    "CampaignToolObservation",
    "CaseSupportStatus",
    "campaign_plan_id",
]
