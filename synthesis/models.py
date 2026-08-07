"""Strict contracts for cited testcase execution-specification synthesis."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from urllib.parse import urlsplit
from xml.etree import ElementTree

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from knowledge.models import SHA256_PATTERN


SYNTHESIS_SCHEMA_VERSION = "1.0"
PROMPT_VERSION = "1.0"
_ENVIRONMENT_VALUE = re.compile(r"^\{\{[A-Za-z_][A-Za-z0-9_]*\}\}$")
_TEMPLATE_VALUE = re.compile(r"\{\{[A-Za-z_][A-Za-z0-9_]*\}\}")
_SENSITIVE_HEADER_NAMES = {
    "api-key",
    "apikey",
    "authorization",
    "cookie",
    "proxy-authorization",
    "x-api-key",
    "x-auth-token",
    "x-client-secret",
    "x-vendor-key",
}


class SynthesisModel(BaseModel):
    """Immutable base model for Phase 9 artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class HTTPMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class RequestBodyMode(str, Enum):
    NONE = "none"
    JSON = "json"
    XML = "xml"
    FORM_URLENCODED = "form_urlencoded"
    RAW = "raw"


class AssertionSource(str, Enum):
    HTTP_STATUS = "http_status"
    RESPONSE_HEADER = "response_header"
    JSON_PATH = "json_path"
    XPATH = "xpath"
    RESPONSE_BODY = "response_body"
    RESPONSE_TIME = "response_time"


class AssertionOperator(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    MATCHES = "matches"
    LESS_THAN = "less_than"
    GREATER_THAN = "greater_than"


class SynthesisDisposition(str, Enum):
    READY = "ready"
    NEEDS_REVIEW = "needs_review"
    BLOCKED = "blocked"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TemplateVariableSource(str, Enum):
    ENVIRONMENT = "environment"
    PORTAL_FIELD = "portal_field"
    EVIDENCE_LITERAL = "evidence_literal"
    GENERATED = "generated"
    DEPENDENCY = "dependency"


class GeneratedValueKind(str, Enum):
    UUID = "uuid"
    TIMESTAMP = "timestamp"
    RANDOM_ALPHANUMERIC = "random_alphanumeric"


class NamedValue(SynthesisModel):
    """One header or query parameter in a future Postman request."""

    name: str = Field(min_length=1)
    value: str
    enabled: bool = True
    description: Optional[str] = None


class RequestBodySpec(SynthesisModel):
    """A syntactically checked request body template."""

    mode: RequestBodyMode
    content_type: Optional[str] = None
    template: Optional[str] = None

    @model_validator(mode="after")
    def validate_body(self) -> "RequestBodySpec":
        if self.mode == RequestBodyMode.NONE:
            if self.content_type is not None or self.template is not None:
                raise ValueError("a body with mode=none cannot carry content")
            return self
        if not self.template:
            raise ValueError("a non-empty request body mode requires a template")
        if self.mode == RequestBodyMode.XML and re.search(
            r"<!\s*(?:DOCTYPE|ENTITY)", self.template, re.IGNORECASE
        ):
            raise ValueError("XML request templates cannot contain DTD declarations")
        if self.mode == RequestBodyMode.JSON:
            try:
                json.loads(self.template)
            except json.JSONDecodeError as exc:
                raise ValueError("JSON request template is not valid JSON") from exc
        elif self.mode == RequestBodyMode.XML:
            try:
                ElementTree.fromstring(self.template)
            except ElementTree.ParseError as exc:
                raise ValueError("XML request template is not well formed") from exc
        return self


class HTTPRequestSpec(SynthesisModel):
    """Transport-neutral HTTP request definition for Phase 10 rendering."""

    method: HTTPMethod
    path: str = Field(min_length=1)
    headers: tuple[NamedValue, ...] = ()
    query_parameters: tuple[NamedValue, ...] = ()
    body: RequestBodySpec = Field(
        default_factory=lambda: RequestBodySpec(mode=RequestBodyMode.NONE)
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc:
            raise ValueError("request path must not contain a fixed origin")
        if not value.startswith("/"):
            raise ValueError("request path must begin with /")
        if parsed.fragment:
            raise ValueError("request path must not contain a fragment")
        if parsed.query:
            raise ValueError("query values belong in query_parameters")
        return value

    @model_validator(mode="after")
    def validate_parameters(self) -> "HTTPRequestSpec":
        for label, parameters in (
            ("header", self.headers),
            ("query parameter", self.query_parameters),
        ):
            names = [item.name.casefold() for item in parameters]
            if len(names) != len(set(names)):
                raise ValueError(f"{label} names must be unique")
        for header in self.headers:
            if (
                header.enabled
                and header.name.casefold() in _SENSITIVE_HEADER_NAMES
                and not _ENVIRONMENT_VALUE.fullmatch(header.value)
            ):
                raise ValueError(
                    f"sensitive header {header.name!r} must use one environment "
                    "placeholder such as {{API_KEY}}"
                )
        return self


class ResponseAssertion(SynthesisModel):
    """One deterministic assertion to render as a Postman test."""

    source: AssertionSource
    operator: AssertionOperator
    target: Optional[str] = None
    expected: Optional[str] = None
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_assertion(self) -> "ResponseAssertion":
        value_free = {
            AssertionOperator.EXISTS,
            AssertionOperator.NOT_EXISTS,
        }
        if self.operator in value_free and self.expected is not None:
            raise ValueError("existence assertions cannot define expected")
        if self.operator not in value_free and self.expected is None:
            raise ValueError("comparison assertions require expected")
        if self.source == AssertionSource.HTTP_STATUS and self.target is not None:
            raise ValueError("HTTP status assertions do not use target")
        if self.source != AssertionSource.HTTP_STATUS and not self.target:
            raise ValueError("non-status assertions require target")
        return self


class TemplateVariableBinding(SynthesisModel):
    """Typed source for one placeholder used by a request or assertion."""

    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    source: TemplateVariableSource
    value: Optional[str] = None
    source_key: Optional[str] = None
    dependency_case_id: Optional[str] = None
    generator: Optional[GeneratedValueKind] = None
    sensitive: bool = False
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source(self) -> "TemplateVariableBinding":
        if self.source == TemplateVariableSource.ENVIRONMENT:
            if any(
                item is not None
                for item in (
                    self.value,
                    self.source_key,
                    self.dependency_case_id,
                    self.generator,
                )
            ):
                raise ValueError("environment bindings cannot embed source values")
        elif self.source == TemplateVariableSource.PORTAL_FIELD:
            if not self.source_key or self.value is None:
                raise ValueError("portal-field bindings require source_key and value")
            if (
                self.dependency_case_id is not None
                or self.generator is not None
                or self.sensitive
            ):
                raise ValueError("portal-field binding contains incompatible metadata")
        elif self.source == TemplateVariableSource.EVIDENCE_LITERAL:
            if self.value is None:
                raise ValueError("evidence-literal bindings require value")
            if (
                self.source_key is not None
                or self.dependency_case_id is not None
                or self.generator is not None
                or self.sensitive
            ):
                raise ValueError(
                    "evidence-literal binding contains incompatible metadata"
                )
            if "[REDACTED]" in self.value:
                raise ValueError(
                    "evidence-literal bindings cannot contain redaction markers"
                )
        elif self.source == TemplateVariableSource.GENERATED:
            if self.generator is None:
                raise ValueError("generated bindings require a generator")
            if (
                self.value is not None
                or self.source_key is not None
                or self.dependency_case_id is not None
                or self.sensitive
            ):
                raise ValueError("generated binding contains incompatible metadata")
        elif self.source == TemplateVariableSource.DEPENDENCY:
            if not self.source_key or not self.dependency_case_id:
                raise ValueError(
                    "dependency bindings require dependency_case_id and source_key"
                )
            if self.value is not None or self.generator is not None or self.sensitive:
                raise ValueError("dependency binding contains incompatible metadata")
        return self


class TestCaseExecutionSpec(SynthesisModel):
    """One cited and review-aware testcase execution specification."""

    test_case_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    dependency_case_ids: tuple[str, ...] = ()
    disposition: SynthesisDisposition
    request: Optional[HTTPRequestSpec] = None
    assertions: tuple[ResponseAssertion, ...] = ()
    variable_bindings: tuple[TemplateVariableBinding, ...] = ()
    evidence_snippet_ids: tuple[str, ...] = ()
    confidence: Confidence
    rationale: str = Field(min_length=1)
    unresolved_requirements: tuple[str, ...] = ()
    human_review_required: bool

    @model_validator(mode="after")
    def validate_execution_state(self) -> "TestCaseExecutionSpec":
        for label, values in (
            ("dependency_case_ids", self.dependency_case_ids),
            ("evidence_snippet_ids", self.evidence_snippet_ids),
            ("unresolved_requirements", self.unresolved_requirements),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        binding_names = [binding.name for binding in self.variable_bindings]
        if len(binding_names) != len(set(binding_names)):
            raise ValueError("variable binding names must be unique")
        if self.test_case_id in self.dependency_case_ids:
            raise ValueError("a testcase cannot depend on itself")

        used_variables = self._used_template_variables()
        unknown_variables = used_variables - set(binding_names)
        if unknown_variables:
            raise ValueError(
                f"template variables lack typed bindings: {sorted(unknown_variables)}"
            )
        for binding in self.variable_bindings:
            if (
                binding.source == TemplateVariableSource.DEPENDENCY
                and binding.dependency_case_id not in self.dependency_case_ids
            ):
                raise ValueError("variable binding references an undeclared dependency")

        if self.disposition == SynthesisDisposition.READY:
            if self.request is None or not self.assertions:
                raise ValueError(
                    "ready specifications require a request and assertions"
                )
            if not self.evidence_snippet_ids:
                raise ValueError("ready specifications require external citations")
            if self.unresolved_requirements or self.human_review_required:
                raise ValueError("ready specifications cannot retain unresolved review")
            unused_bindings = set(binding_names) - used_variables
            if unused_bindings:
                raise ValueError(
                    f"ready specification has unused bindings: {sorted(unused_bindings)}"
                )
            serialized_request = self.request.model_dump_json()
            if "[REDACTED]" in serialized_request:
                raise ValueError("ready requests cannot contain redaction markers")
            bindings = {binding.name: binding for binding in self.variable_bindings}
            for header in self.request.headers:
                if header.enabled and header.name.casefold() in _SENSITIVE_HEADER_NAMES:
                    variable_name = header.value[2:-2]
                    binding = bindings.get(variable_name)
                    if (
                        binding is None
                        or binding.source != TemplateVariableSource.ENVIRONMENT
                        or not binding.sensitive
                    ):
                        raise ValueError(
                            "sensitive header placeholders require a sensitive "
                            "environment binding"
                        )
        else:
            if not self.human_review_required or not self.unresolved_requirements:
                raise ValueError(
                    "non-ready specifications require explicit unresolved review items"
                )
            if (
                self.disposition == SynthesisDisposition.BLOCKED
                and self.request is not None
            ):
                raise ValueError(
                    "blocked specifications cannot carry an executable request"
                )
        return self

    def _used_template_variables(self) -> set[str]:
        values: list[str] = []
        if self.request is not None:
            values.append(self.request.path)
            values.extend(item.value for item in self.request.headers)
            values.extend(item.value for item in self.request.query_parameters)
            if self.request.body.template:
                values.append(self.request.body.template)
        values.extend(
            assertion.expected
            for assertion in self.assertions
            if assertion.expected is not None
        )
        return {
            match.group(0)[2:-2]
            for value in values
            for match in _TEMPLATE_VALUE.finditer(value)
        }


class SynthesisBatchResponse(SynthesisModel):
    """Exact structured output requested from one model call."""

    specifications: tuple[TestCaseExecutionSpec, ...]


class TokenUsage(SynthesisModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> "TokenUsage":
        if self.total_tokens < self.prompt_tokens + self.completion_tokens:
            raise ValueError("total token usage cannot be smaller than its components")
        return self


class SynthesisCallRecord(SynthesisModel):
    """Secret-free audit metadata for one returned LiteLLM response."""

    call_id: str = Field(pattern=SHA256_PATTERN)
    batch_id: str = Field(pattern=SHA256_PATTERN)
    attempt: int = Field(ge=1)
    test_case_ids: tuple[str, ...]
    request_sha256: str = Field(pattern=SHA256_PATTERN)
    response_sha256: str = Field(pattern=SHA256_PATTERN)
    valid: bool
    validation_error: Optional[str] = None
    usage: TokenUsage = Field(default_factory=TokenUsage)

    @model_validator(mode="after")
    def validate_record(self) -> "SynthesisCallRecord":
        if not self.test_case_ids or len(self.test_case_ids) != len(
            set(self.test_case_ids)
        ):
            raise ValueError("call testcase IDs must be non-empty and unique")
        if self.valid == (self.validation_error is not None):
            raise ValueError(
                "valid calls cannot have errors; invalid calls require one"
            )
        expected = _hash_json(
            {
                "batch_id": self.batch_id,
                "attempt": self.attempt,
                "request_sha256": self.request_sha256,
                "response_sha256": self.response_sha256,
            }
        )
        if self.call_id != expected:
            raise ValueError("call_id does not match call evidence")
        return self


class LLMProviderSummary(SynthesisModel):
    """Non-secret LiteLLM configuration and aggregate call metrics."""

    endpoint: str = Field(min_length=1)
    model: str = Field(min_length=1)
    response_format: str = Field(pattern=r"^(json_schema|json_object)$")
    prompt_version: str = PROMPT_VERSION
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    responses_received: int = Field(ge=0)
    valid_responses: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    errors: tuple[str, ...] = ()

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("LiteLLM endpoint must be absolute HTTP(S)")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "LiteLLM endpoint cannot contain credentials or query data"
            )
        return value

    @model_validator(mode="after")
    def validate_counts(self) -> "LLMProviderSummary":
        if self.valid_responses > self.responses_received:
            raise ValueError("valid responses cannot exceed responses received")
        return self


class SynthesisCoverage(SynthesisModel):
    """Completion and execution-readiness gates for Phase 9."""

    source_grounding_complete: bool
    test_cases: int = Field(ge=0)
    synthesized: int = Field(ge=0)
    ready: int = Field(ge=0)
    needs_review: int = Field(ge=0)
    blocked: int = Field(ge=0)
    failed_test_case_ids: tuple[str, ...] = ()
    synthesis_complete: bool
    execution_ready: bool
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_coverage(self) -> "SynthesisCoverage":
        if self.synthesized != self.ready + self.needs_review + self.blocked:
            raise ValueError("synthesis disposition counts do not reconcile")
        if self.test_cases != self.synthesized + len(self.failed_test_case_ids):
            raise ValueError("synthesis testcase counts do not reconcile")
        if self.synthesis_complete != (
            self.source_grounding_complete and not self.failed_test_case_ids
        ):
            raise ValueError(
                "synthesis_complete does not match source and output coverage"
            )
        if self.execution_ready != (
            self.synthesis_complete
            and self.ready == self.test_cases
            and not self.needs_review
            and not self.blocked
        ):
            raise ValueError("execution_ready does not match review dispositions")
        return self


class SynthesisPackage(SynthesisModel):
    """Audit-grade Phase 9 output ready for deterministic Postman rendering."""

    schema_version: str = SYNTHESIS_SCHEMA_VERSION
    source_run_id: str = Field(min_length=1)
    source_grounding_sha256: str = Field(pattern=SHA256_PATTERN)
    synthesized_at: datetime
    provider: LLMProviderSummary
    calls: tuple[SynthesisCallRecord, ...]
    specifications: tuple[TestCaseExecutionSpec, ...]
    coverage: SynthesisCoverage

    @field_validator("synthesized_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("synthesized_at must include timezone information")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_package(self) -> "SynthesisPackage":
        call_ids = [call.call_id for call in self.calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("synthesis call records must be unique")
        case_ids = [spec.test_case_id for spec in self.specifications]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("synthesized testcase specifications must be unique")
        if self.coverage.synthesized != len(self.specifications):
            raise ValueError("synthesis coverage does not match specifications")
        valid_responses = sum(call.valid for call in self.calls)
        if self.provider.responses_received != len(self.calls):
            raise ValueError("provider response count does not match call records")
        if self.provider.valid_responses != valid_responses:
            raise ValueError(
                "provider valid response count does not match call records"
            )
        return self


class SynthesisFile(SynthesisModel):
    name: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_size: int = Field(ge=0)


class SynthesisExportManifest(SynthesisModel):
    schema_version: str = SYNTHESIS_SCHEMA_VERSION
    source_run_id: str = Field(min_length=1)
    source_grounding_sha256: str = Field(pattern=SHA256_PATTERN)
    synthesized_at: datetime
    files: tuple[SynthesisFile, ...]
    test_cases: int = Field(ge=0)
    synthesized: int = Field(ge=0)
    synthesis_complete: bool
    execution_ready: bool

    @field_validator("synthesized_at")
    @classmethod
    def validate_manifest_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("synthesized_at must include timezone information")
        return value.astimezone(timezone.utc)


class SynthesisCheckpoint(SynthesisModel):
    """Validated, secret-free progress used to resume expensive model calls."""

    source_run_id: str = Field(min_length=1)
    source_grounding_sha256: str = Field(pattern=SHA256_PATTERN)
    configuration_sha256: str = Field(pattern=SHA256_PATTERN)
    model: str = Field(min_length=1)
    updated_at: datetime
    calls: tuple[SynthesisCallRecord, ...] = ()
    specifications: tuple[TestCaseExecutionSpec, ...] = ()
    errors: tuple[str, ...] = ()

    @field_validator("updated_at")
    @classmethod
    def validate_checkpoint_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("updated_at must include timezone information")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_progress(self) -> "SynthesisCheckpoint":
        call_ids = [call.call_id for call in self.calls]
        case_ids = [spec.test_case_id for spec in self.specifications]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("checkpoint call records must be unique")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("checkpoint specifications must be unique")
        return self


def synthesis_call_id(
    batch_id: str,
    attempt: int,
    request_sha256: str,
    response_sha256: str,
) -> str:
    return _hash_json(
        {
            "batch_id": batch_id,
            "attempt": attempt,
            "request_sha256": request_sha256,
            "response_sha256": response_sha256,
        }
    )


def contains_template_value(value: str) -> bool:
    """Return whether a generated value contains a Postman-style placeholder."""

    return bool(_TEMPLATE_VALUE.search(value))


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
    "AssertionOperator",
    "AssertionSource",
    "Confidence",
    "GeneratedValueKind",
    "HTTPRequestSpec",
    "HTTPMethod",
    "LLMProviderSummary",
    "NamedValue",
    "PROMPT_VERSION",
    "RequestBodyMode",
    "RequestBodySpec",
    "ResponseAssertion",
    "SYNTHESIS_SCHEMA_VERSION",
    "SynthesisBatchResponse",
    "SynthesisCallRecord",
    "SynthesisCheckpoint",
    "SynthesisCoverage",
    "SynthesisDisposition",
    "SynthesisExportManifest",
    "SynthesisFile",
    "SynthesisPackage",
    "TemplateVariableBinding",
    "TemplateVariableSource",
    "TestCaseExecutionSpec",
    "TokenUsage",
    "contains_template_value",
    "synthesis_call_id",
]
