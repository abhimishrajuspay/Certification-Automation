"""Strict Postman v2.1 rendering and audit contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from knowledge.models import SHA256_PATTERN


POSTMAN_SCHEMA_VERSION = "1.0"
POSTMAN_COLLECTION_SCHEMA = (
    "https://schema.postman.com/collection/json/v2.1.0/draft-07/collection.json"
)


class PostmanModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
        populate_by_name=True,
    )


class PostmanVariable(PostmanModel):
    key: str = Field(min_length=1)
    value: str = ""
    type: Literal["string", "default", "secret"] = "string"
    description: Optional[str] = None
    disabled: bool = False


class PostmanHeader(PostmanModel):
    key: str = Field(min_length=1)
    value: str
    type: Literal["text"] = "text"
    description: Optional[str] = None
    disabled: bool = False


class PostmanQueryParameter(PostmanModel):
    key: str = Field(min_length=1)
    value: str
    description: Optional[str] = None
    disabled: bool = False


class PostmanUrl(PostmanModel):
    raw: str = Field(min_length=1)
    host: tuple[str, ...]
    path: tuple[str, ...]
    query: tuple[PostmanQueryParameter, ...] = ()

    @model_validator(mode="after")
    def validate_url(self) -> "PostmanUrl":
        if not self.host:
            raise ValueError("Postman URL requires a host expression")
        return self


class PostmanFormParameter(PostmanModel):
    key: str = Field(min_length=1)
    value: str
    type: Literal["text"] = "text"
    description: Optional[str] = None
    disabled: bool = False


class PostmanBody(PostmanModel):
    mode: Literal["raw", "urlencoded"]
    raw: Optional[str] = None
    urlencoded: tuple[PostmanFormParameter, ...] = ()
    options: Optional[dict[str, object]] = None

    @model_validator(mode="after")
    def validate_body(self) -> "PostmanBody":
        if self.mode == "raw":
            if self.raw is None or self.urlencoded:
                raise ValueError("raw Postman body requires only raw content")
        elif self.raw is not None or not self.urlencoded:
            raise ValueError("urlencoded Postman body requires form parameters")
        return self


class PostmanScript(PostmanModel):
    type: Literal["text/javascript"] = "text/javascript"
    exec: tuple[str, ...]

    @model_validator(mode="after")
    def validate_script(self) -> "PostmanScript":
        if not self.exec:
            raise ValueError("Postman script cannot be empty")
        return self


class PostmanEvent(PostmanModel):
    listen: Literal["prerequest", "test"]
    script: PostmanScript


class PostmanRequest(PostmanModel):
    method: str = Field(min_length=1)
    header: tuple[PostmanHeader, ...] = ()
    body: Optional[PostmanBody] = None
    url: PostmanUrl
    description: Optional[str] = None


class PostmanItem(PostmanModel):
    name: str = Field(min_length=1)
    id: str = Field(min_length=1)
    event: tuple[PostmanEvent, ...]
    request: PostmanRequest

    @model_validator(mode="after")
    def validate_events(self) -> "PostmanItem":
        listeners = [event.listen for event in self.event]
        if len(listeners) != len(set(listeners)):
            raise ValueError("Postman item event listeners must be unique")
        if "test" not in listeners:
            raise ValueError("every rendered testcase requires a test script")
        return self


class PostmanInfo(PostmanModel):
    name: str = Field(min_length=1)
    schema_url: str = Field(alias="schema", default=POSTMAN_COLLECTION_SCHEMA)
    postman_id: str = Field(alias="_postman_id", min_length=1)
    description: Optional[str] = None


class PostmanCollection(PostmanModel):
    info: PostmanInfo
    item: tuple[PostmanItem, ...]
    variable: tuple[PostmanVariable, ...] = ()

    @model_validator(mode="after")
    def validate_collection(self) -> "PostmanCollection":
        item_ids = [item.id for item in self.item]
        variable_keys = [item.key for item in self.variable]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("Postman collection item IDs must be unique")
        if len(variable_keys) != len(set(variable_keys)):
            raise ValueError("Postman collection variables must be unique")
        return self


class PostmanEnvironmentValue(PostmanModel):
    key: str = Field(min_length=1)
    value: str = ""
    type: Literal["default", "secret"] = "default"
    enabled: bool = True


class PostmanEnvironment(PostmanModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    values: tuple[PostmanEnvironmentValue, ...]
    schema_url: str = Field(alias="_postman_variable_scope", default="environment")
    exported_at: datetime = Field(alias="_postman_exported_at")
    exported_using: str = Field(
        alias="_postman_exported_using",
        default="cz-certification-automation",
    )

    @field_validator("exported_at")
    @classmethod
    def validate_exported_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Postman environment export time must include timezone")
        return value.astimezone(timezone.utc)


class RenderedTestCase(PostmanModel):
    test_case_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    order: int = Field(ge=0)


class SkippedTestCase(PostmanModel):
    test_case_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class PostmanCoverage(PostmanModel):
    source_synthesis_complete: bool
    source_execution_ready: bool
    test_cases: int = Field(ge=0)
    ready_source_cases: int = Field(ge=0)
    rendered: int = Field(ge=0)
    rendered_needs_review: int = Field(ge=0, default=0)
    skipped: int = Field(ge=0)
    generation_complete: bool
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_coverage(self) -> "PostmanCoverage":
        if self.rendered + self.skipped != self.test_cases:
            raise ValueError("Postman coverage testcase counts do not reconcile")
        expected = bool(
            self.source_synthesis_complete
            and self.source_execution_ready
            and self.rendered == self.test_cases
            and self.skipped == 0
        )
        if self.generation_complete != expected:
            raise ValueError("generation_complete does not match coverage")
        return self


class PostmanPackage(PostmanModel):
    schema_version: str = POSTMAN_SCHEMA_VERSION
    source_run_id: str = Field(min_length=1)
    source_synthesis_sha256: str = Field(pattern=SHA256_PATTERN)
    generated_at: datetime
    collection_id: str = Field(min_length=1)
    environment_id: str = Field(min_length=1)
    rendered_test_cases: tuple[RenderedTestCase, ...]
    skipped_test_cases: tuple[SkippedTestCase, ...]
    coverage: PostmanCoverage

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Postman generation time must include timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_package(self) -> "PostmanPackage":
        rendered_ids = [item.test_case_id for item in self.rendered_test_cases]
        skipped_ids = [item.test_case_id for item in self.skipped_test_cases]
        if len(rendered_ids) != len(set(rendered_ids)):
            raise ValueError("rendered testcase IDs must be unique")
        if len(skipped_ids) != len(set(skipped_ids)):
            raise ValueError("skipped testcase IDs must be unique")
        if set(rendered_ids) & set(skipped_ids):
            raise ValueError("testcases cannot be both rendered and skipped")
        if len(rendered_ids) != self.coverage.rendered:
            raise ValueError("rendered audit records do not match coverage")
        if len(skipped_ids) != self.coverage.skipped:
            raise ValueError("skipped audit records do not match coverage")
        return self


class PostmanFile(PostmanModel):
    name: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_size: int = Field(ge=0)


class PostmanExportManifest(PostmanModel):
    schema_version: str = POSTMAN_SCHEMA_VERSION
    source_run_id: str = Field(min_length=1)
    source_synthesis_sha256: str = Field(pattern=SHA256_PATTERN)
    generated_at: datetime
    files: tuple[PostmanFile, ...]
    rendered: int = Field(ge=0)
    skipped: int = Field(ge=0)
    generation_complete: bool

    @field_validator("generated_at")
    @classmethod
    def validate_manifest_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Postman manifest time must include timezone")
        return value.astimezone(timezone.utc)


__all__ = [
    "POSTMAN_COLLECTION_SCHEMA",
    "POSTMAN_SCHEMA_VERSION",
    "PostmanBody",
    "PostmanCollection",
    "PostmanCoverage",
    "PostmanEnvironment",
    "PostmanEnvironmentValue",
    "PostmanEvent",
    "PostmanExportManifest",
    "PostmanFile",
    "PostmanFormParameter",
    "PostmanHeader",
    "PostmanInfo",
    "PostmanItem",
    "PostmanPackage",
    "PostmanQueryParameter",
    "PostmanRequest",
    "PostmanScript",
    "PostmanUrl",
    "PostmanVariable",
    "RenderedTestCase",
    "SkippedTestCase",
]
