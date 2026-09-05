"""Strict contracts for externally imported (CSV) testcase source data."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from knowledge.models import SHA256_PATTERN


INGEST_SCHEMA_VERSION = "1.0"

# Exact CSV column contract. Column matching is case-insensitive after trimming,
# but every required column must be present; unknown columns become extra fields.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "TC ID",
    "API Type",
    "API Name",
    "Test Data",
    "Version",
    "RC",
    "Description",
    "Steps",
)

_DASH_MARKERS = {"", "-", "—", "–", "n/a", "N/A"}


class IngestModel(BaseModel):
    """Strict immutable base model for external source import artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


def normalize_column_value(value: object) -> str:
    """Collapse whitespace and map explicit dash placeholders to empty text."""

    if value is None:
        return ""
    normalized = " ".join(str(value).split())
    return "" if normalized in _DASH_MARKERS else normalized


class ExternalField(IngestModel):
    """One additional CSV column normalized into the knowledge field shape."""

    key: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str = Field(min_length=1)
    value: str = ""


class ExternalTestCaseRow(IngestModel):
    """One validated CSV row with its deterministic content digest."""

    row_number: int = Field(ge=1)
    row_sha256: str = Field(pattern=SHA256_PATTERN)
    test_case_id: str = Field(min_length=1)
    api_type: str = Field(min_length=1)
    api_name: str = Field(min_length=1)
    test_data: str = ""
    version: str = ""
    response_codes: str = ""
    description: str = Field(min_length=1)
    steps: tuple[str, ...] = ()
    extra_fields: tuple[ExternalField, ...] = ()

    @model_validator(mode="after")
    def validate_row(self) -> "ExternalTestCaseRow":
        keys = [field.key for field in self.extra_fields]
        if len(keys) != len(set(keys)):
            raise ValueError("extra CSV column keys must be unique")
        expected = _hash_json(
            {
                "test_case_id": self.test_case_id,
                "api_type": self.api_type,
                "api_name": self.api_name,
                "test_data": self.test_data,
                "version": self.version,
                "response_codes": self.response_codes,
                "description": self.description,
                "steps": list(self.steps),
                "extra_fields": [
                    {"key": field.key, "label": field.label, "value": field.value}
                    for field in self.extra_fields
                ],
            }
        )
        if self.row_sha256 != expected:
            raise ValueError("row digest does not match parsed CSV content")
        return self


class IngestFile(IngestModel):
    """One exported ingest-source file and its content digest."""

    name: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_size: int = Field(ge=0)


class IngestExportManifest(IngestModel):
    """Small integrity manifest for one external-source import package."""

    schema_version: str = INGEST_SCHEMA_VERSION
    source_kind: str = "external_csv"
    source_label: str = Field(min_length=1)
    source_csv_name: str = Field(min_length=1)
    source_csv_sha256: str = Field(pattern=SHA256_PATTERN)
    run_id: str = Field(min_length=1)
    imported_at: datetime
    rows_total: int = Field(ge=0)
    rows_selected: int = Field(ge=0)
    selected_test_cases: tuple[str, ...] = ()
    test_cases: tuple[str, ...] = ()
    files: tuple[IngestFile, ...]

    @field_validator("imported_at")
    @classmethod
    def validate_imported_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("imported_at must include timezone information")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_manifest(self) -> "IngestExportManifest":
        if self.rows_selected > self.rows_total:
            raise ValueError("selected rows cannot exceed total rows")
        if len(self.test_cases) != self.rows_selected:
            raise ValueError("manifest testcase list does not match selected rows")
        if len(self.test_cases) != len(set(self.test_cases)):
            raise ValueError("manifest testcase IDs must be unique")
        if len(self.selected_test_cases) != len(set(self.selected_test_cases)):
            raise ValueError("requested testcase IDs must be unique")
        unknown = set(self.selected_test_cases) - set(self.test_cases)
        if unknown:
            raise ValueError(f"manifest requests unknown testcases: {sorted(unknown)}")
        names = [item.name for item in self.files]
        if len(names) != len(set(names)):
            raise ValueError("manifest file names must be unique")
        return self


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
    "ExternalField",
    "ExternalTestCaseRow",
    "INGEST_SCHEMA_VERSION",
    "IngestExportManifest",
    "IngestFile",
    "IngestModel",
    "REQUIRED_COLUMNS",
    "normalize_column_value",
]
