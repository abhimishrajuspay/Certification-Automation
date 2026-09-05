"""Parse external CSV testcases into the Phase 7 knowledge contract."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from ingest.models import (
    ExternalField,
    ExternalTestCaseRow,
    REQUIRED_COLUMNS,
    _hash_json,
    normalize_column_value,
)
from knowledge.models import (
    EvidencePointer,
    KnowledgeCoverage,
    KnowledgeField,
    KnowledgeSourceKind,
    PortalKnowledge,
    TestCaseKnowledge,
)
from scraper.models import ScrapeRunStatus


class IngestBuildError(RuntimeError):
    """Raised when external CSV data cannot become a trusted knowledge package."""


# Split "1. step one.2. step two." while leaving ranges like "1–14. ..." intact.
_STEP_MARKER = re.compile(r"(?=(?:(?<=^)|(?<=[\s.;)]))(?:\d{1,3}[.)]\s*))")

_FIELD_KEYS: dict[str, str] = {
    "TC ID": "test_case_id",
    "API Type": "api_type",
    "API Name": "api_name",
    "Test Data": "test_data",
    "Version": "version",
    "RC": "rc",
    "Description": "description",
    "Steps": "steps",
}


@dataclass(frozen=True)
class ParsedCsv:
    """Validated rows plus source identity for a CSV import."""

    source_sha256: str
    header_labels: tuple[str, ...]
    rows: tuple[ExternalTestCaseRow, ...]


def parse_csv_bytes(data: bytes) -> ParsedCsv:
    """Parse the strict BCRP-style testcase CSV into validated rows."""

    if not data.strip():
        raise IngestBuildError("the CSV file is empty")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IngestBuildError(f"the CSV file is not valid UTF-8: {exc}") from exc
    source_sha256 = hashlib.sha256(data).hexdigest()

    reader = csv.DictReader(io.StringIO(text))
    raw_headers = [(header or "").strip() for header in (reader.fieldnames or [])]
    if not raw_headers or not any(raw_headers):
        raise IngestBuildError("the CSV header row is missing")
    duplicated = sorted({h for h in raw_headers if h and raw_headers.count(h) > 1})
    if duplicated:
        raise IngestBuildError(f"the CSV repeats columns: {duplicated}")
    missing = [name for name in REQUIRED_COLUMNS if name not in raw_headers]
    if missing:
        raise IngestBuildError(
            f"the CSV is missing required columns: {missing}; found {raw_headers}"
        )
    header_labels = tuple(raw_headers)

    extra_keys: dict[str, str] = {}
    errors: list[str] = []
    for header in raw_headers:
        if header in _FIELD_KEYS:
            continue
        key = re.sub(r"[^a-z0-9]+", "_", header.lower()).strip("_")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key or ""):
            errors.append(
                f"extra column {header!r} cannot become a knowledge field key"
            )
        elif key in _FIELD_KEYS.values():
            errors.append(f"extra column {header!r} duplicates a required field key")
        elif key in extra_keys:
            errors.append(f"extra columns normalize to the same key {key!r}")
        else:
            extra_keys[key] = header

    rows: list[ExternalTestCaseRow] = []
    seen_ids: dict[str, int] = {}
    for index, record in enumerate(reader, start=1):
        values = {
            (key or "").strip(): normalize_column_value(value)
            for key, value in record.items()
            if key is not None
        }
        if not any(values.values()):
            continue
        overflow = values.get("")
        if overflow:
            errors.append(f"row {index}: has overflow cells past the header width")
            continue
        test_case_id = values["TC ID"]
        api_type = values["API Type"]
        api_name = values["API Name"]
        description = values["Description"]
        if not test_case_id:
            errors.append(f"row {index}: TC ID is empty")
            continue
        if test_case_id in seen_ids:
            errors.append(
                f"row {index}: duplicate TC ID {test_case_id!r}"
                f" (first seen on row {seen_ids[test_case_id]})"
            )
            continue
        for label in ("API Type", "API Name", "Description"):
            if not values[label]:
                errors.append(f"row {index}: column {label!r} is empty")
        if not (test_case_id and api_type and api_name and description):
            continue
        seen_ids[test_case_id] = index
        steps = _split_steps(values["Steps"])
        extra_fields = tuple(
            ExternalField(key=key, label=label, value=values[label])
            for key, label in extra_keys.items()
            if values[label]
        )
        row_sha256 = _hash_json(
            {
                "test_case_id": test_case_id,
                "api_type": api_type,
                "api_name": api_name,
                "test_data": values["Test Data"],
                "version": values["Version"],
                "response_codes": values["RC"],
                "description": description,
                "steps": list(steps),
                "extra_fields": [
                    {"key": field.key, "label": field.label, "value": field.value}
                    for field in extra_fields
                ],
            }
        )
        try:
            rows.append(
                ExternalTestCaseRow(
                    row_number=index,
                    row_sha256=row_sha256,
                    test_case_id=test_case_id,
                    api_type=api_type,
                    api_name=api_name,
                    test_data=values["Test Data"],
                    version=values["Version"],
                    response_codes=values["RC"],
                    description=description,
                    steps=steps,
                    extra_fields=extra_fields,
                )
            )
        except ValueError as exc:
            errors.append(f"row {index}: {exc}")
    if errors:
        raise IngestBuildError("invalid CSV rows: " + "; ".join(errors))
    if not rows:
        raise IngestBuildError("the CSV contains no usable testcase rows")
    return ParsedCsv(
        source_sha256=source_sha256, header_labels=header_labels, rows=tuple(rows)
    )


def select_rows(
    parsed: ParsedCsv,
    selected_ids: Sequence[str],
) -> ParsedCsv:
    """Deterministically filter parsed rows to the requested testcase IDs."""

    if not selected_ids:
        return parsed
    requested = tuple(dict.fromkeys(item.strip() for item in selected_ids))
    if any(not item for item in requested):
        raise IngestBuildError("selected testcase IDs must not be empty")
    available = {row.test_case_id for row in parsed.rows}
    unknown = [item for item in requested if item not in available]
    if unknown:
        raise IngestBuildError(
            f"unknown testcase IDs requested: {unknown}; available: {sorted(available)}"
        )
    selected = set(requested)
    rows = tuple(row for row in parsed.rows if row.test_case_id in selected)
    return ParsedCsv(
        source_sha256=parsed.source_sha256,
        header_labels=parsed.header_labels,
        rows=rows,
    )


def build_portal_knowledge(
    parsed: ParsedCsv,
    *,
    run_id: str,
    imported_at: datetime,
    rows_total: int,
    selected_ids: Sequence[str],
) -> PortalKnowledge:
    """Translate imported rows into the strict Phase 7 handoff contract."""

    if not run_id.strip():
        raise IngestBuildError("a run ID is required")
    if imported_at.tzinfo is None or imported_at.utcoffset() is None:
        raise IngestBuildError("imported_at must be timezone-aware")
    filtered = len(parsed.rows) != rows_total

    source_identity = f"ingest://csv/{parsed.source_sha256[:16]}"
    test_cases: list[TestCaseKnowledge] = []
    for row in parsed.rows:
        pointer = EvidencePointer(
            state_id=f"csvrow-{row.row_sha256[:24]}",
            state_sequence=row.row_number,
            url=f"{source_identity}/row/{row.row_number:04d}",
            frame_id="csv_row",
            element_ids=(),
            artifact_ids=(parsed.source_sha256,),
        )
        fields: list[KnowledgeField] = [
            KnowledgeField(key="api_type", label="API Type", value=row.api_type),
            KnowledgeField(key="api_name", label="API Name", value=row.api_name),
        ]
        if row.test_data:
            fields.append(
                KnowledgeField(key="test_data", label="Test Data", value=row.test_data)
            )
        if row.version:
            fields.append(
                KnowledgeField(key="version", label="Version", value=row.version)
            )
        if row.response_codes:
            fields.append(
                KnowledgeField(key="rc", label="RC", value=row.response_codes)
            )
        if row.steps:
            fields.append(
                KnowledgeField(key="steps", label="Steps", value="\n".join(row.steps))
            )
        fields.extend(
            KnowledgeField(key=field.key, label=field.label, value=field.value)
            for field in row.extra_fields
        )
        try:
            test_cases.append(
                TestCaseKnowledge(
                    test_case_id=row.test_case_id,
                    fields=tuple(fields),
                    dependency_case_ids=(),
                    description=row.description,
                    controls=(),
                    evidence=(pointer,),
                    description_evidence=(pointer,),
                    conflicts=(),
                )
            )
        except ValueError as exc:
            raise IngestBuildError(
                f"row {row.row_number} cannot become portal knowledge: {exc}"
            ) from exc

    limitations = [
        "source_kind external_csv: evidence pointers reference deterministic CSV-row"
        " digests and the imported file digest, not browser-captured portal states",
        "external rows carry no portal controls, routes, tables, modals, or network"
        " observations; dependencies are not declared by the CSV format",
    ]
    if filtered:
        limitations.append(
            f"filtered import: {len(parsed.rows)} of {rows_total} CSV rows selected"
        )
    coverage = KnowledgeCoverage(
        source_status=ScrapeRunStatus.COMPLETED,
        source_bounded_complete=False,
        testcase_context_complete=True,
        states_examined=len(parsed.rows),
        routes_normalized=0,
        tables_normalized=0,
        declared_test_cases=len(parsed.rows),
        test_cases_normalized=len(test_cases),
        descriptions_captured=len(test_cases),
        missing_description_ids=(),
        conflicting_test_case_ids=(),
        limitations=tuple(limitations),
    )
    try:
        return PortalKnowledge(
            source_kind=KnowledgeSourceKind.EXTERNAL_CSV,
            source_run_id=run_id,
            source_root_url=source_identity,
            source_started_at=imported_at,
            source_ended_at=imported_at,
            normalized_at=imported_at,
            routes=(),
            tables=(),
            test_cases=tuple(test_cases),
            network=(),
            coverage=coverage,
        )
    except ValueError as exc:
        raise IngestBuildError(
            f"imported rows cannot form portal knowledge: {exc}"
        ) from exc


def _split_steps(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    pieces = [piece.strip() for piece in _STEP_MARKER.split(value)]
    return tuple(dict.fromkeys(piece for piece in pieces if piece))


__all__ = [
    "IngestBuildError",
    "ParsedCsv",
    "build_portal_knowledge",
    "parse_csv_bytes",
    "select_rows",
]
