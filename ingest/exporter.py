"""Deterministic filesystem export for external-source import packages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from ingest.builder import ParsedCsv
from ingest.models import ExternalTestCaseRow, IngestExportManifest, IngestFile


class IngestExportError(RuntimeError):
    """Raised when an external-source package cannot be written safely."""


@dataclass(frozen=True)
class IngestExportResult:
    """Paths and manifest returned after a successful import export."""

    output_directory: Path
    csv_path: Path
    rows_path: Path
    manifest_path: Path
    manifest: IngestExportManifest


def export_ingest(
    parsed: ParsedCsv,
    source_csv_bytes: bytes,
    *,
    source_csv_name: str,
    source_label: str,
    run_id: str,
    imported_at: datetime,
    rows_total: int,
    selected_ids: Sequence[str],
    output_directory: Path,
    overwrite: bool = False,
) -> IngestExportResult:
    """Write the immutable source copy, normalized rows, and integrity manifest."""

    destination = output_directory.expanduser().resolve()
    csv_path = destination / "source.csv"
    rows_path = destination / "rows.jsonl"
    manifest_path = destination / "manifest.json"
    targets = (csv_path, rows_path, manifest_path)
    existing = tuple(path for path in targets if path.exists())
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise IngestExportError(
            f"ingest output already exists ({names}); use overwrite explicitly"
        )

    try:
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise IngestExportError(f"failed to create ingest output: {exc}") from exc

    rows_data = b"".join(_json_bytes(_row_record(row)) + b"\n" for row in parsed.rows)
    files = (
        _file(csv_path.name, source_csv_bytes),
        _file(rows_path.name, rows_data),
    )
    manifest = IngestExportManifest(
        source_label=source_label,
        source_csv_name=source_csv_name,
        source_csv_sha256=parsed.source_sha256,
        run_id=run_id,
        imported_at=imported_at,
        rows_total=rows_total,
        rows_selected=len(parsed.rows),
        selected_test_cases=tuple(selected_ids),
        test_cases=tuple(row.test_case_id for row in parsed.rows),
        files=files,
    )
    manifest_data = _json_bytes(manifest.model_dump(mode="json"))

    _atomic_write(csv_path, source_csv_bytes)
    _atomic_write(rows_path, rows_data)
    _atomic_write(manifest_path, manifest_data)
    return IngestExportResult(
        output_directory=destination,
        csv_path=csv_path,
        rows_path=rows_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _row_record(row: ExternalTestCaseRow) -> dict[str, object]:
    return {
        "row_number": row.row_number,
        "row_sha256": row.row_sha256,
        "test_case_id": row.test_case_id,
        "api_type": row.api_type,
        "api_name": row.api_name,
        "test_data": row.test_data,
        "version": row.version,
        "response_codes": row.response_codes,
        "description": row.description,
        "steps": list(row.steps),
        "extra_fields": {field.key: field.value for field in row.extra_fields},
    }


def _file(name: str, data: bytes) -> IngestFile:
    return IngestFile(
        name=name,
        sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
    )


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise IngestExportError(f"failed to write {path.name}: {exc}") from exc


__all__ = [
    "IngestExportError",
    "IngestExportResult",
    "export_ingest",
]
