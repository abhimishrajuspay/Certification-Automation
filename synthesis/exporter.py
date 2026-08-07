"""Atomic export and resumable checkpoint support for Phase 9."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from synthesis.models import (
    SynthesisCallRecord,
    SynthesisCheckpoint,
    SynthesisExportManifest,
    SynthesisFile,
    SynthesisPackage,
    TestCaseExecutionSpec,
)


class SynthesisExportError(RuntimeError):
    """Raised when synthesis progress or final output cannot be persisted safely."""


@dataclass(frozen=True)
class SynthesisExportResult:
    output_directory: Path
    synthesis_path: Path
    specifications_path: Path
    manifest_path: Path
    checkpoint_path: Path
    manifest: SynthesisExportManifest


def export_synthesis(
    package: SynthesisPackage,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> SynthesisExportResult:
    """Write the audit package and compact execution-specification JSONL."""

    destination = output_directory.expanduser().resolve()
    synthesis_path = destination / "synthesis.json"
    specifications_path = destination / "execution_specs.jsonl"
    manifest_path = destination / "manifest.json"
    checkpoint_path = destination / "checkpoint.json"
    targets = (synthesis_path, specifications_path, manifest_path)
    existing = tuple(path for path in targets if path.exists())
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise SynthesisExportError(
            f"synthesis output already exists ({names}); use overwrite explicitly"
        )
    _ensure_directory(destination)

    synthesis_data = _json_bytes(package.model_dump(mode="json"), pretty=True)
    specifications_data = b"".join(
        _json_bytes(spec.model_dump(mode="json"), pretty=False) + b"\n"
        for spec in package.specifications
    )
    files = (
        _synthesis_file(synthesis_path.name, synthesis_data),
        _synthesis_file(specifications_path.name, specifications_data),
    )
    manifest = SynthesisExportManifest(
        source_run_id=package.source_run_id,
        source_grounding_sha256=package.source_grounding_sha256,
        synthesized_at=package.synthesized_at,
        files=files,
        test_cases=package.coverage.test_cases,
        synthesized=package.coverage.synthesized,
        synthesis_complete=package.coverage.synthesis_complete,
        execution_ready=package.coverage.execution_ready,
    )
    manifest_data = _json_bytes(manifest.model_dump(mode="json"), pretty=True)
    _atomic_write(synthesis_path, synthesis_data)
    _atomic_write(specifications_path, specifications_data)
    _atomic_write(manifest_path, manifest_data)
    return SynthesisExportResult(
        output_directory=destination,
        synthesis_path=synthesis_path,
        specifications_path=specifications_path,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        manifest=manifest,
    )


def save_checkpoint(
    path: Path,
    *,
    source_run_id: str,
    source_grounding_sha256: str,
    configuration_sha256: str,
    model: str,
    calls: tuple[SynthesisCallRecord, ...],
    specifications: tuple[TestCaseExecutionSpec, ...],
    errors: tuple[str, ...],
) -> SynthesisCheckpoint:
    """Atomically replace a secret-free checkpoint after a completed batch."""

    checkpoint_path = path.expanduser().resolve()
    _ensure_directory(checkpoint_path.parent)
    checkpoint = SynthesisCheckpoint(
        source_run_id=source_run_id,
        source_grounding_sha256=source_grounding_sha256,
        configuration_sha256=configuration_sha256,
        model=model,
        updated_at=datetime.now(timezone.utc),
        calls=calls,
        specifications=specifications,
        errors=errors,
    )
    _atomic_write(
        checkpoint_path,
        _json_bytes(checkpoint.model_dump(mode="json"), pretty=True),
    )
    return checkpoint


def load_checkpoint(path: Path) -> SynthesisCheckpoint:
    checkpoint_path = path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SynthesisExportError(
            f"synthesis checkpoint does not exist: {checkpoint_path}"
        )
    try:
        return SynthesisCheckpoint.model_validate_json(checkpoint_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise SynthesisExportError(f"invalid synthesis checkpoint: {exc}") from exc


def _ensure_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise SynthesisExportError(f"failed to create synthesis output: {exc}") from exc


def _synthesis_file(name: str, data: bytes) -> SynthesisFile:
    return SynthesisFile(
        name=name,
        sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
    )


def _json_bytes(value: object, *, pretty: bool) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
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
        raise SynthesisExportError(f"failed to write {path.name}: {exc}") from exc


__all__ = [
    "SynthesisExportError",
    "SynthesisExportResult",
    "export_synthesis",
    "load_checkpoint",
    "save_checkpoint",
]
