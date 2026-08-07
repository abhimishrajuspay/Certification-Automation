"""Atomic export of Postman collection, environment, and audit report."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from postman.builder import PostmanBuildResult
from postman.models import PostmanExportManifest, PostmanFile


class PostmanExportError(RuntimeError):
    """Raised when a rendered Postman package cannot be written safely."""


@dataclass(frozen=True)
class PostmanExportResult:
    output_directory: Path
    collection_path: Path
    environment_path: Path
    report_path: Path
    manifest_path: Path
    manifest: PostmanExportManifest


def export_postman(
    result: PostmanBuildResult,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> PostmanExportResult:
    destination = output_directory.expanduser().resolve()
    collection_path = destination / "postman_collection.json"
    environment_path = destination / "postman_environment.json"
    report_path = destination / "postman_report.json"
    manifest_path = destination / "manifest.json"
    targets = (collection_path, environment_path, report_path, manifest_path)
    existing = tuple(path for path in targets if path.exists())
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise PostmanExportError(
            f"Postman output already exists ({names}); use overwrite explicitly"
        )
    try:
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise PostmanExportError(f"failed to create Postman output: {exc}") from exc

    collection_data = _json_bytes(
        result.collection.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
        )
    )
    environment_data = _json_bytes(
        result.environment.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
        )
    )
    report_data = _json_bytes(result.report.model_dump(mode="json"))
    files = (
        _postman_file(collection_path.name, collection_data),
        _postman_file(environment_path.name, environment_data),
        _postman_file(report_path.name, report_data),
    )
    manifest = PostmanExportManifest(
        source_run_id=result.report.source_run_id,
        source_synthesis_sha256=result.report.source_synthesis_sha256,
        generated_at=result.report.generated_at,
        files=files,
        rendered=result.report.coverage.rendered,
        skipped=result.report.coverage.skipped,
        generation_complete=result.report.coverage.generation_complete,
    )
    manifest_data = _json_bytes(manifest.model_dump(mode="json"))
    _atomic_write(collection_path, collection_data)
    _atomic_write(environment_path, environment_data)
    _atomic_write(report_path, report_data)
    _atomic_write(manifest_path, manifest_data)
    return PostmanExportResult(
        output_directory=destination,
        collection_path=collection_path,
        environment_path=environment_path,
        report_path=report_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _postman_file(name: str, data: bytes) -> PostmanFile:
    return PostmanFile(
        name=name,
        sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
    )


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
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
        raise PostmanExportError(f"failed to write {path.name}: {exc}") from exc


__all__ = [
    "PostmanExportError",
    "PostmanExportResult",
    "export_postman",
]
