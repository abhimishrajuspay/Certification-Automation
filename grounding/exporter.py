"""Atomic export of audit-grade and compact grounded testcase packages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from grounding.models import (
    GroundedTestCase,
    GroundingExportManifest,
    GroundingFile,
    GroundingPackage,
    GroundingSnippet,
)


class GroundingExportError(RuntimeError):
    """Raised when a grounding package cannot be written safely."""


@dataclass(frozen=True)
class GroundingExportResult:
    """Paths and manifest produced by one successful export."""

    output_directory: Path
    grounding_path: Path
    testcases_path: Path
    manifest_path: Path
    manifest: GroundingExportManifest


def export_grounding(
    package: GroundingPackage,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> GroundingExportResult:
    """Write a complete package while leaving source evidence untouched."""

    destination = output_directory.expanduser().resolve()
    grounding_path = destination / "grounding.json"
    testcases_path = destination / "grounded_testcases.jsonl"
    manifest_path = destination / "manifest.json"
    targets = (grounding_path, testcases_path, manifest_path)
    existing = tuple(path for path in targets if path.exists())
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise GroundingExportError(
            f"grounding output already exists ({names}); use overwrite explicitly"
        )
    try:
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise GroundingExportError(f"failed to create grounding output: {exc}") from exc

    grounding_data = _json_bytes(package.model_dump(mode="json"), pretty=True)
    by_id = {snippet.snippet_id: snippet for snippet in package.snippets}
    testcase_data = b"".join(
        _json_bytes(_compact_testcase(case, by_id), pretty=False) + b"\n"
        for case in package.test_cases
    )
    files = (
        _grounding_file(grounding_path.name, grounding_data),
        _grounding_file(testcases_path.name, testcase_data),
    )
    manifest = GroundingExportManifest(
        source_run_id=package.source_run_id,
        source_knowledge_sha256=package.source_knowledge_sha256,
        grounded_at=package.grounded_at,
        files=files,
        test_cases=package.coverage.test_cases,
        snippets=len(package.snippets),
        grounding_complete=package.coverage.grounding_complete,
    )
    manifest_data = _json_bytes(manifest.model_dump(mode="json"), pretty=True)

    _atomic_write(grounding_path, grounding_data)
    _atomic_write(testcases_path, testcase_data)
    _atomic_write(manifest_path, manifest_data)
    return GroundingExportResult(
        output_directory=destination,
        grounding_path=grounding_path,
        testcases_path=testcases_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _compact_testcase(
    case: GroundedTestCase,
    snippets: dict[str, GroundingSnippet],
) -> dict[str, object]:
    repository = [
        _compact_snippet(snippets[snippet_id])
        for snippet_id in case.repository_snippet_ids
    ]
    mcp = [
        _compact_snippet(snippets[snippet_id]) for snippet_id in case.mcp_snippet_ids
    ]
    return {
        "test_case_id": case.context.test_case_id,
        "fields": {field.key: field.value for field in case.context.fields},
        "field_labels": {field.key: field.label for field in case.context.fields},
        "dependency_case_ids": list(case.context.dependency_case_ids),
        "description": case.context.description,
        "portal_evidence_state_ids": list(case.context.evidence_state_ids),
        "description_state_ids": list(case.context.description_state_ids),
        "retrieval_query": case.retrieval_query,
        "repository_context": repository,
        "mcp_context": mcp,
        "limitations": list(case.limitations),
    }


def _compact_snippet(snippet: GroundingSnippet) -> dict[str, object]:
    citation: dict[str, object]
    if snippet.repository is not None:
        citation = snippet.repository.model_dump(mode="json")
    elif snippet.mcp is not None:
        citation = snippet.mcp.model_dump(mode="json")
    else:  # Protected by GroundingSnippet validation.
        raise GroundingExportError("snippet has no citation")
    return {
        "snippet_id": snippet.snippet_id,
        "source_kind": snippet.source_kind.value,
        "title": snippet.title,
        "content": snippet.content,
        "relevance_score": snippet.relevance_score,
        "citation": citation,
    }


def _grounding_file(name: str, data: bytes) -> GroundingFile:
    return GroundingFile(
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
        raise GroundingExportError(f"failed to write {path.name}: {exc}") from exc


__all__ = [
    "GroundingExportError",
    "GroundingExportResult",
    "export_grounding",
]
