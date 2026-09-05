"""Deterministic filesystem export for normalized portal knowledge."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from knowledge.models import (
    KnowledgeExportManifest,
    KnowledgeFile,
    PortalKnowledge,
    TestCaseKnowledge,
)


class KnowledgeExportError(RuntimeError):
    """Raised when a knowledge package cannot be written safely."""


@dataclass(frozen=True)
class KnowledgeExportResult:
    """Paths and manifest returned after a successful export."""

    output_directory: Path
    knowledge_path: Path
    testcases_path: Path
    manifest_path: Path
    manifest: KnowledgeExportManifest


def export_knowledge(
    knowledge: PortalKnowledge,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> KnowledgeExportResult:
    """Write stable JSON/JSONL output without mutating crawl evidence."""

    destination = output_directory.expanduser().resolve()
    knowledge_path = destination / "portal_knowledge.json"
    testcases_path = destination / "testcases.jsonl"
    manifest_path = destination / "manifest.json"
    targets = (knowledge_path, testcases_path, manifest_path)
    existing = tuple(path for path in targets if path.exists())
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise KnowledgeExportError(
            f"knowledge output already exists ({names}); use overwrite explicitly"
        )

    try:
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise KnowledgeExportError(f"failed to create knowledge output: {exc}") from exc

    knowledge_data = _json_bytes(knowledge.model_dump(mode="json"), pretty=True)
    testcase_data = b"".join(
        _json_bytes(_compact_testcase(testcase), pretty=False) + b"\n"
        for testcase in knowledge.test_cases
    )
    files = (
        _knowledge_file(knowledge_path.name, knowledge_data),
        _knowledge_file(testcases_path.name, testcase_data),
    )
    manifest = KnowledgeExportManifest(
        source_kind=knowledge.source_kind,
        source_run_id=knowledge.source_run_id,
        normalized_at=knowledge.normalized_at,
        files=files,
        test_cases=knowledge.coverage.test_cases_normalized,
        descriptions=knowledge.coverage.descriptions_captured,
        source_complete=knowledge.coverage.source_bounded_complete,
        testcase_context_complete=knowledge.coverage.testcase_context_complete,
    )
    manifest_data = _json_bytes(manifest.model_dump(mode="json"), pretty=True)

    _atomic_write(knowledge_path, knowledge_data)
    _atomic_write(testcases_path, testcase_data)
    _atomic_write(manifest_path, manifest_data)
    return KnowledgeExportResult(
        output_directory=destination,
        knowledge_path=knowledge_path,
        testcases_path=testcases_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _knowledge_file(name: str, data: bytes) -> KnowledgeFile:
    return KnowledgeFile(
        name=name,
        sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
    )


def _compact_testcase(testcase: TestCaseKnowledge) -> dict[str, object]:
    controls: dict[str, dict[str, object]] = {}
    for control in testcase.controls:
        value = {
            "role": control.role,
            "label": control.label,
            "title": control.title,
            "href": control.href,
            "identifiers": list(control.identifiers),
            "action_kind": control.action_kind.value if control.action_kind else None,
            "risk": control.risk.value if control.risk else None,
            "action_status": (
                control.action_status.value if control.action_status else None
            ),
            "policy_rule": control.policy_rule,
        }
        key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        controls.setdefault(key, value)
    return {
        "test_case_id": testcase.test_case_id,
        "fields": {field.key: field.value for field in testcase.fields},
        "field_labels": {field.key: field.label for field in testcase.fields},
        "dependency_case_ids": list(testcase.dependency_case_ids),
        "description": testcase.description,
        "controls": [controls[key] for key in sorted(controls)],
        "source_urls": sorted({pointer.url for pointer in testcase.evidence}),
        "evidence_state_ids": sorted(
            {pointer.state_id for pointer in testcase.evidence}
        ),
        "description_state_ids": sorted(
            {pointer.state_id for pointer in testcase.description_evidence}
        ),
        "conflicts": list(testcase.conflicts),
    }


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
        raise KnowledgeExportError(f"failed to write {path.name}: {exc}") from exc


__all__ = [
    "KnowledgeExportError",
    "KnowledgeExportResult",
    "export_knowledge",
]
