"""Atomic persistence for remediation plans and apply reports."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from remediation.models import RemediationApplyReport, RemediationPlan


class RemediationIOError(RuntimeError):
    pass


def export_plan(plan: RemediationPlan, path: Path, *, overwrite: bool = False) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() and not overwrite:
        raise RemediationIOError("remediation plan already exists; use --overwrite")
    _atomic_write(destination, _json_bytes(plan.model_dump(mode="json")))
    return destination


def load_plan(path: Path) -> RemediationPlan:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise RemediationIOError(f"remediation plan does not exist: {source}")
    try:
        return RemediationPlan.model_validate_json(source.read_bytes())
    except (OSError, ValueError) as exc:
        raise RemediationIOError(f"invalid remediation plan: {exc}") from exc


def export_apply_report(
    report: RemediationApplyReport,
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() and not overwrite:
        raise RemediationIOError("remediation apply report already exists")
    _atomic_write(destination, _json_bytes(report.model_dump(mode="json")))
    return destination


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


__all__ = [
    "RemediationIOError",
    "export_apply_report",
    "export_plan",
    "load_plan",
]
