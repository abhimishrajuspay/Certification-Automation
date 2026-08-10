"""Atomic campaign plan, checkpoint, and apply-report persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from campaign.models import CampaignCheckpoint, CampaignPlan
from remediation.models import RemediationApplyReport


class CampaignIOError(RuntimeError):
    pass


def export_plan(plan: CampaignPlan, path: Path, *, overwrite: bool = False) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() and not overwrite:
        raise CampaignIOError("campaign plan already exists; use --overwrite")
    _atomic_write(destination, _json_bytes(plan.model_dump(mode="json")))
    return destination


def export_support_matrix(
    plan: CampaignPlan,
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() and not overwrite:
        raise CampaignIOError("campaign support matrix already exists; use --overwrite")
    data = b"".join(
        json.dumps(
            item.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
        for item in plan.assessments
    )
    _atomic_write(destination, data)
    return destination


def load_plan(path: Path) -> CampaignPlan:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise CampaignIOError(f"campaign plan does not exist: {source}")
    try:
        return CampaignPlan.model_validate_json(source.read_bytes())
    except (OSError, ValueError) as exc:
        raise CampaignIOError(f"invalid campaign plan: {exc}") from exc


def save_checkpoint(checkpoint: CampaignCheckpoint, path: Path) -> Path:
    destination = path.expanduser().resolve()
    _atomic_write(destination, _json_bytes(checkpoint.model_dump(mode="json")))
    return destination


def load_checkpoint(path: Path) -> CampaignCheckpoint:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise CampaignIOError(f"campaign checkpoint does not exist: {source}")
    try:
        return CampaignCheckpoint.model_validate_json(source.read_bytes())
    except (OSError, ValueError) as exc:
        raise CampaignIOError(f"invalid campaign checkpoint: {exc}") from exc


def archive_checkpoint_attempt(
    checkpoint_path: Path,
    progress_path: Path,
) -> Path:
    """Archive one resumable attempt exactly once using its content digest."""

    source = checkpoint_path.expanduser().resolve()
    if not source.is_file():
        raise CampaignIOError(f"campaign checkpoint does not exist: {source}")
    checkpoint_bytes = source.read_bytes()
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()[:16]
    destination = source.parent / "history" / digest
    archived_checkpoint = destination / "checkpoint.json"
    if not archived_checkpoint.exists():
        _atomic_write(archived_checkpoint, checkpoint_bytes)
    progress = progress_path.expanduser().resolve()
    archived_progress = destination / "progress.log"
    if progress.is_file() and not archived_progress.exists():
        _atomic_write(archived_progress, progress.read_bytes())
    return destination


def export_apply_report(
    report: RemediationApplyReport,
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() and not overwrite:
        raise CampaignIOError("campaign apply report already exists")
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
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


__all__ = [
    "CampaignIOError",
    "archive_checkpoint_attempt",
    "export_apply_report",
    "export_plan",
    "export_support_matrix",
    "load_checkpoint",
    "load_plan",
    "save_checkpoint",
]
