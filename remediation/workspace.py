"""Confined repository tools and verified change-set application."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Optional

from grounding.models import GroundingSnippet
from grounding.repository import (
    DEFAULT_EXCLUDED_DIRECTORIES,
    DEFAULT_SUFFIXES,
    RepositoryIndex,
    RepositoryIndexConfig,
)
from remediation.models import (
    AppliedFile,
    ApplyStatus,
    FileOperation,
    RemediationApplyReport,
    RemediationPlan,
    RepositoryFileChange,
    VerificationResult,
)
from scraper.models import CapturePolicy
from scraper.redaction import redact_text


CODE_SUFFIXES = tuple(
    sorted(
        set(DEFAULT_SUFFIXES)
        | {
            ".c",
            ".cc",
            ".conf",
            ".cpp",
            ".cs",
            ".css",
            ".go",
            ".gradle",
            ".groovy",
            ".h",
            ".hpp",
            ".hs",
            ".html",
            ".ini",
            ".java",
            ".js",
            ".jsx",
            ".kt",
            ".kts",
            ".lhs",
            ".cabal",
            ".dhall",
            ".nix",
            ".php",
            ".py",
            ".rb",
            ".rs",
            ".scala",
            ".sh",
            ".sql",
            ".toml",
            ".ts",
            ".tsx",
        }
    )
)
_SENSITIVE_FILE_NAMES = {
    ".env",
    ".env.local",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
}


class RepositoryWorkspaceError(RuntimeError):
    """Raised when a repository tool or apply invariant is violated."""


class RepositoryWorkspace:
    """Read/search a repository and apply only hash-locked complete files."""

    def __init__(
        self,
        root: Path,
        *,
        maximum_file_bytes: int = 2_000_000,
        maximum_total_bytes: int = 300_000_000,
    ) -> None:
        expanded = root.expanduser()
        if expanded.is_symlink():
            raise RepositoryWorkspaceError("repository root cannot be a symlink")
        self.root = expanded.resolve()
        if not self.root.is_dir():
            raise RepositoryWorkspaceError(f"repository is not a directory: {root}")
        if self.root == Path(self.root.anchor) or self.root == Path.home().resolve():
            raise RepositoryWorkspaceError("repository root is too broad")
        self.maximum_file_bytes = maximum_file_bytes
        self.index = RepositoryIndex.build(
            self.root,
            RepositoryIndexConfig(
                suffixes=CODE_SUFFIXES,
                maximum_file_bytes=maximum_file_bytes,
                maximum_total_bytes=maximum_total_bytes,
            ),
        )

    @property
    def repository_id(self) -> str:
        return self.index.summary.repository_id

    def search(self, query: str, *, limit: int = 8) -> tuple[GroundingSnippet, ...]:
        return self.index.search(query, limit=limit)

    def read_file(self, relative_path: str) -> tuple[str, str]:
        path = self._resolve(relative_path, must_exist=True)
        self._reject_sensitive(path)
        if not path.is_file():
            raise RepositoryWorkspaceError("repository path is not a regular file")
        data = path.read_bytes()
        if len(data) > self.maximum_file_bytes:
            raise RepositoryWorkspaceError("repository file exceeds the read limit")
        if b"\x00" in data:
            raise RepositoryWorkspaceError("binary repository files cannot be read")
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RepositoryWorkspaceError("repository file is not UTF-8 text") from exc
        if redact_text(content, CapturePolicy().redacted_names) != content:
            raise RepositoryWorkspaceError(
                "repository file contains sensitive values and cannot enter an LLM prompt"
            )
        return content, hashlib.sha256(data).hexdigest()

    def apply(
        self,
        plan: RemediationPlan,
        *,
        approved_plan_id: str,
        verification_commands: tuple[tuple[str, ...], ...],
        verification_timeout_seconds: int = 600,
        allow_unverified: bool = False,
    ) -> RemediationApplyReport:
        """Verify in a disposable copy, then atomically update the real repository."""

        if approved_plan_id != plan.plan_id:
            return self._rejected(plan, "approval hash does not match plan_id")
        if self._current_repository_id() != plan.repository_id:
            return self._rejected(plan, "repository changed since plan generation")
        if not verification_commands and not allow_unverified:
            return self._rejected(plan, "at least one verification command is required")
        try:
            self._validate_changes(plan.proposal.changes)
        except RepositoryWorkspaceError as exc:
            return self._rejected(plan, str(exc))

        verification: tuple[VerificationResult, ...] = ()
        with tempfile.TemporaryDirectory(prefix="cz-remediation-") as temporary:
            stage = Path(temporary) / "repository"
            try:
                shutil.copytree(
                    self.root,
                    stage,
                    symlinks=True,
                    ignore=self._copy_ignore,
                )
                self._reject_stage_symlinks(stage)
                self._apply_to_root(stage, plan.proposal.changes)
                verification = self._verify(
                    stage,
                    verification_commands,
                    timeout_seconds=verification_timeout_seconds,
                )
            except (OSError, RepositoryWorkspaceError) as exc:
                return self._rolled_back(plan, verification, str(exc))
            if any(not result.passed for result in verification):
                return self._rolled_back(
                    plan,
                    verification,
                    "one or more verification commands failed",
                )

        # Compare-and-swap is repeated after potentially long verification.
        try:
            if self._current_repository_id() != plan.repository_id:
                raise RepositoryWorkspaceError(
                    "repository changed while verification was running"
                )
            self._validate_changes(plan.proposal.changes)
            files = self._apply_to_root(self.root, plan.proposal.changes)
        except (OSError, RepositoryWorkspaceError) as exc:
            return self._rolled_back(plan, verification, str(exc))
        return RemediationApplyReport(
            plan_id=plan.plan_id,
            repository_id_before=plan.repository_id,
            applied_at=datetime.now(timezone.utc),
            status=ApplyStatus.APPLIED,
            files=files,
            verification=verification,
        )

    def _current_repository_id(self) -> str:
        return RepositoryIndex.build(self.root, self.index.config).summary.repository_id

    def _validate_changes(self, changes: tuple[RepositoryFileChange, ...]) -> None:
        for change in changes:
            path = self._resolve(
                change.path,
                must_exist=change.operation == FileOperation.REPLACE,
            )
            self._reject_sensitive(path)
            content_bytes = change.content.encode("utf-8")
            if (
                redact_text(change.content, CapturePolicy().redacted_names)
                != change.content
            ):
                raise RepositoryWorkspaceError(
                    f"planned file appears to contain a hardcoded secret: {change.path}"
                )
            if len(content_bytes) > self.maximum_file_bytes:
                raise RepositoryWorkspaceError(
                    f"planned file exceeds size limit: {change.path}"
                )
            if change.operation == FileOperation.CREATE:
                if path.exists():
                    raise RepositoryWorkspaceError(
                        f"create target already exists: {change.path}"
                    )
            else:
                if not path.is_file() or path.is_symlink():
                    raise RepositoryWorkspaceError(
                        f"replace target is not a regular file: {change.path}"
                    )
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != change.expected_sha256:
                    raise RepositoryWorkspaceError(
                        f"replace target changed after planning: {change.path}"
                    )

    def _apply_to_root(
        self,
        root: Path,
        changes: tuple[RepositoryFileChange, ...],
    ) -> tuple[AppliedFile, ...]:
        applied: list[AppliedFile] = []
        backups: list[tuple[Path, Optional[bytes], Optional[int]]] = []
        try:
            for change in changes:
                path = _contained_path(root, change.path)
                before = path.read_bytes() if path.exists() else None
                mode = path.stat().st_mode if path.exists() else None
                backups.append((path, before, mode))
                path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write(path, change.content.encode("utf-8"), mode=mode)
                applied.append(
                    AppliedFile(
                        path=change.path,
                        before_sha256=(
                            hashlib.sha256(before).hexdigest()
                            if before is not None
                            else None
                        ),
                        after_sha256=hashlib.sha256(
                            change.content.encode("utf-8")
                        ).hexdigest(),
                    )
                )
        except OSError:
            for path, before, mode in reversed(backups):
                if before is None:
                    path.unlink(missing_ok=True)
                else:
                    _atomic_write(path, before, mode=mode)
            raise
        return tuple(applied)

    @staticmethod
    def _verify(
        stage: Path,
        commands: tuple[tuple[str, ...], ...],
        *,
        timeout_seconds: int,
    ) -> tuple[VerificationResult, ...]:
        results: list[VerificationResult] = []
        process_environment = _verification_environment(stage)
        for argv in commands:
            if not argv or any(not value for value in argv):
                raise RepositoryWorkspaceError("verification argv cannot be empty")
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    argv,
                    cwd=stage,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                    env=process_environment,
                    timeout=timeout_seconds,
                    check=False,
                )
                timed_out = False
                exit_code: Optional[int] = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                exit_code = None
                stdout = exc.stdout or b""
                stderr = exc.stderr or b""
            duration_ms = int((time.monotonic() - started) * 1_000)
            result = VerificationResult(
                argv=argv,
                exit_code=exit_code,
                timed_out=timed_out,
                duration_ms=duration_ms,
                stdout_sha256=hashlib.sha256(stdout).hexdigest(),
                stderr_sha256=hashlib.sha256(stderr).hexdigest(),
                passed=not timed_out and exit_code == 0,
            )
            results.append(result)
            if not result.passed:
                break
        return tuple(results)

    def _resolve(self, relative_path: str, *, must_exist: bool) -> Path:
        path = _contained_path(self.root, relative_path)
        if must_exist and not path.exists():
            raise RepositoryWorkspaceError(
                f"repository file does not exist: {relative_path}"
            )
        current = self.root
        for part in PurePosixPath(relative_path).parts:
            current = current / part
            if current.exists() and current.is_symlink():
                raise RepositoryWorkspaceError(
                    "repository tools cannot follow symlinks"
                )
        return path

    @staticmethod
    def _reject_sensitive(path: Path) -> None:
        if path.name.casefold() in _SENSITIVE_FILE_NAMES:
            raise RepositoryWorkspaceError(
                "sensitive credential files are out of scope"
            )

    @staticmethod
    def _copy_ignore(directory: str, names: list[str]) -> set[str]:
        ignored = {
            name
            for name in names
            if name in DEFAULT_EXCLUDED_DIRECTORIES
            or name.casefold() in _SENSITIVE_FILE_NAMES
        }
        if Path(directory).name == ".git":
            ignored.update(names)
        return ignored

    @staticmethod
    def _reject_stage_symlinks(stage: Path) -> None:
        for directory, names, files in os.walk(stage, followlinks=False):
            for name in (*names, *files):
                if (Path(directory) / name).is_symlink():
                    raise RepositoryWorkspaceError(
                        "staged verification does not permit repository symlinks"
                    )

    def _rejected(self, plan: RemediationPlan, reason: str) -> RemediationApplyReport:
        return RemediationApplyReport(
            plan_id=plan.plan_id,
            repository_id_before=self.repository_id,
            applied_at=datetime.now(timezone.utc),
            status=ApplyStatus.REJECTED,
            files=(),
            verification=(),
            failure_reason=reason,
        )

    def _rolled_back(
        self,
        plan: RemediationPlan,
        verification: tuple[VerificationResult, ...],
        reason: str,
    ) -> RemediationApplyReport:
        return RemediationApplyReport(
            plan_id=plan.plan_id,
            repository_id_before=plan.repository_id,
            applied_at=datetime.now(timezone.utc),
            status=ApplyStatus.ROLLED_BACK,
            files=(),
            verification=verification,
            failure_reason=reason,
        )


def _contained_path(root: Path, relative_path: str) -> Path:
    if "\\" in relative_path:
        raise RepositoryWorkspaceError("repository paths must use POSIX separators")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise RepositoryWorkspaceError("repository path must be relative and contained")
    candidate = root.joinpath(*pure.parts)
    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise RepositoryWorkspaceError("repository path escapes its root") from exc
    if pure.parts[0] == ".git":
        raise RepositoryWorkspaceError("repository changes cannot target .git")
    return candidate


def _atomic_write(path: Path, data: bytes, *, mode: Optional[int]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _verification_environment(stage: Path) -> dict[str, str]:
    """Retain toolchain discovery while excluding ambient credentials."""

    home = stage / ".cz-verification-home"
    temporary = stage / ".cz-verification-tmp"
    home.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)
    allowed_names = (
        "COMSPEC",
        "JAVA_HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
    )
    environment = {
        name: os.environ[name] for name in allowed_names if name in os.environ
    }
    environment.update(
        {
            "HOME": str(home),
            "TMPDIR": str(temporary),
            "CZ_REMEDIATION_VERIFY": "1",
        }
    )
    return environment


__all__ = [
    "CODE_SUFFIXES",
    "RepositoryWorkspace",
    "RepositoryWorkspaceError",
]
