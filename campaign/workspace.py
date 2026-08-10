"""Staged repository overlay used by the campaign agent."""

from __future__ import annotations

import hashlib
from pathlib import Path

from grounding.models import GroundingSnippet
from remediation.models import FileOperation, RepositoryFileChange
from remediation.workspace import RepositoryWorkspace, RepositoryWorkspaceError


class CampaignWorkspace:
    """Read the real repository while keeping proposed writes in memory."""

    def __init__(self, root: Path) -> None:
        self.repository = RepositoryWorkspace(root)
        self._read_paths: set[str] = set()
        self._staged: dict[str, RepositoryFileChange] = {}

    @property
    def repository_id(self) -> str:
        return self.repository.repository_id

    @property
    def summary(self) -> dict[str, object]:
        return self.repository.index.summary.model_dump(mode="json")

    @property
    def read_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self._read_paths))

    @property
    def staged_changes(self) -> tuple[RepositoryFileChange, ...]:
        return tuple(self._staged[path] for path in sorted(self._staged))

    def restore(
        self,
        *,
        read_paths: tuple[str, ...],
        staged_changes: tuple[RepositoryFileChange, ...],
    ) -> None:
        self.repository.validate_changes(staged_changes)
        self._read_paths.update(read_paths)
        self._staged.update((change.path, change) for change in staged_changes)

    def search(self, query: str, *, limit: int) -> tuple[GroundingSnippet, ...]:
        return self.repository.search(query, limit=limit)

    def read_file(self, relative_path: str) -> tuple[str, str]:
        staged = self._staged.get(relative_path)
        if staged is not None:
            content = staged.content
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        else:
            content, digest = self.repository.read_file(relative_path)
        self._read_paths.add(relative_path)
        return content, digest

    def stage(self, change: RepositoryFileChange) -> RepositoryFileChange:
        """Validate and retain one complete file without touching the repository."""

        if change.path in self._staged:
            raise RepositoryWorkspaceError(
                "a staged file cannot be replaced twice; submit its final content once"
            )
        if change.operation == FileOperation.REPLACE:
            if change.path not in self._read_paths:
                raise RepositoryWorkspaceError(
                    f"replacement was not read first: {change.path}"
                )
            _, digest = self.repository.read_file(change.path)
            if change.expected_sha256 != digest:
                raise RepositoryWorkspaceError(
                    f"replacement digest is stale: {change.path}"
                )
        self.repository.validate_changes((change,))
        self._staged[change.path] = change
        return change


__all__ = ["CampaignWorkspace"]
