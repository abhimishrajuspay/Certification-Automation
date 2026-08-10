"""Bounded, deterministic repository indexing and cited excerpt retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from grounding.models import (
    GroundingSnippet,
    GroundingSourceKind,
    RepositoryCitation,
    RepositoryIndexSummary,
)
from scraper.models import CapturePolicy
from scraper.redaction import redact_text


DEFAULT_SUFFIXES = (
    ".avsc",
    ".graphql",
    ".http",
    ".json",
    ".md",
    ".proto",
    ".properties",
    ".txt",
    ".wsdl",
    ".xml",
    ".xsd",
    ".yaml",
    ".yml",
)
DEFAULT_EXCLUDED_DIRECTORIES = (
    ".deprecated",
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".stack-work",
    ".venv",
    "artifacts",
    "build",
    "dist",
    "dist-newstyle",
    "dummy-portal",
    "dummy_portal",
    "htmlcov",
    "logs",
    "node_modules",
    "target",
    "venv",
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "case",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "test",
    "that",
    "this",
    "with",
}
_GENERIC_RETRIEVAL_TERMS = {
    "api",
    "code",
    "codes",
    "data",
    "default",
    "error",
    "expected",
    "handling",
    "integration",
    "negative",
    "payload",
    "pending",
    "request",
    "response",
    "schema",
    "status",
    "validation",
}


class RepositoryIndexError(RuntimeError):
    """Raised when a repository cannot be indexed safely."""


@dataclass(frozen=True)
class RepositoryIndexConfig:
    """Bounds and safety settings for local repository retrieval."""

    suffixes: tuple[str, ...] = DEFAULT_SUFFIXES
    excluded_directories: tuple[str, ...] = DEFAULT_EXCLUDED_DIRECTORIES
    maximum_file_bytes: int = 2_000_000
    maximum_total_bytes: int = 200_000_000
    maximum_files: int = 20_000
    excerpt_context_lines: int = 4
    maximum_excerpt_characters: int = 6_000
    redacted_names: tuple[str, ...] = CapturePolicy().redacted_names

    def __post_init__(self) -> None:
        if not self.suffixes or any(
            not suffix.startswith(".") for suffix in self.suffixes
        ):
            raise ValueError("repository suffixes must be non-empty dotted extensions")
        for value in (
            self.maximum_file_bytes,
            self.maximum_total_bytes,
            self.maximum_files,
            self.maximum_excerpt_characters,
        ):
            if value <= 0:
                raise ValueError("repository index limits must be positive")
        if self.excerpt_context_lines < 0:
            raise ValueError("excerpt_context_lines cannot be negative")


@dataclass(frozen=True)
class _IndexedDocument:
    path: str
    file_sha256: str
    lines: tuple[str, ...]
    tokens: frozenset[str]
    redacted: bool
    byte_size: int


class RepositoryIndex:
    """In-memory token index containing only bounded redacted text."""

    def __init__(
        self,
        documents: tuple[_IndexedDocument, ...],
        summary: RepositoryIndexSummary,
        config: RepositoryIndexConfig,
    ) -> None:
        self._documents = documents
        self.summary = summary
        self.config = config
        postings: dict[str, list[int]] = defaultdict(list)
        for index, document in enumerate(documents):
            for token in document.tokens:
                postings[token].append(index)
        self._postings = {
            token: tuple(indices) for token, indices in sorted(postings.items())
        }

    @classmethod
    def build(
        cls,
        root: Path,
        config: Optional[RepositoryIndexConfig] = None,
    ) -> "RepositoryIndex":
        """Index supported regular files without following directory symlinks."""

        settings = config or RepositoryIndexConfig()
        expanded = root.expanduser()
        if expanded.is_symlink():
            raise RepositoryIndexError("repository root cannot be a symbolic link")
        resolved = expanded.resolve()
        if not resolved.is_dir():
            raise RepositoryIndexError(f"repository path is not a directory: {root}")
        if resolved == Path(resolved.anchor):
            raise RepositoryIndexError(
                "filesystem root cannot be used as repository path"
            )

        documents: list[_IndexedDocument] = []
        file_digests: list[tuple[str, str]] = []
        examined = 0
        skipped = 0
        indexed_bytes = 0
        limitations: set[str] = set()
        suffixes = {suffix.casefold() for suffix in settings.suffixes}
        excluded = set(settings.excluded_directories)

        try:
            walker = os.walk(resolved, topdown=True, followlinks=False)
            for directory, names, files in walker:
                names[:] = sorted(
                    name
                    for name in names
                    if name not in excluded
                    and not name.startswith(".")
                    and not (Path(directory) / name).is_symlink()
                )
                for name in sorted(files):
                    examined += 1
                    path = Path(directory) / name
                    relative = path.relative_to(resolved).as_posix()
                    if (
                        path.is_symlink()
                        or name.startswith(".")
                        or path.suffix.casefold() not in suffixes
                    ):
                        skipped += 1
                        continue
                    if len(documents) >= settings.maximum_files:
                        skipped += 1
                        limitations.add("maximum repository file count reached")
                        continue
                    try:
                        resolved_file = path.resolve(strict=True)
                        resolved_file.relative_to(resolved)
                        size = path.stat().st_size
                    except (OSError, ValueError):
                        skipped += 1
                        limitations.add(
                            "one or more repository files could not be safely resolved"
                        )
                        continue
                    if size > settings.maximum_file_bytes:
                        skipped += 1
                        limitations.add(
                            "one or more repository files exceeded the size limit"
                        )
                        continue
                    if indexed_bytes + size > settings.maximum_total_bytes:
                        skipped += 1
                        limitations.add("maximum repository byte budget reached")
                        continue
                    try:
                        data = path.read_bytes()
                    except OSError:
                        skipped += 1
                        limitations.add(
                            "one or more repository files could not be read"
                        )
                        continue
                    if len(data) > settings.maximum_file_bytes:
                        skipped += 1
                        limitations.add(
                            "one or more repository files grew beyond the size limit"
                        )
                        continue
                    if indexed_bytes + len(data) > settings.maximum_total_bytes:
                        skipped += 1
                        limitations.add("maximum repository byte budget reached")
                        continue
                    if b"\x00" in data:
                        skipped += 1
                        limitations.add(
                            "one or more supported files contained binary data"
                        )
                        continue
                    raw_text = data.decode("utf-8", errors="replace")
                    safe_text = redact_text(raw_text, settings.redacted_names)
                    file_sha256 = hashlib.sha256(data).hexdigest()
                    documents.append(
                        _IndexedDocument(
                            path=relative,
                            file_sha256=file_sha256,
                            lines=tuple(safe_text.splitlines()) or ("",),
                            tokens=frozenset(_tokenize(f"{relative} {safe_text}")),
                            redacted=safe_text != raw_text,
                            byte_size=len(data),
                        )
                    )
                    file_digests.append((relative, file_sha256))
                    indexed_bytes += len(data)
        except OSError as exc:
            raise RepositoryIndexError(f"failed to scan repository: {exc}") from exc

        repository_id = _hash_json(file_digests)
        summary = RepositoryIndexSummary(
            repository_id=repository_id,
            root_name=resolved.name,
            files_examined=examined,
            files_indexed=len(documents),
            files_skipped=skipped,
            bytes_indexed=indexed_bytes,
            suffixes=tuple(sorted(suffixes)),
            limitations=tuple(sorted(limitations)),
        )
        return cls(tuple(documents), summary, settings)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        anchor_terms: tuple[str, ...] = (),
    ) -> tuple[GroundingSnippet, ...]:
        """Return the best redacted line windows with content-hash citations."""

        if limit <= 0:
            raise ValueError("repository search limit must be positive")
        query_tokens = frozenset(_tokenize(query))
        if not query_tokens:
            return ()
        anchor_tokens = frozenset(_tokenize(" ".join(anchor_terms)))
        anchor_identifiers = frozenset(_identifier_tokens(anchor_terms))
        if anchor_terms:
            required_tokens = anchor_tokens - _GENERIC_RETRIEVAL_TERMS
            if not required_tokens:
                return ()
        else:
            required_tokens = query_tokens - _GENERIC_RETRIEVAL_TERMS
        candidates = {
            index for token in query_tokens for index in self._postings.get(token, ())
        }
        scored: list[tuple[int, str, int, _IndexedDocument]] = []
        for index in candidates:
            document = self._documents[index]
            if anchor_identifiers and not anchor_identifiers.intersection(
                document.tokens
            ):
                continue
            minimum_required_matches = min(2, len(required_tokens))
            if len(required_tokens & document.tokens) < minimum_required_matches:
                continue
            path_overlap = len(query_tokens & set(_tokenize(document.path)))
            best_line = 0
            best_line_score = 0
            total_matches = 0
            for line_number, line in enumerate(document.lines):
                line_score = len(query_tokens & set(_tokenize(line)))
                total_matches += line_score
                if line_score > best_line_score:
                    best_line = line_number
                    best_line_score = line_score
            score = (path_overlap * 20) + (best_line_score * 8) + min(total_matches, 20)
            if score > 0:
                scored.append((score, document.path, best_line, document))

        snippets: list[GroundingSnippet] = []
        for score, _, center, document in sorted(
            scored,
            key=lambda item: (-item[0], item[1], item[2]),
        )[:limit]:
            start_index = max(0, center - self.config.excerpt_context_lines)
            end_index = min(
                len(document.lines), center + self.config.excerpt_context_lines + 1
            )
            excerpt = "\n".join(document.lines[start_index:end_index]).strip()
            if not excerpt:
                continue
            excerpt = excerpt[: self.config.maximum_excerpt_characters]
            excerpt_sha256 = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
            citation = RepositoryCitation(
                repository_id=self.summary.repository_id,
                path=document.path,
                file_sha256=document.file_sha256,
                line_start=start_index + 1,
                line_end=end_index,
                excerpt_sha256=excerpt_sha256,
                redacted=document.redacted,
            )
            snippet_id = _hash_json(
                {
                    "source": GroundingSourceKind.REPOSITORY.value,
                    "citation": citation.model_dump(mode="json"),
                    "content": excerpt,
                }
            )
            snippets.append(
                GroundingSnippet(
                    snippet_id=snippet_id,
                    source_kind=GroundingSourceKind.REPOSITORY,
                    title=(
                        f"{document.path}:{citation.line_start}-{citation.line_end}"
                    ),
                    content=excerpt,
                    relevance_score=score,
                    repository=citation,
                )
            )
        return tuple(snippets)


def _tokenize(value: str) -> tuple[str, ...]:
    originals = re.findall(r"[A-Za-z0-9]+", value.casefold())
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    tokens = re.findall(r"[A-Za-z0-9]+", expanded.casefold())
    return tuple(
        dict.fromkeys(
            token
            for token in (*originals, *tokens)
            if len(token) >= 2 and token not in _STOPWORDS
        )
    )


def _identifier_tokens(values: tuple[str, ...]) -> tuple[str, ...]:
    """Return exact compound identifiers that should anchor retrieval."""

    identifiers: list[str] = []
    for value in values:
        stripped = value.strip()
        if not stripped or any(character.isspace() for character in stripped):
            continue
        normalized = re.sub(r"[^A-Za-z0-9]", "", stripped).casefold()
        if len(normalized) >= 3 and normalized not in _GENERIC_RETRIEVAL_TERMS:
            identifiers.append(normalized)
    return tuple(dict.fromkeys(identifiers))


def _hash_json(value: object) -> str:
    data = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "DEFAULT_EXCLUDED_DIRECTORIES",
    "DEFAULT_SUFFIXES",
    "RepositoryIndex",
    "RepositoryIndexConfig",
    "RepositoryIndexError",
]
