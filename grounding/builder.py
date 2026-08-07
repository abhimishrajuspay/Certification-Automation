"""Assemble normalized portal facts with cited repository and MCP context."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from grounding.mcp import DEFAULT_READ_ONLY_TOOLS, MCPClient, MCPError
from grounding.models import (
    GroundedTestCase,
    GroundingCoverage,
    GroundingPackage,
    GroundingSnippet,
    MCPServerSummary,
    PortalTestCaseContext,
)
from grounding.repository import RepositoryIndex
from knowledge.models import KnowledgeExportManifest, PortalKnowledge, TestCaseKnowledge


_MCP_GROUP_KEYS = {
    "api",
    "api_name",
    "api_type",
    "flow",
    "flow_type",
    "message_type",
    "operation",
    "request",
    "request_type",
    "service",
    "service_name",
}
_LOW_SIGNAL_KEYS = {
    "column_1",
    "dependency",
    "dependency_case",
    "index",
    "no",
    "number",
    "rc",
    "row",
    "serial",
    "sr_no",
    "status",
    "test_case_id",
    "test_data",
    "testdata",
}
_QUERY_STOPWORDS = {
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
    "of",
    "on",
    "or",
    "test",
    "the",
    "to",
    "with",
}


class GroundingBuildError(RuntimeError):
    """Raised when source knowledge cannot safely enter grounding."""


@dataclass(frozen=True)
class GroundingBuildConfig:
    """Retrieval breadth, concurrency, and source-completeness policy."""

    repository_results_per_case: int = 5
    mcp_results_per_tool: int = 3
    mcp_tools: tuple[str, ...] = DEFAULT_READ_ONLY_TOOLS
    maximum_mcp_groups: int = 100
    mcp_concurrency: int = 4
    maximum_query_characters: int = 1_500
    allow_incomplete_source: bool = False

    def __post_init__(self) -> None:
        for value in (
            self.repository_results_per_case,
            self.mcp_results_per_tool,
            self.maximum_mcp_groups,
            self.mcp_concurrency,
            self.maximum_query_characters,
        ):
            if value <= 0:
                raise ValueError("grounding build limits must be positive")
        unsupported = set(self.mcp_tools) - set(DEFAULT_READ_ONLY_TOOLS)
        if unsupported:
            raise ValueError(
                f"automatic MCP grounding supports only read-only search tools: "
                f"{sorted(unsupported)}"
            )
        if len(self.mcp_tools) != len(set(self.mcp_tools)):
            raise ValueError("mcp_tools must be unique")


@dataclass(frozen=True)
class LoadedKnowledge:
    """Validated source package and digest used by the grounding manifest."""

    path: Path
    sha256: str
    knowledge: PortalKnowledge


class GroundingBuilder:
    """Join portal evidence with bounded local and remote retrieval results."""

    def __init__(
        self,
        knowledge: PortalKnowledge,
        knowledge_sha256: str,
        repository: RepositoryIndex,
        *,
        mcp_client: Optional[MCPClient] = None,
        config: Optional[GroundingBuildConfig] = None,
    ) -> None:
        self.knowledge = knowledge
        self.knowledge_sha256 = knowledge_sha256
        self.repository = repository
        self.mcp_client = mcp_client
        self.config = config or GroundingBuildConfig()

    async def build(self) -> GroundingPackage:
        """Build a cited package without invoking an LLM or generative MCP tool."""

        if (
            not self.knowledge.coverage.testcase_context_complete
            and not self.config.allow_incomplete_source
        ):
            raise GroundingBuildError(
                "grounding requires complete normalized testcase context; use "
                "allow_incomplete_source only for diagnostics"
            )

        snippets: dict[str, GroundingSnippet] = {}
        case_queries: dict[str, str] = {}
        repository_ids: dict[str, tuple[str, ...]] = {}
        for case in self.knowledge.test_cases:
            query = _case_query(case, self.config.maximum_query_characters)
            case_queries[case.test_case_id] = query
            matches = self.repository.search(
                query,
                limit=self.config.repository_results_per_case,
                anchor_terms=_mcp_group_key(case),
            )
            for snippet in matches:
                snippets.setdefault(snippet.snippet_id, snippet)
            repository_ids[case.test_case_id] = tuple(
                snippet.snippet_id for snippet in matches
            )

        mcp_ids: dict[str, list[str]] = defaultdict(list)
        mcp_summary, mcp_limitations = await self._retrieve_mcp(snippets, mcp_ids)

        grounded_cases: list[GroundedTestCase] = []
        for case in self.knowledge.test_cases:
            repository_case_ids = repository_ids[case.test_case_id]
            mcp_case_ids = tuple(dict.fromkeys(mcp_ids[case.test_case_id]))
            limitations: list[str] = []
            if not repository_case_ids:
                limitations.append("no relevant repository excerpt was found")
            if self.mcp_client is not None and not mcp_case_ids:
                limitations.append("no MCP search context was captured")
            grounded_cases.append(
                GroundedTestCase(
                    context=_portal_context(case),
                    retrieval_query=case_queries[case.test_case_id],
                    repository_snippet_ids=repository_case_ids,
                    mcp_snippet_ids=mcp_case_ids,
                    limitations=tuple(limitations),
                )
            )

        repository_grounded = sum(
            bool(case.repository_snippet_ids) for case in grounded_cases
        )
        mcp_grounded = sum(bool(case.mcp_snippet_ids) for case in grounded_cases)
        missing = tuple(
            case.context.test_case_id
            for case in grounded_cases
            if not (case.repository_snippet_ids or case.mcp_snippet_ids)
        )
        externally_grounded = len(grounded_cases) - len(missing)
        mcp_required = self.mcp_client is not None
        grounding_complete = bool(
            self.knowledge.coverage.testcase_context_complete
            and not missing
            and (
                not mcp_required
                or (
                    mcp_summary.retrieval_complete
                    and (not grounded_cases or mcp_summary.calls_succeeded > 0)
                )
            )
        )
        limitations = set(self.repository.summary.limitations)
        limitations.update(mcp_limitations)
        if not self.knowledge.coverage.testcase_context_complete:
            limitations.add("source normalized testcase context is incomplete")
        if missing:
            limitations.add(
                f"{len(missing)} testcases have no repository or MCP context"
            )
        if mcp_required and not mcp_summary.available:
            limitations.add("configured MCP server was unavailable")

        return GroundingPackage(
            source_run_id=self.knowledge.source_run_id,
            source_knowledge_sha256=self.knowledge_sha256,
            grounded_at=datetime.now(timezone.utc),
            repository=self.repository.summary,
            mcp=mcp_summary,
            snippets=tuple(sorted(snippets.values(), key=lambda item: item.snippet_id)),
            test_cases=tuple(grounded_cases),
            coverage=GroundingCoverage(
                source_testcase_context_complete=(
                    self.knowledge.coverage.testcase_context_complete
                ),
                mcp_required=mcp_required,
                test_cases=len(grounded_cases),
                repository_grounded=repository_grounded,
                mcp_grounded=mcp_grounded,
                externally_grounded=externally_grounded,
                missing_external_context_ids=missing,
                grounding_complete=grounding_complete,
                limitations=tuple(sorted(limitations)),
            ),
        )

    async def _retrieve_mcp(
        self,
        snippets: dict[str, GroundingSnippet],
        case_snippet_ids: dict[str, list[str]],
    ) -> tuple[MCPServerSummary, tuple[str, ...]]:
        if self.mcp_client is None:
            return MCPServerSummary(configured=False, available=False), ()

        errors: set[str] = set()
        limitations: set[str] = set()
        try:
            identity, tools = await self.mcp_client.connect()
        except MCPError as exc:
            message = str(exc)
            return (
                MCPServerSummary(
                    configured=True,
                    available=False,
                    endpoint=self.mcp_client.config.endpoint,
                    errors=(message,),
                ),
                (message,),
            )

        discovered = tuple(tool.name for tool in tools)
        selected_tools = tuple(
            tool for tool in self.config.mcp_tools if tool in set(discovered)
        )
        absent = set(self.config.mcp_tools) - set(selected_tools)
        if absent:
            limitations.add(f"MCP search tools were not advertised: {sorted(absent)}")
        if not selected_tools:
            return (
                MCPServerSummary(
                    configured=True,
                    available=True,
                    endpoint=self.mcp_client.config.endpoint,
                    server_name=identity.name,
                    server_version=identity.version,
                    protocol_version=identity.protocol_version,
                    discovered_tools=discovered,
                    errors=tuple(sorted(limitations)),
                ),
                tuple(sorted(limitations)),
            )

        groups: dict[tuple[str, ...], list[TestCaseKnowledge]] = defaultdict(list)
        for case in self.knowledge.test_cases:
            groups[_mcp_group_key(case)].append(case)
        ordered_groups = sorted(groups.items(), key=lambda item: item[0])
        groups_truncated = False
        if len(ordered_groups) > self.config.maximum_mcp_groups:
            groups_truncated = True
            limitations.add(
                "maximum MCP query-group count reached; remaining groups were not queried"
            )
            ordered_groups = ordered_groups[: self.config.maximum_mcp_groups]

        semaphore = asyncio.Semaphore(self.config.mcp_concurrency)

        async def retrieve(
            group_key: tuple[str, ...],
            cases: list[TestCaseKnowledge],
            tool_name: str,
        ) -> tuple[tuple[str, ...], str, tuple[GroundingSnippet, ...], Optional[str]]:
            query = _mcp_query(group_key, cases, self.config.maximum_query_characters)
            arguments = _mcp_arguments(
                tool_name,
                query,
                self.config.mcp_results_per_tool,
            )
            try:
                async with semaphore:
                    result = await self.mcp_client.call_tool(tool_name, arguments)
                return (
                    tuple(case.test_case_id for case in cases),
                    tool_name,
                    self.mcp_client.to_snippets(result),
                    None,
                )
            except MCPError as exc:
                return (
                    tuple(case.test_case_id for case in cases),
                    tool_name,
                    (),
                    str(exc),
                )

        results = await asyncio.gather(
            *(
                retrieve(group_key, cases, tool_name)
                for group_key, cases in ordered_groups
                for tool_name in selected_tools
            )
        )
        invoked: set[str] = set()
        succeeded = 0
        for case_ids, tool_name, retrieved, error in results:
            invoked.add(tool_name)
            if error:
                errors.add(error)
                continue
            succeeded += 1
            for snippet in retrieved:
                snippets.setdefault(snippet.snippet_id, snippet)
            ids = [snippet.snippet_id for snippet in retrieved]
            for case_id in case_ids:
                case_snippet_ids[case_id].extend(ids)

        limitations.update(errors)
        return (
            MCPServerSummary(
                configured=True,
                available=True,
                endpoint=self.mcp_client.config.endpoint,
                server_name=identity.name,
                server_version=identity.version,
                protocol_version=identity.protocol_version,
                discovered_tools=discovered,
                invoked_tools=tuple(sorted(invoked)),
                calls_attempted=len(results),
                calls_succeeded=succeeded,
                retrieval_complete=bool(
                    not absent and not groups_truncated and not errors
                ),
                errors=tuple(sorted(errors)),
            ),
            tuple(sorted(limitations)),
        )


def load_knowledge(
    source: Path,
    *,
    verify_manifest: bool = True,
) -> LoadedKnowledge:
    """Load a Phase 7 package and verify its manifest digest when present."""

    expanded = source.expanduser().resolve()
    path = expanded / "portal_knowledge.json" if expanded.is_dir() else expanded
    if not path.is_file():
        raise GroundingBuildError(f"portal knowledge file does not exist: {path}")
    try:
        data = path.read_bytes()
        knowledge = PortalKnowledge.model_validate_json(data)
    except (OSError, ValueError) as exc:
        raise GroundingBuildError(f"failed to load portal knowledge: {exc}") from exc
    digest = hashlib.sha256(data).hexdigest()

    manifest_path = path.parent / "manifest.json"
    if verify_manifest and manifest_path.is_file():
        try:
            manifest = KnowledgeExportManifest.model_validate_json(
                manifest_path.read_bytes()
            )
        except (OSError, ValueError) as exc:
            raise GroundingBuildError(f"invalid knowledge manifest: {exc}") from exc
        expected = next(
            (item.sha256 for item in manifest.files if item.name == path.name),
            None,
        )
        if expected is None or expected != digest:
            raise GroundingBuildError(
                "portal knowledge digest does not match its export manifest"
            )
        if manifest.source_run_id != knowledge.source_run_id:
            raise GroundingBuildError(
                "portal knowledge run ID does not match its export manifest"
            )
    return LoadedKnowledge(path=path, sha256=digest, knowledge=knowledge)


def _portal_context(case: TestCaseKnowledge) -> PortalTestCaseContext:
    return PortalTestCaseContext(
        test_case_id=case.test_case_id,
        fields=case.fields,
        dependency_case_ids=case.dependency_case_ids,
        description=case.description,
        evidence_state_ids=tuple(
            sorted({pointer.state_id for pointer in case.evidence})
        ),
        description_state_ids=tuple(
            sorted({pointer.state_id for pointer in case.description_evidence})
        ),
    )


def _case_query(case: TestCaseKnowledge, maximum_characters: int) -> str:
    values = [case.test_case_id]
    values.extend(field.value for field in case.fields if field.value)
    if case.description:
        values.append(case.description)
    values.extend(case.dependency_case_ids)
    return " ".join(dict.fromkeys(values))[:maximum_characters]


def _mcp_group_key(case: TestCaseKnowledge) -> tuple[str, ...]:
    values = tuple(
        dict.fromkeys(
            field.value
            for field in case.fields
            if field.key in _MCP_GROUP_KEYS and field.value
        )
    )
    if values:
        return values
    fallback = tuple(
        dict.fromkeys(
            field.value
            for field in case.fields
            if field.key not in _LOW_SIGNAL_KEYS and field.value
        )
    )
    return fallback[:2] or (case.test_case_id,)


def _mcp_query(
    group_key: tuple[str, ...],
    cases: list[TestCaseKnowledge],
    maximum_characters: int,
) -> str:
    terms: Counter[str] = Counter()
    for case in cases:
        if case.description:
            terms.update(_query_tokens(case.description))
    supporting_terms = [
        term
        for term, _ in sorted(terms.items(), key=lambda item: (-item[1], item[0]))
        if term not in {value.casefold() for value in group_key}
    ][:20]
    query = " ".join(
        (
            *group_key,
            *supporting_terms,
            "request response payload schema validation error codes integration",
        )
    )
    return query[:maximum_characters]


def _query_tokens(value: str) -> tuple[str, ...]:
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return tuple(
        token
        for token in re.findall(r"[A-Za-z0-9]+", expanded.casefold())
        if len(token) >= 3 and token not in _QUERY_STOPWORDS
    )


def _mcp_arguments(tool_name: str, query: str, limit: int) -> dict[str, object]:
    if tool_name == "search_docs":
        return {"query": query, "limit": limit}
    if tool_name == "search_documents":
        return {"query": query, "top_k": limit}
    raise GroundingBuildError(f"unsupported automatic MCP search tool: {tool_name}")


__all__ = [
    "GroundingBuildConfig",
    "GroundingBuildError",
    "GroundingBuilder",
    "LoadedKnowledge",
    "load_knowledge",
]
