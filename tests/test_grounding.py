"""Grounding tests use synthetic knowledge, repositories, and MCP transport."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Type

import pytest
from pydantic import BaseModel, ValidationError

from grounding.agent import (
    AgenticGroundingBuilder,
    AgenticGroundingConfig,
    _safe_validation_error,
)
from grounding.builder import (
    GroundingBuildError,
    GroundingBuilder,
    load_knowledge,
)
from grounding.cli import _configure_progress_logging, main as grounding_main
from grounding.exporter import GroundingExportError, export_grounding
from grounding.mcp import (
    AGENT_READ_ONLY_TOOLS,
    MCP_SESSION_SENTINEL,
    MCPClient,
    MCPClientConfig,
    MCPProtocolError,
    MCPRetryableTransportError,
    MCPServerIdentity,
    MCPToolResult,
)
from grounding.models import (
    GroundingAgentAction,
    GroundingAgentDecision,
    GroundingAgentSelection,
    GroundingAgentToolKind,
    GroundingAgentToolRequest,
    GroundingSnippet,
    GroundingSourceKind,
    GroundingStrategy,
)
from grounding.repository import RepositoryIndex
from knowledge.exporter import export_knowledge
from knowledge.models import (
    EvidencePointer,
    KnowledgeCoverage,
    KnowledgeField,
    PortalKnowledge,
    PortalRoute,
    TestCaseKnowledge as PortalTestCase,
)
from scraper.models import ScrapeRunStatus
from synthesis.client import LiteLLMCompletion, LiteLLMConfig
from synthesis.models import TokenUsage


NOW = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)


class _FakeMCPTransport:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    async def post_json(
        self,
        endpoint: str,
        payload: dict[str, object],
        *,
        timeout_seconds: float,
        maximum_response_bytes: int,
        session_id: Optional[str] = None,
    ) -> dict[str, object]:
        assert endpoint == "https://mcp.example.test/mcp"
        assert timeout_seconds > 0
        assert maximum_response_bytes > 0
        self.requests.append(payload)
        method = payload["method"]
        if method == "notifications/initialized":
            return {}
        request_id = payload["id"]
        if method == "initialize":
            result: dict[str, object] = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture-mcp", "version": "1.0.0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "search_docs",
                        "description": "search docs",
                        "inputSchema": {"type": "object"},
                    },
                    {
                        "name": "search_documents",
                        "description": "hybrid search",
                        "inputSchema": {"type": "object"},
                    },
                    {
                        "name": "generate_payload",
                        "description": "not allowlisted",
                        "inputSchema": {"type": "object"},
                    },
                ]
            }
        elif method == "tools/call":
            params = payload["params"]
            assert isinstance(params, dict)
            tool = params["name"]
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "# Results\n\n"
                            "### Result 1\n"
                            "**Source:** payment_docs / bill-fetch.md\n"
                            "**Document ID:** `bill_fetch_spec`\n"
                            "**Chunk:** `7`\n\n"
                            f"{tool} says BillFetchRequest requires a request ID and "
                            "valid response-code handling."
                        ),
                    }
                ]
            }
        else:
            raise AssertionError(f"unexpected method: {method}")
        return {"jsonrpc": "2.0", "result": result, "id": request_id}


class _SessionMCPTransport(_FakeMCPTransport):
    """Session-bound server: hands out an id on initialize and requires it."""

    def __init__(self) -> None:
        super().__init__()
        self.observed_session_ids: list[Optional[str]] = []

    async def post_json(
        self,
        endpoint: str,
        payload: dict[str, object],
        *,
        timeout_seconds: float,
        maximum_response_bytes: int,
        session_id: Optional[str] = None,
    ) -> dict[str, object]:
        if payload["method"] != "initialize":
            assert session_id == "session-1"
        self.observed_session_ids.append(session_id)
        response = await super().post_json(
            endpoint,
            payload,
            timeout_seconds=timeout_seconds,
            maximum_response_bytes=maximum_response_bytes,
            session_id=session_id,
        )
        if payload["method"] == "initialize":
            response = {**response, MCP_SESSION_SENTINEL: "session-1"}
        return response


class _FlakyMCPTransport(_FakeMCPTransport):
    def __init__(self) -> None:
        super().__init__()
        self.failures_remaining = 1

    async def post_json(
        self,
        endpoint: str,
        payload: dict[str, object],
        *,
        timeout_seconds: float,
        maximum_response_bytes: int,
        session_id: Optional[str] = None,
    ) -> dict[str, object]:
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise MCPRetryableTransportError("temporary timeout")
        return await super().post_json(
            endpoint,
            payload,
            timeout_seconds=timeout_seconds,
            maximum_response_bytes=maximum_response_bytes,
        )


class _AgentMCPTransport(_FakeMCPTransport):
    async def post_json(
        self,
        endpoint: str,
        payload: dict[str, object],
        *,
        timeout_seconds: float,
        maximum_response_bytes: int,
        session_id: Optional[str] = None,
    ) -> dict[str, object]:
        if payload["method"] != "tools/list":
            return await super().post_json(
                endpoint,
                payload,
                timeout_seconds=timeout_seconds,
                maximum_response_bytes=maximum_response_bytes,
            )
        self.requests.append(payload)
        return {
            "jsonrpc": "2.0",
            "id": payload["id"],
            "result": {
                "tools": [
                    {
                        "name": "get_api_spec",
                        "description": "retrieve one API specification",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"api_name": {"type": "string"}},
                            "required": ["api_name"],
                            "additionalProperties": False,
                        },
                        "annotations": {"readOnlyHint": True},
                    },
                    {
                        "name": "generate_payload",
                        "description": "mutating generator",
                        "inputSchema": {"type": "object"},
                        "annotations": {"readOnlyHint": False},
                    },
                ]
            },
        }


class _AgentGroundingLLM:
    def __init__(self, *, use_mcp: bool = False, unseen_first: bool = False) -> None:
        self._config = LiteLLMConfig(
            endpoint="https://llm.example.test",
            model="fixture-grounding-model",
            response_format="json_schema",
        )
        self.use_mcp = use_mcp
        self.unseen_first = unseen_first
        self.prompts: list[str] = []

    @property
    def config(self) -> LiteLLMConfig:
        return self._config

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        schema_name: str,
        maximum_output_tokens: int | None = None,
    ) -> LiteLLMCompletion:
        del system_prompt, schema_name, maximum_output_tokens
        self.prompts.append(user_prompt)
        context_text = user_prompt.split("INPUT_CONTEXT=", 1)[1]
        context = json.JSONDecoder().raw_decode(context_text)[0]
        case_ids = tuple(context["control"]["target_test_case_ids"])
        candidates = context["candidate_evidence"]["by_test_case"]
        if not any(candidates.values()):
            request = (
                GroundingAgentToolRequest(
                    request_id="mcp-spec",
                    kind=GroundingAgentToolKind.CALL_MCP,
                    test_case_ids=case_ids,
                    tool_name="get_api_spec",
                    arguments={"api_name": "BillFetchRequest"},
                )
                if self.use_mcp
                else GroundingAgentToolRequest(
                    request_id="repo-search",
                    kind=GroundingAgentToolKind.SEARCH_REPOSITORY,
                    test_case_ids=case_ids,
                    query="BillFetchRequest request response schema",
                )
            )
            decision = GroundingAgentDecision(
                action=GroundingAgentAction.SEARCH,
                rationale="Retrieve the API contract shared by these cases.",
                requests=(request,),
            )
        else:
            snippet_ids = {case_id: tuple(candidates[case_id]) for case_id in case_ids}
            if self.unseen_first and len(self.prompts) == 2:
                snippet_ids[case_ids[0]] = ("f" * 64,)
            decision = GroundingAgentDecision(
                action=GroundingAgentAction.FINAL,
                rationale="Select only the evidence returned by bounded tools.",
                selections=tuple(
                    GroundingAgentSelection(
                        test_case_id=case_id,
                        snippet_ids=snippet_ids[case_id],
                        limitations=(
                            ()
                            if snippet_ids[case_id]
                            else ("No applicable evidence was returned",)
                        ),
                    )
                    for case_id in case_ids
                ),
            )
        assert isinstance(decision, response_model)
        content = decision.model_dump_json()
        return LiteLLMCompletion(
            content=content,
            request_sha256=hashlib.sha256(user_prompt.encode()).hexdigest(),
            response_sha256=hashlib.sha256(content.encode()).hexdigest(),
            usage=TokenUsage(prompt_tokens=25, completion_tokens=15, total_tokens=40),
        )


def _pointer(state_id: str) -> EvidencePointer:
    return EvidencePointer(
        state_id=state_id,
        state_sequence=int(state_id[-1]),
        url="https://portal.example.test/cases",
        frame_id="main",
        element_ids=(f"element-{state_id}",),
        artifact_ids=(hashlib.sha256(state_id.encode()).hexdigest(),),
    )


def _knowledge(*, complete: bool = True) -> PortalKnowledge:
    cases: list[PortalTestCase] = []
    for index in (1, 2):
        test_case_id = f"TC_0{index}"
        description = (
            f"{test_case_id}: validates BillFetchRequest response handling"
            if complete or index == 1
            else None
        )
        description_evidence = (_pointer(f"state-{index}"),) if description else ()
        cases.append(
            PortalTestCase(
                test_case_id=test_case_id,
                fields=(
                    KnowledgeField(
                        key="test_case_id", label="TC ID", value=test_case_id
                    ),
                    KnowledgeField(
                        key="api_name",
                        label="API Name",
                        value="BillFetchRequest",
                    ),
                    KnowledgeField(
                        key="test_data",
                        label="Test Data",
                        value=f"bill fetch scenario {index}",
                    ),
                ),
                dependency_case_ids=("TC_01",) if index == 2 else (),
                description=description,
                evidence=(_pointer("state-0"),),
                description_evidence=description_evidence,
            )
        )
    missing = () if complete else ("TC_02",)
    return PortalKnowledge(
        source_run_id="fixture-run",
        source_root_url="https://portal.example.test/start",
        source_started_at=NOW,
        source_ended_at=NOW,
        normalized_at=NOW,
        routes=(
            PortalRoute(
                url="https://portal.example.test/cases",
                titles=("Cases",),
                state_ids=("state-0",),
            ),
        ),
        tables=(),
        test_cases=tuple(cases),
        network=(),
        coverage=KnowledgeCoverage(
            source_status=ScrapeRunStatus.COMPLETED,
            source_bounded_complete=True,
            testcase_context_complete=complete,
            states_examined=3,
            routes_normalized=1,
            tables_normalized=0,
            declared_test_cases=2,
            test_cases_normalized=2,
            descriptions_captured=2 if complete else 1,
            missing_description_ids=missing,
        ),
    )


def _repository(tmp_path: Path) -> RepositoryIndex:
    root = tmp_path / "merchant-repo"
    (root / "schemas").mkdir(parents=True)
    (root / "examples").mkdir()
    (root / "dummy_portal").mkdir()
    (root / "schemas" / "bill_fetch.xsd").write_text(
        "<schema>\n"
        '  <element name="BillFetchRequest"/>\n'
        "  <password>do-not-retain</password>\n"
        "</schema>\n"
    )
    (root / "examples" / "bill_fetch.xml").write_text(
        "<BillFetchRequest><requestId>sample</requestId></BillFetchRequest>\n"
    )
    (root / "ignored.bin").write_bytes(b"BillFetchRequest")
    (root / "dummy_portal" / "ignored.xsd").write_text("BillFetchRequest hidden")
    return RepositoryIndex.build(root)


def test_repository_index_is_bounded_cited_and_secret_safe(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    matches = repository.search("BillFetchRequest request schema", limit=5)

    assert repository.summary.files_indexed == 2
    assert repository.summary.files_skipped == 1
    assert matches
    assert all(match.source_kind == GroundingSourceKind.REPOSITORY for match in matches)
    schema = next(match for match in matches if "bill_fetch.xsd" in match.title)
    assert schema.repository is not None
    assert schema.repository.path == "schemas/bill_fetch.xsd"
    assert schema.repository.line_start >= 1
    assert schema.repository.redacted is True
    assert "do-not-retain" not in schema.content
    assert "[REDACTED]" in schema.content

    tampered = schema.model_dump(mode="json")
    tampered["content"] += " changed"
    with pytest.raises(ValueError, match="digest does not match"):
        GroundingSnippet.model_validate(tampered)


def test_repository_index_excludes_generated_build_trees_and_requires_api_anchor(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "newton"
    generated = repository_root / "dist-newstyle" / "generated"
    generated.mkdir(parents=True)
    (generated / "aws.json").write_text(
        '{"description":"session registration request response payload"}'
    )
    docs = repository_root / "docs"
    docs.mkdir()
    (docs / "unrelated.md").write_text(
        "This discusses session registration request response handling."
    )
    (docs / "sdk.md").write_text(
        "SdkSessionReg calls SessionRegistration at /merchantSDK/getSessionToken."
    )

    repository = RepositoryIndex.build(repository_root)
    matches = repository.search(
        "SdkSessionReg SessionRegistration request response payload",
        limit=5,
        anchor_terms=("SdkSessionReg", "SessionRegistration"),
    )

    assert repository.summary.files_indexed == 2
    assert matches
    assert [
        item.repository.path for item in matches if item.repository is not None
    ] == ["docs/sdk.md"]


@pytest.mark.asyncio
async def test_mcp_client_echoes_server_session_id() -> None:
    transport = _SessionMCPTransport()
    client = MCPClient(
        MCPClientConfig(endpoint="https://mcp.example.test/mcp"),
        transport=transport,
    )
    identity, tools = await client.connect()
    assert identity.name == "fixture-mcp"
    assert {"search_docs", "search_documents"}.issubset({t.name for t in tools})
    result = await client.call_tool("search_docs", {"query": "x", "limit": 1})
    assert result.text_blocks
    methods = [payload["method"] for payload in transport.requests]
    assert methods == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
    ]
    # The session id adopted during initialize is echoed on every later call.
    assert transport.observed_session_ids[0] is None
    assert all(
        observed == "session-1" for observed in transport.observed_session_ids[1:]
    )


def test_mcp_to_snippets_splits_real_keyword_output_and_round_trips() -> None:
    transport = _FakeMCPTransport()
    client = MCPClient(
        MCPClientConfig(endpoint="https://mcp.example.test/mcp"),
        transport=transport,
    )
    client.identity = MCPServerIdentity(
        "2025-06-18", "merchant-integration-mcp", "3.4.7"
    )
    text = (
        '# Keyword Search Results for: "ReqValAdd Validate VPA payload"\n'
        "\n### Result 1\n"
        "**Source:** s2s_api_docs / post-api-apiversion-merchants-vpas-validity.md\n"
        "**Document ID:** `s2s_api_docs__post_api_apiversion_merchants_vpas_validity`\n"
        "**Chunk:** `0`\n"
        "term matches: 3\n"
        "\n```\n# VPA Validity API Integration Guide\n"
        "Newton calls NPCI `ReqValAdd` for off-us VPAs.\n```\n"
        "\n### Result 2\n"
        "**Source:** s2s_api_docs / other.md\n"
        "**Document ID:** `other_doc`\n"
        "**Chunk:** `15`\n"
        "term matches: 1\n"
        "\n```\nother body\n```\n"
        "\n### Result 3\n"
        "**Source:** guides / faq.md\n"
        "\n```\nno doc refs here\n```\n"
    )
    result = MCPToolResult(
        tool_name="search_docs",
        query="q",
        arguments_sha256="a" * 64,
        response_sha256="b" * 64,
        retrieved_at=datetime.now(timezone.utc),
        text_blocks=(text,),
    )
    snippets = client.to_snippets(result)
    assert len(snippets) == 3
    assert snippets[0].mcp and snippets[0].mcp.documents[0].document_id == (
        "s2s_api_docs__post_api_apiversion_merchants_vpas_validity"
    )
    assert snippets[0].mcp.documents[0].chunk_id == "0"
    assert snippets[1].mcp and snippets[1].mcp.documents[0].chunk_id == "15"
    assert snippets[2].mcp and snippets[2].mcp.documents[0].document_id is None
    # The digest validator must accept a full dump/reload round-trip.
    from grounding.models import GroundingSnippet

    for snippet in snippets:
        reloaded = GroundingSnippet.model_validate(
            json.loads(snippet.model_dump_json())
        )
        assert reloaded.snippet_id == snippet.snippet_id


async def test_mcp_client_discovers_allowlists_and_cites_results() -> None:
    transport = _FakeMCPTransport()
    client = MCPClient(
        MCPClientConfig(endpoint="https://mcp.example.test/mcp"),
        transport=transport,
    )

    identity, tools = await client.connect()
    result = await client.call_tool(
        "search_docs",
        {"query": "BillFetchRequest", "limit": 2},
    )
    snippets = client.to_snippets(result)

    assert identity.name == "fixture-mcp"
    assert "generate_payload" in {tool.name for tool in tools}
    assert len(snippets) == 1
    assert snippets[0].mcp is not None
    assert snippets[0].mcp.documents[0].document_id == "bill_fetch_spec"
    assert snippets[0].mcp.documents[0].chunk_id == "7"
    with pytest.raises(MCPProtocolError, match="not allowlisted"):
        await client.call_tool("generate_payload", {"endpoint_id": "bill-fetch"})


@pytest.mark.asyncio
async def test_mcp_client_retries_transient_transport_failures() -> None:
    transport = _FlakyMCPTransport()
    client = MCPClient(
        MCPClientConfig(
            endpoint="https://mcp.example.test/mcp",
            maximum_attempts=2,
            retry_backoff_seconds=0,
        ),
        transport=transport,
    )

    identity, _ = await client.connect()

    assert identity.name == "fixture-mcp"
    assert transport.failures_remaining == 0


@pytest.mark.asyncio
async def test_grounding_groups_mcp_queries_and_exports_llm_ready_jsonl(
    tmp_path: Path,
) -> None:
    transport = _FakeMCPTransport()
    client = MCPClient(
        MCPClientConfig(endpoint="https://mcp.example.test/mcp"),
        transport=transport,
    )
    package = await GroundingBuilder(
        _knowledge(),
        "a" * 64,
        _repository(tmp_path),
        mcp_client=client,
    ).build()

    calls = [
        request for request in transport.requests if request["method"] == "tools/call"
    ]
    assert len(calls) == 2
    assert package.coverage.grounding_complete is True
    assert package.coverage.repository_grounded == 2
    assert package.coverage.mcp_grounded == 2
    assert package.mcp.calls_attempted == 2
    assert package.mcp.calls_succeeded == 2
    assert (
        package.test_cases[0].mcp_snippet_ids == package.test_cases[1].mcp_snippet_ids
    )

    output = export_grounding(package, tmp_path / "grounded")
    assert output.testcases_path.read_text().count("\n") == 2
    for item in output.manifest.files:
        data = (output.output_directory / item.name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item.sha256
    with pytest.raises(GroundingExportError, match="already exists"):
        export_grounding(package, tmp_path / "grounded")


@pytest.mark.asyncio
async def test_agentic_grounding_lets_model_select_repository_evidence(
    tmp_path: Path,
) -> None:
    llm = _AgentGroundingLLM()
    package = await AgenticGroundingBuilder(
        _knowledge(),
        "c" * 64,
        _repository(tmp_path),
        llm,
        config=AgenticGroundingConfig(
            cases_per_batch=2,
            concurrency=1,
            maximum_turns_per_batch=3,
            maximum_snippet_characters=2_000,
        ),
    ).build()

    assert package.strategy == GroundingStrategy.AGENTIC
    assert package.coverage.grounding_complete is True
    assert package.coverage.repository_grounded == 2
    assert package.coverage.mcp_grounded == 0
    assert len(package.agent_calls) == 2
    assert len(package.agent_observations) == 1
    assert package.provider is not None
    assert package.provider.prompt_tokens == 50
    assert all(case.repository_snippet_ids for case in package.test_cases)
    assert len(llm.prompts[1]) < 40_000
    second_context = json.JSONDecoder().raw_decode(
        llm.prompts[1].split("INPUT_CONTEXT=", 1)[1]
    )[0]
    candidate_catalog = second_context["candidate_evidence"]["snippets"]
    assert len(candidate_catalog) == len(package.snippets)
    assert len({item["snippet_id"] for item in candidate_catalog}) == len(
        candidate_catalog
    )
    persisted = package.model_dump_json()
    assert "Retrieve the API contract shared by these cases" not in persisted
    assert "Select only the evidence returned by bounded tools" not in persisted


@pytest.mark.asyncio
async def test_agentic_grounding_discovers_and_uses_safe_mcp_tool(
    tmp_path: Path,
) -> None:
    transport = _AgentMCPTransport()
    mcp = MCPClient(
        MCPClientConfig(
            endpoint="https://mcp.example.test/mcp",
            allowed_tools=AGENT_READ_ONLY_TOOLS,
        ),
        transport=transport,
    )
    package = await AgenticGroundingBuilder(
        _knowledge(),
        "d" * 64,
        _repository(tmp_path),
        _AgentGroundingLLM(use_mcp=True),
        mcp_client=mcp,
        config=AgenticGroundingConfig(cases_per_batch=2, concurrency=1),
    ).build()

    assert package.coverage.grounding_complete is True
    assert package.coverage.mcp_grounded == 2
    assert package.mcp.invoked_tools == ("get_api_spec",)
    assert "generate_payload" not in package.mcp.discovered_tools
    assert package.agent_observations[0].tool_name == "get_api_spec"
    assert all(case.mcp_snippet_ids for case in package.test_cases)


@pytest.mark.asyncio
async def test_agentic_grounding_rejects_unseen_citation_then_self_corrects(
    tmp_path: Path,
) -> None:
    package = await AgenticGroundingBuilder(
        _knowledge(),
        "e" * 64,
        _repository(tmp_path),
        _AgentGroundingLLM(unseen_first=True),
        config=AgenticGroundingConfig(
            cases_per_batch=2,
            concurrency=1,
            maximum_turns_per_batch=3,
        ),
    ).build()

    assert package.coverage.grounding_complete is True
    assert len(package.agent_calls) == 3
    assert package.agent_calls[1].valid is False
    assert "unseen snippets" in (package.agent_calls[1].validation_error or "")


@pytest.mark.asyncio
async def test_incomplete_source_context_is_rejected_by_default(tmp_path: Path) -> None:
    builder = GroundingBuilder(
        _knowledge(complete=False),
        "b" * 64,
        _repository(tmp_path),
    )

    with pytest.raises(GroundingBuildError, match="complete normalized"):
        await builder.build()


def test_knowledge_loader_verifies_phase7_manifest_hash(tmp_path: Path) -> None:
    package = export_knowledge(_knowledge(), tmp_path / "knowledge")

    loaded = load_knowledge(package.output_directory)

    assert loaded.knowledge.source_run_id == "fixture-run"
    package.knowledge_path.write_bytes(package.knowledge_path.read_bytes() + b"\n")
    with pytest.raises(GroundingBuildError, match="does not match"):
        load_knowledge(package.output_directory)


def test_cli_exports_incomplete_diagnostics_with_distinct_exit_status(
    tmp_path: Path,
) -> None:
    knowledge = export_knowledge(_knowledge(), tmp_path / "knowledge")
    repository = tmp_path / "empty-repository"
    repository.mkdir()
    output_root = tmp_path / "grounding-output"
    arguments = [
        "--knowledge",
        str(knowledge.output_directory),
        "--repo-path",
        str(repository),
        "--no-mcp",
        "--strategy",
        "deterministic",
        "--output-root",
        str(output_root),
    ]

    assert grounding_main(arguments) == 2
    assert (output_root / "fixture-run" / "grounding.json").is_file()
    assert (
        grounding_main([*arguments, "--allow-incomplete-grounding", "--overwrite"]) == 0
    )


def test_grounding_progress_log_includes_shared_litellm_transport(
    tmp_path: Path,
) -> None:
    path = tmp_path / "progress.log"
    logger_names = ("cz.grounding", "cz.synthesis.transport")
    try:
        _configure_progress_logging(path, quiet=True)

        logging.getLogger("cz.grounding.agent").info("agent batch started")
        logging.getLogger("cz.synthesis.transport").info("model request waiting")
        for logger_name in logger_names:
            for handler in logging.getLogger(logger_name).handlers:
                handler.flush()

        content = path.read_text()
        assert "agent batch started" in content
        assert "model request waiting" in content
    finally:
        handlers = {
            handler
            for logger_name in logger_names
            for handler in logging.getLogger(logger_name).handlers
        }
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            logger.propagate = True
            logger.setLevel(logging.NOTSET)
        for handler in handlers:
            handler.close()


def test_agentic_validation_errors_do_not_persist_model_input() -> None:
    secret_marker = "raw-model-content-must-not-persist"
    with pytest.raises(ValidationError) as captured:
        GroundingAgentDecision.model_validate(
            {
                "action": "search",
                "rationale": "search",
                "requests": [
                    {
                        "request_id": "search-1",
                        "kind": "search_repository",
                        "test_case_ids": ["TC_01"],
                        "query": "BillFetchRequest",
                        "unexpected": secret_marker,
                    }
                ],
            }
        )

    sanitized = _safe_validation_error(captured.value)
    assert secret_marker not in sanitized
    assert "unexpected" in sanitized


def test_repository_suffix_resolution_defaults_merges_and_validates() -> None:
    from grounding.repository import (
        CODE_SUFFIXES,
        DEFAULT_SUFFIXES,
        resolve_repository_suffixes,
    )

    assert set(resolve_repository_suffixes()) == set(DEFAULT_SUFFIXES)
    assert resolve_repository_suffixes() == tuple(sorted(DEFAULT_SUFFIXES))
    assert resolve_repository_suffixes(include_code=True) == tuple(CODE_SUFFIXES)
    assert tuple(CODE_SUFFIXES) == tuple(sorted(CODE_SUFFIXES))
    assert {".hs", ".lhs", ".py", ".java", ".sh"} <= set(CODE_SUFFIXES)

    merged = resolve_repository_suffixes([".hs", " .LHS "])
    assert ".hs" in merged and ".lhs" in merged
    assert len(merged) == len(set(merged))
    assert merged == tuple(sorted(merged))

    for invalid in ("hs", "", ".hs hs", "hs.", ".hs,.lhs"):
        with pytest.raises(ValueError, match="dotted extension"):
            resolve_repository_suffixes([invalid])


def test_repository_include_code_indexes_haskell_sources(tmp_path: Path) -> None:
    from grounding.repository import RepositoryIndexConfig, resolve_repository_suffixes

    repository_root = tmp_path / "newton-hs"
    handlers = repository_root / "handlers"
    handlers.mkdir(parents=True)
    (handlers / "ValAdd.hs").write_text(
        "module ValAdd (handler) where\n"
        "-- reqValAdd validates a payee VPA for P2P transfers\n"
        "validatePayeeVpa :: Vpa -> IO (Either ApiError RespValAdd)\n"
    )
    (repository_root / "changelog.md").write_text("unrelated portal notes\n")

    default_index = RepositoryIndex.build(repository_root)
    assert default_index.summary.files_indexed == 1
    assert default_index.summary.files_skipped == 1
    assert not default_index.search(
        "reqValAdd validate payee VPA P2P", limit=5, anchor_terms=("reqValAdd",)
    )

    code_index = RepositoryIndex.build(
        repository_root,
        RepositoryIndexConfig(
            suffixes=resolve_repository_suffixes([".lhs"], include_code=True)
        ),
    )
    assert code_index.summary.files_indexed == 2
    matches = code_index.search(
        "reqValAdd validate payee VPA P2P", limit=5, anchor_terms=("reqValAdd",)
    )
    assert [item.repository.path for item in matches if item.repository] == [
        "handlers/ValAdd.hs"
    ]


def test_grounding_and_synthesis_cli_expose_repository_suffix_flags() -> None:
    from grounding.cli import build_parser as grounding_parser
    from synthesis.cli import build_parser as synthesis_parser

    grounding_args = grounding_parser().parse_args(
        [
            "--knowledge",
            "knowledge.json",
            "--repo-path",
            "repo",
            "--plan-only",
            "--repo-include-code",
            "--repo-suffix",
            ".hs",
        ]
    )
    assert grounding_args.repo_include_code is True
    assert grounding_args.repo_suffix == [".hs"]

    synthesis_args = synthesis_parser().parse_args(
        [
            "--grounding",
            "package",
            "--strategy",
            "bulk",
            "--plan-only",
            "--repo-path",
            "repo",
            "--repo-suffix",
            ".lhs",
        ]
    )
    assert synthesis_args.repo_include_code is False
    assert synthesis_args.repo_suffix == [".lhs"]


def test_repository_search_diversifies_results_across_directories(
    tmp_path: Path,
) -> None:
    from grounding.repository import RepositoryIndexConfig

    repository_root = tmp_path / "newton-hs"
    flow = repository_root / "src" / "product" / "reqvaladd"
    flow.mkdir(parents=True)
    payload = "reqValAdd validates a payee VPA for P2P transfers\n"
    for name in ("Controller.hs", "Customer.hs", "Helper.hs", "DynamicVpa.hs"):
        (flow / name).write_text(f"module {name} where\n{payload * 4}")
    routes = repository_root / "src" / "app" / "routes"
    routes.mkdir(parents=True)
    (routes / "Npci.hs").write_text(
        'module Npci where\n-- route "reqvaladd" XML POST reqValAdd\n'
    )
    api_types = repository_root / "src" / "npci"
    api_types.mkdir(parents=True)
    (api_types / "ApiTypes.hs").write_text(
        "module ApiTypes where\n-- reqValAdd ReqBody XML Post AckContract\n"
    )
    index = RepositoryIndex.build(
        repository_root,
        RepositoryIndexConfig(suffixes=(".hs",)),
    )

    matches = index.search("reqValAdd route", limit=5, anchor_terms=("reqValAdd",))
    directories = {
        item.repository.path.rsplit("/", 1)[0] for item in matches if item.repository
    }
    assert "src/product/reqvaladd" in directories
    assert "src/app/routes" in directories
    assert "src/npci" in directories


def test_repository_read_around_match_centers_on_first_hit(tmp_path: Path) -> None:
    from grounding.repository import RepositoryIndexConfig

    repository_root = tmp_path / "newton-hs"
    routes = repository_root / "routes"
    routes.mkdir(parents=True)
    lines = ["module Routes where"]
    lines += [f"-- filler {index}" for index in range(2, 55)]
    lines.append('"ReqValAdd" :> ReqBody XML Post Ack')
    lines += [f"-- tail {index}" for index in range(2, 40)]
    (routes / "Npc.hs").write_text("\n".join(lines))
    elsewhere = repository_root / "types"
    elsewhere.mkdir()
    (elsewhere / "Common.hs").write_text(
        'module Common where\n-- "ReqValAdd" :> also referenced here\n'
    )
    index = RepositoryIndex.build(
        repository_root,
        RepositoryIndexConfig(suffixes=(".hs",)),
    )

    snippet = index.read_around_match("routes/Npc.hs", '"ReqValAdd" :>')
    assert snippet.repository is not None
    assert '"ReqValAdd" :> ReqBody XML Post Ack' in snippet.content
    assert snippet.repository.line_start <= 55 - 4
    assert snippet.repository.line_end >= 56 >= snippet.repository.line_start

    with pytest.raises(Exception, match=r"not found.*types/Common\.hs"):
        index.read_around_match("routes/Npc.hs", '"ReqValAdd" :> also')
    with pytest.raises(Exception, match="not found"):
        index.read_around_match("routes/Npc.hs", "missing-anchor-xyz")
    with pytest.raises(Exception, match="anchor"):
        index.read_around_match("routes/Npc.hs", "  ")
