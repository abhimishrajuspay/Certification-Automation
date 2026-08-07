"""Grounding tests use synthetic knowledge, repositories, and MCP transport."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from grounding.builder import (
    GroundingBuildError,
    GroundingBuilder,
    load_knowledge,
)
from grounding.cli import main as grounding_main
from grounding.exporter import GroundingExportError, export_grounding
from grounding.mcp import (
    MCPClient,
    MCPClientConfig,
    MCPProtocolError,
    MCPRetryableTransportError,
)
from grounding.models import GroundingSnippet, GroundingSourceKind
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
    ) -> dict[str, object]:
        assert endpoint == "https://mcp.example.test/mcp"
        assert timeout_seconds > 0
        assert maximum_response_bytes > 0
        self.requests.append(payload)
        request_id = payload["id"]
        method = payload["method"]
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


@pytest.mark.asyncio
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
        "--output-root",
        str(output_root),
    ]

    assert grounding_main(arguments) == 2
    assert (output_root / "fixture-run" / "grounding.json").is_file()
    assert (
        grounding_main([*arguments, "--allow-incomplete-grounding", "--overwrite"]) == 0
    )
