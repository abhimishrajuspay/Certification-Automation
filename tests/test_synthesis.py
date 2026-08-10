from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Type

import pytest
from pydantic import BaseModel, SecretStr, ValidationError

from grounding.exporter import export_grounding
from grounding.models import (
    GroundedTestCase,
    GroundingCoverage,
    GroundingPackage,
    GroundingSnippet,
    GroundingSourceKind,
    MCPServerSummary,
    PortalTestCaseContext,
    RepositoryCitation,
    RepositoryIndexSummary,
)
from grounding.repository import RepositoryIndex, RepositoryIndexConfig
from knowledge.models import KnowledgeField
from synthesis.builder import (
    SynthesisBuildConfig,
    SynthesisBuildError,
    SynthesisBuilder,
    load_grounding,
)
from synthesis.agentic import (
    AgenticSynthesisBuilder,
    AgenticSynthesisConfig,
    SynthesisEvidenceTools,
)
from synthesis.cli import _configure_progress_logging, main as synthesis_main
from synthesis.client import (
    LiteLLMClient,
    LiteLLMCompletion,
    LiteLLMConfig,
    LiteLLMRetryableError,
    LiteLLMTransportResponse,
    normalize_litellm_endpoint,
)
from synthesis.exporter import export_synthesis, load_checkpoint, save_checkpoint
from synthesis.models import (
    AssertionOperator,
    AssertionSource,
    Confidence,
    GeneratedValueKind,
    HTTPMethod,
    HTTPRequestSpec,
    NamedValue,
    RequestBodyMode,
    RequestBodySpec,
    ResponseAssertion,
    SynthesisBatchResponse,
    SynthesisAgentAction,
    SynthesisAgentResponse,
    SynthesisDisposition,
    SynthesisStrategy,
    SynthesisToolObservation,
    TemplateVariableBinding,
    TemplateVariableSource,
    TestCaseExecutionSpec as ExecutionSpec,
    TokenUsage,
)


def _hash_json(value: object) -> str:
    data = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(data).hexdigest()


def _grounding() -> GroundingPackage:
    content = (
        "POST /bill/fetch accepts JSON with requestId. "
        "A valid request returns HTTP 200."
    )
    citation = RepositoryCitation(
        repository_id="a" * 64,
        path="docs/bill-fetch.md",
        file_sha256="b" * 64,
        line_start=1,
        line_end=2,
        excerpt_sha256=hashlib.sha256(content.encode()).hexdigest(),
        redacted=False,
    )
    snippet_id = _hash_json(
        {
            "source": GroundingSourceKind.REPOSITORY.value,
            "citation": citation.model_dump(mode="json"),
            "content": content,
        }
    )
    snippet = GroundingSnippet(
        snippet_id=snippet_id,
        source_kind=GroundingSourceKind.REPOSITORY,
        title="docs/bill-fetch.md:1-2",
        content=content,
        relevance_score=10,
        repository=citation,
    )
    cases = []
    for index in (1, 2):
        case_id = f"TC_0{index}"
        cases.append(
            GroundedTestCase(
                context=PortalTestCaseContext(
                    test_case_id=case_id,
                    fields=(
                        KnowledgeField(
                            key="test_case_id", label="TC ID", value=case_id
                        ),
                        KnowledgeField(
                            key="api_name",
                            label="API Name",
                            value="BillFetchRequest",
                        ),
                    ),
                    dependency_case_ids=("TC_01",) if index == 2 else (),
                    description=f"{case_id} validates bill fetch",
                    evidence_state_ids=(f"state-{index}",),
                    description_state_ids=(f"detail-{index}",),
                ),
                retrieval_query=f"{case_id} BillFetchRequest",
                repository_snippet_ids=(snippet_id,),
            )
        )
    return GroundingPackage(
        source_run_id="fixture-run",
        source_knowledge_sha256="c" * 64,
        grounded_at="2026-08-07T00:00:00Z",
        repository=RepositoryIndexSummary(
            repository_id="a" * 64,
            root_name="fixture",
            files_examined=1,
            files_indexed=1,
            files_skipped=0,
            bytes_indexed=len(content),
            suffixes=(".md",),
        ),
        mcp=MCPServerSummary(configured=False, available=False),
        snippets=(snippet,),
        test_cases=tuple(cases),
        coverage=GroundingCoverage(
            source_testcase_context_complete=True,
            mcp_required=False,
            test_cases=2,
            repository_grounded=2,
            mcp_grounded=0,
            externally_grounded=2,
            grounding_complete=True,
        ),
    )


def _ready_spec(case: dict[str, object]) -> ExecutionSpec:
    snippet_ids = case["available_evidence_snippet_ids"]
    assert isinstance(snippet_ids, list)
    dependencies = case["dependency_case_ids"]
    assert isinstance(dependencies, list)
    return ExecutionSpec(
        test_case_id=str(case["test_case_id"]),
        title=f"Execute {case['test_case_id']}",
        dependency_case_ids=tuple(str(item) for item in dependencies),
        disposition=SynthesisDisposition.READY,
        request=HTTPRequestSpec(
            method=HTTPMethod.POST,
            path="/bill/fetch",
            headers=(NamedValue(name="x-api-key", value="{{VENDOR_KEY}}"),),
            body=RequestBodySpec(
                mode=RequestBodyMode.JSON,
                content_type="application/json",
                template='{"requestId":"{{REQUEST_ID}}"}',
            ),
        ),
        assertions=(
            ResponseAssertion(
                source=AssertionSource.HTTP_STATUS,
                operator=AssertionOperator.EQUALS,
                expected="200",
                description="transport succeeds",
            ),
        ),
        variable_bindings=(
            TemplateVariableBinding(
                name="VENDOR_KEY",
                source=TemplateVariableSource.ENVIRONMENT,
                sensitive=True,
                description="Vendor credential supplied at runtime",
            ),
            TemplateVariableBinding(
                name="REQUEST_ID",
                source=TemplateVariableSource.GENERATED,
                generator=GeneratedValueKind.UUID,
                description="Unique request correlation identifier",
            ),
        ),
        evidence_snippet_ids=tuple(str(item) for item in snippet_ids),
        confidence=Confidence.HIGH,
        rationale="The cited API document supplies method, path, body, and status.",
        human_review_required=False,
    )


class _FakeLLM:
    def __init__(self, *, invalid_first: bool = False) -> None:
        self._config = LiteLLMConfig(
            endpoint="https://llm.example.test",
            model="fixture-model",
            response_format="json_schema",
        )
        self.invalid_first = invalid_first
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
    ) -> LiteLLMCompletion:
        del system_prompt, response_model, schema_name
        self.prompts.append(user_prompt)
        context_text = user_prompt.split("INPUT_CONTEXT=", 1)[1]
        context, _ = json.JSONDecoder().raw_decode(context_text)
        specifications = [_ready_spec(case) for case in context["test_cases"]]
        if self.invalid_first and len(self.prompts) == 1:
            specifications[0] = specifications[0].model_copy(
                update={"test_case_id": "INVENTED_TC"}
            )
        content = SynthesisBatchResponse(
            specifications=tuple(specifications)
        ).model_dump_json()
        return LiteLLMCompletion(
            content=content,
            request_sha256=hashlib.sha256(user_prompt.encode()).hexdigest(),
            response_sha256=hashlib.sha256(content.encode()).hexdigest(),
            usage=TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )


class _FakeTransport:
    def __init__(self) -> None:
        self.api_key: SecretStr | None = None
        self.request: dict[str, object] | None = None

    async def post_json(
        self,
        endpoint: str,
        payload: bytes,
        *,
        api_key: SecretStr | None,
        timeout_seconds: float,
        maximum_response_bytes: int,
    ) -> LiteLLMTransportResponse:
        del endpoint, timeout_seconds, maximum_response_bytes
        self.api_key = api_key
        self.request = json.loads(payload)
        body = json.dumps(
            {
                "choices": [{"message": {"content": '{"specifications":[]}'}}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            }
        ).encode()
        return LiteLLMTransportResponse(
            payload=json.loads(body),
            response_sha256=hashlib.sha256(body).hexdigest(),
        )


class _AgentLLM:
    def __init__(self, responses: list[SynthesisAgentResponse]) -> None:
        self._config = LiteLLMConfig(
            endpoint="https://llm.example.test",
            model="fixture-agent-model",
            response_format="json_schema",
        )
        self.responses = responses
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
    ) -> LiteLLMCompletion:
        del system_prompt, response_model, schema_name
        self.prompts.append(user_prompt)
        response = self.responses[len(self.prompts) - 1]
        content = response.model_dump_json()
        return LiteLLMCompletion(
            content=content,
            request_sha256=hashlib.sha256(user_prompt.encode()).hexdigest(),
            response_sha256=hashlib.sha256(content.encode()).hexdigest(),
            usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        )


class _SlowTransport(_FakeTransport):
    async def post_json(
        self,
        endpoint: str,
        payload: bytes,
        *,
        api_key: SecretStr | None,
        timeout_seconds: float,
        maximum_response_bytes: int,
    ) -> LiteLLMTransportResponse:
        await asyncio.sleep(0.03)
        return await super().post_json(
            endpoint,
            payload,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            maximum_response_bytes=maximum_response_bytes,
        )


class _UnavailableLLM:
    def __init__(self, error: str) -> None:
        self._config = LiteLLMConfig(
            endpoint="https://llm.example.test",
            model="fixture-agent-model",
        )
        self.error = error
        self.calls = 0

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
    ) -> LiteLLMCompletion:
        del system_prompt, user_prompt, response_model, schema_name
        self.calls += 1
        raise LiteLLMRetryableError(self.error)


def test_request_models_reject_unsafe_or_invalid_ready_specs() -> None:
    with pytest.raises(ValidationError, match="environment placeholder"):
        HTTPRequestSpec(
            method=HTTPMethod.GET,
            path="/resource",
            headers=(NamedValue(name="Authorization", value="Bearer secret"),),
        )
    with pytest.raises(ValidationError, match="not valid JSON"):
        RequestBodySpec(mode=RequestBodyMode.JSON, template="{invalid")
    with pytest.raises(ValidationError, match="fixed origin"):
        HTTPRequestSpec(method=HTTPMethod.GET, path="https://api.example.test/path")
    with pytest.raises(ValidationError, match="supported extraction_source"):
        TemplateVariableBinding(
            name="PARENT_VALUE",
            source=TemplateVariableSource.DEPENDENCY,
            source_key="$.id",
            dependency_case_id="TC_PARENT",
            description="Missing extraction source is rejected",
        )


def _portal_ready_spec(case_id: str, state_id: str) -> ExecutionSpec:
    grounding = _grounding()
    source_case = next(
        case for case in grounding.test_cases if case.context.test_case_id == case_id
    )
    data = _ready_spec(
        {
            "test_case_id": case_id,
            "dependency_case_ids": list(source_case.context.dependency_case_ids),
            "available_evidence_snippet_ids": [grounding.snippets[0].snippet_id],
        }
    ).model_dump(mode="python")
    data["portal_evidence_state_ids"] = (state_id,)
    data["evidence_snippet_ids"] = ()
    return ExecutionSpec.model_validate(data)


@pytest.mark.asyncio
async def test_agentic_synthesis_finishes_from_portal_facts_without_tools() -> None:
    grounding = _grounding()
    specification = _portal_ready_spec("TC_01", "state-1")
    llm = _AgentLLM(
        [
            SynthesisAgentResponse(
                action=SynthesisAgentAction.FINAL,
                rationale="Portal facts contain the complete request and assertions",
                specification=specification,
            )
        ]
    )
    tools = SynthesisEvidenceTools(grounding)

    package = await AgenticSynthesisBuilder(
        grounding,
        "d" * 64,
        llm,
        tools,
        config=AgenticSynthesisConfig(maximum_turns_per_case=2),
    ).build(initial_specifications=(_portal_ready_spec("TC_02", "state-2"),))

    assert package.strategy == SynthesisStrategy.AGENTIC
    assert package.coverage.synthesis_complete is True
    assert package.agent_observations == ()
    assert package.retrieved_snippets == ()
    assert len(package.calls) == 1
    assert len(llm.prompts) == 1
    assert len(llm.prompts[0]) < 10_000


@pytest.mark.asyncio
async def test_agentic_synthesis_searches_then_reads_bounded_evidence(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repository"
    docs = repository_root / "docs"
    docs.mkdir(parents=True)
    content = (
        "BillFetchRequest POST /bill/fetch accepts requestId. "
        + ("supporting-context " * 30)
        + "TAIL_ASSERTION HTTP 200"
    )
    (docs / "BillFetchRequest.md").write_text(content)
    repository = RepositoryIndex.build(
        repository_root,
        RepositoryIndexConfig(maximum_excerpt_characters=2_000),
    )
    live_snippet = repository.search(
        "BillFetchRequest requestId endpoint",
        limit=1,
        anchor_terms=("BillFetchRequest",),
    )[0]
    case = {
        "test_case_id": "TC_01",
        "dependency_case_ids": [],
        "available_evidence_snippet_ids": [live_snippet.snippet_id],
    }
    llm = _AgentLLM(
        [
            SynthesisAgentResponse(
                action=SynthesisAgentAction.SEARCH_REPOSITORY,
                rationale="Find the exact API contract",
                query="BillFetchRequest requestId endpoint",
            ),
            SynthesisAgentResponse(
                action=SynthesisAgentAction.READ_EVIDENCE,
                rationale="Read the matching API contract",
                snippet_id=live_snippet.snippet_id,
            ),
            SynthesisAgentResponse(
                action=SynthesisAgentAction.FINAL,
                rationale="The read contract supports the execution specification",
                specification=_ready_spec(case),
            ),
        ]
    )
    grounding = _grounding()
    tools = SynthesisEvidenceTools(
        grounding,
        repository=repository,
        config=AgenticSynthesisConfig(
            maximum_turns_per_case=4,
            evidence_preview_characters=120,
            maximum_evidence_characters=2_000,
        ),
    )

    package = await AgenticSynthesisBuilder(
        grounding,
        "d" * 64,
        llm,
        tools,
        config=tools.config,
    ).build(initial_specifications=(_portal_ready_spec("TC_02", "state-2"),))

    assert [item.action for item in package.agent_observations] == [
        SynthesisAgentAction.SEARCH_REPOSITORY,
        SynthesisAgentAction.READ_EVIDENCE,
    ]
    assert package.specifications[0].evidence_snippet_ids == (live_snippet.snippet_id,)
    assert [item.snippet_id for item in package.retrieved_snippets] == [
        live_snippet.snippet_id
    ]
    assert "TAIL_ASSERTION" not in llm.prompts[1]
    assert "TAIL_ASSERTION" in llm.prompts[2]
    assert max(map(len, llm.prompts)) < 15_000


@pytest.mark.asyncio
async def test_agent_rejects_unread_external_citation_then_self_corrects() -> None:
    grounding = _grounding()
    unread = grounding.snippets[0].snippet_id
    invalid_case = {
        "test_case_id": "TC_01",
        "dependency_case_ids": [],
        "available_evidence_snippet_ids": [unread],
    }
    llm = _AgentLLM(
        [
            SynthesisAgentResponse(
                action=SynthesisAgentAction.FINAL,
                rationale="Attempt an unread citation",
                specification=_ready_spec(invalid_case),
            ),
            SynthesisAgentResponse(
                action=SynthesisAgentAction.FINAL,
                rationale="Use the directly observed portal evidence",
                specification=_portal_ready_spec("TC_01", "state-1"),
            ),
        ]
    )

    package = await AgenticSynthesisBuilder(
        grounding,
        "d" * 64,
        llm,
        SynthesisEvidenceTools(grounding),
        config=AgenticSynthesisConfig(maximum_turns_per_case=2),
    ).build(initial_specifications=(_portal_ready_spec("TC_02", "state-2"),))

    assert package.coverage.synthesis_complete is True
    assert [item.valid for item in package.calls] == [False, True]
    assert "evidence that was not read" in llm.prompts[1]


@pytest.mark.asyncio
async def test_agent_rejects_resumed_specification_with_unread_evidence() -> None:
    grounding = _grounding()
    snippet = grounding.snippets[0]
    case = {
        "test_case_id": "TC_01",
        "dependency_case_ids": [],
        "available_evidence_snippet_ids": [snippet.snippet_id],
    }

    with pytest.raises(ValueError, match="evidence that was not read"):
        await AgenticSynthesisBuilder(
            grounding,
            "d" * 64,
            _AgentLLM([]),
            SynthesisEvidenceTools(grounding),
        ).build(
            initial_specifications=(_ready_spec(case),),
            initial_retrieved_snippets=(snippet,),
            initial_observations=(),
        )

    read_observation = SynthesisToolObservation(
        sequence=1,
        test_case_id="TC_01",
        turn=1,
        action=SynthesisAgentAction.READ_EVIDENCE,
        request_sha256="a" * 64,
        result_sha256="b" * 64,
        result_characters=len(snippet.content),
        evidence_snippet_ids=(snippet.snippet_id,),
    )
    resumed = await AgenticSynthesisBuilder(
        grounding,
        "d" * 64,
        _AgentLLM([]),
        SynthesisEvidenceTools(grounding),
    ).build(
        initial_specifications=(
            _ready_spec(case),
            _portal_ready_spec("TC_02", "state-2"),
        ),
        initial_retrieved_snippets=(snippet,),
        initial_observations=(read_observation,),
    )

    assert resumed.coverage.synthesis_complete is True
    assert resumed.agent_observations == (read_observation,)


@pytest.mark.parametrize(
    "provider_error",
    ("LiteLLM HTTP 429", "LiteLLM request failed: read operation timed out"),
)
@pytest.mark.asyncio
async def test_agent_opens_provider_circuit_after_transport_failure(
    provider_error: str,
) -> None:
    grounding = _grounding()
    llm = _UnavailableLLM(provider_error)

    package = await AgenticSynthesisBuilder(
        grounding,
        "d" * 64,
        llm,
        SynthesisEvidenceTools(grounding),
        config=AgenticSynthesisConfig(concurrency=1, maximum_turns_per_case=2),
    ).build()

    assert llm.calls == 1
    assert package.coverage.synthesized == 0
    assert any("provider circuit opened" in error for error in package.provider.errors)


@pytest.mark.asyncio
async def test_builder_validates_citations_retries_and_exports(tmp_path: Path) -> None:
    grounding = _grounding()
    llm = _FakeLLM(invalid_first=True)
    progress: list[int] = []
    package = await SynthesisBuilder(
        grounding,
        "d" * 64,
        llm,
        config=SynthesisBuildConfig(
            maximum_cases_per_batch=2,
            concurrency=1,
            maximum_validation_attempts=2,
        ),
    ).build(progress=lambda specs, calls, errors: progress.append(len(specs)))

    assert package.coverage.synthesis_complete is True
    assert package.coverage.execution_ready is True
    assert package.coverage.ready == 2
    assert len(package.calls) == 2
    assert package.calls[0].valid is False
    assert package.calls[1].valid is True
    assert "previous response was rejected" in llm.prompts[1]
    assert progress == [2]

    exported = export_synthesis(package, tmp_path / "synthesis")
    assert exported.specifications_path.read_text().count("\n") == 2
    for item in exported.manifest.files:
        data = (exported.output_directory / item.name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item.sha256


@pytest.mark.asyncio
async def test_builder_logs_batch_progress_without_prompt_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="cz.synthesis.builder")

    await SynthesisBuilder(
        _grounding(),
        "d" * 64,
        _FakeLLM(),
        config=SynthesisBuildConfig(maximum_cases_per_batch=2, concurrency=1),
    ).build()

    assert "batch queued" in caplog.text
    assert "batch started" in caplog.text
    assert "batch completed" in caplog.text
    assert "build progress batches_finished=1/1" in caplog.text
    assert "INPUT_CONTEXT" not in caplog.text


@pytest.mark.asyncio
async def test_builder_resumes_completed_specs_without_new_calls() -> None:
    grounding = _grounding()
    first_llm = _FakeLLM()
    first = await SynthesisBuilder(
        grounding,
        "d" * 64,
        first_llm,
        config=SynthesisBuildConfig(maximum_cases_per_batch=2),
    ).build()
    resumed_llm = _FakeLLM()
    resumed = await SynthesisBuilder(
        grounding,
        "d" * 64,
        resumed_llm,
        config=SynthesisBuildConfig(maximum_cases_per_batch=2),
    ).build(
        initial_specifications=first.specifications,
        initial_calls=first.calls,
    )

    assert resumed.specifications == first.specifications
    assert resumed.calls == first.calls
    assert resumed_llm.prompts == []


def test_checkpoint_is_validated_and_secret_free(tmp_path: Path) -> None:
    grounding = _grounding()
    case = {
        "test_case_id": "TC_01",
        "dependency_case_ids": [],
        "available_evidence_snippet_ids": [grounding.snippets[0].snippet_id],
    }
    checkpoint_path = tmp_path / "checkpoint.json"
    saved = save_checkpoint(
        checkpoint_path,
        source_run_id="fixture-run",
        source_grounding_sha256="d" * 64,
        configuration_sha256="e" * 64,
        model="fixture-model",
        calls=(),
        specifications=(_ready_spec(case),),
        errors=(),
    )

    assert load_checkpoint(checkpoint_path) == saved
    assert "top-secret" not in checkpoint_path.read_text().casefold()


def test_agentic_checkpoint_preserves_retrieved_evidence_and_observations(
    tmp_path: Path,
) -> None:
    grounding = _grounding()
    snippet = grounding.snippets[0]
    observation = SynthesisToolObservation(
        sequence=1,
        test_case_id="TC_01",
        turn=1,
        action=SynthesisAgentAction.READ_EVIDENCE,
        request_sha256="a" * 64,
        result_sha256="b" * 64,
        result_characters=len(snippet.content),
        evidence_snippet_ids=(snippet.snippet_id,),
    )
    checkpoint_path = tmp_path / "agentic-checkpoint.json"

    saved = save_checkpoint(
        checkpoint_path,
        source_run_id="fixture-run",
        source_grounding_sha256="d" * 64,
        configuration_sha256="e" * 64,
        model="fixture-model",
        strategy=SynthesisStrategy.AGENTIC,
        calls=(),
        retrieved_snippets=(snippet,),
        agent_observations=(observation,),
        specifications=(_portal_ready_spec("TC_01", "state-1"),),
        errors=(),
    )

    loaded = load_checkpoint(checkpoint_path)
    assert loaded == saved
    assert loaded.strategy == SynthesisStrategy.AGENTIC
    assert loaded.retrieved_snippets == (snippet,)
    assert loaded.agent_observations == (observation,)


@pytest.mark.asyncio
async def test_litellm_client_uses_structured_response_without_key_in_payload() -> None:
    transport = _FakeTransport()
    client = LiteLLMClient(
        LiteLLMConfig(
            endpoint="https://llm.example.test/v1",
            model="fixture-model",
            api_key=SecretStr("top-secret"),
        ),
        transport=transport,
    )

    completion = await client.complete(
        system_prompt="system",
        user_prompt="user",
        response_model=SynthesisBatchResponse,
        schema_name="test_schema",
    )

    assert completion.content == '{"specifications":[]}'
    assert transport.request is not None
    assert transport.request["response_format"]["type"] == "json_schema"
    response_schema = transport.request["response_format"]["json_schema"]["schema"]
    assert response_schema["required"] == ["specifications"]
    assert response_schema["additionalProperties"] is False
    assert "top-secret" not in json.dumps(transport.request)
    assert transport.api_key is not None
    assert transport.api_key.get_secret_value() == "top-secret"
    assert normalize_litellm_endpoint("https://llm.example.test") == (
        "https://llm.example.test/v1/chat/completions"
    )


@pytest.mark.asyncio
async def test_litellm_client_logs_safe_waiting_heartbeats(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="cz.synthesis.transport")
    client = LiteLLMClient(
        LiteLLMConfig(
            endpoint="https://llm.example.test/v1",
            model="fixture-model",
            api_key=SecretStr("top-secret"),
            progress_heartbeat_seconds=0.01,
        ),
        transport=_SlowTransport(),
    )

    await client.complete(
        system_prompt="private-system-prompt",
        user_prompt="private-user-prompt",
        response_model=SynthesisBatchResponse,
        schema_name="test_schema",
    )

    assert "model request started" in caplog.text
    assert "model request waiting" in caplog.text
    assert "model response received" in caplog.text
    assert "top-secret" not in caplog.text
    assert "private-system-prompt" not in caplog.text
    assert "private-user-prompt" not in caplog.text


def test_progress_logger_writes_run_local_file(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress.log"
    logger = logging.getLogger("cz.synthesis")
    try:
        _configure_progress_logging(progress_path, quiet=True, append=False)
        logging.getLogger("cz.synthesis.builder").info(
            "batch completed batch=abc123 elapsed_seconds=1.0"
        )
        for handler in logger.handlers:
            handler.flush()

        content = progress_path.read_text()
        assert "batch completed batch=abc123" in content
        assert "INFO" in content
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


@pytest.mark.asyncio
async def test_builder_rejects_checkpoint_with_forged_citation() -> None:
    grounding = _grounding()
    case = {
        "test_case_id": "TC_01",
        "dependency_case_ids": [],
        "available_evidence_snippet_ids": [grounding.snippets[0].snippet_id],
    }
    forged = _ready_spec(case).model_copy(update={"evidence_snippet_ids": ("f" * 64,)})

    with pytest.raises(SynthesisBuildError, match="invalid citations"):
        await SynthesisBuilder(
            grounding,
            "d" * 64,
            _FakeLLM(),
        ).build(initial_specifications=(forged,))


def test_grounding_loader_and_plan_only_cli_verify_phase_boundary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exported = export_grounding(_grounding(), tmp_path / "grounding")

    loaded = load_grounding(exported.output_directory)
    assert loaded.grounding.source_run_id == "fixture-run"
    assert (
        synthesis_main(
            [
                "--grounding",
                str(exported.output_directory),
                "--plan-only",
                "--maximum-cases-per-batch",
                "1",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["test_cases"] == 2
    assert plan["planned_model_calls"] == 2

    exported.grounding_path.write_bytes(exported.grounding_path.read_bytes() + b"\n")
    with pytest.raises(SynthesisBuildError, match="does not match"):
        load_grounding(exported.output_directory)
