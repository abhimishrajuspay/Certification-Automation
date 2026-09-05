from __future__ import annotations

import datetime as _dt

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
from grounding.repository import (
    RepositoryIndex,
    RepositoryIndexConfig,
    resolve_repository_suffixes,
)
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
    _budgeted_catalog,
    _budgeted_read_payloads,
)
from synthesis.cli import _configure_progress_logging, main as synthesis_main
from synthesis.client import (
    LiteLLMClient,
    LiteLLMCompletion,
    LiteLLMConfig,
    LiteLLMError,
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
    SynthesisAgentDecision,
    SynthesisCallStage,
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
    def __init__(
        self,
        responses: list[BaseModel],
        *,
        response_format: str = "json_schema",
    ) -> None:
        self._config = LiteLLMConfig(
            endpoint="https://llm.example.test",
            model="fixture-agent-model",
            response_format=response_format,
        )
        self.responses = responses
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []
        self.response_models: list[Type[BaseModel]] = []
        self.maximum_output_tokens: list[int | None] = []

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
        del schema_name
        self.prompts.append(user_prompt)
        self.system_prompts.append(system_prompt)
        self.response_models.append(response_model)
        self.maximum_output_tokens.append(maximum_output_tokens)
        response = self.responses[len(self.prompts) - 1]
        if isinstance(response, LiteLLMError):
            raise response
        assert isinstance(response, response_model)
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
        maximum_output_tokens: int | None = None,
    ) -> LiteLLMCompletion:
        del (
            system_prompt,
            user_prompt,
            response_model,
            schema_name,
            maximum_output_tokens,
        )
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
            SynthesisAgentDecision(
                action=SynthesisAgentAction.FINAL,
                rationale="Portal facts contain the complete request and assertions",
            ),
            specification,
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
    assert [item.stage for item in package.calls] == [
        SynthesisCallStage.AGENT_TURN,
        SynthesisCallStage.SPECIFICATION,
    ]
    assert len(llm.prompts) == 2
    assert len(llm.prompts[0]) < len(llm.prompts[1])
    assert llm.maximum_output_tokens == [2_048, None]


@pytest.mark.asyncio
async def test_json_object_mode_keeps_decision_schema_separate() -> None:
    grounding = _grounding()
    specification = _portal_ready_spec("TC_01", "state-1")
    llm = _AgentLLM(
        [
            SynthesisAgentDecision(
                action=SynthesisAgentAction.FINAL,
                rationale="Portal evidence is complete",
            ),
            specification,
        ],
        response_format="json_object",
    )

    await AgenticSynthesisBuilder(
        grounding,
        "d" * 64,
        llm,
        SynthesisEvidenceTools(grounding),
        config=AgenticSynthesisConfig(maximum_turns_per_case=2),
    ).build(initial_specifications=(_portal_ready_spec("TC_02", "state-2"),))

    assert "variable_bindings" not in llm.prompts[0]
    assert "variable_bindings" in llm.prompts[1]
    assert len(llm.prompts[0]) < 6_000


@pytest.mark.asyncio
async def test_agent_normalizes_description_literal_binding_without_retry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    grounding = _grounding()
    first = grounding.test_cases[0]
    context = first.context.model_copy(
        update={
            "description": (
                "Request payload has merchantCustomerId CUST0001 and returns 200"
            )
        }
    )
    grounding = grounding.model_copy(
        update={
            "test_cases": (
                first.model_copy(update={"context": context}),
                grounding.test_cases[1],
            )
        }
    )
    base = _portal_ready_spec("TC_01", "state-1")
    assert base.request is not None
    request = base.request.model_copy(
        update={
            "body": RequestBodySpec(
                mode=RequestBodyMode.JSON,
                content_type="application/json",
                template=('{"merchantCustomerId":"{{MERCHANT_CUSTOMER_ID}}"}'),
            )
        }
    )
    bad_binding = TemplateVariableBinding(
        name="MERCHANT_CUSTOMER_ID",
        source=TemplateVariableSource.PORTAL_FIELD,
        source_key="api_name",
        value="CUST0001",
        description="Customer ID copied from the portal description",
    )
    specification = ExecutionSpec.model_validate(
        base.model_dump(mode="python")
        | {
            "request": request,
            "variable_bindings": (
                base.variable_bindings[0],
                bad_binding,
            ),
        }
    )
    llm = _AgentLLM(
        [
            SynthesisAgentDecision(
                action=SynthesisAgentAction.FINAL,
                rationale="Portal description contains the complete request",
            ),
            specification,
        ]
    )
    caplog.set_level(logging.INFO, logger="cz.synthesis.agent")

    package = await AgenticSynthesisBuilder(
        grounding,
        "d" * 64,
        llm,
        SynthesisEvidenceTools(grounding),
        config=AgenticSynthesisConfig(maximum_turns_per_case=2),
    ).build(initial_specifications=(_portal_ready_spec("TC_02", "state-2"),))

    normalized = package.specifications[0]
    customer = next(
        item
        for item in normalized.variable_bindings
        if item.name == "MERCHANT_CUSTOMER_ID"
    )
    assert customer.source == TemplateVariableSource.EVIDENCE_LITERAL
    assert customer.source_key is None
    assert "detail-1" in normalized.portal_evidence_state_ids
    assert len(package.calls) == 2
    assert "MERCHANT_CUSTOMER_ID:evidence_literal" in caplog.text
    assert "CUST0001" not in caplog.text


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
            SynthesisAgentDecision(
                action=SynthesisAgentAction.SEARCH_REPOSITORY,
                rationale="Find the exact API contract",
                query="BillFetchRequest requestId endpoint",
            ),
            SynthesisAgentDecision(
                action=SynthesisAgentAction.READ_EVIDENCE,
                rationale="Read the matching API contract",
                snippet_id=live_snippet.snippet_id,
            ),
            SynthesisAgentDecision(
                action=SynthesisAgentAction.FINAL,
                rationale="The read contract supports the execution specification",
            ),
            _ready_spec(case),
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
            SynthesisAgentDecision(
                action=SynthesisAgentAction.FINAL,
                rationale="Attempt an unread citation",
            ),
            _ready_spec(invalid_case),
            _portal_ready_spec("TC_01", "state-1"),
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
    assert [item.valid for item in package.calls] == [True, False, True]
    assert "evidence that was not read" in llm.prompts[2]


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

    file_read_observation = read_observation.model_copy(
        update={"action": SynthesisAgentAction.READ_REPOSITORY_FILE}
    )
    resumed_via_file_read = await AgenticSynthesisBuilder(
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
        initial_observations=(file_read_observation,),
    )

    assert resumed_via_file_read.coverage.synthesis_complete is True


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

    # Decision transport failure forces one specification attempt before the
    # circuit opens, so the provider sees the decision call plus that fallback.
    assert llm.calls == 2
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
        maximum_output_tokens=321,
    )

    assert completion.content == '{"specifications":[]}'
    assert transport.request is not None
    assert transport.request["response_format"]["type"] == "json_schema"
    assert transport.request["max_tokens"] == 321
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
    assert plan["minimum_model_calls"] == 4
    assert plan["planned_model_calls"] == 4

    exported.grounding_path.write_bytes(exported.grounding_path.read_bytes() + b"\n")
    with pytest.raises(SynthesisBuildError, match="does not match"):
        load_grounding(exported.output_directory)


def test_decision_validates_read_repository_file_payload() -> None:
    with pytest.raises(ValueError, match="lacks its required payload"):
        SynthesisAgentDecision.model_validate(
            {
                "action": "read_repository_file",
                "rationale": "open the window",
                "repository_path": "api/apitypes.md",
                "line_start": 80,
                "line_end": 79,
            }
        )
    with pytest.raises(ValueError, match="lacks its required payload"):
        SynthesisAgentDecision.model_validate(
            {"action": "read_repository_file", "rationale": "open the window"}
        )
    with pytest.raises(ValueError, match="valid only for read_repository_file"):
        SynthesisAgentDecision.model_validate(
            {
                "action": "search_repository",
                "rationale": "search instead",
                "query": "ReqValAdd",
                "repository_path": "api/apitypes.md",
            }
        )


@pytest.mark.asyncio
async def test_agent_reads_repository_file_window_and_cites_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    (root / "api").mkdir(parents=True)
    payload_lines = (
        ["# API contract"]
        + [f"context line {index}" for index in range(2, 88)]
        + ["ReqValAdd POST /upi/2/{path} XML payload contract"]
        + [f"tail line {index}" for index in range(1, 60)]
    )
    (root / "api" / "apitypes.md").write_text("\n".join(payload_lines))
    repository = RepositoryIndex.build(
        root,
        RepositoryIndexConfig(maximum_excerpt_characters=2_000),
    )
    expected = repository.read_lines("api/apitypes.md", 80, 90)
    assert "ReqValAdd POST /upi/2/{path} XML" in expected.content

    grounding = _grounding()
    case = {
        "test_case_id": "TC_01",
        "dependency_case_ids": [],
        "available_evidence_snippet_ids": [expected.snippet_id],
    }
    llm = _AgentLLM(
        [
            SynthesisAgentDecision(
                action=SynthesisAgentAction.READ_REPOSITORY_FILE,
                rationale="Open the revealed contract window in the indexed file",
                repository_path="api/apitypes.md",
                line_start=80,
                line_end=90,
            ),
            SynthesisAgentDecision(
                action=SynthesisAgentAction.FINAL,
                rationale="The contract window answers the missing facts",
            ),
            _ready_spec(case),
        ]
    )
    tools = SynthesisEvidenceTools(
        grounding,
        repository=repository,
        config=AgenticSynthesisConfig(maximum_turns_per_case=3),
    )

    package = await AgenticSynthesisBuilder(
        grounding,
        "d" * 64,
        llm,
        tools,
        config=tools.config,
    ).build(initial_specifications=(_portal_ready_spec("TC_02", "state-2"),))

    assert [item.action for item in package.agent_observations] == [
        SynthesisAgentAction.READ_REPOSITORY_FILE
    ]
    assert package.retrieved_snippets == (expected,)
    assert package.specifications[0].disposition == SynthesisDisposition.READY
    assert package.specifications[0].evidence_snippet_ids == (expected.snippet_id,)


@pytest.mark.asyncio
async def test_agent_rejects_repeated_evidence_action_then_finalizes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    (root / "contract.md").write_text(
        "BillFetchRequest contract line 42. "
        + "BillFetchRequest supporting context. " * 12
    )
    repository = RepositoryIndex.build(
        root,
        RepositoryIndexConfig(maximum_excerpt_characters=2_000),
    )
    grounding = _grounding()
    decision = SynthesisAgentDecision(
        action=SynthesisAgentAction.SEARCH_REPOSITORY,
        rationale="Search the contract",
        query="BillFetchRequest contract line",
    )
    repeated = SynthesisAgentDecision(
        action=SynthesisAgentAction.SEARCH_REPOSITORY,
        rationale="Search the contract again",
        query="BillFetchRequest contract line",
    )
    tools = SynthesisEvidenceTools(
        grounding,
        repository=repository,
        config=AgenticSynthesisConfig(maximum_turns_per_case=4),
    )
    grounded_case = grounding.test_cases[0]
    first_result, first_ids, first_error = await tools.execute(grounded_case, decision)
    assert first_error is None
    assert first_ids
    second_result, second_ids, second_error = await tools.execute(
        grounded_case, repeated
    )
    assert second_error is None
    assert second_ids == ()
    assert "Duplicate action rejected" in second_result


@pytest.mark.asyncio
async def test_agent_forces_specification_after_turn_exhaustion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    (root / "api").mkdir(parents=True)
    lines = ["# XML API contract"] + [
        f"context line {index}" for index in range(2, 240)
    ]
    lines[87] = "ReqValAdd -> ReqBody '[XML] (Payload ReqValAdd) :> Post '[XML] Ack"
    (root / "api" / "apitypes.md").write_text("\n".join(lines))
    repository = RepositoryIndex.build(
        root,
        RepositoryIndexConfig(maximum_excerpt_characters=2_000),
    )
    grounding = _grounding()
    first_window = repository.read_lines("api/apitypes.md", 1, 60)
    second_window = repository.read_lines("api/apitypes.md", 61, 120)
    assert "ReqValAdd" in second_window.content
    case = {
        "test_case_id": "TC_01",
        "dependency_case_ids": [],
        "available_evidence_snippet_ids": [second_window.snippet_id],
    }
    llm = _AgentLLM(
        [
            SynthesisAgentDecision(
                action=SynthesisAgentAction.READ_REPOSITORY_FILE,
                rationale="Open the contract window",
                repository_path="api/apitypes.md",
                line_start=1,
                line_end=60,
            ),
            SynthesisAgentDecision(
                action=SynthesisAgentAction.READ_REPOSITORY_FILE,
                rationale="Open the neighboring window",
                repository_path="api/apitypes.md",
                line_start=61,
                line_end=120,
            ),
            _ready_spec(case),
        ]
    )
    tools = SynthesisEvidenceTools(
        grounding,
        repository=repository,
        config=AgenticSynthesisConfig(maximum_turns_per_case=2),
    )

    package = await AgenticSynthesisBuilder(
        grounding,
        "d" * 64,
        llm,
        tools,
        config=tools.config,
    ).build(initial_specifications=(_portal_ready_spec("TC_02", "state-2"),))

    actions = [item.action for item in package.agent_observations]
    assert actions == [
        SynthesisAgentAction.READ_REPOSITORY_FILE,
        SynthesisAgentAction.READ_REPOSITORY_FILE,
    ]
    assert package.specifications[0].disposition == SynthesisDisposition.READY
    assert len(package.specifications) == 2
    assert {item.snippet_id for item in package.retrieved_snippets} == {
        first_window.snippet_id,
        second_window.snippet_id,
    }


@pytest.mark.asyncio
async def test_agent_prefetch_surfaces_route_definitions_in_first_prompt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    flow = root / "src" / "product" / "billfetchrequest"
    flow.mkdir(parents=True)
    (flow / "Flow.hs").write_text(
        "module Flow where\n-- BillFetchRequest validates bills\n" * 4
    )
    routes = root / "src" / "app" / "routes"
    routes.mkdir(parents=True)
    (routes / "Bill.hs").write_text(
        "module Bill where\n-- BillFetchRequest route POST /bill/fetch JSON\n"
    )
    repository = RepositoryIndex.build(
        root,
        RepositoryIndexConfig(
            maximum_excerpt_characters=2_000,
            suffixes=resolve_repository_suffixes(include_code=True),
        ),
    )
    grounding = _grounding()
    llm = _AgentLLM(
        [
            SynthesisAgentDecision(
                action=SynthesisAgentAction.FINAL,
                rationale="Preview already shows the route definition",
            ),
            _portal_ready_spec("TC_01", "state-1"),
        ]
    )
    tools = SynthesisEvidenceTools(
        grounding,
        repository=repository,
        config=AgenticSynthesisConfig(maximum_turns_per_case=3),
    )

    package = await AgenticSynthesisBuilder(
        grounding,
        "d" * 64,
        llm,
        tools,
        config=tools.config,
    ).build(initial_specifications=(_portal_ready_spec("TC_02", "state-2"),))

    assert "src/app/routes/Bill.hs" in llm.prompts[0]
    assert package.specifications[0].disposition == SynthesisDisposition.READY


def test_prefetch_auto_anchor_reads_make_contract_tiles_citable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    routes = root / "src" / "App" / "Routes"
    routes.mkdir(parents=True)
    (routes / "Bill.hs").write_text(
        "module Bill where\n\nroutes :: Proxy BillAPI\n"
        '"BillFetchRequest" :> Capture "version" Text :> ReqBody \'[JSON] Body :> Post \'[JSON] Ack\n'
        "handler = undefined\n"
    )
    types = root / "src" / "product" / "billfetchrequest"
    types.mkdir(parents=True)
    (types / "Types.hs").write_text(
        "module Types where\n\ndata BillFetchRequest = BillFetchRequest\n"
        "  { _head :: Head, _txn :: Txn }\n  deriving (Show, Eq)\n\n"
        "data Head = Head { _ver :: Text } deriving (Show)\n\n"
        "data Txn = Txn { _txnId :: Text } deriving (Show)\n"
    )
    repository = RepositoryIndex.build(
        root,
        RepositoryIndexConfig(
            maximum_excerpt_characters=2_000,
            suffixes=resolve_repository_suffixes(include_code=True),
        ),
    )
    grounding = _grounding()
    tools = SynthesisEvidenceTools(
        grounding,
        repository=repository,
        config=AgenticSynthesisConfig(),
    )
    case = grounding.test_cases[0]

    tools.initialize_case(case)
    read_title_keys = set()
    for snippet_id in tools.read_ids(case):
        citation = tools.catalog[snippet_id].repository
        assert citation is not None
        read_title_keys.add(citation.path)
    read_content = "\n".join(
        tools.catalog[snippet_id].content for snippet_id in tools.read_ids(case)
    )
    assert {
        "src/App/Routes/Bill.hs",
        "src/product/billfetchrequest/Types.hs",
    } == read_title_keys
    assert '"BillFetchRequest" :>' in read_content
    assert "data BillFetchRequest" in read_content
    assert "data Head = Head" in read_content
    assert "data Txn = Txn" in read_content


@pytest.mark.asyncio
async def test_resume_accepts_completed_spec_citing_turn0_auto_reads(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    routes = root / "src" / "App" / "Routes"
    routes.mkdir(parents=True)
    (routes / "Bill.hs").write_text(
        "module Bill where\n\nroutes :: Proxy BillAPI\n"
        '"BillFetchRequest" :> Capture "version" Text :> ReqBody \'[JSON] Body :> Post \'[JSON] Ack\n'
        "handler = undefined\n"
    )
    types = root / "src" / "product" / "billfetchrequest"
    types.mkdir(parents=True)
    (types / "Types.hs").write_text(
        "module Types where\n\ndata BillFetchRequest = BillFetchRequest\n"
        "  { _head :: Head, _txn :: Txn }\n  deriving (Show, Eq)\n\n"
        "data Head = Head { _ver :: Text } deriving (Show)\n\n"
        "data Txn = Txn { _txnId :: Text } deriving (Show)\n"
    )

    def build_tools(grounding: GroundingPackage) -> SynthesisEvidenceTools:
        repository = RepositoryIndex.build(
            root,
            RepositoryIndexConfig(
                maximum_excerpt_characters=2_000,
                suffixes=resolve_repository_suffixes(include_code=True),
            ),
        )
        return SynthesisEvidenceTools(
            grounding,
            repository=repository,
            config=AgenticSynthesisConfig(),
        )

    grounding = _grounding()
    tools = build_tools(grounding)
    case = grounding.test_cases[0]
    tools.initialize_case(case)
    read_ids = list(tools.read_ids(case))
    assert read_ids

    completed = _ready_spec(
        {
            "test_case_id": case.context.test_case_id,
            "dependency_case_ids": [],
            "available_evidence_snippet_ids": read_ids,
        }
    )
    snippets = tools.case_snippets(case)

    # Cold resume: no observation records the turn-0 anchor reads; acceptance
    # must come from the deterministic auto-read rebuild union.
    resumed = await AgenticSynthesisBuilder(
        grounding,
        "d" * 64,
        _AgentLLM([]),
        build_tools(grounding),
        config=AgenticSynthesisConfig(),
    ).build(
        initial_specifications=(completed, _portal_ready_spec("TC_02", "state-2")),
        initial_retrieved_snippets=tuple(snippets),
        initial_observations=(),
    )

    assert resumed.coverage.synthesis_complete is True
    assert len(resumed.specifications) == 2


def test_auto_cascade_reads_route_return_type_and_xml_instances(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    routes = root / "src" / "App" / "Routes"
    routes.mkdir(parents=True)
    (routes / "Bill.hs").write_text(
        "module Bill where\n\nroutes :: Proxy BillAPI\n"
        '"BillFetchRequest" :> Capture "version" Text :> ReqBody \'[JSON] Body :> Post \'[XML] Ack\n'
        "handler = undefined\n"
    )
    types = root / "src" / "product" / "billfetchrequest"
    types.mkdir(parents=True)
    (types / "Types.hs").write_text(
        "module Types where\n\ndata BillFetchRequest = BillFetchRequest\n"
        "  { _head :: Head, _txn :: Txn }\n  deriving (Show, Eq)\n\n"
        "data Head = Head { _ver :: Text } deriving (Show)\n\n"
        'instance ToXml Head where\n  toXml h = [XAttr "ver" (_ver h)]\n\n'
        "data Txn = Txn { _txnId :: Text } deriving (Show)\n"
    )
    (types / "Ack.hs").write_text(
        "module Ack where\n\ndata Ack = Ack { code :: Text, err :: Maybe Text }\n"
        "  deriving (Show)\n\ninstance FromXml Ack where\n  fromXml = undefined\n"
    )

    repository = RepositoryIndex.build(
        root,
        RepositoryIndexConfig(
            maximum_excerpt_characters=2_000,
            suffixes=resolve_repository_suffixes(include_code=True),
        ),
    )
    grounding = _grounding()
    tools = SynthesisEvidenceTools(
        grounding,
        repository=repository,
        config=AgenticSynthesisConfig(),
    )
    case = grounding.test_cases[0]

    tools.initialize_case(case)
    read_content = "\n".join(
        tools.catalog[snippet_id].content for snippet_id in tools.read_ids(case)
    )
    read_paths = {
        tools.catalog[snippet_id].repository.path for snippet_id in tools.read_ids(case)
    }
    assert "data Ack = Ack" in read_content  # cascade C: route return type
    assert "instance FromXml Ack where" in read_content  # cascade D: instance
    assert "instance ToXml Head where" in read_content
    assert "src/product/billfetchrequest/Ack.hs" in read_paths


def test_auto_route_context_reads_mount_prefix_and_handler_body(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    routes = root / "src" / "App" / "Routes"
    routes.mkdir(parents=True)
    (routes / "Bill.hs").write_text(
        "module Bill where\n\ntype BillAPI =\n"
        '  "BillFetchRequest" :> Capture "version" Text :> Capture "path" Text\n'
        "    :> Vault :> ReqBody '[XML] Body :> Post '[XML] Ack\n\n"
        "billFetchRequest :: V.Key (KM.KeyMap Text) -> Text -> Text -> Vault -> Body -> Handler Ack\n"
        "billFetchRequest key version path vault reqBody = do\n"
        '  logAPIEnc "BillFetch API called." reqBody\n'
        "  handleBillRequest reqBody version\n"
    )
    (routes / "Core.hs").write_text(
        "module Core where\n\ntype RootAPIs =\n"
        '  "health" :> Get \'[JSON] Head :<|> "upi" :> Bill.BillAPI\n'
        "rootServer = undefined\n"
    )
    types = root / "src" / "product" / "billfetchrequest"
    types.mkdir(parents=True)
    (types / "Types.hs").write_text(
        "module Types where\n\ndata BillFetchRequest = BillFetchRequest\n"
        "  { _head :: Head, _txn :: Txn }\n  deriving (Show, Eq)\n\n"
        "data Head = Head { _ver :: Text } deriving (Show)\n\n"
        "data Txn = Txn { _txnId :: Text } deriving (Show)\n"
    )
    repository = RepositoryIndex.build(
        root,
        RepositoryIndexConfig(
            maximum_excerpt_characters=2_000,
            suffixes=resolve_repository_suffixes(include_code=True),
        ),
    )
    grounding = _grounding()
    tools = SynthesisEvidenceTools(
        grounding,
        repository=repository,
        config=AgenticSynthesisConfig(),
    )
    case = grounding.test_cases[0]

    tools.initialize_case(case)
    read_content = "\n".join(
        tools.catalog[snippet_id].content for snippet_id in tools.read_ids(case)
    )
    assert '"upi" :> Bill.BillAPI' in read_content  # mount prefix (cascade A)
    assert (
        "billFetchRequest key version path vault" in read_content
    )  # handler body (cascade B)


@pytest.mark.asyncio
async def test_complete_turn0_evidence_goes_straight_to_specification(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    routes = root / "src" / "App" / "Routes"
    routes.mkdir(parents=True)
    (routes / "Bill.hs").write_text(
        "module Bill where\n\ntype BillAPI =\n"
        '  "BillFetchRequest" :> Capture "version" Text :> Capture "path" Text\n'
        "    :> Vault :> ReqBody '[XML] Body :> Post '[XML] Ack\n\n"
        "billFetchRequest :: V.Key (KM.KeyMap Text) -> Text -> Text -> Vault -> Body -> Handler Ack\n"
        "billFetchRequest key version path vault reqBody = handleBill reqBody version\n"
    )
    (routes / "Core.hs").write_text(
        "module Core where\n\ntype RootAPIs =\n"
        '  "health" :> Get \'[JSON] Head :<|> "upi" :> Bill.BillAPI\n'
        "rootServer = undefined\n"
    )
    types = root / "src" / "product" / "billfetchrequest"
    types.mkdir(parents=True)
    (types / "Types.hs").write_text(
        "module Types where\n\ndata BillFetchRequest = BillFetchRequest\n"
        "  { _head :: Head, _txn :: Txn }\n  deriving (Show, Eq)\n\n"
        "data Head = Head { _ver :: Text } deriving (Show)\n\n"
        'instance ToXml Head where\n  toXml h = [XAttr "ver" (_ver h)]\n\n'
        "data Txn = Txn { _txnId :: Text } deriving (Show)\n"
    )
    (types / "Ack.hs").write_text(
        "module Ack where\n\ndata Ack = Ack { code :: Text, err :: Maybe Text }\n"
        "  deriving (Show)\n\ninstance FromXml Ack where\n  fromXml = undefined\n"
    )
    repository = RepositoryIndex.build(
        root,
        RepositoryIndexConfig(
            maximum_excerpt_characters=2_000,
            suffixes=resolve_repository_suffixes(include_code=True),
        ),
    )
    grounding = _grounding()
    tools = SynthesisEvidenceTools(
        grounding,
        repository=repository,
        config=AgenticSynthesisConfig(),
    )
    case = grounding.test_cases[0]
    tools.initialize_case(case)
    completeness = tools.evidence_completeness(case)
    assert all(completeness.values()), completeness

    read_ids = list(tools.read_ids(case))
    spec = _ready_spec(
        {
            "test_case_id": case.context.test_case_id,
            "dependency_case_ids": [],
            "available_evidence_snippet_ids": read_ids,
        }
    )
    llm = _AgentLLM([spec])
    builder = AgenticSynthesisBuilder(
        grounding,
        "d" * 64,
        llm,
        SynthesisEvidenceTools(
            grounding,
            repository=repository,
            config=AgenticSynthesisConfig(),
        ),
        config=AgenticSynthesisConfig(),
    )
    package = await builder.build(
        initial_specifications=(_portal_ready_spec("TC_02", "state-2"),)
    )

    assert len(llm.prompts) == 1  # ONE model call: the specification, no decision turns
    assert package.specifications[0].disposition == SynthesisDisposition.READY


def test_completeness_gate_requires_read_mcp_docs_when_grounding_selected_them() -> (
    None
):
    from grounding.mcp import (
        MCPClient,
        MCPClientConfig,
        MCPServerIdentity,
        MCPToolResult,
    )

    mcp_client = MCPClient(MCPClientConfig(endpoint="https://mcp.example.test/mcp"))
    mcp_client.identity = MCPServerIdentity(
        "2025-06-18", "merchant-integration-mcp", "3.4.7"
    )
    mcp_result = MCPToolResult(
        tool_name="search_docs",
        query="q",
        arguments_sha256="a" * 64,
        response_sha256="b" * 64,
        retrieved_at=_dt.datetime(2026, 8, 31, tzinfo=_dt.timezone.utc),
        text_blocks=(
            "### Result 1\n**Source:** s2s_api_docs / vpas-validity.md\n"
            "**Document ID:** `vpa_doc`\n**Chunk:** `0`\n\n```\nPOST /api/{apiVersion}/merchants/vpas/validity JSON body customerVpa.\n```\n",
        ),
    )
    mcp_snippet = mcp_client.to_snippets(mcp_result)[0]

    grounding = _grounding()
    case0 = grounding.test_cases[0]
    case = case0.model_copy(update={"mcp_snippet_ids": (mcp_snippet.snippet_id,)})
    grounding2 = grounding.model_copy(
        update={
            "snippets": grounding.snippets + (mcp_snippet,),
            "test_cases": (case,) + grounding.test_cases[1:],
        }
    )
    config = AgenticSynthesisConfig(prefetch_repository_evidence=False)
    tools = SynthesisEvidenceTools(grounding2, config=config)
    markers = tools.evidence_completeness(case)
    assert "mcp_documentation_read" in markers
    assert markers["mcp_documentation_read"] is False
    assert not all(markers.values())

    # A case whose grounding selected no MCP docs carries no MCP marker.
    markers2 = tools.evidence_completeness(grounding2.test_cases[1])
    assert "mcp_documentation_read" not in markers2


def test_search_mcp_arguments_are_bounded_to_the_advertised_schema() -> None:
    base = {
        "action": SynthesisAgentAction.SEARCH_MCP,
        "rationale": "Fetch the endpoint spec",
        "query": "vpas validity endpoint spec",
        "tool_name": "get_api_spec",
    }
    SynthesisAgentDecision.model_validate(
        {
            **base,
            "arguments": {"endpoint_id": "newton.s2s.post.merchants.vpas.validity"},
        }
    )
    with pytest.raises(ValidationError):
        SynthesisAgentDecision.model_validate({**base, "arguments": {"bad-key!": "x"}})
    with pytest.raises(ValidationError):
        SynthesisAgentDecision.model_validate(
            {**base, "arguments": {"endpoint_id": "v" * 513}}
        )
    with pytest.raises(ValidationError):
        SynthesisAgentDecision.model_validate(
            {**base, "arguments": {f"k{i}": "x" for i in range(7)}}
        )
    with pytest.raises(ValidationError):
        SynthesisAgentDecision.model_validate(
            {
                "action": SynthesisAgentAction.READ_EVIDENCE,
                "rationale": "wrong action carries arguments",
                "snippet_id": "f" * 64,
                "arguments": {"a": "b"},
            }
        )


@pytest.mark.asyncio
async def test_search_mcp_honours_model_constructed_arguments() -> None:
    from grounding.mcp import (
        MCPClient,
        MCPClientConfig,
        MCPServerIdentity,
        MCPTool,
        MCPToolResult,
    )

    captured: list[tuple[str, dict[str, object]]] = []

    class _SpecClient(MCPClient):
        async def call_tool(self, name, arguments):  # type: ignore[override]
            captured.append((name, dict(arguments)))
            return MCPToolResult(
                tool_name=name,
                query=str(arguments.get("endpoint_id", "")),
                arguments_sha256="a" * 64,
                response_sha256="b" * 64,
                retrieved_at=_dt.datetime(2026, 8, 31, tzinfo=_dt.timezone.utc),
                text_blocks=(
                    "### Result 1\n**Document ID:** `vpa-spec`\n\n```\n"
                    "POST /api/{apiVersion}/merchants/vpas/validity JSON body "
                    "customerVpa and payerVpa.\n```\n",
                ),
            )

    client = _SpecClient(
        MCPClientConfig(
            endpoint="https://mcp.example.test/mcp",
            allowed_tools=("get_api_spec",),
        )
    )
    client.identity = MCPServerIdentity(
        "2025-06-18", "merchant-integration-mcp", "3.4.7"
    )
    client.tools = (
        MCPTool(
            name="get_api_spec",
            description="Get complete API specification for an endpoint",
            input_schema={
                "type": "object",
                "properties": {"endpoint_id": {"type": "string"}},
                "required": ["endpoint_id"],
            },
        ),
    )
    grounding = _grounding()
    tools = SynthesisEvidenceTools(grounding, mcp=client)
    assert tools.available_mcp_tools == ("get_api_spec",)
    descriptors = tools.advertised_mcp_tool_descriptors()
    assert descriptors[0]["name"] == "get_api_spec"
    assert "endpoint_id" in str(descriptors[0]["input_schema"])

    decision = SynthesisAgentDecision(
        action=SynthesisAgentAction.SEARCH_MCP,
        rationale="Pull the endpoint specification",
        query="vpas validity endpoint specification",
        tool_name="get_api_spec",
        arguments={"endpoint_id": "newton.s2s.post.merchants.vpas.validity"},
    )
    result, snippet_ids, error = await tools.execute(grounding.test_cases[0], decision)
    assert error is None
    assert captured == [
        ("get_api_spec", {"endpoint_id": "newton.s2s.post.merchants.vpas.validity"})
    ]
    assert len(snippet_ids) == 1
    assert snippet_ids[0] in tools.catalog
    assert "customerVpa" in result

    # Same tool+query with different arguments is NOT a duplicate.
    decision2 = decision.model_copy(
        update={"arguments": {"endpoint_id": "newton.s2s.other.endpoint"}}
    )
    result2, _, error2 = await tools.execute(grounding.test_cases[0], decision2)
    assert error2 is None
    assert "Duplicate action rejected" not in result2


def test_specification_catalog_is_id_preserved_under_budget() -> None:
    catalog = [
        {"snippet_id": f"id-{index}", "title": f"t{index}", "content": "x" * 500}
        for index in range(20)
    ]
    budgeted = _budgeted_catalog(catalog, budget_characters=40_000)
    assert [item["snippet_id"] for item in budgeted] == [
        item["snippet_id"] for item in catalog
    ]
    full = [item for item in budgeted if "title" in item]
    id_only = [item for item in budgeted if set(item) == {"snippet_id"}]
    assert full and id_only
    serialized = json.dumps(full)
    assert len(serialized) <= int(40_000 * 0.25) + 4_000
    small = [{"snippet_id": "a", "title": "t"}]
    assert _budgeted_catalog(small, budget_characters=40_000) == small


def test_specification_read_payloads_are_budget_shaped() -> None:
    payloads = [
        {"snippet_id": str(index), "content": "x" * 2_000, "title": f"t{index}"}
        for index in range(15)
    ]
    budgeted = _budgeted_read_payloads(payloads, 40_000)
    assert sum(len(item["content"]) for item in budgeted) <= int(40_000 * 0.45) + 4_000
    assert any(
        "characters elided for prompt budget" in item["content"] for item in budgeted
    )
    small = [{"snippet_id": "a", "content": "y" * 100}]
    assert _budgeted_read_payloads(small, 40_000) == small


def test_case_snippets_persist_reads_from_static_prefetch() -> None:
    grounding = _grounding()
    tools = SynthesisEvidenceTools(grounding, config=AgenticSynthesisConfig())
    case = grounding.test_cases[0]

    tools.initialize_case(case)
    outcome_ids = {snippet.snippet_id for snippet in tools.case_snippets(case)}
    # Completed specs may cite ANY read evidence, including static prefetch
    # battery snippets; the persisted store must therefore cover read union.
    assert set(tools.read_ids(case)).issubset(outcome_ids)


class _FlakyEmptyContentTransport:
    def __init__(self, empties: int) -> None:
        self.empties = empties
        self.calls = 0

    async def post_json(
        self,
        endpoint: str,
        payload: bytes,
        *,
        api_key: SecretStr | None,
        timeout_seconds: float,
        maximum_response_bytes: int,
    ) -> LiteLLMTransportResponse:
        del endpoint, payload, api_key, timeout_seconds, maximum_response_bytes
        self.calls += 1
        content: object = (
            None if self.calls <= self.empties else '{"specifications":[]}'
        )
        body = json.dumps(
            {"choices": [{"message": {"content": content}}], "usage": {}}
        ).encode()
        return LiteLLMTransportResponse(
            payload=json.loads(body),
            response_sha256=hashlib.sha256(body).hexdigest(),
        )


@pytest.mark.asyncio
async def test_litellm_client_retries_transient_empty_assistant_content() -> None:
    transport = _FlakyEmptyContentTransport(empties=1)
    client = LiteLLMClient(
        LiteLLMConfig(
            endpoint="https://llm.example.test/v1",
            model="fixture-model",
            api_key=SecretStr("token"),
            retry_backoff_seconds=0.01,
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
    assert transport.calls == 2

    exhausted = _FlakyEmptyContentTransport(empties=99)
    client = LiteLLMClient(
        LiteLLMConfig(
            endpoint="https://llm.example.test/v1",
            model="fixture-model",
            api_key=SecretStr("token"),
            retry_backoff_seconds=0.01,
        ),
        transport=exhausted,
    )
    with pytest.raises(LiteLLMRetryableError, match="assistant content is empty"):
        await client.complete(
            system_prompt="system",
            user_prompt="user",
            response_model=SynthesisBatchResponse,
            schema_name="test_schema",
        )
    assert exhausted.calls == 3


@pytest.mark.asyncio
async def test_agent_memory_compaction_keeps_prompts_bounded(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    (root / "api").mkdir(parents=True)
    lines: list[str] = []
    for window in range(4):
        lines.append(f"MARKER_WINDOW_{window + 1}_START")
        lines.extend(f"filler {window + 1}.{index}" for index in range(55))
    (root / "api" / "contract.md").write_text("\n".join(lines))
    repository = RepositoryIndex.build(
        root,
        RepositoryIndexConfig(maximum_excerpt_characters=2_000),
    )
    grounding = _grounding()
    last_window = repository.read_lines("api/contract.md", 169, 224)
    case = {
        "test_case_id": "TC_01",
        "dependency_case_ids": [],
        "available_evidence_snippet_ids": [last_window.snippet_id],
    }
    decisions: list[object] = [
        SynthesisAgentDecision(
            action=SynthesisAgentAction.READ_REPOSITORY_FILE,
            rationale=f"Read window {window}",
            repository_path="api/contract.md",
            line_start=1 + 56 * window,
            line_end=56 + 56 * window,
        )
        for window in range(4)
    ]
    decisions += [
        SynthesisAgentDecision(
            action=SynthesisAgentAction.FINAL,
            rationale="Four windows read",
        ),
        _ready_spec(case),
    ]
    llm = _AgentLLM(decisions)
    tools = SynthesisEvidenceTools(
        grounding,
        repository=repository,
        config=AgenticSynthesisConfig(maximum_turns_per_case=6),
    )

    package = await AgenticSynthesisBuilder(
        grounding,
        "d" * 64,
        llm,
        tools,
        config=tools.config,
    ).build(initial_specifications=(_portal_ready_spec("TC_02", "state-2"),))

    assert package.specifications[0].disposition == SynthesisDisposition.READY
    decision_prompts = llm.prompts[:5]
    spec_prompt = llm.prompts[5]
    assert max(map(len, llm.prompts)) < 40_000
    final_decision = decision_prompts[-1]
    assert "MARKER_WINDOW_1_START" not in final_decision
    assert "MARKER_WINDOW_2_START" not in final_decision
    assert "MARKER_WINDOW_4_START" in final_decision
    assert "read_evidence_content" in spec_prompt
    for window in range(1, 5):
        assert f"MARKER_WINDOW_{window}_START" in spec_prompt
    assert len(package.specifications) == 2


@pytest.mark.asyncio
async def test_agent_decision_context_overflow_forces_specification(
    caplog: pytest.LogCaptureFixture,
) -> None:
    grounding = _grounding()
    # GroundingSnippet content is whitespace-stripped on validation, so the
    # digest must be taken over the normalized text.
    big_content = (
        "POST /bill/fetch accepts JSON. " + ("payload padding " * 320)
    ).strip()
    big_citation = RepositoryCitation(
        repository_id="a" * 64,
        path="docs/bill-fetch-big.md",
        file_sha256="c" * 64,
        line_start=1,
        line_end=2,
        excerpt_sha256=hashlib.sha256(big_content.encode()).hexdigest(),
        redacted=False,
    )
    big_id = _hash_json(
        {
            "source": GroundingSourceKind.REPOSITORY.value,
            "citation": big_citation.model_dump(mode="json"),
            "content": big_content,
        }
    )
    big_snippet = GroundingSnippet(
        snippet_id=big_id,
        source_kind=GroundingSourceKind.REPOSITORY,
        title="docs/bill-fetch-big.md:1-2",
        content=big_content,
        relevance_score=10,
        repository=big_citation,
    )
    case0 = grounding.test_cases[0].model_copy(
        update={
            "repository_snippet_ids": grounding.test_cases[0].repository_snippet_ids
            + (big_id,)
        }
    )
    grounding2 = grounding.model_copy(
        update={
            "snippets": grounding.snippets + (big_snippet,),
            "test_cases": (case0,) + grounding.test_cases[1:],
        }
    )
    case = {
        "test_case_id": "TC_01",
        "dependency_case_ids": [],
        "available_evidence_snippet_ids": [big_id],
    }
    llm = _AgentLLM(
        [
            SynthesisAgentDecision(
                action=SynthesisAgentAction.READ_EVIDENCE,
                rationale="Read the oversized seeded snippet",
                snippet_id=big_id,
            ),
            _ready_spec(case),
        ]
    )
    # Turn 1: preview-only decision prompt fits under the cap and the model
    # reads the oversized snippet. Turn 2: the raw 4KB read body in memory
    # pushes the decision prompt well past 2_000 -> forced specification with
    # the read evidence intact.
    # 4_500 sits in the window between the oversized turn-2 decision prompt
    # (~5_746 with the raw read body in memory) and the budgeted specification
    # prompt (~4_322), so the overflow forces the spec stage without failing it.
    tools = SynthesisEvidenceTools(
        grounding2,
        config=AgenticSynthesisConfig(
            maximum_turns_per_case=4,
            maximum_prompt_characters=4_500,
            maximum_evidence_characters=4_000,
        ),
    )

    with caplog.at_level("WARNING"):
        package = await AgenticSynthesisBuilder(
            grounding2,
            "d" * 64,
            llm,
            tools,
            config=tools.config,
        ).build(initial_specifications=(_portal_ready_spec("TC_02", "state-2"),))

    assert "agent decision context exceeded" in caplog.text
    assert package.specifications[0].test_case_id == "TC_01"
    assert package.specifications[0].disposition == SynthesisDisposition.READY


@pytest.mark.asyncio
async def test_agent_decision_transport_failure_forces_specification(
    caplog: pytest.LogCaptureFixture,
) -> None:
    grounding = _grounding()
    snippet_id = grounding.test_cases[0].repository_snippet_ids[0]
    case = {
        "test_case_id": "TC_01",
        "dependency_case_ids": [],
        "available_evidence_snippet_ids": [snippet_id],
    }
    llm = _AgentLLM(
        [
            SynthesisAgentDecision(
                action=SynthesisAgentAction.READ_EVIDENCE,
                rationale="Read the only seeded snippet",
                snippet_id=snippet_id,
            ),
            LiteLLMError("LiteLLM assistant content is empty"),
            _ready_spec(case),
        ]
    )
    tools = SynthesisEvidenceTools(
        grounding,
        config=AgenticSynthesisConfig(maximum_turns_per_case=4),
    )

    with caplog.at_level("WARNING"):
        package = await AgenticSynthesisBuilder(
            grounding,
            "d" * 64,
            llm,
            tools,
            config=tools.config,
        ).build(initial_specifications=(_portal_ready_spec("TC_02", "state-2"),))

    assert "decision transport failed" in caplog.text
    assert len(llm.prompts) == 3
    assert package.specifications[0].test_case_id == "TC_01"
    assert package.specifications[0].disposition == SynthesisDisposition.READY
    assert package.specifications[0].evidence_snippet_ids == (snippet_id,)
    assert len(package.specifications) == 2


@pytest.mark.asyncio
async def test_specification_prompt_excludes_unread_candidate_previews(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    read_text = (
        "READ_TARGET_CONTENT BillFetchRequest accepts requestId field. "
        + "padding " * 40
    )
    unseen_text = (
        "UNREAD_PREVIEW_MARKER BillFetchRequest route placeholder. " + "trailer " * 40
    )
    (root / "a-read.md").write_text(read_text)
    (root / "b-unread.md").write_text(unseen_text)
    repository = RepositoryIndex.build(
        root,
        RepositoryIndexConfig(maximum_excerpt_characters=2_000),
    )
    search_hits = repository.search(
        "BillFetchRequest requestId", limit=5, anchor_terms=("BillFetchRequest",)
    )
    assert len(search_hits) == 2
    read_hit = next(hit for hit in search_hits if "read" in hit.title)
    unseen_marker = "UNREAD_PREVIEW_MARKER"

    grounding = _grounding()
    case = {
        "test_case_id": "TC_01",
        "dependency_case_ids": [],
        "available_evidence_snippet_ids": [read_hit.snippet_id],
    }
    llm = _AgentLLM(
        [
            SynthesisAgentDecision(
                action=SynthesisAgentAction.SEARCH_REPOSITORY,
                rationale="Locate the request documentation",
                query="BillFetchRequest requestId",
            ),
            SynthesisAgentDecision(
                action=SynthesisAgentAction.READ_EVIDENCE,
                rationale="Read the marked request file",
                snippet_id=read_hit.snippet_id,
            ),
            SynthesisAgentDecision(
                action=SynthesisAgentAction.FINAL,
                rationale="Request contract confirmed",
            ),
            _ready_spec(case),
        ]
    )
    tools = SynthesisEvidenceTools(
        grounding,
        repository=repository,
        config=AgenticSynthesisConfig(maximum_turns_per_case=5),
    )

    package = await AgenticSynthesisBuilder(
        grounding,
        "d" * 64,
        llm,
        tools,
        config=tools.config,
    ).build(initial_specifications=(_portal_ready_spec("TC_02", "state-2"),))

    assert package.specifications[0].disposition == SynthesisDisposition.READY
    decision_prompts = llm.prompts[:3]
    assert any(unseen_marker in prompt for prompt in decision_prompts)
    spec_prompt = llm.prompts[3]
    assert unseen_marker not in spec_prompt
    assert "READ_TARGET_CONTENT" in spec_prompt


def test_decision_validates_read_repository_file_anchor_mode() -> None:
    anchor_only = SynthesisAgentDecision(
        action=SynthesisAgentAction.READ_REPOSITORY_FILE,
        rationale="Open the route declaration by its exact term",
        repository_path="api/apitypes.md",
        anchor="ReqValAdd :>",
    )
    assert anchor_only.anchor == "ReqValAdd :>"
    with pytest.raises(ValueError, match="mutually exclusive"):
        SynthesisAgentDecision(
            action=SynthesisAgentAction.READ_REPOSITORY_FILE,
            rationale="Anchor plus explicit window",
            repository_path="api/apitypes.md",
            anchor="ReqValAdd :>",
            line_start=80,
            line_end=90,
        )
    with pytest.raises(ValueError, match="read_repository_file"):
        SynthesisAgentDecision(
            action=SynthesisAgentAction.SEARCH_REPOSITORY,
            rationale="Anchor passed to a search",
            query="ReqValAdd",
            anchor="ReqValAdd :>",
        )


@pytest.mark.asyncio
async def test_agent_reads_repository_file_by_anchor_and_cites_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    (root / "api").mkdir(parents=True)
    lines = ["# contract"] + [f"context {index}" for index in range(2, 55)]
    lines.append('"ReqValAdd" :> Capture version :> ReqBody XML :> Post XML Ack')
    lines += [f"tail {index}" for index in range(2, 120)]
    (root / "api" / "routes.md").write_text("\n".join(lines))
    repository = RepositoryIndex.build(
        root,
        RepositoryIndexConfig(maximum_excerpt_characters=2_000),
    )
    expected = repository.read_around_match("api/routes.md", '"ReqValAdd" :>')
    grounding = _grounding()
    case = {
        "test_case_id": "TC_01",
        "dependency_case_ids": [],
        "available_evidence_snippet_ids": [expected.snippet_id],
    }
    llm = _AgentLLM(
        [
            SynthesisAgentDecision(
                action=SynthesisAgentAction.READ_REPOSITORY_FILE,
                rationale="Open the route declaration via its anchor",
                repository_path="api/routes.md",
                anchor='"ReqValAdd" :>',
            ),
            SynthesisAgentDecision(
                action=SynthesisAgentAction.FINAL,
                rationale="Route found by anchor",
            ),
            _ready_spec(case),
        ]
    )
    tools = SynthesisEvidenceTools(
        grounding,
        repository=repository,
        config=AgenticSynthesisConfig(maximum_turns_per_case=3),
    )

    package = await AgenticSynthesisBuilder(
        grounding,
        "d" * 64,
        llm,
        tools,
        config=tools.config,
    ).build(initial_specifications=(_portal_ready_spec("TC_02", "state-2"),))

    assert package.retrieved_snippets == (expected,)
    assert 'ReqValAdd"' in package.specifications[0].evidence_snippet_ids[0] or (
        package.specifications[0].evidence_snippet_ids == (expected.snippet_id,)
    )
