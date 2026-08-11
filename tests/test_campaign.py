"""Tests for repository-wide lazy certification campaigns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Type

import pytest
from pydantic import BaseModel

from campaign.builder import (
    CampaignBuildConfig,
    CampaignBuildError,
    CertificationCampaignAgent,
    _legacy_campaign_configuration_sha256,
    campaign_groups,
)
from campaign.cli import _reopen_needs_review
from campaign.io import archive_checkpoint_attempt
from campaign.models import (
    CampaignAction,
    CampaignAgentResponse,
    CampaignAssessment,
    CampaignCallRecord,
    CampaignCheckpoint,
    CampaignConclusion,
    CampaignPhase,
    CampaignPlan,
    CampaignToolObservation,
    CaseSupportStatus,
    campaign_plan_id,
)
from campaign.workspace import CampaignWorkspace
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
from knowledge.models import KnowledgeField
from remediation.models import FileOperation, RepositoryFileChange
from remediation.workspace import RepositoryWorkspaceError
from synthesis.client import LiteLLMClient, LiteLLMCompletion, LiteLLMConfig
from synthesis.models import TokenUsage


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_json(value: object) -> str:
    return _sha(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _grounding(case_count: int = 2) -> GroundingPackage:
    content = "BillFetchRequest requires POST /bill/fetch and response code 000."
    citation = RepositoryCitation(
        repository_id="a" * 64,
        path="docs/bill-fetch.md",
        file_sha256="b" * 64,
        line_start=1,
        line_end=1,
        excerpt_sha256=_sha(content),
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
        title="docs/bill-fetch.md:1",
        content=content,
        relevance_score=10,
        repository=citation,
    )
    cases = tuple(
        GroundedTestCase(
            context=PortalTestCaseContext(
                test_case_id=f"TC_0{index}",
                fields=(
                    KnowledgeField(
                        key="test_case_id",
                        label="TC ID",
                        value=f"TC_0{index}",
                    ),
                    KnowledgeField(
                        key="api_name",
                        label="API Name",
                        value="BillFetchRequest",
                    ),
                ),
                dependency_case_ids=("TC_01",) if index == 2 else (),
                description=f"TC_0{index} validates the bill fetch API",
                evidence_state_ids=(f"state-{index}",),
                description_state_ids=(f"detail-{index}",),
            ),
            retrieval_query=f"TC_0{index} BillFetchRequest",
            repository_snippet_ids=(snippet_id,),
        )
        for index in range(1, case_count + 1)
    )
    return GroundingPackage(
        source_run_id="fixture-run",
        source_knowledge_sha256="c" * 64,
        grounded_at="2026-08-10T00:00:00Z",
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
        test_cases=cases,
        coverage=GroundingCoverage(
            source_testcase_context_complete=True,
            mcp_required=False,
            test_cases=case_count,
            repository_grounded=case_count,
            mcp_grounded=0,
            externally_grounded=case_count,
            grounding_complete=True,
        ),
    )


class _CampaignLLM:
    def __init__(self, source: str, snippet_id: str, group_id: str) -> None:
        self._config = LiteLLMConfig(
            endpoint="https://llm.example.test",
            model="fixture-model",
            response_format="json_schema",
        )
        self.source = source
        self.snippet_id = snippet_id
        self.group_id = group_id
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
        context = json.loads(user_prompt.split("INPUT_CONTEXT=", 1)[1])
        phase = context["control"]["phase"]
        if phase == CampaignPhase.READ_CASES.value:
            response = CampaignAgentResponse(
                action=CampaignAction.READ_TEST_CASES,
                rationale="Read the related API group in one bounded request",
                group_id=self.group_id,
                test_case_ids=("TC_01", "TC_02"),
            )
        elif phase == CampaignPhase.RESOLVE_EVIDENCE.value:
            response = CampaignAgentResponse(
                action=CampaignAction.READ_EVIDENCE,
                rationale="Read the retained repository evidence",
                snippet_ids=tuple(context["control"]["target_snippet_ids"]),
            )
        elif phase == CampaignPhase.DISCOVER.value and len(self.prompts) == 2:
            response = CampaignAgentResponse(
                action=CampaignAction.READ_EVIDENCE,
                rationale="Read the shared API requirement",
                snippet_id=self.snippet_id,
            )
        elif phase == CampaignPhase.DISCOVER.value:
            response = CampaignAgentResponse(
                action=CampaignAction.SEARCH_REPOSITORY,
                query="bill_fetch",
                limit=50,
                group_id=self.group_id,
                test_case_ids=("TC_01", "TC_02"),
            )
        elif phase == CampaignPhase.ASSESS.value:
            response = CampaignAgentResponse(
                action=CampaignAction.RECORD_ASSESSMENTS,
                rationale="Both cases share the same missing handler capability",
                assessments=tuple(
                    CampaignAssessment(
                        test_case_id=f"TC_0{index}",
                        status=CaseSupportStatus.SUPPORTED_AFTER_CHANGE,
                        rationale="The route is documented but the handler is incomplete",
                        required_capabilities=("bill fetch handler",),
                        missing_capabilities=("implemented response",),
                        portal_evidence_state_ids=(f"state-{index}",),
                        evidence_snippet_ids=(self.snippet_id,),
                    )
                    for index in (1, 2)
                ),
            )
        elif (
            phase == CampaignPhase.PLAN_CHANGES.value
            and "service.py" not in context["progress"]["read_repository_paths"]
        ):
            response = CampaignAgentResponse(
                action=CampaignAction.READ_REPOSITORY_FILE,
                rationale="Inspect the existing implementation before replacement",
                path="service.py",
            )
        elif phase == CampaignPhase.PLAN_CHANGES.value:
            response = CampaignAgentResponse(
                action=CampaignAction.STAGE_FILE,
                rationale="Implement the shared capability once for both cases",
                change=RepositoryFileChange(
                    path="service.py",
                    operation=FileOperation.REPLACE,
                    expected_sha256=_sha(self.source),
                    content="def bill_fetch():\n    return {'responseCode': '000'}\n",
                    rationale="Implement the documented bill fetch behavior",
                    test_case_ids=("TC_01", "TC_02"),
                    evidence_snippet_ids=(self.snippet_id,),
                ),
            )
        elif phase == CampaignPhase.FINALIZE.value:
            response = CampaignAgentResponse(
                action=CampaignAction.FINAL,
                rationale="Every case is assessed and the shared change is staged",
                conclusion=CampaignConclusion(
                    summary="Implement bill fetch support for both certification cases",
                    verification_notes=("Run the focused service tests",),
                ),
            )
        else:
            raise AssertionError(f"unexpected campaign phase: {phase}")
        payload = response.model_dump_json()
        return LiteLLMCompletion(
            content=payload,
            request_sha256=_sha(user_prompt),
            response_sha256=_sha(payload),
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


class _ControlledLLM:
    def __init__(self, responses: list[CampaignAgentResponse]) -> None:
        self._config = LiteLLMConfig(
            endpoint="https://llm.example.test",
            model="fixture-model",
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
        response = self.responses.pop(0)
        payload = response.model_dump_json()
        return LiteLLMCompletion(
            content=payload,
            request_sha256=_sha(user_prompt),
            response_sha256=_sha(payload),
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


@pytest.mark.asyncio
async def test_campaign_reads_lazily_and_builds_one_shared_change(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    source = "def bill_fetch():\n    return None\n"
    (repository / "service.py").write_text(source)
    grounding = _grounding()
    snippet_id = grounding.snippets[0].snippet_id
    groups = campaign_groups(grounding.test_cases)
    llm = _CampaignLLM(source, snippet_id, groups[0].group_id)
    checkpoints = []

    plan = await CertificationCampaignAgent(
        grounding=grounding,
        grounding_sha256="d" * 64,
        workspace=CampaignWorkspace(repository),
        llm=llm,
        objective="Support every grounded testcase and stage required code",
    ).build(progress=checkpoints.append)

    assert len(groups) == 1
    assert len(plan.assessments) == 2
    assert plan.proposal is not None
    assert len(plan.proposal.changes) == 1
    assert plan.proposal.changes[0].test_case_ids == ("TC_01", "TC_02")
    assert plan.approval_required is True
    assert "2 supported_after_change" in plan.conclusion.summary
    assert len(checkpoints) == 8
    assert checkpoints[-1].assessments == plan.assessments
    assert "validates the bill fetch API" not in llm.prompts[0]
    assert "POST /bill/fetch" not in llm.prompts[0]
    assert (repository / "service.py").read_text() == source


@pytest.mark.asyncio
async def test_campaign_forces_assessment_after_bounded_discovery(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "service.py").write_text("def bill_fetch():\n    return None\n")
    grounding = _grounding()
    group = campaign_groups(grounding.test_cases)[0]
    assessments = tuple(
        CampaignAssessment(
            test_case_id=case.context.test_case_id,
            status=CaseSupportStatus.SUPPORTED_AS_IS,
            rationale="The repository search confirms the shared handler",
            required_capabilities=("bill fetch handler",),
            portal_evidence_state_ids=case.context.evidence_state_ids,
        )
        for case in grounding.test_cases
    )
    llm = _ControlledLLM(
        [
            CampaignAgentResponse(
                action=CampaignAction.READ_TEST_CASES,
                group_id=group.group_id,
                test_case_ids=("TC_01", "TC_02"),
            ),
            CampaignAgentResponse(
                action=CampaignAction.SEARCH_REPOSITORY,
                query="definitely absent alpha",
            ),
            CampaignAgentResponse(
                action=CampaignAction.SEARCH_REPOSITORY,
                query="definitely absent beta",
            ),
            CampaignAgentResponse(
                action=CampaignAction.RECORD_ASSESSMENTS,
                assessments=assessments,
            ),
            CampaignAgentResponse(
                action=CampaignAction.FINAL,
                conclusion=CampaignConclusion(summary="All cases are supported"),
            ),
        ]
    )

    plan = await CertificationCampaignAgent(
        grounding=grounding,
        grounding_sha256="d" * 64,
        workspace=CampaignWorkspace(repository),
        llm=llm,
        objective="Assess support",
        config=CampaignBuildConfig(maximum_discovery_actions_per_group=2),
    ).build()

    assert len(plan.assessments) == 2
    assert '"phase":"assess"' in llm.prompts[3]
    assert '"required_action":"record_assessments"' in llm.prompts[3]
    assert len(plan.calls) == 5


@pytest.mark.asyncio
async def test_assessment_context_prioritizes_read_snippets_over_large_files(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "large.txt").write_text("x" * 25_000)
    grounding = _grounding()
    group = campaign_groups(grounding.test_cases)[0]
    agent = CertificationCampaignAgent(
        grounding=grounding,
        grounding_sha256="d" * 64,
        workspace=CampaignWorkspace(repository),
        llm=_CampaignLLM(
            "x" * 25_000,
            grounding.snippets[0].snippet_id,
            group.group_id,
        ),
        objective="Assess support",
    )
    _, _, cases_error = await agent._execute_tool(
        CampaignAgentResponse(
            action=CampaignAction.READ_TEST_CASES,
            test_case_ids=("TC_01", "TC_02"),
        )
    )
    _, _, evidence_error = await agent._execute_tool(
        CampaignAgentResponse(
            action=CampaignAction.READ_EVIDENCE,
            snippet_id=grounding.snippets[0].snippet_id,
        )
    )
    _, _, repository_error = await agent._execute_tool(
        CampaignAgentResponse(
            action=CampaignAction.READ_REPOSITORY_FILE,
            path="large.txt",
        )
    )

    context = agent._bounded_evidence_context(("TC_01", "TC_02"))

    assert cases_error is None
    assert evidence_error is None
    assert repository_error is None
    assert context["read_evidence"][0]["snippet_id"] == grounding.snippets[0].snippet_id
    assert (
        sum(
            len(item["content"])
            for label in ("read_evidence", "repository_evidence")
            for item in context[label]
        )
        <= agent.config.maximum_assessment_evidence_characters
    )


@pytest.mark.asyncio
async def test_assessment_turn_caps_cases_and_omits_duplicated_context(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    grounding = _grounding(case_count=6)
    group = campaign_groups(grounding.test_cases)[0]
    agent = CertificationCampaignAgent(
        grounding=grounding,
        grounding_sha256="d" * 64,
        workspace=CampaignWorkspace(repository),
        llm=_CampaignLLM(
            "",
            grounding.snippets[0].snippet_id,
            group.group_id,
        ),
        objective="Assess support",
        config=CampaignBuildConfig(assessment_batch_size=6),
    )
    all_ids = tuple(item.context.test_case_id for item in grounding.test_cases)
    _, _, cases_error = await agent._execute_tool(
        CampaignAgentResponse(
            action=CampaignAction.READ_TEST_CASES,
            test_case_ids=all_ids,
        )
    )
    agent._add_dynamic(grounding.snippets)
    _, _, evidence_error = await agent._execute_tool(
        CampaignAgentResponse(
            action=CampaignAction.READ_EVIDENCE,
            snippet_id=grounding.snippets[0].snippet_id,
        )
    )

    directive = agent._directive()
    agent._observations.append(
        CampaignToolObservation(
            sequence=1,
            turn=1,
            action=CampaignAction.RECORD_ASSESSMENTS,
            request="TC_01,TC_02",
            result_sha256=_sha("validation-error"),
            result_characters=16,
            error="assessment citation needs correction",
        )
    )
    prompt = agent._prompt(directive)
    context_text = prompt.split("INPUT_CONTEXT=", 1)[1].split(
        "\nOUTPUT_JSON_SCHEMA=", 1
    )[0]
    context = json.loads(context_text)

    assert cases_error is None
    assert evidence_error is None
    assert directive.phase == CampaignPhase.ASSESS
    assert directive.target_test_case_ids == ("TC_01", "TC_02")
    assert "available_evidence" not in context
    assert "recent_tool_results" not in context
    assert "recent_errors" not in context
    assert "completed_action_keys" not in context["progress"]
    assert context["assessment_context"]["repository_evidence"] == []
    assert len(context["assessment_context"]["test_cases"]) == 2
    assert (
        context["assessment_context"]["last_validation_error"]
        == "assessment citation needs correction"
    )


@pytest.mark.asyncio
async def test_assessment_accepts_path_backed_by_its_cited_read_snippet(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    grounding = _grounding()
    group = campaign_groups(grounding.test_cases)[0]
    snippet_id = grounding.snippets[0].snippet_id
    agent = CertificationCampaignAgent(
        grounding=grounding,
        grounding_sha256="d" * 64,
        workspace=CampaignWorkspace(repository),
        llm=_CampaignLLM("", snippet_id, group.group_id),
        objective="Assess support",
    )
    await agent._execute_tool(
        CampaignAgentResponse(
            action=CampaignAction.READ_TEST_CASES,
            test_case_ids=("TC_01", "TC_02"),
        )
    )
    await agent._execute_tool(
        CampaignAgentResponse(
            action=CampaignAction.READ_EVIDENCE,
            snippet_id=snippet_id,
        )
    )
    base = CampaignAssessment(
        test_case_id="TC_01",
        status=CaseSupportStatus.SUPPORTED_AS_IS,
        rationale="The cited repository snippet confirms the route",
        required_capabilities=("bill fetch handler",),
        portal_evidence_state_ids=("state-1",),
        evidence_snippet_ids=(snippet_id,),
        repository_paths=("docs/bill-fetch.md",),
    )

    _, _, invalid_error = await agent._execute_tool(
        CampaignAgentResponse(
            action=CampaignAction.RECORD_ASSESSMENTS,
            assessments=(
                base.model_copy(update={"repository_paths": ("service.py",)}),
            ),
        )
    )
    _, cited_snippets, valid_error = await agent._execute_tool(
        CampaignAgentResponse(
            action=CampaignAction.RECORD_ASSESSMENTS,
            assessments=(base,),
        )
    )

    assert invalid_error is not None
    assert "full read or a cited read snippet" in invalid_error
    assert valid_error is None
    assert cited_snippets == (snippet_id,)
    assert agent._assessments["TC_01"].repository_paths == ("docs/bill-fetch.md",)


@pytest.mark.asyncio
async def test_campaign_stops_repeated_no_progress_actions(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    grounding = _grounding()
    group = campaign_groups(grounding.test_cases)[0]
    read_cases = CampaignAgentResponse(
        action=CampaignAction.READ_TEST_CASES,
        group_id=group.group_id,
        test_case_ids=("TC_01", "TC_02"),
    )
    repeated_search = CampaignAgentResponse(
        action=CampaignAction.SEARCH_REPOSITORY,
        query="bill fetch handler",
    )
    llm = _ControlledLLM([read_cases, *([repeated_search] * 4)])
    checkpoints: list[CampaignCheckpoint] = []

    with pytest.raises(CampaignBuildError, match="3 consecutive no-progress"):
        await CertificationCampaignAgent(
            grounding=grounding,
            grounding_sha256="d" * 64,
            workspace=CampaignWorkspace(repository),
            llm=llm,
            objective="Assess support",
            config=CampaignBuildConfig(maximum_no_progress_turns=3),
        ).build(progress=checkpoints.append)

    assert len(llm.prompts) == 5
    assert checkpoints[-1].no_progress_turns == 3
    assert checkpoints[-1].phase == CampaignPhase.DISCOVER
    assert any(
        "duplicate campaign action rejected" in item for item in checkpoints[-1].errors
    )
    assert checkpoints[-1].errors[-1].startswith("campaign stopped after 3")


@pytest.mark.asyncio
async def test_legacy_loop_checkpoint_recovers_directly_into_assessment(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    source = "def bill_fetch():\n    return {'responseCode': '000'}\n"
    (repository / "service.py").write_text(source)
    grounding = _grounding()
    selected = tuple(item.context.test_case_id for item in grounding.test_cases)
    workspace = CampaignWorkspace(repository)
    config = CampaignBuildConfig()
    llm_config = LiteLLMConfig(
        endpoint="https://llm.example.test",
        model="fixture-model",
        response_format="json_schema",
    )
    legacy_llm = LiteLLMClient(llm_config)
    objective = "Assess support"
    legacy_hash = _legacy_campaign_configuration_sha256(
        objective,
        selected,
        config,
        legacy_llm,
    )
    calls = tuple(
        CampaignCallRecord(
            turn=index,
            action=CampaignAction.SEARCH_REPOSITORY,
            request_sha256=_sha(f"request-{index}"),
            response_sha256=_sha(f"response-{index}"),
            valid=True,
        )
        for index in range(1, 13)
    )
    observations = tuple(
        CampaignToolObservation(
            sequence=index,
            turn=index,
            action=CampaignAction.SEARCH_REPOSITORY,
            request=f"query-{index}",
            result_sha256=_sha(f"result-{index}"),
            result_characters=10,
        )
        for index in range(1, 13)
    )
    checkpoint = CampaignCheckpoint(
        source_run_id=grounding.source_run_id,
        source_grounding_sha256="d" * 64,
        repository_id=workspace.repository_id,
        configuration_sha256=legacy_hash,
        model="fixture-model",
        objective=objective,
        updated_at="2026-08-10T00:00:00Z",
        selected_test_case_ids=selected,
        read_test_case_ids=selected,
        read_repository_paths=("service.py",),
        calls=calls,
        observations=observations,
    )
    assessments = tuple(
        CampaignAssessment(
            test_case_id=case.context.test_case_id,
            status=CaseSupportStatus.SUPPORTED_AS_IS,
            rationale="The existing shared handler supports this case",
            required_capabilities=("bill fetch handler",),
            repository_paths=("service.py",),
            portal_evidence_state_ids=case.context.evidence_state_ids,
        )
        for case in grounding.test_cases
    )
    llm = _ControlledLLM(
        [
            CampaignAgentResponse(
                action=CampaignAction.RECORD_ASSESSMENTS,
                assessments=assessments,
            ),
            CampaignAgentResponse(
                action=CampaignAction.FINAL,
                conclusion=CampaignConclusion(summary="Recovered campaign"),
            ),
        ]
    )

    plan = await CertificationCampaignAgent(
        grounding=grounding,
        grounding_sha256="d" * 64,
        workspace=workspace,
        llm=llm,
        objective=objective,
        config=config,
    ).build(initial=checkpoint)

    assert '"phase":"assess"' in llm.prompts[0]
    assert '"required_action":"record_assessments"' in llm.prompts[0]
    assert "def bill_fetch" in llm.prompts[0]
    assert len(plan.assessments) == 2
    assert len(plan.calls) == 14


def test_campaign_workspace_stages_without_mutating_and_rejects_stale_hash(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    source = "value = 1\n"
    (repository / "settings.py").write_text(source)
    workspace = CampaignWorkspace(repository)
    workspace.read_file("settings.py")
    change = RepositoryFileChange(
        path="settings.py",
        operation=FileOperation.REPLACE,
        expected_sha256=_sha(source),
        content="value = 2\n",
        rationale="Support the selected behavior",
        test_case_ids=("TC_01",),
        evidence_snippet_ids=("e" * 64,),
    )

    workspace.stage(change)

    assert (repository / "settings.py").read_text() == source
    assert workspace.read_file("settings.py")[0] == "value = 2"

    stale_workspace = CampaignWorkspace(repository)
    stale_workspace.read_file("settings.py")
    with pytest.raises(RepositoryWorkspaceError, match="digest is stale"):
        stale_workspace.stage(change.model_copy(update={"expected_sha256": "0" * 64}))


def test_campaign_resume_archive_is_content_addressed_and_idempotent(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    progress = tmp_path / "progress.log"
    plan = tmp_path / "plan.json"
    checkpoint.write_text('{"saved":true}\n')
    progress.write_text("one completed turn\n")
    plan.write_text('{"planned":true}\n')

    first = archive_checkpoint_attempt(checkpoint, progress, plan)
    second = archive_checkpoint_attempt(checkpoint, progress, plan)

    assert first == second
    assert (first / "checkpoint.json").read_bytes() == checkpoint.read_bytes()
    assert (first / "progress.log").read_bytes() == progress.read_bytes()
    assert (first / "plan.json").read_bytes() == plan.read_bytes()


@pytest.mark.asyncio
async def test_large_repository_files_require_safe_bounded_range_reads(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    large_content = "".join(
        f"line_{index:04d} = '{('x' * 180)}'\n" for index in range(1, 501)
    )
    (repository / "large.py").write_text(large_content)
    grounding = _grounding()
    group = campaign_groups(grounding.test_cases)[0]
    workspace = CampaignWorkspace(repository)
    agent = CertificationCampaignAgent(
        grounding=grounding,
        grounding_sha256="d" * 64,
        workspace=workspace,
        llm=_CampaignLLM(
            large_content,
            grounding.snippets[0].snippet_id,
            group.group_id,
        ),
        objective="Inspect safely",
    )

    full_result, _, full_error = await agent._execute_tool(
        CampaignAgentResponse(
            action=CampaignAction.READ_REPOSITORY_FILE,
            path="large.py",
        )
    )
    range_result, _, range_error = await agent._execute_tool(
        CampaignAgentResponse(
            action=CampaignAction.READ_REPOSITORY_FILE,
            path="large.py",
            line_start=1,
            line_end=100,
        )
    )

    assert full_error is not None
    assert "retry read_repository_file" in full_result
    assert range_error is None
    assert json.loads(range_result)["replacement_allowed"] is False
    assert "large.py" not in workspace.read_paths


def test_campaign_action_contract_rejects_mixed_tool_payloads() -> None:
    with pytest.raises(ValueError, match="invalid fields"):
        CampaignAgentResponse(
            action=CampaignAction.SEARCH_REPOSITORY,
            rationale="Search for the handler",
            query="bill fetch handler",
            path="service.py",
        )


def test_read_evidence_accepts_one_bounded_batch_and_rejects_ambiguity() -> None:
    first = "a" * 64
    second = "b" * 64
    response = CampaignAgentResponse(
        action=CampaignAction.READ_EVIDENCE,
        snippet_ids=(first, second),
    )

    assert response.snippet_ids == (first, second)

    with pytest.raises(ValueError, match="lacks its required payload"):
        CampaignAgentResponse(
            action=CampaignAction.READ_EVIDENCE,
            snippet_id=first,
            snippet_ids=(second,),
        )
    with pytest.raises(ValueError, match="must be unique"):
        CampaignAgentResponse(
            action=CampaignAction.READ_EVIDENCE,
            snippet_ids=(first, first),
        )


def test_reassessment_reopens_only_needs_review_cases() -> None:
    assessments = (
        CampaignAssessment(
            test_case_id="TC_01",
            status=CaseSupportStatus.NEEDS_REVIEW,
            rationale="More repository evidence is required",
            required_capabilities=("bill fetch handler",),
        ),
        CampaignAssessment(
            test_case_id="TC_02",
            status=CaseSupportStatus.SUPPORTED_AS_IS,
            rationale="The existing handler supports this case",
            required_capabilities=("bill fetch handler",),
        ),
    )
    checkpoint = CampaignCheckpoint(
        source_run_id="fixture-run",
        source_grounding_sha256="c" * 64,
        repository_id="a" * 64,
        configuration_sha256="d" * 64,
        model="fixture-model",
        objective="Assess support",
        updated_at="2026-08-10T00:00:00Z",
        selected_test_case_ids=("TC_01", "TC_02"),
        phase=CampaignPhase.FINALIZE,
        no_progress_turns=2,
        read_test_case_ids=("TC_01", "TC_02"),
        assessments=assessments,
    )

    reopened = _reopen_needs_review(checkpoint)

    assert tuple(item.test_case_id for item in reopened.assessments) == ("TC_02",)
    assert reopened.phase is None
    assert reopened.no_progress_turns == 0
    assert reopened.read_test_case_ids == checkpoint.read_test_case_ids
    assert "TC_01" in reopened.errors[-1]

    resumed = _reopen_needs_review(
        reopened,
        source_plan=CampaignPlan.model_construct(assessments=assessments),
    )
    assert resumed.assessments == reopened.assessments
    assert resumed.errors == reopened.errors


def test_read_testcases_accepts_optional_group_context() -> None:
    response = CampaignAgentResponse(
        action=CampaignAction.READ_TEST_CASES,
        rationale="Read one bounded API group",
        group_id="group-001-fixture",
        test_case_ids=("TC_01", "TC_02"),
    )

    assert response.group_id == "group-001-fixture"


def test_search_accepts_and_locally_bounds_model_scope() -> None:
    response = CampaignAgentResponse(
        action=CampaignAction.SEARCH_REPOSITORY,
        query="session registration route",
        limit=50,
        group_id="group-001-fixture",
        test_case_ids=("TC_01", "TC_02"),
    )

    assert response.limit == 50
    assert response.rationale == "Model selected this bounded campaign action."


def test_repository_range_contract_requires_complete_ordered_bounds() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        CampaignAgentResponse(
            action=CampaignAction.READ_REPOSITORY_FILE,
            path="large.py",
            line_start=1,
        )

    response = CampaignAgentResponse(
        action=CampaignAction.READ_REPOSITORY_FILE,
        path="large.py",
        line_start=200,
        line_end=400,
    )
    assert response.line_start == 200
    assert response.line_end == 400


def test_complete_supported_campaign_needs_no_repository_approval() -> None:
    grounding = _grounding()
    assessments = tuple(
        CampaignAssessment(
            test_case_id=case.context.test_case_id,
            status=CaseSupportStatus.SUPPORTED_AS_IS,
            rationale="The existing implementation supports the requirement",
            required_capabilities=("bill fetch handler",),
            portal_evidence_state_ids=case.context.evidence_state_ids,
        )
        for case in grounding.test_cases
    )
    identifiers = {
        "source_run_id": grounding.source_run_id,
        "source_grounding_sha256": "d" * 64,
        "repository_id": "e" * 64,
        "objective": "Assess support",
        "selected_test_case_ids": tuple(
            item.context.test_case_id for item in grounding.test_cases
        ),
        "assessments": assessments,
        "proposal": None,
    }

    plan = CampaignPlan(
        plan_id=campaign_plan_id(**identifiers),
        groups=campaign_groups(grounding.test_cases),
        generated_at="2026-08-10T00:00:00Z",
        model="fixture-model",
        prompt_sha256="f" * 64,
        conclusion=CampaignConclusion(summary="All cases are already supported"),
        observations=(),
        calls=(),
        approval_required=False,
        **identifiers,
    )

    assert plan.proposal is None
    assert plan.approval_required is False

    previous_payload = plan.model_dump(mode="json")
    previous_payload["plan_id"] = campaign_plan_id(
        **identifiers,
        prompt_version="1.1",
    )
    assert (
        CampaignPlan.model_validate(previous_payload).plan_id
        == previous_payload["plan_id"]
    )
