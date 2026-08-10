"""Tests for repository-wide lazy certification campaigns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Type

import pytest
from pydantic import BaseModel

from campaign.builder import CertificationCampaignAgent, campaign_groups
from campaign.models import (
    CampaignAction,
    CampaignAgentResponse,
    CampaignAssessment,
    CampaignConclusion,
    CampaignPlan,
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
from synthesis.client import LiteLLMCompletion, LiteLLMConfig
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


def _grounding() -> GroundingPackage:
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
        for index in (1, 2)
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
            test_cases=2,
            repository_grounded=2,
            mcp_grounded=0,
            externally_grounded=2,
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
        turn = len(self.prompts)
        if turn == 1:
            response = CampaignAgentResponse(
                action=CampaignAction.READ_TEST_CASES,
                rationale="Read the related API group in one bounded request",
                group_id=self.group_id,
                test_case_ids=("TC_01", "TC_02"),
            )
        elif turn == 2:
            response = CampaignAgentResponse(
                action=CampaignAction.READ_EVIDENCE,
                rationale="Read the shared API requirement",
                snippet_id=self.snippet_id,
            )
        elif turn == 3:
            response = CampaignAgentResponse(
                action=CampaignAction.SEARCH_REPOSITORY,
                query="bill_fetch",
                limit=50,
                group_id=self.group_id,
                test_case_ids=("TC_01", "TC_02"),
            )
        elif turn == 4:
            response = CampaignAgentResponse(
                action=CampaignAction.READ_REPOSITORY_FILE,
                rationale="Inspect the existing implementation before replacement",
                path="service.py",
            )
        elif turn == 5:
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
                        repository_paths=("service.py",),
                        portal_evidence_state_ids=(f"state-{index}",),
                        evidence_snippet_ids=(self.snippet_id,),
                    )
                    for index in (1, 2)
                ),
            )
        elif turn == 6:
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
        else:
            response = CampaignAgentResponse(
                action=CampaignAction.FINAL,
                rationale="Every case is assessed and the shared change is staged",
                conclusion=CampaignConclusion(
                    summary="Implement bill fetch support for both certification cases",
                    verification_notes=("Run the focused service tests",),
                ),
            )
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
    assert len(checkpoints) == 7
    assert checkpoints[-1].assessments == plan.assessments
    assert "validates the bill fetch API" not in llm.prompts[0]
    assert "POST /bill/fetch" not in llm.prompts[0]
    assert (repository / "service.py").read_text() == source


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
