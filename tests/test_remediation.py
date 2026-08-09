"""Tests for bounded repository planning and isolated verified application."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Type

import pytest
from pydantic import BaseModel

from postman.builder import LoadedSynthesis
from remediation.builder import CodeRemediationAgent
from remediation.models import (
    AgentAction,
    ApplyStatus,
    CodeAgentResponse,
    FileOperation,
    RemediationPlan,
    RemediationProposal,
    RepositoryFileChange,
    remediation_plan_id,
)
from remediation.workspace import RepositoryWorkspace
from synthesis.builder import LoadedGrounding
from synthesis.client import LiteLLMCompletion, LiteLLMConfig
from synthesis.models import TokenUsage


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _plan(workspace: RepositoryWorkspace, before: str) -> RemediationPlan:
    proposal = RemediationProposal(
        summary="Implement bill fetch API configuration",
        changes=(
            RepositoryFileChange(
                path="service.py",
                operation=FileOperation.REPLACE,
                expected_sha256=_sha(before),
                content="def bill_fetch():\n    return {'status': 'ready'}\n",
                rationale="Expose the required bill fetch behavior",
                test_case_ids=("TC_01",),
                evidence_snippet_ids=("e" * 64,),
            ),
            RepositoryFileChange(
                path="config/api.yaml",
                operation=FileOperation.CREATE,
                content="bill_fetch:\n  enabled: true\n",
                rationale="Enable the new API",
                test_case_ids=("TC_01",),
                evidence_snippet_ids=("e" * 64,),
            ),
        ),
        risks=("New route must remain backward compatible",),
    )
    fields = {
        "source_run_id": "fixture-run",
        "source_synthesis_sha256": "a" * 64,
        "source_grounding_sha256": "b" * 64,
        "repository_id": workspace.repository_id,
        "objective": "Implement TC_01",
        "selected_test_case_ids": ("TC_01",),
        "proposal": proposal,
    }
    return RemediationPlan(
        plan_id=remediation_plan_id(**fields),
        generated_at="2026-08-09T00:00:00Z",
        model="fixture-model",
        prompt_sha256="c" * 64,
        observations=(),
        calls=(),
        **fields,
    )


def test_wrong_approval_or_failed_verification_never_changes_repository(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    before = "def bill_fetch():\n    return None\n"
    (repository / "service.py").write_text(before)
    workspace = RepositoryWorkspace(repository)
    plan = _plan(workspace, before)

    rejected = workspace.apply(
        plan,
        approved_plan_id="0" * 64,
        verification_commands=((sys.executable, "-c", "raise SystemExit(0)"),),
    )
    failed = workspace.apply(
        plan,
        approved_plan_id=plan.plan_id,
        verification_commands=((sys.executable, "-c", "raise SystemExit(7)"),),
    )

    assert rejected.status == ApplyStatus.REJECTED
    assert failed.status == ApplyStatus.ROLLED_BACK
    assert (repository / "service.py").read_text() == before
    assert not (repository / "config/api.yaml").exists()
    assert failed.verification[0].exit_code == 7


def test_verified_plan_is_applied_after_isolated_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    before = "def bill_fetch():\n    return None\n"
    (repository / "service.py").write_text(before)
    workspace = RepositoryWorkspace(repository)
    plan = _plan(workspace, before)
    monkeypatch.setenv("CZ_LITELLM_API_KEY", "must-not-enter-verification")
    check = (
        sys.executable,
        "-c",
        (
            "import os; from pathlib import Path; "
            "assert 'CZ_LITELLM_API_KEY' not in os.environ; "
            "assert 'ready' in Path('service.py').read_text(); "
            "assert Path('config/api.yaml').is_file()"
        ),
    )

    report = workspace.apply(
        plan,
        approved_plan_id=plan.plan_id,
        verification_commands=(check,),
    )

    assert report.status == ApplyStatus.APPLIED
    assert report.verification[0].passed is True
    assert {item.path for item in report.files} == {
        "service.py",
        "config/api.yaml",
    }
    assert "ready" in (repository / "service.py").read_text()
    assert (repository / "config/api.yaml").is_file()


class _Dump:
    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)

    def model_dump(self, *, mode: str) -> dict[str, object]:
        del mode
        return dict(self.__dict__)


class _AgentLLM:
    def __init__(self, *, file_digest: str, evidence_id: str) -> None:
        self._config = LiteLLMConfig(
            endpoint="https://llm.example.test",
            model="fixture-model",
        )
        self.file_digest = file_digest
        self.evidence_id = evidence_id
        self.turn = 0

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
        self.turn += 1
        if self.turn == 1:
            response = CodeAgentResponse(
                action=AgentAction.SEARCH_REPOSITORY,
                rationale="Locate the bill fetch implementation",
                query="bill fetch implementation",
            )
        elif self.turn == 2:
            response = CodeAgentResponse(
                action=AgentAction.READ_REPOSITORY_FILE,
                rationale="Read before replacement",
                path="service.py",
            )
        else:
            response = CodeAgentResponse(
                action=AgentAction.FINAL,
                rationale="The implementation is grounded and complete",
                proposal=RemediationProposal(
                    summary="Implement bill fetch",
                    changes=(
                        RepositoryFileChange(
                            path="service.py",
                            operation=FileOperation.REPLACE,
                            expected_sha256=self.file_digest,
                            content=(
                                "# bill fetch implementation\n"
                                "def fetch_bill():\n    return 'ready'\n"
                            ),
                            rationale="Implement the selected testcase",
                            test_case_ids=("TC_01",),
                            evidence_snippet_ids=(self.evidence_id,),
                        ),
                    ),
                ),
            )
        payload = response.model_dump_json()
        return LiteLLMCompletion(
            content=payload,
            request_sha256=_sha(f"request-{self.turn}"),
            response_sha256=_sha(payload),
            usage=TokenUsage(),
        )


@pytest.mark.asyncio
async def test_agent_searches_and_reads_before_emitting_hash_locked_plan(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    source = "# bill fetch implementation\ndef fetch_bill():\n    return None\n"
    (repository / "service.py").write_text(source)
    workspace = RepositoryWorkspace(repository)
    snippet = workspace.search("bill fetch implementation", limit=1)[0]

    context = _Dump(
        test_case_id="TC_01",
        fields=(),
        dependency_case_ids=(),
        description="Implement bill fetch",
        evidence_state_ids=("state-1",),
        description_state_ids=("detail-1",),
    )
    case = _Dump(
        context=context,
        repository_snippet_ids=(),
        mcp_snippet_ids=(),
        limitations=(),
    )
    spec = _Dump(test_case_id="TC_01", disposition="ready")
    grounding_sha = "b" * 64
    loaded_synthesis = LoadedSynthesis(
        path=tmp_path / "synthesis.json",
        sha256="a" * 64,
        synthesis=SimpleNamespace(
            source_run_id="fixture-run",
            source_grounding_sha256=grounding_sha,
            specifications=(spec,),
        ),
    )
    loaded_grounding = LoadedGrounding(
        path=tmp_path / "grounding.json",
        sha256=grounding_sha,
        grounding=SimpleNamespace(
            source_run_id="fixture-run",
            test_cases=(case,),
            snippets=(),
        ),
    )
    llm = _AgentLLM(file_digest=_sha(source), evidence_id=snippet.snippet_id)

    plan = await CodeRemediationAgent(
        synthesis=loaded_synthesis,
        grounding=loaded_grounding,
        workspace=workspace,
        llm=llm,
        objective="Implement TC_01",
        selected_test_case_ids=("TC_01",),
    ).build()

    assert [item.action for item in plan.observations] == [
        AgentAction.SEARCH_REPOSITORY,
        AgentAction.READ_REPOSITORY_FILE,
    ]
    assert plan.proposal.changes[0].expected_sha256 == _sha(source)
    assert plan.proposal.changes[0].evidence_snippet_ids == (snippet.snippet_id,)
    assert plan.approval_required is True
    assert llm.turn == 3
