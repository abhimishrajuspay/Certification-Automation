"""Bounded LiteLLM tool loop for cited repository change planning."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from grounding.mcp import MCPClient, MCPError
from grounding.models import GroundingSnippet
from postman.builder import LoadedSynthesis, load_synthesis
from remediation.models import (
    AgentAction,
    CodeAgentResponse,
    FileOperation,
    REMEDIATION_PROMPT_VERSION,
    RemediationCallRecord,
    RemediationPlan,
    RemediationProposal,
    ToolObservation,
    remediation_plan_id,
)
from remediation.workspace import RepositoryWorkspace, RepositoryWorkspaceError
from synthesis.builder import LoadedGrounding, load_grounding
from synthesis.client import LiteLLMCompletion, LiteLLMError, SynthesisLLM


SYSTEM_PROMPT = """You are a senior integration engineer operating in a bounded code-planning loop.
Your goal is to implement the selected certification requirements in the target repository.

Use tools before proposing changes:
- search_repository finds relevant code/config/schema excerpts.
- read_repository_file returns the complete current UTF-8 file and its SHA-256.
- search_mcp retrieves requirement details using only an advertised read-only search tool.
- final returns a complete, cited change set.

Rules:
1. Never invent repository contents. Read every file before proposing replacement.
2. A replacement must use the exact SHA-256 returned by read_repository_file.
3. Create only files that are genuinely required. Do not delete or rename files.
4. Implement production code, configuration, API routing, validation, and focused tests when required.
5. Preserve the repository's conventions and dependency direction.
6. Never hardcode credentials, tokens, cookies, private hosts, or environment secrets.
7. Every change must cite selected testcase IDs and available evidence snippet IDs.
8. Do not emit shell commands. Verification commands are chosen by the human/operator.
9. If evidence is insufficient, search/read again instead of guessing.
10. Return action=final only when all file contents are complete and internally consistent.
"""


class RemediationBuildError(RuntimeError):
    """Raised when a safe repository plan cannot be produced."""


@dataclass(frozen=True)
class RemediationBuildConfig:
    maximum_turns: int = 24
    maximum_prompt_characters: int = 240_000
    maximum_changes: int = 30
    maximum_change_bytes: int = 2_000_000
    repository_search_results: int = 8

    def __post_init__(self) -> None:
        if (
            min(
                self.maximum_turns,
                self.maximum_prompt_characters,
                self.maximum_changes,
                self.maximum_change_bytes,
                self.repository_search_results,
            )
            <= 0
        ):
            raise ValueError("remediation build limits must be positive")


class CodeRemediationAgent:
    """Let an LLM inspect bounded tools, then validate its immutable proposal."""

    def __init__(
        self,
        *,
        synthesis: LoadedSynthesis,
        grounding: LoadedGrounding,
        workspace: RepositoryWorkspace,
        llm: SynthesisLLM,
        objective: str,
        selected_test_case_ids: tuple[str, ...],
        mcp: Optional[MCPClient] = None,
        config: Optional[RemediationBuildConfig] = None,
    ) -> None:
        if synthesis.synthesis.source_run_id != grounding.grounding.source_run_id:
            raise RemediationBuildError("synthesis and grounding run IDs differ")
        if synthesis.synthesis.source_grounding_sha256 != grounding.sha256:
            raise RemediationBuildError(
                "synthesis does not cite this grounding package"
            )
        if not objective.strip():
            raise ValueError("remediation objective cannot be empty")
        if not selected_test_case_ids:
            raise ValueError("at least one testcase must be selected")
        self.synthesis = synthesis
        self.grounding = grounding
        self.workspace = workspace
        self.llm = llm
        self.objective = objective.strip()
        self.selected_ids = selected_test_case_ids
        self.mcp = mcp
        self.config = config or RemediationBuildConfig()
        self._specs = {
            item.test_case_id: item for item in synthesis.synthesis.specifications
        }
        self._cases = {
            item.context.test_case_id: item for item in grounding.grounding.test_cases
        }
        unknown = set(selected_test_case_ids) - set(self._specs) - set(self._cases)
        if unknown:
            raise RemediationBuildError(
                f"selected testcases are absent from source packages: {sorted(unknown)}"
            )
        missing_specs = set(selected_test_case_ids) - set(self._specs)
        if missing_specs:
            raise RemediationBuildError(
                f"selected testcases lack synthesis specs: {sorted(missing_specs)}"
            )
        missing_cases = set(selected_test_case_ids) - set(self._cases)
        if missing_cases:
            raise RemediationBuildError(
                f"selected testcases lack grounding context: {sorted(missing_cases)}"
            )
        self._snippets = {
            item.snippet_id: item for item in grounding.grounding.snippets
        }

    async def build(self) -> RemediationPlan:
        if self.mcp is not None and self.mcp.identity is None:
            await self.mcp.connect()
        available_snippets = dict(self._selected_snippets())
        observations: list[ToolObservation] = []
        calls: list[RemediationCallRecord] = []
        memory: list[dict[str, object]] = []
        read_files: dict[str, str] = {}

        for turn in range(1, self.config.maximum_turns + 1):
            prompt = self._prompt(memory, available_snippets)
            if len(prompt) > self.config.maximum_prompt_characters:
                raise RemediationBuildError(
                    "repository-agent context exceeded the configured prompt limit"
                )
            try:
                completion = await self.llm.complete(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    response_model=CodeAgentResponse,
                    schema_name="cz_repository_agent_turn",
                )
                response = CodeAgentResponse.model_validate_json(
                    _extract_json(completion.content)
                )
            except (LiteLLMError, ValidationError, ValueError) as exc:
                raise RemediationBuildError(
                    f"repository-agent turn failed: {exc}"
                ) from exc

            if response.action == AgentAction.FINAL:
                assert response.proposal is not None
                try:
                    self._validate_proposal(
                        response.proposal,
                        read_files=read_files,
                        available_snippet_ids=set(available_snippets),
                    )
                except ValueError as exc:
                    calls.append(
                        _call_record(turn, response.action, completion, error=str(exc))
                    )
                    memory.append(
                        {
                            "turn": turn,
                            "action": "validation_feedback",
                            "result": str(exc)[:2_000],
                        }
                    )
                    continue
                calls.append(_call_record(turn, response.action, completion))
                return self._plan(
                    response.proposal,
                    observations=tuple(observations),
                    calls=tuple(calls),
                )

            calls.append(_call_record(turn, response.action, completion))
            result, snippet_ids, error = await self._execute_tool(
                response,
                available_snippets=available_snippets,
                read_files=read_files,
            )
            request = response.query or response.path or response.action.value
            observation = ToolObservation(
                sequence=len(observations) + 1,
                action=response.action,
                request=request,
                result_sha256=hashlib.sha256(result.encode("utf-8")).hexdigest(),
                result_characters=len(result),
                evidence_snippet_ids=snippet_ids,
                error=error,
            )
            observations.append(observation)
            memory.append(
                {
                    "turn": turn,
                    "action": response.action.value,
                    "request": request,
                    "result": result,
                    "error": error,
                }
            )
        raise RemediationBuildError("repository agent exhausted its turn limit")

    async def _execute_tool(
        self,
        response: CodeAgentResponse,
        *,
        available_snippets: dict[str, GroundingSnippet],
        read_files: dict[str, str],
    ) -> tuple[str, tuple[str, ...], Optional[str]]:
        try:
            if response.action == AgentAction.SEARCH_REPOSITORY:
                assert response.query is not None
                snippets = self.workspace.search(
                    response.query,
                    limit=self.config.repository_search_results,
                )
                available_snippets.update(
                    (snippet.snippet_id, snippet) for snippet in snippets
                )
                return (
                    _json_text([_snippet_prompt(item) for item in snippets]),
                    tuple(item.snippet_id for item in snippets),
                    None,
                )
            if response.action == AgentAction.READ_REPOSITORY_FILE:
                assert response.path is not None
                content, digest = self.workspace.read_file(response.path)
                read_files[response.path] = digest
                return (
                    _json_text(
                        {
                            "path": response.path,
                            "sha256": digest,
                            "content": content,
                        }
                    ),
                    (),
                    None,
                )
            if response.action == AgentAction.SEARCH_MCP:
                if self.mcp is None:
                    raise RemediationBuildError("MCP search was not configured")
                assert response.query is not None and response.tool_name is not None
                tool_result = await self.mcp.call_tool(
                    response.tool_name,
                    {"query": response.query, "limit": 5},
                )
                snippets = self.mcp.to_snippets(tool_result)
                available_snippets.update(
                    (snippet.snippet_id, snippet) for snippet in snippets
                )
                return (
                    _json_text([_snippet_prompt(item) for item in snippets]),
                    tuple(item.snippet_id for item in snippets),
                    None,
                )
            raise RemediationBuildError("unsupported repository-agent action")
        except (MCPError, RepositoryWorkspaceError, RemediationBuildError) as exc:
            safe_error = str(exc)[:2_000]
            return _json_text({"error": safe_error}), (), safe_error

    def _validate_proposal(
        self,
        proposal: RemediationProposal,
        *,
        read_files: dict[str, str],
        available_snippet_ids: set[str],
    ) -> None:
        if len(proposal.changes) > self.config.maximum_changes:
            raise ValueError("proposal exceeds the maximum file-change count")
        total_bytes = sum(
            len(change.content.encode("utf-8")) for change in proposal.changes
        )
        if total_bytes > self.config.maximum_change_bytes:
            raise ValueError("proposal exceeds the maximum change byte budget")
        selected = set(self.selected_ids)
        for change in proposal.changes:
            if not set(change.test_case_ids).issubset(selected):
                raise ValueError(f"{change.path} cites an unselected testcase")
            if not set(change.evidence_snippet_ids).issubset(available_snippet_ids):
                raise ValueError(f"{change.path} cites unavailable evidence")
            if change.operation == FileOperation.REPLACE:
                expected = read_files.get(change.path)
                if expected is None:
                    raise ValueError(f"replacement was not read first: {change.path}")
                if change.expected_sha256 != expected:
                    raise ValueError(f"replacement digest is stale: {change.path}")

    def _plan(
        self,
        proposal: RemediationProposal,
        *,
        observations: tuple[ToolObservation, ...],
        calls: tuple[RemediationCallRecord, ...],
    ) -> RemediationPlan:
        plan_id = remediation_plan_id(
            source_run_id=self.synthesis.synthesis.source_run_id,
            source_synthesis_sha256=self.synthesis.sha256,
            source_grounding_sha256=self.grounding.sha256,
            repository_id=self.workspace.repository_id,
            objective=self.objective,
            selected_test_case_ids=self.selected_ids,
            proposal=proposal,
        )
        return RemediationPlan(
            plan_id=plan_id,
            source_run_id=self.synthesis.synthesis.source_run_id,
            source_synthesis_sha256=self.synthesis.sha256,
            source_grounding_sha256=self.grounding.sha256,
            repository_id=self.workspace.repository_id,
            objective=self.objective,
            selected_test_case_ids=self.selected_ids,
            generated_at=datetime.now(timezone.utc),
            model=self.llm.config.model,
            prompt_sha256=hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            proposal=proposal,
            observations=observations,
            calls=calls,
        )

    def _prompt(
        self,
        memory: list[dict[str, object]],
        available_snippets: dict[str, GroundingSnippet],
    ) -> str:
        context = {
            "prompt_version": REMEDIATION_PROMPT_VERSION,
            "objective": self.objective,
            "selected_test_cases": [
                {
                    "execution_spec": self._specs[case_id].model_dump(mode="json"),
                    "portal_context": self._cases[case_id].context.model_dump(
                        mode="json"
                    ),
                    "grounding_limitations": list(self._cases[case_id].limitations),
                }
                for case_id in self.selected_ids
            ],
            "initial_evidence": [
                _snippet_prompt(snippet)
                for snippet in self._selected_snippets().values()
            ],
            "repository": self.workspace.index.summary.model_dump(mode="json"),
            "mcp_tools": (
                [tool.name for tool in self.mcp.tools] if self.mcp is not None else []
            ),
            "previous_tool_results": memory,
            "currently_available_evidence_snippet_ids": sorted(available_snippets),
        }
        prompt = "Choose exactly one next action.\nINPUT_CONTEXT=" + _json_text(context)
        if self.llm.config.response_format == "json_object":
            prompt += "\nOUTPUT_JSON_SCHEMA=" + _json_text(
                CodeAgentResponse.model_json_schema()
            )
        return prompt

    def _selected_snippets(self) -> dict[str, GroundingSnippet]:
        identifiers = {
            snippet_id
            for case_id in self.selected_ids
            for snippet_id in (
                *self._cases[case_id].repository_snippet_ids,
                *self._cases[case_id].mcp_snippet_ids,
            )
        }
        return {
            snippet_id: self._snippets[snippet_id] for snippet_id in sorted(identifiers)
        }


def load_remediation_sources(
    *,
    synthesis_path: Path,
    grounding_path: Path,
    verify_manifests: bool = True,
) -> tuple[LoadedSynthesis, LoadedGrounding]:
    return (
        load_synthesis(synthesis_path, verify_manifest=verify_manifests),
        load_grounding(grounding_path, verify_manifest=verify_manifests),
    )


def _call_record(
    turn: int,
    action: AgentAction,
    completion: LiteLLMCompletion,
    error: Optional[str] = None,
) -> RemediationCallRecord:
    return RemediationCallRecord(
        turn=turn,
        action=action,
        request_sha256=completion.request_sha256,
        response_sha256=completion.response_sha256,
        valid=error is None,
        validation_error=error,
        usage=completion.usage,
    )


def _snippet_prompt(snippet: GroundingSnippet) -> dict[str, object]:
    return {
        "snippet_id": snippet.snippet_id,
        "source_kind": snippet.source_kind.value,
        "title": snippet.title,
        "content": snippet.content,
        "repository": (
            snippet.repository.model_dump(mode="json")
            if snippet.repository is not None
            else None
        ),
        "mcp": snippet.mcp.model_dump(mode="json") if snippet.mcp is not None else None,
    }


def _extract_json(value: str) -> str:
    stripped = value.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    return fence.group(1) if fence else stripped


def _json_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "CodeRemediationAgent",
    "RemediationBuildConfig",
    "RemediationBuildError",
    "SYSTEM_PROMPT",
    "load_remediation_sources",
]
