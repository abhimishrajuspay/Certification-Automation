"""CLI for lazy repository-wide certification campaigns."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
import sys
from pathlib import Path
from typing import Optional, Sequence

from pydantic import SecretStr

from campaign.builder import (
    CampaignBuildConfig,
    CampaignBuildError,
    CertificationCampaignAgent,
    campaign_groups,
)
from campaign.io import (
    CampaignIOError,
    archive_checkpoint_attempt,
    export_apply_report,
    export_plan,
    export_support_matrix,
    load_checkpoint,
    load_plan,
    save_checkpoint,
)
from campaign.models import (
    CampaignAssessment,
    CampaignCheckpoint,
    CampaignPlan,
    CaseSupportStatus,
)
from campaign.workspace import CampaignWorkspace
from grounding.mcp import DEFAULT_READ_ONLY_TOOLS, MCPClient, MCPClientConfig
from grounding.models import GroundedTestCase
from remediation.models import ApplyStatus
from remediation.workspace import RepositoryWorkspace, RepositoryWorkspaceError
from synthesis.builder import load_grounding
from synthesis.client import LiteLLMClient, LiteLLMConfig


LOGGER = logging.getLogger("cz.campaign")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cz-campaign",
        description=(
            "Run one lazy LLM campaign across grounded testcases, repository code, "
            "and MCP knowledge"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser(
        "inspect",
        help="show deterministic semantic groups without calling an LLM",
    )
    inspect.add_argument("--grounding", type=Path, required=True)
    inspect.add_argument("--test-case", action="append", default=[])
    inspect.add_argument("--skip-source-manifest-check", action="store_true")

    plan = subparsers.add_parser(
        "plan",
        help="assess all selected cases and stage a reviewed code plan",
    )
    plan.add_argument("--grounding", type=Path, required=True)
    plan.add_argument("--repo-path", type=Path, required=True)
    plan.add_argument("--objective", required=True)
    plan.add_argument(
        "--test-case",
        action="append",
        default=[],
        help="optional testcase filter; omitted means every grounded testcase",
    )
    plan.add_argument("--litellm-url", default=os.environ.get("CZ_LITELLM_URL"))
    plan.add_argument("--model", default=os.environ.get("CZ_LITELLM_MODEL"))
    plan.add_argument("--api-key-env", default="CZ_LITELLM_API_KEY")
    plan.add_argument("--no-api-key", action="store_true")
    plan.add_argument(
        "--response-format",
        choices=("json_schema", "json_object"),
        default="json_object",
    )
    plan.add_argument("--mcp-url", default=os.environ.get("CZ_MCP_URL"))
    plan.add_argument(
        "--mcp-tool",
        action="append",
        default=[],
        help="allowlisted read-only MCP search tool; repeatable",
    )
    plan.add_argument("--maximum-turns", type=int, default=160)
    plan.add_argument("--assessment-batch-size", type=int, default=6)
    plan.add_argument(
        "--maximum-evidence-snippets-per-batch",
        type=int,
        default=6,
    )
    plan.add_argument(
        "--maximum-discovery-actions-per-group",
        type=int,
        default=12,
    )
    plan.add_argument("--maximum-change-actions", type=int, default=12)
    plan.add_argument("--maximum-no-progress-turns", type=int, default=3)
    plan.add_argument("--maximum-output-tokens", type=int, default=20_000)
    plan.add_argument("--timeout-seconds", type=float, default=180.0)
    plan.add_argument("--maximum-transport-attempts", type=int, default=3)
    plan.add_argument("--retry-backoff-seconds", type=float, default=15.0)
    plan.add_argument("--output-root", type=Path, default=Path("artifacts/campaign"))
    plan.add_argument("--resume", action="store_true")
    plan.add_argument(
        "--reassess-needs-review",
        action="store_true",
        help=("with --resume, reopen needs_review cases and resolve retained evidence"),
    )
    plan.add_argument("--overwrite", action="store_true")
    plan.add_argument("--skip-source-manifest-check", action="store_true")

    status = subparsers.add_parser("status", help="show saved campaign progress")
    source = status.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id")
    source.add_argument("--campaign", type=Path)
    status.add_argument("--output-root", type=Path, default=Path("artifacts/campaign"))

    apply = subparsers.add_parser(
        "apply",
        help="verify an approved campaign change set in isolation, then apply it",
    )
    apply.add_argument("--plan", type=Path, required=True)
    apply.add_argument("--repo-path", type=Path, required=True)
    apply.add_argument("--approve-plan-id", required=True)
    apply.add_argument(
        "--verify-command",
        action="append",
        default=[],
        help="command parsed to argv and executed without a shell; repeatable",
    )
    apply.add_argument("--verification-timeout-seconds", type=int, default=600)
    apply.add_argument("--allow-unverified", action="store_true")
    apply.add_argument("--report", type=Path)
    apply.add_argument("--overwrite-report", action="store_true")
    return parser


def _inspect(args: argparse.Namespace) -> int:
    loaded = load_grounding(
        args.grounding,
        verify_manifest=not args.skip_source_manifest_check,
    )
    selected = _selected_cases(loaded.grounding.test_cases, tuple(args.test_case))
    groups = campaign_groups(selected)
    print(
        json.dumps(
            {
                "source_run_id": loaded.grounding.source_run_id,
                "test_cases": len(selected),
                "semantic_groups": len(groups),
                "groups": [item.model_dump(mode="json") for item in groups],
                "llm_calls": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


async def _plan(args: argparse.Namespace) -> int:
    loaded = load_grounding(
        args.grounding,
        verify_manifest=not args.skip_source_manifest_check,
    )
    selected_cases = _selected_cases(
        loaded.grounding.test_cases,
        tuple(args.test_case),
    )
    selected_ids = tuple(item.context.test_case_id for item in selected_cases)
    output_directory = args.output_root / loaded.grounding.source_run_id
    checkpoint_path = output_directory / "checkpoint.json"
    plan_path = output_directory / "plan.json"
    matrix_path = output_directory / "support_matrix.jsonl"
    progress_path = output_directory / "progress.log"
    if args.reassess_needs_review and not args.resume:
        raise ValueError("--reassess-needs-review requires --resume")
    if args.resume and plan_path.is_file() and not args.reassess_needs_review:
        existing = load_plan(plan_path)
        current_repository_id = CampaignWorkspace(args.repo_path).repository_id
        if (
            existing.source_run_id != loaded.grounding.source_run_id
            or existing.source_grounding_sha256 != loaded.sha256
            or existing.repository_id != current_repository_id
            or existing.objective != args.objective.strip()
            or existing.selected_test_case_ids != selected_ids
        ):
            raise CampaignIOError(
                "existing campaign plan differs from the requested campaign"
            )
        print(
            json.dumps(
                _planned_payload(
                    existing,
                    plan_path=plan_path,
                    matrix_path=matrix_path,
                    checkpoint_path=checkpoint_path,
                    progress_path=progress_path,
                    reused_existing=True,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not args.litellm_url:
        raise ValueError("LiteLLM URL is required")
    if not args.model:
        raise ValueError("LiteLLM model is required")
    api_key_value = None if args.no_api_key else os.environ.get(args.api_key_env)
    if not args.no_api_key and not api_key_value:
        raise ValueError(
            f"LiteLLM API key environment variable {args.api_key_env!r} is not set"
        )
    initial = load_checkpoint(checkpoint_path) if args.resume else None
    archived_attempt = (
        archive_checkpoint_attempt(checkpoint_path, progress_path, plan_path)
        if initial is not None
        else None
    )
    if args.reassess_needs_review:
        assert initial is not None
        if not plan_path.is_file():
            raise CampaignIOError(
                "--reassess-needs-review requires a completed campaign plan"
            )
        initial = _reopen_needs_review(initial, source_plan=load_plan(plan_path))
        save_checkpoint(initial, checkpoint_path)
    _configure_logging(progress_path)
    if archived_attempt is not None:
        LOGGER.info("campaign attempt archived path=%s", archived_attempt)

    allowed_tools = tuple(args.mcp_tool) or DEFAULT_READ_ONLY_TOOLS
    mcp = (
        MCPClient(
            MCPClientConfig(
                endpoint=args.mcp_url,
                allowed_tools=tuple(dict.fromkeys(allowed_tools)),
            )
        )
        if args.mcp_url
        else None
    )
    llm = LiteLLMClient(
        LiteLLMConfig(
            endpoint=args.litellm_url,
            model=args.model,
            api_key=SecretStr(api_key_value) if api_key_value else None,
            timeout_seconds=args.timeout_seconds,
            maximum_transport_attempts=args.maximum_transport_attempts,
            retry_backoff_seconds=args.retry_backoff_seconds,
            maximum_output_tokens=args.maximum_output_tokens,
            response_format=args.response_format,
        )
    )
    if not args.resume and checkpoint_path.exists() and not args.overwrite:
        raise CampaignIOError(
            "campaign checkpoint already exists; use --resume or --overwrite"
        )
    agent = CertificationCampaignAgent(
        grounding=loaded.grounding,
        grounding_sha256=loaded.sha256,
        workspace=CampaignWorkspace(args.repo_path),
        llm=llm,
        objective=args.objective,
        selected_test_case_ids=selected_ids,
        mcp=mcp,
        config=CampaignBuildConfig(
            maximum_turns=args.maximum_turns,
            assessment_batch_size=args.assessment_batch_size,
            maximum_evidence_snippets_per_batch=(
                args.maximum_evidence_snippets_per_batch
            ),
            maximum_discovery_actions_per_group=(
                args.maximum_discovery_actions_per_group
            ),
            maximum_change_actions=args.maximum_change_actions,
            maximum_no_progress_turns=args.maximum_no_progress_turns,
        ),
    )
    result = await agent.build(
        initial=initial,
        progress=lambda item: save_checkpoint(item, checkpoint_path),
    )
    export_plan(result, plan_path, overwrite=args.overwrite or args.resume)
    export_support_matrix(
        result,
        matrix_path,
        overwrite=args.overwrite or args.resume,
    )
    print(
        json.dumps(
            _planned_payload(
                result,
                plan_path=plan_path,
                matrix_path=matrix_path,
                checkpoint_path=checkpoint_path,
                progress_path=progress_path,
                reused_existing=False,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _status(args: argparse.Namespace) -> int:
    directory = args.campaign if args.campaign else args.output_root / args.run_id
    assert directory is not None
    plan_path = directory / "plan.json"
    checkpoint_path = directory / "checkpoint.json"
    plan = load_plan(plan_path) if plan_path.is_file() else None
    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path.is_file() else None
    reassessment_active = (
        plan is not None
        and checkpoint is not None
        and (
            len(checkpoint.calls) > len(plan.calls)
            or len(checkpoint.assessments) < len(plan.assessments)
        )
    )
    if plan is not None and not reassessment_active:
        payload = {
            "status": "planned",
            "source_run_id": plan.source_run_id,
            "plan_id": plan.plan_id,
            "test_cases": len(plan.selected_test_case_ids),
            "assessed": len(plan.assessments),
            "support": _status_counts(plan.assessments),
            "staged_files": len(plan.proposal.changes) if plan.proposal else 0,
            "approval_required": plan.approval_required,
            "agent_turns": len(plan.calls),
            "tool_calls": len(plan.observations),
        }
    elif checkpoint is not None:
        payload = {
            "status": "checkpointed",
            "source_run_id": checkpoint.source_run_id,
            "updated_at": checkpoint.updated_at.isoformat(),
            "test_cases": len(checkpoint.selected_test_case_ids),
            "read": len(checkpoint.read_test_case_ids),
            "assessed": len(checkpoint.assessments),
            "support": _status_counts(checkpoint.assessments),
            "staged_files": len(checkpoint.staged_changes),
            "phase": (
                checkpoint.phase.value
                if checkpoint.phase is not None
                else (
                    "reassessment_pending"
                    if any(
                        item.startswith("reopened needs_review cases")
                        for item in checkpoint.errors
                    )
                    else "legacy_recovery"
                )
            ),
            "no_progress_turns": checkpoint.no_progress_turns,
            "agent_turns": len(checkpoint.calls),
            "tool_calls": len(checkpoint.observations),
            "errors": list(checkpoint.errors),
            "resume_command_hint": (
                "rerun the original cz-campaign plan command with --resume"
            ),
        }
    else:
        raise CampaignIOError(f"no campaign checkpoint or plan exists in {directory}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _apply(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    if plan.proposal is None:
        print(
            json.dumps(
                {
                    "status": "not_required",
                    "plan_id": plan.plan_id,
                    "files_applied": 0,
                    "message": "campaign found no repository changes to approve",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    commands = tuple(_command_argv(item) for item in args.verify_command)
    workspace = RepositoryWorkspace(args.repo_path)
    report = workspace.apply_changes(
        plan_id=plan.plan_id,
        repository_id=plan.repository_id,
        changes=plan.proposal.changes,
        approved_plan_id=args.approve_plan_id,
        verification_commands=commands,
        verification_timeout_seconds=args.verification_timeout_seconds,
        allow_unverified=args.allow_unverified,
    )
    report_path = args.report or args.plan.with_name("apply_report.json")
    export_apply_report(report, report_path, overwrite=args.overwrite_report)
    print(
        json.dumps(
            {
                "status": report.status.value,
                "plan_id": report.plan_id,
                "files_applied": len(report.files),
                "verification_commands": len(report.verification),
                "verification_passed": all(item.passed for item in report.verification),
                "failure_reason": report.failure_reason,
                "report_path": str(report_path.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.status == ApplyStatus.APPLIED else 2


def _selected_cases(
    cases: tuple[GroundedTestCase, ...],
    requested: tuple[str, ...],
) -> tuple[GroundedTestCase, ...]:
    if not requested:
        return cases
    requested_set = set(requested)
    available = {item.context.test_case_id for item in cases}
    unknown = requested_set - available
    if unknown:
        raise ValueError(f"unknown testcases: {sorted(unknown)}")
    return tuple(item for item in cases if item.context.test_case_id in requested_set)


def _status_counts(
    assessments: tuple[CampaignAssessment, ...],
) -> dict[str, int]:
    return {
        status.value: sum(item.status == status for item in assessments)
        for status in CaseSupportStatus
    }


def _reopen_needs_review(
    checkpoint: CampaignCheckpoint,
    *,
    source_plan: CampaignPlan | None = None,
) -> CampaignCheckpoint:
    source_assessments = (
        source_plan.assessments if source_plan is not None else checkpoint.assessments
    )
    reopened = tuple(
        item.test_case_id
        for item in source_assessments
        if item.status == CaseSupportStatus.NEEDS_REVIEW
    )
    if not reopened:
        raise CampaignIOError("campaign has no needs_review cases to reassess")
    reopened_set = set(reopened)
    retained = tuple(
        item for item in checkpoint.assessments if item.test_case_id not in reopened_set
    )
    audit_message = (
        "reopened needs_review cases for retained-evidence reassessment: "
        + ", ".join(reopened)
    )
    return checkpoint.model_copy(
        update={
            "phase": None,
            "no_progress_turns": 0,
            "assessments": retained,
            "errors": (
                checkpoint.errors
                if audit_message in checkpoint.errors
                else (*checkpoint.errors, audit_message)
            ),
        }
    )


def _planned_payload(
    plan: CampaignPlan,
    *,
    plan_path: Path,
    matrix_path: Path,
    checkpoint_path: Path,
    progress_path: Path,
    reused_existing: bool,
) -> dict[str, object]:
    return {
        "status": "planned",
        "reused_existing": reused_existing,
        "source_run_id": plan.source_run_id,
        "plan_id": plan.plan_id,
        "test_cases": len(plan.assessments),
        "semantic_groups": len(plan.groups),
        "support": _status_counts(plan.assessments),
        "file_changes": len(plan.proposal.changes) if plan.proposal else 0,
        "approval_required": plan.approval_required,
        "agent_turns": len(plan.calls),
        "tool_calls": len(plan.observations),
        "plan_path": str(plan_path.resolve()),
        "support_matrix": str(matrix_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "progress_log": str(progress_path.resolve()),
    }


def _command_argv(value: str) -> tuple[str, ...]:
    argv = tuple(shlex.split(value))
    if not argv:
        raise ValueError("verification command cannot be empty")
    return argv


def _configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.propagate = False
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)
    transport_logger = logging.getLogger("cz.synthesis.transport")
    transport_logger.setLevel(logging.INFO)
    transport_logger.handlers.clear()
    transport_logger.propagate = False
    transport_logger.addHandler(stream)
    transport_logger.addHandler(file_handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            return _inspect(args)
        if args.command == "plan":
            return asyncio.run(_plan(args))
        if args.command == "status":
            return _status(args)
        return _apply(args)
    except (
        CampaignBuildError,
        CampaignIOError,
        RepositoryWorkspaceError,
        ValueError,
    ) as exc:
        print(f"cz-campaign: {exc}", file=sys.stderr)
        return 1


def entrypoint() -> None:
    raise SystemExit(main())


__all__ = ["build_parser", "entrypoint", "main"]
