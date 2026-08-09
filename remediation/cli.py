"""CLI for agentic repository planning and verified application."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
from pathlib import Path
from typing import Optional, Sequence

from pydantic import SecretStr

from grounding.mcp import MCPClient, MCPClientConfig
from remediation.builder import (
    CodeRemediationAgent,
    RemediationBuildConfig,
    RemediationBuildError,
    load_remediation_sources,
)
from remediation.io import (
    RemediationIOError,
    export_apply_report,
    export_plan,
    load_plan,
)
from remediation.models import ApplyStatus
from remediation.workspace import RepositoryWorkspace, RepositoryWorkspaceError
from synthesis.client import LiteLLMClient, LiteLLMConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cz-remediate",
        description="Plan and safely apply LLM-guided repository changes",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="generate a cited, hash-locked plan")
    plan.add_argument("--synthesis", type=Path, required=True)
    plan.add_argument("--grounding", type=Path, required=True)
    plan.add_argument("--repo-path", type=Path, required=True)
    plan.add_argument("--objective", required=True)
    plan.add_argument("--test-case", action="append", required=True)
    plan.add_argument("--litellm-url", default=os.environ.get("CZ_LITELLM_URL"))
    plan.add_argument("--model", default=os.environ.get("CZ_LITELLM_MODEL"))
    plan.add_argument("--api-key-env", default="CZ_LITELLM_API_KEY")
    plan.add_argument("--no-api-key", action="store_true")
    plan.add_argument(
        "--response-format",
        choices=("json_schema", "json_object"),
        default="json_schema",
    )
    plan.add_argument("--mcp-url", default=os.environ.get("CZ_MCP_URL"))
    plan.add_argument("--maximum-turns", type=int, default=24)
    plan.add_argument("--maximum-output-tokens", type=int, default=24_000)
    plan.add_argument("--timeout-seconds", type=float, default=120.0)
    plan.add_argument("--output-root", type=Path, default=Path("artifacts/remediation"))
    plan.add_argument("--overwrite", action="store_true")
    plan.add_argument("--skip-source-manifest-check", action="store_true")

    apply = subparsers.add_parser(
        "apply", help="verify in isolation and apply one approved plan"
    )
    apply.add_argument("--plan", type=Path, required=True)
    apply.add_argument("--repo-path", type=Path, required=True)
    apply.add_argument("--approve-plan-id", required=True)
    apply.add_argument(
        "--verify-command",
        action="append",
        default=[],
        help="shell-like command parsed to argv and executed without a shell; repeatable",
    )
    apply.add_argument("--verification-timeout-seconds", type=int, default=600)
    apply.add_argument("--allow-unverified", action="store_true")
    apply.add_argument("--report", type=Path)
    apply.add_argument("--overwrite-report", action="store_true")
    return parser


async def _plan(args: argparse.Namespace) -> int:
    if not args.litellm_url:
        raise ValueError("LiteLLM URL is required")
    if not args.model:
        raise ValueError("LiteLLM model is required")
    api_key_value = None if args.no_api_key else os.environ.get(args.api_key_env)
    if not args.no_api_key and not api_key_value:
        raise ValueError(
            f"LiteLLM API key environment variable {args.api_key_env!r} is not set"
        )
    synthesis, grounding = load_remediation_sources(
        synthesis_path=args.synthesis,
        grounding_path=args.grounding,
        verify_manifests=not args.skip_source_manifest_check,
    )
    selected = tuple(dict.fromkeys(args.test_case))
    workspace = RepositoryWorkspace(args.repo_path)
    llm = LiteLLMClient(
        LiteLLMConfig(
            endpoint=args.litellm_url,
            model=args.model,
            api_key=SecretStr(api_key_value) if api_key_value else None,
            timeout_seconds=args.timeout_seconds,
            maximum_output_tokens=args.maximum_output_tokens,
            response_format=args.response_format,
        )
    )
    mcp = MCPClient(MCPClientConfig(endpoint=args.mcp_url)) if args.mcp_url else None
    plan = await CodeRemediationAgent(
        synthesis=synthesis,
        grounding=grounding,
        workspace=workspace,
        llm=llm,
        objective=args.objective,
        selected_test_case_ids=selected,
        mcp=mcp,
        config=RemediationBuildConfig(maximum_turns=args.maximum_turns),
    ).build()
    path = args.output_root / plan.source_run_id / "plan.json"
    export_plan(plan, path, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": "planned",
                "source_run_id": plan.source_run_id,
                "plan_id": plan.plan_id,
                "approval_required": plan.approval_required,
                "selected_test_cases": len(plan.selected_test_case_ids),
                "file_changes": len(plan.proposal.changes),
                "agent_turns": len(plan.calls),
                "repository_tools_used": len(plan.observations),
                "plan_path": str(path.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _apply(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    commands = tuple(_command_argv(value) for value in args.verify_command)
    workspace = RepositoryWorkspace(args.repo_path)
    report = workspace.apply(
        plan,
        approved_plan_id=args.approve_plan_id,
        verification_commands=commands,
        verification_timeout_seconds=args.verification_timeout_seconds,
        allow_unverified=args.allow_unverified,
    )
    report_path = args.report or args.plan.with_name("apply_report.json")
    export_apply_report(
        report,
        report_path,
        overwrite=args.overwrite_report,
    )
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


def _command_argv(value: str) -> tuple[str, ...]:
    argv = tuple(shlex.split(value))
    if not argv:
        raise ValueError("verification command cannot be empty")
    return argv


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            return asyncio.run(_plan(args))
        return _apply(args)
    except (
        RemediationBuildError,
        RemediationIOError,
        RepositoryWorkspaceError,
        ValueError,
    ) as exc:
        print(f"cz-remediate: {exc}")
        return 1


def entrypoint() -> None:
    raise SystemExit(main())


__all__ = ["build_parser", "entrypoint", "main"]
