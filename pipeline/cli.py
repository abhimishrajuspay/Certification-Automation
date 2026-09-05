"""One-command driver: external CSV -> grounding -> synthesis -> Postman."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from grounding.cli import main as grounding_main
from ingest.cli import main as ingest_main
from postman.cli import main as postman_main
from synthesis.cli import main as synthesis_main


_PHASE_ORDER = ("ingest", "grounding", "synthesis", "postman")

# Phases map onto each package's own CLI so this driver never duplicates
# phase logic and always inherits the same gates, caps, and progress logging.
_PHASE_MAIN: Mapping[str, Callable[[Optional[Sequence[str]]], int]] = {
    "ingest": ingest_main,
    "grounding": grounding_main,
    "synthesis": synthesis_main,
    "postman": postman_main,
}


class PipelineError(RuntimeError):
    """Raised when the combined flow cannot be configured."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cz-flow",
        description=(
            "Run the complete external-CSV flow in one command: ingest, then"
            " repository/MCP grounding, then LLM synthesis, then Postman"
            " collection generation"
        ),
    )
    parser.add_argument("--csv", type=Path, required=True, help="source testcase CSV")
    parser.add_argument(
        "--run-id",
        help="flow run identifier (default: <csv-name>-<UTC timestamp>); reuse with "
        "--overwrite to re-do the same run",
    )
    parser.add_argument(
        "--test-case",
        action="append",
        default=[],
        help="process only these testcase IDs (repeatable); default is every CSV row",
    )
    parser.add_argument("--source-label", help="audit label for the source sheet")
    parser.add_argument(
        "--repo-path",
        type=Path,
        required=True,
        help="integration repository for grounding and synthesis searches",
    )
    parser.add_argument(
        "--repo-include-code",
        action="store_true",
        help="also index source-code files such as .hs/.java/.py",
    )
    parser.add_argument(
        "--repo-suffix",
        action="append",
        default=None,
        metavar=".EXT",
        help="extra indexed repository file extension (repeatable)",
    )
    parser.add_argument("--litellm-url", help="override CZ_LITELLM_URL")
    parser.add_argument("--model", help="override CZ_LITELLM_MODEL")
    parser.add_argument(
        "--api-key-env",
        help="environment variable containing the LiteLLM API key (default phase settings)",
    )
    parser.add_argument(
        "--no-api-key",
        action="store_true",
        help="connect to a trusted LiteLLM proxy without a key",
    )
    parser.add_argument(
        "--response-format",
        choices=("json_schema", "json_object"),
        help="structured-output mode for grounding and synthesis",
    )
    parser.add_argument("--mcp-url", help="override CZ_MCP_URL")
    parser.add_argument(
        "--no-mcp", action="store_true", help="perform repository-only grounding"
    )
    parser.add_argument(
        "--mcp-tool",
        action="append",
        default=None,
        help="allowlisted MCP search tool for synthesis (repeatable)",
    )
    parser.add_argument(
        "--allow-incomplete-grounding",
        action="store_true",
        help="continue when some testcases found no external context",
    )
    parser.add_argument(
        "--allow-incomplete-source",
        action="store_true",
        help="synthesize from incomplete grounding (diagnostics)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="render only the dependency-closed ready Postman subset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace this run's existing phase outputs in every phase",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate and print every phase plan without writing or calling models",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress phase progress on stderr"
    )
    parser.add_argument(
        "--mode",
        choices=("agent", "opencode"),
        default="agent",
        help=(
            "flow mode: 'agent' (repo-internal LLM loops, legacy default) or"
            " 'opencode' (harness-driven: run ingest only and print an"
            " OpenCode handoff bundle, no LLM builder calls)"
        ),
    )
    return parser


def derive_run_id(csv_path: Path, source_label: Optional[str], now: datetime) -> str:
    """Deterministic per-import run ID: source slug plus a UTC timestamp."""

    base = (source_label or csv_path.stem or "flow").strip() or "flow"
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-") or "flow"
    return f"{slug[:40]}-{now.strftime('%Y%m%d-%H%M%S')}"


def _phase_argv(args: argparse.Namespace, run_id: str) -> list[tuple[str, list[str]]]:
    ingest = [
        "--csv",
        str(args.csv),
        "--run-id",
        run_id,
    ]
    if args.source_label:
        ingest.extend(("--source-label", args.source_label))
    for test_case in args.test_case:
        ingest.extend(("--test-case", test_case))

    grounding = [
        "--run-id",
        run_id,
        "--repo-path",
        str(args.repo_path),
        "--strategy",
        "agentic",
    ]
    synthesis = ["--run-id", run_id, "--strategy", "agentic"]
    postman = ["--run-id", run_id]

    for phase in (grounding, synthesis):
        if args.repo_include_code:
            phase.append("--repo-include-code")
        for suffix in args.repo_suffix or []:
            phase.extend(("--repo-suffix", suffix))
        if args.litellm_url:
            phase.extend(("--litellm-url", args.litellm_url))
        if args.model:
            phase.extend(("--model", args.model))
        if args.api_key_env:
            phase.extend(("--api-key-env", args.api_key_env))
        if args.no_api_key:
            phase.append("--no-api-key")
        if args.response_format:
            phase.extend(("--response-format", args.response_format))
        if args.mcp_url:
            phase.extend(("--mcp-url", args.mcp_url))
        if args.quiet:
            phase.append("--quiet")
    if args.no_mcp:
        grounding.append("--no-mcp")
        synthesis.append("--no-mcp")
    for tool in args.mcp_tool or []:
        synthesis.extend(("--mcp-tool", tool))

    if args.allow_incomplete_grounding:
        grounding.append("--allow-incomplete-grounding")
    if args.allow_incomplete_source:
        synthesis.append("--allow-incomplete-source")
    if args.allow_partial:
        postman.append("--allow-partial")

    if args.overwrite:
        for phase in (ingest, grounding, synthesis, postman):
            phase.append("--overwrite")
    return [
        ("ingest", ingest),
        ("grounding", grounding),
        ("synthesis", synthesis),
        ("postman", postman),
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = (
        args.run_id.strip()
        if args.run_id and args.run_id.strip()
        else derive_run_id(args.csv, args.source_label, datetime.now(timezone.utc))
    )
    phases = _phase_argv(args, run_id)

    if args.plan_only:
        # Downstream plan validation needs a real package chain, which a dry run
        # must not create. Parse/validate the CSV and report the staged phases.
        ingest_name, ingest_argv = phases[0]
        ingest_code = _PHASE_MAIN[ingest_name]([*ingest_argv, "--plan-only"])
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "plan_only": True,
                    "staged_phases": [name for name, _ in phases],
                    "phases": {ingest_name: ingest_code},
                    "status": "plan_only_complete"
                    if ingest_code == 0
                    else "plan_only_failed",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return ingest_code

    if args.mode == "opencode":
        # Run deterministic ingest only, then hand off to the harness.
        # No in-repo LLM loops (grounding/synthesis/campaign/remediation)
        # participate in this mode.
        ingest_name, ingest_argv = phases[0]
        code = _PHASE_MAIN[ingest_name](ingest_argv)
        if code != 0:
            results = {
                "run_id": run_id,
                "mode": "opencode",
                "status": f"ingest failed: {code}",
            }
            print(json.dumps(results, indent=2, sort_keys=True))
            return code
        handoff = {
            "run_id": run_id,
            "mode": "opencode",
            "status": "handoff",
            "message": (
                "Ingest complete. All further judgement lives in the agent"
                " harness. Suggested phase order in the harness: grounding →"
                " synthesis → postman → live-replay. Legacy agentic builders"
                " exist only for --mode=agent; do not invoke their"
                " interactive callers here."
            ),
            "artifacts": {
                "knowledge": f"artifacts/knowledge/{run_id}",
                "ingest": f"artifacts/ingest/{run_id}",
            },
        }
        print(json.dumps(handoff, indent=2, sort_keys=True))
        return 0

    results: dict[str, object] = {"run_id": run_id, "phases": {}}
    phase_results: dict[str, int] = {}
    for name, phase_argv in phases:
        code = _PHASE_MAIN[name](phase_argv)
        phase_results[name] = code
        if code != 0:
            results["phases"] = phase_results
            results["status"] = f"stopped: {name} exited {code}"
            print(json.dumps(results, indent=2, sort_keys=True))
            return code
    results["phases"] = phase_results
    results["status"] = "flow_complete"
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


__all__ = [
    "PipelineError",
    "build_parser",
    "derive_run_id",
    "entrypoint",
    "main",
]
