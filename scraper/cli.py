"""Safe command-line entrypoint for deterministic portal crawling."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import NoReturn, Optional, Sequence, cast
from urllib.parse import urlsplit, urlunsplit

from pydantic import ValidationError

from scraper.browser import BrowserName
from scraper.explorer import ExplorerConfig
from scraper.guidance import CrawlStrategy, ParallelSessionMode
from scraper.models import CapturePolicy, CrawlCompletionGoal, CrawlLimits
from scraper.runner import (
    AuthenticationMode,
    CrawlRequest,
    CrawlRunner,
    CrawlRunnerError,
    ExistingRunPolicy,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without causing browser or filesystem side effects."""

    parser = argparse.ArgumentParser(
        prog="cz-crawl",
        description="Capture deterministic evidence from a generic CZ portal.",
    )
    parser.add_argument("--url", required=True, help="absolute portal start URL")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/crawls"),
        help="directory beneath which isolated crawl runs are created",
    )
    parser.add_argument("--run-id", help="stable run identifier")
    parser.add_argument(
        "--existing-run",
        choices=[item.value for item in ExistingRunPolicy],
        default=ExistingRunPolicy.ERROR.value,
    )
    parser.add_argument(
        "--auth",
        choices=[
            AuthenticationMode.NONE.value,
            AuthenticationMode.STORAGE_STATE.value,
            AuthenticationMode.MANUAL.value,
        ],
        default=AuthenticationMode.NONE.value,
    )
    parser.add_argument(
        "--storage-state",
        type=Path,
        help="existing Playwright storage-state JSON (path is never persisted)",
    )
    parser.add_argument(
        "--save-storage-state",
        type=Path,
        help=(
            "explicit sensitive output for reusable Playwright session state; "
            "written with owner-only permissions and never added to crawl evidence"
        ),
    )
    parser.add_argument(
        "--overwrite-storage-state",
        action="store_true",
        help="allow replacing an existing --save-storage-state file",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="show the browser; required for manual authentication",
    )
    parser.add_argument(
        "--ready-selector",
        help="caller-supplied selector that marks the authenticated portal ready",
    )
    parser.add_argument(
        "--strategy",
        choices=[item.value for item in CrawlStrategy],
        help=(
            "exhaustive discovery, guide-only replay, or guide-first discovery; "
            "defaults to hybrid when a guide/teaching output is supplied"
        ),
    )
    parser.add_argument(
        "--crawl-guide",
        type=Path,
        help="strict JSON guide to replay after authentication",
    )
    parser.add_argument(
        "--teach-guide",
        type=Path,
        help="record post-login operator clicks into this reusable JSON guide",
    )
    parser.add_argument(
        "--overwrite-taught-guide",
        action="store_true",
        help="allow replacing an existing --teach-guide output",
    )
    parser.add_argument(
        "--teaching-timeout-seconds",
        type=_positive_int,
        default=900,
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="isolated browser workers used for sibling branches (maximum 32)",
    )
    parser.add_argument(
        "--parallel-session-mode",
        choices=[item.value for item in ParallelSessionMode],
        help="off, probe-and-fallback, or force cloned authenticated sessions",
    )
    parser.add_argument(
        "--authentication-timeout-seconds",
        type=_positive_int,
        default=300,
    )
    parser.add_argument(
        "--allowed-origin",
        action="append",
        default=[],
        help="additional in-scope origin; repeat for multiple origins",
    )
    parser.add_argument(
        "--browser",
        choices=("chromium", "firefox", "webkit"),
        default="chromium",
    )
    parser.add_argument("--viewport-width", type=_positive_int, default=1920)
    parser.add_argument("--viewport-height", type=_positive_int, default=1080)
    parser.add_argument("--ignore-https-errors", action="store_true")
    parser.add_argument("--maximum-depth", type=_nonnegative_int, default=20)
    parser.add_argument("--maximum-states", type=_positive_int, default=10_000)
    parser.add_argument("--maximum-actions", type=_positive_int, default=100_000)
    parser.add_argument(
        "--maximum-runtime-seconds",
        type=_positive_int,
        default=14_400,
    )
    parser.add_argument(
        "--maximum-artifact-bytes",
        type=_positive_int,
        default=10_737_418_240,
    )
    parser.add_argument(
        "--completion-goal",
        choices=[item.value for item in CrawlCompletionGoal],
        default=CrawlCompletionGoal.BOUNDED_FRONTIER.value,
        help=(
            "successful completion condition; testcase_context stops after the "
            "declared testcase total, unique IDs, and descriptions reconcile"
        ),
    )
    parser.add_argument(
        "--testcase-context-stability-observations",
        type=_positive_int,
        default=2,
        help="complete captures required before testcase-context early stop",
    )
    parser.add_argument(
        "--screenshots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dom",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--accessibility-tree",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--storage-summaries",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--request-bodies",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--response-bodies",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--har",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--allow-review-actions",
        action="store_true",
        help="execute side-effect-like actions once as terminal branches",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and print a secret-safe summary without launching a browser",
    )
    return parser


def request_from_args(args: argparse.Namespace) -> CrawlRequest:
    """Translate parsed CLI flags into the typed runner boundary."""

    limits = CrawlLimits(
        maximum_depth=args.maximum_depth,
        maximum_states=args.maximum_states,
        maximum_actions=args.maximum_actions,
        maximum_runtime_seconds=args.maximum_runtime_seconds,
        maximum_artifact_bytes=args.maximum_artifact_bytes,
    )
    capture_policy = CapturePolicy(
        allow_review_required_actions=args.allow_review_actions,
        capture_dom=args.dom,
        capture_screenshots=args.screenshots,
        capture_accessibility_tree=args.accessibility_tree,
        capture_storage_state=args.storage_summaries,
        capture_request_bodies=args.request_bodies,
        capture_response_bodies=args.response_bodies,
        capture_trace=args.trace,
        capture_har=args.har,
    )
    allowed_origins: tuple[str, ...] = ()
    if args.allowed_origin:
        values = (_url_origin(args.url), *args.allowed_origin)
        allowed_origins = tuple(dict.fromkeys(values))
    strategy = (
        CrawlStrategy(args.strategy)
        if args.strategy is not None
        else (
            CrawlStrategy.HYBRID
            if args.crawl_guide is not None or args.teach_guide is not None
            else CrawlStrategy.EXHAUSTIVE
        )
    )
    parallel_mode = (
        ParallelSessionMode(args.parallel_session_mode)
        if args.parallel_session_mode is not None
        else (
            ParallelSessionMode.PROBE if args.workers > 1 else ParallelSessionMode.OFF
        )
    )
    return CrawlRequest(
        root_url=args.url,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        allowed_origins=allowed_origins,
        authentication_mode=AuthenticationMode(args.auth),
        storage_state_path=args.storage_state,
        storage_state_output_path=args.save_storage_state,
        overwrite_storage_state_output=args.overwrite_storage_state,
        existing_run_policy=ExistingRunPolicy(args.existing_run),
        browser_name=cast(BrowserName, args.browser),
        headless=not args.headed,
        viewport_width=args.viewport_width,
        viewport_height=args.viewport_height,
        ignore_https_errors=args.ignore_https_errors,
        authentication_timeout_ms=args.authentication_timeout_seconds * 1_000,
        ready_selector=args.ready_selector,
        guide_path=args.crawl_guide,
        taught_guide_output_path=args.teach_guide,
        overwrite_taught_guide=args.overwrite_taught_guide,
        teaching_timeout_ms=args.teaching_timeout_seconds * 1_000,
        limits=limits,
        capture_policy=capture_policy,
        explorer_config=ExplorerConfig(
            completion_goal=CrawlCompletionGoal(args.completion_goal),
            testcase_context_stability_observations=(
                args.testcase_context_stability_observations
            ),
            strategy=strategy,
            worker_count=args.workers,
            parallel_session_mode=parallel_mode,
        ),
    )


async def async_main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI workflow and return a process exit status."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        request = request_from_args(args)
        if args.validate_only:
            print(json.dumps(_validation_summary(request), indent=2, sort_keys=True))
            return 0
        result = await CrawlRunner(request).run()
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "run_directory": str(result.run_directory),
                    "status": result.status.value,
                    "reused_existing": result.reused_existing,
                    "completion_reason": result.exploration.completion_reason,
                    "coverage": result.exploration.coverage.model_dump(mode="json"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if result.exploration.coverage.configured_goal_complete else 2
    except (CrawlRunnerError, ValidationError, ValueError) as exc:
        print(f"cz-crawl: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            "cz-crawl: crawl failed "
            f"({type(exc).__name__}); inspect the run manifest when available",
            file=sys.stderr,
        )
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Synchronous wrapper used by ``python -m scraper`` and console scripts."""

    return asyncio.run(async_main(argv))


def entrypoint() -> NoReturn:
    """Installed console-script boundary."""

    raise SystemExit(main())


def _validation_summary(request: CrawlRequest) -> dict[str, object]:
    return {
        "valid": True,
        "root_target": _query_free_url(request.root_url),
        "artifact_root": str(request.artifact_root),
        "run_id": request.run_id or "<auto-generated>",
        "allowed_origins": request.effective_allowed_origins,
        "authentication_mode": request.authentication_mode.value,
        "storage_state_configured": request.storage_state_path is not None,
        "storage_state_output_configured": (
            request.storage_state_output_path is not None
        ),
        "headless": request.headless,
        "browser": request.browser_name,
        "ready_selector_configured": request.ready_selector is not None,
        "existing_run_policy": request.existing_run_policy.value,
        "limits": request.limits.model_dump(mode="json"),
        "completion_goal": request.explorer_config.completion_goal.value,
        "testcase_context_stability_observations": (
            request.explorer_config.testcase_context_stability_observations
        ),
        "strategy": request.explorer_config.strategy.value,
        "crawl_guide_configured": request.guide_path is not None,
        "teaching_configured": request.taught_guide_output_path is not None,
        "worker_count": request.explorer_config.worker_count,
        "parallel_session_mode": (request.explorer_config.parallel_session_mode.value),
        "capture_policy": request.capture_policy.model_dump(mode="json"),
    }


def _query_free_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _url_origin(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


__all__ = [
    "async_main",
    "build_parser",
    "entrypoint",
    "main",
    "request_from_args",
]
