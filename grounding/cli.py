"""CLI for building repository and MCP-grounded testcase context."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Optional, Sequence

from grounding.builder import (
    GroundingBuildConfig,
    GroundingBuildError,
    GroundingBuilder,
    load_knowledge,
)
from grounding.exporter import GroundingExportError, export_grounding
from grounding.mcp import DEFAULT_READ_ONLY_TOOLS, MCPClient, MCPClientConfig
from grounding.repository import (
    RepositoryIndex,
    RepositoryIndexConfig,
    RepositoryIndexError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cz-ground",
        description="Ground normalized CZ testcases in repository and MCP evidence",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--knowledge",
        type=Path,
        help="portal_knowledge.json file or its package directory",
    )
    source.add_argument(
        "--run-id",
        help="run ID beneath --knowledge-root",
    )
    parser.add_argument(
        "--knowledge-root",
        type=Path,
        default=Path("artifacts/knowledge"),
        help="normalized package root used with --run-id",
    )
    parser.add_argument(
        "--repo-path",
        type=Path,
        required=True,
        help="repository to index for schemas, templates, examples, and docs",
    )
    parser.add_argument(
        "--mcp-url",
        default=os.environ.get("CZ_MCP_URL"),
        help="MCP HTTP endpoint (or set CZ_MCP_URL)",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="perform repository-only grounding even if CZ_MCP_URL is set",
    )
    parser.add_argument(
        "--mcp-tool",
        action="append",
        choices=DEFAULT_READ_ONLY_TOOLS,
        help="read-only MCP search tool to invoke; may be repeated",
    )
    parser.add_argument(
        "--mcp-timeout-seconds",
        type=float,
        default=20.0,
        help="timeout for each MCP request",
    )
    parser.add_argument(
        "--mcp-concurrency",
        type=int,
        default=4,
        help="maximum concurrent MCP search calls",
    )
    parser.add_argument(
        "--maximum-mcp-groups",
        type=int,
        default=100,
        help="maximum deduplicated API groups queried through MCP",
    )
    parser.add_argument(
        "--results-per-source",
        type=int,
        default=3,
        help="maximum excerpts requested from each retrieval source",
    )
    parser.add_argument(
        "--maximum-repo-file-bytes",
        type=int,
        default=2_000_000,
        help="skip individual repository files larger than this",
    )
    parser.add_argument(
        "--maximum-repo-total-bytes",
        type=int,
        default=200_000_000,
        help="maximum total repository bytes indexed",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/grounding"),
        help="root for exported grounding packages",
    )
    parser.add_argument(
        "--allow-incomplete-source",
        action="store_true",
        help="diagnostically ground incomplete Phase 7 testcase context",
    )
    parser.add_argument(
        "--allow-incomplete-grounding",
        action="store_true",
        help="return success even when some testcases have no external context",
    )
    parser.add_argument(
        "--skip-knowledge-manifest-check",
        action="store_true",
        help="skip Phase 7 package hash verification when a manifest exists",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace this run's existing grounding output files",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    source = (
        args.knowledge
        if args.knowledge is not None
        else args.knowledge_root / args.run_id
    )
    tools = tuple(args.mcp_tool or DEFAULT_READ_ONLY_TOOLS)
    loaded = load_knowledge(
        source,
        verify_manifest=not args.skip_knowledge_manifest_check,
    )
    repository = RepositoryIndex.build(
        args.repo_path,
        RepositoryIndexConfig(
            maximum_file_bytes=args.maximum_repo_file_bytes,
            maximum_total_bytes=args.maximum_repo_total_bytes,
        ),
    )
    mcp_url = None if args.no_mcp else args.mcp_url
    mcp_client = (
        MCPClient(
            MCPClientConfig(
                endpoint=mcp_url,
                timeout_seconds=args.mcp_timeout_seconds,
                allowed_tools=tools,
            )
        )
        if mcp_url
        else None
    )
    package = await GroundingBuilder(
        loaded.knowledge,
        loaded.sha256,
        repository,
        mcp_client=mcp_client,
        config=GroundingBuildConfig(
            repository_results_per_case=args.results_per_source,
            mcp_results_per_tool=args.results_per_source,
            mcp_tools=tools,
            maximum_mcp_groups=args.maximum_mcp_groups,
            mcp_concurrency=args.mcp_concurrency,
            allow_incomplete_source=args.allow_incomplete_source,
        ),
    ).build()
    output = export_grounding(
        package,
        args.output_root / package.source_run_id,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "source_run_id": package.source_run_id,
                "test_cases": package.coverage.test_cases,
                "repository_files": package.repository.files_indexed,
                "repository_grounded": package.coverage.repository_grounded,
                "mcp_available": package.mcp.available,
                "mcp_retrieval_complete": package.mcp.retrieval_complete,
                "mcp_calls_succeeded": package.mcp.calls_succeeded,
                "mcp_grounded": package.coverage.mcp_grounded,
                "externally_grounded": package.coverage.externally_grounded,
                "grounding_complete": package.coverage.grounding_complete,
                "output_directory": str(output.output_directory),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not package.coverage.grounding_complete and not args.allow_incomplete_grounding:
        return 2
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except (
        GroundingBuildError,
        GroundingExportError,
        RepositoryIndexError,
        ValueError,
    ) as exc:
        print(f"cz-ground: {exc}")
        return 1


def entrypoint() -> None:
    raise SystemExit(main())


__all__ = ["build_parser", "entrypoint", "main"]
