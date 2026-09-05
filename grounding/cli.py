"""CLI for building repository and MCP-grounded testcase context."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Optional, Sequence

from pydantic import SecretStr

from grounding.agent import (
    AgenticGroundingBuilder,
    AgenticGroundingConfig,
    GroundingAgentError,
    GroundingAgentTransportError,
)

from grounding.builder import (
    GroundingBuildConfig,
    GroundingBuildError,
    GroundingBuilder,
    load_knowledge,
)
from grounding.exporter import GroundingExportError, export_grounding
from grounding.mcp import (
    AGENT_READ_ONLY_TOOLS,
    DEFAULT_READ_ONLY_TOOLS,
    MCPClient,
    MCPClientConfig,
)
from grounding.models import GroundingStrategy
from grounding.repository import (
    RepositoryIndex,
    RepositoryIndexConfig,
    RepositoryIndexError,
    resolve_repository_suffixes,
)


LOGGER = logging.getLogger("cz.grounding")


class _LiteLLMAdapter:
    """Translate the shared transport's errors into the grounding boundary."""

    def __init__(self, client: Any, error_type: type[BaseException]) -> None:
        self.client = client
        self.error_type = error_type

    @property
    def config(self) -> Any:
        return self.client.config

    async def complete(self, **kwargs: object) -> Any:
        try:
            return await self.client.complete(**kwargs)
        except self.error_type as exc:
            raise GroundingAgentTransportError(str(exc)) from exc


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
        "--repo-include-code",
        action="store_true",
        help="also index source-code files such as .hs/.java/.py (or use --repo-suffix)",
    )
    parser.add_argument(
        "--repo-suffix",
        action="append",
        default=None,
        metavar=".EXT",
        help="extra indexed repository file extension (repeatable), for example "
        "--repo-suffix .hs",
    )
    parser.add_argument(
        "--strategy",
        choices=tuple(item.value for item in GroundingStrategy),
        default=GroundingStrategy.AGENTIC.value,
        help="AI-directed bounded retrieval or legacy deterministic token search",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate Phase 7 input and print the grounding plan without credentials",
    )
    parser.add_argument(
        "--litellm-url",
        default=os.environ.get("CZ_LITELLM_URL"),
        help="LiteLLM base URL or chat endpoint (or CZ_LITELLM_URL)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CZ_LITELLM_MODEL"),
        help="model/deployment name understood by LiteLLM (or CZ_LITELLM_MODEL)",
    )
    parser.add_argument(
        "--api-key-env",
        default="CZ_LITELLM_API_KEY",
        help="environment variable containing the LiteLLM API key",
    )
    parser.add_argument(
        "--no-api-key",
        action="store_true",
        help="connect to a trusted LiteLLM proxy that does not require a key",
    )
    parser.add_argument(
        "--response-format",
        choices=("json_schema", "json_object"),
        default="json_schema",
        help="structured-output mode supported by the selected model",
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
        help="legacy deterministic MCP search tool; agentic mode discovers tools",
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
        "--agent-cases-per-batch",
        type=int,
        default=16,
        help="maximum semantically related testcases handled by one grounding agent",
    )
    parser.add_argument(
        "--agent-concurrency",
        type=int,
        default=2,
        help="maximum concurrent AI grounding batches",
    )
    parser.add_argument(
        "--maximum-agent-turns",
        type=int,
        default=4,
        help="maximum search/final decisions per grounding batch",
    )
    parser.add_argument(
        "--maximum-agent-tool-requests",
        type=int,
        default=8,
        help="maximum independent read-only requests in one agent turn",
    )
    parser.add_argument(
        "--maximum-agent-prompt-characters",
        type=int,
        default=40_000,
        help="hard input-context limit for one grounding turn",
    )
    parser.add_argument(
        "--agent-maximum-snippet-characters",
        type=int,
        default=2_000,
        help="maximum cited evidence characters exposed per retrieval result",
    )
    parser.add_argument(
        "--agent-maximum-snippets-per-case",
        type=int,
        default=3,
        help="maximum candidate citations retained in a testcase prompt",
    )
    parser.add_argument(
        "--agent-snippet-preview-characters",
        type=int,
        default=400,
        help="maximum characters shown for one candidate citation",
    )
    parser.add_argument(
        "--agent-maximum-description-characters",
        type=int,
        default=1_000,
        help="maximum portal-description characters supplied per testcase",
    )
    parser.add_argument(
        "--maximum-output-tokens",
        type=int,
        default=4_096,
        help="maximum completion tokens for one grounding decision",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="timeout for one LiteLLM request",
    )
    parser.add_argument(
        "--maximum-transport-attempts",
        type=int,
        default=2,
        help="retries for timeouts, rate limits, and server failures",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=15.0,
        help="initial exponential LiteLLM retry delay",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=15.0,
        help="seconds between safe LiteLLM heartbeat logs",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="safe progress log (default artifacts/grounding/<run-id>/progress.log)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress on stderr; progress.log is still written",
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
    loaded = load_knowledge(
        source,
        verify_manifest=not args.skip_knowledge_manifest_check,
    )
    strategy = GroundingStrategy(args.strategy)
    if args.plan_only:
        batches = (
            (len(loaded.knowledge.test_cases) + args.agent_cases_per_batch - 1)
            // args.agent_cases_per_batch
            if strategy == GroundingStrategy.AGENTIC
            else 0
        )
        print(
            json.dumps(
                {
                    "source_run_id": loaded.knowledge.source_run_id,
                    "strategy": strategy.value,
                    "test_cases": len(loaded.knowledge.test_cases),
                    "source_testcase_context_complete": (
                        loaded.knowledge.coverage.testcase_context_complete
                    ),
                    "minimum_batches_by_capacity": batches,
                    "minimum_model_calls": batches * 2,
                    "maximum_model_calls": batches * args.maximum_agent_turns,
                    "mcp_tools_are_model_selected": (
                        strategy == GroundingStrategy.AGENTIC
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output_directory = (args.output_root / loaded.knowledge.source_run_id).resolve()
    progress_log_path = (
        args.log_file.expanduser().resolve()
        if args.log_file is not None
        else output_directory / "progress.log"
    )
    _configure_progress_logging(progress_log_path, quiet=args.quiet)
    LOGGER.info(
        "grounding started run_id=%s strategy=%s testcases=%d",
        loaded.knowledge.source_run_id,
        strategy.value,
        len(loaded.knowledge.test_cases),
    )

    repository_excerpt_limit = (
        args.agent_maximum_snippet_characters
        if strategy == GroundingStrategy.AGENTIC
        else 6_000
    )
    repository = RepositoryIndex.build(
        args.repo_path,
        RepositoryIndexConfig(
            suffixes=resolve_repository_suffixes(
                args.repo_suffix, include_code=args.repo_include_code
            ),
            maximum_file_bytes=args.maximum_repo_file_bytes,
            maximum_total_bytes=args.maximum_repo_total_bytes,
            maximum_excerpt_characters=repository_excerpt_limit,
        ),
    )
    LOGGER.info(
        "repository indexed files=%d bytes=%d repository_id=%s",
        repository.summary.files_indexed,
        repository.summary.bytes_indexed,
        repository.summary.repository_id[:12],
    )
    mcp_url = None if args.no_mcp else args.mcp_url
    tools = (
        tuple(args.mcp_tool or DEFAULT_READ_ONLY_TOOLS)
        if strategy == GroundingStrategy.DETERMINISTIC
        else tuple(args.mcp_tool or AGENT_READ_ONLY_TOOLS)
    )
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
    if strategy == GroundingStrategy.AGENTIC:
        if not args.litellm_url:
            raise ValueError(
                "LiteLLM URL is required via --litellm-url or CZ_LITELLM_URL"
            )
        if not args.model:
            raise ValueError(
                "LiteLLM model is required via --model or CZ_LITELLM_MODEL"
            )
        api_key_value = None if args.no_api_key else os.environ.get(args.api_key_env)
        if not args.no_api_key and not api_key_value:
            raise ValueError(
                f"LiteLLM API key environment variable {args.api_key_env!r} is not "
                "set; use --no-api-key only for a trusted local proxy"
            )
        # Import lazily: synthesis reuses grounding models, so importing its package
        # while grounding itself is initializing would create a package cycle.
        from synthesis.client import LiteLLMClient, LiteLLMConfig, LiteLLMError

        client = LiteLLMClient(
            LiteLLMConfig(
                endpoint=args.litellm_url,
                model=args.model,
                api_key=SecretStr(api_key_value) if api_key_value else None,
                timeout_seconds=args.timeout_seconds,
                maximum_transport_attempts=args.maximum_transport_attempts,
                retry_backoff_seconds=args.retry_backoff_seconds,
                maximum_output_tokens=args.maximum_output_tokens,
                response_format=args.response_format,
                progress_heartbeat_seconds=args.progress_interval_seconds,
            )
        )
        llm = _LiteLLMAdapter(client, LiteLLMError)
        package = await AgenticGroundingBuilder(
            loaded.knowledge,
            loaded.sha256,
            repository,
            llm,
            mcp_client=mcp_client,
            config=AgenticGroundingConfig(
                cases_per_batch=args.agent_cases_per_batch,
                concurrency=args.agent_concurrency,
                maximum_turns_per_batch=args.maximum_agent_turns,
                maximum_tool_requests_per_turn=args.maximum_agent_tool_requests,
                maximum_prompt_characters=args.maximum_agent_prompt_characters,
                maximum_output_tokens=args.maximum_output_tokens,
                repository_results_per_request=args.results_per_source,
                maximum_snippet_characters=args.agent_maximum_snippet_characters,
                maximum_snippet_preview_characters=(
                    args.agent_snippet_preview_characters
                ),
                maximum_snippets_per_case=args.agent_maximum_snippets_per_case,
                maximum_description_characters=(
                    args.agent_maximum_description_characters
                ),
                allow_incomplete_source=args.allow_incomplete_source,
            ),
        ).build()
    else:
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
                "strategy": package.strategy.value,
                "test_cases": package.coverage.test_cases,
                "repository_files": package.repository.files_indexed,
                "repository_grounded": package.coverage.repository_grounded,
                "mcp_available": package.mcp.available,
                "mcp_retrieval_complete": package.mcp.retrieval_complete,
                "mcp_calls_succeeded": package.mcp.calls_succeeded,
                "mcp_grounded": package.coverage.mcp_grounded,
                "externally_grounded": package.coverage.externally_grounded,
                "grounding_complete": package.coverage.grounding_complete,
                "model_calls": len(package.agent_calls),
                "agent_tool_calls": len(package.agent_observations),
                "prompt_tokens": (
                    package.provider.prompt_tokens if package.provider else 0
                ),
                "completion_tokens": (
                    package.provider.completion_tokens if package.provider else 0
                ),
                "output_directory": str(output.output_directory),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not package.coverage.grounding_complete and not args.allow_incomplete_grounding:
        return 2
    return 0


def _configure_progress_logging(path: Path, *, quiet: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger_names = ("cz.grounding", "cz.synthesis.transport")
    existing_handlers = {
        handler
        for logger_name in logger_names
        for handler in logging.getLogger(logger_name).handlers
    }
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
    for handler in existing_handlers:
        handler.close()
    formatter = logging.Formatter(
        fmt="%(asctime)sZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    handlers: list[logging.Handler] = []
    if not quiet:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.extend(handlers)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except (
        GroundingBuildError,
        GroundingAgentError,
        GroundingExportError,
        RepositoryIndexError,
        ValueError,
    ) as exc:
        print(f"cz-ground: {exc}")
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("grounding interrupted before a final package was written")
        print("cz-ground: interrupted before a final package was written")
        return 130


def entrypoint() -> None:
    raise SystemExit(main())


__all__ = ["build_parser", "entrypoint", "main"]
