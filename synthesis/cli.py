"""CLI for constrained LiteLLM synthesis of cited execution specifications."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Optional, Sequence

from pydantic import SecretStr

from grounding.mcp import (
    AGENT_READ_ONLY_TOOLS,
    DEFAULT_READ_ONLY_TOOLS,
    MCPClient,
    MCPClientConfig,
)
from grounding.models import GroundingSnippet
from grounding.repository import (
    RepositoryIndex,
    RepositoryIndexConfig,
    RepositoryIndexError,
    resolve_repository_suffixes,
)
from synthesis.agentic import (
    AgenticSynthesisBuilder,
    AgenticSynthesisConfig,
    SynthesisEvidenceTools,
    agentic_configuration_sha256,
)
from synthesis.builder import (
    SynthesisBuildConfig,
    SynthesisBuildError,
    SynthesisBuilder,
    load_grounding,
    plan_synthesis,
    synthesis_configuration_sha256,
)
from synthesis.client import LiteLLMClient, LiteLLMConfig
from synthesis.exporter import (
    SynthesisExportError,
    export_synthesis,
    load_checkpoint,
    save_checkpoint,
)
from synthesis.models import (
    SynthesisCallRecord,
    SynthesisStrategy,
    SynthesisToolObservation,
    TestCaseExecutionSpec,
)


LOGGER = logging.getLogger("cz.synthesis")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cz-synthesize",
        description=(
            "Compile grounded CZ testcases into strict, cited HTTP execution specs"
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--grounding",
        type=Path,
        help="grounding.json file or its Phase 8 package directory",
    )
    source.add_argument(
        "--run-id",
        help="run ID beneath --grounding-root",
    )
    parser.add_argument(
        "--grounding-root",
        type=Path,
        default=Path("artifacts/grounding"),
        help="grounding package root used with --run-id",
    )
    parser.add_argument(
        "--litellm-url",
        default=os.environ.get("CZ_LITELLM_URL"),
        help="LiteLLM base URL or chat-completions endpoint (or CZ_LITELLM_URL)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CZ_LITELLM_MODEL"),
        help="model/deployment name understood by LiteLLM (or CZ_LITELLM_MODEL)",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate Phase 8 input and print the model-call plan without credentials",
    )
    parser.add_argument(
        "--api-key-env",
        default="CZ_LITELLM_API_KEY",
        help="name of the environment variable containing the proxy API key",
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
        "--strategy",
        choices=tuple(item.value for item in SynthesisStrategy),
        default=SynthesisStrategy.AGENTIC.value,
        help="agentic on-demand evidence retrieval or legacy bulk prompting",
    )
    parser.add_argument(
        "--repo-path",
        type=Path,
        help="repository exposed to the agent's bounded search tool",
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
        "--mcp-url",
        default=os.environ.get("CZ_MCP_URL"),
        help="MCP endpoint exposed to the agent (or CZ_MCP_URL)",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="disable live MCP tools even when CZ_MCP_URL is set",
    )
    parser.add_argument(
        "--mcp-tool",
        action="append",
        choices=AGENT_READ_ONLY_TOOLS,
        help=(
            "code-owned read-only MCP tool exposed to the agent; may be "
            f"repeated (defaults: {', '.join(DEFAULT_READ_ONLY_TOOLS)})"
        ),
    )
    parser.add_argument(
        "--mcp-timeout-seconds",
        type=float,
        default=20.0,
        help="timeout for each agent MCP request",
    )
    parser.add_argument(
        "--maximum-cases-per-batch",
        type=int,
        default=8,
        help="maximum grouped testcases in one legacy bulk model request",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="maximum concurrent LiteLLM requests",
    )
    parser.add_argument(
        "--maximum-validation-attempts",
        type=int,
        default=3,
        help="legacy bulk responses allowed after schema/evidence rejection",
    )
    parser.add_argument(
        "--maximum-agent-turns",
        type=int,
        default=8,
        help="maximum evidence decisions and specification corrections per testcase",
    )
    parser.add_argument(
        "--maximum-agent-prompt-characters",
        type=int,
        default=40_000,
        help="hard context limit for one agent turn",
    )
    parser.add_argument(
        "--agent-decision-output-tokens",
        type=int,
        default=2_048,
        help="completion-token cap for the small evidence-decision call",
    )
    parser.add_argument(
        "--agent-repository-results",
        type=int,
        default=5,
        help="maximum repository candidates returned by one agent search",
    )
    parser.add_argument(
        "--agent-mcp-results",
        type=int,
        default=3,
        help="maximum MCP candidates returned by one agent search",
    )
    parser.add_argument(
        "--agent-evidence-preview-characters",
        type=int,
        default=320,
        help="short preview size returned by evidence searches",
    )
    parser.add_argument(
        "--agent-maximum-evidence-characters",
        type=int,
        default=2_000,
        help="maximum evidence content returned by read_evidence",
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
        help="maximum total repository bytes indexed for agent search",
    )
    parser.add_argument(
        "--maximum-prompt-characters",
        type=int,
        default=180_000,
        help="hard character limit for one legacy bulk prompt",
    )
    parser.add_argument(
        "--maximum-output-tokens",
        type=int,
        default=16_000,
        help="maximum completion tokens requested for one model turn",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=90.0,
        help="timeout for each LiteLLM HTTP request",
    )
    parser.add_argument(
        "--no-direct-specification",
        action="store_true",
        help=(
            "always run the full decision loop even when turn-0 evidence is "
            "already complete (route, types, response type, instances)"
        ),
    )
    parser.add_argument(
        "--maximum-transport-attempts",
        type=int,
        default=3,
        help="retries for timeouts, rate limits, and server failures",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=15.0,
        help="initial exponential retry delay; Retry-After takes precedence",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=15.0,
        help="seconds between safe heartbeat logs while awaiting LiteLLM",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help=(
            "safe progress log path (default: "
            "artifacts/synthesis/<run-id>/progress.log)"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress on stderr; progress.log is still written",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/synthesis"),
        help="root for Phase 9 packages and resumable checkpoints",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from a compatible per-testcase checkpoint",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="start fresh and replace this run's existing Phase 9 files",
    )
    parser.add_argument(
        "--allow-incomplete-source",
        action="store_true",
        help="diagnostically synthesize from incomplete Phase 8 grounding",
    )
    parser.add_argument(
        "--allow-incomplete-synthesis",
        action="store_true",
        help="return success when one or more testcases could not be synthesized",
    )
    parser.add_argument(
        "--skip-grounding-manifest-check",
        action="store_true",
        help="skip the Phase 8 package hash check for diagnostics",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    source = (
        args.grounding
        if args.grounding is not None
        else args.grounding_root / args.run_id
    )
    loaded = load_grounding(
        source,
        verify_manifest=not args.skip_grounding_manifest_check,
    )
    plan = plan_synthesis(
        loaded.grounding,
        loaded.sha256,
        args.maximum_cases_per_batch,
    )
    if args.plan_only:
        strategy = SynthesisStrategy(args.strategy)
        print(
            json.dumps(
                {
                    "source_run_id": loaded.grounding.source_run_id,
                    "strategy": strategy.value,
                    "grounding_complete": (
                        loaded.grounding.coverage.grounding_complete
                    ),
                    "test_cases": len(loaded.grounding.test_cases),
                    "semantic_groups": plan.semantic_groups,
                    "minimum_model_calls": (
                        len(loaded.grounding.test_cases) * 2
                        if strategy == SynthesisStrategy.AGENTIC
                        else plan.batches
                    ),
                    "planned_model_calls": (
                        len(loaded.grounding.test_cases) * 2
                        if strategy == SynthesisStrategy.AGENTIC
                        else plan.batches
                    ),
                    "maximum_model_calls": (
                        len(loaded.grounding.test_cases)
                        * (args.maximum_agent_turns + 1)
                        if strategy == SynthesisStrategy.AGENTIC
                        else plan.batches * args.maximum_validation_attempts
                    ),
                    "batch_sizes": (
                        [1] * len(loaded.grounding.test_cases)
                        if strategy == SynthesisStrategy.AGENTIC
                        else plan.batch_sizes
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.litellm_url:
        raise ValueError("LiteLLM URL is required via --litellm-url or CZ_LITELLM_URL")
    if not args.model:
        raise ValueError("LiteLLM model is required via --model or CZ_LITELLM_MODEL")
    api_key_value = None if args.no_api_key else os.environ.get(args.api_key_env)
    if not args.no_api_key and not api_key_value:
        raise ValueError(
            f"LiteLLM API key environment variable {args.api_key_env!r} is not set; "
            "use --no-api-key only for an authenticated/trusted local proxy"
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
            progress_heartbeat_seconds=args.progress_interval_seconds,
        )
    )
    strategy = SynthesisStrategy(args.strategy)
    output_directory = (args.output_root / loaded.grounding.source_run_id).resolve()
    checkpoint_path = output_directory / "checkpoint.json"
    final_paths = (
        output_directory / "synthesis.json",
        output_directory / "execution_specs.jsonl",
        output_directory / "manifest.json",
    )
    if not args.resume and not args.overwrite:
        existing = [
            path.name for path in (*final_paths, checkpoint_path) if path.exists()
        ]
        if existing:
            raise SynthesisExportError(
                f"synthesis output already exists ({', '.join(existing)}); use "
                "--resume or --overwrite explicitly"
            )

    progress_log_path = (
        args.log_file.expanduser().resolve()
        if args.log_file is not None
        else output_directory / "progress.log"
    )
    _configure_progress_logging(
        progress_log_path,
        quiet=args.quiet,
        append=args.resume,
    )
    bulk_config: Optional[SynthesisBuildConfig] = None
    agent_config: Optional[AgenticSynthesisConfig] = None
    evidence_tools: Optional[SynthesisEvidenceTools] = None
    if strategy == SynthesisStrategy.AGENTIC:
        agent_config = AgenticSynthesisConfig(
            concurrency=args.concurrency,
            maximum_turns_per_case=args.maximum_agent_turns,
            maximum_prompt_characters=args.maximum_agent_prompt_characters,
            decision_maximum_output_tokens=args.agent_decision_output_tokens,
            repository_search_results=args.agent_repository_results,
            mcp_search_results=args.agent_mcp_results,
            evidence_preview_characters=args.agent_evidence_preview_characters,
            maximum_evidence_characters=args.agent_maximum_evidence_characters,
            allow_incomplete_source=args.allow_incomplete_source,
            direct_specification_on_complete_evidence=(
                not args.no_direct_specification
            ),
        )
        repository = None
        if args.repo_path is not None:
            LOGGER.info("repository index started path=%s", args.repo_path.resolve())
            repository = RepositoryIndex.build(
                args.repo_path,
                RepositoryIndexConfig(
                    suffixes=resolve_repository_suffixes(
                        args.repo_suffix, include_code=args.repo_include_code
                    ),
                    maximum_file_bytes=args.maximum_repo_file_bytes,
                    maximum_total_bytes=args.maximum_repo_total_bytes,
                    maximum_excerpt_characters=(args.agent_maximum_evidence_characters),
                ),
            )
            LOGGER.info(
                "repository index completed files=%d bytes=%d repository_id=%s",
                repository.summary.files_indexed,
                repository.summary.bytes_indexed,
                repository.summary.repository_id[:12],
            )
        mcp = None
        if args.mcp_url and not args.no_mcp:
            mcp = MCPClient(
                MCPClientConfig(
                    endpoint=args.mcp_url,
                    timeout_seconds=args.mcp_timeout_seconds,
                    allowed_tools=tuple(args.mcp_tool or DEFAULT_READ_ONLY_TOOLS),
                )
            )
        evidence_tools = SynthesisEvidenceTools(
            loaded.grounding,
            repository=repository,
            mcp=mcp,
            config=agent_config,
        )
        configuration_sha256 = agentic_configuration_sha256(
            agent_config,
            llm,
            repository_id=(
                repository.summary.repository_id if repository is not None else None
            ),
            mcp_endpoint=mcp.config.endpoint if mcp is not None else None,
        )
        planned_units = len(loaded.grounding.test_cases)
        planned_sizes = "1"
        concurrency = agent_config.concurrency
    else:
        bulk_config = SynthesisBuildConfig(
            maximum_cases_per_batch=args.maximum_cases_per_batch,
            concurrency=args.concurrency,
            maximum_validation_attempts=args.maximum_validation_attempts,
            maximum_prompt_characters=args.maximum_prompt_characters,
            allow_incomplete_source=args.allow_incomplete_source,
        )
        configuration_sha256 = synthesis_configuration_sha256(bulk_config, llm)
        planned_units = plan.batches
        planned_sizes = ",".join(str(size) for size in plan.batch_sizes)
        concurrency = bulk_config.concurrency
    LOGGER.info(
        "synthesis started run_id=%s strategy=%s testcases=%d units=%d sizes=%s "
        "model=%s concurrency=%d timeout_seconds=%.1f max_output_tokens=%d "
        "checkpoint=%s",
        loaded.grounding.source_run_id,
        strategy.value,
        len(loaded.grounding.test_cases),
        planned_units,
        planned_sizes,
        llm.config.model,
        concurrency,
        llm.config.timeout_seconds,
        llm.config.maximum_output_tokens,
        checkpoint_path,
    )

    initial_specifications = ()
    initial_calls = ()
    initial_retrieved_snippets: tuple[GroundingSnippet, ...] = ()
    initial_observations: tuple[SynthesisToolObservation, ...] = ()
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint.source_run_id != loaded.grounding.source_run_id:
            raise SynthesisExportError("checkpoint run ID does not match grounding")
        if checkpoint.source_grounding_sha256 != loaded.sha256:
            raise SynthesisExportError("checkpoint grounding digest does not match")
        if checkpoint.configuration_sha256 != configuration_sha256:
            raise SynthesisExportError(
                "checkpoint synthesis configuration does not match this command"
            )
        if checkpoint.model != llm.config.model:
            raise SynthesisExportError("checkpoint model does not match this command")
        if checkpoint.strategy != strategy:
            raise SynthesisExportError("checkpoint synthesis strategy does not match")
        initial_specifications = checkpoint.specifications
        initial_calls = checkpoint.calls
        initial_retrieved_snippets = checkpoint.retrieved_snippets
        initial_observations = checkpoint.agent_observations
        LOGGER.info(
            "checkpoint resumed specifications=%d model_responses=%d "
            "retrieved_snippets=%d tool_calls=%d errors=%d",
            len(checkpoint.specifications),
            len(checkpoint.calls),
            len(checkpoint.retrieved_snippets),
            len(checkpoint.agent_observations),
            len(checkpoint.errors),
        )
    else:
        save_checkpoint(
            checkpoint_path,
            source_run_id=loaded.grounding.source_run_id,
            source_grounding_sha256=loaded.sha256,
            configuration_sha256=configuration_sha256,
            model=llm.config.model,
            calls=(),
            specifications=(),
            errors=(),
            strategy=strategy,
        )
        LOGGER.info("checkpoint initialized path=%s", checkpoint_path)

    def save_progress(
        specifications: tuple[TestCaseExecutionSpec, ...],
        calls: tuple[SynthesisCallRecord, ...],
        errors: tuple[str, ...],
        retrieved_snippets: tuple[GroundingSnippet, ...] = (),
        observations: tuple[SynthesisToolObservation, ...] = (),
    ) -> None:
        checkpoint = save_checkpoint(
            checkpoint_path,
            source_run_id=loaded.grounding.source_run_id,
            source_grounding_sha256=loaded.sha256,
            configuration_sha256=configuration_sha256,
            model=llm.config.model,
            calls=calls,
            specifications=specifications,
            errors=errors,
            strategy=strategy,
            retrieved_snippets=retrieved_snippets,
            agent_observations=observations,
        )
        LOGGER.info(
            "checkpoint updated at=%s specifications=%d model_responses=%d "
            "retrieved_snippets=%d tool_calls=%d errors=%d path=%s",
            checkpoint.updated_at.isoformat(),
            len(specifications),
            len(calls),
            len(retrieved_snippets),
            len(observations),
            len(errors),
            checkpoint_path,
        )

    if strategy == SynthesisStrategy.AGENTIC:
        assert agent_config is not None and evidence_tools is not None

        def persist_agent_progress(
            specifications: tuple[TestCaseExecutionSpec, ...],
            calls: tuple[SynthesisCallRecord, ...],
            snippets: tuple[GroundingSnippet, ...],
            observations: tuple[SynthesisToolObservation, ...],
            errors: tuple[str, ...],
        ) -> None:
            save_progress(specifications, calls, errors, snippets, observations)

        package = await AgenticSynthesisBuilder(
            loaded.grounding,
            loaded.sha256,
            llm,
            evidence_tools,
            config=agent_config,
        ).build(
            initial_specifications=initial_specifications,
            initial_calls=initial_calls,
            initial_retrieved_snippets=initial_retrieved_snippets,
            initial_observations=initial_observations,
            progress=persist_agent_progress,
        )
    else:
        assert bulk_config is not None

        def persist_bulk_progress(
            specifications: tuple[TestCaseExecutionSpec, ...],
            calls: tuple[SynthesisCallRecord, ...],
            errors: tuple[str, ...],
        ) -> None:
            save_progress(specifications, calls, errors)

        package = await SynthesisBuilder(
            loaded.grounding,
            loaded.sha256,
            llm,
            config=bulk_config,
        ).build(
            initial_specifications=initial_specifications,
            initial_calls=initial_calls,
            progress=persist_bulk_progress,
        )
    # Persist even a zero-batch resumed run with its final validated state.
    save_progress(
        package.specifications,
        package.calls,
        package.provider.errors,
        package.retrieved_snippets,
        package.agent_observations,
    )
    output = export_synthesis(
        package,
        output_directory,
        overwrite=args.overwrite or args.resume,
    )
    LOGGER.info(
        "synthesis finished synthesized=%d/%d ready=%d needs_review=%d "
        "blocked=%d failed=%d output=%s",
        package.coverage.synthesized,
        package.coverage.test_cases,
        package.coverage.ready,
        package.coverage.needs_review,
        package.coverage.blocked,
        len(package.coverage.failed_test_case_ids),
        output.output_directory,
    )
    print(
        json.dumps(
            {
                "source_run_id": package.source_run_id,
                "strategy": package.strategy.value,
                "test_cases": package.coverage.test_cases,
                "synthesized": package.coverage.synthesized,
                "ready": package.coverage.ready,
                "needs_review": package.coverage.needs_review,
                "blocked": package.coverage.blocked,
                "failed": len(package.coverage.failed_test_case_ids),
                "model_responses": package.provider.responses_received,
                "valid_model_responses": package.provider.valid_responses,
                "prompt_tokens": package.provider.prompt_tokens,
                "completion_tokens": package.provider.completion_tokens,
                "retrieved_snippets": len(package.retrieved_snippets),
                "agent_tool_calls": len(package.agent_observations),
                "synthesis_complete": package.coverage.synthesis_complete,
                "execution_ready": package.coverage.execution_ready,
                "output_directory": str(output.output_directory),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not package.coverage.synthesis_complete and not args.allow_incomplete_synthesis:
        return 2
    return 0


def _configure_progress_logging(
    path: Path,
    *,
    quiet: bool,
    append: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cz.synthesis")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)sZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    if not quiet:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(
        path,
        mode="a" if append else "w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except (
        RepositoryIndexError,
        SynthesisBuildError,
        SynthesisExportError,
        ValueError,
    ) as exc:
        LOGGER.error("synthesis stopped error=%s", exc)
        print(f"cz-synthesize: {exc}")
        return 1
    except KeyboardInterrupt:
        LOGGER.warning(
            "synthesis interrupted; completed checkpoint batches are preserved"
        )
        print("cz-synthesize: interrupted; completed checkpoint batches are preserved")
        return 130


def entrypoint() -> None:
    raise SystemExit(main())


__all__ = ["build_parser", "entrypoint", "main"]
