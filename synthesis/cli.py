"""CLI for constrained LiteLLM synthesis of cited execution specifications."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Optional, Sequence

from pydantic import SecretStr

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
from synthesis.models import SynthesisCallRecord, TestCaseExecutionSpec


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
        "--maximum-cases-per-batch",
        type=int,
        default=8,
        help="maximum semantically grouped testcases in one model request",
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
        help="fresh structured responses allowed after schema/evidence rejection",
    )
    parser.add_argument(
        "--maximum-prompt-characters",
        type=int,
        default=180_000,
        help="hard character limit for one unredacted-in-memory model prompt",
    )
    parser.add_argument(
        "--maximum-output-tokens",
        type=int,
        default=16_000,
        help="maximum completion tokens requested for one batch",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=90.0,
        help="timeout for each LiteLLM HTTP request",
    )
    parser.add_argument(
        "--maximum-transport-attempts",
        type=int,
        default=3,
        help="retries for timeouts, rate limits, and server failures",
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
        help="resume from a compatible per-batch checkpoint",
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
        help="return success when one or more model batches could not be synthesized",
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
        print(
            json.dumps(
                {
                    "source_run_id": loaded.grounding.source_run_id,
                    "grounding_complete": (
                        loaded.grounding.coverage.grounding_complete
                    ),
                    "test_cases": len(loaded.grounding.test_cases),
                    "semantic_groups": plan.semantic_groups,
                    "planned_model_calls": plan.batches,
                    "batch_sizes": plan.batch_sizes,
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
            maximum_output_tokens=args.maximum_output_tokens,
            response_format=args.response_format,
        )
    )
    config = SynthesisBuildConfig(
        maximum_cases_per_batch=args.maximum_cases_per_batch,
        concurrency=args.concurrency,
        maximum_validation_attempts=args.maximum_validation_attempts,
        maximum_prompt_characters=args.maximum_prompt_characters,
        allow_incomplete_source=args.allow_incomplete_source,
    )
    configuration_sha256 = synthesis_configuration_sha256(config, llm)
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

    initial_specifications = ()
    initial_calls = ()
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
        initial_specifications = checkpoint.specifications
        initial_calls = checkpoint.calls
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
        )

    def persist_progress(
        specifications: tuple[TestCaseExecutionSpec, ...],
        calls: tuple[SynthesisCallRecord, ...],
        errors: tuple[str, ...],
    ) -> None:
        save_checkpoint(
            checkpoint_path,
            source_run_id=loaded.grounding.source_run_id,
            source_grounding_sha256=loaded.sha256,
            configuration_sha256=configuration_sha256,
            model=llm.config.model,
            calls=calls,
            specifications=specifications,
            errors=errors,
        )

    package = await SynthesisBuilder(
        loaded.grounding,
        loaded.sha256,
        llm,
        config=config,
    ).build(
        initial_specifications=initial_specifications,
        initial_calls=initial_calls,
        progress=persist_progress,
    )
    # Persist even a zero-batch resumed run with its final validated state.
    persist_progress(package.specifications, package.calls, package.provider.errors)
    output = export_synthesis(
        package,
        output_directory,
        overwrite=args.overwrite or args.resume,
    )
    print(
        json.dumps(
            {
                "source_run_id": package.source_run_id,
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except (SynthesisBuildError, SynthesisExportError, ValueError) as exc:
        print(f"cz-synthesize: {exc}")
        return 1


def entrypoint() -> None:
    raise SystemExit(main())


__all__ = ["build_parser", "entrypoint", "main"]
