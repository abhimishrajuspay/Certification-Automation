"""CLI for deterministic Postman v2.1 generation from Phase 9 output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from postman.builder import (
    PostmanBuildConfig,
    PostmanBuildError,
    PostmanBuilder,
    load_synthesis,
)
from postman.exporter import PostmanExportError, export_postman


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cz-postman",
        description="Render validated CZ execution specs as Postman Collection v2.1",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--synthesis",
        type=Path,
        help="synthesis.json file or its Phase 9 package directory",
    )
    source.add_argument(
        "--run-id",
        help="run ID beneath --synthesis-root",
    )
    parser.add_argument(
        "--synthesis-root",
        type=Path,
        default=Path("artifacts/synthesis"),
        help="Phase 9 package root used with --run-id",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/postman"),
        help="root for generated Postman packages",
    )
    parser.add_argument(
        "--base-url-variable",
        default="base_url",
        help="name of the empty Postman environment variable for the API origin",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="show safe render coverage without writing Postman files",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="render only dependency-closed ready cases when Phase 9 needs review",
    )
    parser.add_argument(
        "--include-needs-review",
        action="store_true",
        help=(
            "explicit operator opt-in: additionally render needs_review specs as "
            "DRAFT requests marked REVIEW REQUIRED with their unresolved "
            "requirements; blocked specs remain skipped; implies --allow-partial"
        ),
    )
    parser.add_argument(
        "--skip-synthesis-manifest-check",
        action="store_true",
        help="skip the Phase 9 package hash check for diagnostics",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace this run's existing Postman output files",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source = (
            args.synthesis
            if args.synthesis is not None
            else args.synthesis_root / args.run_id
        )
        loaded = load_synthesis(
            source,
            verify_manifest=not args.skip_synthesis_manifest_check,
        )
        result = PostmanBuilder(
            loaded.synthesis,
            loaded.sha256,
            config=PostmanBuildConfig(
                base_url_variable=args.base_url_variable,
                allow_partial=(
                    args.allow_partial or args.include_needs_review or args.plan_only
                ),
                include_needs_review=args.include_needs_review,
            ),
        ).build()
        summary = {
            "source_run_id": result.report.source_run_id,
            "source_synthesis_complete": (
                result.report.coverage.source_synthesis_complete
            ),
            "source_execution_ready": result.report.coverage.source_execution_ready,
            "test_cases": result.report.coverage.test_cases,
            "ready_source_cases": result.report.coverage.ready_source_cases,
            "rendered": result.report.coverage.rendered,
            "rendered_needs_review": result.report.coverage.rendered_needs_review,
            "skipped": result.report.coverage.skipped,
            "generation_complete": result.report.coverage.generation_complete,
        }
        if args.plan_only:
            summary["skipped_test_cases"] = [
                item.model_dump(mode="json")
                for item in result.report.skipped_test_cases
            ]
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        output = export_postman(
            result,
            args.output_root / result.report.source_run_id,
            overwrite=args.overwrite,
        )
        summary["output_directory"] = str(output.output_directory)
        summary["collection"] = str(output.collection_path)
        summary["environment"] = str(output.environment_path)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (PostmanBuildError, PostmanExportError, ValueError) as exc:
        print(f"cz-postman: {exc}")
        return 1


def entrypoint() -> None:
    raise SystemExit(main())


__all__ = ["build_parser", "entrypoint", "main"]
