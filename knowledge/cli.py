"""Command-line entrypoint for crawl-evidence normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from knowledge.builder import (
    KnowledgeBuildConfig,
    KnowledgeBuildError,
    PortalKnowledgeBuilder,
)
from knowledge.exporter import KnowledgeExportError, export_knowledge
from scraper.artifact_store import ArtifactStore, ArtifactStoreError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cz-normalize",
        description="Normalize immutable crawl evidence into LLM-ready knowledge",
    )
    parser.add_argument("--run-id", required=True, help="source crawl run identifier")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/crawls"),
        help="crawl artifact root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/knowledge"),
        help="root for normalized packages",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="produce diagnostic knowledge from a non-completed crawl",
    )
    parser.add_argument(
        "--skip-integrity-check",
        action="store_true",
        help="skip blob hashing after structural stream validation",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="omit normalized network observations",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace this run's existing normalized output files",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        store = ArtifactStore.open(args.artifact_root, args.run_id)
        knowledge = PortalKnowledgeBuilder(
            store,
            KnowledgeBuildConfig(
                allow_incomplete=args.allow_incomplete,
                verify_integrity=not args.skip_integrity_check,
                include_network=not args.no_network,
            ),
        ).build()
        output = export_knowledge(
            knowledge,
            args.output_root / args.run_id,
            overwrite=args.overwrite,
        )
    except (ArtifactStoreError, KnowledgeBuildError, KnowledgeExportError) as exc:
        print(f"cz-normalize: {exc}")
        return 1

    print(
        json.dumps(
            {
                "source_run_id": knowledge.source_run_id,
                "source_complete": knowledge.coverage.source_bounded_complete,
                "testcase_context_complete": (
                    knowledge.coverage.testcase_context_complete
                ),
                "test_cases": knowledge.coverage.test_cases_normalized,
                "descriptions": knowledge.coverage.descriptions_captured,
                "declared_test_cases": knowledge.coverage.declared_test_cases,
                "output_directory": str(output.output_directory),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


__all__ = ["build_parser", "entrypoint", "main"]
