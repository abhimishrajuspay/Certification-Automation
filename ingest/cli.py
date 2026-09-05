"""CLI for importing externally scraped CSV testcases into the pipeline."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from ingest.builder import (
    IngestBuildError,
    build_portal_knowledge,
    parse_csv_bytes,
    select_rows,
)
from ingest.exporter import IngestExportError, export_ingest
from knowledge.exporter import KnowledgeExportError, export_knowledge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cz-ingest",
        description=(
            "Import an externally scraped testcase CSV (columns: TC ID, API Type,"
            " API Name, Test Data, Version, RC, Description, Steps) into an"
            " honest external_csv knowledge package for grounding and beyond"
        ),
    )
    parser.add_argument("--csv", type=Path, required=True, help="source testcase CSV")
    parser.add_argument("--run-id", required=True, help="import run identifier")
    parser.add_argument(
        "--test-case",
        action="append",
        default=[],
        help="import only these testcase IDs (repeatable); default is every row",
    )
    parser.add_argument(
        "--source-label",
        help="audit label for the source sheet (default: CSV file stem)",
    )
    parser.add_argument(
        "--ingest-root",
        type=Path,
        default=Path("artifacts/ingest"),
        help="root for immutable import evidence copies",
    )
    parser.add_argument(
        "--knowledge-root",
        type=Path,
        default=Path("artifacts/knowledge"),
        help="root for the generated knowledge package",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="parse and summarize without writing any package",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace this run's existing import and knowledge packages",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        csv_path = args.csv.expanduser().resolve()
        data = csv_path.read_bytes()
    except OSError as exc:
        print(f"cz-ingest: cannot read {args.csv}: {exc}")
        return 1
    try:
        parsed_all = parse_csv_bytes(data)
        parsed = select_rows(parsed_all, list(args.test_case))
    except IngestBuildError as exc:
        print(f"cz-ingest: {exc}")
        return 1

    imported_at = datetime.now(timezone.utc)
    source_label = (
        args.source_label.strip()
        if args.source_label and args.source_label.strip()
        else re.sub(r"\s+", " ", csv_path.stem).strip() or csv_path.stem
    )
    summary: dict[str, object] = {
        "run_id": args.run_id,
        "source_kind": "external_csv",
        "source_label": source_label,
        "csv_sha256": parsed.source_sha256,
        "rows_total": len(parsed_all.rows),
        "rows_selected": len(parsed.rows),
        "test_cases": [row.test_case_id for row in parsed.rows],
    }

    if args.plan_only:
        summary["plan_only"] = True
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    try:
        knowledge = build_portal_knowledge(
            parsed,
            run_id=args.run_id,
            imported_at=imported_at,
            rows_total=len(parsed_all.rows),
            selected_ids=list(args.test_case),
        )
        ingest_result = export_ingest(
            parsed,
            data,
            source_csv_name=csv_path.name,
            source_label=source_label,
            run_id=args.run_id,
            imported_at=imported_at,
            rows_total=len(parsed_all.rows),
            selected_ids=list(args.test_case),
            output_directory=args.ingest_root / args.run_id,
            overwrite=args.overwrite,
        )
        knowledge_result = export_knowledge(
            knowledge,
            args.knowledge_root / args.run_id,
            overwrite=args.overwrite,
        )
    except (
        IngestBuildError,
        IngestExportError,
        KnowledgeExportError,
        ValueError,
    ) as exc:
        print(f"cz-ingest: {exc}")
        return 1

    summary.update(
        {
            "testcase_context_complete": knowledge.coverage.testcase_context_complete,
            "limitations": list(knowledge.coverage.limitations),
            "ingest_directory": str(ingest_result.output_directory),
            "knowledge_directory": str(knowledge_result.output_directory),
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


__all__ = ["build_parser", "entrypoint", "main"]
