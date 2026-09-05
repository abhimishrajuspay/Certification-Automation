"""Tests for the external CSV testcase import boundary."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from grounding.builder import load_knowledge
from ingest.builder import (
    IngestBuildError,
    build_portal_knowledge,
    parse_csv_bytes,
    select_rows,
)
from ingest.cli import main as ingest_main
from ingest.exporter import IngestExportError, export_ingest
from ingest.models import IngestExportManifest, normalize_column_value
from knowledge.models import KnowledgeSourceKind, PortalKnowledge


IMPORTED_AT = datetime(2026, 8, 27, 6, 0, 0, tzinfo=timezone.utc)

SAMPLE_CSV = (
    "TC ID,API Type,API Name,Test Data,Version,RC,Description,Steps\n"
    "MT_01,ValAdd,ReqValAdd,mt01@vpa,2,0,Validate payee VPA for P2P.,"
    '"1. Payer enters payee VPA.2. PSP sends Validate Address request."\n'
    'MT_02,BalEnq,ReqBalEnq,"resultType=SUCCESS, Consent=Y",2,U17,'
    '"Balance enquiry fails, negative ACK.",1. User selects account.\n'
    "MT_03,SetCre,ReqSetCre,—,2,RM,Customer changes the PIN.,"
    '"1–14. Follow the standard registration flow.15. Bank validates details."\n'
)


def test_parse_validates_rows_and_splits_steps() -> None:
    parsed = parse_csv_bytes(SAMPLE_CSV.encode("utf-8"))
    assert (
        parsed.source_sha256 == hashlib.sha256(SAMPLE_CSV.encode("utf-8")).hexdigest()
    )
    assert len(parsed.rows) == 3

    first = parsed.rows[0]
    assert first.test_case_id == "MT_01"
    assert first.steps == (
        "1. Payer enters payee VPA.",
        "2. PSP sends Validate Address request.",
    )

    second = parsed.rows[1]
    assert second.test_data == "resultType=SUCCESS, Consent=Y"
    assert second.response_codes == "U17"

    ranged = parsed.rows[2]
    assert ranged.test_data == ""  # dash placeholders normalize to empty
    assert ranged.steps == (
        "1–14. Follow the standard registration flow.",
        "15. Bank validates details.",
    )


def test_row_digests_are_recognized_by_models() -> None:
    parsed = parse_csv_bytes(SAMPLE_CSV.encode("utf-8"))
    for row in parsed.rows:
        # ExternalTestCaseRow re-validates its digest on construction.
        assert len(row.row_sha256) == 64


def test_parse_rejects_empty_and_bad_csv() -> None:
    with pytest.raises(IngestBuildError, match="empty"):
        parse_csv_bytes(b"   ")
    with pytest.raises(IngestBuildError, match="missing required columns"):
        parse_csv_bytes(b"TC ID,Description\nA,hello\n")
    with pytest.raises(IngestBuildError, match="repeats columns"):
        parse_csv_bytes(
            b"TC ID,API Type,API Name,Test Data,Version,RC,Description,RC\n"
            b"A,B,C,D,E,F,G,H\n"
        )


def test_parse_rejects_duplicate_and_incomplete_rows() -> None:
    duplicated = (
        "TC ID,API Type,API Name,Test Data,Version,RC,Description,Steps\n"
        "MT_01,ValAdd,ReqValAdd,x,2,0,First.,1. step\n"
        "MT_01,ValAdd,ReqValAdd,x,2,0,Second.,1. step\n"
    )
    with pytest.raises(IngestBuildError, match="duplicate TC ID"):
        parse_csv_bytes(duplicated.encode())
    blank_required = (
        "TC ID,API Type,API Name,Test Data,Version,RC,Description,Steps\n"
        "MT_09,,ReqValAdd,x,2,0,,1. step\n"
    )
    with pytest.raises(IngestBuildError, match=r"column 'API Type' is empty"):
        parse_csv_bytes(blank_required.encode())
    with pytest.raises(IngestBuildError, match="no usable testcase rows"):
        parse_csv_bytes(
            "TC ID,API Type,API Name,Test Data,Version,RC,Description,Steps\n"
            ",,,,,,,\n".encode()
        )


def test_extra_columns_become_fields_or_errors() -> None:
    with_extra = (
        "TC ID,API Type,API Name,Test Data,Version,RC,Description,Steps,Priority\n"
        "MT_01,ValAdd,ReqValAdd,x,2,0,Desc.,1. step,high\n"
    )
    parsed = parse_csv_bytes(with_extra.encode())
    assert parsed.rows[0].extra_fields[0].key == "priority"
    with_bad_extra = (
        "TC ID,API Type,API Name,Test Data,Version,RC,Description,Steps,123 Bad!\n"
        "MT_01,ValAdd,ReqValAdd,x,2,0,Desc.,1. step,v\n"
    )
    with pytest.raises(IngestBuildError, match="cannot become a knowledge field key"):
        parse_csv_bytes(with_bad_extra.encode())


def test_select_rows_filters_or_explains() -> None:
    parsed = parse_csv_bytes(SAMPLE_CSV.encode())
    subset = select_rows(parsed, ["mt_01".upper()])
    assert [row.test_case_id for row in subset.rows] == ["MT_01"]
    with pytest.raises(IngestBuildError, match="unknown testcase IDs"):
        select_rows(parsed, ["NOPE"])
    ordered = select_rows(parsed, ["MT_03", "MT_01"])
    assert [row.test_case_id for row in ordered.rows] == ["MT_01", "MT_03"]


def test_build_package_has_honest_external_coverage() -> None:
    parsed = select_rows(parse_csv_bytes(SAMPLE_CSV.encode()), ["MT_01"])
    knowledge = build_portal_knowledge(
        parsed,
        run_id="csv-test-01",
        imported_at=IMPORTED_AT,
        rows_total=3,
        selected_ids=["MT_01"],
    )
    assert knowledge.source_kind == KnowledgeSourceKind.EXTERNAL_CSV
    assert knowledge.coverage.testcase_context_complete
    assert not knowledge.coverage.source_bounded_complete
    assert knowledge.coverage.declared_test_cases == 1
    case = knowledge.test_cases[0]
    assert case.test_case_id == "MT_01"
    assert case.dependency_case_ids == ()
    assert case.description == "Validate payee VPA for P2P."
    assert case.evidence == case.description_evidence
    pointer = case.evidence[0]
    assert pointer.state_id.startswith("csvrow-")
    assert pointer.element_ids == ()
    assert pointer.artifact_ids == (parsed.source_sha256,)
    labels = {field.label: field.value for field in case.fields}
    assert labels["API Name"] == "ReqValAdd"
    assert labels["RC"] == "0"
    assert labels["Steps"].startswith("1. Payer enters")
    assert any("external_csv" in text for text in knowledge.coverage.limitations)
    assert any(
        "filtered import: 1 of 3" in text for text in knowledge.coverage.limitations
    )


def test_build_rejects_naive_timestamps() -> None:
    parsed = parse_csv_bytes(SAMPLE_CSV.encode())
    with pytest.raises(IngestBuildError, match="timezone-aware"):
        build_portal_knowledge(
            parsed,
            run_id="csv-test-naive",
            imported_at=datetime(2026, 8, 27),
            rows_total=3,
            selected_ids=[],
        )


def test_export_and_grounding_loader_accept_the_package(tmp_path: Path) -> None:
    data = SAMPLE_CSV.encode()
    parsed_all = parse_csv_bytes(data)
    parsed = select_rows(parsed_all, ["MT_01"])
    knowledge = build_portal_knowledge(
        parsed,
        run_id="csv-test-01",
        imported_at=IMPORTED_AT,
        rows_total=3,
        selected_ids=["MT_01"],
    )

    result = export_ingest(
        parsed,
        data,
        source_csv_name="sheet.csv",
        source_label="BCRP Comfort v1",
        run_id="csv-test-01",
        imported_at=IMPORTED_AT,
        rows_total=3,
        selected_ids=["MT_01"],
        output_directory=tmp_path / "ingest" / "csv-test-01",
    )
    manifest = IngestExportManifest.model_validate_json(
        result.manifest_path.read_bytes()
    )
    assert manifest.rows_total == 3
    assert manifest.rows_selected == 1
    assert manifest.test_cases == ("MT_01",)
    assert manifest.source_csv_sha256 == hashlib.sha256(data).hexdigest()
    for item in manifest.files:
        written = (result.output_directory / item.name).read_bytes()
        assert hashlib.sha256(written).hexdigest() == item.sha256
    assert result.csv_path.read_bytes() == data

    with pytest.raises(IngestExportError, match="already exists"):
        export_ingest(
            parsed,
            data,
            source_csv_name="sheet.csv",
            source_label="BCRP Comfort v1",
            run_id="csv-test-01",
            imported_at=IMPORTED_AT,
            rows_total=3,
            selected_ids=["MT_01"],
            output_directory=tmp_path / "ingest" / "csv-test-01",
        )

    from knowledge.exporter import export_knowledge

    export_knowledge(knowledge, tmp_path / "knowledge" / "csv-test-01")
    loaded = load_knowledge(tmp_path / "knowledge" / "csv-test-01")
    assert loaded.knowledge.test_cases[0].test_case_id == "MT_01"
    assert loaded.knowledge.source_kind == KnowledgeSourceKind.EXTERNAL_CSV
    assert loaded.knowledge.coverage.testcase_context_complete


def test_import_is_deterministic(tmp_path: Path) -> None:
    def build() -> PortalKnowledge:
        return build_portal_knowledge(
            select_rows(parse_csv_bytes(SAMPLE_CSV.encode()), ["MT_01"]),
            run_id="csv-test-01",
            imported_at=IMPORTED_AT,
            rows_total=3,
            selected_ids=["MT_01"],
        )

    first = build().model_dump(mode="json")
    second = build().model_dump(mode="json")
    assert first == second


def test_cli_plan_only_and_full_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "sheet.csv"
    csv_path.write_bytes(SAMPLE_CSV.encode())

    status = ingest_main(
        [
            "--csv",
            str(csv_path),
            "--run-id",
            "csv-cli-01",
            "--test-case",
            "MT_02",
            "--plan-only",
        ]
    )
    assert status == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan == {
        "plan_only": True,
        "run_id": "csv-cli-01",
        "rows_selected": 1,
        "rows_total": 3,
        "csv_sha256": hashlib.sha256(SAMPLE_CSV.encode()).hexdigest(),
        "source_kind": "external_csv",
        "source_label": "sheet",
        "test_cases": ["MT_02"],
    }
    assert not (tmp_path / "ingest").exists()

    status = ingest_main(
        [
            "--csv",
            str(csv_path),
            "--run-id",
            "csv-cli-01",
            "--test-case",
            "MT_02",
            "--ingest-root",
            str(tmp_path / "ingest"),
            "--knowledge-root",
            str(tmp_path / "knowledge"),
        ]
    )
    assert status == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["testcase_context_complete"] is True
    loaded = load_knowledge(tmp_path / "knowledge" / "csv-cli-01")
    assert [case.test_case_id for case in loaded.knowledge.test_cases] == ["MT_02"]

    status = ingest_main(
        [
            "--csv",
            str(csv_path),
            "--run-id",
            "csv-cli-01",
            "--test-case",
            "MT_02",
            "--ingest-root",
            str(tmp_path / "ingest"),
            "--knowledge-root",
            str(tmp_path / "knowledge"),
        ]
    )
    assert status == 1  # no implicit overwrite

    status = ingest_main(
        [
            "--csv",
            str(tmp_path / "missing.csv"),
            "--run-id",
            "csv-cli-00",
        ]
    )
    assert status == 1


def test_normalize_column_value_maps_dashes() -> None:
    assert normalize_column_value(" — ") == ""
    assert normalize_column_value(None) == ""
    assert normalize_column_value(" 00-00 ") == "00-00"
    assert normalize_column_value("line one\nline two") == "line one line two"
