"""Tests for the single-command external-source flow driver."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import pytest

from pipeline import cli as pipeline_cli
from pipeline.cli import build_parser, derive_run_id


def _base_args(tmp_path: Path) -> list[str]:
    return [
        "--csv",
        str(tmp_path / "cases.csv"),
        "--run-id",
        "flow-test-01",
        "--repo-path",
        str(tmp_path / "repo"),
    ]


def test_derive_run_id_is_slug_plus_utc_timestamp(tmp_path: Path) -> None:
    now = datetime(2026, 8, 27, 14, 5, 6, tzinfo=timezone.utc)
    run_id = derive_run_id(tmp_path / "BCRP Comfort v1 Copy.csv", None, now)
    assert run_id == "bcrp-comfort-v1-copy-20260827-140506"
    assert derive_run_id(tmp_path / "x.csv", "Custom Label", now).startswith(
        "custom-label-20260827-140506"
    )
    assert derive_run_id(tmp_path / "---.csv", None, now).startswith("flow-")


def test_parser_requires_csv_and_repo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--csv", "x.csv"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--repo-path", "repo"])


def _fake_mains(
    monkeypatch: pytest.MonkeyPatch, codes: dict[str, int]
) -> dict[str, Sequence[str]]:
    calls: dict[str, Sequence[str]] = {}

    def make(name: str):
        def fake(argv: Optional[Sequence[str]]) -> int:
            calls[name] = list(argv or [])
            return codes[name]

        return fake

    for name in ("ingest", "grounding", "synthesis", "postman"):
        monkeypatch.setitem(pipeline_cli._PHASE_MAIN, name, make(name))
    return calls


def test_flow_runs_all_phases_in_order_with_threaded_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _fake_mains(
        monkeypatch,
        {"ingest": 0, "grounding": 0, "synthesis": 0, "postman": 0},
    )
    status = pipeline_cli.main(
        [
            "--csv",
            str(tmp_path / "cases.csv"),
            "--run-id",
            "flow-test-01",
            "--test-case",
            "MT_01",
            "--repo-path",
            str(tmp_path / "repo"),
            "--repo-include-code",
            "--repo-suffix",
            ".hs",
            "--response-format",
            "json_object",
            "--mcp-tool",
            "search_docs",
            "--allow-partial",
            "--overwrite",
        ]
    )
    assert status == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "flow_complete"
    assert list(calls) == ["ingest", "grounding", "synthesis", "postman"]
    assert all(code == 0 for code in summary["phases"].values())

    ingest_argv = calls["ingest"]
    assert "--test-case" in ingest_argv and "MT_01" in ingest_argv
    for phase in ("ingest", "grounding", "synthesis", "postman"):
        assert "--overwrite" in calls[phase]
    for phase in ("grounding", "synthesis"):
        assert "--repo-include-code" in calls[phase]
        assert ".hs" in calls[phase]
        assert (
            calls[phase][calls[phase].index("--response-format") + 1] == "json_object"
        )
    assert "--mcp-tool" not in calls["grounding"]
    assert (
        calls["synthesis"][calls["synthesis"].index("--mcp-tool") + 1] == "search_docs"
    )
    assert "--allow-partial" in calls["postman"]
    assert "--allow-partial" not in calls["synthesis"]


def test_flow_stops_at_first_failing_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _fake_mains(
        monkeypatch,
        {"ingest": 0, "grounding": 2, "synthesis": 0, "postman": 0},
    )
    status = pipeline_cli.main(_base_args(tmp_path))
    assert status == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "stopped: grounding exited 2"
    assert summary["phases"] == {"ingest": 0, "grounding": 2}
    assert "synthesis" not in calls and "postman" not in calls


def test_plan_only_validates_ingest_without_touching_downstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _fake_mains(
        monkeypatch,
        {"ingest": 0, "grounding": 0, "synthesis": 0, "postman": 0},
    )
    status = pipeline_cli.main([*_base_args(tmp_path), "--plan-only"])
    assert status == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "plan_only_complete"
    assert summary["staged_phases"] == ["ingest", "grounding", "synthesis", "postman"]
    assert "--plan-only" in calls["ingest"]
    assert "grounding" not in calls and "synthesis" not in calls
    assert "postman" not in calls
