"""Unit verification for the Phase 6 production crawl boundary."""

from __future__ import annotations

import json
import os
import stat
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest
from playwright.async_api import Page

from scraper.actions import ActionPolicyConfig
from scraper.artifact_store import ArtifactStore
from scraper.cli import async_main
from scraper.explorer import ExplorerConfig
from scraper.extractor import SnapshotConfig
from scraper.models import (
    CoverageReport,
    ScrapeRun,
    ScrapeRunStatus,
    StateSnapshot,
    Viewport,
)
from scraper.runner import (
    AuthenticationError,
    AuthenticationMode,
    CrawlRequest,
    CrawlRunner,
    ExistingRunPolicy,
    ResumeUnavailableError,
    StorageStateExportError,
)


NOW = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)
PORTAL_URL = "https://portal.example.test/start"
ORIGIN = "https://portal.example.test"


class _StorageContext:
    async def storage_state(self) -> dict[str, object]:
        return {
            "cookies": [
                {
                    "name": "session",
                    "value": "session-secret",
                    "domain": "portal.example.test",
                    "path": "/",
                }
            ],
            "origins": [],
        }


class _StoragePage:
    context = _StorageContext()


def test_request_rejects_unsafe_or_incomplete_authentication_config(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="headed browser"):
        CrawlRequest(
            root_url=PORTAL_URL,
            authentication_mode=AuthenticationMode.MANUAL,
        )

    with pytest.raises(ValueError, match="storage_state_path"):
        CrawlRequest(
            root_url=PORTAL_URL,
            authentication_mode=AuthenticationMode.STORAGE_STATE,
        )

    state_path = tmp_path / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    request = CrawlRequest(
        root_url=PORTAL_URL,
        authentication_mode=AuthenticationMode.STORAGE_STATE,
        storage_state_path=state_path,
    )
    assert request.storage_state_path == state_path

    with pytest.raises(ValueError, match="run_id"):
        CrawlRequest(root_url=PORTAL_URL, run_id="../escape")

    existing_output = tmp_path / "existing-session.json"
    existing_output.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        CrawlRequest(
            root_url=PORTAL_URL,
            storage_state_output_path=existing_output,
        )
    overwrite = CrawlRequest(
        root_url=PORTAL_URL,
        storage_state_output_path=existing_output,
        overwrite_storage_state_output=True,
    )
    assert overwrite.overwrite_storage_state_output is True

    with pytest.raises(ValueError, match="requires"):
        CrawlRequest(
            root_url=PORTAL_URL,
            overwrite_storage_state_output=True,
        )


def test_callback_mode_requires_and_exclusively_owns_callback() -> None:
    callback_request = CrawlRequest(
        root_url=PORTAL_URL,
        authentication_mode=AuthenticationMode.CALLBACK,
    )
    with pytest.raises(AuthenticationError, match="requires"):
        CrawlRunner(callback_request)

    async def callback(page: Page) -> None:
        return

    with pytest.raises(AuthenticationError, match="only valid"):
        CrawlRunner(CrawlRequest(root_url=PORTAL_URL), login_callback=callback)


@pytest.mark.asyncio
async def test_manual_confirmation_is_injected_and_receives_only_origin() -> None:
    prompts: list[str] = []

    async def confirm(prompt: str) -> None:
        prompts.append(prompt)

    request = CrawlRequest(
        root_url=f"{PORTAL_URL}?token=must-not-appear",
        authentication_mode=AuthenticationMode.MANUAL,
        headless=False,
    )
    runner = CrawlRunner(request, manual_confirmation=confirm)

    await runner._authenticate(cast(Page, object()))

    assert prompts == [
        "Complete login in the opened browser for "
        "https://portal.example.test, then press Enter: "
    ]
    assert "must-not-appear" not in prompts[0]


@pytest.mark.asyncio
async def test_console_confirmation_timeout_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_input(prompt: str) -> str:
        started.set()
        release.wait(timeout=1)
        return ""

    monkeypatch.setattr("builtins.input", blocking_input)
    request = CrawlRequest(
        root_url=PORTAL_URL,
        authentication_mode=AuthenticationMode.MANUAL,
        authentication_timeout_ms=20,
        headless=False,
    )

    try:
        with pytest.raises(AuthenticationError, match="timed out"):
            await CrawlRunner(request)._authenticate(cast(Page, object()))
        assert started.is_set()
    finally:
        release.set()


@pytest.mark.asyncio
async def test_explicit_storage_state_export_is_private_and_not_manifested(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "private" / "session.json"
    request = CrawlRequest(
        root_url=PORTAL_URL,
        artifact_root=tmp_path / "crawls",
        run_id="storage-export",
        storage_state_output_path=output_path,
    )
    runner = CrawlRunner(request)
    store, reused = runner._prepare_store()

    await runner._export_storage_state(cast(Page, _StoragePage()))

    assert reused is False
    assert (
        json.loads(output_path.read_text(encoding="utf-8"))["cookies"][0]["value"]
        == "session-secret"
    )
    if os.name != "nt":
        assert stat.S_IMODE(output_path.stat().st_mode) == 0o600
    manifest = (store.run_directory / "manifest.json").read_text(encoding="utf-8")
    assert str(output_path) not in manifest
    assert "session-secret" not in manifest
    assert store.run.behavior_policy.browser.storage_state_output_configured is True

    with pytest.raises(StorageStateExportError, match="already exists"):
        await runner._export_storage_state(cast(Page, _StoragePage()))


def test_manifest_persists_complete_behavior_policy(tmp_path: Path) -> None:
    request = CrawlRequest(
        root_url=PORTAL_URL,
        artifact_root=tmp_path / "crawls",
        run_id="behavior-policy",
        authentication_timeout_ms=42_000,
        ready_selector="[data-ready='true']",
        action_timeout_ms=8_000,
        popup_detection_timeout_ms=250,
        snapshot_config=SnapshotConfig(quiet_window_ms=250),
        action_policy_config=ActionPolicyConfig(
            include_hover_actions=False,
            deduplicate_nested_targets=False,
        ),
        explorer_config=ExplorerConfig(restore_timeout_ms=9_000),
    )

    store, _ = CrawlRunner(request)._prepare_store()
    behavior = store.run.behavior_policy

    assert behavior.browser.authentication_timeout_ms == 42_000
    assert behavior.browser.ready_selector_configured is True
    assert behavior.browser.ready_selector_sha256 is not None
    assert "data-ready" not in json.dumps(behavior.model_dump(mode="json"))
    assert behavior.snapshot.quiet_window_ms == 250
    assert behavior.action.include_hover_actions is False
    assert behavior.action.deduplicate_nested_targets is False
    assert behavior.action.execution_timeout_ms == 8_000
    assert behavior.action.popup_detection_timeout_ms == 250
    assert behavior.explorer.restore_timeout_ms == 9_000


@pytest.mark.asyncio
async def test_completed_run_is_verified_and_returned_without_browser(
    tmp_path: Path,
) -> None:
    base = tmp_path / "crawls"
    _make_completed_store(base, "completed-run")
    request = CrawlRequest(
        root_url=PORTAL_URL,
        artifact_root=base,
        run_id="completed-run",
        existing_run_policy=ExistingRunPolicy.RETURN_COMPLETED,
    )

    result = await CrawlRunner(request).run()

    assert result.reused_existing is True
    assert result.status == ScrapeRunStatus.COMPLETED
    assert result.exploration.root_state_id == "state-root"
    assert result.exploration.coverage.bounded_complete is True


@pytest.mark.asyncio
async def test_completed_run_cannot_be_reused_for_different_request(
    tmp_path: Path,
) -> None:
    base = tmp_path / "crawls"
    _make_completed_store(base, "completed-run")
    request = CrawlRequest(
        root_url="https://another.example.test/start",
        artifact_root=base,
        run_id="completed-run",
        existing_run_policy=ExistingRunPolicy.RETURN_COMPLETED,
    )

    with pytest.raises(ResumeUnavailableError, match="does not match"):
        await CrawlRunner(request).run()


@pytest.mark.asyncio
async def test_incomplete_run_requires_a_new_attempt(tmp_path: Path) -> None:
    base = tmp_path / "crawls"
    ArtifactStore.create(
        base,
        ScrapeRun(
            run_id="interrupted-run",
            root_url=PORTAL_URL,
            allowed_origins=(ORIGIN,),
        ),
    )
    request = CrawlRequest(
        root_url=PORTAL_URL,
        artifact_root=base,
        run_id="interrupted-run",
        existing_run_policy=ExistingRunPolicy.RETURN_COMPLETED,
    )

    with pytest.raises(ResumeUnavailableError, match="intentionally unsupported"):
        await CrawlRunner(request).run()


def test_new_attempt_policy_never_overwrites_existing_run(tmp_path: Path) -> None:
    base = tmp_path / "crawls"
    ArtifactStore.create(
        base,
        ScrapeRun(
            run_id="crawl-run",
            root_url=PORTAL_URL,
            allowed_origins=(ORIGIN,),
        ),
    )
    request = CrawlRequest(
        root_url=PORTAL_URL,
        artifact_root=base,
        run_id="crawl-run",
        existing_run_policy=ExistingRunPolicy.NEW_ATTEMPT,
    )

    store, reused = CrawlRunner(request)._prepare_store()

    assert reused is False
    assert store.run.run_id == "crawl-run-attempt-1"
    assert (base / "crawl-run" / "manifest.json").is_file()
    assert (base / "crawl-run-attempt-1" / "manifest.json").is_file()


def test_runtime_url_secret_is_redacted_before_manifest_persistence(
    tmp_path: Path,
) -> None:
    request = CrawlRequest(
        root_url=f"{PORTAL_URL}?token=runtime-only&mode=certification",
        artifact_root=tmp_path / "crawls",
        run_id="redacted-root",
    )

    store, reused = CrawlRunner(request)._prepare_store()

    assert reused is False
    assert "runtime-only" not in store.run.root_url
    assert "REDACTED" in store.run.root_url
    assert "mode=certification" in store.run.root_url
    assert request.root_url.endswith("token=runtime-only&mode=certification")


@pytest.mark.asyncio
async def test_validate_only_does_not_expose_url_secret_or_storage_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    storage_path = tmp_path / "sensitive-session-name.json"
    storage_path.write_text("{}", encoding="utf-8")
    output_path = tmp_path / "sensitive-session-output.json"

    status = await async_main(
        [
            "--url",
            f"{PORTAL_URL}?token=must-not-appear#secret=also-hidden",
            "--auth",
            "storage_state",
            "--storage-state",
            str(storage_path),
            "--save-storage-state",
            str(output_path),
            "--validate-only",
        ]
    )

    output = capsys.readouterr().out
    summary = json.loads(output)
    assert status == 0
    assert summary["valid"] is True
    assert summary["storage_state_configured"] is True
    assert summary["storage_state_output_configured"] is True
    assert summary["root_target"] == PORTAL_URL
    assert "must-not-appear" not in output
    assert "also-hidden" not in output
    assert str(storage_path) not in output
    assert str(output_path) not in output


@pytest.mark.asyncio
async def test_cli_allowed_origins_extend_root_scope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = await async_main(
        [
            "--url",
            PORTAL_URL,
            "--allowed-origin",
            "https://api.example.test",
            "--validate-only",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert status == 0
    assert summary["allowed_origins"] == [ORIGIN, "https://api.example.test"]


def _make_completed_store(base: Path, run_id: str) -> ArtifactStore:
    store = ArtifactStore.create(
        base,
        ScrapeRun(
            run_id=run_id,
            root_url=PORTAL_URL,
            allowed_origins=(ORIGIN,),
        ),
    )
    state = StateSnapshot(
        state_id="state-root",
        run_id=run_id,
        sequence=0,
        fingerprint="a" * 64,
        captured_at=NOW,
        page_id="page-root",
        url=PORTAL_URL,
        viewport=Viewport(width=1280, height=720),
    )
    coverage = CoverageReport(
        run_id=run_id,
        generated_at=NOW,
        states_discovered=1,
        routes_discovered=1,
        bounded_complete=True,
        completion_reason="frontier exhausted",
    )
    store.append_record(state)
    store.append_record(coverage)
    completed = store.run.model_copy(
        update={
            "status": ScrapeRunStatus.COMPLETED,
            "started_at": NOW,
            "ended_at": NOW,
            "state_ids": (state.state_id,),
            "completion_reason": "frontier exhausted",
        }
    )
    store.save_manifest(completed)
    store.write_checkpoint(1, last_state_id=state.state_id)
    return store
