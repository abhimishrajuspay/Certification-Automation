"""Real-browser verification for the Phase 6 crawl runner."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from playwright.async_api import Page

from scraper.artifact_store import ArtifactStore
from scraper.extractor import SnapshotConfig
from scraper.models import (
    ActionCandidate,
    ActionStatus,
    CapturePolicy,
    CrawlLimits,
    ScrapeRunStatus,
    StateSnapshot,
)
from scraper.runner import AuthenticationMode, CrawlRequest, CrawlRunner


RUN_BROWSER_TESTS = os.environ.get("CZ_RUN_BROWSER_TESTS") == "1"


class _RunnerFixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/session":
            cookie = self.headers.get("Cookie", "")
            html = (
                _SESSION_PRIVATE_HTML
                if "fixture-session=ready" in cookie
                else _SESSION_PUBLIC_HTML
            )
            self._send_html(html)
            return
        if path != "/":
            self.send_error(404)
            return
        self._send_html(_PORTAL_HTML)

    def _send_html(self, html: str) -> None:
        payload = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format_string: str, *args: object) -> None:
        return


_PORTAL_HTML = """<!doctype html>
<html>
<head><title>Local CZ Fixture</title></head>
<body>
  <h1>Certification cases</h1>
  <table><tr><th>Case</th><th>Action</th></tr>
    <tr><td>TC-001</td><td><button id="details">Show details</button></td></tr>
    <tr><td>TC-002</td><td><button id="run">Run test</button></td></tr>
  </table>
  <main id="view">Choose a case</main>
  <script>
    document.querySelector('#details').addEventListener('click', () => {
      document.body.innerHTML = '<h1>TC-001 details</h1><p>Expected HTTP 200</p>';
    });
  </script>
</body>
</html>
"""


_SESSION_PUBLIC_HTML = """<!doctype html>
<html><head><title>Unauthenticated fixture</title></head>
<body><h1>Login required</h1></body></html>
"""


_SESSION_PRIVATE_HTML = """<!doctype html>
<html><head><title>Authenticated fixture</title></head>
<body><h1>Private portal</h1><a href="/">Dashboard</a></body></html>
"""


@pytest.fixture
def runner_portal() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RunnerFixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.skipif(
    not RUN_BROWSER_TESTS,
    reason="set CZ_RUN_BROWSER_TESTS=1 to run real Chromium verification",
)
@pytest.mark.asyncio
async def test_runner_crawls_local_portal_and_finalizes_integrity(
    tmp_path: Path,
    runner_portal: str,
) -> None:
    request = CrawlRequest(
        root_url=runner_portal,
        artifact_root=tmp_path / "crawls",
        run_id="runner-integration",
        limits=CrawlLimits(
            maximum_depth=1,
            maximum_states=10,
            maximum_actions=10,
            maximum_runtime_seconds=30,
        ),
        capture_policy=CapturePolicy(
            capture_dom=False,
            capture_screenshots=False,
            capture_accessibility_tree=False,
            capture_storage_state=False,
            capture_request_bodies=False,
            capture_response_bodies=False,
            capture_trace=False,
            capture_har=False,
        ),
        snapshot_config=SnapshotConfig(
            quiet_window_ms=100,
            quiet_timeout_ms=2_000,
            full_page_screenshot=False,
        ),
    )

    result = await CrawlRunner(request).run()
    store = ArtifactStore.open(request.artifact_root, result.run_id)
    actions = tuple(store.iter_records(ActionCandidate))

    assert result.status == ScrapeRunStatus.COMPLETED
    assert result.exploration.coverage.bounded_complete is True
    assert result.exploration.coverage.actions_succeeded >= 1
    assert result.exploration.coverage.actions_failed == 0
    review = next(
        action
        for action in actions
        if action.policy_rule == "semantic.review.test_execution"
    )
    assert review.status == ActionStatus.SKIPPED
    assert store.verify_integrity(strict=True).valid is True


@pytest.mark.skipif(
    not RUN_BROWSER_TESTS,
    reason="set CZ_RUN_BROWSER_TESTS=1 to run real Chromium verification",
)
@pytest.mark.asyncio
async def test_storage_state_export_is_reusable_in_a_fresh_browser_context(
    tmp_path: Path,
    runner_portal: str,
) -> None:
    session_url = f"{runner_portal}/session"
    session_path = tmp_path / "sessions" / "portal.json"
    capture_policy = CapturePolicy(
        capture_dom=False,
        capture_screenshots=False,
        capture_accessibility_tree=False,
        capture_storage_state=False,
        capture_request_bodies=False,
        capture_response_bodies=False,
        capture_trace=False,
        capture_har=False,
    )
    limits = CrawlLimits(
        maximum_depth=0,
        maximum_states=5,
        maximum_actions=5,
        maximum_runtime_seconds=30,
    )
    snapshot_config = SnapshotConfig(
        quiet_window_ms=100,
        quiet_timeout_ms=2_000,
        full_page_screenshot=False,
    )

    async def establish_session(page: Page) -> None:
        await page.context.add_cookies(
            [{"name": "fixture-session", "value": "ready", "url": runner_portal}]
        )
        await page.reload(wait_until="load")

    export_request = CrawlRequest(
        root_url=session_url,
        artifact_root=tmp_path / "crawls",
        run_id="session-export",
        authentication_mode=AuthenticationMode.CALLBACK,
        storage_state_output_path=session_path,
        limits=limits,
        capture_policy=capture_policy,
        snapshot_config=snapshot_config,
    )
    exported = await CrawlRunner(
        export_request,
        login_callback=establish_session,
    ).run()

    reuse_request = CrawlRequest(
        root_url=session_url,
        artifact_root=tmp_path / "crawls",
        run_id="session-reuse",
        authentication_mode=AuthenticationMode.STORAGE_STATE,
        storage_state_path=session_path,
        limits=limits,
        capture_policy=capture_policy,
        snapshot_config=snapshot_config,
    )
    reused = await CrawlRunner(reuse_request).run()
    reused_store = ArtifactStore.open(reuse_request.artifact_root, reused.run_id)

    assert exported.status == ScrapeRunStatus.COMPLETED
    assert reused.status == ScrapeRunStatus.COMPLETED
    assert session_path.is_file()
    assert next(reused_store.iter_records(StateSnapshot)).title == (
        "Authenticated fixture"
    )
    assert reused_store.run.behavior_policy.browser.authentication_mode == (
        "storage_state"
    )
