"""Real-browser verification for the Phase 6 crawl runner."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scraper.artifact_store import ArtifactStore
from scraper.extractor import SnapshotConfig
from scraper.models import (
    ActionCandidate,
    ActionStatus,
    CapturePolicy,
    CrawlLimits,
    ScrapeRunStatus,
)
from scraper.runner import CrawlRequest, CrawlRunner


RUN_BROWSER_TESTS = os.environ.get("CZ_RUN_BROWSER_TESTS") == "1"


class _RunnerFixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/":
            self.send_error(404)
            return
        payload = _PORTAL_HTML.encode()
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
    review = next(action for action in actions if action.policy_rule.endswith(".run"))
    assert review.status == ActionStatus.SKIPPED
    assert store.verify_integrity(strict=True).valid is True
