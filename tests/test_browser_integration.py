"""Real-browser verification for Phase 3 instrumentation.

Set ``CZ_RUN_BROWSER_TESTS=1`` to execute this test. Chromium needs process
permissions that are intentionally unavailable in some sandboxed test runners.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scraper.artifact_store import ArtifactStore
from scraper.browser import BrowserLaunchConfig, BrowserManager
from scraper.models import (
    BrowserEvent,
    BrowserEventKind,
    CapturePolicy,
    NetworkExchange,
    ScrapeRun,
)
from scraper.redaction import REDACTED


RUN_BROWSER_TESTS = os.environ.get("CZ_RUN_BROWSER_TESTS") == "1"


class _FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send_html(_PORTAL_HTML)
        elif path == "/frame":
            self._send_html(
                "<title>Frame</title><p id='frame-content'>Frame content</p>"
            )
        elif path == "/popup":
            self._send_html("<title>Popup</title><p>Popup content</p>")
        elif path == "/download":
            payload = b"download evidence"
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header(
                "Content-Disposition", 'attachment; filename="evidence.txt"'
            )
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path != "/api":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        payload = json.dumps({"visible": "done", "token": "response-secret"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format_string: str, *args: object) -> None:
        return

    def _send_html(self, html: str) -> None:
        payload = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


_PORTAL_HTML = """<!doctype html>
<html>
<head><title>Recorder Fixture</title></head>
<body>
  <button id="action">Run action</button>
  <button id="dialog">Open dialog</button>
  <button id="popup">Open popup</button>
  <a id="download" download="evidence.txt"
     href="/download?token=download-secret">Download</a>
  <input id="file" type="file">
  <div id="result"></div>
  <iframe src="/frame"></iframe>
  <script>
    console.log('fixture ready token=console-secret');
    setTimeout(() => { throw new Error('password=page-secret'); }, 0);

    document.querySelector('#action').addEventListener('click', async () => {
      history.pushState({}, '', '/changed?token=url-secret');
      const marker = document.createElement('span');
      marker.id = 'mutation-marker';
      marker.textContent = 'created';
      document.body.appendChild(marker);
      const response = await fetch('/api?token=request-secret', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer header-secret'
        },
        body: JSON.stringify({password: 'request-body-secret', visible: 'yes'})
      });
      const data = await response.json();
      const result = document.querySelector('#result');
      result.textContent = data.visible;
      result.dataset.done = 'true';
    });

    document.querySelector('#dialog').addEventListener('click', () => {
      alert('password=dialog-secret');
    });
    document.querySelector('#popup').addEventListener('click', () => {
      window.open('/popup?token=popup-secret', '_blank');
    });
    const workerSource = "postMessage('ready')";
    new Worker(URL.createObjectURL(new Blob([workerSource])));
  </script>
</body>
</html>
"""


@pytest.fixture
def fixture_portal() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
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
async def test_real_browser_events_are_recorded_before_and_after_actions(
    tmp_path: Path,
    fixture_portal: str,
) -> None:
    policy = CapturePolicy(capture_har=True, capture_trace=True)
    run = ScrapeRun(
        run_id="browser-run",
        root_url=fixture_portal,
        allowed_origins=(fixture_portal,),
        capture_policy=policy,
    )
    store = ArtifactStore.create(tmp_path / "crawls", run)
    manager = BrowserManager(
        store,
        BrowserLaunchConfig(headless=True, viewport_width=1280, viewport_height=720),
    )

    try:
        page = await manager.start()
        assert page.url == "about:blank"
        await manager.navigate(fixture_portal)
        await page.wait_for_selector("#action")

        async with manager.recorder.action_scope("action-fetch"):
            await page.click("#action")
            await page.wait_for_selector("#result[data-done='true']")
            assert await manager.recorder.drain_mutations(page) > 0
            await manager.recorder.flush()

        async with manager.recorder.action_scope("action-dialog"):
            await page.click("#dialog")
            await manager.recorder.flush()

        async with manager.recorder.action_scope("action-popup"):
            async with page.expect_popup() as popup_info:
                await page.click("#popup")
            popup = await popup_info.value
            await popup.wait_for_load_state("domcontentloaded")
            await manager.recorder.flush()

        async with manager.recorder.action_scope("action-download"):
            async with page.expect_download() as download_info:
                await page.click("#download")
            await download_info.value
            await manager.recorder.flush()

        async with manager.recorder.action_scope("action-file-chooser"):
            async with page.expect_file_chooser():
                await page.click("#file")
            await manager.recorder.flush()
    finally:
        await manager.stop()

    events = list(store.iter_records(BrowserEvent))
    exchanges = list(store.iter_records(NetworkExchange))
    event_kinds = {event.kind for event in events}

    assert {
        BrowserEventKind.PAGE_OPENED,
        BrowserEventKind.REQUEST,
        BrowserEventKind.RESPONSE,
        BrowserEventKind.CONSOLE,
        BrowserEventKind.PAGE_ERROR,
        BrowserEventKind.DOM_CONTENT_LOADED,
        BrowserEventKind.LOAD,
        BrowserEventKind.NAVIGATION,
        BrowserEventKind.DOM_MUTATION,
        BrowserEventKind.DIALOG,
        BrowserEventKind.POPUP,
        BrowserEventKind.DOWNLOAD,
        BrowserEventKind.FILE_CHOOSER,
        BrowserEventKind.FRAME_NAVIGATED,
        BrowserEventKind.WORKER_CREATED,
        BrowserEventKind.TRACE_SAVED,
        BrowserEventKind.HAR_SAVED,
    }.issubset(event_kinds)

    page_opened_index = next(
        index
        for index, event in enumerate(events)
        if event.kind == BrowserEventKind.PAGE_OPENED
    )
    first_request_index = next(
        index
        for index, event in enumerate(events)
        if event.kind == BrowserEventKind.REQUEST
    )
    page_opened_details = {
        detail.name: detail.value for detail in events[page_opened_index].details
    }
    assert page_opened_details["url"] == "about:blank"
    assert page_opened_index < first_request_index

    expected_action_events = {
        BrowserEventKind.DOM_MUTATION: "action-fetch",
        BrowserEventKind.DIALOG: "action-dialog",
        BrowserEventKind.POPUP: "action-popup",
        BrowserEventKind.DOWNLOAD: "action-download",
        BrowserEventKind.FILE_CHOOSER: "action-file-chooser",
    }
    for kind, action_id in expected_action_events.items():
        assert any(
            event.kind == kind and event.action_id == action_id for event in events
        )

    api_exchange = next(exchange for exchange in exchanges if "/api" in exchange.url)
    assert api_exchange.action_id == "action-fetch"
    assert "request-secret" not in api_exchange.url
    assert REDACTED in api_exchange.url or "%5BREDACTED%5D" in api_exchange.url

    request_headers = {
        header.name.lower(): header for header in api_exchange.request_headers
    }
    assert request_headers["authorization"].redacted is True
    assert request_headers["authorization"].value is None

    assert api_exchange.request_body is not None
    request_body = store.read_text(api_exchange.request_body)
    assert "request-body-secret" not in request_body
    assert REDACTED in request_body

    assert api_exchange.response_body is not None
    response_body = store.read_text(api_exchange.response_body)
    assert "response-secret" not in response_body
    assert REDACTED in response_body

    trace_event = next(
        event for event in events if event.kind == BrowserEventKind.TRACE_SAVED
    )
    har_event = next(
        event for event in events if event.kind == BrowserEventKind.HAR_SAVED
    )
    assert trace_event.artifacts[0].redacted is False
    assert har_event.artifacts[0].redacted is True
    assert len(store.read_bytes(trace_event.artifacts[0])) > 0
    har_data = store.read_text(har_event.artifacts[0])
    assert "header-secret" not in har_data
    assert "request-body-secret" not in har_data
    assert "response-secret" not in har_data

    serialized_evidence = "\n".join(
        [event.model_dump_json() for event in events]
        + [exchange.model_dump_json() for exchange in exchanges]
    )
    for secret in (
        "console-secret",
        "page-secret",
        "dialog-secret",
        "popup-secret",
        "download-secret",
        "header-secret",
        "request-body-secret",
        "response-secret",
    ):
        assert secret not in serialized_evidence

    assert store.verify_integrity(strict=True).valid is True
