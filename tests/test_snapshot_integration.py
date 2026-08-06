"""Real-browser verification for deterministic Phase 4 state snapshots."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scraper.artifact_store import ArtifactStore
from scraper.browser import BrowserLaunchConfig, BrowserManager
from scraper.extractor import PageStateExtractor, SnapshotConfig
from scraper.models import (
    ArtifactKind,
    CapturePolicy,
    FrameElementCollection,
    ScrapeRun,
    StateSnapshot,
)


RUN_BROWSER_TESTS = os.environ.get("CZ_RUN_BROWSER_TESTS") == "1"


class _SnapshotFixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            port = self.server.server_address[1]
            html = _SNAPSHOT_HTML.replace("__PORT__", str(port))
            self._send_html(html, set_cookie=True)
        elif path in {"/frame", "/cross-frame"}:
            self._send_html(_FRAME_HTML)
        else:
            self.send_error(404)

    def log_message(self, format_string: str, *args: object) -> None:
        return

    def _send_html(self, html: str, *, set_cookie: bool = False) -> None:
        payload = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        if set_cookie:
            self.send_header(
                "Set-Cookie",
                "sessionToken=cookie-value-secret; HttpOnly; SameSite=Lax; Path=/",
            )
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


_FRAME_HTML = """<!doctype html>
<html>
<head><title>Child frame</title></head>
<body>
  <button data-testid="frame-action">Frame action</button>
  <label for="frame-input">Frame input</label>
  <input id="frame-input" value="frame-input-secret">
</body>
</html>
"""


_SNAPSHOT_HTML = """<!doctype html>
<html>
<head>
  <title>Snapshot Fixture</title>
  <style>
    body { min-height: 1800px; font-family: sans-serif; }
    .hidden { display: none; }
  </style>
</head>
<body>
  <main aria-labelledby="page-heading">
    <h1 id="page-heading">Certification test cases</h1>
    <section aria-labelledby="controls-heading">
      <h2 id="controls-heading">Execution controls</h2>
      <button data-testid="run-case">Run case</button>
      <div id="listener-action" style="cursor: pointer">Listener action</div>
      <button id="disabled-action" disabled>Disabled action</button>
      <button id="hidden-action" class="hidden">Hidden action</button>

      <form id="customer-form">
        <label for="customer">Customer number</label>
        <input id="customer" name="customerNumber" value="customer-input-secret">
        <label for="password">Password</label>
        <input id="password" name="password" type="password" value="password-input-secret">
        <label for="environment">Environment</label>
        <select id="environment" name="environment">
          <option value="sandbox" selected>Sandbox</option>
          <option value="production">Production</option>
        </select>
        <label><input id="confirmed" type="checkbox" checked> Confirmed</label>
        <input id="invalid-field" aria-invalid="true" value="invalid-input-secret">
      </form>
    </section>

    <table id="cases">
      <caption>Available test cases</caption>
      <thead><tr><th>ID</th><th>Status</th></tr></thead>
      <tbody><tr><th scope="row">TC-001</th><td>Ready</td></tr></tbody>
    </table>

    <dialog id="review-dialog" open aria-label="Review payload">
      <p>Payload review required</p>
      <button id="approve">Approve</button>
    </dialog>
    <div role="alert">Validation failed</div>
    <progress max="100" value="25">25%</progress>
    <div class="loading">Loading details</div>
    <details open><summary>Request details</summary><p>POST /payments</p></details>
    <div id="shadow-host"></div>
    <iframe id="same-frame" src="/frame"></iframe>
    <iframe id="cross-frame" src="http://127.0.0.1:__PORT__/cross-frame?token=frame-url-secret"></iframe>
  </main>
  <script>
    localStorage.setItem('apiToken', 'local-storage-secret');
    sessionStorage.setItem('sessionSecret', 'session-storage-secret');
    const shadow = document.querySelector('#shadow-host').attachShadow({mode: 'open'});
    shadow.innerHTML = '<button data-testid="shadow-action">Shadow action</button>';
    document.querySelector('#listener-action').addEventListener('click', () => {});
    document.querySelector('#customer').focus();
  </script>
</body>
</html>
"""


@pytest.fixture
def snapshot_portal() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SnapshotFixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _, port = server.server_address
    try:
        yield f"http://localhost:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _attribute(element: object, name: str) -> str | None:
    attributes = getattr(element, "attributes")
    return next(
        (attribute.value for attribute in attributes if attribute.name == name),
        None,
    )


@pytest.mark.skipif(
    not RUN_BROWSER_TESTS,
    reason="set CZ_RUN_BROWSER_TESTS=1 to run real Chromium verification",
)
@pytest.mark.asyncio
async def test_real_page_snapshot_is_complete_stable_and_secret_safe(
    tmp_path: Path,
    snapshot_portal: str,
) -> None:
    run = ScrapeRun(
        run_id="snapshot-run",
        root_url=snapshot_portal,
        allowed_origins=(
            snapshot_portal,
            snapshot_portal.replace("localhost", "127.0.0.1"),
        ),
        capture_policy=CapturePolicy(capture_trace=False, capture_har=False),
    )
    store = ArtifactStore.create(tmp_path / "crawls", run)
    manager = BrowserManager(
        store,
        BrowserLaunchConfig(headless=True, viewport_width=1280, viewport_height=720),
    )

    try:
        page = await manager.start()
        await manager.navigate(snapshot_portal)
        await page.wait_for_selector("iframe#cross-frame")
        await page.wait_for_function("document.querySelectorAll('iframe').length === 2")
        extractor = PageStateExtractor(
            store,
            manager.recorder,
            SnapshotConfig(quiet_window_ms=200, quiet_timeout_ms=5_000),
        )
        first = await extractor.capture(page, sequence=0)
        second = await extractor.capture(page, sequence=1)
    finally:
        await manager.stop()

    assert first.state.stable is True
    assert first.state.fingerprint == second.state.fingerprint
    assert first.state.state_id == second.state.state_id
    assert first.state.element_ids == second.state.element_ids
    assert first.receipt.sequence == 0
    assert second.receipt.sequence == 1

    assert len(first.state.frames) == 3
    assert sum(frame.is_cross_origin for frame in first.state.frames) == 1
    assert first.state.modal_count >= 1
    assert first.state.loading_indicators
    assert first.state.notifications
    assert first.state.errors
    assert first.state.scroll.maximum_y > 0
    assert first.state.active_element_id is not None

    run_button = next(
        element
        for element in first.elements
        if _attribute(element, "data-testid") == "run-case"
    )
    hidden_button = next(
        element
        for element in first.elements
        if _attribute(element, "id") == "hidden-action"
    )
    disabled_button = next(
        element
        for element in first.elements
        if _attribute(element, "id") == "disabled-action"
    )
    shadow_button = next(
        element
        for element in first.elements
        if _attribute(element, "data-testid") == "shadow-action"
    )
    customer_input = next(
        element for element in first.elements if _attribute(element, "id") == "customer"
    )
    listener_action = next(
        element
        for element in first.elements
        if _attribute(element, "id") == "listener-action"
    )

    assert run_button.visible is True
    assert hidden_button.visible is False
    assert disabled_button.enabled is False
    assert shadow_button.shadow_host_path
    assert customer_input.editable is True
    assert customer_input.value_redacted is True
    assert customer_input.element_id == first.state.active_element_id
    assert listener_action.interactive is True
    assert "cursor:pointer" in listener_action.interaction_signals
    assert "listener:click" in listener_action.interaction_signals
    assert any(
        option.selected for element in first.elements for option in element.options
    )
    assert any(element.context.table_id == "cases" for element in first.elements)

    screenshot = next(
        artifact
        for artifact in first.state.artifacts
        if artifact.kind == ArtifactKind.SCREENSHOT
    )
    storage = next(
        artifact
        for artifact in first.state.artifacts
        if artifact.kind == ArtifactKind.STORAGE_STATE
    )
    assert screenshot.redacted is False
    assert storage.redacted is True
    assert store.read_bytes(screenshot).startswith(b"\x89PNG")

    serialized_elements = "\n".join(
        element.model_dump_json() for element in first.elements
    )
    storage_data = store.read_text(storage)
    dom_data = "\n".join(
        store.read_text(frame.dom_artifact)
        for frame in first.state.frames
        if frame.dom_artifact is not None
    )
    for secret in (
        "customer-input-secret",
        "password-input-secret",
        "invalid-input-secret",
        "frame-input-secret",
        "cookie-value-secret",
        "local-storage-secret",
        "session-storage-secret",
        "frame-url-secret",
    ):
        assert secret not in serialized_elements
        assert secret not in storage_data
        assert secret not in dom_data

    for frame in first.state.frames:
        assert frame.elements_artifact is not None
        collection = FrameElementCollection.model_validate_json(
            store.read_bytes(frame.elements_artifact)
        )
        assert collection.frame_id == frame.frame_id
        assert (
            tuple(element.element_id for element in collection.elements)
            == frame.element_ids
        )

    states = list(store.iter_records(StateSnapshot))
    assert states == [first.state, second.state]
    assert store.verify_integrity(strict=True).valid is True
