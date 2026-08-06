"""Real-browser verification for bounded Phase 5 graph exploration."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scraper.actions import ActionExecutor, ActionPlanner
from scraper.artifact_store import ArtifactStore
from scraper.browser import BrowserLaunchConfig, BrowserManager
from scraper.explorer import StateGraphExplorer
from scraper.extractor import PageStateExtractor, SnapshotConfig
from scraper.models import (
    ActionCandidate,
    ActionStatus,
    CapturePolicy,
    CrawlLimits,
    EffectKind,
    InteractionTransition,
    ScrapeRun,
    ScrapeRunStatus,
    StateSnapshot,
)


RUN_BROWSER_TESTS = os.environ.get("CZ_RUN_BROWSER_TESTS") == "1"


class _ExplorerFixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send_html(_EXPLORER_HTML)
        elif path == "/frame":
            self._send_html(_FRAME_HTML)
        elif path == "/popup":
            self._send_html(
                "<!doctype html><title>Help</title><main><h1>Help panel</h1></main>"
            )
        elif path == "/data":
            payload = json.dumps({"message": "Network details loaded"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)

    def log_message(self, format_string: str, *args: object) -> None:
        return

    def _send_html(self, html: str) -> None:
        payload = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


_FRAME_HTML = """<!doctype html>
<html><body>
  <button data-testid="frame-view">Show frame view</button>
  <script>
    document.querySelector('[data-testid=frame-view]').addEventListener('click', () => {
      parent.postMessage({type: 'frame-view'}, '*');
      document.body.textContent = 'Frame view visible';
    });
  </script>
</body></html>
"""


_EXPLORER_HTML = """<!doctype html>
<html>
<head><title>Explorer Fixture</title></head>
<body>
  <h1>Portal explorer</h1>
  <div id="controls">
    <button data-testid="show-alpha">Show alpha</button>
    <button data-testid="show-beta">Show beta</button>
    <button data-testid="load-details">Load details</button>
    <button data-testid="open-help">Open help</button>
    <button data-testid="run-test">Run test</button>
    <button data-testid="delete-account">Delete account</button>
    <button data-testid="hidden-action" style="display:none">Hidden navigation</button>
    <button data-testid="disabled-action" disabled>Disabled navigation</button>
    <a data-testid="external-link" href="https://outside.test/docs">External docs</a>
    <label><input id="logs" type="checkbox"> Reveal logs</label>
    <label for="mode">Mode</label>
    <select id="mode">
      <option value="a" selected>Mode A</option>
      <option value="b">Mode B</option>
    </select>
    <iframe src="/frame"></iframe>
  </div>
  <main id="view"><p>Root view</p></main>
  <script>
    const controls = document.querySelector('#controls');
    const view = document.querySelector('#view');
    const finish = (html) => {
      controls.replaceChildren();
      view.innerHTML = html;
    };

    document.querySelector('[data-testid=show-alpha]').addEventListener('click', () => {
      finish('<section><h2>Alpha view</h2><button data-testid="alpha-child">Show child</button></section>');
      document.querySelector('[data-testid=alpha-child]').addEventListener('click', () => {
        view.innerHTML = '<section><h2>Alpha child view</h2></section>';
      });
    });
    document.querySelector('[data-testid=show-beta]').addEventListener('click', () => {
      finish('<section><h2>Beta view</h2></section>');
    });
    document.querySelector('[data-testid=load-details]').addEventListener('click', async () => {
      controls.replaceChildren();
      view.innerHTML = '<p>Loading network details</p>';
      const response = await fetch('/data');
      const data = await response.json();
      view.innerHTML = `<section><h2>${data.message}</h2></section>`;
    });
    document.querySelector('[data-testid=open-help]').addEventListener('click', () => {
      window.open('/popup', '_blank');
    });
    document.querySelector('#logs').addEventListener('change', () => {
      finish('<section><h2>Logs visible</h2></section>');
    });
    document.querySelector('#mode').addEventListener('change', () => {
      finish('<section><h2>Mode B visible</h2></section>');
    });
    window.addEventListener('message', (event) => {
      if (event.data && event.data.type === 'frame-view') {
        finish('<section><h2>Frame result visible</h2></section>');
      }
    });
  </script>
</body>
</html>
"""


@pytest.fixture
def explorer_portal() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ExplorerFixtureHandler)
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
async def test_real_graph_exploration_restores_parents_and_records_effects(
    tmp_path: Path,
    explorer_portal: str,
) -> None:
    run = ScrapeRun(
        run_id="explorer-run",
        root_url=explorer_portal,
        allowed_origins=(explorer_portal,),
        limits=CrawlLimits(
            maximum_depth=2,
            maximum_states=20,
            maximum_actions=20,
            maximum_runtime_seconds=60,
        ),
        capture_policy=CapturePolicy(
            capture_dom=False,
            capture_screenshots=False,
            capture_accessibility_tree=False,
            capture_trace=False,
            capture_har=False,
        ),
    )
    store = ArtifactStore.create(tmp_path / "crawls", run)
    manager = BrowserManager(
        store,
        BrowserLaunchConfig(headless=True, viewport_width=1280, viewport_height=720),
    )

    try:
        page = await manager.start()
        await manager.navigate(explorer_portal)
        await page.wait_for_selector("iframe")
        extractor = PageStateExtractor(
            store,
            manager.recorder,
            SnapshotConfig(
                quiet_window_ms=250,
                quiet_timeout_ms=3_000,
                full_page_screenshot=False,
            ),
        )
        planner = ActionPlanner(store)
        executor = ActionExecutor(store, manager.recorder, extractor)
        result = await StateGraphExplorer(
            store,
            manager.recorder,
            extractor,
            planner,
            executor,
        ).explore(page)
    finally:
        await manager.stop()

    actions = list(store.iter_records(ActionCandidate))
    transitions = list(store.iter_records(InteractionTransition))
    transitions_by_action = {
        transition.action_id: transition for transition in transitions
    }

    assert result.coverage.bounded_complete is True
    assert result.coverage.actions_pending == 0
    assert result.coverage.actions_failed == 0
    assert result.coverage.actions_succeeded >= 7
    assert len(result.state_ids) >= 8
    assert store.run.status == ScrapeRunStatus.COMPLETED

    review_action = next(
        action for action in actions if action.policy_rule == "semantic.review.run"
    )
    destructive_action = next(
        action for action in actions if action.policy_rule == "semantic.blocked.delete"
    )
    external_action = next(
        action for action in actions if action.policy_rule == "navigation.cross_origin"
    )
    assert review_action.status == ActionStatus.SKIPPED
    assert destructive_action.status == ActionStatus.SKIPPED
    assert external_action.status == ActionStatus.SKIPPED
    assert transitions_by_action[review_action.action_id].status == ActionStatus.SKIPPED
    assert (
        transitions_by_action[destructive_action.action_id].status
        == ActionStatus.SKIPPED
    )
    assert (
        transitions_by_action[external_action.action_id].status == ActionStatus.SKIPPED
    )

    succeeded = [
        transition
        for transition in transitions
        if transition.status == ActionStatus.SUCCEEDED
    ]
    assert any(transition.network_exchange_ids for transition in succeeded)
    assert any(
        any(effect.kind == EffectKind.POPUP_OPENED for effect in transition.effects)
        for transition in succeeded
    )
    assert any(state.title == "Help" for state in store.iter_records(StateSnapshot))
    assert any(
        transition.parent_state_id != result.root_state_id for transition in succeeded
    )
    root_children = {
        transition.resulting_state_id
        for transition in succeeded
        if transition.parent_state_id == result.root_state_id
    }
    assert len(root_children) >= 7

    checkpoint = store.load_checkpoint()
    assert checkpoint.last_state_id in result.state_ids
    assert checkpoint.last_transition_id in result.transition_ids
    assert store.verify_integrity(strict=True).valid is True
