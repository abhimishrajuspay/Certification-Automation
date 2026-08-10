"""Real-browser verification for bounded Phase 5 graph exploration."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from knowledge.builder import PortalKnowledgeBuilder
from scraper.actions import ActionExecutor, ActionPlanner
from scraper.artifact_store import ArtifactStore
from scraper.browser import BrowserLaunchConfig, BrowserManager
from scraper.explorer import ExplorerConfig, ExplorationWorker, StateGraphExplorer
from scraper.extractor import PageStateExtractor, SnapshotConfig
from scraper.guidance import (
    CrawlGuide,
    CrawlStrategy,
    GuideRepeatRule,
    GuideStep,
    GuideTarget,
    ParallelSessionMode,
)
from scraper.models import (
    ActionCandidate,
    ActionStatus,
    BrowserEvent,
    CapturePolicy,
    CrawlCompletionGoal,
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
        elif path == "/testcase-context":
            self._send_html(_TESTCASE_CONTEXT_HTML)
        elif path == "/guided-entry":
            self._send_html(_GUIDED_ENTRY_HTML)
        elif path == "/spa-guided":
            self._send_html(_SPA_GUIDED_HTML)
        elif path == "/guided-pagination-entry":
            self._send_html(_GUIDED_PAGINATION_ENTRY_HTML)
        elif path == "/guided-paginated":
            page = int(parse_qs(urlsplit(self.path).query).get("page", ["1"])[0])
            self._send_html(_guided_paginated_html(page))
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
  <table><tbody><tr><td>Anonymous table evidence</td></tr></tbody></table>
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


_TESTCASE_CONTEXT_HTML = """<!doctype html>
<html>
<head><title>Testcase context fixture</title></head>
<body>
  <table>
    <thead><tr><th>API Name</th><th>Total TCs</th></tr></thead>
    <tbody><tr><td>Payments</td><td>2</td></tr></tbody>
  </table>
  <table>
    <thead><tr><th>TC ID</th><th>API Name</th><th>Status</th><th>Info</th></tr></thead>
    <tbody>
      <tr><td>TC_01</td><td>Payments</td><td>pending</td><td><button data-case="TC_01" aria-label="Information for TC_01">i</button></td></tr>
      <tr><td>TC_02</td><td>Payments</td><td>pending</td><td><button data-case="TC_02" aria-label="Information for TC_02">i</button></td></tr>
    </tbody>
  </table>
  <div id="modal-root"></div>
  <script>
    const modalRoot = document.querySelector('#modal-root');
    for (const button of document.querySelectorAll('button[data-case]')) {
      button.addEventListener('click', () => {
        const caseId = button.dataset.case;
        modalRoot.innerHTML = `<div role="dialog" aria-label="Test Case Details"><div role="document" class="modal-body">${caseId}: validates the payment flow</div><button aria-label="Close details">Close</button></div>`;
        modalRoot.querySelector('button').addEventListener('click', () => {
          modalRoot.replaceChildren();
        });
      });
    }
  </script>
</body>
</html>
"""


_GUIDED_ENTRY_HTML = """<!doctype html>
<html>
<head><title>Guided entry fixture</title></head>
<body>
  <nav><a href="/">Home</a></nav>
  <main>
    <h1>Certification portal</h1>
    <button data-testid="open-testcases" onclick="location.href='/testcase-context'">Open testcases</button>
  </main>
</body>
</html>
"""


_SPA_GUIDED_HTML = """<!doctype html>
<html>
<head><title>SPA guided fixture</title></head>
<body>
  <main><button data-testid="open-spa-testcases">Open SPA testcases</button></main>
  <script>
    document.querySelector('[data-testid=open-spa-testcases]').addEventListener('click', () => {
      document.body.innerHTML = `
        <main>
          <table><thead><tr><th>API Name</th><th>Total TCs</th></tr></thead>
            <tbody><tr><td>Payments</td><td>2</td></tr></tbody></table>
          <table><thead><tr><th>TC ID</th><th>API Name</th><th>Status</th><th>Info</th></tr></thead>
            <tbody>
              <tr><td>TC_01</td><td>Payments</td><td>pending</td><td><button data-case="TC_01" aria-label="Information for TC_01">i</button></td></tr>
              <tr><td>TC_02</td><td>Payments</td><td>pending</td><td><button data-case="TC_02" aria-label="Information for TC_02">i</button></td></tr>
            </tbody>
          </table>
          <div id="modal-root"></div>
        </main>`;
      const modalRoot = document.querySelector('#modal-root');
      for (const button of document.querySelectorAll('button[data-case]')) {
        button.addEventListener('click', () => {
          const caseId = button.dataset.case;
          modalRoot.innerHTML = `<div role="dialog" aria-label="Test Case Details"><div role="document">${caseId}: validates the SPA payment flow</div><button aria-label="Close details">Close</button></div>`;
          modalRoot.querySelector('button').addEventListener('click', () => modalRoot.replaceChildren());
        });
      }
    });
  </script>
</body>
</html>
"""


_GUIDED_PAGINATION_ENTRY_HTML = """<!doctype html>
<html>
<head><title>Paginated guide entry</title></head>
<body><main>
  <table>
    <thead><tr><th>API Name</th><th>Total TCs</th><th>Test</th></tr></thead>
    <tbody>
      <tr><td>Payments</td><td>12</td><td><a data-testid="open-paginated" href="/guided-paginated?page=1">Test</a></td></tr>
      <tr><td>Refunds</td><td>9</td><td><a href="/unused">Test</a></td></tr>
    </tbody>
  </table>
</main></body>
</html>
"""


def _guided_paginated_html(page: int) -> str:
    start = 1 if page <= 1 else 11
    end = 10 if page <= 1 else 12
    rows = "".join(
        f"""<tr><td>{index}</td><td>TC_PAGE_{index:02d}</td><td>Payments</td><td>pending</td>
        <td><button data-case="TC_PAGE_{index:02d}" title="Test case details">i</button></td></tr>"""
        for index in range(start, end + 1)
    )
    pager = (
        '<a href="/guided-paginated?page=2" rel="next" data-page="2">Next</a>'
        if page <= 1
        else '<a href="/guided-paginated?page=1" rel="prev">Previous</a>'
    )
    return f"""<!doctype html>
<html>
<head><title>Paginated visual-modal fixture</title>
<style>
  .hidden {{ display: none; }}
  .modal-layer {{ position: fixed; inset: 0; background: rgba(0,0,0,.4); }}
  .modal {{ width: 500px; margin: 50px auto; padding: 20px; background: white; }}
</style></head>
<body><main>
  <table>
    <thead><tr><th>#</th><th>TC ID</th><th>API Name</th><th>Status</th><th>Info</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <div class="showing">Showing {start} to {end} of 12 entries</div>
  <nav>{pager}</nav>
  <div data-modal class="modal-layer hidden">
    <section class="modal">
      <header><span class="modal-title">Test Case Details</span><button class="close">×</button></header>
      <div class="modal-body"></div>
    </section>
  </div>
</main>
<script>
  const layer = document.querySelector('[data-modal]');
  const body = document.querySelector('.modal-body');
  for (const button of document.querySelectorAll('button[data-case]')) {{
    button.addEventListener('click', () => {{
      const caseId = button.dataset.case;
      body.textContent = `${{caseId}}: validates a paginated CSS-modal payment flow`;
      layer.classList.remove('hidden');
    }});
  }}
  document.querySelector('button.close').addEventListener('click', () => {{
    layer.classList.add('hidden');
    body.textContent = '';
  }});
</script>
</body></html>"""


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
    assert result.coverage.tables_discovered == 1
    assert len(result.state_ids) >= 8
    assert store.run.status == ScrapeRunStatus.COMPLETED

    review_action = next(
        action
        for action in actions
        if action.policy_rule == "semantic.review.test_execution"
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


@pytest.mark.skipif(
    not RUN_BROWSER_TESTS,
    reason="set CZ_RUN_BROWSER_TESTS=1 to run real Chromium verification",
)
@pytest.mark.asyncio
async def test_testcase_context_goal_stops_with_success_before_frontier_exhaustion(
    tmp_path: Path,
    explorer_portal: str,
) -> None:
    url = f"{explorer_portal}/testcase-context"
    run = ScrapeRun(
        run_id="testcase-context-run",
        root_url=url,
        allowed_origins=(explorer_portal,),
        limits=CrawlLimits(
            maximum_depth=5,
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
        await manager.navigate(url)
        extractor = PageStateExtractor(
            store,
            manager.recorder,
            SnapshotConfig(
                quiet_window_ms=100,
                quiet_timeout_ms=2_000,
                full_page_screenshot=False,
            ),
        )
        result = await StateGraphExplorer(
            store,
            manager.recorder,
            extractor,
            ActionPlanner(store),
            ActionExecutor(store, manager.recorder, extractor),
            ExplorerConfig(
                completion_goal=CrawlCompletionGoal.TESTCASE_CONTEXT,
                testcase_context_stability_observations=2,
            ),
        ).explore(page)
    finally:
        await manager.stop()

    context = result.coverage.testcase_context
    assert context is not None
    assert result.completion_reason == "testcase context complete"
    assert result.coverage.configured_goal_complete is True
    assert result.coverage.bounded_complete is False
    assert context.declared_test_cases == 2
    assert context.test_cases_discovered == 2
    assert context.descriptions_captured == 2
    assert context.stable_for_early_stop is True
    assert store.run.status == ScrapeRunStatus.COMPLETED


@pytest.mark.skipif(
    not RUN_BROWSER_TESTS,
    reason="set CZ_RUN_BROWSER_TESTS=1 to run real Chromium verification",
)
@pytest.mark.asyncio
async def test_guided_route_promotes_testcase_root_and_sweeps_rows_in_place(
    tmp_path: Path,
    explorer_portal: str,
) -> None:
    entry_url = f"{explorer_portal}/guided-entry"
    run = ScrapeRun(
        run_id="guided-root-run",
        root_url=entry_url,
        allowed_origins=(explorer_portal,),
        limits=CrawlLimits(
            maximum_depth=5,
            maximum_states=30,
            maximum_actions=30,
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
    guide = CrawlGuide(
        steps=(
            GuideStep(
                name="open_testcases",
                target=GuideTarget(
                    test_id="open-testcases",
                    test_id_attribute="data-testid",
                ),
            ),
        ),
        repeat_rules=(
            GuideRepeatRule(
                name="open_each_description",
                target=GuideTarget(tag="button", role="button", text="i"),
                close_target=GuideTarget(
                    role="button",
                    accessible_name="Close details",
                ),
            ),
        ),
        root_scope_selector=None,
    )

    try:
        page = await manager.start()
        await manager.navigate(entry_url)
        extractor = PageStateExtractor(
            store,
            manager.recorder,
            SnapshotConfig(
                quiet_window_ms=100,
                quiet_timeout_ms=2_000,
                full_page_screenshot=False,
            ),
        )
        result = await StateGraphExplorer(
            store,
            manager.recorder,
            extractor,
            ActionPlanner(store),
            ActionExecutor(store, manager.recorder, extractor),
            ExplorerConfig(
                completion_goal=CrawlCompletionGoal.TESTCASE_CONTEXT,
                testcase_context_stability_observations=2,
                strategy=CrawlStrategy.HYBRID,
            ),
            guide=guide,
        ).explore(page)
    finally:
        await manager.stop()

    context = result.coverage.testcase_context
    root = next(
        state
        for state in store.iter_records(StateSnapshot)
        if state.state_id == result.root_state_id
    )
    actions = tuple(store.iter_records(ActionCandidate))
    transitions = tuple(store.iter_records(InteractionTransition))

    assert root.url.endswith("/testcase-context")
    assert context is not None
    assert context.test_cases_discovered == 2
    assert context.descriptions_captured == 2
    assert context.stable_for_early_stop is True
    assert result.completion_reason == "testcase context complete"
    assert all(action.status == ActionStatus.PENDING for action in actions)
    assert len(actions) == 5
    assert len(transitions) == 5
    assert all(
        transition.status == ActionStatus.SUCCEEDED for transition in transitions
    )
    assert any(
        "promoted to exploration root" in limitation
        for limitation in result.coverage.limitations
    )


@pytest.mark.skipif(
    not RUN_BROWSER_TESTS,
    reason="set CZ_RUN_BROWSER_TESTS=1 to run real Chromium verification",
)
@pytest.mark.asyncio
async def test_parallel_worker_probe_fans_out_safe_root_siblings(
    tmp_path: Path,
    explorer_portal: str,
) -> None:
    run = ScrapeRun(
        run_id="parallel-root-run",
        root_url=explorer_portal,
        allowed_origins=(explorer_portal,),
        limits=CrawlLimits(
            maximum_depth=1,
            maximum_states=20,
            maximum_actions=4,
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
    managers = tuple(
        BrowserManager(
            store,
            BrowserLaunchConfig(
                headless=True,
                viewport_width=1280,
                viewport_height=720,
            ),
        )
        for _ in range(2)
    )

    try:
        primary_page = await managers[0].start()
        await managers[0].navigate(explorer_portal)
        secondary_page = await managers[1].start()
        primary_extractor = PageStateExtractor(
            store,
            managers[0].recorder,
            SnapshotConfig(
                quiet_window_ms=100,
                quiet_timeout_ms=2_000,
                full_page_screenshot=False,
            ),
        )
        secondary_extractor = PageStateExtractor(
            store,
            managers[1].recorder,
            SnapshotConfig(
                quiet_window_ms=100,
                quiet_timeout_ms=2_000,
                full_page_screenshot=False,
            ),
        )
        result = await StateGraphExplorer(
            store,
            managers[0].recorder,
            primary_extractor,
            ActionPlanner(store),
            ActionExecutor(store, managers[0].recorder, primary_extractor),
            ExplorerConfig(
                worker_count=2,
                parallel_session_mode=ParallelSessionMode.PROBE,
            ),
            additional_workers=(
                ExplorationWorker(
                    worker_id="worker-2",
                    page=secondary_page,
                    recorder=managers[1].recorder,
                    extractor=secondary_extractor,
                    executor=ActionExecutor(
                        store,
                        managers[1].recorder,
                        secondary_extractor,
                    ),
                ),
            ),
        ).explore(primary_page)
    finally:
        for manager in reversed(managers):
            await manager.stop()

    transitions = tuple(
        transition
        for transition in store.iter_records(InteractionTransition)
        if transition.status == ActionStatus.SUCCEEDED
    )
    event_by_id = {event.event_id: event for event in store.iter_records(BrowserEvent)}
    action_page_ids = {
        event_by_id[event_id].page_id
        for transition in transitions
        for event_id in transition.event_ids
        if event_id in event_by_id and event_by_id[event_id].page_id is not None
    }

    assert result.coverage.actions_succeeded == 4
    assert len(action_page_ids) >= 2
    assert any(
        "used 2 isolated browser workers" in limitation
        for limitation in result.coverage.limitations
    )


@pytest.mark.skipif(
    not RUN_BROWSER_TESTS,
    reason="set CZ_RUN_BROWSER_TESTS=1 to run real Chromium verification",
)
@pytest.mark.asyncio
async def test_parallel_workers_partition_guided_testcase_rows(
    tmp_path: Path,
    explorer_portal: str,
) -> None:
    url = f"{explorer_portal}/spa-guided"
    run = ScrapeRun(
        run_id="parallel-row-run",
        root_url=url,
        allowed_origins=(explorer_portal,),
        limits=CrawlLimits(
            maximum_depth=5,
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
    managers = tuple(
        BrowserManager(
            store,
            BrowserLaunchConfig(
                headless=True,
                viewport_width=1280,
                viewport_height=720,
            ),
        )
        for _ in range(2)
    )
    guide = CrawlGuide(
        steps=(
            GuideStep(
                name="open_spa_testcases",
                target=GuideTarget(
                    test_id="open-spa-testcases",
                    test_id_attribute="data-testid",
                ),
            ),
        ),
        repeat_rules=(
            GuideRepeatRule(
                name="parallel_descriptions",
                target=GuideTarget(tag="button", role="button", text="i"),
                close_target=GuideTarget(
                    role="button",
                    accessible_name="Close details",
                ),
            ),
        ),
        root_scope_selector=None,
    )

    try:
        primary_page = await managers[0].start()
        await managers[0].navigate(url)
        secondary_page = await managers[1].start()
        extractors = tuple(
            PageStateExtractor(
                store,
                manager.recorder,
                SnapshotConfig(
                    quiet_window_ms=100,
                    quiet_timeout_ms=2_000,
                    full_page_screenshot=False,
                ),
            )
            for manager in managers
        )
        result = await StateGraphExplorer(
            store,
            managers[0].recorder,
            extractors[0],
            ActionPlanner(store),
            ActionExecutor(store, managers[0].recorder, extractors[0]),
            ExplorerConfig(
                completion_goal=CrawlCompletionGoal.TESTCASE_CONTEXT,
                worker_count=2,
                parallel_session_mode=ParallelSessionMode.PROBE,
            ),
            guide=guide,
            additional_workers=(
                ExplorationWorker(
                    worker_id="worker-2",
                    page=secondary_page,
                    recorder=managers[1].recorder,
                    extractor=extractors[1],
                    executor=ActionExecutor(
                        store,
                        managers[1].recorder,
                        extractors[1],
                    ),
                ),
            ),
        ).explore(primary_page)
    finally:
        for manager in reversed(managers):
            await manager.stop()

    context = result.coverage.testcase_context
    transitions = tuple(store.iter_records(InteractionTransition))
    events = {event.event_id: event for event in store.iter_records(BrowserEvent)}
    page_ids = {
        events[event_id].page_id
        for transition in transitions
        for event_id in transition.event_ids
        if event_id in events and events[event_id].page_id is not None
    }

    assert context is not None and context.stable_for_early_stop is True
    assert context.descriptions_captured == 2
    assert result.coverage.actions_succeeded == 5
    assert len(page_ids) >= 2
    assert any(
        "row sweep distributed across 2 workers" in limitation
        for limitation in result.coverage.limitations
    )
    assert any(
        "URL drift falls back" in limitation
        for limitation in result.coverage.limitations
    )


@pytest.mark.skipif(
    not RUN_BROWSER_TESTS,
    reason="set CZ_RUN_BROWSER_TESTS=1 to run real Chromium verification",
)
@pytest.mark.asyncio
async def test_guided_parallel_pagination_and_visual_modals_complete_context(
    tmp_path: Path,
    explorer_portal: str,
) -> None:
    entry_url = f"{explorer_portal}/guided-pagination-entry"
    run = ScrapeRun(
        run_id="guided-pagination-run",
        root_url=entry_url,
        allowed_origins=(explorer_portal,),
        limits=CrawlLimits(
            maximum_depth=5,
            maximum_states=60,
            maximum_actions=40,
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
    managers = tuple(
        BrowserManager(
            store,
            BrowserLaunchConfig(
                headless=True,
                viewport_width=1280,
                viewport_height=720,
            ),
        )
        for _ in range(2)
    )
    guide = CrawlGuide(
        steps=(
            GuideStep(
                name="open_paginated_testcases",
                target=GuideTarget(
                    test_id="open-paginated",
                    test_id_attribute="data-testid",
                ),
            ),
        ),
        repeat_rules=(
            GuideRepeatRule(
                name="capture_paginated_descriptions",
                target=GuideTarget(
                    tag="button",
                    role="button",
                    title="Test case details",
                ),
                close_target=GuideTarget(
                    tag="button",
                    role="button",
                    text="×",
                ),
            ),
        ),
        root_scope_selector="main",
    )

    try:
        primary_page = await managers[0].start()
        await managers[0].navigate(entry_url)
        secondary_page = await managers[1].start()
        extractors = tuple(
            PageStateExtractor(
                store,
                manager.recorder,
                SnapshotConfig(
                    quiet_window_ms=100,
                    quiet_timeout_ms=2_000,
                    full_page_screenshot=False,
                ),
            )
            for manager in managers
        )
        result = await StateGraphExplorer(
            store,
            managers[0].recorder,
            extractors[0],
            ActionPlanner(store),
            ActionExecutor(store, managers[0].recorder, extractors[0]),
            ExplorerConfig(
                completion_goal=CrawlCompletionGoal.TESTCASE_CONTEXT,
                testcase_context_stability_observations=2,
                strategy=CrawlStrategy.GUIDED,
                worker_count=2,
                parallel_session_mode=ParallelSessionMode.PROBE,
            ),
            guide=guide,
            additional_workers=(
                ExplorationWorker(
                    worker_id="worker-2",
                    page=secondary_page,
                    recorder=managers[1].recorder,
                    extractor=extractors[1],
                    executor=ActionExecutor(
                        store,
                        managers[1].recorder,
                        extractors[1],
                    ),
                ),
            ),
        ).explore(primary_page)
    finally:
        for manager in reversed(managers):
            await manager.stop()

    context = result.coverage.testcase_context
    actions = tuple(store.iter_records(ActionCandidate))
    knowledge = PortalKnowledgeBuilder(store).build()

    assert context is not None
    assert result.completion_reason == "testcase context complete"
    assert result.coverage.configured_goal_complete is True
    assert context.declared_test_cases == 12
    assert context.test_cases_discovered == 12
    assert context.descriptions_captured == 12
    assert context.stable_for_early_stop is True
    assert result.coverage.actions_succeeded == 26
    assert result.coverage.actions_failed == 0
    assert result.coverage.actions_skipped == 0
    assert result.coverage.modals_discovered >= 12
    assert not any(
        action.policy_rule == "semantic.review.test_execution" for action in actions
    )
    assert any(
        "followed 1 pagination transition" in limitation
        for limitation in result.coverage.limitations
    )
    assert knowledge.coverage.declared_test_cases == 12
    assert knowledge.coverage.test_cases_normalized == 12
    assert knowledge.coverage.descriptions_captured == 12
    assert knowledge.coverage.testcase_context_complete is True
