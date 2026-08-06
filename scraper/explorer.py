"""Bounded, restorable state-graph exploration for generic portals."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from urllib.parse import urlsplit

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Frame, Page

from scraper.actions import (
    ActionExecutionError,
    ActionExecutionOutcome,
    ActionExecutor,
    ActionPlanner,
    PlannedAction,
)
from scraper.artifact_store import ArtifactStore
from scraper.extractor import CapturedState, PageStateExtractor
from scraper.models import (
    ActionCandidate,
    ActionRisk,
    ActionStatus,
    BrowserEvent,
    BrowserEventKind,
    CoverageReport,
    EffectKind,
    InteractionTransition,
    ScrapeRunStatus,
    utc_now,
)
from scraper.recorder import BrowserRecorder, RecorderError
from scraper.redaction import redact_text


RESTORE_STORAGE_SCRIPT = r"""
(payload) => {
    localStorage.clear();
    for (const [name, value] of payload.localStorage) localStorage.setItem(name, value);
    sessionStorage.clear();
    for (const [name, value] of payload.sessionStorage) sessionStorage.setItem(name, value);
}
"""


class ExplorationError(RuntimeError):
    """Raised when the graph explorer cannot preserve its invariants."""


@dataclass(frozen=True)
class ExplorerConfig:
    """Runtime controls not already serialized in ``CrawlLimits``."""

    restore_timeout_ms: int = 15_000
    capture_initial_state: bool = True

    def __post_init__(self) -> None:
        if self.restore_timeout_ms <= 0:
            raise ValueError("restore_timeout_ms must be positive")


@dataclass(frozen=True)
class ExplorationResult:
    """In-memory summary of the durable state graph."""

    root_state_id: str
    state_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    transition_ids: tuple[str, ...]
    coverage: CoverageReport
    completion_reason: str


@dataclass(frozen=True)
class _GraphNode:
    capture: CapturedState
    path: tuple[PlannedAction, ...]
    depth: int


@dataclass(frozen=True, repr=False)
class _BrowserCheckpoint:
    root_url: str
    cookies: tuple[dict[str, object], ...]
    local_storage: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    session_storage: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]


@dataclass
class _ExplorationLedger:
    states: dict[str, CapturedState] = field(default_factory=dict)
    actions: list[ActionCandidate] = field(default_factory=list)
    transitions: list[InteractionTransition] = field(default_factory=list)
    duplicate_states: int = 0
    limitations: set[str] = field(default_factory=set)
    sequence: int = 0


class StateGraphExplorer:
    """Explore safe actions from independently restored parent states."""

    def __init__(
        self,
        store: ArtifactStore,
        recorder: BrowserRecorder,
        extractor: PageStateExtractor,
        planner: ActionPlanner,
        executor: ActionExecutor,
        config: Optional[ExplorerConfig] = None,
    ) -> None:
        self.store = store
        self.recorder = recorder
        self.extractor = extractor
        self.planner = planner
        self.executor = executor
        self.config = config or ExplorerConfig()
        self.limits = store.run.limits

    async def explore(self, root_page: Page) -> ExplorationResult:
        """Build a bounded graph and finalize manifest, checkpoint, and coverage."""

        started_at = utc_now()
        running_run = self.store.run.model_copy(
            update={
                "status": ScrapeRunStatus.RUNNING,
                "started_at": started_at,
                "ended_at": None,
                "completion_reason": None,
            }
        )
        await asyncio.to_thread(self.store.save_manifest, running_run)

        ledger = _ExplorationLedger()
        try:
            checkpoint = await self._capture_checkpoint(root_page)
            if not self.config.capture_initial_state:
                raise ExplorationError(
                    "an initial state is required for graph exploration"
                )
            root_capture = await self.extractor.capture(
                root_page,
                sequence=ledger.sequence,
            )
        except Exception as exc:
            await self._finalize_failed_run(started_at, ledger, str(exc))
            raise
        ledger.sequence += 1
        ledger.states[root_capture.state.fingerprint] = root_capture
        frontier: deque[_GraphNode] = deque(
            [_GraphNode(capture=root_capture, path=(), depth=0)]
        )
        navigation_frontier: deque[_GraphNode] = deque()
        observation_frontier: deque[_GraphNode] = deque()
        completion_reason = "frontier exhausted"
        runtime_limited = False
        attempted_actions = 0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.limits.maximum_runtime_seconds

        try:
            while observation_frontier or navigation_frontier or frontier:
                if loop.time() >= deadline:
                    completion_reason = "maximum runtime reached"
                    ledger.limitations.add(completion_reason)
                    runtime_limited = True
                    break
                if observation_frontier:
                    node = observation_frontier.popleft()
                elif navigation_frontier:
                    node = navigation_frontier.popleft()
                else:
                    node = frontier.popleft()
                planned_actions = await self.planner.plan(node.capture)
                ledger.actions.extend(planned.candidate for planned in planned_actions)

                for planned in planned_actions:
                    if loop.time() >= deadline:
                        completion_reason = "maximum runtime reached"
                        ledger.limitations.add(completion_reason)
                        runtime_limited = True
                        break

                    limit_reason = self._limit_reason(
                        node,
                        attempted_actions=attempted_actions,
                        discovered_states=len(ledger.states),
                    )
                    if planned.candidate.status != ActionStatus.PENDING:
                        outcome = await self.executor.skip(
                            root_page,
                            planned,
                            node.capture.state,
                            reason=planned.candidate.rationale,
                        )
                    elif limit_reason is not None:
                        ledger.limitations.add(limit_reason)
                        outcome = await self.executor.skip(
                            root_page,
                            planned,
                            node.capture.state,
                            reason=limit_reason,
                        )
                    else:
                        attempted_actions += 1
                        outcome = await self._execute_from_parent(
                            root_page,
                            checkpoint,
                            node,
                            planned,
                            sequence=ledger.sequence,
                        )
                        if outcome.capture is not None:
                            ledger.sequence += 1

                    ledger.transitions.append(outcome.transition)
                    capture = outcome.capture
                    if capture is None:
                        continue
                    existing = ledger.states.get(capture.state.fingerprint)
                    if existing is not None:
                        ledger.duplicate_states += 1
                        continue
                    ledger.states[capture.state.fingerprint] = capture
                    if planned.candidate.risk == ActionRisk.REVIEW_REQUIRED:
                        ledger.limitations.add(
                            "review-required action results are terminal branches "
                            "to prevent repeated external side effects"
                        )
                        continue
                    child = _GraphNode(
                        capture=capture,
                        path=(*node.path, planned),
                        depth=node.depth + 1,
                    )
                    navigated = any(
                        effect.kind == EffectKind.NAVIGATION
                        for effect in outcome.transition.effects
                    )
                    if navigated and _has_row_observation_controls(capture):
                        observation_frontier.append(child)
                    elif navigated:
                        navigation_frontier.append(child)
                    else:
                        frontier.append(child)
                if runtime_limited:
                    break
        except Exception as exc:
            await self._finalize_failed_run(started_at, ledger, str(exc))
            raise

        coverage = await self._coverage(
            ledger,
            bounded_complete=not runtime_limited,
            completion_reason=completion_reason,
        )
        await asyncio.to_thread(self.store.append_record, coverage)
        ended_at = utc_now()
        unique_states = tuple(
            capture.state.state_id for capture in ledger.states.values()
        )
        transition_ids = tuple(
            transition.transition_id for transition in ledger.transitions
        )
        final_status = (
            ScrapeRunStatus.COMPLETED
            if coverage.bounded_complete
            else ScrapeRunStatus.PARTIAL
        )
        final_run = self.store.run.model_copy(
            update={
                "status": final_status,
                "started_at": started_at,
                "ended_at": ended_at,
                "completion_reason": completion_reason,
                "state_ids": unique_states,
                "transition_ids": transition_ids,
            }
        )
        await asyncio.to_thread(self.store.save_manifest, final_run)
        await asyncio.to_thread(
            self.store.write_checkpoint,
            final_run.checkpoint_sequence + 1,
            last_state_id=unique_states[-1] if unique_states else None,
            last_transition_id=transition_ids[-1] if transition_ids else None,
        )
        return ExplorationResult(
            root_state_id=root_capture.state.state_id,
            state_ids=unique_states,
            action_ids=tuple(action.action_id for action in ledger.actions),
            transition_ids=transition_ids,
            coverage=coverage,
            completion_reason=completion_reason,
        )

    def _limit_reason(
        self,
        node: _GraphNode,
        *,
        attempted_actions: int,
        discovered_states: int,
    ) -> Optional[str]:
        if node.depth >= self.limits.maximum_depth:
            return "maximum graph depth reached"
        if attempted_actions >= self.limits.maximum_actions:
            return "maximum executed actions reached"
        if discovered_states >= self.limits.maximum_states:
            return "maximum discovered states reached"
        return None

    async def _execute_from_parent(
        self,
        root_page: Page,
        checkpoint: _BrowserCheckpoint,
        node: _GraphNode,
        planned: PlannedAction,
        *,
        sequence: int,
    ) -> ActionExecutionOutcome:
        try:
            active_page = await self._restore_parent(
                root_page,
                checkpoint,
                node.path,
            )
        except (
            ActionExecutionError,
            ExplorationError,
            PlaywrightError,
            RecorderError,
        ) as exc:
            return await self.executor.fail(
                root_page,
                planned,
                node.capture.state,
                error=f"parent-state restoration failed: {exc}",
            )
        return await self.executor.execute(
            active_page,
            planned,
            node.capture,
            sequence=sequence,
        )

    async def _capture_checkpoint(self, root_page: Page) -> _BrowserCheckpoint:
        state = await root_page.context.storage_state()
        raw_cookies = state.get("cookies", []) if isinstance(state, dict) else []
        cookies = tuple(
            {str(key): value for key, value in cookie.items()}
            for cookie in raw_cookies
            if isinstance(cookie, dict)
        )
        raw_origins = state.get("origins", []) if isinstance(state, dict) else []
        local_storage: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        for origin in raw_origins if isinstance(raw_origins, list) else []:
            if not isinstance(origin, dict):
                continue
            items = origin.get("localStorage", [])
            pairs = tuple(
                sorted(
                    (
                        (str(item.get("name", "")), str(item.get("value", "")))
                        for item in items
                        if isinstance(item, dict)
                    ),
                    key=lambda pair: pair[0],
                )
            )
            local_storage.append((str(origin.get("origin", "")), pairs))

        sessions: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        for frame_path, frame in self._ordered_frames(root_page.main_frame):
            try:
                values = await frame.evaluate("() => Object.entries(sessionStorage)")
            except PlaywrightError:
                values = []
            pairs = tuple(
                sorted(
                    (
                        (str(item[0]), str(item[1]))
                        for item in values
                        if isinstance(item, list) and len(item) == 2
                    ),
                    key=lambda pair: pair[0],
                )
            )
            sessions.append((frame_path, pairs))
        return _BrowserCheckpoint(
            root_url=root_page.url,
            cookies=cookies,
            local_storage=tuple(sorted(local_storage, key=lambda item: item[0])),
            session_storage=tuple(sessions),
        )

    async def _restore_parent(
        self,
        root_page: Page,
        checkpoint: _BrowserCheckpoint,
        path: tuple[PlannedAction, ...],
    ) -> Page:
        if root_page.is_closed():
            raise ExplorationError("root page was closed and cannot be restored")
        for page in tuple(root_page.context.pages):
            if page != root_page and not page.is_closed():
                await page.close()
        await root_page.context.clear_cookies()
        if checkpoint.cookies:
            await root_page.context.add_cookies(list(checkpoint.cookies))  # type: ignore[arg-type]
        await root_page.goto(
            checkpoint.root_url,
            wait_until="load",
            timeout=self.config.restore_timeout_ms,
        )
        local_by_origin = dict(checkpoint.local_storage)
        sessions_by_path = dict(checkpoint.session_storage)
        for frame_path, frame in self._ordered_frames(root_page.main_frame):
            origin = _origin(frame.url)
            payload = {
                "localStorage": list(local_by_origin.get(origin or "", ())),
                "sessionStorage": list(sessions_by_path.get(frame_path, ())),
            }
            try:
                await frame.evaluate(RESTORE_STORAGE_SCRIPT, payload)
            except PlaywrightError:
                continue
        await root_page.reload(
            wait_until="load",
            timeout=self.config.restore_timeout_ms,
        )
        await self.extractor.wait_for_quiet(root_page)
        await self.recorder.drain_mutations(root_page)
        await self.recorder.flush()

        active_page = root_page
        for step in path:
            active_page = await self.executor.replay(active_page, step)
        return active_page

    async def _coverage(
        self,
        ledger: _ExplorationLedger,
        *,
        bounded_complete: bool,
        completion_reason: str,
    ) -> CoverageReport:
        transitioned = {transition.action_id for transition in ledger.transitions}
        pending_ids = tuple(
            sorted(
                {
                    action.action_id
                    for action in ledger.actions
                    if action.action_id not in transitioned
                }
            )
        )
        status_counts = {
            status: sum(
                transition.status == status for transition in ledger.transitions
            )
            for status in (
                ActionStatus.SUCCEEDED,
                ActionStatus.FAILED,
                ActionStatus.SKIPPED,
            )
        }
        captures = tuple(ledger.states.values())
        elements = [element for capture in captures for element in capture.elements]
        action_ids = {transition.action_id for transition in ledger.transitions}
        events = tuple(
            event
            for event in self.store.iter_records(BrowserEvent)
            if event.action_id in action_ids
        )
        limitations = set(ledger.limitations)
        limitations.add(
            "restoration cannot reset storage for origins absent from the root state"
        )
        limitations.add(
            "replayed parent paths are locator-validated but are not fingerprint-"
            "validated before the next action"
        )
        if any(
            frame.is_cross_origin
            for capture in captures
            for frame in capture.state.frames
        ):
            limitations.add(
                "cross-origin frames are restored only when present in the root frame tree"
            )
        return CoverageReport(
            run_id=self.store.run.run_id,
            states_discovered=len(captures),
            duplicate_states=ledger.duplicate_states,
            elements_discovered=len(elements),
            action_candidates=len(ledger.actions),
            actions_succeeded=status_counts[ActionStatus.SUCCEEDED],
            actions_failed=status_counts[ActionStatus.FAILED],
            actions_skipped=status_counts[ActionStatus.SKIPPED],
            actions_pending=len(pending_ids),
            routes_discovered=len({capture.state.url for capture in captures}),
            frames_discovered=sum(len(capture.state.frames) for capture in captures),
            tables_discovered=len(_table_identities(captures)),
            modals_discovered=sum(capture.state.modal_count for capture in captures),
            downloads_observed=sum(
                event.kind == BrowserEventKind.DOWNLOAD for event in events
            ),
            console_errors=sum(
                event.kind == BrowserEventKind.CONSOLE
                and event.summary.lower().startswith("console.error")
                for event in events
            ),
            page_errors=sum(
                event.kind == BrowserEventKind.PAGE_ERROR for event in events
            ),
            unexplored_action_ids=pending_ids,
            limitations=tuple(sorted(limitations)),
            bounded_complete=bounded_complete and not pending_ids,
            completion_reason=completion_reason,
        )

    async def _finalize_failed_run(
        self,
        started_at: datetime,
        ledger: _ExplorationLedger,
        error: str,
    ) -> None:
        safe_error = (
            redact_text(
                error,
                self.store.run.capture_policy.redacted_names,
            )
            or "graph exploration failed"
        )
        failed_run = self.store.run.model_copy(
            update={
                "status": ScrapeRunStatus.FAILED,
                "started_at": started_at,
                "ended_at": utc_now(),
                "completion_reason": safe_error,
                "state_ids": tuple(
                    capture.state.state_id for capture in ledger.states.values()
                ),
                "transition_ids": tuple(
                    transition.transition_id for transition in ledger.transitions
                ),
            }
        )
        await asyncio.to_thread(self.store.save_manifest, failed_run)

    @staticmethod
    def _ordered_frames(main_frame: Frame) -> list[tuple[str, Frame]]:
        result: list[tuple[str, Frame]] = []

        def visit(frame: Frame, path: str) -> None:
            result.append((path, frame))
            active_children = [
                child for child in frame.child_frames if not child.is_detached()
            ]
            for index, child in enumerate(active_children):
                visit(child, f"{path}/{index}")

        visit(main_frame, "main")
        return result


def _table_identities(
    captures: tuple[CapturedState, ...],
) -> set[tuple[str, ...]]:
    """Identify named and anonymous HTML/ARIA tables across captured routes."""

    identities: set[tuple[str, ...]] = set()
    for capture in captures:
        for element in capture.elements:
            if element.context.table_id:
                identities.add(("named", element.frame_id, element.context.table_id))
            elif element.tag == "table" or (element.role or "").lower() == "table":
                identities.add(
                    (
                        "anonymous",
                        capture.state.url,
                        element.frame_id,
                        element.element_id,
                    )
                )
    return identities


def _has_row_observation_controls(capture: CapturedState) -> bool:
    """Detect table controls that reveal context without leaving the page."""

    inert_destinations = {"", "#", "javascript:void(0)"}
    for element in capture.elements:
        if element.context.row_label is None or element.tag != "a":
            continue
        href = next(
            (
                attribute.value or attribute.safe_value or ""
                for attribute in element.attributes
                if attribute.name.lower() == "href"
            ),
            None,
        )
        if href is not None and href.strip().lower() in inert_destinations:
            return True
    return False


def _origin(url: str) -> Optional[str]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


__all__ = [
    "ExplorerConfig",
    "ExplorationError",
    "ExplorationResult",
    "StateGraphExplorer",
]
