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
from scraper.guidance import (
    CrawlGuide,
    CrawlStrategy,
    GuideRepeatRule,
    GuideStep,
    ParallelSessionMode,
    expectation_errors,
    target_elements,
)
from scraper.models import (
    ActionCandidate,
    ActionKind,
    ActionRisk,
    ActionStatus,
    BrowserEvent,
    BrowserEventKind,
    CoverageReport,
    CrawlCompletionGoal,
    EffectKind,
    InteractionTransition,
    ScrapeRunStatus,
    TestcaseContextCoverage,
    utc_now,
)
from scraper.recorder import BrowserRecorder, RecorderError
from scraper.redaction import redact_text
from scraper.testcase_context import TestcaseContextTracker


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
    completion_goal: CrawlCompletionGoal = CrawlCompletionGoal.BOUNDED_FRONTIER
    testcase_context_stability_observations: int = 2
    strategy: CrawlStrategy = CrawlStrategy.EXHAUSTIVE
    worker_count: int = 1
    parallel_session_mode: ParallelSessionMode = ParallelSessionMode.OFF

    def __post_init__(self) -> None:
        if self.restore_timeout_ms <= 0:
            raise ValueError("restore_timeout_ms must be positive")
        if self.testcase_context_stability_observations <= 0:
            raise ValueError("testcase_context_stability_observations must be positive")
        if self.worker_count <= 0 or self.worker_count > 32:
            raise ValueError("worker_count must be between 1 and 32")
        if (
            self.worker_count == 1
            and self.parallel_session_mode != ParallelSessionMode.OFF
        ):
            raise ValueError("single-worker exploration requires parallel mode off")
        if (
            self.worker_count > 1
            and self.parallel_session_mode == ParallelSessionMode.OFF
        ):
            raise ValueError("multiple workers require probe or force parallel mode")


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
class ExplorationWorker:
    """One isolated browser/recorder stack available to execute graph branches."""

    worker_id: str
    page: Page
    recorder: BrowserRecorder
    extractor: PageStateExtractor
    executor: ActionExecutor


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
        *,
        guide: Optional[CrawlGuide] = None,
        additional_workers: tuple[ExplorationWorker, ...] = (),
    ) -> None:
        self.store = store
        self.recorder = recorder
        self.extractor = extractor
        self.planner = planner
        self.executor = executor
        self.config = config or ExplorerConfig()
        self.guide = guide
        self.additional_workers = additional_workers
        self.limits = store.run.limits
        self._promoted_root_capture: Optional[CapturedState] = None
        self._recovery_checkpoint: Optional[_BrowserCheckpoint] = None
        self._recovery_path: tuple[PlannedAction, ...] = ()

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
        context_tracker = (
            TestcaseContextTracker(self.config.testcase_context_stability_observations)
            if self.config.completion_goal == CrawlCompletionGoal.TESTCASE_CONTEXT
            else None
        )
        testcase_context: Optional[TestcaseContextCoverage] = None
        try:
            if not self.config.capture_initial_state:
                raise ExplorationError(
                    "an initial state is required for graph exploration"
                )
            initial_capture = await self.extractor.capture(
                root_page,
                sequence=ledger.sequence,
            )
            ledger.sequence += 1
            self._retain_capture(ledger, initial_capture)
            entry_checkpoint = await self._capture_checkpoint(root_page)
            active_page = root_page
            root_capture = initial_capture
            recovery_path: tuple[PlannedAction, ...] = ()
            if self.guide is not None and self.guide.steps:
                active_page, root_capture, recovery_path = await self._execute_guide(
                    active_page,
                    root_capture,
                    ledger,
                )
            checkpoint = await self._capture_checkpoint(active_page)
            root_page = active_page
            self._promoted_root_capture = root_capture
            self._recovery_checkpoint = entry_checkpoint
            self._recovery_path = recovery_path
        except Exception as exc:
            await self._finalize_failed_run(started_at, ledger, str(exc))
            raise
        testcase_context = (
            context_tracker.observe(root_capture) if context_tracker else None
        )
        exploration_root_state_id = root_capture.state.state_id
        testcase_goal_stopped = False
        attempted_actions = len(ledger.transitions)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.limits.maximum_runtime_seconds
        primary_worker = ExplorationWorker(
            worker_id="worker-1",
            page=root_page,
            recorder=self.recorder,
            extractor=self.extractor,
            executor=self.executor,
        )
        workers = await self._active_workers(
            primary_worker,
            checkpoint,
            root_capture,
            ledger,
        )
        if self.guide is not None and self.guide.repeat_rules:
            try:
                (
                    active_page,
                    root_capture,
                    testcase_context,
                    repeat_attempts,
                ) = (
                    await self._execute_repeat_rules(
                        active_page,
                        root_capture,
                        ledger,
                        context_tracker=context_tracker,
                        testcase_context=testcase_context,
                        deadline=deadline,
                        attempted_actions=attempted_actions,
                    )
                    if len(workers) == 1
                    else await self._execute_repeat_rules_parallel(
                        workers,
                        checkpoint,
                        root_capture,
                        ledger,
                        context_tracker=context_tracker,
                        testcase_context=testcase_context,
                        deadline=deadline,
                        attempted_actions=attempted_actions,
                    )
                )
                attempted_actions += repeat_attempts
                testcase_goal_stopped = bool(
                    testcase_context and testcase_context.stable_for_early_stop
                )
            except Exception as exc:
                if self.config.strategy == CrawlStrategy.GUIDED:
                    await self._finalize_failed_run(started_at, ledger, str(exc))
                    raise
                ledger.limitations.add(f"guided row sweep fell back: {exc}")
        frontier: deque[_GraphNode] = deque(
            [_GraphNode(capture=root_capture, path=(), depth=0)]
        )
        navigation_frontier: deque[_GraphNode] = deque()
        observation_frontier: deque[_GraphNode] = deque()
        completion_reason = (
            "testcase context complete"
            if testcase_goal_stopped
            else "frontier exhausted"
        )
        runtime_limited = False
        if testcase_goal_stopped:
            ledger.limitations.add(
                "configured testcase-context goal reached before bounded frontier exhaustion"
            )
            frontier.clear()
        guided_only = self.config.strategy == CrawlStrategy.GUIDED
        if guided_only and not testcase_goal_stopped:
            frontier.clear()
            completion_reason = "guided actions exhausted"

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
                scoped_ids = self._scoped_element_ids(node.capture)
                planned_actions = await self.planner.plan(
                    node.capture,
                    element_ids=scoped_ids,
                )
                ledger.actions.extend(planned.candidate for planned in planned_actions)

                action_index = 0
                while action_index < len(planned_actions):
                    planned = planned_actions[action_index]
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
                        outcome = await primary_worker.executor.skip(
                            primary_worker.page,
                            planned,
                            node.capture.state,
                            reason=planned.candidate.rationale,
                        )
                        outcomes = ((planned, outcome),)
                        action_index += 1
                    elif limit_reason is not None:
                        ledger.limitations.add(limit_reason)
                        outcome = await primary_worker.executor.skip(
                            primary_worker.page,
                            planned,
                            node.capture.state,
                            reason=limit_reason,
                        )
                        outcomes = ((planned, outcome),)
                        action_index += 1
                    else:
                        maximum_batch = (
                            1
                            if planned.candidate.risk == ActionRisk.REVIEW_REQUIRED
                            else len(workers)
                        )
                        batch: list[PlannedAction] = []
                        while (
                            action_index < len(planned_actions)
                            and len(batch) < maximum_batch
                        ):
                            candidate = planned_actions[action_index]
                            if candidate.candidate.status != ActionStatus.PENDING:
                                break
                            if (
                                candidate.candidate.risk == ActionRisk.REVIEW_REQUIRED
                                and batch
                            ):
                                break
                            next_limit = self._limit_reason(
                                node,
                                attempted_actions=attempted_actions + len(batch),
                                discovered_states=len(ledger.states) + len(batch),
                            )
                            if next_limit is not None:
                                break
                            batch.append(candidate)
                            action_index += 1
                            if candidate.candidate.risk == ActionRisk.REVIEW_REQUIRED:
                                break
                        if not batch:
                            # Re-evaluate the current action through the bounded-skip
                            # branch without losing its durable candidate.
                            continue
                        sequences = tuple(
                            range(ledger.sequence, ledger.sequence + len(batch))
                        )
                        ledger.sequence += len(batch)
                        attempted_actions += len(batch)
                        executed = await asyncio.gather(
                            *(
                                self._execute_from_parent(
                                    workers[index],
                                    checkpoint,
                                    node,
                                    candidate,
                                    sequence=sequences[index],
                                )
                                for index, candidate in enumerate(batch)
                            )
                        )
                        outcomes = tuple(zip(batch, executed))

                    for completed_action, outcome in outcomes:
                        ledger.transitions.append(outcome.transition)
                        capture = outcome.capture
                        if capture is None:
                            continue
                        if context_tracker is not None:
                            testcase_context = context_tracker.observe(capture)
                            if testcase_context.stable_for_early_stop:
                                completion_reason = "testcase context complete"
                                ledger.limitations.add(
                                    "configured testcase-context goal reached before "
                                    "bounded frontier exhaustion"
                                )
                                testcase_goal_stopped = True
                        if not self._retain_capture(ledger, capture):
                            if testcase_goal_stopped:
                                break
                            continue
                        if testcase_goal_stopped:
                            break
                        if (
                            completed_action.candidate.risk
                            == ActionRisk.REVIEW_REQUIRED
                        ):
                            ledger.limitations.add(
                                "review-required action results are terminal branches "
                                "to prevent repeated external side effects"
                            )
                            continue
                        child = _GraphNode(
                            capture=capture,
                            path=(*node.path, completed_action),
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
                    if testcase_goal_stopped:
                        break
                if runtime_limited:
                    break
                if testcase_goal_stopped:
                    break
        except Exception as exc:
            await self._finalize_failed_run(started_at, ledger, str(exc))
            raise

        coverage = await self._coverage(
            ledger,
            bounded_complete=(
                not runtime_limited and not testcase_goal_stopped and not guided_only
            )
            or (guided_only and not runtime_limited),
            completion_goal=self.config.completion_goal,
            testcase_context=testcase_context,
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
            if coverage.configured_goal_complete
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
            root_state_id=exploration_root_state_id,
            state_ids=unique_states,
            action_ids=tuple(action.action_id for action in ledger.actions),
            transition_ids=transition_ids,
            coverage=coverage,
            completion_reason=completion_reason,
        )

    async def _execute_guide(
        self,
        active_page: Page,
        capture: CapturedState,
        ledger: _ExplorationLedger,
    ) -> tuple[Page, CapturedState, tuple[PlannedAction, ...]]:
        """Replay explicit setup steps once, then promote the result to root."""

        if self.guide is None:  # pragma: no cover - caller invariant
            return active_page, capture, ()
        current_page = active_page
        current_capture = capture
        recovery_path: list[PlannedAction] = []
        for step in self.guide.steps:
            matches = target_elements(current_capture, step.target, allow_many=False)
            if len(matches) != 1:
                message = f"guide step {step.name!r} did not resolve uniquely"
                if self.config.strategy == CrawlStrategy.HYBRID:
                    ledger.limitations.add(f"{message}; generic discovery resumed")
                    break
                raise ExplorationError(message)
            planned = await self.planner.plan(
                current_capture,
                element_ids={matches[0].element_id},
                action_kinds={step.action},
            )
            selected = self._one_pending_guide_action(step, planned)
            ledger.actions.extend(item.candidate for item in planned)
            outcome = await self.executor.execute(
                current_page,
                selected,
                current_capture,
                sequence=ledger.sequence,
            )
            ledger.transitions.append(outcome.transition)
            if outcome.capture is None:
                message = f"guide step {step.name!r} failed"
                if self.config.strategy == CrawlStrategy.HYBRID:
                    ledger.limitations.add(f"{message}; generic discovery resumed")
                    break
                raise ExplorationError(message)
            ledger.sequence += 1
            self._retain_capture(ledger, outcome.capture)
            recovery_path.append(selected)
            errors = expectation_errors(outcome.capture, step.expect)
            if errors:
                message = f"guide step {step.name!r}: {'; '.join(errors)}"
                if self.config.strategy == CrawlStrategy.HYBRID:
                    ledger.limitations.add(f"{message}; generic discovery resumed")
                    current_page = outcome.active_page
                    current_capture = outcome.capture
                    break
                raise ExplorationError(message)
            current_page = outcome.active_page
            current_capture = outcome.capture
        if self.guide.promote_final_state_to_root:
            ledger.limitations.add(
                "configured guide final state was promoted to exploration root"
            )
        return current_page, current_capture, tuple(recovery_path)

    async def _execute_repeat_rules(
        self,
        active_page: Page,
        capture: CapturedState,
        ledger: _ExplorationLedger,
        *,
        context_tracker: Optional[TestcaseContextTracker],
        testcase_context: Optional[TestcaseContextCoverage],
        deadline: float,
        attempted_actions: int,
    ) -> tuple[
        Page,
        CapturedState,
        Optional[TestcaseContextCoverage],
        int,
    ]:
        """Sweep demonstrated row controls in place without restoring the root."""

        current_page = active_page
        current_capture = capture
        attempts = 0
        for rule in self.guide.repeat_rules if self.guide else ():
            processed_rows: set[str] = set()
            while len(processed_rows) < rule.maximum_rows:
                if asyncio.get_running_loop().time() >= deadline:
                    raise ExplorationError(
                        "maximum runtime reached during guided row sweep"
                    )
                if attempted_actions + attempts >= self.limits.maximum_actions:
                    raise ExplorationError(
                        "maximum executed actions reached during guided row sweep"
                    )
                matches = tuple(
                    element
                    for element in target_elements(
                        current_capture,
                        rule.target,
                        allow_many=True,
                    )
                    if element.visible
                    and element.enabled
                    and (not rule.require_row_label or bool(element.context.row_label))
                    and (element.context.row_label or element.element_id)
                    not in processed_rows
                )
                if not matches:
                    if not processed_rows:
                        raise ExplorationError(
                            f"repeat rule {rule.name!r} matched no row controls"
                        )
                    break
                selected_element = min(
                    matches,
                    key=lambda element: (
                        element.context.row_label or "",
                        element.element_id,
                    ),
                )
                row_key = (
                    selected_element.context.row_label or selected_element.element_id
                )
                planned = await self.planner.plan(
                    current_capture,
                    element_ids={selected_element.element_id},
                    action_kinds={ActionKind.CLICK},
                )
                selected = self._one_pending_repeat_action(rule, planned)
                ledger.actions.extend(item.candidate for item in planned)
                outcome = await self.executor.execute(
                    current_page,
                    selected,
                    current_capture,
                    sequence=ledger.sequence,
                )
                ledger.transitions.append(outcome.transition)
                attempts += 1
                if outcome.capture is None:
                    raise ExplorationError(
                        f"repeat rule {rule.name!r} failed for row {row_key!r}"
                    )
                ledger.sequence += 1
                self._retain_capture(ledger, outcome.capture)
                modal_capture = outcome.capture
                if (
                    rule.expect_dialog_contains_row_label
                    and selected_element.context.row_label
                    and not _capture_contains(
                        modal_capture,
                        selected_element.context.row_label,
                    )
                ):
                    raise ExplorationError(
                        f"repeat rule {rule.name!r} dialog did not contain row label"
                    )
                if context_tracker is not None:
                    testcase_context = context_tracker.observe(modal_capture)
                current_page = outcome.active_page
                current_capture = modal_capture
                processed_rows.add(row_key)

                (
                    current_page,
                    current_capture,
                    close_attempts,
                ) = await self._close_repeated_observation(
                    current_page,
                    current_capture,
                    rule,
                    ledger,
                )
                attempts += close_attempts
                if context_tracker is not None:
                    testcase_context = context_tracker.observe(current_capture)
                if testcase_context and testcase_context.stable_for_early_stop:
                    return (
                        current_page,
                        current_capture,
                        testcase_context,
                        attempts,
                    )
        return current_page, current_capture, testcase_context, attempts

    async def _close_repeated_observation(
        self,
        page: Page,
        capture: CapturedState,
        rule: GuideRepeatRule,
        ledger: _ExplorationLedger,
    ) -> tuple[Page, CapturedState, int]:
        matches = target_elements(capture, rule.close_target, allow_many=False)
        if len(matches) != 1:
            raise ExplorationError(
                f"repeat rule {rule.name!r} close target did not resolve uniquely"
            )
        planned = await self.planner.plan(
            capture,
            element_ids={matches[0].element_id},
            action_kinds={ActionKind.CLICK},
        )
        selected = self._one_pending_repeat_action(rule, planned)
        ledger.actions.extend(item.candidate for item in planned)
        outcome = await self.executor.execute(
            page,
            selected,
            capture,
            sequence=ledger.sequence,
        )
        ledger.transitions.append(outcome.transition)
        if outcome.capture is None:
            raise ExplorationError(f"repeat rule {rule.name!r} could not close dialog")
        ledger.sequence += 1
        self._retain_capture(ledger, outcome.capture)
        return outcome.active_page, outcome.capture, 1

    async def _execute_repeat_rules_parallel(
        self,
        workers: tuple[ExplorationWorker, ...],
        checkpoint: _BrowserCheckpoint,
        capture: CapturedState,
        ledger: _ExplorationLedger,
        *,
        context_tracker: Optional[TestcaseContextTracker],
        testcase_context: Optional[TestcaseContextCoverage],
        deadline: float,
        attempted_actions: int,
    ) -> tuple[
        Page,
        CapturedState,
        Optional[TestcaseContextCoverage],
        int,
    ]:
        """Partition repeated root-row observations across validated workers."""

        root_capture = capture
        primary_capture = capture
        attempts = 0
        for rule in self.guide.repeat_rules if self.guide else ():
            matches = tuple(
                sorted(
                    (
                        element
                        for element in target_elements(
                            root_capture,
                            rule.target,
                            allow_many=True,
                        )
                        if element.visible
                        and element.enabled
                        and (
                            not rule.require_row_label
                            or bool(element.context.row_label)
                        )
                    ),
                    key=lambda element: (
                        element.context.row_label or "",
                        element.element_id,
                    ),
                )[: rule.maximum_rows]
            )
            if not matches:
                raise ExplorationError(
                    f"repeat rule {rule.name!r} matched no row controls"
                )

            for offset in range(0, len(matches), len(workers)):
                if asyncio.get_running_loop().time() >= deadline:
                    raise ExplorationError(
                        "maximum runtime reached during parallel row sweep"
                    )
                batch = matches[offset : offset + len(workers)]
                remaining = self.limits.maximum_actions - (attempted_actions + attempts)
                if remaining < 2:
                    raise ExplorationError(
                        "maximum executed actions reached during parallel row sweep"
                    )
                batch = batch[: remaining // 2]

                info_actions: list[PlannedAction] = []
                for element in batch:
                    planned = await self.planner.plan(
                        root_capture,
                        element_ids={element.element_id},
                        action_kinds={ActionKind.CLICK},
                    )
                    ledger.actions.extend(item.candidate for item in planned)
                    info_actions.append(self._one_pending_repeat_action(rule, planned))

                info_sequences = tuple(
                    range(ledger.sequence, ledger.sequence + len(info_actions))
                )
                ledger.sequence += len(info_actions)
                node = _GraphNode(capture=root_capture, path=(), depth=0)
                info_outcomes = await asyncio.gather(
                    *(
                        self._execute_from_parent(
                            workers[index],
                            checkpoint,
                            node,
                            action,
                            sequence=info_sequences[index],
                        )
                        for index, action in enumerate(info_actions)
                    )
                )
                attempts += len(info_actions)

                modal_captures: list[CapturedState] = []
                for element, outcome in zip(batch, info_outcomes):
                    ledger.transitions.append(outcome.transition)
                    if outcome.capture is None:
                        raise ExplorationError(
                            f"repeat rule {rule.name!r} failed for row "
                            f"{(element.context.row_label or element.element_id)!r}"
                        )
                    modal_capture = outcome.capture
                    self._retain_capture(ledger, modal_capture)
                    if (
                        rule.expect_dialog_contains_row_label
                        and element.context.row_label
                        and not _capture_contains(
                            modal_capture,
                            element.context.row_label,
                        )
                    ):
                        raise ExplorationError(
                            f"repeat rule {rule.name!r} dialog did not contain row label"
                        )
                    modal_captures.append(modal_capture)
                    if context_tracker is not None:
                        testcase_context = context_tracker.observe(modal_capture)

                close_actions: list[PlannedAction] = []
                for modal_capture in modal_captures:
                    close_matches = target_elements(
                        modal_capture,
                        rule.close_target,
                        allow_many=False,
                    )
                    if len(close_matches) != 1:
                        raise ExplorationError(
                            f"repeat rule {rule.name!r} close target did not resolve uniquely"
                        )
                    planned = await self.planner.plan(
                        modal_capture,
                        element_ids={close_matches[0].element_id},
                        action_kinds={ActionKind.CLICK},
                    )
                    ledger.actions.extend(item.candidate for item in planned)
                    close_actions.append(self._one_pending_repeat_action(rule, planned))

                close_sequences = tuple(
                    range(ledger.sequence, ledger.sequence + len(close_actions))
                )
                ledger.sequence += len(close_actions)
                close_outcomes = await asyncio.gather(
                    *(
                        workers[index].executor.execute(
                            info_outcomes[index].active_page,
                            action,
                            modal_captures[index],
                            sequence=close_sequences[index],
                        )
                        for index, action in enumerate(close_actions)
                    )
                )
                attempts += len(close_actions)

                for outcome in close_outcomes:
                    ledger.transitions.append(outcome.transition)
                    if outcome.capture is None:
                        raise ExplorationError(
                            f"repeat rule {rule.name!r} could not close dialog"
                        )
                    self._retain_capture(ledger, outcome.capture)
                    if context_tracker is not None:
                        testcase_context = context_tracker.observe(outcome.capture)
                primary_capture = close_outcomes[0].capture or primary_capture
                if testcase_context and testcase_context.stable_for_early_stop:
                    ledger.limitations.add(
                        f"guided row sweep distributed across {len(workers)} workers"
                    )
                    return (
                        workers[0].page,
                        primary_capture,
                        testcase_context,
                        attempts,
                    )

        ledger.limitations.add(
            f"guided row sweep distributed across {len(workers)} workers"
        )
        return workers[0].page, primary_capture, testcase_context, attempts

    @staticmethod
    def _one_pending_guide_action(
        step: GuideStep,
        planned: tuple[PlannedAction, ...],
    ) -> PlannedAction:
        pending = tuple(
            item
            for item in planned
            if item.candidate.kind == step.action
            and item.candidate.status == ActionStatus.PENDING
        )
        if len(pending) != 1:
            raise ExplorationError(
                f"guide step {step.name!r} was blocked or ambiguous by action policy"
            )
        return pending[0]

    @staticmethod
    def _one_pending_repeat_action(
        rule: GuideRepeatRule,
        planned: tuple[PlannedAction, ...],
    ) -> PlannedAction:
        pending = tuple(
            item for item in planned if item.candidate.status == ActionStatus.PENDING
        )
        if len(pending) != 1:
            raise ExplorationError(
                f"repeat rule {rule.name!r} was blocked or ambiguous by action policy"
            )
        return pending[0]

    def _scoped_element_ids(
        self,
        capture: CapturedState,
    ) -> Optional[set[str]]:
        if self.guide is None or not self.guide.root_scope_selector:
            return None
        selector = self.guide.root_scope_selector.strip()
        token = selector.casefold()
        selected: set[str] = set()
        for element in capture.elements:
            ancestors = tuple(
                value.casefold() for value in element.context.ancestor_summary
            )
            own_css = {
                locator.value.casefold()
                for locator in element.locators
                if locator.strategy.value == "css"
            }
            in_scope = False
            if token == "main":
                in_scope = (
                    element.context.inside_main
                    or element.tag == "main"
                    or any(
                        ancestor == "main" or ancestor.startswith("main[")
                        for ancestor in ancestors
                    )
                )
            elif token.startswith("#"):
                in_scope = token in own_css or any(
                    token in value for value in ancestors
                )
            else:
                in_scope = token in own_css or any(
                    token == value for value in ancestors
                )
            if in_scope:
                selected.add(element.element_id)
        if selected:
            return selected
        return None

    @staticmethod
    def _retain_capture(
        ledger: _ExplorationLedger,
        capture: CapturedState,
    ) -> bool:
        if capture.state.fingerprint in ledger.states:
            ledger.duplicate_states += 1
            return False
        ledger.states[capture.state.fingerprint] = capture
        return True

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
        worker: ExplorationWorker,
        checkpoint: _BrowserCheckpoint,
        node: _GraphNode,
        planned: PlannedAction,
        *,
        sequence: int,
    ) -> ActionExecutionOutcome:
        try:
            active_page = await self._restore_parent(
                worker,
                checkpoint,
                node.path,
            )
        except (
            ActionExecutionError,
            ExplorationError,
            PlaywrightError,
            RecorderError,
        ) as exc:
            return await worker.executor.fail(
                worker.page,
                planned,
                node.capture.state,
                error=f"parent-state restoration failed: {exc}",
            )
        return await worker.executor.execute(
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

    async def _active_workers(
        self,
        primary: ExplorationWorker,
        checkpoint: _BrowserCheckpoint,
        root_capture: CapturedState,
        ledger: _ExplorationLedger,
    ) -> tuple[ExplorationWorker, ...]:
        """Probe cloned sessions once and safely reduce to valid workers."""

        requested = self.config.worker_count
        if requested == 1:
            return (primary,)
        candidates = self.additional_workers[: requested - 1]
        if len(candidates) < requested - 1:
            message = (
                f"requested {requested} workers but only "
                f"{len(candidates) + 1} browser stacks started"
            )
            if self.config.parallel_session_mode == ParallelSessionMode.FORCE:
                raise ExplorationError(message)
            ledger.limitations.add(f"{message}; continuing with available workers")

        async def probe(worker: ExplorationWorker) -> bool:
            try:
                await self._restore_parent(worker, checkpoint, ())
                return await self._root_matches(worker.page, root_capture)
            except (
                ActionExecutionError,
                ExplorationError,
                PlaywrightError,
                RecorderError,
            ):
                return False

        results = await asyncio.gather(*(probe(worker) for worker in candidates))
        valid = tuple(
            worker for worker, accepted in zip(candidates, results) if accepted
        )
        rejected = len(candidates) - len(valid)
        if rejected:
            message = f"{rejected} parallel worker session probe(s) failed"
            if self.config.parallel_session_mode == ParallelSessionMode.FORCE:
                raise ExplorationError(message)
            ledger.limitations.add(f"{message}; falling back to valid workers")
        active = (primary, *valid)
        if len(active) > 1:
            ledger.limitations.add(
                f"safe sibling actions used {len(active)} isolated browser workers"
            )
        else:
            ledger.limitations.add(
                "parallel session probe left one worker; exploration remained serial"
            )
        return active

    async def _root_matches(
        self,
        page: Page,
        root_capture: CapturedState,
    ) -> bool:
        """Validate a cloned worker without persisting raw authentication state."""

        expected_url = urlsplit(root_capture.state.url)
        actual_url = urlsplit(page.url)
        if (
            expected_url.scheme.lower(),
            expected_url.netloc.lower(),
            expected_url.path,
        ) != (
            actual_url.scheme.lower(),
            actual_url.netloc.lower(),
            actual_url.path,
        ):
            return False
        try:
            title = await page.title()
            actual_rows = await page.locator("tr").evaluate_all(
                """rows => rows.map((row) => {
                    const cell = row.querySelector('th[scope=row], th, td');
                    return String(cell ? (cell.innerText || cell.textContent || '') : '')
                        .replace(/\\s+/g, ' ').trim();
                }).filter(Boolean)"""
            )
            actual_table_count = await page.locator("table, [role=table]").count()
            actual_modal_count = await page.locator(
                "dialog, [role=dialog], [aria-modal=true]"
            ).count()
        except PlaywrightError:
            return False
        expected_rows = {
            element.context.row_label
            for element in root_capture.elements
            if element.context.row_label
        }
        row_values = {
            str(value) for value in actual_rows if isinstance(value, str) and value
        }
        expected_has_table = any(
            element.tag == "table" or (element.role or "").lower() == "table"
            for element in root_capture.elements
        )
        return bool(
            title == root_capture.state.title
            and expected_rows.issubset(row_values)
            and (actual_table_count > 0) == expected_has_table
            and (actual_modal_count > 0) == (root_capture.state.modal_count > 0)
        )

    async def _restore_parent(
        self,
        worker: ExplorationWorker,
        checkpoint: _BrowserCheckpoint,
        path: tuple[PlannedAction, ...],
    ) -> Page:
        active_root = await self._restore_checkpoint(worker, checkpoint)
        promoted = self._promoted_root_capture
        if promoted is not None and not await self._root_matches(active_root, promoted):
            recovery = self._recovery_checkpoint
            if recovery is None or not self._recovery_path:
                raise ExplorationError(
                    "promoted root could not be restored from its URL and storage"
                )
            active_root = await self._restore_checkpoint(worker, recovery)
            for step in self._recovery_path:
                active_root = await worker.executor.replay(active_root, step)
            if not await self._root_matches(active_root, promoted):
                raise ExplorationError(
                    "promoted root recovery guide failed structural validation"
                )

        active_page = active_root
        for step in path:
            active_page = await worker.executor.replay(active_page, step)
        return active_page

    async def _restore_checkpoint(
        self,
        worker: ExplorationWorker,
        checkpoint: _BrowserCheckpoint,
    ) -> Page:
        """Restore one in-memory browser checkpoint without replaying graph edges."""

        root_page = worker.page
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
        await worker.extractor.wait_for_quiet(root_page)
        await worker.recorder.drain_mutations(root_page)
        await worker.recorder.flush()
        return root_page

    async def _coverage(
        self,
        ledger: _ExplorationLedger,
        *,
        bounded_complete: bool,
        completion_goal: CrawlCompletionGoal,
        testcase_context: Optional[TestcaseContextCoverage],
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
            "promoted roots are structurally validated; replayed deep parents are "
            "locator-validated but not fingerprint-validated before the next action"
        )
        if self._recovery_path:
            limitations.add(
                "promoted-root URL drift falls back to one replay of the validated guide"
            )
        if any(
            frame.is_cross_origin
            for capture in captures
            for frame in capture.state.frames
        ):
            limitations.add(
                "cross-origin frames are restored only when present in the root frame tree"
            )
        effective_bounded_complete = bounded_complete and not pending_ids
        goal_complete = (
            effective_bounded_complete
            if completion_goal == CrawlCompletionGoal.BOUNDED_FRONTIER
            else bool(testcase_context and testcase_context.context_complete)
        )
        terminal_count = sum(status_counts.values()) + len(pending_ids)
        if terminal_count != len(ledger.actions):
            raise ExplorationError(
                "action ledger mismatch: "
                f"candidates={len(ledger.actions)}, "
                f"transitions={len(ledger.transitions)}, pending={len(pending_ids)}, "
                f"terminal={terminal_count}"
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
            bounded_complete=effective_bounded_complete,
            completion_goal=completion_goal,
            goal_complete=goal_complete,
            testcase_context=testcase_context,
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


def _capture_contains(capture: CapturedState, value: str) -> bool:
    """Return whether visible structural evidence contains an exact row token."""

    needle = value.casefold()
    return any(
        needle in candidate.casefold()
        for element in capture.elements
        for candidate in (
            element.text or "",
            element.accessible_name or "",
            element.label or "",
            element.title or "",
        )
        if candidate
    )


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
