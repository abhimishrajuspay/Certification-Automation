"""Deterministic action planning, replay, execution, and effect derivation."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin, urlsplit

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Frame, Locator, Page

from scraper.artifact_store import ArtifactStore
from scraper.extractor import (
    CapturedState,
    PageStateExtractor,
    SnapshotExtractionError,
)
from scraper.models import (
    ActionCandidate,
    ActionKind,
    ActionRisk,
    ActionStatus,
    BrowserEvent,
    BrowserEventKind,
    EffectKind,
    ElementSnapshot,
    InteractionTransition,
    LocatorCandidate,
    LocatorStrategy,
    NetworkExchange,
    StateSnapshot,
    TransitionEffect,
    ValueCapture,
    utc_now,
)
from scraper.recorder import BrowserRecorder, RecorderError
from scraper.redaction import hash_text, redact_text


class ActionPlanningError(RuntimeError):
    """Raised when action evidence cannot be planned or persisted."""


class ActionExecutionError(RuntimeError):
    """Raised when an approved action cannot be located or performed."""


@dataclass(frozen=True)
class ActionPolicyConfig:
    """Portal-independent safety vocabulary for browser-side exploration."""

    blocked_keywords: tuple[str, ...] = (
        "delete",
        "destroy",
        "erase",
        "logout",
        "purge",
        "remove account",
        "revoke",
        "sign out",
        "terminate",
        "wipe",
    )
    review_keywords: tuple[str, ...] = (
        "approve",
        "complete",
        "confirm",
        "create",
        "download",
        "execute",
        "authenticate",
        "log in",
        "login",
        "pay",
        "play",
        "production",
        "publish",
        "refund",
        "reject",
        "run",
        "save",
        "send",
        "sign in",
        "start",
        "submit",
        "transfer",
        "update",
        "upload",
    )
    allow_hidden_actions: bool = False
    allow_disabled_actions: bool = False
    include_hover_actions: bool = True
    include_scroll_actions: bool = True
    scroll_viewport_fraction: float = 0.8

    def __post_init__(self) -> None:
        if not 0 < self.scroll_viewport_fraction <= 1:
            raise ValueError("scroll_viewport_fraction must be in (0, 1]")


@dataclass(frozen=True)
class PlannedAction:
    """An action candidate paired with the evidence required for replay."""

    candidate: ActionCandidate
    element: ElementSnapshot
    frame_path: str


@dataclass(frozen=True)
class ActionExecutionOutcome:
    """One durable transition and its optional resulting browser state."""

    transition: InteractionTransition
    capture: Optional[CapturedState]
    active_page: Page


class ActionPlanner:
    """Derive deterministic candidates without portal-specific selectors."""

    def __init__(
        self,
        store: ArtifactStore,
        config: Optional[ActionPolicyConfig] = None,
    ) -> None:
        self.store = store
        self.config = config or ActionPolicyConfig()
        self.policy = store.run.capture_policy

    async def plan(self, capture: CapturedState) -> tuple[PlannedAction, ...]:
        """Derive and append all candidates for a newly discovered state."""

        frame_paths = {
            frame.frame_id: frame.frame_path for frame in capture.state.frames
        }
        planned: list[PlannedAction] = []
        for element in capture.elements:
            frame_path = frame_paths.get(element.frame_id)
            if frame_path is None:
                raise ActionPlanningError(
                    f"element {element.element_id} references an unknown frame"
                )
            for kind, parameters, option_text in self._action_shapes(element):
                risk, rule, rationale = self._classify(
                    capture.state,
                    element,
                    kind,
                    option_text=option_text,
                )
                status = ActionStatus.PENDING
                if not element.visible and not self.config.allow_hidden_actions:
                    status = ActionStatus.SKIPPED
                    rule = "visibility.hidden"
                    rationale = "element is not rendered as visible"
                elif not element.enabled and not self.config.allow_disabled_actions:
                    status = ActionStatus.SKIPPED
                    rule = "availability.disabled"
                    rationale = "element is disabled or inert"
                elif risk == ActionRisk.BLOCKED:
                    status = ActionStatus.SKIPPED
                elif (
                    risk == ActionRisk.REVIEW_REQUIRED
                    and not self.policy.allow_review_required_actions
                ):
                    status = ActionStatus.SKIPPED
                    rationale = f"{rationale}; manual-review actions are disabled"

                action_id = self._action_id(
                    capture.state.state_id,
                    element.element_id,
                    kind,
                    parameters,
                )
                candidate = ActionCandidate(
                    action_id=action_id,
                    run_id=self.store.run.run_id,
                    state_id=capture.state.state_id,
                    element_id=element.element_id,
                    kind=kind,
                    risk=risk,
                    status=status,
                    policy_rule=rule,
                    rationale=rationale,
                    parameters=parameters,
                )
                await asyncio.to_thread(self.store.append_record, candidate)
                planned.append(
                    PlannedAction(
                        candidate=candidate,
                        element=element,
                        frame_path=frame_path,
                    )
                )

        if self.config.include_scroll_actions:
            scroll_action = self._scroll_action(capture, frame_paths)
            if scroll_action is not None:
                await asyncio.to_thread(
                    self.store.append_record,
                    scroll_action.candidate,
                )
                planned.append(scroll_action)

        planned.sort(key=self._sort_key)
        return tuple(planned)

    def _action_shapes(
        self,
        element: ElementSnapshot,
    ) -> tuple[tuple[ActionKind, tuple[ValueCapture, ...], str], ...]:
        role = (element.role or "").lower()
        input_type = (element.input_type or "").lower()
        signals = set(element.interaction_signals)
        result: list[tuple[ActionKind, tuple[ValueCapture, ...], str]] = []

        if input_type == "file":
            return ((ActionKind.CLICK, (), "file chooser"),)
        if role == "checkbox" or input_type == "checkbox":
            result.append(
                (
                    ActionKind.UNCHECK if element.checked else ActionKind.CHECK,
                    (),
                    "",
                )
            )
        elif role == "radio" or input_type == "radio":
            if not element.checked:
                result.append((ActionKind.CHECK, (), ""))
        elif element.tag == "select" or role in {"combobox", "listbox"}:
            for option in element.options:
                if option.selected or option.disabled:
                    continue
                if option.redacted or option.value is None:
                    parameters = (
                        ValueCapture(
                            name="option_value",
                            value_hash=option.value_hash,
                            redacted=True,
                        ),
                    )
                else:
                    parameters = (
                        ValueCapture(name="option_value", value=option.value),
                    )
                result.append((ActionKind.SELECT_OPTION, parameters, option.label))
        else:
            clickable_role = role in {
                "button",
                "link",
                "menuitem",
                "option",
                "switch",
                "tab",
                "treeitem",
            }
            clickable_tag = element.tag in {"a", "area", "button", "summary"}
            click_signal = any(
                signal in {"cursor:pointer", "listener:click", "inline:click"}
                for signal in signals
            )
            if clickable_role or clickable_tag or click_signal:
                result.append((ActionKind.CLICK, (), ""))

        if "listener:dblclick" in signals or "inline:dblclick" in signals:
            result.append((ActionKind.DOUBLE_CLICK, (), ""))
        if self.config.include_hover_actions and any(
            signal
            in {
                "listener:mouseenter",
                "listener:mouseover",
                "listener:pointerenter",
                "inline:mouseenter",
                "inline:mouseover",
            }
            for signal in signals
        ):
            result.append((ActionKind.HOVER, (), ""))
        return tuple(result)

    def _classify(
        self,
        state: StateSnapshot,
        element: ElementSnapshot,
        kind: ActionKind,
        *,
        option_text: str,
    ) -> tuple[ActionRisk, str, str]:
        if (element.input_type or "").lower() == "file":
            return (
                ActionRisk.BLOCKED,
                "input.file_requires_authorized_path",
                "file chooser cannot be explored without an authorized local file",
            )
        href = _attribute_value(element, "href")
        if href and self.policy.same_origin_only:
            destination = urljoin(state.url, href)
            if _origin(destination) not in set(self.store.run.allowed_origins):
                return (
                    ActionRisk.BLOCKED,
                    "navigation.cross_origin",
                    "destination is outside the configured allowed origins",
                )

        description = " ".join(
            value
            for value in (
                element.accessible_name,
                element.label,
                element.text,
                element.title,
                element.context.section_heading,
                option_text,
            )
            if value
        ).lower()
        blocked = _matching_keyword(description, self.config.blocked_keywords)
        if blocked:
            return (
                ActionRisk.BLOCKED,
                f"semantic.blocked.{_rule_token(blocked)}",
                f"semantic evidence contains blocked operation '{blocked}'",
            )
        if kind == ActionKind.CLICK and (element.input_type or "").lower() in {
            "image",
            "submit",
        }:
            return (
                ActionRisk.REVIEW_REQUIRED,
                "form.submit_requires_review",
                "form submission can cause authentication or server-side effects",
            )
        review = _matching_keyword(description, self.config.review_keywords)
        if review:
            return (
                ActionRisk.REVIEW_REQUIRED,
                f"semantic.review.{_rule_token(review)}",
                f"semantic evidence suggests side effect '{review}'",
            )
        if kind == ActionKind.SELECT_OPTION:
            return (
                ActionRisk.SAFE,
                "form.select_reversible",
                "selecting a non-sensitive enabled option is browser-reversible",
            )
        if kind in {ActionKind.CHECK, ActionKind.UNCHECK}:
            return (
                ActionRisk.SAFE,
                "form.toggle_reversible",
                "toggling this local control is browser-reversible",
            )
        if kind == ActionKind.HOVER:
            return (
                ActionRisk.SAFE,
                "interaction.hover_observation",
                "hover observes a registered hover interaction without activation",
            )
        return (
            ActionRisk.SAFE,
            "interaction.discovery_safe",
            "no destructive or side-effect semantic signal was detected",
        )

    def _scroll_action(
        self,
        capture: CapturedState,
        frame_paths: dict[str, str],
    ) -> Optional[PlannedAction]:
        scroll = capture.state.scroll
        remaining = scroll.maximum_y - scroll.y
        if remaining <= 1:
            return None
        root = next(
            (
                element
                for element in capture.elements
                if frame_paths.get(element.frame_id) == "main"
                and element.tag in {"html", "body"}
            ),
            None,
        )
        if root is None:
            return None
        delta = max(
            1,
            min(
                int(remaining),
                int(
                    capture.state.viewport.height * self.config.scroll_viewport_fraction
                ),
            ),
        )
        parameters = (ValueCapture(name="delta_y", value=str(delta)),)
        candidate = ActionCandidate(
            action_id=self._action_id(
                capture.state.state_id,
                root.element_id,
                ActionKind.SCROLL,
                parameters,
            ),
            run_id=self.store.run.run_id,
            state_id=capture.state.state_id,
            element_id=root.element_id,
            kind=ActionKind.SCROLL,
            risk=ActionRisk.SAFE,
            policy_rule="viewport.scroll_bounded",
            rationale="more document content exists below the current viewport",
            parameters=parameters,
        )
        return PlannedAction(
            candidate=candidate,
            element=root,
            frame_path="main",
        )

    @staticmethod
    def _action_id(
        state_id: str,
        element_id: str,
        kind: ActionKind,
        parameters: tuple[ValueCapture, ...],
    ) -> str:
        payload = json.dumps(
            {
                "state_id": state_id,
                "element_id": element_id,
                "kind": kind.value,
                "parameters": [item.model_dump(mode="json") for item in parameters],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return f"action-{hash_text(payload)[:32]}"

    @staticmethod
    def _sort_key(planned: PlannedAction) -> tuple[int, str, str]:
        status_order = 0 if planned.candidate.status == ActionStatus.PENDING else 1
        return (
            status_order,
            planned.frame_path,
            planned.candidate.action_id,
        )


class ActionExecutor:
    """Replay locators, execute one action, and persist its causal transition."""

    def __init__(
        self,
        store: ArtifactStore,
        recorder: BrowserRecorder,
        extractor: PageStateExtractor,
        *,
        action_timeout_ms: int = 10_000,
    ) -> None:
        if action_timeout_ms <= 0:
            raise ValueError("action_timeout_ms must be positive")
        self.store = store
        self.recorder = recorder
        self.extractor = extractor
        self.action_timeout_ms = action_timeout_ms

    async def execute(
        self,
        page: Page,
        planned: PlannedAction,
        parent: CapturedState,
        *,
        sequence: int,
    ) -> ActionExecutionOutcome:
        """Execute an approved candidate and capture its resulting state."""

        candidate = planned.candidate
        if candidate.status != ActionStatus.PENDING:
            return await self.skip(
                page,
                planned,
                parent.state,
                reason=candidate.rationale,
            )

        started_at = utc_now()
        locator_used: Optional[LocatorCandidate] = None
        active_page = page
        try:
            async with self.recorder.action_scope(candidate.action_id):
                pages_before = set(page.context.pages)
                locator, locator_used = await self._resolve_locator(page, planned)
                await self._perform(page, locator, candidate)
                active_page = await self._result_page(page, pages_before)
                capture = await self.extractor.capture(
                    active_page,
                    sequence=sequence,
                )
            await self.recorder.flush()
            completed_at = utc_now()
            event_ids, exchange_ids, events, exchanges = self._correlated_evidence(
                candidate.action_id
            )
            transition = InteractionTransition(
                transition_id=_transition_id(
                    candidate.action_id,
                    ActionStatus.SUCCEEDED,
                    capture.state.state_id,
                ),
                run_id=self.store.run.run_id,
                action_id=candidate.action_id,
                parent_state_id=parent.state.state_id,
                immediate_state_id=capture.state.state_id,
                resulting_state_id=capture.state.state_id,
                status=ActionStatus.SUCCEEDED,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=_duration_ms(started_at, completed_at),
                locator_used=locator_used,
                event_ids=event_ids,
                network_exchange_ids=exchange_ids,
                effects=self._effects(
                    parent.state,
                    capture.state,
                    candidate,
                    events,
                    exchanges,
                ),
                quiescence_reason=(
                    "configured DOM quiet window reached"
                    if capture.state.stable
                    else "quiet timeout reached; bounded snapshot retained"
                ),
            )
            await asyncio.to_thread(self.store.append_record, transition)
            return ActionExecutionOutcome(
                transition=transition,
                capture=capture,
                active_page=active_page,
            )
        except (
            ActionExecutionError,
            PlaywrightError,
            RecorderError,
            SnapshotExtractionError,
        ) as exc:
            try:
                await self.recorder.flush()
            except RecorderError:
                pass
            completed_at = utc_now()
            event_ids, exchange_ids, _, _ = self._correlated_evidence(
                candidate.action_id
            )
            safe_error = redact_text(
                str(exc),
                self.store.run.capture_policy.redacted_names,
            )
            transition = InteractionTransition(
                transition_id=_transition_id(
                    candidate.action_id,
                    ActionStatus.FAILED,
                    safe_error,
                ),
                run_id=self.store.run.run_id,
                action_id=candidate.action_id,
                parent_state_id=parent.state.state_id,
                status=ActionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=_duration_ms(started_at, completed_at),
                locator_used=locator_used,
                event_ids=event_ids,
                network_exchange_ids=exchange_ids,
                error_message=safe_error or "action execution failed",
            )
            await asyncio.to_thread(self.store.append_record, transition)
            return ActionExecutionOutcome(
                transition=transition,
                capture=None,
                active_page=active_page,
            )

    async def skip(
        self,
        page: Page,
        planned: PlannedAction,
        parent: StateSnapshot,
        *,
        reason: str,
    ) -> ActionExecutionOutcome:
        """Persist why a candidate was not executed."""

        now = utc_now()
        transition = InteractionTransition(
            transition_id=_transition_id(
                planned.candidate.action_id,
                ActionStatus.SKIPPED,
                reason,
            ),
            run_id=self.store.run.run_id,
            action_id=planned.candidate.action_id,
            parent_state_id=parent.state_id,
            status=ActionStatus.SKIPPED,
            started_at=now,
            completed_at=now,
            duration_ms=0,
            skip_reason=reason,
        )
        await asyncio.to_thread(self.store.append_record, transition)
        return ActionExecutionOutcome(
            transition=transition,
            capture=None,
            active_page=page,
        )

    async def fail(
        self,
        page: Page,
        planned: PlannedAction,
        parent: StateSnapshot,
        *,
        error: str,
    ) -> ActionExecutionOutcome:
        """Persist a failure that occurred while restoring the parent path."""

        now = utc_now()
        safe_error = (
            redact_text(
                error,
                self.store.run.capture_policy.redacted_names,
            )
            or "parent-state restoration failed"
        )
        transition = InteractionTransition(
            transition_id=_transition_id(
                planned.candidate.action_id,
                ActionStatus.FAILED,
                safe_error,
            ),
            run_id=self.store.run.run_id,
            action_id=planned.candidate.action_id,
            parent_state_id=parent.state_id,
            status=ActionStatus.FAILED,
            started_at=now,
            completed_at=now,
            duration_ms=0,
            error_message=safe_error,
        )
        await asyncio.to_thread(self.store.append_record, transition)
        return ActionExecutionOutcome(
            transition=transition,
            capture=None,
            active_page=page,
        )

    async def replay(self, page: Page, planned: PlannedAction) -> Page:
        """Replay an already-approved path step without creating graph records."""

        pages_before = set(page.context.pages)
        locator, _ = await self._resolve_locator(page, planned)
        await self._perform(page, locator, planned.candidate)
        active_page = await self._result_page(page, pages_before)
        await self.extractor.wait_for_quiet(active_page)
        await self.recorder.drain_mutations(active_page)
        await self.recorder.flush()
        return active_page

    async def _resolve_locator(
        self,
        page: Page,
        planned: PlannedAction,
    ) -> tuple[Optional[Locator], LocatorCandidate]:
        candidates = sorted(
            planned.element.locators,
            key=lambda item: (
                not item.is_primary,
                -item.confidence,
                item.strategy.value,
            ),
        )
        if planned.candidate.kind == ActionKind.SCROLL:
            locator = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.strategy == LocatorStrategy.CSS
                ),
                None,
            )
            if locator is None:
                raise ActionExecutionError("scroll action has no CSS provenance")
            return None, locator

        errors: list[str] = []
        for attempt in range(20):
            errors = []
            for candidate in candidates:
                try:
                    frame = await self._frame_by_path(page, planned.frame_path)
                    if frame.is_detached():
                        raise ActionExecutionError("frame is detached")
                    locator = self._locator(frame, planned.element, candidate)
                    count = await locator.count()
                    if count == 1:
                        return locator, candidate
                    errors.append(f"{candidate.strategy.value} matched {count}")
                except (
                    ActionExecutionError,
                    PlaywrightError,
                    ValueError,
                    json.JSONDecodeError,
                ) as exc:
                    errors.append(f"{candidate.strategy.value}: {exc}")
            if attempt < 19:
                await asyncio.sleep(0.05)
        raise ActionExecutionError(
            "no replay locator uniquely matched the element: " + "; ".join(errors)
        )

    def _locator(
        self,
        frame: Frame,
        element: ElementSnapshot,
        candidate: LocatorCandidate,
    ) -> Locator:
        strategy = candidate.strategy
        if strategy == LocatorStrategy.TEST_ID:
            value = _json_object(candidate.value)
            return frame.locator(
                f"[{_css_identifier(value['attribute'])}={json.dumps(value['value'])}]"
            )
        if strategy == LocatorStrategy.ROLE:
            value = _json_object(candidate.value)
            return frame.get_by_role(
                value["role"],
                name=value["name"],
                exact=True,
            )
        if strategy == LocatorStrategy.LABEL:
            return frame.get_by_label(candidate.value, exact=True)
        if strategy == LocatorStrategy.PLACEHOLDER:
            return frame.get_by_placeholder(candidate.value, exact=True)
        if strategy == LocatorStrategy.ALT_TEXT:
            return frame.get_by_alt_text(candidate.value, exact=True)
        if strategy == LocatorStrategy.TITLE:
            return frame.get_by_title(candidate.value, exact=True)
        if strategy == LocatorStrategy.TEXT:
            return frame.get_by_text(candidate.value, exact=True)
        if strategy == LocatorStrategy.STABLE_ATTRIBUTE:
            value = _json_object(candidate.value)
            tag = _css_identifier(value.get("tag", "")) if value.get("tag") else ""
            return frame.locator(
                f"{tag}[{_css_identifier(value['attribute'])}={json.dumps(value['value'])}]"
            )
        if strategy == LocatorStrategy.CSS:
            if element.shadow_host_path:
                current = frame.locator(element.shadow_host_path[0])
                for host in element.shadow_host_path[1:]:
                    current = current.locator(host)
                return current.locator(candidate.value)
            return frame.locator(candidate.value)
        if strategy == LocatorStrategy.XPATH:
            return frame.locator(f"xpath={candidate.value}")
        raise ValueError(f"unsupported locator strategy: {strategy.value}")

    async def _perform(
        self,
        page: Page,
        locator: Optional[Locator],
        candidate: ActionCandidate,
    ) -> None:
        kind = candidate.kind
        if kind == ActionKind.SCROLL:
            delta = int(_parameter(candidate, "delta_y"))
            await page.evaluate("delta => window.scrollBy(0, delta)", delta)
            return
        if locator is None:
            raise ActionExecutionError(f"{kind.value} requires a locator")
        if kind == ActionKind.CLICK:
            await locator.click(timeout=self.action_timeout_ms)
        elif kind == ActionKind.DOUBLE_CLICK:
            await locator.dblclick(timeout=self.action_timeout_ms)
        elif kind == ActionKind.HOVER:
            await locator.hover(timeout=self.action_timeout_ms)
        elif kind == ActionKind.CHECK:
            await locator.click(timeout=self.action_timeout_ms)
        elif kind == ActionKind.UNCHECK:
            await locator.click(timeout=self.action_timeout_ms)
        elif kind == ActionKind.SELECT_OPTION:
            await locator.select_option(
                value=_parameter(candidate, "option_value"),
                timeout=self.action_timeout_ms,
            )
        elif kind == ActionKind.FILL:
            await locator.fill(
                _parameter(candidate, "value"),
                timeout=self.action_timeout_ms,
            )
        elif kind == ActionKind.CLEAR:
            await locator.clear(timeout=self.action_timeout_ms)
        elif kind == ActionKind.PRESS:
            await locator.press(
                _parameter(candidate, "key"),
                timeout=self.action_timeout_ms,
            )
        elif kind == ActionKind.SUBMIT:
            await locator.press("Enter", timeout=self.action_timeout_ms)
        else:
            raise ActionExecutionError(f"unsupported action kind: {kind.value}")

    async def _result_page(self, page: Page, pages_before: set[Page]) -> Page:
        await asyncio.sleep(0)
        new_pages = [
            candidate
            for candidate in page.context.pages
            if candidate not in pages_before
        ]
        result = new_pages[-1] if new_pages else page
        if result.is_closed():
            remaining = [
                candidate
                for candidate in page.context.pages
                if not candidate.is_closed()
            ]
            if not remaining:
                raise ActionExecutionError("action closed every page in the context")
            result = remaining[-1]
        try:
            await result.wait_for_load_state(
                "domcontentloaded",
                timeout=self.action_timeout_ms,
            )
        except PlaywrightError:
            pass
        return result

    @staticmethod
    async def _frame_by_path(page: Page, frame_path: str) -> Frame:
        current = page.main_frame
        if frame_path == "main":
            return current
        parts = frame_path.split("/")
        if not parts or parts[0] != "main":
            raise ActionExecutionError(f"invalid frame path: {frame_path}")
        for part in parts[1:]:
            try:
                index = int(part)
                handles = await current.locator("iframe, frame").element_handles()
                handle = handles[index]
                content_frame = await handle.content_frame()
                if content_frame is None:
                    raise ActionExecutionError(
                        f"frame element has no active document: {frame_path}"
                    )
                current = content_frame
            except (ValueError, IndexError, PlaywrightError) as exc:
                raise ActionExecutionError(
                    f"frame path no longer exists: {frame_path}"
                ) from exc
        return current

    def _correlated_evidence(
        self,
        action_id: str,
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[BrowserEvent, ...],
        tuple[NetworkExchange, ...],
    ]:
        events = tuple(
            event
            for event in self.store.iter_records(BrowserEvent)
            if event.action_id == action_id
        )
        exchanges = tuple(
            exchange
            for exchange in self.store.iter_records(NetworkExchange)
            if exchange.action_id == action_id
        )
        return (
            tuple(event.event_id for event in events),
            tuple(exchange.exchange_id for exchange in exchanges),
            events,
            exchanges,
        )

    def _effects(
        self,
        parent: StateSnapshot,
        result: StateSnapshot,
        candidate: ActionCandidate,
        events: tuple[BrowserEvent, ...],
        exchanges: tuple[NetworkExchange, ...],
    ) -> tuple[TransitionEffect, ...]:
        effects: list[TransitionEffect] = []

        def add(kind: EffectKind, summary: str) -> None:
            if any(effect.kind == kind for effect in effects):
                return
            effects.append(
                TransitionEffect(
                    kind=kind,
                    summary=summary,
                    artifact_ids=tuple(
                        artifact.artifact_id for artifact in result.artifacts
                    ),
                )
            )

        if parent.url != result.url:
            add(EffectKind.NAVIGATION, f"URL changed to {result.url}")
        if parent.fingerprint != result.fingerprint:
            add(EffectKind.DOM_CHANGE, "resulting state fingerprint changed")
        if candidate.kind in {
            ActionKind.CHECK,
            ActionKind.CLEAR,
            ActionKind.FILL,
            ActionKind.SELECT_OPTION,
            ActionKind.UNCHECK,
        }:
            add(EffectKind.FORM_CHANGE, f"{candidate.kind.value} changed form state")
        if result.modal_count > parent.modal_count:
            add(EffectKind.MODAL_OPENED, "visible modal count increased")
        elif result.modal_count < parent.modal_count:
            add(EffectKind.MODAL_CLOSED, "visible modal count decreased")
        if set(result.notifications) - set(parent.notifications):
            add(EffectKind.NOTIFICATION, "new notification evidence appeared")
        if set(result.errors) - set(parent.errors):
            add(EffectKind.ERROR, "new error evidence appeared")
        if parent.storage_fingerprint != result.storage_fingerprint:
            add(EffectKind.STORAGE_CHANGE, "browser storage fingerprint changed")
        if exchanges:
            add(
                EffectKind.NETWORK_ACTIVITY,
                f"{len(exchanges)} network exchanges observed",
            )

        event_kinds = {event.kind for event in events}
        if BrowserEventKind.POPUP in event_kinds:
            add(EffectKind.POPUP_OPENED, "popup page opened")
        if BrowserEventKind.DOWNLOAD in event_kinds:
            add(EffectKind.DOWNLOAD_STARTED, "download started")
        if BrowserEventKind.DIALOG in event_kinds:
            add(EffectKind.DIALOG_OPENED, "browser dialog opened")
        if BrowserEventKind.CONSOLE in event_kinds:
            add(EffectKind.CONSOLE_OUTPUT, "console output was emitted")
        if BrowserEventKind.PAGE_ERROR in event_kinds:
            add(EffectKind.ERROR, "unhandled page error was emitted")
        if not effects:
            add(EffectKind.NO_OP, "action completed without an observable state change")
        return tuple(effects)


def _attribute_value(element: ElementSnapshot, name: str) -> Optional[str]:
    return next(
        (
            attribute.value or attribute.safe_value
            for attribute in element.attributes
            if attribute.name.lower() == name.lower()
        ),
        None,
    )


def _parameter(candidate: ActionCandidate, name: str) -> str:
    parameter = next(
        (item for item in candidate.parameters if item.name == name),
        None,
    )
    if parameter is None or parameter.redacted or parameter.value is None:
        raise ActionExecutionError(f"action parameter is unavailable: {name}")
    return parameter.value


def _matching_keyword(text: str, keywords: tuple[str, ...]) -> Optional[str]:
    for keyword in sorted(keywords, key=len, reverse=True):
        pattern = rf"(?<![a-z0-9]){re.escape(keyword.lower())}(?![a-z0-9])"
        if re.search(pattern, text):
            return keyword
    return None


def _rule_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _origin(url: str) -> Optional[str]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def _transition_id(action_id: str, status: ActionStatus, outcome: str) -> str:
    return f"transition-{hash_text(f'{action_id}|{status.value}|{outcome}')[:32]}"


def _duration_ms(started: datetime, completed: datetime) -> int:
    return max(0, int((completed - started).total_seconds() * 1_000))


def _json_object(value: str) -> dict[str, str]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in parsed.items()
    ):
        raise ValueError("locator value must be a string mapping")
    return parsed


def _css_identifier(value: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", value) is None:
        raise ValueError(f"unsafe CSS identifier: {value}")
    return value


__all__ = [
    "ActionExecutionError",
    "ActionExecutionOutcome",
    "ActionExecutor",
    "ActionPlanner",
    "ActionPlanningError",
    "ActionPolicyConfig",
    "PlannedAction",
]
