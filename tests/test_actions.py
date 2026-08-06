"""Deterministic action planning and safety-policy tests."""

from pathlib import Path

import pytest

from scraper.actions import ActionPlanner
from scraper.artifact_store import ArtifactStore
from scraper.extractor import CapturedState
from scraper.models import (
    ActionCandidate,
    ActionKind,
    ActionRisk,
    ActionStatus,
    CapturePolicy,
    ElementSnapshot,
    FrameSnapshot,
    ScrapeRun,
    ScrollPosition,
    SelectOptionSnapshot,
    StateSnapshot,
    ValueCapture,
    Viewport,
)


FINGERPRINT = "a" * 64


def _element(
    element_id: str,
    *,
    tag: str = "button",
    role: str = "button",
    name: str,
    visible: bool = True,
    enabled: bool = True,
    attributes: tuple[ValueCapture, ...] = (),
    checked: bool | None = None,
    options: tuple[SelectOptionSnapshot, ...] = (),
) -> ElementSnapshot:
    return ElementSnapshot(
        element_id=element_id,
        frame_id="frame-main",
        tag=tag,
        role=role,
        accessible_name=name,
        attributes=attributes,
        interactive=True,
        interaction_signals=(f"role:{role}",),
        visible=visible,
        enabled=enabled,
        checked=checked,
        options=options,
    )


def _capture(
    store: ArtifactStore,
    elements: tuple[ElementSnapshot, ...],
) -> CapturedState:
    state = StateSnapshot(
        state_id="state-root",
        run_id=store.run.run_id,
        sequence=0,
        fingerprint=FINGERPRINT,
        page_id="page-main",
        url="https://portal.test/start",
        viewport=Viewport(width=1280, height=720),
        scroll=ScrollPosition(maximum_y=1200),
        frames=(
            FrameSnapshot(
                frame_id="frame-main",
                frame_path="main",
                url="https://portal.test/start",
                is_main=True,
                element_ids=tuple(element.element_id for element in elements),
            ),
        ),
        element_ids=tuple(element.element_id for element in elements),
    )
    receipt = store.append_record(state)
    return CapturedState(state=state, elements=elements, receipt=receipt)


@pytest.mark.asyncio
async def test_planner_records_every_candidate_with_auditable_risk(
    tmp_path: Path,
) -> None:
    run = ScrapeRun(
        run_id="action-run",
        root_url="https://portal.test/start",
        allowed_origins=("https://portal.test",),
    )
    store = ArtifactStore.create(tmp_path / "crawls", run)
    elements = (
        ElementSnapshot(
            element_id="root",
            frame_id="frame-main",
            tag="html",
            visible=True,
            enabled=True,
        ),
        _element("open", name="Open details"),
        _element("run", name="Run certification test"),
        _element("delete", name="Delete account"),
        _element("hidden", name="Hidden navigation", visible=False),
        _element(
            "external",
            tag="a",
            role="link",
            name="External documentation",
            attributes=(ValueCapture(name="href", value="https://outside.test/docs"),),
        ),
        _element(
            "toggle",
            tag="input",
            role="checkbox",
            name="Show logs",
            checked=False,
        ),
        _element(
            "environment",
            tag="select",
            role="combobox",
            name="Environment",
            options=(
                SelectOptionSnapshot(label="Sandbox", value="sandbox", selected=True),
                SelectOptionSnapshot(label="Production", value="production"),
            ),
        ),
    )
    capture = _capture(store, elements)

    planned = await ActionPlanner(store).plan(capture)
    by_element = {
        (item.element.element_id, item.candidate.kind): item.candidate
        for item in planned
    }

    assert by_element[("open", ActionKind.CLICK)].risk == ActionRisk.SAFE
    assert by_element[("open", ActionKind.CLICK)].status == ActionStatus.PENDING
    assert by_element[("run", ActionKind.CLICK)].risk == ActionRisk.REVIEW_REQUIRED
    assert by_element[("run", ActionKind.CLICK)].status == ActionStatus.SKIPPED
    assert by_element[("delete", ActionKind.CLICK)].risk == ActionRisk.BLOCKED
    assert by_element[("delete", ActionKind.CLICK)].status == ActionStatus.SKIPPED
    assert by_element[("hidden", ActionKind.CLICK)].policy_rule == "visibility.hidden"
    assert by_element[("external", ActionKind.CLICK)].policy_rule == (
        "navigation.cross_origin"
    )
    assert by_element[("toggle", ActionKind.CHECK)].status == ActionStatus.PENDING
    assert by_element[("environment", ActionKind.SELECT_OPTION)].risk == (
        ActionRisk.REVIEW_REQUIRED
    )
    assert any(item.candidate.kind == ActionKind.SCROLL for item in planned)
    assert len(list(store.iter_records(ActionCandidate))) == len(planned)
    assert len({item.candidate.action_id for item in planned}) == len(planned)


@pytest.mark.asyncio
async def test_redacted_external_href_still_enforces_origin_policy(
    tmp_path: Path,
) -> None:
    run = ScrapeRun(
        run_id="redacted-link-run",
        root_url="https://portal.test/start",
        allowed_origins=("https://portal.test",),
    )
    store = ArtifactStore.create(tmp_path / "crawls", run)
    link = _element(
        "external",
        tag="a",
        role="link",
        name="View result",
        attributes=(
            ValueCapture(
                name="href",
                safe_value="https://outside.test/result?token=[REDACTED]",
                value_hash="b" * 64,
                redacted=True,
            ),
        ),
    )

    candidate = (await ActionPlanner(store).plan(_capture(store, (link,))))[0].candidate

    assert candidate.risk == ActionRisk.BLOCKED
    assert candidate.status == ActionStatus.SKIPPED
    assert candidate.policy_rule == "navigation.cross_origin"


@pytest.mark.asyncio
async def test_review_actions_can_be_explicitly_enabled(tmp_path: Path) -> None:
    run = ScrapeRun(
        run_id="review-run",
        root_url="https://portal.test/start",
        allowed_origins=("https://portal.test",),
        capture_policy=CapturePolicy(allow_review_required_actions=True),
    )
    store = ArtifactStore.create(tmp_path / "crawls", run)
    action = _element("run", name="Run test")

    candidate = (await ActionPlanner(store).plan(_capture(store, (action,))))[
        0
    ].candidate

    assert candidate.risk == ActionRisk.REVIEW_REQUIRED
    assert candidate.status == ActionStatus.PENDING
