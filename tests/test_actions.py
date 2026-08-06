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
    ElementContext,
    ElementSnapshot,
    FrameSnapshot,
    LocatorCandidate,
    LocatorStrategy,
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
    title: str | None = None,
    visible: bool = True,
    enabled: bool = True,
    attributes: tuple[ValueCapture, ...] = (),
    checked: bool | None = None,
    options: tuple[SelectOptionSnapshot, ...] = (),
    input_type: str | None = None,
    interaction_signals: tuple[str, ...] | None = None,
    css_path: str | None = None,
    parent_css_path: str | None = None,
    row_label: str | None = None,
) -> ElementSnapshot:
    signals = interaction_signals or (f"role:{role}",)
    locators = (
        (
            LocatorCandidate(
                strategy=LocatorStrategy.CSS,
                value=css_path,
                confidence=0.5,
                unique_match_count=1,
                is_primary=True,
            ),
        )
        if css_path is not None
        else ()
    )
    return ElementSnapshot(
        element_id=element_id,
        frame_id="frame-main",
        tag=tag,
        role=role,
        accessible_name=name,
        title=title,
        input_type=input_type,
        attributes=attributes,
        interactive=True,
        interaction_signals=signals,
        visible=visible,
        enabled=enabled,
        checked=checked,
        options=options,
        locators=locators,
        parent_css_path=parent_css_path,
        context=ElementContext(row_label=row_label),
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


@pytest.mark.asyncio
async def test_authentication_and_structural_submit_actions_require_review(
    tmp_path: Path,
) -> None:
    store = ArtifactStore.create(
        tmp_path / "crawls",
        ScrapeRun(
            run_id="authentication-actions",
            root_url="https://portal.test/login",
            allowed_origins=("https://portal.test",),
        ),
    )
    elements = (
        _element("login", name="Login"),
        _element("unnamed-submit", name="Continue", input_type="submit"),
    )

    planned = await ActionPlanner(store).plan(_capture(store, elements))
    candidates = {item.element.element_id: item.candidate for item in planned}

    assert candidates["login"].risk == ActionRisk.REVIEW_REQUIRED
    assert candidates["login"].status == ActionStatus.SKIPPED
    assert candidates["unnamed-submit"].risk == ActionRisk.REVIEW_REQUIRED
    assert candidates["unnamed-submit"].policy_rule == "form.submit_requires_review"


@pytest.mark.asyncio
async def test_test_execution_requires_review_but_navigation_and_details_are_safe(
    tmp_path: Path,
) -> None:
    store = ArtifactStore.create(
        tmp_path / "crawls",
        ScrapeRun(
            run_id="test-control-safety",
            root_url="https://portal.test/start",
            allowed_origins=("https://portal.test",),
        ),
    )
    elements = (
        _element(
            "category",
            tag="a",
            role="link",
            name="▶",
            title="Test",
            attributes=(
                ValueCapture(name="href", value="/TestCases/Cases?api=BillFetch"),
            ),
        ),
        _element(
            "execute",
            tag="button",
            role="button",
            name="▶",
            title="Test",
            input_type="button",
        ),
        _element(
            "details",
            tag="a",
            role="link",
            name="i",
            title="Test case details",
            attributes=(ValueCapture(name="href", value="#"),),
        ),
    )

    planned = await ActionPlanner(store).plan(_capture(store, elements))
    candidates = {item.element.element_id: item.candidate for item in planned}

    assert candidates["category"].risk == ActionRisk.SAFE
    assert candidates["category"].status == ActionStatus.PENDING
    assert candidates["execute"].risk == ActionRisk.REVIEW_REQUIRED
    assert candidates["execute"].status == ActionStatus.SKIPPED
    assert candidates["execute"].policy_rule == "semantic.review.test_execution"
    assert candidates["details"].risk == ActionRisk.SAFE
    assert candidates["details"].status == ActionStatus.PENDING


@pytest.mark.asyncio
async def test_table_row_observations_are_planned_before_navigation(
    tmp_path: Path,
) -> None:
    store = ArtifactStore.create(
        tmp_path / "crawls",
        ScrapeRun(
            run_id="observation-priority",
            root_url="https://portal.test/start",
            allowed_origins=("https://portal.test",),
        ),
    )
    elements = (
        _element(
            "unrelated-listener",
            tag="div",
            role="button",
            name="Toggle sidebar",
            interaction_signals=("listener:click",),
        ),
        _element(
            "next-page",
            tag="a",
            role="link",
            name="Next",
            attributes=(ValueCapture(name="href", value="/cases?page=2"),),
        ),
        _element(
            "row-details",
            tag="a",
            role="link",
            name="Information",
            attributes=(ValueCapture(name="href", value="#"),),
            # Many portals use a delegated listener on the table instead of a
            # listener attached directly to each details anchor.
            interaction_signals=("native-control", "role:link"),
            row_label="case-1",
        ),
    )

    planned = await ActionPlanner(store).plan(_capture(store, elements))

    assert [item.element.element_id for item in planned] == [
        "row-details",
        "next-page",
        "unrelated-listener",
    ]


@pytest.mark.asyncio
async def test_nested_visual_targets_are_recorded_but_not_executed_twice(
    tmp_path: Path,
) -> None:
    store = ArtifactStore.create(
        tmp_path / "crawls",
        ScrapeRun(
            run_id="nested-targets",
            root_url="https://portal.test/start",
            allowed_origins=("https://portal.test",),
        ),
    )
    elements = (
        _element(
            "link",
            tag="a",
            role="link",
            name="Dashboard",
            interaction_signals=("cursor:pointer", "native-control", "role:link"),
            css_path="#dashboard",
            parent_css_path="html > body",
        ),
        _element(
            "label",
            tag="span",
            role="",
            name="Dashboard",
            interaction_signals=("cursor:pointer",),
            css_path="#dashboard-label",
            parent_css_path="#dashboard",
        ),
    )

    planned = await ActionPlanner(store).plan(_capture(store, elements))
    candidates = {item.element.element_id: item.candidate for item in planned}

    assert candidates["link"].status == ActionStatus.PENDING
    assert candidates["label"].status == ActionStatus.SKIPPED
    assert candidates["label"].policy_rule == "dedup.nested_target"
    assert len(list(store.iter_records(ActionCandidate))) == 2


@pytest.mark.asyncio
async def test_ambiguous_delegated_container_is_audited_but_skipped(
    tmp_path: Path,
) -> None:
    store = ArtifactStore.create(
        tmp_path / "crawls",
        ScrapeRun(
            run_id="delegated-container",
            root_url="https://portal.test/start",
            allowed_origins=("https://portal.test",),
        ),
    )
    elements = (
        _element(
            "container",
            tag="div",
            role="",
            name="Portal navigation",
            interaction_signals=("listener:click",),
            css_path="#portal-navigation",
            parent_css_path="html > body",
        ),
        _element(
            "first",
            tag="a",
            role="link",
            name="First",
            interaction_signals=("native-control", "role:link"),
            css_path="#first",
            parent_css_path="#portal-navigation",
        ),
        _element(
            "second",
            tag="a",
            role="link",
            name="Second",
            interaction_signals=("native-control", "role:link"),
            css_path="#second",
            parent_css_path="#portal-navigation",
        ),
    )

    planned = await ActionPlanner(store).plan(_capture(store, elements))
    candidates = {item.element.element_id: item.candidate for item in planned}

    assert candidates["container"].status == ActionStatus.SKIPPED
    assert candidates["container"].policy_rule == "delegation.ambiguous_container"
    assert candidates["first"].status == ActionStatus.PENDING
    assert candidates["second"].status == ActionStatus.PENDING
