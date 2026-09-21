"""Typed crawl-guide, matching, and teaching compiler tests."""

from pathlib import Path

import pytest

from scraper.artifact_store import ArtifactStore
from scraper.extractor import CapturedState
from scraper.guidance import (
    CrawlGuide,
    GuidanceError,
    GuideRepeatRule,
    GuideStep,
    GuideTarget,
    OperatorClick,
    compile_taught_guide,
    guide_sha256,
    load_crawl_guide,
    target_elements,
    write_crawl_guide,
)
from scraper.models import (
    ElementContext,
    ElementSnapshot,
    FrameSnapshot,
    LocatorCandidate,
    LocatorStrategy,
    ScrapeRun,
    StateSnapshot,
    ValueCapture,
    Viewport,
)


def _target(name: str, css: str) -> GuideTarget:
    return GuideTarget(role="button", accessible_name=name, css=css)


def _capture(tmp_path: Path) -> CapturedState:
    store = ArtifactStore.create(
        tmp_path / "crawls",
        ScrapeRun(
            run_id="guide-run",
            root_url="https://portal.test/cases",
            allowed_origins=("https://portal.test",),
        ),
    )
    elements = (
        ElementSnapshot(
            element_id="info-1",
            frame_id="main",
            tag="button",
            role="button",
            accessible_name="Info",
            attributes=(ValueCapture(name="data-testid", value="info-one"),),
            visible=True,
            enabled=True,
            interactive=True,
            interaction_signals=("role:button",),
            context=ElementContext(row_label="TC_01"),
            locators=(
                LocatorCandidate(
                    strategy=LocatorStrategy.CSS,
                    value="#info-one",
                    confidence=0.5,
                    unique_match_count=1,
                    is_primary=True,
                ),
            ),
        ),
        ElementSnapshot(
            element_id="info-2",
            frame_id="main",
            tag="button",
            role="button",
            accessible_name="Info",
            attributes=(ValueCapture(name="data-testid", value="info-two"),),
            visible=True,
            enabled=True,
            interactive=True,
            interaction_signals=("role:button",),
            context=ElementContext(row_label="TC_02"),
            locators=(
                LocatorCandidate(
                    strategy=LocatorStrategy.CSS,
                    value="#info-two",
                    confidence=0.5,
                    unique_match_count=1,
                    is_primary=True,
                ),
            ),
        ),
    )
    state = StateSnapshot(
        state_id="root",
        run_id="guide-run",
        sequence=0,
        fingerprint="a" * 64,
        page_id="page",
        url="https://portal.test/cases",
        viewport=Viewport(width=1280, height=720),
        frames=(
            FrameSnapshot(
                frame_id="main",
                frame_path="main",
                url="https://portal.test/cases",
                is_main=True,
                element_ids=("info-1", "info-2"),
            ),
        ),
        element_ids=("info-1", "info-2"),
    )
    receipt = store.append_record(state)
    return CapturedState(state=state, elements=elements, receipt=receipt)


def test_guide_round_trip_is_strict_and_content_hashed(tmp_path: Path) -> None:
    guide = CrawlGuide(
        steps=(GuideStep(name="open", target=_target("Open", "#open")),),
        repeat_rules=(
            GuideRepeatRule(
                name="rows",
                target=_target("Info", ".info"),
                close_target=_target("Close", ".close"),
            ),
        ),
    )
    path = tmp_path / "guides" / "portal.json"

    write_crawl_guide(path, guide)
    loaded, digest = load_crawl_guide(path)

    assert loaded == guide
    assert digest == guide_sha256(guide)
    assert len(digest) == 64
    with pytest.raises(GuidanceError, match="exists"):
        write_crawl_guide(path, guide)


def test_taught_clicks_compile_route_and_repeated_row_observation() -> None:
    clicks = (
        OperatorClick(sequence=0, target=_target("Manage Test", "#manage")),
        OperatorClick(
            sequence=1,
            target=_target("Info", "#row-1-info"),
            row_label="TC_01",
        ),
        OperatorClick(
            sequence=2,
            target=_target("Close", "#close"),
            inside_dialog=True,
        ),
    )

    guide = compile_taught_guide(clicks)

    assert guide.source == "taught"
    assert [step.target.accessible_name for step in guide.steps] == ["Manage Test"]
    assert len(guide.repeat_rules) == 1
    assert guide.repeat_rules[0].target.accessible_name == "Info"
    assert guide.repeat_rules[0].close_target is not None
    assert guide.repeat_rules[0].close_target.accessible_name == "Close"


def test_taught_portal_wide_clicks_compile_nested_branch_fanout() -> None:
    clicks = (
        OperatorClick(sequence=0, target=_target("Manage Test", "#manage")),
        OperatorClick(sequence=1, target=_target("HS51", "#bou")),
        OperatorClick(
            sequence=2,
            target=GuideTarget(
                tag="a",
                role="link",
                accessible_name="UPI",
                css="#group-list > a:nth-of-type(1)",
            ),
        ),
        OperatorClick(
            sequence=3,
            target=GuideTarget(
                tag="a",
                role="link",
                accessible_name="Test",
                title="Test",
                css="#slots > tr:nth-of-type(1) a",
            ),
            row_label="SessionRegistration",
        ),
        OperatorClick(
            sequence=4,
            target=_target("Test case details", "#info"),
            row_label="SDKREG_UAT_01",
        ),
        OperatorClick(
            sequence=5,
            target=_target("×", "#close"),
            inside_dialog=True,
        ),
    )

    guide = compile_taught_guide(clicks, branch_depth=2)

    assert [step.target.accessible_name for step in guide.steps] == [
        "Manage Test",
        "HS51",
    ]
    assert len(guide.branch_rules) == 2
    assert guide.branch_rules[0].target.css_regex is not None
    assert guide.branch_rules[0].target.accessible_name is None
    assert guide.branch_rules[1].target.accessible_name == "Test"
    assert guide.branch_rules[1].require_row_label is True
    assert guide.repeat_rules[0].target.accessible_name == "Test case details"
    assert guide.repeat_rules[0].close_target.accessible_name == "×"


def test_css_regex_target_matches_all_taught_siblings(tmp_path: Path) -> None:
    capture = _capture(tmp_path)
    target = GuideTarget(css_regex=r"#info-(?:one|two)")

    matches = target_elements(capture, target, allow_many=True)

    assert [item.element_id for item in matches] == ["info-1", "info-2"]


def test_taught_clicks_without_modal_remain_explicit_steps() -> None:
    guide = compile_taught_guide(
        (
            OperatorClick(sequence=0, target=_target("One", "#one")),
            OperatorClick(sequence=1, target=_target("Two", "#two")),
        )
    )

    assert len(guide.steps) == 2
    assert guide.repeat_rules == ()


def test_taught_dialog_link_is_rejected_as_a_close_control() -> None:
    with pytest.raises(GuidanceError, match="actual button-like close"):
        compile_taught_guide(
            (
                OperatorClick(
                    sequence=0,
                    target=_target("Info", "#info"),
                    row_label="TC_01",
                ),
                OperatorClick(
                    sequence=1,
                    target=GuideTarget(
                        tag="a",
                        role="link",
                        accessible_name="Open full page",
                    ),
                    inside_dialog=True,
                ),
            )
        )


def test_target_matching_uses_css_to_disambiguate_repeated_roles(
    tmp_path: Path,
) -> None:
    capture = _capture(tmp_path)

    repeated = target_elements(
        capture,
        GuideTarget(role="button", accessible_name="Info"),
        allow_many=True,
    )
    unique = target_elements(
        capture,
        GuideTarget(
            role="button",
            accessible_name="Info",
            css="#info-two",
        ),
        allow_many=False,
    )

    assert [element.context.row_label for element in repeated] == ["TC_01", "TC_02"]
    assert [element.element_id for element in unique] == ["info-2"]


def test_close_target_resolves_uniquely_among_hidden_dialog_controls(
    tmp_path: Path,
) -> None:
    """Only the opened dialog's close control is visible, so a close target
    must disambiguate same-shaped hidden dialogs by visibility."""

    store = ArtifactStore.create(
        tmp_path / "crawls",
        ScrapeRun(
            run_id="close-run",
            root_url="https://portal.test/cases",
            allowed_origins=("https://portal.test",),
        ),
    )

    def close_button(element_id: str, visible: bool) -> ElementSnapshot:
        return ElementSnapshot(
            element_id=element_id,
            frame_id="main",
            tag="button",
            role="button",
            accessible_name="×",
            text="×",
            visible=visible,
            enabled=True,
            interactive=True,
            interaction_signals=("role:button",),
            attributes=(ValueCapture(name="class", value="close"),),
            locators=(
                LocatorCandidate(
                    strategy=LocatorStrategy.CSS,
                    value=f"#{element_id}",
                    confidence=0.5,
                    unique_match_count=1,
                    is_primary=True,
                ),
            ),
        )

    elements = (
        close_button("close-hidden-1", False),
        close_button("close-open", True),
        close_button("close-hidden-2", False),
    )
    state = StateSnapshot(
        state_id="modal",
        run_id="close-run",
        sequence=1,
        fingerprint="b" * 64,
        page_id="page",
        url="https://portal.test/cases",
        viewport=Viewport(width=1280, height=720),
        frames=(
            FrameSnapshot(
                frame_id="main",
                frame_path="main",
                url="https://portal.test/cases",
                is_main=True,
                element_ids=tuple(item.element_id for item in elements),
            ),
        ),
        element_ids=tuple(item.element_id for item in elements),
    )
    capture = CapturedState(
        state=state,
        elements=elements,
        receipt=store.append_record(state),
    )
    target = GuideTarget(
        tag="button",
        role="button",
        accessible_name="×",
        text="×",
        css_classes=("close",),
    )

    ambiguous = target_elements(capture, target, allow_many=False)
    resolved = target_elements(
        capture,
        target,
        allow_many=False,
        require_visible=True,
    )

    assert ambiguous == ()
    assert [item.element_id for item in resolved] == ["close-open"]


def test_icon_only_row_target_compiles_and_resolves_via_class_membership(
    tmp_path: Path,
) -> None:
    """A bare icon anchor carries only its class list; generalization and
    matching must both work from that single signal."""

    clicks = (
        OperatorClick(
            sequence=0,
            target=GuideTarget(
                tag="a",
                role="link",
                accessible_name="Manage Test",
                css="aside > ul > li:nth-of-type(2) > a",
            ),
        ),
        OperatorClick(
            sequence=1,
            target=GuideTarget(tag="a", css_classes=("getTestCaseDet",)),
            row_label="MP_03",
        ),
        OperatorClick(
            sequence=2,
            target=GuideTarget(
                tag="button",
                role="button",
                accessible_name="×",
                text="×",
                css_classes=("close",),
            ),
            inside_dialog=True,
        ),
    )
    guide = compile_taught_guide(clicks, branch_depth=1)
    repeat = guide.repeat_rules[0]
    assert repeat.target.css_classes == ("getTestCaseDet",)
    assert repeat.target.test_id is None
    assert repeat.target.css_regex is None

    elements = tuple(
        ElementSnapshot(
            element_id=element_id,
            frame_id="main",
            tag="a",
            visible=True,
            enabled=True,
            interactive=True,
            interaction_signals=("click",),
            attributes=(ValueCapture(name="class", value=class_value),),
            locators=(
                LocatorCandidate(
                    strategy=LocatorStrategy.CSS,
                    value=f"#{element_id}",
                    confidence=0.5,
                    unique_match_count=1,
                    is_primary=True,
                ),
            ),
        )
        for element_id, class_value in (
            ("info-1", "getTestCaseDet"),
            ("info-2", "getTestCaseDet fa"),
            ("ready", "ready"),
        )
    )
    store = ArtifactStore.create(
        tmp_path / "crawls2",
        ScrapeRun(
            run_id="icon-run",
            root_url="https://portal.test/cases",
            allowed_origins=("https://portal.test",),
        ),
    )
    state = StateSnapshot(
        state_id="s",
        run_id="icon-run",
        sequence=2,
        fingerprint="d" * 64,
        page_id="page",
        url="https://portal.test/cases",
        viewport=Viewport(width=1280, height=720),
        frames=(
            FrameSnapshot(
                frame_id="main",
                frame_path="main",
                url="https://portal.test/cases",
                is_main=True,
                element_ids=tuple(item.element_id for item in elements),
            ),
        ),
        element_ids=tuple(item.element_id for item in elements),
    )
    capture = CapturedState(
        state=state,
        elements=elements,
        receipt=store.append_record(state),
    )
    matches = target_elements(capture, repeat.target, allow_many=True)
    assert [item.element_id for item in matches] == ["info-1", "info-2"]
