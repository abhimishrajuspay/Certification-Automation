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


def test_taught_clicks_without_modal_remain_explicit_steps() -> None:
    guide = compile_taught_guide(
        (
            OperatorClick(sequence=0, target=_target("One", "#one")),
            OperatorClick(sequence=1, target=_target("Two", "#two")),
        )
    )

    assert len(guide.steps) == 2
    assert guide.repeat_rules == ()


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
