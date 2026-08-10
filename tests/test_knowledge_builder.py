"""Deterministic normalization tests using only synthetic crawl evidence."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest

from knowledge.builder import (
    KnowledgeBuildConfig,
    KnowledgeBuildError,
    PortalKnowledgeBuilder,
    _contains_identifier,
)
from knowledge.exporter import KnowledgeExportError, export_knowledge
from scraper.artifact_store import AppendReceipt, ArtifactStore
from scraper.extractor import CapturedState
from scraper.models import (
    ActionCandidate,
    ActionKind,
    ActionRisk,
    ActionStatus,
    ArtifactKind,
    CoverageReport,
    CrawlCompletionGoal,
    ElementContext,
    ElementSnapshot,
    FrameElementCollection,
    FrameSnapshot,
    InteractionTransition,
    NetworkExchange,
    ScrapeRun,
    ScrapeRunStatus,
    StateSnapshot,
    TestcaseContextCoverage as ContextCoverage,
    ValueCapture,
    Viewport,
)
from scraper.testcase_context import TestcaseContextTracker as ContextTracker


NOW = datetime(2026, 8, 6, 10, 0, tzinfo=timezone.utc)
FRAME_ID = "frame-main"
ROOT = "html > body > main"
SUMMARY_TABLE = f"{ROOT} > table:nth-of-type(1)"
CASE_TABLE = f"{ROOT} > table:nth-of-type(2)"


def _element(
    element_id: str,
    *,
    tag: str,
    text: str | None,
    parent_css_path: str | None,
    role: str | None = None,
    accessible_name: str | None = None,
    title: str | None = None,
    attributes: tuple[ValueCapture, ...] = (),
    interactive: bool = False,
    visible: bool = True,
    row_label: str | None = None,
    ancestors: tuple[str, ...] = (),
    inside_dialog: bool = False,
) -> ElementSnapshot:
    return ElementSnapshot(
        element_id=element_id,
        frame_id=FRAME_ID,
        tag=tag,
        role=role,
        accessible_name=accessible_name,
        text=text,
        title=title,
        attributes=attributes,
        interactive=interactive,
        visible=visible,
        enabled=True,
        parent_css_path=parent_css_path,
        context=ElementContext(
            row_label=row_label,
            inside_dialog=inside_dialog,
            ancestor_summary=ancestors,
        ),
    )


def _headers(table: str, labels: tuple[str, ...], prefix: str) -> list[ElementSnapshot]:
    row = f"{table} > thead > tr"
    return [
        _element(
            f"{prefix}-header-{index}",
            tag="th",
            role="columnheader",
            text=label,
            parent_css_path=row,
            row_label=labels[0],
            ancestors=("tr[role=row]", "thead[role=rowgroup]", "table[role=table]"),
        )
        for index, label in enumerate(labels, start=1)
    ]


def _cells(
    table: str,
    row_number: int,
    values: tuple[str, ...],
    prefix: str,
) -> list[ElementSnapshot]:
    row = f"{table} > tbody > tr:nth-of-type({row_number})"
    return [
        _element(
            f"{prefix}-cell-{index}",
            tag="td",
            role="cell",
            text=value,
            parent_css_path=row,
            row_label=str(row_number),
            ancestors=("tr[role=row]", "tbody[role=rowgroup]", "table[role=table]"),
        )
        for index, value in enumerate(values, start=1)
    ]


def _case_controls(row_number: int, test_case_id: str) -> list[ElementSnapshot]:
    row = f"{CASE_TABLE} > tbody > tr:nth-of-type({row_number})"
    return [
        _element(
            f"{test_case_id}-details",
            tag="a",
            role="link",
            text="i",
            accessible_name="Information",
            title="Details",
            parent_css_path=f"{row} > td:nth-of-type(2)",
            attributes=(
                ValueCapture(name="href", value="#"),
                ValueCapture(name="data-case", value=test_case_id),
            ),
            interactive=True,
            row_label=str(row_number),
            ancestors=("td[role=cell]", "tr[role=row]", "table[role=table]"),
        ),
        _element(
            f"{test_case_id}-link",
            tag="a",
            role="link",
            text=test_case_id,
            accessible_name=test_case_id,
            parent_css_path=f"{row} > td:nth-of-type(2)",
            attributes=(ValueCapture(name="href", value=f"/logs?case={test_case_id}"),),
            interactive=True,
            row_label=str(row_number),
            ancestors=("td[role=cell]", "tr[role=row]", "table[role=table]"),
        ),
        _element(
            f"{test_case_id}-execute",
            tag="button",
            role="button",
            text="Run",
            accessible_name="Run",
            title="Test",
            parent_css_path=f"{row} > td:nth-of-type(8)",
            interactive=True,
            row_label=str(row_number),
            ancestors=("td[role=cell]", "tr[role=row]", "table[role=table]"),
        ),
    ]


def _base_elements() -> tuple[ElementSnapshot, ...]:
    summary_headers = _headers(
        SUMMARY_TABLE,
        ("API Name", "Total TCs"),
        "summary",
    )
    summary_cells = _cells(SUMMARY_TABLE, 1, ("Payments", "2"), "summary-row")
    case_headers = _headers(
        CASE_TABLE,
        (
            "#",
            "TC ID",
            "API Name",
            "Dependency Case",
            "Test Data",
            "RC",
            "Status",
            "Test/Log",
        ),
        "case",
    )
    first_cells = _cells(
        CASE_TABLE,
        1,
        ("1", "i TC_01", "Payments", "", "positive payload", "000", "pending", "▶"),
        "tc1",
    )
    second_cells = _cells(
        CASE_TABLE,
        2,
        (
            "2",
            "i TC_02",
            "Payments",
            "TC_01",
            "dependent payload",
            "000",
            "pending",
            "▶",
        ),
        "tc2",
    )
    return tuple(
        [
            *summary_headers,
            *summary_cells,
            *case_headers,
            *first_cells,
            *_case_controls(1, "TC_01"),
            *second_cells,
            *_case_controls(2, "TC_02"),
        ]
    )


def _state(
    store: ArtifactStore,
    *,
    state_id: str,
    sequence: int,
    url: str,
    elements: tuple[ElementSnapshot, ...],
    modal_count: int = 0,
) -> StateSnapshot:
    collection = FrameElementCollection(frame_id=FRAME_ID, elements=elements)
    reference = store.put_model(ArtifactKind.ELEMENTS, collection)
    frame = FrameSnapshot(
        frame_id=FRAME_ID,
        frame_path="main",
        url=url,
        is_main=True,
        element_ids=tuple(element.element_id for element in elements),
        elements_artifact=reference,
    )
    state = StateSnapshot(
        state_id=state_id,
        run_id=store.run.run_id,
        sequence=sequence,
        fingerprint=f"{sequence + 1:064x}",
        captured_at=NOW + timedelta(seconds=sequence),
        page_id="page-main",
        url=url,
        title="Certification cases",
        viewport=Viewport(width=1280, height=720),
        frames=(frame,),
        element_ids=frame.element_ids,
        modal_count=modal_count,
    )
    store.append_record(state)
    return state


def _modal_elements(
    test_case_id: str,
    sequence: int,
    *,
    dependency_case_id: str | None = None,
) -> tuple[ElementSnapshot, ...]:
    dependency = f" Dependency Case {dependency_case_id}" if dependency_case_id else ""
    return (
        _element(
            f"dialog-{sequence}",
            tag="div",
            role="dialog",
            text=f"Test Case Details {test_case_id}: validates the payment flow Close",
            parent_css_path=ROOT,
            ancestors=("main[role=main]",),
        ),
        _element(
            f"dialog-body-{sequence}",
            tag="div",
            text=f"{test_case_id}: validates the payment flow{dependency}",
            parent_css_path=f"{ROOT} > div[role=dialog]",
            ancestors=("div[role=dialog]", "main[role=main]"),
        ),
    )


def _build_store(
    tmp_path: Path,
    *,
    completed: bool,
    description_dependency_collision: bool = False,
) -> ArtifactStore:
    run = ScrapeRun(
        run_id="knowledge-run",
        root_url="https://portal.example.test/start",
        allowed_origins=("https://portal.example.test",),
        status=ScrapeRunStatus.RUNNING,
        started_at=NOW,
    )
    store = ArtifactStore.create(tmp_path / "crawls", run)
    base = _state(
        store,
        state_id="state-base",
        sequence=0,
        url="https://portal.example.test/cases?api=Payments&page=1",
        elements=_base_elements(),
    )
    first_modal = _state(
        store,
        state_id="state-modal-1",
        sequence=1,
        url=base.url,
        elements=_modal_elements("TC_01", 1),
        modal_count=1,
    )
    second_modal = _state(
        store,
        state_id="state-modal-2",
        sequence=2,
        url=base.url,
        elements=_modal_elements(
            "TC_02",
            2,
            dependency_case_id=("TC_01" if description_dependency_collision else None),
        ),
        modal_count=1,
    )

    for index, (test_case_id, resulting_state) in enumerate(
        (("TC_01", first_modal), ("TC_02", second_modal)),
        start=1,
    ):
        action = ActionCandidate(
            action_id=f"action-details-{index}",
            run_id=run.run_id,
            state_id=base.state_id,
            element_id=f"{test_case_id}-details",
            kind=ActionKind.CLICK,
            risk=ActionRisk.SAFE,
            status=ActionStatus.PENDING,
            policy_rule="interaction.discovery_safe",
            rationale="opens an informational details panel",
            discovered_at=NOW,
        )
        store.append_record(action)
        store.append_record(
            InteractionTransition(
                transition_id=f"transition-details-{index}",
                run_id=run.run_id,
                action_id=action.action_id,
                parent_state_id=base.state_id,
                resulting_state_id=resulting_state.state_id,
                status=ActionStatus.SUCCEEDED,
                started_at=NOW,
                completed_at=NOW,
                duration_ms=10,
            )
        )

    execute = ActionCandidate(
        action_id="action-execute-1",
        run_id=run.run_id,
        state_id=base.state_id,
        element_id="TC_01-execute",
        kind=ActionKind.CLICK,
        risk=ActionRisk.REVIEW_REQUIRED,
        status=ActionStatus.SKIPPED,
        policy_rule="semantic.review.test_execution",
        rationale="test execution requires review",
        discovered_at=NOW,
    )
    store.append_record(execute)
    store.append_record(
        InteractionTransition(
            transition_id="transition-execute-1",
            run_id=run.run_id,
            action_id=execute.action_id,
            parent_state_id=base.state_id,
            status=ActionStatus.SKIPPED,
            started_at=NOW,
            completed_at=NOW,
            duration_ms=0,
            skip_reason="test execution requires review",
        )
    )
    store.append_record(
        NetworkExchange(
            exchange_id="exchange-1",
            run_id=run.run_id,
            request_id="request-1",
            requested_at=NOW,
            url="https://portal.example.test/api/details?case=TC_01",
            method="GET",
            resource_type="fetch",
            response_status=200,
            completed_at=NOW,
        )
    )
    store.append_record(
        CoverageReport(
            run_id=run.run_id,
            generated_at=NOW,
            states_discovered=3,
            action_candidates=3,
            actions_succeeded=2,
            actions_skipped=1,
            routes_discovered=1,
            tables_discovered=2,
            modals_discovered=2,
            bounded_complete=completed,
            completion_reason="frontier exhausted" if completed else "still running",
        )
    )
    if completed:
        store.save_manifest(
            run.model_copy(
                update={
                    "status": ScrapeRunStatus.COMPLETED,
                    "ended_at": NOW + timedelta(seconds=3),
                    "completion_reason": "frontier exhausted",
                    "state_ids": (
                        base.state_id,
                        first_modal.state_id,
                        second_modal.state_id,
                    ),
                    "transition_ids": (
                        "transition-details-1",
                        "transition-details-2",
                        "transition-execute-1",
                    ),
                }
            )
        )
    return store


def test_completed_crawl_normalizes_testcases_dependencies_and_descriptions(
    tmp_path: Path,
) -> None:
    store = _build_store(tmp_path, completed=True)

    knowledge = PortalKnowledgeBuilder(store).build()

    assert knowledge.coverage.source_bounded_complete is True
    assert knowledge.coverage.testcase_context_complete is True
    assert knowledge.coverage.declared_test_cases == 2
    assert knowledge.coverage.test_cases_normalized == 2
    assert knowledge.coverage.descriptions_captured == 2
    assert knowledge.coverage.missing_description_ids == ()
    assert len(knowledge.tables) == 2
    assert len(knowledge.routes) == 1
    assert len(knowledge.network) == 1
    assert knowledge.network[0].occurrences == 1

    cases = {case.test_case_id: case for case in knowledge.test_cases}
    assert tuple(cases) == ("TC_01", "TC_02")
    assert cases["TC_01"].dependency_case_ids == ()
    assert cases["TC_02"].dependency_case_ids == ("TC_01",)
    assert cases["TC_01"].description == "TC_01: validates the payment flow"
    assert cases["TC_02"].description == "TC_02: validates the payment flow"
    execute = next(
        control
        for control in cases["TC_01"].controls
        if control.element_id == "TC_01-execute"
    )
    assert execute.risk == ActionRisk.REVIEW_REQUIRED
    assert execute.action_status == ActionStatus.SKIPPED
    assert execute.policy_rule == "semantic.review.test_execution"
    assert store.verify_integrity(strict=True).valid is True


def test_modal_dependency_id_is_not_misattributed_as_parent_description(
    tmp_path: Path,
) -> None:
    store = _build_store(
        tmp_path,
        completed=True,
        description_dependency_collision=True,
    )

    knowledge = PortalKnowledgeBuilder(store).build()
    cases = {case.test_case_id: case for case in knowledge.test_cases}

    assert knowledge.coverage.testcase_context_complete is True
    assert knowledge.coverage.conflicting_test_case_ids == ()
    assert cases["TC_01"].description == "TC_01: validates the payment flow"
    assert cases["TC_01"].conflicts == ()
    assert cases["TC_02"].description == (
        "TC_02: validates the payment flow Dependency Case TC_01"
    )


def test_tracker_uses_clicked_case_to_disambiguate_dependency_ids(
    tmp_path: Path,
) -> None:
    store = _build_store(
        tmp_path,
        completed=True,
        description_dependency_collision=True,
    )
    states = tuple(store.iter_records(StateSnapshot))
    tracker = ContextTracker(required_stable_observations=2)
    coverage: ContextCoverage | None = None

    for state, expected_id in zip(states, (None, "TC_01", "TC_02")):
        frame = state.frames[0]
        assert frame.elements_artifact is not None
        elements = FrameElementCollection.model_validate_json(
            store.read_bytes(frame.elements_artifact)
        ).elements
        coverage = tracker.observe(
            CapturedState(
                state=state,
                elements=elements,
                receipt=cast(AppendReceipt, object()),
            ),
            description_test_case_id=expected_id,
        )

    assert coverage is not None
    assert coverage.context_complete is True
    assert coverage.conflicting_test_case_ids == ()
    assert coverage.descriptions_captured == 2


def test_incomplete_crawl_is_rejected_unless_explicitly_allowed(
    tmp_path: Path,
) -> None:
    store = _build_store(tmp_path, completed=False)

    with pytest.raises(KnowledgeBuildError, match="completed crawl goal"):
        PortalKnowledgeBuilder(store).build()

    diagnostic = PortalKnowledgeBuilder(
        store,
        KnowledgeBuildConfig(allow_incomplete=True, verify_integrity=False),
    ).build()

    assert diagnostic.coverage.source_bounded_complete is False
    assert diagnostic.coverage.testcase_context_complete is True
    assert "knowledge was generated from incomplete crawl evidence" in (
        diagnostic.coverage.limitations
    )


def test_incremental_completion_tracker_matches_normalized_context_gate(
    tmp_path: Path,
) -> None:
    store = _build_store(tmp_path, completed=True)
    tracker = ContextTracker(required_stable_observations=2)
    observations = []
    for state in store.iter_records(StateSnapshot):
        frame = state.frames[0]
        assert frame.elements_artifact is not None
        elements = FrameElementCollection.model_validate_json(
            store.read_bytes(frame.elements_artifact)
        ).elements
        observations.append(
            tracker.observe(
                CapturedState(
                    state=state,
                    elements=elements,
                    receipt=cast(AppendReceipt, object()),
                )
            )
        )

    assert observations[0].declared_test_cases == 2
    assert observations[0].test_cases_discovered == 2
    assert observations[0].descriptions_captured == 0
    assert observations[1].descriptions_captured == 1
    assert observations[2].context_complete is True
    assert observations[2].stable_observations == 1
    assert observations[2].stable_for_early_stop is False

    stable = tracker.observe(
        CapturedState(
            state=tuple(store.iter_records(StateSnapshot))[-1],
            elements=elements,
            receipt=cast(AppendReceipt, object()),
        )
    )
    knowledge = PortalKnowledgeBuilder(store).build()

    assert stable.stable_for_early_stop is True
    assert stable.declared_test_cases == knowledge.coverage.declared_test_cases
    assert stable.test_cases_discovered == knowledge.coverage.test_cases_normalized
    assert stable.descriptions_captured == knowledge.coverage.descriptions_captured


def test_tracker_recognizes_visual_modal_and_pagination_total(tmp_path: Path) -> None:
    run = ScrapeRun(
        run_id="visual-modal-run",
        root_url="https://portal.example.test/cases?page=1",
        allowed_origins=("https://portal.example.test",),
    )
    store = ArtifactStore.create(tmp_path / "visual-crawls", run)
    headers = _headers(
        CASE_TABLE,
        ("#", "TC ID", "API Name", "Status"),
        "visual-case",
    )
    cells = _cells(
        CASE_TABLE,
        1,
        ("1", "TC_VISUAL_01", "Payments", "pending"),
        "visual-row",
    )
    controls = _case_controls(1, "TC_VISUAL_01")
    showing = _element(
        "showing-total",
        tag="div",
        text="Showing 1 to 1 of 1 entries",
        parent_css_path=ROOT,
    )
    base_elements = tuple([*headers, *cells, *controls, showing])
    base = _state(
        store,
        state_id="visual-base",
        sequence=0,
        url=run.root_url,
        elements=base_elements,
    )
    visual_modal_elements = (
        _element(
            "visual-modal",
            tag="section",
            text="TC_VISUAL_01: validates a CSS modal flow",
            parent_css_path=ROOT,
            inside_dialog=True,
        ),
    )
    first_modal = _state(
        store,
        state_id="visual-modal-1",
        sequence=1,
        url=run.root_url,
        elements=visual_modal_elements,
        modal_count=1,
    )
    second_modal = _state(
        store,
        state_id="visual-modal-2",
        sequence=2,
        url=run.root_url,
        elements=visual_modal_elements,
        modal_count=1,
    )
    tracker = ContextTracker(required_stable_observations=2)

    initial = tracker.observe(
        CapturedState(base, base_elements, cast(AppendReceipt, object()))
    )
    first = tracker.observe(
        CapturedState(
            first_modal,
            visual_modal_elements,
            cast(AppendReceipt, object()),
        )
    )
    stable = tracker.observe(
        CapturedState(
            second_modal,
            visual_modal_elements,
            cast(AppendReceipt, object()),
        )
    )

    assert initial.declared_test_cases == 1
    assert initial.test_cases_discovered == 1
    assert first.descriptions_captured == 1
    assert first.context_complete is True
    assert stable.stable_for_early_stop is True


def test_tracker_carries_only_the_selected_summary_row_total(tmp_path: Path) -> None:
    run = ScrapeRun(
        run_id="selected-total-run",
        root_url="https://portal.example.test/apis",
        allowed_origins=("https://portal.example.test",),
    )
    store = ArtifactStore.create(tmp_path / "selected-crawls", run)
    headers = _headers(
        SUMMARY_TABLE,
        ("API Name", "Total TCs", "Test"),
        "selected-summary",
    )
    first = _cells(
        SUMMARY_TABLE,
        1,
        ("Payments", "12", "▶"),
        "selected-first",
    )
    second = _cells(
        SUMMARY_TABLE,
        2,
        ("Refunds", "9", "▶"),
        "selected-second",
    )
    first_row = f"{SUMMARY_TABLE} > tbody > tr:nth-of-type(1)"
    control = _element(
        "open-payments",
        tag="a",
        role="link",
        text="▶",
        accessible_name="Test",
        parent_css_path=f"{first_row} > td:nth-of-type(3)",
        interactive=True,
        row_label="Payments",
    )
    elements = tuple([*headers, *first, control, *second])
    state = _state(
        store,
        state_id="selected-summary-state",
        sequence=0,
        url=run.root_url,
        elements=elements,
    )
    tracker = ContextTracker()

    coverage = tracker.observe_selected_total(
        CapturedState(state, elements, cast(AppendReceipt, object())),
        control,
    )

    assert coverage.declared_test_cases == 12


def test_completed_testcase_goal_is_a_normalizable_non_frontier_handoff(
    tmp_path: Path,
) -> None:
    store = _build_store(tmp_path, completed=True)
    store.append_record(
        CoverageReport(
            run_id=store.run.run_id,
            generated_at=NOW + timedelta(seconds=4),
            bounded_complete=False,
            completion_goal=CrawlCompletionGoal.TESTCASE_CONTEXT,
            goal_complete=True,
            testcase_context=ContextCoverage(
                declared_test_cases=2,
                test_cases_discovered=2,
                descriptions_captured=2,
                stable_observations=2,
                required_stable_observations=2,
                context_complete=True,
            ),
            completion_reason="testcase context complete",
        )
    )
    store.save_manifest(
        store.run.model_copy(update={"completion_reason": "testcase context complete"})
    )

    knowledge = PortalKnowledgeBuilder(store).build()

    assert knowledge.coverage.source_status == ScrapeRunStatus.COMPLETED
    assert knowledge.coverage.source_bounded_complete is False
    assert knowledge.coverage.testcase_context_complete is True


def test_knowledge_export_is_deterministic_and_requires_explicit_overwrite(
    tmp_path: Path,
) -> None:
    knowledge = PortalKnowledgeBuilder(_build_store(tmp_path, completed=True)).build()
    output_directory = tmp_path / "knowledge"

    first = export_knowledge(knowledge, output_directory)

    assert first.knowledge_path.is_file()
    assert first.testcases_path.read_text().count("\n") == 2
    for item in first.manifest.files:
        data = (output_directory / item.name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item.sha256
        assert len(data) == item.byte_size

    with pytest.raises(KnowledgeExportError, match="already exists"):
        export_knowledge(knowledge, output_directory)

    before = first.knowledge_path.read_bytes()
    second = export_knowledge(knowledge, output_directory, overwrite=True)
    assert second.knowledge_path.read_bytes() == before


def test_identifier_matching_does_not_confuse_prefixed_testcase_ids() -> None:
    assert _contains_identifier("TC_01: exact details", "TC_01") is True
    assert _contains_identifier("BillerTC_01: different case", "TC_01") is False
