"""Focused tests for the Phase 1 scraper evidence contract."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from scraper.models import (
    ActionStatus,
    ArtifactKind,
    ArtifactReference,
    CoverageReport,
    ElementSnapshot,
    FrameElementCollection,
    FrameSnapshot,
    InteractionTransition,
    ScrapeRun,
    ScrapeRunStatus,
    ScrollPosition,
    StateSnapshot,
    ValueCapture,
    Viewport,
)


SHA256 = "a" * 64
NOW = datetime(2026, 8, 6, 7, 0, tzinfo=timezone.utc)


def make_artifact() -> ArtifactReference:
    """Create a valid content-addressed artifact reference."""

    return ArtifactReference(
        artifact_id="dom-1",
        kind=ArtifactKind.DOM,
        relative_path="dom/aa/page.html.gz",
        sha256=SHA256,
        media_type="text/html",
        byte_size=128,
        captured_at=NOW,
    )


def test_models_are_frozen_and_forbid_unknown_fields() -> None:
    run = ScrapeRun(
        run_id="run-1",
        root_url="https://portal.example.test/start",
        allowed_origins=("https://portal.example.test",),
    )

    with pytest.raises(ValidationError, match="frozen"):
        run.status = ScrapeRunStatus.RUNNING  # type: ignore[misc]

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ScrapeRun.model_validate(
            {
                "run_id": "run-1",
                "root_url": "https://portal.example.test",
                "allowed_origins": ["https://portal.example.test"],
                "unexpected": True,
            }
        )


def test_sensitive_values_and_artifact_paths_are_guarded() -> None:
    with pytest.raises(ValidationError, match="must not contain plaintext"):
        ValueCapture(name="authorization", value="Bearer secret", redacted=True)

    redacted = ValueCapture(
        name="authorization",
        value_hash=SHA256,
        redacted=True,
    )
    assert redacted.value is None

    safe_url = ValueCapture(
        name="href",
        safe_value="/next?token=[REDACTED]",
        value_hash=SHA256,
        redacted=True,
    )
    assert safe_url.safe_value == "/next?token=[REDACTED]"

    with pytest.raises(ValidationError, match="only valid for redacted"):
        ValueCapture(name="href", safe_value="/unsafe", redacted=False)

    with pytest.raises(ValidationError, match="explicit redaction marker"):
        ValueCapture(
            name="href",
            safe_value="/still-plaintext",
            value_hash=SHA256,
            redacted=True,
        )

    with pytest.raises(ValidationError, match="must be relative"):
        ArtifactReference(
            artifact_id="escape",
            kind=ArtifactKind.DOM,
            relative_path="../outside.html",
            sha256=SHA256,
            media_type="text/html",
            byte_size=1,
        )

    with pytest.raises(ValidationError, match="POSIX separators"):
        ArtifactReference(
            artifact_id="windows-escape",
            kind=ArtifactKind.DOM,
            relative_path="..\\outside.html",
            sha256=SHA256,
            media_type="text/html",
            byte_size=1,
        )


def test_state_snapshot_round_trips_without_losing_types() -> None:
    artifact = make_artifact()
    frame = FrameSnapshot(
        frame_id="frame-main",
        url="https://portal.example.test/start",
        is_main=True,
        element_ids=("button-1",),
        dom_artifact=artifact,
    )
    state = StateSnapshot(
        state_id="state-1",
        run_id="run-1",
        sequence=0,
        fingerprint=SHA256,
        captured_at=NOW,
        page_id="page-1",
        url="https://portal.example.test/start",
        title="Portal",
        viewport=Viewport(width=1920, height=1080),
        scroll=ScrollPosition(maximum_y=2000, y=200),
        frames=(frame,),
        element_ids=("button-1",),
        active_element_id="button-1",
        artifacts=(artifact,),
    )

    restored = StateSnapshot.model_validate_json(state.model_dump_json())

    assert restored == state
    assert isinstance(restored.frames, tuple)
    assert restored.frames[0].dom_artifact == artifact


def test_state_snapshot_rejects_dangling_active_element() -> None:
    with pytest.raises(ValidationError, match="active_element_id"):
        StateSnapshot(
            state_id="state-1",
            run_id="run-1",
            sequence=0,
            fingerprint=SHA256,
            page_id="page-1",
            url="https://portal.example.test",
            viewport=Viewport(width=1280, height=720),
            element_ids=("button-1",),
            active_element_id="missing",
        )


def test_frame_element_collection_guards_frame_and_element_identity() -> None:
    element = ElementSnapshot(
        element_id="button-1",
        frame_id="frame-main",
        tag="button",
    )
    collection = FrameElementCollection(
        frame_id="frame-main",
        elements=(element,),
    )

    assert collection.elements == (element,)

    with pytest.raises(ValidationError, match="collection frame_id"):
        FrameElementCollection(frame_id="other-frame", elements=(element,))

    with pytest.raises(ValidationError, match="must be unique"):
        FrameElementCollection(
            frame_id="frame-main",
            elements=(element, element),
        )


@pytest.mark.parametrize(
    ("status", "kwargs", "message"),
    [
        (ActionStatus.SUCCEEDED, {}, "resulting_state_id"),
        (ActionStatus.FAILED, {}, "error_message"),
        (ActionStatus.SKIPPED, {}, "skip_reason"),
    ],
)
def test_transition_requires_status_specific_evidence(
    status: ActionStatus,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        InteractionTransition(
            transition_id="transition-1",
            run_id="run-1",
            action_id="action-1",
            parent_state_id="state-1",
            status=status,
            started_at=NOW,
            completed_at=NOW + timedelta(seconds=1),
            **kwargs,
        )


def test_successful_transition_captures_a_complete_causal_edge() -> None:
    transition = InteractionTransition(
        transition_id="transition-1",
        run_id="run-1",
        action_id="action-1",
        parent_state_id="state-1",
        resulting_state_id="state-2",
        status=ActionStatus.SUCCEEDED,
        started_at=NOW,
        completed_at=NOW + timedelta(seconds=1),
        duration_ms=1_000,
        quiescence_reason="DOM and network quiet windows reached",
    )

    assert transition.resulting_state_id == "state-2"


def test_terminal_run_requires_auditable_completion() -> None:
    with pytest.raises(ValidationError, match="start and end timestamps"):
        ScrapeRun(
            run_id="run-1",
            root_url="https://portal.example.test",
            allowed_origins=("https://portal.example.test",),
            status=ScrapeRunStatus.COMPLETED,
        )

    completed = ScrapeRun(
        run_id="run-1",
        root_url="https://portal.example.test/start",
        allowed_origins=("https://portal.example.test/",),
        status=ScrapeRunStatus.COMPLETED,
        started_at=NOW,
        ended_at=NOW + timedelta(seconds=10),
        completion_reason="frontier exhausted",
        state_ids=("state-1",),
        transition_ids=("transition-1",),
    )

    assert completed.allowed_origins == ("https://portal.example.test",)

    with pytest.raises(ValidationError, match="root_url origin"):
        ScrapeRun(
            run_id="run-2",
            root_url="https://other.example.test/start",
            allowed_origins=("https://portal.example.test",),
        )


def test_urls_and_timestamps_are_normalized_without_leaking_credentials() -> None:
    india = timezone(timedelta(hours=5, minutes=30))
    run = ScrapeRun(
        run_id="run-1",
        root_url="https://PORTAL.Example.Test/start",
        allowed_origins=("https://portal.example.test/",),
        status=ScrapeRunStatus.RUNNING,
        started_at=datetime(2026, 8, 6, 12, 30, tzinfo=india),
    )

    assert run.allowed_origins == ("https://portal.example.test",)
    assert run.started_at == NOW
    assert run.started_at is not None
    assert run.started_at.tzinfo == timezone.utc

    with pytest.raises(ValidationError, match="embedded credentials"):
        ScrapeRun(
            run_id="run-secret",
            root_url="https://user:password@portal.example.test/start",
            allowed_origins=("https://portal.example.test",),
        )


def test_coverage_counts_must_reconcile() -> None:
    with pytest.raises(ValidationError, match="must equal action_candidates"):
        CoverageReport(
            run_id="run-1",
            action_candidates=2,
            actions_succeeded=1,
            completion_reason="still crawling",
        )

    coverage = CoverageReport(
        run_id="run-1",
        action_candidates=3,
        actions_succeeded=1,
        actions_failed=1,
        actions_skipped=1,
        bounded_complete=True,
        completion_reason="frontier exhausted",
    )

    assert coverage.bounded_complete is True
