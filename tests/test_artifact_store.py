"""Persistence, recovery, and integrity tests for the Phase 2 artifact store."""

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scraper.artifact_store import (
    ArtifactStore,
    ArtifactStoreError,
    RecordStream,
    StoreAlreadyExistsError,
    StoreIntegrityError,
    StoreNotFoundError,
)
from scraper.models import (
    ActionCandidate,
    ActionKind,
    ActionRisk,
    ActionStatus,
    ArtifactKind,
    BrowserEvent,
    BrowserEventKind,
    CoverageReport,
    InteractionTransition,
    NetworkExchange,
    ScrapeRun,
    StateSnapshot,
    Viewport,
)


NOW = datetime(2026, 8, 6, 8, 0, tzinfo=timezone.utc)


def make_run(run_id: str = "run-1") -> ScrapeRun:
    """Create a minimal valid run manifest."""

    return ScrapeRun(
        run_id=run_id,
        root_url="https://portal.example.test/start",
        allowed_origins=("https://portal.example.test",),
    )


def make_state(sequence: int = 0, run_id: str = "run-1") -> StateSnapshot:
    """Create a minimal state record for stream tests."""

    return StateSnapshot(
        state_id=f"state-{sequence}",
        run_id=run_id,
        sequence=sequence,
        fingerprint=f"{sequence:064x}",
        captured_at=NOW,
        page_id="page-1",
        url=f"https://portal.example.test/state/{sequence}",
        viewport=Viewport(width=1280, height=720),
    )


def test_store_creation_is_exclusive_and_manifest_is_reopenable(tmp_path: Path) -> None:
    base = tmp_path / "crawls"
    run = make_run()

    store = ArtifactStore.create(base, run)

    assert store.run == run
    assert (store.run_directory / "manifest.json").is_file()
    assert (store.run_directory / "blobs").is_dir()

    reopened = ArtifactStore.open(base, run.run_id)
    assert reopened.run == run

    with pytest.raises(StoreAlreadyExistsError):
        ArtifactStore.create(base, run)


def test_blobs_are_content_addressed_deduplicated_and_verified(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "crawls", make_run())

    first = store.put_text(
        ArtifactKind.DOM,
        "<html>stable</html>",
        media_type="text/html",
    )
    second = store.put_text(
        ArtifactKind.DOM,
        "<html>stable</html>",
        media_type="text/html",
    )

    assert first.sha256 == second.sha256
    assert first.relative_path == second.relative_path
    assert store.read_text(first) == "<html>stable</html>"

    blob_files = [
        path for path in (store.run_directory / "blobs").rglob("*") if path.is_file()
    ]
    assert len(blob_files) == 1

    report = store.verify_integrity(strict=True)
    assert report.valid is True
    assert report.blob_count == 1


def test_canonical_json_deduplicates_equivalent_objects(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "crawls", make_run())

    first = store.put_json(ArtifactKind.RESPONSE_BODY, {"b": 2, "a": 1})
    second = store.put_json(ArtifactKind.RESPONSE_BODY, {"a": 1, "b": 2})

    assert first.sha256 == second.sha256
    assert store.read_bytes(first) == b'{"a":1,"b":2}'


def test_records_append_with_monotonic_sequences_and_resume(tmp_path: Path) -> None:
    base = tmp_path / "crawls"
    store = ArtifactStore.create(base, make_run())

    first = store.append_record(make_state(0))
    second = store.append_record(make_state(1))

    assert first.stream == RecordStream.STATES
    assert first.sequence == 0
    assert second.sequence == 1
    assert second.byte_offset == first.byte_offset + first.byte_length
    assert list(store.iter_records(StateSnapshot)) == [make_state(0), make_state(1)]

    reopened = ArtifactStore.open(base, "run-1")
    third = reopened.append_record(make_state(2))

    assert third.sequence == 2
    assert list(reopened.iter_records(StateSnapshot)) == [
        make_state(0),
        make_state(1),
        make_state(2),
    ]


def test_in_process_concurrent_appends_do_not_interleave(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "crawls", make_run())

    with ThreadPoolExecutor(max_workers=4) as executor:
        receipts = list(executor.map(store.append_record, map(make_state, range(20))))

    assert sorted(receipt.sequence for receipt in receipts) == list(range(20))
    assert len(list(store.iter_records(StateSnapshot))) == 20


def test_records_from_another_run_are_rejected(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "crawls", make_run())

    with pytest.raises(ArtifactStoreError, match="run_id"):
        store.append_record(make_state(0, run_id="another-run"))


def test_every_evidence_model_uses_its_own_typed_stream(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "crawls", make_run())
    records = [
        ActionCandidate(
            action_id="action-1",
            run_id="run-1",
            state_id="state-0",
            element_id="button-1",
            kind=ActionKind.CLICK,
            risk=ActionRisk.SAFE,
            policy_rule="read-only navigation",
            rationale="opens a details panel",
            discovered_at=NOW,
        ),
        InteractionTransition(
            transition_id="transition-1",
            run_id="run-1",
            action_id="action-1",
            parent_state_id="state-0",
            resulting_state_id="state-1",
            status=ActionStatus.SUCCEEDED,
            started_at=NOW,
            completed_at=NOW,
        ),
        BrowserEvent(
            event_id="event-1",
            run_id="run-1",
            kind=BrowserEventKind.LOAD,
            observed_at=NOW,
        ),
        NetworkExchange(
            exchange_id="exchange-1",
            run_id="run-1",
            request_id="request-1",
            requested_at=NOW,
            url="https://portal.example.test/api/details",
            method="GET",
        ),
        CoverageReport(
            run_id="run-1",
            generated_at=NOW,
            completion_reason="crawl still active",
        ),
    ]
    expected_streams = [
        RecordStream.ACTIONS,
        RecordStream.TRANSITIONS,
        RecordStream.EVENTS,
        RecordStream.NETWORK,
        RecordStream.COVERAGE,
    ]

    for record, expected_stream in zip(records, expected_streams):
        receipt = store.append_record(record)
        assert receipt.stream == expected_stream
        assert list(store.iter_records(type(record))) == [record]


def test_checkpoint_captures_stream_boundaries_and_updates_manifest(
    tmp_path: Path,
) -> None:
    base = tmp_path / "crawls"
    store = ArtifactStore.create(base, make_run())
    store.append_record(make_state(0))

    checkpoint = store.write_checkpoint(1, last_state_id="state-0")

    state_position = next(
        position
        for position in checkpoint.streams
        if position.stream == RecordStream.STATES
    )
    assert state_position.record_count == 1
    assert state_position.byte_length > 0
    assert checkpoint.last_state_id == "state-0"
    assert store.reload_manifest().checkpoint_sequence == 1
    assert store.load_checkpoint() == checkpoint

    reopened = ArtifactStore.open(base, "run-1")
    assert reopened.load_checkpoint() == checkpoint

    with pytest.raises(ArtifactStoreError, match="must increase"):
        reopened.write_checkpoint(1)


def test_manifest_changes_require_a_new_checkpoint_for_integrity(
    tmp_path: Path,
) -> None:
    store = ArtifactStore.create(tmp_path / "crawls", make_run())
    store.append_record(make_state(0))
    store.write_checkpoint(1, last_state_id="state-0")

    store.save_manifest(store.run.model_copy(update={"state_ids": ("state-0",)}))

    stale_report = store.verify_integrity()
    assert stale_report.valid is False
    assert "checkpoint does not reference" in stale_report.issues[0]

    store.write_checkpoint(2, last_state_id="state-0")
    assert store.verify_integrity(strict=True).valid is True


def test_open_can_repair_only_a_partial_trailing_record(tmp_path: Path) -> None:
    base = tmp_path / "crawls"
    store = ArtifactStore.create(base, make_run())
    store.append_record(make_state(0))
    stream_path = store.run_directory / "states.jsonl"

    with stream_path.open("ab") as handle:
        handle.write(b'{"partial"')

    with pytest.raises(StoreIntegrityError, match="partial trailing record"):
        ArtifactStore.open(base, "run-1")

    repaired = ArtifactStore.open(base, "run-1", repair_trailing_partial=True)

    assert list(repaired.iter_records(StateSnapshot)) == [make_state(0)]
    assert stream_path.read_bytes().endswith(b"\n")
    assert repaired.append_record(make_state(1)).sequence == 1


def test_blob_tampering_is_reported_and_strict_mode_raises(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "crawls", make_run())
    reference = store.put_bytes(
        ArtifactKind.SCREENSHOT,
        b"original-image",
        media_type="image/png",
    )
    blob_path = store.run_directory / reference.relative_path
    blob_path.write_bytes(b"tampered")

    with pytest.raises(StoreIntegrityError, match="integrity check failed"):
        store.read_bytes(reference)

    report = store.verify_integrity()
    assert report.valid is False
    assert any("blob hash" in issue for issue in report.issues)

    with pytest.raises(StoreIntegrityError, match="blob hash"):
        store.verify_integrity(strict=True)


def test_missing_artifact_referenced_by_state_is_reported(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "crawls", make_run())
    reference = store.put_text(
        ArtifactKind.DOM,
        "<html></html>",
        media_type="text/html",
    )
    state = make_state(0).model_copy(update={"artifacts": (reference,)})
    store.append_record(state)
    (store.run_directory / reference.relative_path).unlink()

    report = store.verify_integrity()

    assert report.valid is False
    assert any("invalid referenced artifact" in issue for issue in report.issues)
    with pytest.raises(StoreIntegrityError, match="invalid referenced artifact"):
        store.verify_integrity(strict=True)


def test_artifact_references_cannot_read_store_metadata(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "crawls", make_run())
    reference = store.put_text(ArtifactKind.DOM, "safe", media_type="text/plain")
    forged = reference.model_copy(update={"relative_path": "manifest.json"})

    with pytest.raises(ArtifactStoreError, match="content-addressed layout"):
        store.read_bytes(forged)


def test_run_paths_cannot_escape_the_storage_root(tmp_path: Path) -> None:
    base = tmp_path / "crawls"

    with pytest.raises(ArtifactStoreError, match="not safe"):
        ArtifactStore.create(base, make_run("../escape"))

    with pytest.raises(ArtifactStoreError, match="not safe"):
        ArtifactStore.create(base, make_run("run with spaces"))

    with pytest.raises(StoreNotFoundError):
        ArtifactStore.open(base, "missing-run")


def test_corrupt_payload_is_detected_during_reopen(tmp_path: Path) -> None:
    base = tmp_path / "crawls"
    store = ArtifactStore.create(base, make_run())
    store.append_record(make_state(0))
    stream_path = store.run_directory / "states.jsonl"
    line = stream_path.read_text(encoding="utf-8")
    stream_path.write_text(
        line.replace('"page_id":"page-1"', '"page_id":null'),
        encoding="utf-8",
    )

    with pytest.raises(StoreIntegrityError, match="invalid StateSnapshot"):
        ArtifactStore.open(base, "run-1")
