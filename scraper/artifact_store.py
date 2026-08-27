"""Durable append-only storage for portal crawl evidence.

The store owns one directory per scrape run. Structured evidence is appended to
homogeneous JSONL streams, while large payloads are written as content-addressed
blobs. Mutable manifest and checkpoint files are replaced atomically and are
the only non-append-only files in a run directory.

The implementation supports one writer per :class:`ArtifactStore` instance.
Its in-process lock prevents thread interleaving; coordinating multiple writer
processes is intentionally outside the Phase 2 contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional, Type, TypeVar, Union, cast

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from scraper.models import (
    SCHEMA_VERSION,
    ActionCandidate,
    ArtifactKind,
    ArtifactReference,
    BrowserEvent,
    CoverageReport,
    EvidenceModel,
    InteractionTransition,
    NetworkExchange,
    ScrapeRun,
    StateSnapshot,
    utc_now,
)


RUN_DIRECTORY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ArtifactStoreError(RuntimeError):
    """Base error raised by the artifact store."""


class StoreAlreadyExistsError(ArtifactStoreError):
    """Raised when creation would overwrite an existing run."""


class StoreNotFoundError(ArtifactStoreError):
    """Raised when a run directory or required file does not exist."""


class StoreIntegrityError(ArtifactStoreError):
    """Raised when persisted evidence fails validation or hashing."""


class RecordStream(str, Enum):
    """Homogeneous append-only record streams stored for each run."""

    STATES = "states"
    ACTIONS = "actions"
    TRANSITIONS = "transitions"
    EVENTS = "events"
    NETWORK = "network"
    COVERAGE = "coverage"


StorableRecord = Union[
    StateSnapshot,
    ActionCandidate,
    InteractionTransition,
    BrowserEvent,
    NetworkExchange,
    CoverageReport,
]
RecordModel = TypeVar("RecordModel", bound=BaseModel)


class AppendReceipt(EvidenceModel):
    """Durable position of one appended JSONL evidence record."""

    stream: RecordStream
    sequence: int = Field(ge=0)
    byte_offset: int = Field(ge=0)
    byte_length: int = Field(gt=0)
    record_type: str = Field(min_length=1)
    recorded_at: datetime = Field(default_factory=utc_now)

    @field_validator("recorded_at")
    @classmethod
    def validate_recorded_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("recorded_at must include timezone information")
        return value.astimezone(timezone.utc)


class StreamPosition(EvidenceModel):
    """Count and byte boundary for a record stream at checkpoint time."""

    stream: RecordStream
    record_count: int = Field(ge=0)
    byte_length: int = Field(ge=0)


class StoreCheckpoint(EvidenceModel):
    """Atomic recovery pointer for one artifact store."""

    schema_version: str = SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    streams: tuple[StreamPosition, ...]
    last_state_id: Optional[str] = None
    last_transition_id: Optional[str] = None

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include timezone information")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_streams(self) -> "StoreCheckpoint":
        streams = [position.stream for position in self.streams]
        if len(streams) != len(set(streams)):
            raise ValueError("checkpoint streams must be unique")
        if set(streams) != set(RecordStream):
            raise ValueError("checkpoint must describe every record stream")
        return self


class StoreIntegrityReport(EvidenceModel):
    """Result of validating manifests, streams, and stored blobs."""

    run_id: str = Field(min_length=1)
    checked_at: datetime = Field(default_factory=utc_now)
    streams: tuple[StreamPosition, ...]
    blob_count: int = Field(ge=0)
    blob_bytes: int = Field(ge=0)
    valid: bool
    issues: tuple[str, ...] = ()

    @field_validator("checked_at")
    @classmethod
    def validate_checked_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("checked_at must include timezone information")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_validity(self) -> "StoreIntegrityReport":
        streams = [position.stream for position in self.streams]
        if len(streams) != len(set(streams)):
            raise ValueError("integrity report streams must be unique")
        if set(streams) != set(RecordStream):
            raise ValueError("integrity report must describe every record stream")
        if self.valid == bool(self.issues):
            raise ValueError("valid must be true exactly when issues is empty")
        return self


_STREAM_FILES: dict[RecordStream, str] = {
    RecordStream.STATES: "states.jsonl",
    RecordStream.ACTIONS: "actions.jsonl",
    RecordStream.TRANSITIONS: "transitions.jsonl",
    RecordStream.EVENTS: "events.jsonl",
    RecordStream.NETWORK: "network.jsonl",
    RecordStream.COVERAGE: "coverage.jsonl",
}

_MODEL_STREAMS: dict[type[BaseModel], RecordStream] = {
    StateSnapshot: RecordStream.STATES,
    ActionCandidate: RecordStream.ACTIONS,
    InteractionTransition: RecordStream.TRANSITIONS,
    BrowserEvent: RecordStream.EVENTS,
    NetworkExchange: RecordStream.NETWORK,
    CoverageReport: RecordStream.COVERAGE,
}


class ArtifactStore:
    """Content-addressed blob store and append-only evidence journal."""

    MANIFEST_FILE = "manifest.json"
    CHECKPOINT_FILE = "checkpoint.json"
    BLOBS_DIRECTORY = "blobs"

    def __init__(self, run_directory: Path, run: ScrapeRun):
        self._run_directory = run_directory
        self._run = run
        self._lock = threading.RLock()
        self._events_by_action: dict[str, list[BrowserEvent]] = {}
        self._exchanges_by_action: dict[str, list[NetworkExchange]] = {}
        self._stream_counts: dict[RecordStream, int] = {
            stream: 0 for stream in RecordStream
        }
        self._stream_bytes: dict[RecordStream, int] = {
            stream: 0 for stream in RecordStream
        }

    @property
    def run(self) -> ScrapeRun:
        """Return the most recently saved run manifest."""

        return self._run

    @property
    def run_directory(self) -> Path:
        """Return the absolute directory containing this run's evidence."""

        return self._run_directory

    @classmethod
    def create(cls, base_directory: Path, run: ScrapeRun) -> "ArtifactStore":
        """Create a new run directory without overwriting existing evidence."""

        base = base_directory.expanduser().resolve()
        run_directory = cls._resolve_run_directory(base, run.run_id)
        if run_directory.exists():
            raise StoreAlreadyExistsError(f"scrape run already exists: {run.run_id}")

        try:
            run_directory.mkdir(parents=True, mode=0o700)
            (run_directory / cls.BLOBS_DIRECTORY).mkdir(mode=0o700)
        except OSError as exc:
            raise ArtifactStoreError(f"failed to create run directory: {exc}") from exc

        store = cls(run_directory=run_directory, run=run)
        store.save_manifest(run)
        return store

    @classmethod
    def open(
        cls,
        base_directory: Path,
        run_id: str,
        *,
        repair_trailing_partial: bool = False,
    ) -> "ArtifactStore":
        """Open an existing run and validate every append-only stream."""

        base = base_directory.expanduser().resolve()
        run_directory = cls._resolve_run_directory(base, run_id)
        if not run_directory.is_dir():
            raise StoreNotFoundError(f"scrape run does not exist: {run_id}")

        manifest_path = run_directory / cls.MANIFEST_FILE
        run = cls._read_model_file(manifest_path, ScrapeRun)
        if run.run_id != run_id:
            raise StoreIntegrityError(
                f"manifest run_id {run.run_id!r} does not match directory {run_id!r}"
            )

        store = cls(run_directory=run_directory, run=run)
        for stream in RecordStream:
            count, byte_length = store._scan_stream(
                stream,
                repair_trailing_partial=repair_trailing_partial,
            )
            store._stream_counts[stream] = count
            store._stream_bytes[stream] = byte_length
        store._rebuild_action_indexes()
        return store

    def save_manifest(self, run: ScrapeRun) -> None:
        """Atomically replace the current manifest snapshot."""

        if run.run_id != self._run.run_id:
            raise ArtifactStoreError("cannot save a manifest for another run")
        if run.checkpoint_sequence < self._run.checkpoint_sequence:
            raise ArtifactStoreError(
                "manifest checkpoint_sequence cannot move backwards"
            )

        payload = self._model_json_bytes(run, pretty=True)
        with self._lock:
            self._atomic_write(self._run_directory / self.MANIFEST_FILE, payload)
            self._run = run

    def reload_manifest(self) -> ScrapeRun:
        """Reload and validate the manifest from disk."""

        with self._lock:
            run = self._read_model_file(
                self._run_directory / self.MANIFEST_FILE,
                ScrapeRun,
            )
            if run.run_id != self._run.run_id:
                raise StoreIntegrityError("manifest run_id changed unexpectedly")
            self._run = run
            return run

    def put_bytes(
        self,
        kind: ArtifactKind,
        data: bytes,
        *,
        media_type: str,
        redacted: bool = False,
        captured_at: Optional[datetime] = None,
    ) -> ArtifactReference:
        """Persist bytes once and return a content-addressed reference."""

        if not media_type.strip():
            raise ValueError("media_type cannot be empty")
        digest = hashlib.sha256(data).hexdigest()
        relative_path = Path(
            self.BLOBS_DIRECTORY,
            kind.value,
            digest[:2],
            digest,
        )
        destination = self._resolve_relative_path(relative_path.as_posix())

        with self._lock:
            if destination.exists():
                self._verify_blob_file(destination, digest, len(data))
            else:
                destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
                self._atomic_write(destination, data)

        return ArtifactReference(
            artifact_id=f"{kind.value}:{digest}",
            kind=kind,
            relative_path=relative_path.as_posix(),
            sha256=digest,
            media_type=media_type.strip(),
            byte_size=len(data),
            redacted=redacted,
            captured_at=captured_at or utc_now(),
        )

    def put_text(
        self,
        kind: ArtifactKind,
        text: str,
        *,
        media_type: str = "text/plain",
        encoding: str = "utf-8",
        redacted: bool = False,
        captured_at: Optional[datetime] = None,
    ) -> ArtifactReference:
        """Encode and store text as a content-addressed artifact."""

        return self.put_bytes(
            kind,
            text.encode(encoding),
            media_type=media_type,
            redacted=redacted,
            captured_at=captured_at,
        )

    def put_json(
        self,
        kind: ArtifactKind,
        value: object,
        *,
        redacted: bool = False,
        captured_at: Optional[datetime] = None,
    ) -> ArtifactReference:
        """Serialize canonical JSON and store it as a blob."""

        try:
            data = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"value is not valid canonical JSON: {exc}") from exc
        return self.put_bytes(
            kind,
            data,
            media_type="application/json",
            redacted=redacted,
            captured_at=captured_at,
        )

    def put_model(
        self,
        kind: ArtifactKind,
        model: BaseModel,
        *,
        redacted: bool = False,
        captured_at: Optional[datetime] = None,
    ) -> ArtifactReference:
        """Store a Pydantic model using canonical JSON serialization."""

        return self.put_json(
            kind,
            model.model_dump(mode="json"),
            redacted=redacted,
            captured_at=captured_at,
        )

    def read_bytes(self, reference: ArtifactReference, *, verify: bool = True) -> bytes:
        """Read a referenced artifact and optionally verify its integrity."""

        path = self._resolve_artifact_reference(reference)
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise StoreNotFoundError(
                f"artifact does not exist: {reference.relative_path}"
            ) from exc
        except OSError as exc:
            raise ArtifactStoreError(f"failed to read artifact: {exc}") from exc

        if verify:
            digest = hashlib.sha256(data).hexdigest()
            if digest != reference.sha256 or len(data) != reference.byte_size:
                raise StoreIntegrityError(
                    f"artifact integrity check failed: {reference.relative_path}"
                )
        return data

    def read_text(
        self,
        reference: ArtifactReference,
        *,
        encoding: str = "utf-8",
        verify: bool = True,
    ) -> str:
        """Read and decode a referenced text artifact."""

        return self.read_bytes(reference, verify=verify).decode(encoding)

    def append_record(self, record: StorableRecord) -> AppendReceipt:
        """Append one validated evidence model to its homogeneous stream."""

        stream = self._stream_for_model(type(record))
        record_run_id = getattr(record, "run_id", None)
        if record_run_id != self._run.run_id:
            raise ArtifactStoreError("record run_id does not match this store")

        recorded_at = utc_now()
        with self._lock:
            sequence = self._stream_counts[stream]
            envelope = {
                "schema_version": SCHEMA_VERSION,
                "stream": stream.value,
                "sequence": sequence,
                "record_type": type(record).__name__,
                "recorded_at": recorded_at.isoformat(),
                "payload": record.model_dump(mode="json"),
            }
            data = self._canonical_json_bytes(envelope) + b"\n"
            path = self._stream_path(stream)
            try:
                descriptor = os.open(
                    path,
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                    0o600,
                )
                with os.fdopen(descriptor, "ab") as handle:
                    byte_offset = handle.tell()
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise ArtifactStoreError(
                    f"failed to append {stream.value}: {exc}"
                ) from exc

            self._stream_counts[stream] = sequence + 1
            self._stream_bytes[stream] = byte_offset + len(data)
            self._index_action_record(record)

        return AppendReceipt(
            stream=stream,
            sequence=sequence,
            byte_offset=byte_offset,
            byte_length=len(data),
            record_type=type(record).__name__,
            recorded_at=recorded_at,
        )

    def browser_events_for_action(self, action_id: str) -> tuple[BrowserEvent, ...]:
        """Return already-durable browser events correlated to one action."""

        with self._lock:
            return tuple(self._events_by_action.get(action_id, ()))

    def network_exchanges_for_action(
        self,
        action_id: str,
    ) -> tuple[NetworkExchange, ...]:
        """Return already-durable network exchanges correlated to one action."""

        with self._lock:
            return tuple(self._exchanges_by_action.get(action_id, ()))

    def _index_action_record(self, record: StorableRecord) -> None:
        if isinstance(record, BrowserEvent) and record.action_id is not None:
            self._events_by_action.setdefault(record.action_id, []).append(record)
        elif isinstance(record, NetworkExchange) and record.action_id is not None:
            self._exchanges_by_action.setdefault(record.action_id, []).append(record)

    def _rebuild_action_indexes(self) -> None:
        for event in self.iter_records(BrowserEvent):
            self._index_action_record(event)
        for exchange in self.iter_records(NetworkExchange):
            self._index_action_record(exchange)

    def iter_records(self, model_type: Type[RecordModel]) -> Iterator[RecordModel]:
        """Yield a validated snapshot of records for the requested model type."""

        stream = self._stream_for_model(model_type)
        path = self._stream_path(stream)
        if not path.exists():
            return

        expected_type = model_type.__name__
        try:
            size_limit = path.stat().st_size
            with path.open("rb") as handle:
                sequence = 0
                while handle.tell() < size_limit:
                    line = handle.readline()
                    if not line.endswith(b"\n"):
                        raise StoreIntegrityError(
                            f"partial trailing record in {stream.value} at "
                            f"byte {handle.tell() - len(line)}"
                        )
                    envelope = self._decode_envelope(
                        line,
                        stream=stream,
                        expected_sequence=sequence,
                        expected_record_type=expected_type,
                    )
                    try:
                        yield model_type.model_validate(envelope["payload"])
                    except ValidationError as exc:
                        raise StoreIntegrityError(
                            f"invalid {expected_type} at {stream.value}:{sequence}: {exc}"
                        ) from exc
                    sequence += 1
        except OSError as exc:
            raise ArtifactStoreError(f"failed to read {stream.value}: {exc}") from exc

    def write_checkpoint(
        self,
        sequence: int,
        *,
        last_state_id: Optional[str] = None,
        last_transition_id: Optional[str] = None,
    ) -> StoreCheckpoint:
        """Atomically save current stream boundaries for crash recovery."""

        if sequence < 0:
            raise ValueError("checkpoint sequence cannot be negative")

        with self._lock:
            previous = self.load_checkpoint(required=False)
            if previous is not None and sequence <= previous.sequence:
                raise ArtifactStoreError("checkpoint sequence must increase")
            if sequence < self._run.checkpoint_sequence:
                raise ArtifactStoreError("checkpoint sequence cannot move backwards")
            if sequence > self._run.checkpoint_sequence:
                self.save_manifest(
                    self._run.model_copy(update={"checkpoint_sequence": sequence})
                )
            manifest_data = self._manifest_path().read_bytes()
            positions = self._current_stream_positions()
            checkpoint = StoreCheckpoint(
                run_id=self._run.run_id,
                sequence=sequence,
                manifest_sha256=hashlib.sha256(manifest_data).hexdigest(),
                streams=positions,
                last_state_id=last_state_id,
                last_transition_id=last_transition_id,
            )
            self._atomic_write(
                self._run_directory / self.CHECKPOINT_FILE,
                self._model_json_bytes(checkpoint, pretty=True),
            )
            return checkpoint

    def load_checkpoint(self, *, required: bool = True) -> Optional[StoreCheckpoint]:
        """Load the latest checkpoint, if one has been written."""

        path = self._run_directory / self.CHECKPOINT_FILE
        if not path.exists():
            if required:
                raise StoreNotFoundError("checkpoint does not exist")
            return None
        checkpoint = self._read_model_file(path, StoreCheckpoint)
        if checkpoint.run_id != self._run.run_id:
            raise StoreIntegrityError("checkpoint run_id does not match manifest")
        return checkpoint

    def verify_integrity(self, *, strict: bool = False) -> StoreIntegrityReport:
        """Validate persisted streams and every content-addressed blob."""

        issues: list[str] = []
        positions: list[StreamPosition] = []

        try:
            self.reload_manifest()
        except ArtifactStoreError as exc:
            issues.append(str(exc))

        for stream in RecordStream:
            try:
                count, byte_length = self._scan_stream(
                    stream,
                    repair_trailing_partial=False,
                )
                positions.append(
                    StreamPosition(
                        stream=stream,
                        record_count=count,
                        byte_length=byte_length,
                    )
                )
            except ArtifactStoreError as exc:
                issues.append(str(exc))
                positions.append(
                    StreamPosition(
                        stream=stream,
                        record_count=self._stream_counts[stream],
                        byte_length=self._stream_bytes[stream],
                    )
                )

        checkpoint = None
        try:
            checkpoint = self.load_checkpoint(required=False)
        except ArtifactStoreError as exc:
            issues.append(str(exc))
        if checkpoint is not None:
            try:
                manifest_data = self._manifest_path().read_bytes()
                manifest_sha256 = hashlib.sha256(manifest_data).hexdigest()
                if checkpoint.manifest_sha256 != manifest_sha256:
                    issues.append("checkpoint does not reference the current manifest")
            except OSError as exc:
                issues.append(f"failed to hash manifest for checkpoint: {exc}")
            actual_positions = {position.stream: position for position in positions}
            for checkpoint_position in checkpoint.streams:
                actual = actual_positions.get(checkpoint_position.stream)
                if actual is None:
                    issues.append(
                        f"checkpoint references unknown stream: "
                        f"{checkpoint_position.stream.value}"
                    )
                    continue
                if (
                    checkpoint_position.record_count > actual.record_count
                    or checkpoint_position.byte_length > actual.byte_length
                ):
                    issues.append(
                        f"checkpoint exceeds durable {checkpoint_position.stream.value} data"
                    )

        blob_count = 0
        blob_bytes = 0
        blobs_root = self._run_directory / self.BLOBS_DIRECTORY
        if blobs_root.exists():
            for path in sorted(blobs_root.rglob("*")):
                if not path.is_file():
                    continue
                blob_count += 1
                if path.is_symlink():
                    issues.append(
                        f"blob must not be a symbolic link: "
                        f"{path.relative_to(self._run_directory).as_posix()}"
                    )
                    continue
                try:
                    data = path.read_bytes()
                    blob_bytes += len(data)
                    digest = hashlib.sha256(data).hexdigest()
                    relative = path.relative_to(self._run_directory)
                    parts = relative.parts
                    valid_kinds = {kind.value for kind in ArtifactKind}
                    if (
                        len(parts) != 4
                        or parts[0] != self.BLOBS_DIRECTORY
                        or parts[1] not in valid_kinds
                        or parts[2] != path.name[:2]
                    ):
                        issues.append(
                            f"invalid content-addressed blob path: {relative.as_posix()}"
                        )
                    if path.name != digest:
                        issues.append(
                            f"blob hash does not match filename: {relative.as_posix()}"
                        )
                except OSError as exc:
                    issues.append(f"failed to inspect blob {path.name}: {exc}")

        references: dict[tuple[str, str, int], ArtifactReference] = {}
        for model_type in _MODEL_STREAMS:
            try:
                for record in self.iter_records(model_type):
                    for reference in self._walk_artifact_references(record):
                        references[
                            (
                                reference.relative_path,
                                reference.sha256,
                                reference.byte_size,
                            )
                        ] = reference
            except ArtifactStoreError:
                # The stream scan above already reports the structural issue.
                continue
        for reference in references.values():
            try:
                self.read_bytes(reference)
            except ArtifactStoreError as exc:
                issues.append(
                    f"invalid referenced artifact {reference.artifact_id}: {exc}"
                )

        report = StoreIntegrityReport(
            run_id=self._run.run_id,
            streams=tuple(positions),
            blob_count=blob_count,
            blob_bytes=blob_bytes,
            valid=not issues,
            issues=tuple(issues),
        )
        if strict and issues:
            raise StoreIntegrityError("; ".join(issues))
        return report

    @classmethod
    def _walk_artifact_references(
        cls,
        value: object,
    ) -> Iterator[ArtifactReference]:
        if isinstance(value, ArtifactReference):
            yield value
            return
        if isinstance(value, BaseModel):
            for field_name in type(value).model_fields:
                yield from cls._walk_artifact_references(getattr(value, field_name))
            return
        if isinstance(value, dict):
            for item in value.values():
                yield from cls._walk_artifact_references(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                yield from cls._walk_artifact_references(item)

    def _scan_stream(
        self,
        stream: RecordStream,
        *,
        repair_trailing_partial: bool,
    ) -> tuple[int, int]:
        """Validate a stream and optionally remove only a partial final line."""

        path = self._stream_path(stream)
        if not path.exists():
            return 0, 0

        expected_model = self._model_for_stream(stream)
        count = 0
        last_valid_offset = 0
        try:
            with path.open("rb") as handle:
                while True:
                    line_offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        if repair_trailing_partial:
                            self._truncate_stream(path, line_offset)
                            return count, line_offset
                        raise StoreIntegrityError(
                            f"partial trailing record in {stream.value} at byte {line_offset}"
                        )
                    envelope = self._decode_envelope(
                        line,
                        stream=stream,
                        expected_sequence=count,
                        expected_record_type=expected_model.__name__,
                    )
                    try:
                        expected_model.model_validate(envelope["payload"])
                    except ValidationError as exc:
                        raise StoreIntegrityError(
                            f"invalid {expected_model.__name__} at "
                            f"{stream.value}:{count}: {exc}"
                        ) from exc
                    last_valid_offset = handle.tell()
                    count += 1
        except StoreIntegrityError:
            raise
        except OSError as exc:
            raise ArtifactStoreError(f"failed to scan {stream.value}: {exc}") from exc
        return count, last_valid_offset

    def _decode_envelope(
        self,
        line: bytes,
        *,
        stream: RecordStream,
        expected_sequence: int,
        expected_record_type: str,
    ) -> dict[str, object]:
        try:
            raw = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise StoreIntegrityError(
                f"invalid JSON in {stream.value} at sequence {expected_sequence}"
            ) from exc
        if not isinstance(raw, dict):
            raise StoreIntegrityError(
                f"record envelope must be an object in {stream.value}:{expected_sequence}"
            )

        required = {
            "schema_version",
            "stream",
            "sequence",
            "record_type",
            "recorded_at",
            "payload",
        }
        if set(raw) != required:
            raise StoreIntegrityError(
                f"unexpected envelope fields in {stream.value}:{expected_sequence}"
            )
        if raw["schema_version"] != SCHEMA_VERSION:
            raise StoreIntegrityError(
                f"unsupported schema version in {stream.value}:{expected_sequence}"
            )
        if raw["stream"] != stream.value:
            raise StoreIntegrityError(
                f"stream mismatch in {stream.value}:{expected_sequence}"
            )
        if raw["sequence"] != expected_sequence:
            raise StoreIntegrityError(
                f"sequence mismatch in {stream.value}: expected {expected_sequence}"
            )
        if raw["record_type"] != expected_record_type:
            raise StoreIntegrityError(
                f"record type mismatch in {stream.value}:{expected_sequence}"
            )
        if not isinstance(raw["payload"], dict):
            raise StoreIntegrityError(
                f"payload must be an object in {stream.value}:{expected_sequence}"
            )
        recorded_at = raw["recorded_at"]
        if not isinstance(recorded_at, str):
            raise StoreIntegrityError(
                f"recorded_at must be a string in {stream.value}:{expected_sequence}"
            )
        try:
            timestamp = datetime.fromisoformat(recorded_at)
        except ValueError as exc:
            raise StoreIntegrityError(
                f"invalid recorded_at in {stream.value}:{expected_sequence}"
            ) from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise StoreIntegrityError(
                f"recorded_at lacks timezone in {stream.value}:{expected_sequence}"
            )
        return cast(dict[str, object], raw)

    def _current_stream_positions(self) -> tuple[StreamPosition, ...]:
        return tuple(
            StreamPosition(
                stream=stream,
                record_count=self._stream_counts[stream],
                byte_length=self._stream_bytes[stream],
            )
            for stream in RecordStream
        )

    @classmethod
    def _resolve_run_directory(cls, base: Path, run_id: str) -> Path:
        if RUN_DIRECTORY_PATTERN.fullmatch(run_id) is None:
            raise ArtifactStoreError("run_id is not safe for use as a directory name")
        candidate = (base / run_id).resolve()
        if candidate.parent != base:
            raise ArtifactStoreError("run directory escapes the configured base")
        return candidate

    def _resolve_relative_path(self, relative_path: str) -> Path:
        if "\\" in relative_path:
            raise ArtifactStoreError("artifact path must use POSIX separators")
        candidate = (self._run_directory / relative_path).resolve()
        try:
            candidate.relative_to(self._run_directory)
        except ValueError as exc:
            raise ArtifactStoreError("artifact path escapes the run directory") from exc
        if candidate == self._run_directory:
            raise ArtifactStoreError("artifact path must reference a file")
        return candidate

    def _resolve_artifact_reference(self, reference: ArtifactReference) -> Path:
        expected_path = Path(
            self.BLOBS_DIRECTORY,
            reference.kind.value,
            reference.sha256[:2],
            reference.sha256,
        ).as_posix()
        if reference.relative_path != expected_path:
            raise ArtifactStoreError(
                "artifact reference does not match the content-addressed layout"
            )
        return self._resolve_relative_path(reference.relative_path)

    def _stream_path(self, stream: RecordStream) -> Path:
        return self._run_directory / _STREAM_FILES[stream]

    def _manifest_path(self) -> Path:
        path = self._run_directory / self.MANIFEST_FILE
        if not path.is_file():
            raise StoreNotFoundError("manifest does not exist")
        return path

    @staticmethod
    def _stream_for_model(model_type: type[BaseModel]) -> RecordStream:
        try:
            return _MODEL_STREAMS[model_type]
        except KeyError as exc:
            raise TypeError(
                f"unsupported evidence record type: {model_type.__name__}"
            ) from exc

    @staticmethod
    def _model_for_stream(stream: RecordStream) -> type[BaseModel]:
        for model_type, candidate_stream in _MODEL_STREAMS.items():
            if candidate_stream == stream:
                return model_type
        raise TypeError(f"no evidence model registered for {stream.value}")

    @staticmethod
    def _canonical_json_bytes(value: object) -> bytes:
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ArtifactStoreError(
                f"value cannot be serialized as JSON: {exc}"
            ) from exc

    @classmethod
    def _model_json_bytes(cls, model: BaseModel, *, pretty: bool) -> bytes:
        value = model.model_dump(mode="json")
        try:
            text = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                indent=2 if pretty else None,
                separators=None if pretty else (",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise ArtifactStoreError(
                f"model cannot be serialized as JSON: {exc}"
            ) from exc
        return (text + "\n").encode("utf-8")

    @classmethod
    def _read_model_file(
        cls,
        path: Path,
        model_type: Type[RecordModel],
    ) -> RecordModel:
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise StoreNotFoundError(
                f"required store file does not exist: {path.name}"
            ) from exc
        except OSError as exc:
            raise ArtifactStoreError(f"failed to read {path.name}: {exc}") from exc
        try:
            return model_type.model_validate_json(data)
        except ValidationError as exc:
            raise StoreIntegrityError(f"invalid {path.name}: {exc}") from exc

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
            )
        except OSError as exc:
            raise ArtifactStoreError(
                f"failed to create temporary file for {path.name}: {exc}"
            ) from exc
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, path)
            ArtifactStore._fsync_directory(path.parent)
        except OSError as exc:
            try:
                os.close(descriptor)
            except OSError:
                pass
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise ArtifactStoreError(f"failed to write {path.name}: {exc}") from exc

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        try:
            descriptor = os.open(directory, flags)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    @staticmethod
    def _verify_blob_file(path: Path, digest: str, expected_size: int) -> None:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ArtifactStoreError(f"failed to verify existing blob: {exc}") from exc
        if len(data) != expected_size or hashlib.sha256(data).hexdigest() != digest:
            raise StoreIntegrityError(f"content-addressed blob is corrupt: {path.name}")

    @staticmethod
    def _truncate_stream(path: Path, byte_length: int) -> None:
        try:
            with path.open("r+b") as handle:
                handle.truncate(byte_length)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ArtifactStoreError(f"failed to repair {path.name}: {exc}") from exc


__all__ = [
    "AppendReceipt",
    "ArtifactStore",
    "ArtifactStoreError",
    "RecordStream",
    "StoreAlreadyExistsError",
    "StoreCheckpoint",
    "StoreIntegrityError",
    "StoreIntegrityReport",
    "StoreNotFoundError",
    "StreamPosition",
]
