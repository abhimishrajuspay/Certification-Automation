"""Immutable evidence models for deterministic portal scraping.

These models define the storage contract between the future browser recorder,
snapshotter, state-graph explorer, and artifact store.  They intentionally do
not depend on Playwright, the LLM agent, or the certification execution DAG.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SHA256_PATTERN = r"^[a-f0-9]{64}$"
SCHEMA_VERSION = "1.0"


def utc_now() -> datetime:
    """Return an aware UTC timestamp for evidence records."""

    return datetime.now(timezone.utc)


class EvidenceModel(BaseModel):
    """Base configuration shared by all immutable evidence models."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class ScrapeRunStatus(str, Enum):
    """Lifecycle status of one scrape run."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CrawlCompletionGoal(str, Enum):
    """Configured condition that makes a crawl successful."""

    BOUNDED_FRONTIER = "bounded_frontier"
    TESTCASE_CONTEXT = "testcase_context"


class ArtifactKind(str, Enum):
    """Kinds of content-addressed artifacts produced by the scraper."""

    MANIFEST = "manifest"
    DOM = "dom"
    ELEMENTS = "elements"
    SCREENSHOT = "screenshot"
    ACCESSIBILITY_TREE = "accessibility_tree"
    STORAGE_STATE = "storage_state"
    REQUEST_BODY = "request_body"
    RESPONSE_BODY = "response_body"
    DOWNLOAD = "download"
    TRACE = "trace"
    HAR = "har"
    VIDEO = "video"
    OTHER = "other"


class LocatorStrategy(str, Enum):
    """Replay locator strategies, ordered independently by confidence."""

    TEST_ID = "test_id"
    ROLE = "role"
    LABEL = "label"
    PLACEHOLDER = "placeholder"
    ALT_TEXT = "alt_text"
    TITLE = "title"
    STABLE_ATTRIBUTE = "stable_attribute"
    TEXT = "text"
    CSS = "css"
    XPATH = "xpath"


class ActionKind(str, Enum):
    """Browser interactions that may form edges in the crawl graph."""

    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    HOVER = "hover"
    SCROLL = "scroll"
    FILL = "fill"
    CLEAR = "clear"
    SELECT_OPTION = "select_option"
    CHECK = "check"
    UNCHECK = "uncheck"
    PRESS = "press"
    SUBMIT = "submit"
    SWITCH_PAGE = "switch_page"
    HANDLE_DIALOG = "handle_dialog"
    OBSERVE_DOWNLOAD = "observe_download"


class ActionRisk(str, Enum):
    """Safety classification applied before an action is executed."""

    SAFE = "safe"
    REVIEW_REQUIRED = "review_required"
    BLOCKED = "blocked"


class ActionStatus(str, Enum):
    """Observed status of an action candidate or transition."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class BrowserEventKind(str, Enum):
    """Asynchronous browser events captured around an action."""

    REQUEST = "request"
    RESPONSE = "response"
    REQUEST_FAILED = "request_failed"
    CONSOLE = "console"
    PAGE_ERROR = "page_error"
    DIALOG = "dialog"
    POPUP = "popup"
    DOWNLOAD = "download"
    FRAME_ATTACHED = "frame_attached"
    FRAME_DETACHED = "frame_detached"
    FRAME_NAVIGATED = "frame_navigated"
    WEBSOCKET_OPENED = "websocket_opened"
    WEBSOCKET_FRAME_SENT = "websocket_frame_sent"
    WEBSOCKET_FRAME_RECEIVED = "websocket_frame_received"
    WEBSOCKET_CLOSED = "websocket_closed"
    WORKER_CREATED = "worker_created"
    DOM_MUTATION = "dom_mutation"
    NAVIGATION = "navigation"
    DOM_CONTENT_LOADED = "dom_content_loaded"
    LOAD = "load"
    PAGE_OPENED = "page_opened"
    PAGE_CLOSED = "page_closed"
    PAGE_CRASH = "page_crash"
    FILE_CHOOSER = "file_chooser"
    TRACE_SAVED = "trace_saved"
    HAR_SAVED = "har_saved"


class EffectKind(str, Enum):
    """Normalized effects observed after an interaction."""

    NAVIGATION = "navigation"
    DOM_CHANGE = "dom_change"
    FORM_CHANGE = "form_change"
    MODAL_OPENED = "modal_opened"
    MODAL_CLOSED = "modal_closed"
    NOTIFICATION = "notification"
    NETWORK_ACTIVITY = "network_activity"
    POPUP_OPENED = "popup_opened"
    DOWNLOAD_STARTED = "download_started"
    DIALOG_OPENED = "dialog_opened"
    STORAGE_CHANGE = "storage_change"
    CONSOLE_OUTPUT = "console_output"
    ERROR = "error"
    NO_OP = "no_op"


class ValueCapture(EvidenceModel):
    """A named value that can be safely redacted without losing provenance."""

    name: str = Field(min_length=1)
    value: Optional[str] = None
    safe_value: Optional[str] = None
    value_hash: Optional[str] = Field(default=None, pattern=SHA256_PATTERN)
    redacted: bool = False

    @model_validator(mode="after")
    def validate_redaction(self) -> "ValueCapture":
        """Prevent a redacted field from retaining its plaintext value."""

        if self.redacted and self.value is not None:
            raise ValueError("redacted values must not contain plaintext")
        if self.safe_value is not None and not self.redacted:
            raise ValueError("safe_value is only valid for redacted values")
        if self.safe_value is not None and not any(
            marker in self.safe_value.lower()
            for marker in ("[redacted]", "%5bredacted%5d")
        ):
            raise ValueError("safe_value must contain an explicit redaction marker")
        return self


class BoundingBox(EvidenceModel):
    """Element geometry in CSS pixels."""

    x: float
    y: float
    width: float = Field(ge=0)
    height: float = Field(ge=0)


class Viewport(EvidenceModel):
    """Browser viewport dimensions at snapshot time."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    device_scale_factor: float = Field(default=1.0, gt=0)


class ScrollPosition(EvidenceModel):
    """Current and maximum scroll offsets for a page or container."""

    x: float = Field(default=0, ge=0)
    y: float = Field(default=0, ge=0)
    maximum_x: float = Field(default=0, ge=0)
    maximum_y: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "ScrollPosition":
        """Ensure current offsets do not exceed measured maximums."""

        if self.x > self.maximum_x or self.y > self.maximum_y:
            raise ValueError("scroll position cannot exceed its maximum")
        return self


class ArtifactReference(EvidenceModel):
    """Reference to a content-addressed artifact within a scrape run."""

    artifact_id: str = Field(min_length=1)
    kind: ArtifactKind
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    media_type: str = Field(min_length=1)
    byte_size: int = Field(ge=0)
    redacted: bool = False
    captured_at: datetime = Field(default_factory=utc_now)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        """Keep artifacts inside their run directory."""

        if "\\" in value:
            raise ValueError("artifact paths must use POSIX separators")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
            raise ValueError("artifact path must be relative and cannot contain '..'")
        return str(path)

    @field_validator("captured_at")
    @classmethod
    def validate_captured_at(cls, value: datetime) -> datetime:
        return _require_aware_datetime(value)


class LocatorCandidate(EvidenceModel):
    """One replay strategy for locating an element after a rerender."""

    strategy: LocatorStrategy
    value: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    unique_match_count: Optional[int] = Field(default=None, ge=0)
    is_primary: bool = False


class SelectOptionSnapshot(EvidenceModel):
    """One option exposed by a select/listbox control."""

    label: str
    value: Optional[str] = None
    value_hash: Optional[str] = Field(default=None, pattern=SHA256_PATTERN)
    selected: bool = False
    disabled: bool = False
    redacted: bool = False

    @model_validator(mode="after")
    def validate_redaction(self) -> "SelectOptionSnapshot":
        if self.redacted and self.value is not None:
            raise ValueError("redacted option values must not contain plaintext")
        return self


class ElementContext(EvidenceModel):
    """Semantic context surrounding an element."""

    section_heading: Optional[str] = None
    form_id: Optional[str] = None
    table_id: Optional[str] = None
    row_label: Optional[str] = None
    inside_main: bool = False
    inside_dialog: bool = False
    ancestor_summary: tuple[str, ...] = ()


class ElementSnapshot(EvidenceModel):
    """Stable evidence describing one DOM element in a page state."""

    element_id: str = Field(min_length=1)
    frame_id: str = Field(min_length=1)
    tag: str = Field(min_length=1)
    role: Optional[str] = None
    accessible_name: Optional[str] = None
    label: Optional[str] = None
    text: Optional[str] = None
    title: Optional[str] = None
    input_type: Optional[str] = None
    attributes: tuple[ValueCapture, ...] = ()
    value_hash: Optional[str] = Field(default=None, pattern=SHA256_PATTERN)
    value_redacted: bool = False
    interactive: bool = False
    interaction_signals: tuple[str, ...] = ()
    visible: bool = False
    enabled: bool = False
    editable: bool = False
    checked: Optional[bool] = None
    selected: Optional[bool] = None
    expanded: Optional[bool] = None
    read_only: Optional[bool] = None
    bounding_box: Optional[BoundingBox] = None
    shadow_host_path: tuple[str, ...] = ()
    parent_css_path: Optional[str] = None
    context: ElementContext = Field(default_factory=ElementContext)
    options: tuple[SelectOptionSnapshot, ...] = ()
    locators: tuple[LocatorCandidate, ...] = ()
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_locators(self) -> "ElementSnapshot":
        """Allow at most one primary replay locator."""

        primary_count = sum(locator.is_primary for locator in self.locators)
        if primary_count > 1:
            raise ValueError("an element can have at most one primary locator")
        if len(self.interaction_signals) != len(set(self.interaction_signals)):
            raise ValueError("interaction_signals must be unique")
        return self


class FrameElementCollection(EvidenceModel):
    """Content-addressed structured element evidence for one frame."""

    frame_id: str = Field(min_length=1)
    elements: tuple[ElementSnapshot, ...] = ()

    @model_validator(mode="after")
    def validate_elements(self) -> "FrameElementCollection":
        element_ids = [element.element_id for element in self.elements]
        if len(element_ids) != len(set(element_ids)):
            raise ValueError("element identifiers must be unique within a frame")
        if any(element.frame_id != self.frame_id for element in self.elements):
            raise ValueError("all elements must reference the collection frame_id")
        return self


class FrameSnapshot(EvidenceModel):
    """Evidence for one frame in a browser state."""

    frame_id: str = Field(min_length=1)
    frame_path: str = Field(default="main", min_length=1)
    parent_frame_id: Optional[str] = None
    name: Optional[str] = None
    url: str = Field(min_length=1)
    is_main: bool = False
    is_cross_origin: bool = False
    element_ids: tuple[str, ...] = ()
    child_frame_ids: tuple[str, ...] = ()
    dom_artifact: Optional[ArtifactReference] = None
    elements_artifact: Optional[ArtifactReference] = None
    accessibility_artifact: Optional[ArtifactReference] = None
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_artifact_kinds(self) -> "FrameSnapshot":
        expected = (
            (self.dom_artifact, ArtifactKind.DOM, "dom_artifact"),
            (self.elements_artifact, ArtifactKind.ELEMENTS, "elements_artifact"),
            (
                self.accessibility_artifact,
                ArtifactKind.ACCESSIBILITY_TREE,
                "accessibility_artifact",
            ),
        )
        for reference, kind, field_name in expected:
            if reference is not None and reference.kind != kind:
                raise ValueError(f"{field_name} must reference a {kind.value} artifact")
        return self


class StateSnapshot(EvidenceModel):
    """Immutable representation of one discovered browser state."""

    state_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    fingerprint: str = Field(pattern=SHA256_PATTERN)
    captured_at: datetime = Field(default_factory=utc_now)
    page_id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    title: str = ""
    viewport: Viewport
    scroll: ScrollPosition = Field(default_factory=ScrollPosition)
    frames: tuple[FrameSnapshot, ...] = ()
    element_ids: tuple[str, ...] = ()
    active_element_id: Optional[str] = None
    artifacts: tuple[ArtifactReference, ...] = ()
    modal_count: int = Field(default=0, ge=0)
    loading_indicators: tuple[str, ...] = ()
    notifications: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    storage_fingerprint: Optional[str] = Field(default=None, pattern=SHA256_PATTERN)
    stable: bool = True
    limitations: tuple[str, ...] = ()

    @field_validator("captured_at")
    @classmethod
    def validate_captured_at(cls, value: datetime) -> datetime:
        return _require_aware_datetime(value)

    @model_validator(mode="after")
    def validate_references(self) -> "StateSnapshot":
        """Ensure frame and element identifiers are unique within the state."""

        frame_ids = [frame.frame_id for frame in self.frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("frame identifiers must be unique within a state")
        if len(self.element_ids) != len(set(self.element_ids)):
            raise ValueError("element identifiers must be unique within a state")
        if self.active_element_id and self.active_element_id not in self.element_ids:
            raise ValueError("active_element_id must reference an element in the state")
        return self


class ActionCandidate(EvidenceModel):
    """An interaction considered for a specific element and state."""

    action_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    state_id: str = Field(min_length=1)
    element_id: str = Field(min_length=1)
    kind: ActionKind
    risk: ActionRisk
    status: ActionStatus = ActionStatus.PENDING
    policy_rule: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    parameters: tuple[ValueCapture, ...] = ()
    discovered_at: datetime = Field(default_factory=utc_now)

    @field_validator("discovered_at")
    @classmethod
    def validate_discovered_at(cls, value: datetime) -> datetime:
        return _require_aware_datetime(value)


class BrowserEvent(EvidenceModel):
    """One timestamped event emitted by Playwright or page instrumentation."""

    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    kind: BrowserEventKind
    observed_at: datetime = Field(default_factory=utc_now)
    page_id: Optional[str] = None
    frame_id: Optional[str] = None
    action_id: Optional[str] = None
    summary: str = ""
    details: tuple[ValueCapture, ...] = ()
    artifacts: tuple[ArtifactReference, ...] = ()

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        return _require_aware_datetime(value)


class NetworkExchange(EvidenceModel):
    """Request/response evidence correlated with a page action when possible."""

    exchange_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    requested_at: datetime = Field(default_factory=utc_now)
    action_id: Optional[str] = None
    page_id: Optional[str] = None
    frame_id: Optional[str] = None
    url: str = Field(min_length=1)
    method: str = Field(min_length=1)
    resource_type: Optional[str] = None
    request_headers: tuple[ValueCapture, ...] = ()
    request_body: Optional[ArtifactReference] = None
    response_status: Optional[int] = Field(default=None, ge=100, le=599)
    response_headers: tuple[ValueCapture, ...] = ()
    response_body: Optional[ArtifactReference] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = Field(default=None, ge=0)
    from_cache: bool = False
    from_service_worker: bool = False
    failure_text: Optional[str] = None
    redirected_from_request_id: Optional[str] = None

    @field_validator("requested_at", "completed_at")
    @classmethod
    def validate_timestamps(cls, value: Optional[datetime]) -> Optional[datetime]:
        return _require_aware_datetime(value) if value is not None else None

    @model_validator(mode="after")
    def validate_timing(self) -> "NetworkExchange":
        if self.completed_at and self.completed_at < self.requested_at:
            raise ValueError("completed_at cannot precede requested_at")
        return self


class TransitionEffect(EvidenceModel):
    """A normalized effect derived from before/after evidence."""

    kind: EffectKind
    summary: str = Field(min_length=1)
    details: tuple[ValueCapture, ...] = ()
    artifact_ids: tuple[str, ...] = ()


class InteractionTransition(EvidenceModel):
    """Causal envelope for one attempted browser interaction."""

    transition_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    parent_state_id: str = Field(min_length=1)
    immediate_state_id: Optional[str] = None
    resulting_state_id: Optional[str] = None
    status: ActionStatus
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = Field(default=None, ge=0)
    locator_used: Optional[LocatorCandidate] = None
    event_ids: tuple[str, ...] = ()
    network_exchange_ids: tuple[str, ...] = ()
    effects: tuple[TransitionEffect, ...] = ()
    quiescence_reason: Optional[str] = None
    error_message: Optional[str] = None
    skip_reason: Optional[str] = None

    @field_validator("started_at", "completed_at")
    @classmethod
    def validate_timestamps(cls, value: Optional[datetime]) -> Optional[datetime]:
        return _require_aware_datetime(value) if value is not None else None

    @model_validator(mode="after")
    def validate_outcome(self) -> "InteractionTransition":
        """Require status-specific evidence for terminal transitions."""

        if self.completed_at and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        terminal = {
            ActionStatus.SUCCEEDED,
            ActionStatus.FAILED,
            ActionStatus.SKIPPED,
        }
        if self.status in terminal and self.completed_at is None:
            raise ValueError("terminal transitions require completed_at")
        if self.status == ActionStatus.SUCCEEDED and not self.resulting_state_id:
            raise ValueError("successful transitions require a resulting_state_id")
        if self.status == ActionStatus.FAILED and not self.error_message:
            raise ValueError("failed transitions require an error_message")
        if self.status == ActionStatus.SKIPPED and not self.skip_reason:
            raise ValueError("skipped transitions require a skip_reason")
        return self


class CrawlLimits(EvidenceModel):
    """Limits captured in the manifest so completeness is reproducible."""

    maximum_depth: int = Field(default=20, ge=0)
    maximum_states: int = Field(default=10_000, gt=0)
    maximum_actions: int = Field(default=100_000, gt=0)
    maximum_runtime_seconds: int = Field(default=14_400, gt=0)
    maximum_artifact_bytes: int = Field(default=10_737_418_240, gt=0)


class CapturePolicy(EvidenceModel):
    """Serializable snapshot of the run's capture and safety policy."""

    same_origin_only: bool = True
    allow_review_required_actions: bool = False
    capture_dom: bool = True
    capture_screenshots: bool = True
    capture_accessibility_tree: bool = True
    capture_storage_state: bool = True
    capture_request_bodies: bool = True
    capture_response_bodies: bool = True
    capture_trace: bool = True
    capture_har: bool = False
    maximum_body_bytes: int = Field(default=1_048_576, gt=0)
    allowed_body_media_types: tuple[str, ...] = (
        "application/json",
        "application/xml",
        "text/html",
        "text/plain",
        "text/xml",
    )
    redacted_names: tuple[str, ...] = (
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "password",
        "token",
        "csrf",
        "otp",
        "secret",
    )


class BrowserBehaviorPolicy(EvidenceModel):
    """Secret-free browser and authentication settings affecting a run."""

    browser_name: Literal["chromium", "firefox", "webkit"] = "chromium"
    headless: bool = True
    viewport_width: int = Field(default=1920, gt=0)
    viewport_height: int = Field(default=1080, gt=0)
    device_scale_factor: float = Field(default=1.0, gt=0)
    ignore_https_errors: bool = False
    default_timeout_ms: int = Field(default=30_000, gt=0)
    authentication_timeout_ms: int = Field(default=300_000, gt=0)
    authentication_mode: Literal[
        "none",
        "storage_state",
        "manual",
        "callback",
    ] = "none"
    ready_selector_configured: bool = False
    ready_selector_sha256: Optional[str] = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[a-f0-9]{64}$",
    )
    storage_state_input_configured: bool = False
    storage_state_output_configured: bool = False

    @model_validator(mode="after")
    def validate_ready_selector_marker(self) -> "BrowserBehaviorPolicy":
        if self.ready_selector_configured != (self.ready_selector_sha256 is not None):
            raise ValueError(
                "ready selector configuration requires a matching selector hash"
            )
        return self


class SnapshotBehaviorPolicy(EvidenceModel):
    """Serializable page-state extraction behavior."""

    maximum_elements_per_frame: int = Field(default=50_000, gt=0)
    maximum_text_chars: int = Field(default=4_000, gt=0)
    quiet_window_ms: int = Field(default=400, ge=0)
    quiet_timeout_ms: int = Field(default=5_000, gt=0)
    quiet_poll_interval_ms: int = Field(default=100, gt=0)
    full_page_screenshot: bool = True


class ActionBehaviorPolicy(EvidenceModel):
    """Serializable interaction discovery and safety behavior."""

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
    execution_control_keywords: tuple[str, ...] = (
        "test",
        "trigger",
    )
    allow_hidden_actions: bool = False
    allow_disabled_actions: bool = False
    include_hover_actions: bool = True
    include_scroll_actions: bool = True
    scroll_viewport_fraction: float = Field(default=0.8, gt=0, le=1)
    deduplicate_nested_targets: bool = True
    skip_ambiguous_delegated_containers: bool = True
    execution_timeout_ms: int = Field(default=10_000, gt=0)
    popup_detection_timeout_ms: int = Field(default=100, gt=0)

    @model_validator(mode="after")
    def validate_keywords(self) -> "ActionBehaviorPolicy":
        for name, keywords in (
            ("blocked_keywords", self.blocked_keywords),
            ("review_keywords", self.review_keywords),
            ("execution_control_keywords", self.execution_control_keywords),
        ):
            if any(not keyword.strip() for keyword in keywords):
                raise ValueError(f"{name} cannot contain empty values")
            if len(keywords) != len(set(keywords)):
                raise ValueError(f"{name} must be unique")
        return self


class ExplorerBehaviorPolicy(EvidenceModel):
    """Serializable graph restoration behavior not represented by limits."""

    restore_timeout_ms: int = Field(default=15_000, gt=0)
    capture_initial_state: bool = True
    completion_goal: CrawlCompletionGoal = CrawlCompletionGoal.BOUNDED_FRONTIER
    testcase_context_stability_observations: int = Field(default=2, gt=0)
    strategy: Literal["exhaustive", "guided", "hybrid"] = "exhaustive"
    guide_configured: bool = False
    guide_sha256: Optional[str] = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=SHA256_PATTERN,
    )
    teaching_enabled: bool = False
    root_scope_configured: bool = False
    worker_count: int = Field(default=1, gt=0, le=32)
    parallel_session_mode: Literal["off", "probe", "force"] = "off"

    @model_validator(mode="after")
    def validate_guidance_and_workers(self) -> "ExplorerBehaviorPolicy":
        if self.guide_configured != (self.guide_sha256 is not None):
            raise ValueError("configured guide requires a matching guide hash")
        if self.worker_count == 1 and self.parallel_session_mode != "off":
            raise ValueError("single-worker exploration requires parallel mode off")
        if self.worker_count > 1 and self.parallel_session_mode == "off":
            raise ValueError("multiple workers require probe or force parallel mode")
        return self


class CrawlBehaviorPolicy(EvidenceModel):
    """Complete secret-free behavior snapshot for reproducible crawling."""

    browser: BrowserBehaviorPolicy = Field(default_factory=BrowserBehaviorPolicy)
    snapshot: SnapshotBehaviorPolicy = Field(default_factory=SnapshotBehaviorPolicy)
    action: ActionBehaviorPolicy = Field(default_factory=ActionBehaviorPolicy)
    explorer: ExplorerBehaviorPolicy = Field(default_factory=ExplorerBehaviorPolicy)


class ScrapeRun(EvidenceModel):
    """Top-level manifest model for a deterministic scrape run."""

    schema_version: str = SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    root_url: str = Field(min_length=1)
    allowed_origins: tuple[str, ...]
    status: ScrapeRunStatus = ScrapeRunStatus.CREATED
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    limits: CrawlLimits = Field(default_factory=CrawlLimits)
    capture_policy: CapturePolicy = Field(default_factory=CapturePolicy)
    behavior_policy: CrawlBehaviorPolicy = Field(default_factory=CrawlBehaviorPolicy)
    state_ids: tuple[str, ...] = ()
    transition_ids: tuple[str, ...] = ()
    checkpoint_sequence: int = Field(default=0, ge=0)
    completion_reason: Optional[str] = None

    @field_validator("root_url")
    @classmethod
    def validate_root_url(cls, value: str) -> str:
        return _require_http_url(value, "root_url")

    @field_validator("allowed_origins")
    @classmethod
    def validate_allowed_origins(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("at least one allowed origin is required")
        normalized = tuple(_require_origin(value) for value in values)
        if len(normalized) != len(set(normalized)):
            raise ValueError("allowed origins must be unique")
        return normalized

    @field_validator("started_at", "ended_at")
    @classmethod
    def validate_timestamps(cls, value: Optional[datetime]) -> Optional[datetime]:
        return _require_aware_datetime(value) if value is not None else None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "ScrapeRun":
        terminal = {
            ScrapeRunStatus.COMPLETED,
            ScrapeRunStatus.PARTIAL,
            ScrapeRunStatus.FAILED,
            ScrapeRunStatus.CANCELLED,
        }
        if self.status == ScrapeRunStatus.RUNNING and self.started_at is None:
            raise ValueError("running scrape runs require started_at")
        if self.status in terminal:
            if self.started_at is None or self.ended_at is None:
                raise ValueError(
                    "terminal scrape runs require start and end timestamps"
                )
            if not self.completion_reason:
                raise ValueError("terminal scrape runs require a completion_reason")
        if self.started_at and self.ended_at and self.ended_at < self.started_at:
            raise ValueError("ended_at cannot precede started_at")
        root_origin = _origin_from_url(self.root_url)
        if root_origin not in self.allowed_origins:
            raise ValueError("root_url origin must be included in allowed_origins")
        if len(self.state_ids) != len(set(self.state_ids)):
            raise ValueError("state_ids must be unique")
        if len(self.transition_ids) != len(set(self.transition_ids)):
            raise ValueError("transition_ids must be unique")
        return self


class TestcaseContextCoverage(EvidenceModel):
    """Progress toward a complete, internally consistent testcase context."""

    declared_test_cases: Optional[int] = Field(default=None, ge=0)
    test_cases_discovered: int = Field(default=0, ge=0)
    descriptions_captured: int = Field(default=0, ge=0)
    missing_description_ids: tuple[str, ...] = ()
    conflicting_test_case_ids: tuple[str, ...] = ()
    declared_total_conflicts: tuple[str, ...] = ()
    stable_observations: int = Field(default=0, ge=0)
    required_stable_observations: int = Field(default=2, gt=0)
    context_complete: bool = False

    @model_validator(mode="after")
    def validate_context_counts(self) -> "TestcaseContextCoverage":
        if self.descriptions_captured > self.test_cases_discovered:
            raise ValueError("description count cannot exceed testcase count")
        if len(self.missing_description_ids) != (
            self.test_cases_discovered - self.descriptions_captured
        ):
            raise ValueError("missing description IDs do not match coverage counts")
        for values, name in (
            (self.missing_description_ids, "missing description IDs"),
            (self.conflicting_test_case_ids, "conflicting testcase IDs"),
            (self.declared_total_conflicts, "declared total conflicts"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        complete = bool(
            self.declared_test_cases is not None
            and self.declared_test_cases == self.test_cases_discovered
            and not self.missing_description_ids
            and not self.conflicting_test_case_ids
            and not self.declared_total_conflicts
        )
        if self.context_complete != complete:
            raise ValueError("context_complete does not match testcase coverage")
        if not self.context_complete and self.stable_observations:
            raise ValueError("incomplete testcase context cannot be stable")
        return self

    @property
    def stable_for_early_stop(self) -> bool:
        """Return whether complete evidence remained stable long enough to stop."""

        return bool(
            self.context_complete
            and self.stable_observations >= self.required_stable_observations
        )


class CoverageReport(EvidenceModel):
    """Machine-readable bounded-completeness report for a scrape run."""

    run_id: str = Field(min_length=1)
    generated_at: datetime = Field(default_factory=utc_now)
    states_discovered: int = Field(default=0, ge=0)
    duplicate_states: int = Field(default=0, ge=0)
    elements_discovered: int = Field(default=0, ge=0)
    action_candidates: int = Field(default=0, ge=0)
    actions_succeeded: int = Field(default=0, ge=0)
    actions_failed: int = Field(default=0, ge=0)
    actions_skipped: int = Field(default=0, ge=0)
    actions_pending: int = Field(default=0, ge=0)
    routes_discovered: int = Field(default=0, ge=0)
    frames_discovered: int = Field(default=0, ge=0)
    tables_discovered: int = Field(default=0, ge=0)
    modals_discovered: int = Field(default=0, ge=0)
    downloads_observed: int = Field(default=0, ge=0)
    console_errors: int = Field(default=0, ge=0)
    page_errors: int = Field(default=0, ge=0)
    unexplored_action_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    bounded_complete: bool = False
    completion_goal: CrawlCompletionGoal = CrawlCompletionGoal.BOUNDED_FRONTIER
    goal_complete: Optional[bool] = None
    testcase_context: Optional[TestcaseContextCoverage] = None
    completion_reason: str = Field(min_length=1)

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _require_aware_datetime(value)

    @model_validator(mode="after")
    def validate_action_counts(self) -> "CoverageReport":
        resolved = self.actions_succeeded + self.actions_failed + self.actions_skipped
        total = resolved + self.actions_pending
        if total != self.action_candidates:
            raise ValueError("action outcome counts must equal action_candidates")
        if self.actions_pending != len(self.unexplored_action_ids):
            raise ValueError("pending count must match unexplored_action_ids")
        if len(self.unexplored_action_ids) != len(set(self.unexplored_action_ids)):
            raise ValueError("unexplored_action_ids must be unique")
        if self.bounded_complete and self.actions_pending:
            raise ValueError("a bounded-complete run cannot have pending actions")
        if self.goal_complete is not None:
            if (
                self.completion_goal == CrawlCompletionGoal.BOUNDED_FRONTIER
                and self.goal_complete != self.bounded_complete
            ):
                raise ValueError("frontier goal must match bounded completeness")
            if self.completion_goal == CrawlCompletionGoal.TESTCASE_CONTEXT:
                if self.testcase_context is None:
                    raise ValueError("testcase goal requires testcase context coverage")
                if self.goal_complete != self.testcase_context.context_complete:
                    raise ValueError(
                        "testcase goal must match testcase context coverage"
                    )
        return self

    @property
    def configured_goal_complete(self) -> bool:
        """Resolve legacy reports to their historical frontier-success meaning."""

        if self.goal_complete is not None:
            return self.goal_complete
        return self.bounded_complete


def _require_aware_datetime(value: datetime) -> datetime:
    """Reject ambiguous timestamps and normalize valid values to UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must include timezone information")
    return value.astimezone(timezone.utc)


def _require_http_url(value: str, field_name: str) -> str:
    """Validate an HTTP(S) URL while preserving its original string form."""

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} must not contain embedded credentials")
    return value


def _require_origin(value: str) -> str:
    """Validate and normalize an allowed origin to scheme://host[:port]."""

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("allowed origins must be absolute HTTP(S) origins")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("allowed origins must not contain embedded credentials")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("allowed origins cannot contain paths, queries, or fragments")
    return _origin_from_url(value)


def _origin_from_url(value: str) -> str:
    """Return a canonical, credential-free origin for an HTTP(S) URL."""

    parsed = urlparse(value)
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("URL must contain a hostname")
    host = f"[{hostname.lower()}]" if ":" in hostname else hostname.lower()
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme.lower()}://{host}{port}"


__all__ = [
    "SCHEMA_VERSION",
    "ActionCandidate",
    "ActionKind",
    "ActionRisk",
    "ActionStatus",
    "ArtifactKind",
    "ArtifactReference",
    "BoundingBox",
    "BrowserEvent",
    "BrowserEventKind",
    "BrowserBehaviorPolicy",
    "CapturePolicy",
    "CrawlBehaviorPolicy",
    "CoverageReport",
    "CrawlLimits",
    "EffectKind",
    "ElementContext",
    "ElementSnapshot",
    "ExplorerBehaviorPolicy",
    "FrameElementCollection",
    "FrameSnapshot",
    "InteractionTransition",
    "LocatorCandidate",
    "LocatorStrategy",
    "NetworkExchange",
    "ActionBehaviorPolicy",
    "ScrapeRun",
    "ScrapeRunStatus",
    "ScrollPosition",
    "SelectOptionSnapshot",
    "SnapshotBehaviorPolicy",
    "StateSnapshot",
    "TransitionEffect",
    "ValueCapture",
    "Viewport",
    "utc_now",
]
