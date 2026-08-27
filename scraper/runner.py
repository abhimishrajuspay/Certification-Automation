"""Production orchestration boundary for one deterministic portal crawl."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import threading
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from playwright.async_api import Page

from scraper.actions import ActionExecutor, ActionPlanner, ActionPolicyConfig
from scraper.artifact_store import (
    ArtifactStore,
    StoreIntegrityError,
    StoreNotFoundError,
)
from scraper.browser import BrowserLaunchConfig, BrowserManager, BrowserName
from scraper.explorer import (
    ExplorerConfig,
    ExplorationResult,
    ExplorationWorker,
    StateGraphExplorer,
)
from scraper.extractor import PageStateExtractor, SnapshotConfig
from scraper.guidance import (
    CrawlGuide,
    CrawlStrategy,
    GuidanceError,
    OperatorActionRecorder,
    ParallelSessionMode,
    compile_taught_guide,
    guide_sha256,
    load_crawl_guide,
    write_crawl_guide,
)
from scraper.models import (
    ActionBehaviorPolicy,
    ActionCandidate,
    BrowserBehaviorPolicy,
    CapturePolicy,
    CoverageReport,
    CrawlBehaviorPolicy,
    CrawlLimits,
    ExplorerBehaviorPolicy,
    ScrapeRun,
    ScrapeRunStatus,
    SnapshotBehaviorPolicy,
    StateSnapshot,
    utc_now,
)
from scraper.redaction import hash_text, redact_text, redact_url


_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class CrawlRunnerError(RuntimeError):
    """Base error raised by the production crawl runner."""


class AuthenticationError(CrawlRunnerError):
    """Raised when the configured authentication boundary cannot complete."""


class ResumeUnavailableError(CrawlRunnerError):
    """Raised when an existing run cannot be safely reused or resumed."""


class StorageStateExportError(CrawlRunnerError):
    """Raised when an explicitly requested reusable session cannot be saved."""


class TeachingError(CrawlRunnerError):
    """Raised when an explicit operator teaching session cannot be completed."""


class AuthenticationMode(str, Enum):
    """Supported ways to establish a portal session before extraction."""

    NONE = "none"
    STORAGE_STATE = "storage_state"
    MANUAL = "manual"
    CALLBACK = "callback"


class ExistingRunPolicy(str, Enum):
    """Behavior when the requested run identifier already exists."""

    ERROR = "error"
    RETURN_COMPLETED = "return_completed"
    NEW_ATTEMPT = "new_attempt"


LoginCallback = Callable[[Page], Awaitable[None]]
ManualConfirmation = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class CrawlRequest:
    """Validated runtime configuration with explicit persistence boundaries."""

    root_url: str
    artifact_root: Path = Path("artifacts/crawls")
    run_id: Optional[str] = None
    allowed_origins: tuple[str, ...] = ()
    authentication_mode: AuthenticationMode = AuthenticationMode.NONE
    storage_state_path: Optional[Path] = None
    storage_state_output_path: Optional[Path] = None
    overwrite_storage_state_output: bool = False
    existing_run_policy: ExistingRunPolicy = ExistingRunPolicy.ERROR
    browser_name: BrowserName = "chromium"
    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080
    ignore_https_errors: bool = False
    default_timeout_ms: int = 30_000
    authentication_timeout_ms: int = 300_000
    action_timeout_ms: int = 10_000
    popup_detection_timeout_ms: int = 100
    ready_selector: Optional[str] = None
    guide_path: Optional[Path] = None
    taught_guide_output_path: Optional[Path] = None
    overwrite_taught_guide: bool = False
    teaching_timeout_ms: int = 900_000
    teaching_branch_depth: int = 0
    limits: CrawlLimits = field(default_factory=CrawlLimits)
    capture_policy: CapturePolicy = field(default_factory=CapturePolicy)
    snapshot_config: SnapshotConfig = field(default_factory=SnapshotConfig)
    action_policy_config: ActionPolicyConfig = field(default_factory=ActionPolicyConfig)
    explorer_config: ExplorerConfig = field(default_factory=ExplorerConfig)

    def __post_init__(self) -> None:
        _origin_from_url(self.root_url)
        if self.run_id is not None and not _RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ValueError(
                "run_id must be 1-128 characters using letters, numbers, '.', "
                "'_', or '-', and must start with a letter or number"
            )
        if self.authentication_timeout_ms <= 0:
            raise ValueError("authentication_timeout_ms must be positive")
        if self.teaching_timeout_ms <= 0:
            raise ValueError("teaching_timeout_ms must be positive")
        if self.teaching_branch_depth < 0 or self.teaching_branch_depth > 8:
            raise ValueError("teaching_branch_depth must be between 0 and 8")
        if self.action_timeout_ms <= 0:
            raise ValueError("action_timeout_ms must be positive")
        if self.popup_detection_timeout_ms <= 0:
            raise ValueError("popup_detection_timeout_ms must be positive")
        if self.ready_selector is not None and not self.ready_selector.strip():
            raise ValueError("ready_selector cannot be empty")
        if self.authentication_mode == AuthenticationMode.MANUAL and self.headless:
            raise ValueError("manual authentication requires a headed browser")
        if self.authentication_mode == AuthenticationMode.STORAGE_STATE:
            if self.storage_state_path is None:
                raise ValueError(
                    "storage_state authentication requires storage_state_path"
                )
            if not self.storage_state_path.expanduser().is_file():
                raise ValueError("storage_state_path must reference an existing file")
        elif self.storage_state_path is not None:
            raise ValueError(
                "storage_state_path is only valid with storage_state authentication"
            )
        if self.storage_state_output_path is not None:
            output = self.storage_state_output_path.expanduser()
            if output.exists() and not output.is_file():
                raise ValueError("storage_state_output_path must reference a file")
            if output.exists() and not self.overwrite_storage_state_output:
                raise ValueError(
                    "storage_state_output_path already exists; explicit overwrite "
                    "permission is required"
                )
        elif self.overwrite_storage_state_output:
            raise ValueError(
                "overwrite_storage_state_output requires storage_state_output_path"
            )
        if self.guide_path is not None and not self.guide_path.expanduser().is_file():
            raise ValueError("guide_path must reference an existing file")
        if self.guide_path is not None and self.taught_guide_output_path is not None:
            raise ValueError("guide replay and guide teaching are mutually exclusive")
        if self.explorer_config.strategy == CrawlStrategy.EXHAUSTIVE:
            if self.guide_path is not None or self.taught_guide_output_path is not None:
                raise ValueError(
                    "guide replay or teaching requires guided/hybrid strategy"
                )
        elif self.guide_path is None and self.taught_guide_output_path is None:
            raise ValueError(
                "guided/hybrid strategy requires a guide or teaching output"
            )
        if self.taught_guide_output_path is not None:
            if self.authentication_mode != AuthenticationMode.MANUAL or self.headless:
                raise ValueError("guide teaching requires headed manual authentication")
            output = self.taught_guide_output_path.expanduser()
            if output.exists() and not output.is_file():
                raise ValueError("taught guide output must reference a file")
            if output.exists() and not self.overwrite_taught_guide:
                raise ValueError(
                    "taught guide output already exists; explicit overwrite is required"
                )
        elif self.overwrite_taught_guide:
            raise ValueError("overwrite_taught_guide requires taught_guide_output_path")
        if self.teaching_branch_depth and self.taught_guide_output_path is None:
            raise ValueError("teaching_branch_depth requires taught_guide_output_path")
        if not self.explorer_config.capture_initial_state:
            raise ValueError("the crawl runner requires capture_initial_state=True")

        # Pydantic performs the final URL/origin and run-id validation here without
        # retaining authentication material in the durable manifest.
        ScrapeRun(
            run_id=self.run_id or "validation-run",
            root_url=self.evidence_root_url,
            allowed_origins=self.effective_allowed_origins,
            limits=self.limits,
            capture_policy=self.capture_policy,
        )
        BrowserLaunchConfig(
            browser_name=self.browser_name,
            headless=self.headless,
            viewport_width=self.viewport_width,
            viewport_height=self.viewport_height,
            ignore_https_errors=self.ignore_https_errors,
            storage_state_path=self.storage_state_path,
            default_timeout_ms=self.default_timeout_ms,
        )

    @property
    def effective_allowed_origins(self) -> tuple[str, ...]:
        """Return explicit origins or the canonical root origin by default."""

        values = self.allowed_origins or (_origin_from_url(self.root_url),)
        return tuple(_normalize_allowed_origin(value) for value in values)

    @property
    def evidence_root_url(self) -> str:
        """Return the redacted URL that is safe to persist in the manifest."""

        return redact_url(self.root_url, self.capture_policy.redacted_names)


@dataclass(frozen=True)
class CrawlRunResult:
    """Stable caller-facing result for a new or reused crawl run."""

    run_id: str
    run_directory: Path
    status: ScrapeRunStatus
    exploration: ExplorationResult
    reused_existing: bool = False


class CrawlRunner:
    """Connect authentication, browser instrumentation, and graph exploration."""

    def __init__(
        self,
        request: CrawlRequest,
        *,
        login_callback: Optional[LoginCallback] = None,
        manual_confirmation: Optional[ManualConfirmation] = None,
    ) -> None:
        if (
            request.authentication_mode == AuthenticationMode.CALLBACK
            and login_callback is None
        ):
            raise AuthenticationError(
                "callback authentication requires a login_callback"
            )
        if (
            request.authentication_mode != AuthenticationMode.CALLBACK
            and login_callback is not None
        ):
            raise AuthenticationError(
                "login_callback is only valid with callback authentication"
            )
        self.request = request
        self.login_callback = login_callback
        self.manual_confirmation = manual_confirmation
        self.guide: Optional[CrawlGuide] = None
        self.guide_sha256: Optional[str] = None
        if request.guide_path is not None:
            try:
                self.guide, self.guide_sha256 = load_crawl_guide(request.guide_path)
            except GuidanceError as exc:
                raise CrawlRunnerError(str(exc)) from exc
        self.teacher = (
            OperatorActionRecorder(request.capture_policy.redacted_names)
            if request.taught_guide_output_path is not None
            else None
        )

    async def run(self) -> CrawlRunResult:
        """Execute a new run or return a verified completed run by policy."""

        store, reused = await asyncio.to_thread(self._prepare_store)
        if reused:
            return await asyncio.to_thread(self._existing_result, store)

        manager = BrowserManager(store, self._browser_config())
        managers = [manager]
        result: Optional[ExplorationResult] = None
        try:
            page = await manager.start()
            if self.teacher is not None:
                await self.teacher.install(manager.context)
            await manager.navigate(self.request.root_url)
            await self._authenticate(page)
            taught = False
            if self.teacher is not None:
                await self._teach()
                await asyncio.to_thread(
                    store.save_manifest,
                    store.run.model_copy(
                        update={"behavior_policy": self._behavior_policy()}
                    ),
                )
                taught = True
                if self.guide is not None and self.guide.branch_rules:
                    await manager.navigate(self.request.root_url)
            await self._export_storage_state(page)

            extractor = PageStateExtractor(
                store,
                manager.recorder,
                self.request.snapshot_config,
            )
            planner = ActionPlanner(store, self.request.action_policy_config)
            executor = ActionExecutor(
                store,
                manager.recorder,
                extractor,
                action_timeout_ms=self.request.action_timeout_ms,
                popup_detection_timeout_ms=(self.request.popup_detection_timeout_ms),
            )
            additional_workers: list[ExplorationWorker] = []
            for worker_number in range(
                2, self.request.explorer_config.worker_count + 1
            ):
                worker_manager = BrowserManager(store, self._browser_config())
                try:
                    worker_page = await worker_manager.start()
                except Exception:
                    if (
                        self.request.explorer_config.parallel_session_mode
                        == ParallelSessionMode.FORCE
                    ):
                        raise
                    break
                managers.append(worker_manager)
                worker_extractor = PageStateExtractor(
                    store,
                    worker_manager.recorder,
                    self.request.snapshot_config,
                )
                additional_workers.append(
                    ExplorationWorker(
                        worker_id=f"worker-{worker_number}",
                        page=worker_page,
                        recorder=worker_manager.recorder,
                        extractor=worker_extractor,
                        executor=ActionExecutor(
                            store,
                            worker_manager.recorder,
                            worker_extractor,
                            action_timeout_ms=self.request.action_timeout_ms,
                            popup_detection_timeout_ms=(
                                self.request.popup_detection_timeout_ms
                            ),
                        ),
                    )
                )
            runtime_guide = self.guide
            if taught and runtime_guide is not None:
                if runtime_guide.branch_rules:
                    runtime_guide = self.guide
                else:
                    runtime_guide = (
                        runtime_guide.model_copy(update={"steps": ()})
                        if runtime_guide.repeat_rules
                        else None
                    )
            result = await StateGraphExplorer(
                store,
                manager.recorder,
                extractor,
                planner,
                executor,
                self.request.explorer_config,
                guide=runtime_guide,
                additional_workers=tuple(additional_workers),
            ).explore(page)
        except BaseException as exc:
            await self._finalize_early_failure(store, exc)
            await self._stop_managers(managers, suppress_errors=True)
            raise

        try:
            await self._stop_managers(managers, suppress_errors=False)
        except Exception as exc:
            await self._mark_partial_after_shutdown_failure(store, exc)
            raise

        if result is None:  # pragma: no cover - defensive invariant
            raise CrawlRunnerError("crawl finished without an exploration result")
        return CrawlRunResult(
            run_id=store.run.run_id,
            run_directory=store.run_directory,
            status=store.run.status,
            exploration=result,
        )

    @staticmethod
    async def _stop_managers(
        managers: list[BrowserManager],
        *,
        suppress_errors: bool,
    ) -> None:
        first_error: Optional[Exception] = None
        for manager in reversed(managers):
            try:
                await manager.stop()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None and not suppress_errors:
            raise first_error

    def _prepare_store(self) -> tuple[ArtifactStore, bool]:
        requested_id = self.request.run_id or _new_run_id()
        existing = self._open_if_present(requested_id)
        if existing is not None:
            if self.request.existing_run_policy == ExistingRunPolicy.ERROR:
                raise ResumeUnavailableError(
                    f"scrape run already exists: {requested_id}"
                )
            if self.request.existing_run_policy == ExistingRunPolicy.RETURN_COMPLETED:
                self._validate_existing_compatibility(existing)
                if existing.run.status != ScrapeRunStatus.COMPLETED:
                    raise ResumeUnavailableError(
                        "only a completed run can be returned; partial frontier "
                        "resume is intentionally unsupported because browser "
                        "authentication and live page state are not persisted"
                    )
                existing.verify_integrity(strict=True)
                return existing, True
            requested_id = self._next_attempt_id(requested_id)

        run = ScrapeRun(
            run_id=requested_id,
            root_url=self.request.evidence_root_url,
            allowed_origins=self.request.effective_allowed_origins,
            limits=self.request.limits,
            capture_policy=self.request.capture_policy,
            behavior_policy=self._behavior_policy(),
        )
        return ArtifactStore.create(self.request.artifact_root, run), False

    def _validate_existing_compatibility(self, store: ArtifactStore) -> None:
        expected = (
            self.request.evidence_root_url,
            self.request.effective_allowed_origins,
            self.request.limits,
            self.request.capture_policy,
            self._behavior_policy(),
        )
        actual = (
            store.run.root_url,
            store.run.allowed_origins,
            store.run.limits,
            store.run.capture_policy,
            store.run.behavior_policy,
        )
        if actual != expected:
            raise ResumeUnavailableError(
                "existing run configuration does not match the requested portal, "
                "origin scope, limits, capture policy, or behavior policy"
            )

    def _behavior_policy(self) -> CrawlBehaviorPolicy:
        snapshot = self.request.snapshot_config
        action = self.request.action_policy_config
        explorer = self.request.explorer_config
        return CrawlBehaviorPolicy(
            browser=BrowserBehaviorPolicy(
                browser_name=self.request.browser_name,
                headless=self.request.headless,
                viewport_width=self.request.viewport_width,
                viewport_height=self.request.viewport_height,
                ignore_https_errors=self.request.ignore_https_errors,
                default_timeout_ms=self.request.default_timeout_ms,
                authentication_timeout_ms=self.request.authentication_timeout_ms,
                authentication_mode=self.request.authentication_mode.value,
                ready_selector_configured=self.request.ready_selector is not None,
                ready_selector_sha256=(
                    hash_text(self.request.ready_selector)
                    if self.request.ready_selector is not None
                    else None
                ),
                storage_state_input_configured=(
                    self.request.storage_state_path is not None
                ),
                storage_state_output_configured=(
                    self.request.storage_state_output_path is not None
                ),
            ),
            snapshot=SnapshotBehaviorPolicy(
                maximum_elements_per_frame=snapshot.maximum_elements_per_frame,
                maximum_text_chars=snapshot.maximum_text_chars,
                quiet_window_ms=snapshot.quiet_window_ms,
                quiet_timeout_ms=snapshot.quiet_timeout_ms,
                quiet_poll_interval_ms=snapshot.quiet_poll_interval_ms,
                full_page_screenshot=snapshot.full_page_screenshot,
            ),
            action=ActionBehaviorPolicy(
                blocked_keywords=action.blocked_keywords,
                review_keywords=action.review_keywords,
                execution_control_keywords=action.execution_control_keywords,
                allow_hidden_actions=action.allow_hidden_actions,
                allow_disabled_actions=action.allow_disabled_actions,
                include_hover_actions=action.include_hover_actions,
                include_scroll_actions=action.include_scroll_actions,
                scroll_viewport_fraction=action.scroll_viewport_fraction,
                deduplicate_nested_targets=action.deduplicate_nested_targets,
                skip_ambiguous_delegated_containers=(
                    action.skip_ambiguous_delegated_containers
                ),
                execution_timeout_ms=self.request.action_timeout_ms,
                popup_detection_timeout_ms=(self.request.popup_detection_timeout_ms),
            ),
            explorer=ExplorerBehaviorPolicy(
                restore_timeout_ms=explorer.restore_timeout_ms,
                capture_initial_state=explorer.capture_initial_state,
                completion_goal=explorer.completion_goal,
                testcase_context_stability_observations=(
                    explorer.testcase_context_stability_observations
                ),
                strategy=explorer.strategy.value,
                guide_configured=self.guide_sha256 is not None,
                guide_sha256=self.guide_sha256,
                teaching_enabled=self.teacher is not None,
                root_scope_configured=bool(
                    self.guide and self.guide.root_scope_selector
                ),
                worker_count=explorer.worker_count,
                parallel_session_mode=explorer.parallel_session_mode.value,
            ),
        )

    def _open_if_present(self, run_id: str) -> Optional[ArtifactStore]:
        try:
            return ArtifactStore.open(self.request.artifact_root, run_id)
        except StoreNotFoundError:
            return None

    def _next_attempt_id(self, base_run_id: str) -> str:
        for attempt in range(1, 10_000):
            suffix = f"-attempt-{attempt}"
            candidate = f"{base_run_id[: 128 - len(suffix)]}{suffix}"
            if self._open_if_present(candidate) is None:
                return candidate
        raise ResumeUnavailableError("could not allocate a new attempt run_id")

    def _existing_result(self, store: ArtifactStore) -> CrawlRunResult:
        states = tuple(store.iter_records(StateSnapshot))
        actions = tuple(store.iter_records(ActionCandidate))
        coverage_records = tuple(store.iter_records(CoverageReport))
        if not states or not coverage_records:
            raise StoreIntegrityError(
                "completed run is missing state or coverage evidence"
            )
        coverage = coverage_records[-1]
        exploration = ExplorationResult(
            root_state_id=states[0].state_id,
            state_ids=store.run.state_ids,
            action_ids=tuple(action.action_id for action in actions),
            transition_ids=store.run.transition_ids,
            coverage=coverage,
            completion_reason=store.run.completion_reason or coverage.completion_reason,
        )
        return CrawlRunResult(
            run_id=store.run.run_id,
            run_directory=store.run_directory,
            status=store.run.status,
            exploration=exploration,
            reused_existing=True,
        )

    def _browser_config(self) -> BrowserLaunchConfig:
        return BrowserLaunchConfig(
            browser_name=self.request.browser_name,
            headless=self.request.headless,
            viewport_width=self.request.viewport_width,
            viewport_height=self.request.viewport_height,
            ignore_https_errors=self.request.ignore_https_errors,
            storage_state_path=self.request.storage_state_path,
            default_timeout_ms=self.request.default_timeout_ms,
        )

    async def _authenticate(self, page: Page) -> None:
        mode = self.request.authentication_mode
        try:
            if mode == AuthenticationMode.CALLBACK:
                if self.login_callback is None:  # pragma: no cover - constructor guard
                    raise AuthenticationError("login callback is unavailable")
                await asyncio.wait_for(
                    self.login_callback(page),
                    timeout=self.request.authentication_timeout_ms / 1000,
                )
            elif mode == AuthenticationMode.MANUAL:
                prompt = (
                    "Complete login in the opened browser for "
                    f"{_origin_from_url(self.request.root_url)}, then press Enter: "
                )
                confirmation = self.manual_confirmation or _console_confirmation
                await asyncio.wait_for(
                    confirmation(prompt),
                    timeout=self.request.authentication_timeout_ms / 1000,
                )

            if self.request.ready_selector is not None:
                await page.wait_for_selector(
                    self.request.ready_selector,
                    state="visible",
                    timeout=self.request.authentication_timeout_ms,
                )
        except asyncio.TimeoutError as exc:
            raise AuthenticationError("authentication readiness timed out") from exc
        except AuthenticationError:
            raise
        except Exception as exc:
            safe_error = redact_text(
                str(exc),
                self.request.capture_policy.redacted_names,
            )
            raise AuthenticationError(
                f"authentication readiness failed: {safe_error}"
            ) from exc

    async def _export_storage_state(self, page: Page) -> None:
        output_path = self.request.storage_state_output_path
        if output_path is None:
            return
        try:
            state = await page.context.storage_state()
            payload = json.dumps(
                state,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            await asyncio.to_thread(
                _write_private_file,
                output_path,
                payload,
                overwrite=self.request.overwrite_storage_state_output,
            )
        except StorageStateExportError:
            raise
        except Exception as exc:
            raise StorageStateExportError(
                "failed to export reusable storage state"
            ) from exc

    async def _teach(self) -> CrawlGuide:
        """Record post-login operator clicks and persist their reusable guide."""

        teacher = self.teacher
        output = self.request.taught_guide_output_path
        if teacher is None or output is None:  # pragma: no cover - caller invariant
            raise TeachingError("teaching was not configured")
        prompt = (
            "Teaching active: click the route to the testcase page"
            + (
                f", including {self.request.teaching_branch_depth} sibling "
                "branch level(s) to fan out"
                if self.request.teaching_branch_depth
                else ""
            )
            + ", demonstrate one testcase info button, click the actual "
            "button-like control that closes its dialog, then press Enter: "
        )
        confirmation = self.manual_confirmation or _console_confirmation
        teacher.activate()
        try:
            await asyncio.wait_for(
                confirmation(prompt),
                timeout=self.request.teaching_timeout_ms / 1000,
            )
            # Let the final pointer binding cross the Playwright transport before
            # freezing the immutable click sequence.
            await asyncio.sleep(0.05)
        except asyncio.TimeoutError as exc:
            raise TeachingError("operator teaching timed out") from exc
        finally:
            teacher.deactivate()
        try:
            guide = compile_taught_guide(
                teacher.clicks,
                branch_depth=self.request.teaching_branch_depth,
            )
            write_crawl_guide(
                output,
                guide,
                overwrite=self.request.overwrite_taught_guide,
            )
        except GuidanceError as exc:
            raise TeachingError(str(exc)) from exc
        self.guide = guide
        self.guide_sha256 = guide_sha256(guide)
        return guide

    async def _finalize_early_failure(
        self,
        store: ArtifactStore,
        error: BaseException,
    ) -> None:
        if store.run.status in {
            ScrapeRunStatus.COMPLETED,
            ScrapeRunStatus.PARTIAL,
            ScrapeRunStatus.FAILED,
            ScrapeRunStatus.CANCELLED,
        }:
            return
        started_at = store.run.started_at or utc_now()
        reason = _safe_error_reason(
            error,
            store.run.capture_policy.redacted_names,
        )
        status = (
            ScrapeRunStatus.CANCELLED
            if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt))
            else ScrapeRunStatus.FAILED
        )
        failed = store.run.model_copy(
            update={
                "status": status,
                "started_at": started_at,
                "ended_at": utc_now(),
                "completion_reason": reason,
            }
        )
        await asyncio.to_thread(store.save_manifest, failed)

    async def _mark_partial_after_shutdown_failure(
        self,
        store: ArtifactStore,
        error: Exception,
    ) -> None:
        reason = _safe_error_reason(
            error,
            store.run.capture_policy.redacted_names,
        )
        started_at = store.run.started_at or utc_now()
        partial = store.run.model_copy(
            update={
                "status": ScrapeRunStatus.PARTIAL,
                "started_at": started_at,
                "ended_at": utc_now(),
                "completion_reason": f"browser shutdown failed: {reason}",
            }
        )
        await asyncio.to_thread(store.save_manifest, partial)
        await asyncio.to_thread(
            store.write_checkpoint,
            partial.checkpoint_sequence + 1,
            last_state_id=partial.state_ids[-1] if partial.state_ids else None,
            last_transition_id=(
                partial.transition_ids[-1] if partial.transition_ids else None
            ),
        )


async def _console_confirmation(prompt: str) -> None:
    loop = asyncio.get_running_loop()
    completed: asyncio.Future[None] = loop.create_future()

    def resolve(error: Optional[BaseException]) -> None:
        if completed.done():
            return
        if error is None:
            completed.set_result(None)
        else:
            completed.set_exception(error)

    def notify(error: Optional[BaseException]) -> None:
        try:
            loop.call_soon_threadsafe(resolve, error)
        except RuntimeError:
            # The CLI may already have exited after a confirmation timeout.
            pass

    def read_confirmation() -> None:
        try:
            input(prompt)
        except BaseException as exc:
            notify(exc)
        else:
            notify(None)

    # A daemon is deliberate: unlike the default asyncio executor, a cancelled
    # blocking stdin read cannot hold process shutdown open after auth timeout.
    threading.Thread(
        target=read_confirmation,
        name="cz-manual-login-confirmation",
        daemon=True,
    ).start()
    await completed


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"crawl-{timestamp}-{uuid.uuid4().hex[:8]}"


def _write_private_file(path: Path, data: bytes, *, overwrite: bool) -> Path:
    destination = path.expanduser().absolute()
    if destination.is_symlink():
        raise StorageStateExportError("storage state output cannot be a symbolic link")
    if destination.exists() and not overwrite:
        raise StorageStateExportError(
            "storage state output already exists; overwrite was not authorized"
        )
    if destination.exists() and not destination.is_file():
        raise StorageStateExportError("storage state output must be a regular file")
    destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise StorageStateExportError(
                    "storage state output already exists; overwrite was not authorized"
                ) from exc
            temporary.unlink()
        os.chmod(destination, 0o600)
        return destination
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _origin_from_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("root_url must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("root_url must not contain embedded credentials")
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("root_url must contain a hostname")
    host = f"[{hostname.lower()}]" if ":" in hostname else hostname.lower()
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme.lower()}://{host}{port}"


def _normalize_allowed_origin(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("allowed origins cannot contain paths, queries, or fragments")
    return _origin_from_url(value)


def _safe_error_reason(error: BaseException, redacted_names: tuple[str, ...]) -> str:
    raw = str(error).strip() or type(error).__name__
    safe = redact_text(raw, redacted_names)
    safe = re.sub(r"[\r\n]+", " ", safe).strip()
    return safe[:2_000] or "crawl failed"


__all__ = [
    "AuthenticationError",
    "AuthenticationMode",
    "CrawlRequest",
    "CrawlRunResult",
    "CrawlRunner",
    "CrawlRunnerError",
    "ExistingRunPolicy",
    "LoginCallback",
    "ManualConfirmation",
    "ResumeUnavailableError",
    "StorageStateExportError",
    "TeachingError",
]
