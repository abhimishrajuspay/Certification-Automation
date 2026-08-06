"""Instrumented Playwright browser lifecycle for deterministic crawl runs."""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Error as PlaywrightError,
    Page,
    Playwright,
    async_playwright,
)

from scraper.artifact_store import ArtifactStore
from scraper.models import ArtifactKind, ArtifactReference, BrowserEventKind
from scraper.recorder import BrowserRecorder, MUTATION_INIT_SCRIPT, RecorderError
from scraper.redaction import redact_har


BrowserName = Literal["chromium", "firefox", "webkit"]
WaitUntil = Literal["commit", "domcontentloaded", "load", "networkidle"]


class BrowserLifecycleError(RuntimeError):
    """Raised when the instrumented browser cannot start or stop safely."""


@dataclass(frozen=True)
class BrowserLaunchConfig:
    """Runtime-only browser configuration; secrets are deliberately excluded."""

    browser_name: BrowserName = "chromium"
    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080
    device_scale_factor: float = 1.0
    ignore_https_errors: bool = False
    accept_downloads: bool = True
    user_agent: Optional[str] = None
    locale: Optional[str] = None
    timezone_id: Optional[str] = None
    storage_state_path: Optional[Path] = None
    launch_args: tuple[str, ...] = ()
    default_timeout_ms: int = 30_000

    def __post_init__(self) -> None:
        if self.viewport_width <= 0 or self.viewport_height <= 0:
            raise ValueError("viewport dimensions must be positive")
        if self.device_scale_factor <= 0:
            raise ValueError("device_scale_factor must be positive")
        if self.default_timeout_ms <= 0:
            raise ValueError("default_timeout_ms must be positive")
        if (
            self.storage_state_path is not None
            and not self.storage_state_path.is_file()
        ):
            raise ValueError("storage_state_path must reference an existing file")


class BrowserManager:
    """Own Playwright resources and guarantee recording precedes navigation."""

    def __init__(
        self,
        store: ArtifactStore,
        config: Optional[BrowserLaunchConfig] = None,
        recorder: Optional[BrowserRecorder] = None,
    ) -> None:
        self.store = store
        self.config = config or BrowserLaunchConfig()
        self.recorder = recorder or BrowserRecorder(store)
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._temporary_directory: Optional[tempfile.TemporaryDirectory[str]] = None
        self._trace_path: Optional[Path] = None
        self._har_path: Optional[Path] = None
        self._trace_started = False

    @property
    def context(self) -> BrowserContext:
        if self._context is None:
            raise BrowserLifecycleError("browser context is not started")
        return self._context

    @property
    def page(self) -> Page:
        if self._page is None:
            raise BrowserLifecycleError("browser page is not started")
        return self._page

    async def __aenter__(self) -> "BrowserManager":
        await self.start()
        return self

    async def __aexit__(
        self,
        exception_type: Optional[type[BaseException]],
        exception: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        await self.stop()

    async def start(self) -> Page:
        """Launch a fully instrumented blank page without navigating it."""

        if self._page is not None:
            return self._page
        try:
            self._playwright = await async_playwright().start()
            browser_type = getattr(self._playwright, self.config.browser_name)
            self._browser = await browser_type.launch(
                headless=self.config.headless,
                args=list(self.config.launch_args),
            )

            self._temporary_directory = tempfile.TemporaryDirectory(
                prefix=f"cz-crawl-{self.store.run.run_id}-"
            )
            temporary_root = Path(self._temporary_directory.name)
            context_options = self._context_options(temporary_root)
            self._context = await self._browser.new_context(**context_options)

            await self._context.add_init_script(script=MUTATION_INIT_SCRIPT)
            if self.store.run.capture_policy.capture_trace:
                await self._context.tracing.start(
                    screenshots=True,
                    snapshots=True,
                    sources=True,
                )
                self._trace_started = True

            await self.recorder.start(self._context)
            self._page = await self._context.new_page()
            self._page.set_default_timeout(self.config.default_timeout_ms)
            return self._page
        except Exception as exc:
            await self._cleanup_failed_start()
            raise BrowserLifecycleError(
                f"failed to start instrumented browser: {exc}"
            ) from exc

    async def navigate(
        self,
        url: str,
        *,
        wait_until: WaitUntil = "domcontentloaded",
        timeout_ms: Optional[int] = None,
    ) -> Page:
        """Navigate only after all context and page listeners are attached."""

        page = self.page
        await page.goto(
            url,
            wait_until=wait_until,
            timeout=timeout_ms or self.config.default_timeout_ms,
        )
        return page

    async def stop(self) -> None:
        """Drain evidence, ingest trace/HAR files, and release browser resources."""

        errors: list[str] = []
        context = self._context

        if context is not None:
            for page in tuple(context.pages):
                try:
                    await self.recorder.drain_mutations(page)
                except (PlaywrightError, RecorderError) as exc:
                    errors.append(f"mutation drain failed: {exc}")

            if self._trace_started and self._trace_path is not None:
                try:
                    await context.tracing.stop(path=self._trace_path)
                    reference = await self._ingest_file(
                        self._trace_path,
                        ArtifactKind.TRACE,
                        "application/zip",
                    )
                    self.recorder.record_artifact_event(
                        BrowserEventKind.TRACE_SAVED,
                        reference,
                        summary="Playwright trace saved",
                    )
                except (PlaywrightError, OSError, ValueError) as exc:
                    errors.append(f"trace capture failed: {exc}")
                finally:
                    self._trace_started = False

            try:
                await context.close()
            except PlaywrightError as exc:
                errors.append(f"context close failed: {exc}")

            if self._har_path is not None and self._har_path.is_file():
                try:
                    data = await asyncio.to_thread(self._har_path.read_bytes)
                    if len(data) > self.store.run.limits.maximum_artifact_bytes:
                        raise ValueError("HAR exceeds maximum artifact size")
                    redacted_data, changed = redact_har(
                        data,
                        self.store.run.capture_policy.redacted_names,
                    )
                    reference = await asyncio.to_thread(
                        self.store.put_bytes,
                        ArtifactKind.HAR,
                        redacted_data,
                        media_type="application/json",
                        redacted=changed,
                    )
                    self.recorder.record_artifact_event(
                        BrowserEventKind.HAR_SAVED,
                        reference,
                        summary="Redacted Playwright HAR saved",
                    )
                except (OSError, ValueError) as exc:
                    errors.append(f"HAR capture failed: {exc}")

        try:
            await self.recorder.stop()
        except RecorderError as exc:
            errors.append(str(exc))

        if self._browser is not None:
            try:
                await self._browser.close()
            except PlaywrightError as exc:
                errors.append(f"browser close failed: {exc}")
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except PlaywrightError as exc:
                errors.append(f"Playwright stop failed: {exc}")
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()

        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._temporary_directory = None
        self._trace_path = None
        self._har_path = None

        if errors:
            raise BrowserLifecycleError("; ".join(errors))

    def _context_options(self, temporary_root: Path) -> dict[str, Any]:
        options: dict[str, Any] = {
            "viewport": {
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            },
            "device_scale_factor": self.config.device_scale_factor,
            "ignore_https_errors": self.config.ignore_https_errors,
            "accept_downloads": self.config.accept_downloads,
        }
        if self.config.user_agent:
            options["user_agent"] = self.config.user_agent
        if self.config.locale:
            options["locale"] = self.config.locale
        if self.config.timezone_id:
            options["timezone_id"] = self.config.timezone_id
        if self.config.storage_state_path:
            options["storage_state"] = str(self.config.storage_state_path)
        if self.store.run.capture_policy.capture_har:
            self._har_path = temporary_root / "capture.har"
            options.update(
                {
                    "record_har_path": str(self._har_path),
                    "record_har_content": "embed",
                    "record_har_mode": "full",
                }
            )
        if self.store.run.capture_policy.capture_trace:
            self._trace_path = temporary_root / "trace.zip"
        return options

    async def _ingest_file(
        self,
        path: Path,
        kind: ArtifactKind,
        media_type: str,
    ) -> ArtifactReference:
        data = await asyncio.to_thread(path.read_bytes)
        if len(data) > self.store.run.limits.maximum_artifact_bytes:
            raise ValueError(f"{kind.value} exceeds maximum artifact size")
        return await asyncio.to_thread(
            self.store.put_bytes,
            kind,
            data,
            media_type=media_type,
            redacted=False,
        )

    async def _cleanup_failed_start(self) -> None:
        if self._context is not None:
            try:
                await self._context.close()
            except PlaywrightError:
                pass
        try:
            await self.recorder.stop()
        except RecorderError:
            pass
        if self._browser is not None:
            try:
                await self._browser.close()
            except PlaywrightError:
                pass
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except PlaywrightError:
                pass
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._temporary_directory = None


__all__ = [
    "BrowserLaunchConfig",
    "BrowserLifecycleError",
    "BrowserManager",
    "BrowserName",
    "WaitUntil",
]
