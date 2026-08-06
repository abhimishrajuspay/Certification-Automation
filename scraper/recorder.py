"""Playwright event recorder for deterministic portal evidence collection."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Awaitable, Optional
from uuid import uuid4

from playwright.async_api import (
    BrowserContext,
    ConsoleMessage,
    Dialog,
    Download,
    Error as PlaywrightError,
    FileChooser,
    Frame,
    Page,
    Request,
    Response,
    WebError,
    WebSocket,
    Worker,
)

from scraper.artifact_store import ArtifactStore, ArtifactStoreError, StorableRecord
from scraper.models import (
    ArtifactKind,
    ArtifactReference,
    BrowserEvent,
    BrowserEventKind,
    NetworkExchange,
    ValueCapture,
    utc_now,
)
from scraper.redaction import capture_mapping, redact_body, redact_text, redact_url


MUTATION_INIT_SCRIPT = r"""
(() => {
    const BUFFER_KEY = '__czEvidenceMutationBuffer';
    const LIMIT = 50000;
    const buffer = [];
    let overflow = 0;

    const safeText = (value) => {
        if (value === null || value === undefined) return null;
        return String(value).slice(0, 2000);
    };

    const selectorHint = (node) => {
        if (!node || node.nodeType !== Node.ELEMENT_NODE) return '';
        const el = node;
        const id = el.id ? `#${el.id}` : '';
        const role = el.getAttribute('role');
        const rolePart = role ? `[role="${role}"]` : '';
        return `${el.tagName.toLowerCase()}${id}${rolePart}`;
    };

    const push = (entry) => {
        const item = { timestamp: Date.now(), ...entry };
        if (buffer.length < LIMIT) buffer.push(item);
        else overflow += 1;
    };

    Object.defineProperty(window, BUFFER_KEY, {
        value: buffer,
        configurable: false,
        enumerable: false,
        writable: false,
    });

    Object.defineProperty(window, '__czEvidenceDrainMutations', {
        value: () => {
            const items = buffer.splice(0, buffer.length);
            if (overflow > 0) {
                items.push({
                    timestamp: Date.now(),
                    type: 'overflow',
                    dropped: overflow,
                });
                overflow = 0;
            }
            return items;
        },
        configurable: false,
        enumerable: false,
        writable: false,
    });

    const observer = new MutationObserver((records) => {
        for (const record of records) {
            const entry = {
                type: record.type,
                target: selectorHint(record.target),
            };
            if (record.type === 'attributes') {
                entry.attribute = record.attributeName || '';
                entry.oldValue = safeText(record.oldValue);
                entry.newValue = safeText(
                    record.target.getAttribute(record.attributeName || '')
                );
            } else if (record.type === 'characterData') {
                entry.oldValue = safeText(record.oldValue);
                entry.newValue = safeText(record.target.data);
            } else if (record.type === 'childList') {
                entry.added = record.addedNodes.length;
                entry.removed = record.removedNodes.length;
                entry.addedSummary = Array.from(record.addedNodes)
                    .slice(0, 20)
                    .map(selectorHint);
                entry.removedSummary = Array.from(record.removedNodes)
                    .slice(0, 20)
                    .map(selectorHint);
            }
            push(entry);
        }
    });

    const observe = () => {
        if (!document.documentElement) return;
        observer.observe(document.documentElement, {
            subtree: true,
            childList: true,
            attributes: true,
            characterData: true,
            attributeOldValue: true,
            characterDataOldValue: true,
        });
    };

    if (document.documentElement) observe();
    else document.addEventListener('DOMContentLoaded', observe, { once: true });

    for (const method of ['pushState', 'replaceState']) {
        const original = history[method].bind(history);
        history[method] = (...args) => {
            const result = original(...args);
            push({
                type: 'history',
                operation: method,
                url: location.href,
            });
            return result;
        };
    }
    addEventListener('popstate', () => {
        push({ type: 'history', operation: 'popstate', url: location.href });
    });
})();
"""


class RecorderError(RuntimeError):
    """Raised when browser evidence cannot be recorded durably."""


class RecorderStateError(RecorderError):
    """Raised when recorder lifecycle or action scopes are misused."""


@dataclass(frozen=True)
class _RequestCapture:
    request_id: str
    requested_at: datetime
    action_id: Optional[str]
    page_id: Optional[str]
    frame_id: Optional[str]
    url: str
    method: str
    resource_type: str
    headers: tuple[ValueCapture, ...]
    body: Optional[ArtifactReference]
    redirected_from_request_id: Optional[str]


class BrowserRecorder:
    """Capture Playwright events and persist them through an ArtifactStore."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        auto_dismiss_dialogs: bool = True,
    ) -> None:
        self.store = store
        self.policy = store.run.capture_policy
        self.auto_dismiss_dialogs = auto_dismiss_dialogs
        self._queue: asyncio.Queue[Optional[StorableRecord]] = asyncio.Queue()
        self._writer_task: Optional[asyncio.Task[None]] = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._errors: list[BaseException] = []
        self._request_tasks: dict[int, asyncio.Task[_RequestCapture]] = {}
        self._attached_pages: set[int] = set()
        self._page_ids: dict[int, str] = {}
        self._frame_ids: dict[int, str] = {}
        self._active_action_id: Optional[str] = None
        self._started = False

    @property
    def active_action_id(self) -> Optional[str]:
        """Return the action currently owning asynchronous browser effects."""

        return self._active_action_id

    async def start(self, context: BrowserContext) -> None:
        """Attach context-wide listeners before the first page navigation."""

        if self._started:
            raise RecorderStateError("recorder is already started")
        self._started = True
        self._writer_task = asyncio.create_task(self._writer_loop())

        context.on("request", self._on_request)
        context.on("response", self._on_response)
        context.on("requestfailed", self._on_request_failed)
        context.on("console", self._on_console)
        context.on("weberror", self._on_web_error)
        context.on("dialog", self._on_dialog)
        context.on("download", self._on_download)
        context.on("frameattached", self._on_frame_attached)
        context.on("framedetached", self._on_frame_detached)
        context.on("framenavigated", self._on_frame_navigated)
        context.on("page", self.attach_page)
        context.on("pageclose", self._on_page_closed)
        context.on("pageload", self._on_page_load)
        context.on("serviceworker", self._on_service_worker)

        for page in context.pages:
            self.attach_page(page)

    async def stop(self) -> None:
        """Flush all pending evidence and stop the ordered writer."""

        if not self._started:
            return
        try:
            await self._drain_pending()
        finally:
            await self._queue.put(None)
            if self._writer_task is not None:
                await self._writer_task
            self._writer_task = None
            self._started = False
        self._raise_background_errors()

    async def flush(self) -> None:
        """Wait until callbacks and queued writes are durably persisted."""

        await self._drain_pending()
        self._raise_background_errors()

    async def _drain_pending(self) -> None:
        while self._tasks:
            tasks = tuple(self._tasks)
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._queue.join()

    @asynccontextmanager
    async def action_scope(self, action_id: str) -> AsyncIterator[None]:
        """Correlate all effects until the caller finishes action settling."""

        if not action_id.strip():
            raise ValueError("action_id cannot be empty")
        if self._active_action_id is not None:
            raise RecorderStateError(
                "parallel or nested action scopes are not supported"
            )
        self._active_action_id = action_id
        try:
            yield
        finally:
            self._active_action_id = None

    def attach_page(self, page: Page) -> None:
        """Attach page-specific listeners once for popups and new tabs."""

        page_key = id(page)
        if page_key in self._attached_pages:
            return
        self._attached_pages.add(page_key)
        page_id = self.page_id(page)
        self._record_event(
            BrowserEventKind.PAGE_OPENED,
            summary="Page opened",
            page_id=page_id,
            details={"url": self._safe_url(page.url)},
        )

        page.on("domcontentloaded", lambda _: self._on_dom_content_loaded(page))
        page.on("popup", lambda popup: self._on_popup(page, popup))
        page.on("websocket", lambda websocket: self._on_websocket(page, websocket))
        page.on("worker", lambda worker: self._on_worker(page, worker))
        page.on("crash", lambda _: self._on_page_crash(page))
        page.on("filechooser", lambda chooser: self._on_file_chooser(page, chooser))

    def page_id(self, page: Page) -> str:
        """Return a stable identifier for a Page object during this run."""

        return self._page_ids.setdefault(id(page), f"page-{uuid4().hex}")

    def frame_id(self, frame: Frame) -> str:
        """Return a stable identifier for a Frame object during this run."""

        return self._frame_ids.setdefault(id(frame), f"frame-{uuid4().hex}")

    async def drain_mutations(self, page: Page) -> int:
        """Drain init-script mutation evidence from every accessible frame."""

        recorded = 0
        for frame in page.frames:
            try:
                mutations = await frame.evaluate(
                    """() => window.__czEvidenceDrainMutations
                        ? window.__czEvidenceDrainMutations()
                        : []"""
                )
            except PlaywrightError:
                continue
            if not isinstance(mutations, list):
                continue
            for mutation in mutations:
                if not isinstance(mutation, dict):
                    continue
                safe_details = {
                    str(name): self._safe_text(str(value))
                    for name, value in mutation.items()
                }
                self._record_event(
                    BrowserEventKind.DOM_MUTATION,
                    summary=f"DOM mutation: {safe_details.get('type', 'unknown')}",
                    page_id=self.page_id(page),
                    frame_id=self.frame_id(frame),
                    details=safe_details,
                )
                recorded += 1
        return recorded

    def record_artifact_event(
        self,
        kind: BrowserEventKind,
        reference: ArtifactReference,
        *,
        summary: str,
    ) -> None:
        """Record a trace/HAR artifact produced by browser shutdown."""

        self._record_event(
            kind,
            summary=summary,
            artifacts=(reference,),
        )

    def _on_request(self, request: Request) -> None:
        task = self._spawn(
            self._prepare_request(
                request,
                requested_at=utc_now(),
                action_id=self._active_action_id,
            )
        )
        self._request_tasks[id(request)] = task

    def _on_response(self, response: Response) -> None:
        self._spawn(self._finalize_response(response))

    def _on_request_failed(self, request: Request) -> None:
        self._spawn(self._finalize_failed_request(request))

    async def _prepare_request(
        self,
        request: Request,
        *,
        requested_at: datetime,
        action_id: Optional[str],
    ) -> _RequestCapture:
        headers = await self._request_headers(request)
        body = await self._capture_request_body(request, headers)
        page_id, frame_id = self._request_context(request)
        redirected_from_request_id = await self._redirected_request_id(request)
        request_id = f"request-{uuid4().hex}"
        safe_url = self._safe_url(request.url)
        capture = _RequestCapture(
            request_id=request_id,
            requested_at=requested_at,
            action_id=action_id,
            page_id=page_id,
            frame_id=frame_id,
            url=safe_url,
            method=request.method,
            resource_type=request.resource_type,
            headers=headers,
            body=body,
            redirected_from_request_id=redirected_from_request_id,
        )
        self._record_event(
            BrowserEventKind.REQUEST,
            summary=f"{request.method} {safe_url}",
            page_id=page_id,
            frame_id=frame_id,
            action_id=action_id,
            details={
                "request_id": request_id,
                "method": request.method,
                "resource_type": request.resource_type,
                "url": safe_url,
            },
            artifacts=(body,) if body else (),
        )
        return capture

    async def _finalize_response(self, response: Response) -> None:
        request = response.request
        capture = await self._request_capture(request)
        try:
            await response.finished()
        except PlaywrightError:
            pass
        headers = await self._response_headers(response)
        body = await self._capture_response_body(response, headers)
        completed_at = utc_now()
        duration_ms = max(
            0,
            int((completed_at - capture.requested_at).total_seconds() * 1000),
        )
        exchange = NetworkExchange(
            exchange_id=f"exchange-{uuid4().hex}",
            run_id=self.store.run.run_id,
            request_id=capture.request_id,
            requested_at=capture.requested_at,
            action_id=capture.action_id,
            page_id=capture.page_id,
            frame_id=capture.frame_id,
            url=capture.url,
            method=capture.method,
            resource_type=capture.resource_type,
            request_headers=capture.headers,
            request_body=capture.body,
            response_status=response.status,
            response_headers=headers,
            response_body=body,
            completed_at=completed_at,
            duration_ms=duration_ms,
            from_service_worker=response.from_service_worker,
            redirected_from_request_id=capture.redirected_from_request_id,
        )
        self._enqueue(exchange)
        self._record_event(
            BrowserEventKind.RESPONSE,
            summary=f"{response.status} {capture.method} {capture.url}",
            page_id=capture.page_id,
            frame_id=capture.frame_id,
            action_id=capture.action_id,
            details={
                "request_id": capture.request_id,
                "status": response.status,
                "status_text": response.status_text,
                "url": capture.url,
            },
            artifacts=(body,) if body else (),
        )
        self._request_tasks.pop(id(request), None)

    async def _finalize_failed_request(self, request: Request) -> None:
        capture = await self._request_capture(request)
        completed_at = utc_now()
        failure = request.failure or "unknown network failure"
        exchange = NetworkExchange(
            exchange_id=f"exchange-{uuid4().hex}",
            run_id=self.store.run.run_id,
            request_id=capture.request_id,
            requested_at=capture.requested_at,
            action_id=capture.action_id,
            page_id=capture.page_id,
            frame_id=capture.frame_id,
            url=capture.url,
            method=capture.method,
            resource_type=capture.resource_type,
            request_headers=capture.headers,
            request_body=capture.body,
            completed_at=completed_at,
            duration_ms=max(
                0,
                int((completed_at - capture.requested_at).total_seconds() * 1000),
            ),
            failure_text=self._safe_text(failure),
            redirected_from_request_id=capture.redirected_from_request_id,
        )
        self._enqueue(exchange)
        self._record_event(
            BrowserEventKind.REQUEST_FAILED,
            summary=f"Request failed: {capture.method} {capture.url}",
            page_id=capture.page_id,
            frame_id=capture.frame_id,
            action_id=capture.action_id,
            details={
                "request_id": capture.request_id,
                "failure": self._safe_text(failure),
                "url": capture.url,
            },
        )
        self._request_tasks.pop(id(request), None)

    async def _request_capture(self, request: Request) -> _RequestCapture:
        task = self._request_tasks.get(id(request))
        if task is None:
            task = self._spawn(
                self._prepare_request(
                    request,
                    requested_at=utc_now(),
                    action_id=self._active_action_id,
                )
            )
            self._request_tasks[id(request)] = task
        return await task

    async def _redirected_request_id(self, request: Request) -> Optional[str]:
        previous = request.redirected_from
        if previous is None:
            return None
        task = self._request_tasks.get(id(previous))
        if task is None:
            return None
        try:
            return (await task).request_id
        except (PlaywrightError, RecorderError):
            return None

    async def _request_headers(self, request: Request) -> tuple[ValueCapture, ...]:
        try:
            headers = await request.all_headers()
        except PlaywrightError:
            headers = request.headers
        return capture_mapping(headers, self.policy.redacted_names)

    async def _response_headers(self, response: Response) -> tuple[ValueCapture, ...]:
        try:
            headers = await response.all_headers()
        except PlaywrightError:
            headers = response.headers
        return capture_mapping(headers, self.policy.redacted_names)

    async def _capture_request_body(
        self,
        request: Request,
        headers: tuple[ValueCapture, ...],
    ) -> Optional[ArtifactReference]:
        if not self.policy.capture_request_bodies:
            return None
        try:
            data = request.post_data_buffer
        except PlaywrightError:
            return None
        return await self._capture_body(
            ArtifactKind.REQUEST_BODY,
            data,
            self._content_type(headers),
        )

    async def _capture_response_body(
        self,
        response: Response,
        headers: tuple[ValueCapture, ...],
    ) -> Optional[ArtifactReference]:
        if not self.policy.capture_response_bodies:
            return None
        try:
            data = await response.body()
        except PlaywrightError:
            return None
        return await self._capture_body(
            ArtifactKind.RESPONSE_BODY,
            data,
            self._content_type(headers),
        )

    async def _capture_body(
        self,
        kind: ArtifactKind,
        data: Optional[bytes],
        media_type: str,
    ) -> Optional[ArtifactReference]:
        if data is None or len(data) > self.policy.maximum_body_bytes:
            return None
        base_media_type = media_type.split(";", 1)[0].strip().lower()
        allowed = {item.lower() for item in self.policy.allowed_body_media_types}
        if base_media_type not in allowed and not any(
            base_media_type.endswith(suffix)
            for suffix in ("+json", "+xml")
            if f"application/{suffix[1:]}" in allowed
        ):
            return None
        redacted_data, changed = redact_body(
            data,
            media_type,
            self.policy.redacted_names,
        )
        return await asyncio.to_thread(
            self.store.put_bytes,
            kind,
            redacted_data,
            media_type=media_type or "application/octet-stream",
            redacted=changed,
        )

    def _on_console(self, message: ConsoleMessage) -> None:
        page = message.page
        self._record_event(
            BrowserEventKind.CONSOLE,
            summary=f"console.{message.type}: {self._safe_text(message.text)}",
            page_id=self.page_id(page) if page else None,
            details={
                "type": message.type,
                "text": self._safe_text(message.text),
                "location": str(message.location),
            },
        )

    def _on_web_error(self, web_error: WebError) -> None:
        page = web_error.page
        self._record_event(
            BrowserEventKind.PAGE_ERROR,
            summary=f"Unhandled page error: {self._safe_text(str(web_error.error))}",
            page_id=self.page_id(page) if page else None,
            details={"error": self._safe_text(str(web_error.error))},
        )

    def _on_dialog(self, dialog: Dialog) -> None:
        page = dialog.page
        self._record_event(
            BrowserEventKind.DIALOG,
            summary=f"{dialog.type} dialog: {self._safe_text(dialog.message)}",
            page_id=self.page_id(page) if page else None,
            details={
                "type": dialog.type,
                "message": self._safe_text(dialog.message),
                "default_value": self._safe_text(dialog.default_value),
                "handled": "dismiss" if self.auto_dismiss_dialogs else "pending",
            },
        )
        if self.auto_dismiss_dialogs:
            self._spawn(self._dismiss_dialog(dialog))

    async def _dismiss_dialog(self, dialog: Dialog) -> None:
        try:
            await dialog.dismiss()
        except PlaywrightError:
            return

    def _on_download(self, download: Download) -> None:
        page = download.page
        self._spawn(self._record_download(download, page))

    async def _record_download(self, download: Download, page: Page) -> None:
        failure = None
        try:
            failure = await download.failure()
        except PlaywrightError:
            failure = "download status unavailable"
        self._record_event(
            BrowserEventKind.DOWNLOAD,
            summary=f"Download started: {self._safe_text(download.suggested_filename)}",
            page_id=self.page_id(page),
            details={
                "url": self._safe_url(download.url),
                "suggested_filename": self._safe_text(download.suggested_filename),
                "failure": self._safe_text(failure or ""),
            },
        )

    def _on_frame_attached(self, frame: Frame) -> None:
        self._record_frame_event(BrowserEventKind.FRAME_ATTACHED, frame)

    def _on_frame_detached(self, frame: Frame) -> None:
        self._record_frame_event(BrowserEventKind.FRAME_DETACHED, frame)

    def _on_frame_navigated(self, frame: Frame) -> None:
        self._record_frame_event(BrowserEventKind.FRAME_NAVIGATED, frame)
        if frame.parent_frame is None:
            self._record_event(
                BrowserEventKind.NAVIGATION,
                summary=f"Page navigated: {self._safe_url(frame.url)}",
                page_id=self.page_id(frame.page),
                frame_id=self.frame_id(frame),
                details={"url": self._safe_url(frame.url)},
            )

    def _record_frame_event(self, kind: BrowserEventKind, frame: Frame) -> None:
        parent = frame.parent_frame
        self._record_event(
            kind,
            summary=f"{kind.value}: {self._safe_url(frame.url)}",
            page_id=self.page_id(frame.page),
            frame_id=self.frame_id(frame),
            details={
                "url": self._safe_url(frame.url),
                "name": self._safe_text(frame.name),
                "parent_frame_id": self.frame_id(parent) if parent else "",
            },
        )

    def _on_page_load(self, page: Page) -> None:
        self._record_event(
            BrowserEventKind.LOAD,
            summary=f"Page load: {self._safe_url(page.url)}",
            page_id=self.page_id(page),
            details={"url": self._safe_url(page.url)},
        )

    def _on_dom_content_loaded(self, page: Page) -> None:
        self._record_event(
            BrowserEventKind.DOM_CONTENT_LOADED,
            summary=f"DOM content loaded: {self._safe_url(page.url)}",
            page_id=self.page_id(page),
            details={"url": self._safe_url(page.url)},
        )

    def _on_page_closed(self, page: Page) -> None:
        self._record_event(
            BrowserEventKind.PAGE_CLOSED,
            summary="Page closed",
            page_id=self.page_id(page),
        )

    def _on_page_crash(self, page: Page) -> None:
        self._record_event(
            BrowserEventKind.PAGE_CRASH,
            summary="Page crashed",
            page_id=self.page_id(page),
        )

    def _on_popup(self, parent: Page, popup: Page) -> None:
        self.attach_page(popup)
        self._record_event(
            BrowserEventKind.POPUP,
            summary=f"Popup opened: {self._safe_url(popup.url)}",
            page_id=self.page_id(popup),
            details={
                "parent_page_id": self.page_id(parent),
                "url": self._safe_url(popup.url),
            },
        )

    def _on_file_chooser(self, page: Page, chooser: FileChooser) -> None:
        self._record_event(
            BrowserEventKind.FILE_CHOOSER,
            summary="File chooser opened",
            page_id=self.page_id(page),
            details={"multiple": chooser.is_multiple()},
        )

    def _on_worker(self, page: Page, worker: Worker) -> None:
        self._record_event(
            BrowserEventKind.WORKER_CREATED,
            summary=f"Worker created: {self._safe_url(worker.url)}",
            page_id=self.page_id(page),
            details={"url": self._safe_url(worker.url), "worker_type": "dedicated"},
        )

    def _on_service_worker(self, worker: Worker) -> None:
        self._record_event(
            BrowserEventKind.WORKER_CREATED,
            summary=f"Service worker created: {self._safe_url(worker.url)}",
            details={"url": self._safe_url(worker.url), "worker_type": "service"},
        )

    def _on_websocket(self, page: Page, websocket: WebSocket) -> None:
        websocket_id = f"websocket-{uuid4().hex}"
        self._record_event(
            BrowserEventKind.WEBSOCKET_OPENED,
            summary=f"WebSocket opened: {self._safe_url(websocket.url)}",
            page_id=self.page_id(page),
            details={
                "websocket_id": websocket_id,
                "url": self._safe_url(websocket.url),
            },
        )
        websocket.on(
            "framesent",
            lambda payload: self._record_websocket_frame(
                BrowserEventKind.WEBSOCKET_FRAME_SENT,
                page,
                websocket_id,
                payload,
            ),
        )
        websocket.on(
            "framereceived",
            lambda payload: self._record_websocket_frame(
                BrowserEventKind.WEBSOCKET_FRAME_RECEIVED,
                page,
                websocket_id,
                payload,
            ),
        )
        websocket.on(
            "close",
            lambda _: self._record_event(
                BrowserEventKind.WEBSOCKET_CLOSED,
                summary="WebSocket closed",
                page_id=self.page_id(page),
                details={"websocket_id": websocket_id},
            ),
        )

    def _record_websocket_frame(
        self,
        kind: BrowserEventKind,
        page: Page,
        websocket_id: str,
        payload: str | bytes,
    ) -> None:
        if isinstance(payload, bytes):
            payload_type = "binary"
            preview = f"{len(payload)} bytes"
            byte_length = len(payload)
        else:
            payload_type = "text"
            preview = self._safe_text(payload[:2000])
            byte_length = len(payload.encode("utf-8"))
        self._record_event(
            kind,
            summary=f"{kind.value}: {preview}",
            page_id=self.page_id(page),
            details={
                "websocket_id": websocket_id,
                "payload_type": payload_type,
                "byte_length": byte_length,
                "preview": preview,
            },
        )

    def _request_context(self, request: Request) -> tuple[Optional[str], Optional[str]]:
        try:
            frame = request.frame
            return self.page_id(frame.page), self.frame_id(frame)
        except PlaywrightError:
            return None, None

    def _record_event(
        self,
        kind: BrowserEventKind,
        *,
        summary: str,
        page_id: Optional[str] = None,
        frame_id: Optional[str] = None,
        action_id: Optional[str] = None,
        details: Optional[dict[str, object]] = None,
        artifacts: tuple[ArtifactReference, ...] = (),
    ) -> None:
        event = BrowserEvent(
            event_id=f"event-{uuid4().hex}",
            run_id=self.store.run.run_id,
            kind=kind,
            page_id=page_id,
            frame_id=frame_id,
            action_id=action_id if action_id is not None else self._active_action_id,
            summary=self._safe_text(summary),
            details=capture_mapping(details or {}, self.policy.redacted_names),
            artifacts=artifacts,
        )
        self._enqueue(event)

    def _enqueue(self, record: StorableRecord) -> None:
        if not self._started:
            raise RecorderStateError("recorder is not started")
        self._queue.put_nowait(record)

    async def _writer_loop(self) -> None:
        while True:
            record = await self._queue.get()
            try:
                if record is None:
                    return
                await asyncio.to_thread(self.store.append_record, record)
            except (ArtifactStoreError, OSError, ValueError) as exc:
                self._errors.append(exc)
            finally:
                self._queue.task_done()

    def _spawn(self, awaitable: Awaitable[Any]) -> asyncio.Task[Any]:
        task = asyncio.create_task(awaitable)
        self._tasks.add(task)
        task.add_done_callback(self._task_finished)
        return task

    def _task_finished(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            self._errors.append(RecorderError("browser recorder task was cancelled"))
            return
        exception = task.exception()
        if exception is not None:
            self._errors.append(exception)

    def _raise_background_errors(self) -> None:
        if not self._errors:
            return
        errors = tuple(self._errors)
        self._errors.clear()
        message = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        raise RecorderError(f"browser evidence recording failed: {message}")

    def _safe_url(self, url: str) -> str:
        return redact_url(url, self.policy.redacted_names)

    def _safe_text(self, text: str) -> str:
        return redact_text(text, self.policy.redacted_names)

    @staticmethod
    def _content_type(headers: tuple[ValueCapture, ...]) -> str:
        for header in headers:
            if header.name.lower() == "content-type" and header.value:
                return header.value
        return "application/octet-stream"


__all__ = [
    "MUTATION_INIT_SCRIPT",
    "BrowserRecorder",
    "RecorderError",
    "RecorderStateError",
]
