"""Deterministic page-state extraction built on the instrumented browser.

The extractor does not interact with the portal. It waits for a bounded quiet
window, reads every render-relevant DOM element (including open shadow roots and
child frames), converts the raw browser data into immutable evidence models,
and commits the state only after all referenced artifacts are durable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlsplit

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Frame, Page

from scraper.artifact_store import AppendReceipt, ArtifactStore
from scraper.models import (
    ArtifactKind,
    ArtifactReference,
    BoundingBox,
    ElementContext,
    ElementSnapshot,
    FrameElementCollection,
    FrameSnapshot,
    LocatorCandidate,
    LocatorStrategy,
    ScrollPosition,
    SelectOptionSnapshot,
    StateSnapshot,
    ValueCapture,
    Viewport,
)
from scraper.recorder import BrowserRecorder, RecorderError
from scraper.redaction import (
    hash_text,
    is_sensitive_name,
    redact_text,
    redact_url,
)


FRAME_EXTRACTION_SCRIPT = r"""
(config) => {
    const REDACTED = '[REDACTED]';
    const excludedTags = new Set([
        'base', 'head', 'link', 'meta', 'noscript', 'script', 'style',
        'template', 'title'
    ]);
    const normalized = (value) => String(value || '')
        .replace(/\s+/g, ' ')
        .trim();
    const normalizedName = (value) => String(value || '')
        .toLowerCase()
        .replace(/[^a-z0-9]/g, '');
    const sensitiveName = (value) => {
        const name = normalizedName(value);
        return config.redactedNames.some((marker) => {
            const candidate = normalizedName(marker);
            return candidate && name.includes(candidate);
        });
    };
    const stableId = (value) => {
        if (!value || value.length > 128) return false;
        return !(/[0-9]{6,}/.test(value)
            || /[a-f0-9]{12,}/i.test(value)
            || /^[a-f0-9-]{32,}$/i.test(value));
    };
    const cssEscape = (value) => {
        if (window.CSS && CSS.escape) return CSS.escape(value);
        return String(value).replace(/[^a-zA-Z0-9_-]/g, (char) => `\\${char}`);
    };
    const cssPath = (element) => {
        if (!element || element.nodeType !== Node.ELEMENT_NODE) return '';
        if (stableId(element.id)) return `#${cssEscape(element.id)}`;
        const parts = [];
        let current = element;
        while (current && current.nodeType === Node.ELEMENT_NODE) {
            let part = current.tagName.toLowerCase();
            const parent = current.parentElement;
            if (parent) {
                const siblings = Array.from(parent.children)
                    .filter((candidate) => candidate.tagName === current.tagName);
                if (siblings.length > 1) {
                    part += `:nth-of-type(${siblings.indexOf(current) + 1})`;
                }
            }
            parts.unshift(part);
            if (!parent || current === document.documentElement) break;
            current = parent;
        }
        return parts.join(' > ');
    };
    const visible = (element) => {
        const style = getComputedStyle(element);
        if (style.display === 'none'
            || style.visibility === 'hidden'
            || style.visibility === 'collapse'
            || Number(style.opacity) === 0) return false;
        const rect = element.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0 && element.getClientRects().length > 0;
    };
    const implicitRole = (element) => {
        const tag = element.tagName.toLowerCase();
        const type = String(element.getAttribute('type') || '').toLowerCase();
        if ((tag === 'a' || tag === 'area') && element.hasAttribute('href')) return 'link';
        if (tag === 'button') return 'button';
        if (tag === 'summary') return 'button';
        if (tag === 'textarea') return 'textbox';
        if (tag === 'select') return element.multiple || element.size > 1 ? 'listbox' : 'combobox';
        if (tag === 'img' && element.getAttribute('alt') !== '') return 'img';
        if (/^h[1-6]$/.test(tag)) return 'heading';
        if (tag === 'table') return 'table';
        if (tag === 'thead' || tag === 'tbody' || tag === 'tfoot') return 'rowgroup';
        if (tag === 'tr') return 'row';
        if (tag === 'th') return 'columnheader';
        if (tag === 'td') return 'cell';
        if (tag === 'ul' || tag === 'ol') return 'list';
        if (tag === 'li') return 'listitem';
        if (tag === 'nav') return 'navigation';
        if (tag === 'main') return 'main';
        if (tag === 'form' && (element.getAttribute('aria-label') || element.getAttribute('aria-labelledby'))) return 'form';
        if (tag === 'dialog') return 'dialog';
        if (tag === 'progress') return 'progressbar';
        if (tag !== 'input') return null;
        if (['button', 'image', 'reset', 'submit'].includes(type)) return 'button';
        if (type === 'checkbox') return 'checkbox';
        if (type === 'radio') return 'radio';
        if (type === 'range') return 'slider';
        if (type === 'number') return 'spinbutton';
        if (type === 'search') return 'searchbox';
        if (type === 'hidden') return null;
        return 'textbox';
    };
    const role = (element) => {
        const explicit = normalized(element.getAttribute('role')).split(' ')[0];
        return explicit || implicitRole(element);
    };
    const referencedText = (element, attribute) => {
        const root = element.getRootNode();
        const ids = normalized(element.getAttribute(attribute)).split(' ').filter(Boolean);
        return normalized(ids.map((id) => {
            const candidate = root.getElementById ? root.getElementById(id) : document.getElementById(id);
            return candidate ? candidate.textContent : '';
        }).join(' '));
    };
    const labelText = (element) => {
        if (element.labels && element.labels.length) {
            return normalized(Array.from(element.labels).map((label) => label.innerText || label.textContent).join(' '));
        }
        const parentLabel = element.closest ? element.closest('label') : null;
        return parentLabel ? normalized(parentLabel.innerText || parentLabel.textContent) : '';
    };
    const accessibleName = (element, elementRole) => {
        const labelled = referencedText(element, 'aria-labelledby');
        if (labelled) return labelled;
        const ariaLabel = normalized(element.getAttribute('aria-label'));
        if (ariaLabel) return ariaLabel;
        const label = labelText(element);
        if (label) return label;
        const alt = normalized(element.getAttribute('alt'));
        if (alt) return alt;
        if (element.tagName.toLowerCase() === 'input'
            && ['button', 'image', 'reset', 'submit'].includes(String(element.type).toLowerCase())) {
            return normalized(element.value);
        }
        if (['button', 'link', 'heading', 'columnheader', 'rowheader'].includes(elementRole)) {
            const text = normalized(element.innerText || element.textContent);
            if (text) return text;
        }
        const title = normalized(element.getAttribute('title'));
        if (title) return title;
        return normalized(element.getAttribute('placeholder'));
    };
    const sectionHeading = (element) => {
        let current = element.parentElement;
        while (current && current !== document.body) {
            const labelled = referencedText(current, 'aria-labelledby');
            if (labelled) return labelled;
            const heading = current.querySelector(':scope > h1, :scope > h2, :scope > h3, :scope > h4, :scope > h5, :scope > h6, :scope > legend');
            if (heading) return normalized(heading.innerText || heading.textContent);
            current = current.parentElement;
        }
        return '';
    };
    const ancestorSummary = (element) => {
        const result = [];
        let current = element.parentElement;
        while (current && result.length < 6) {
            let item = current.tagName.toLowerCase();
            if (stableId(current.id)) item += `#${current.id}`;
            const currentRole = role(current);
            if (currentRole) item += `[role=${currentRole}]`;
            result.push(item);
            current = current.parentElement;
        }
        return result;
    };
    const currentValue = (element) => {
        const tag = element.tagName.toLowerCase();
        if (tag === 'textarea' || tag === 'select') return String(element.value || '');
        if (tag === 'input') {
            const type = String(element.type || '').toLowerCase();
            if (['button', 'image', 'reset', 'submit'].includes(type)) return null;
            return String(element.value || '');
        }
        if (element.isContentEditable) return String(element.textContent || '');
        return null;
    };
    const context = (element) => {
        const form = element.form || (element.closest ? element.closest('form') : null);
        const table = element.closest ? element.closest('table') : null;
        const row = element.closest ? element.closest('tr') : null;
        const rowHeading = row ? row.querySelector('th[scope=row], th, td') : null;
        const caption = table ? table.querySelector(':scope > caption') : null;
        return {
            sectionHeading: sectionHeading(element),
            formId: form ? (form.id || form.getAttribute('name') || form.getAttribute('action') || '') : '',
            tableId: table ? (table.id || normalized(caption ? caption.textContent : '')) : '',
            rowLabel: rowHeading ? normalized(rowHeading.innerText || rowHeading.textContent) : '',
            ancestors: ancestorSummary(element),
        };
    };
    const elementRecord = (element, shadowHosts) => {
        const rect = element.getBoundingClientRect();
        const attributes = {};
        for (const attribute of Array.from(element.attributes)) attributes[attribute.name] = attribute.value;
        const elementRole = role(element);
        const rawText = normalized(element.innerText || element.textContent);
        const textTruncated = rawText.length > config.maximumTextChars;
        const options = element.tagName.toLowerCase() === 'select'
            ? Array.from(element.options).map((option) => ({
                label: normalized(option.label || option.textContent),
                value: String(option.value || ''),
                selected: Boolean(option.selected),
                disabled: Boolean(option.disabled),
            }))
            : [];
        const disabled = Boolean(element.disabled)
            || element.getAttribute('aria-disabled') === 'true'
            || Boolean(element.closest && element.closest('[inert]'));
        const readOnly = typeof element.readOnly === 'boolean'
            ? element.readOnly
            : element.getAttribute('aria-readonly') === 'true';
        const editable = !disabled && !readOnly && (
            element.isContentEditable
            || element.tagName.toLowerCase() === 'textarea'
            || element.tagName.toLowerCase() === 'select'
            || (element.tagName.toLowerCase() === 'input'
                && !['button', 'checkbox', 'file', 'hidden', 'image', 'radio', 'reset', 'submit']
                    .includes(String(element.type || '').toLowerCase()))
        );
        const interactionSignals = [];
        const nativeSelector = 'button, input, select, textarea, summary, a[href], area[href]';
        if (element.matches(nativeSelector)) interactionSignals.push('native-control');
        if (elementRole && [
            'button', 'checkbox', 'combobox', 'link', 'listbox', 'menuitem',
            'option', 'radio', 'searchbox', 'slider', 'spinbutton', 'switch',
            'tab', 'textbox', 'treeitem'
        ].includes(elementRole)) interactionSignals.push(`role:${elementRole}`);
        if (element.isContentEditable) interactionSignals.push('contenteditable');
        if (element.hasAttribute('tabindex')) interactionSignals.push(`tabindex:${element.getAttribute('tabindex')}`);
        if (element.matches('[draggable=true]')) interactionSignals.push('draggable');
        if (getComputedStyle(element).cursor === 'pointer') interactionSignals.push('cursor:pointer');
        for (const attribute of Array.from(element.attributes)) {
            if (attribute.name.toLowerCase().startsWith('on')) {
                interactionSignals.push(`inline:${attribute.name.slice(2).toLowerCase()}`);
            }
        }
        const listenerTypes = window.__czEvidenceEventTypesFor
            ? window.__czEvidenceEventTypesFor(element)
            : [];
        for (const eventType of listenerTypes) interactionSignals.push(`listener:${eventType}`);
        const interactive = interactionSignals.length > 0;
        const ariaChecked = element.getAttribute('aria-checked');
        const ariaSelected = element.getAttribute('aria-selected');
        const ariaExpanded = element.getAttribute('aria-expanded');
        return {
            domPath: cssPath(element),
            parentDomPath: cssPath(
                element.parentElement
                || (element.getRootNode && element.getRootNode().host)
            ),
            shadowHostPath: shadowHosts,
            tag: element.tagName.toLowerCase(),
            role: elementRole,
            accessibleName: accessibleName(element, elementRole),
            label: labelText(element),
            text: rawText.slice(0, config.maximumTextChars),
            textTruncated,
            title: normalized(element.getAttribute('title')),
            inputType: element.tagName.toLowerCase() === 'input'
                ? String(element.type || '').toLowerCase()
                : (element.tagName.toLowerCase() === 'button'
                    ? String(
                        element.getAttribute('type')
                        || (element.closest('form') ? 'submit' : 'button')
                    ).toLowerCase()
                    : ''),
            attributes,
            value: currentValue(element),
            visible: visible(element),
            enabled: !disabled,
            editable,
            checked: typeof element.checked === 'boolean' ? element.checked : (ariaChecked ? ariaChecked === 'true' : null),
            selected: typeof element.selected === 'boolean' ? element.selected : (ariaSelected ? ariaSelected === 'true' : null),
            expanded: ariaExpanded ? ariaExpanded === 'true' : null,
            readOnly: typeof readOnly === 'boolean' ? readOnly : null,
            boundingBox: {x: rect.x, y: rect.y, width: rect.width, height: rect.height},
            options,
            context: context(element),
            interactive,
            interactionSignals: Array.from(new Set(interactionSignals)).sort(),
        };
    };
    const records = [];
    let eligibleCount = 0;
    let closedShadowHosts = 0;
    const visit = (root, shadowHosts) => {
        for (const element of Array.from(root.querySelectorAll('*'))) {
            const tag = element.tagName.toLowerCase();
            if (!excludedTags.has(tag)) {
                eligibleCount += 1;
                if (records.length < config.maximumElements) records.push(elementRecord(element, shadowHosts));
            }
            if (element.shadowRoot) visit(element.shadowRoot, [...shadowHosts, cssPath(element)]);
            else if (element.hasAttribute && element.hasAttribute('data-closed-shadow-root')) closedShadowHosts += 1;
        }
    };
    visit(document, []);

    const activeDescriptor = () => {
        let active = document.activeElement;
        const hosts = [];
        while (active && active.shadowRoot && active.shadowRoot.activeElement) {
            hosts.push(cssPath(active));
            active = active.shadowRoot.activeElement;
        }
        if (!active || active === document.body || active === document.documentElement) return null;
        return {domPath: cssPath(active), shadowHostPath: hosts};
    };
    const visibleMatches = (selector) => Array.from(document.querySelectorAll(selector)).filter(visible);
    const describeMatches = (selector) => visibleMatches(selector).map((element) => ({
        path: cssPath(element),
        text: normalized(element.innerText || element.textContent).slice(0, config.maximumTextChars),
    }));
    const sanitizeDom = () => {
        const root = document.documentElement;
        if (!root) return {html: '', changed: false};
        const clone = root.cloneNode(true);
        let changed = false;
        for (const input of Array.from(clone.querySelectorAll('input'))) {
            const type = String(input.getAttribute('type') || 'text').toLowerCase();
            if (!['button', 'image', 'reset', 'submit'].includes(type) && input.hasAttribute('value')) {
                input.setAttribute('value', REDACTED);
                changed = true;
            }
        }
        for (const textarea of Array.from(clone.querySelectorAll('textarea'))) {
            if (textarea.textContent) {
                textarea.textContent = REDACTED;
                changed = true;
            }
        }
        for (const editable of Array.from(clone.querySelectorAll('[contenteditable]:not([contenteditable=false])'))) {
            if (editable.textContent) {
                editable.textContent = REDACTED;
                changed = true;
            }
        }
        for (const script of Array.from(clone.querySelectorAll('script'))) {
            if (script.textContent) {
                script.textContent = '[SCRIPT CONTENT OMITTED FROM SANITIZED DOM]';
                changed = true;
            }
        }
        for (const meta of Array.from(clone.querySelectorAll('meta[content]'))) {
            const semanticName = meta.getAttribute('name')
                || meta.getAttribute('http-equiv')
                || meta.getAttribute('property')
                || '';
            if (sensitiveName(semanticName)) {
                meta.setAttribute('content', REDACTED);
                changed = true;
            }
        }
        for (const element of Array.from(clone.querySelectorAll('*'))) {
            for (const attribute of Array.from(element.attributes)) {
                if (sensitiveName(attribute.name)) {
                    element.setAttribute(attribute.name, REDACTED);
                    changed = true;
                }
            }
        }
        const doctype = document.doctype ? `<!DOCTYPE ${document.doctype.name}>` : '';
        return {html: doctype + clone.outerHTML, changed};
    };
    const dom = sanitizeDom();
    let sessionStorageEntries = [];
    try {
        sessionStorageEntries = Object.entries(sessionStorage);
    } catch (_) {
        sessionStorageEntries = [];
    }
    return {
        dom: dom.html,
        domRedacted: dom.changed,
        elements: records,
        eligibleCount,
        truncated: eligibleCount > records.length,
        closedShadowHosts,
        activeElement: activeDescriptor(),
        modalCount: visibleMatches('dialog[open], [role=dialog], [aria-modal=true]').length,
        loadingIndicators: describeMatches('[aria-busy=true], progress, [role=progressbar], .loading, .spinner, [class*=loading], [class*=spinner]'),
        notifications: describeMatches('[role=alert], [role=status], [aria-live]'),
        errors: describeMatches('[aria-invalid=true], [role=alert], .error, .errors, [class*=error]'),
        scroll: {
            x: Math.max(0, window.scrollX || 0),
            y: Math.max(0, window.scrollY || 0),
            maximumX: Math.max(0, Math.max(document.documentElement.scrollWidth, document.body ? document.body.scrollWidth : 0) - window.innerWidth),
            maximumY: Math.max(0, Math.max(document.documentElement.scrollHeight, document.body ? document.body.scrollHeight : 0) - window.innerHeight),
        },
        viewport: {
            width: window.innerWidth,
            height: window.innerHeight,
            deviceScaleFactor: window.devicePixelRatio || 1,
        },
        sessionStorage: sessionStorageEntries,
    };
}
"""


QUIET_SIGNATURE_SCRIPT = r"""
() => ({
    readyState: document.readyState,
    mutationVersion: window.__czEvidenceMutationVersion
        ? window.__czEvidenceMutationVersion()
        : -1,
    elementCount: document.getElementsByTagName('*').length,
    scrollWidth: document.documentElement ? document.documentElement.scrollWidth : 0,
    scrollHeight: document.documentElement ? document.documentElement.scrollHeight : 0,
})
"""


class SnapshotExtractionError(RuntimeError):
    """Raised when a page state cannot be captured durably."""


@dataclass(frozen=True)
class SnapshotConfig:
    """Bounded extraction settings independent of portal-specific selectors."""

    maximum_elements_per_frame: int = 50_000
    maximum_text_chars: int = 4_000
    quiet_window_ms: int = 400
    quiet_timeout_ms: int = 5_000
    quiet_poll_interval_ms: int = 100
    full_page_screenshot: bool = True

    def __post_init__(self) -> None:
        if self.maximum_elements_per_frame <= 0:
            raise ValueError("maximum_elements_per_frame must be positive")
        if self.maximum_text_chars <= 0:
            raise ValueError("maximum_text_chars must be positive")
        if self.quiet_window_ms < 0:
            raise ValueError("quiet_window_ms cannot be negative")
        if self.quiet_timeout_ms <= 0:
            raise ValueError("quiet_timeout_ms must be positive")
        if self.quiet_poll_interval_ms <= 0:
            raise ValueError("quiet_poll_interval_ms must be positive")


@dataclass(frozen=True)
class CapturedState:
    """In-memory capture result plus its durable append receipt."""

    state: StateSnapshot
    elements: tuple[ElementSnapshot, ...]
    receipt: AppendReceipt


@dataclass(frozen=True)
class _RawFrame:
    frame: Frame
    frame_key: str
    frame_id: str
    parent_frame_id: Optional[str]
    child_frame_ids: tuple[str, ...]
    data: dict[str, Any]


class PageStateExtractor:
    """Extract and persist an immutable state without performing interactions."""

    def __init__(
        self,
        store: ArtifactStore,
        recorder: Optional[BrowserRecorder] = None,
        config: Optional[SnapshotConfig] = None,
    ) -> None:
        self.store = store
        self.recorder = recorder
        self.config = config or SnapshotConfig()
        self.policy = store.run.capture_policy

    async def capture(self, page: Page, *, sequence: int) -> CapturedState:
        """Capture artifacts first, then append their immutable state envelope."""

        if sequence < 0:
            raise ValueError("sequence cannot be negative")
        limitations: list[str] = []
        stable = await self.wait_for_quiet(page)
        if not stable:
            limitations.append("page did not reach the configured DOM quiet window")

        if self.recorder is not None:
            try:
                await self.recorder.drain_mutations(page)
                await self.recorder.flush()
            except RecorderError as exc:
                raise SnapshotExtractionError(
                    f"failed to flush browser evidence before snapshot: {exc}"
                ) from exc

        raw_frames = await self._extract_frames(page)
        frame_snapshots: list[FrameSnapshot] = []
        all_elements: list[ElementSnapshot] = []
        active_element_id: Optional[str] = None
        modal_count = 0
        loading_indicators: list[str] = []
        notifications: list[str] = []
        errors: list[str] = []

        main_origin = _origin(page.url)
        for raw_frame in raw_frames:
            frame_snapshot, elements, active = await self._materialize_frame(
                raw_frame,
                main_origin=main_origin,
            )
            frame_snapshots.append(frame_snapshot)
            all_elements.extend(elements)
            if active is not None:
                active_element_id = active
            modal_count += _nonnegative_int(raw_frame.data.get("modalCount"))
            loading_indicators.extend(
                self._descriptions(raw_frame, "loadingIndicators")
            )
            notifications.extend(self._descriptions(raw_frame, "notifications"))
            errors.extend(self._descriptions(raw_frame, "errors"))

        artifacts: list[ArtifactReference] = []
        screenshot = await self._capture_screenshot(page, limitations)
        if screenshot is not None:
            artifacts.append(screenshot)
        storage_reference, storage_fingerprint = await self._capture_storage(
            page,
            raw_frames,
            limitations,
        )
        if storage_reference is not None:
            artifacts.append(storage_reference)

        main_data = raw_frames[0].data if raw_frames else {}
        viewport = self._viewport(main_data)
        scroll = self._scroll(main_data)
        safe_url = redact_url(page.url, self.policy.redacted_names)
        safe_title = redact_text(await page.title(), self.policy.redacted_names)
        page_id = (
            self.recorder.page_id(page)
            if self.recorder is not None
            else f"page-{hash_text(safe_url)[:24]}"
        )
        fingerprint = self._fingerprint(
            url=safe_url,
            title=safe_title,
            viewport=viewport,
            scroll=scroll,
            frames=raw_frames,
            elements=all_elements,
            active_element_id=active_element_id,
            storage_fingerprint=storage_fingerprint,
            stable=stable,
        )
        state = StateSnapshot(
            state_id=f"state-{fingerprint[:32]}",
            run_id=self.store.run.run_id,
            sequence=sequence,
            fingerprint=fingerprint,
            page_id=page_id,
            url=safe_url,
            title=safe_title,
            viewport=viewport,
            scroll=scroll,
            frames=tuple(frame_snapshots),
            element_ids=tuple(element.element_id for element in all_elements),
            active_element_id=active_element_id,
            artifacts=tuple(artifacts),
            modal_count=modal_count,
            loading_indicators=tuple(sorted(set(loading_indicators))),
            notifications=tuple(sorted(set(notifications))),
            errors=tuple(sorted(set(errors))),
            storage_fingerprint=storage_fingerprint,
            stable=stable,
            limitations=tuple(limitations),
        )
        receipt = await asyncio.to_thread(self.store.append_record, state)
        return CapturedState(
            state=state,
            elements=tuple(all_elements),
            receipt=receipt,
        )

    async def wait_for_quiet(self, page: Page) -> bool:
        """Return whether every current frame reaches the configured quiet window."""

        if self.config.quiet_window_ms == 0:
            return True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.config.quiet_timeout_ms / 1_000
        unchanged_since = loop.time()
        previous: Optional[str] = None
        while loop.time() < deadline:
            signatures: list[object] = []
            for _, frame in self._ordered_frames(page.main_frame):
                try:
                    signatures.append(await frame.evaluate(QUIET_SIGNATURE_SCRIPT))
                except PlaywrightError:
                    signatures.append(
                        {
                            "unavailable": redact_url(
                                frame.url, self.policy.redacted_names
                            )
                        }
                    )
            signature = _canonical_json(signatures)
            ready = all(
                not isinstance(item, dict) or item.get("readyState") != "loading"
                for item in signatures
            )
            now = loop.time()
            if signature != previous or not ready:
                previous = signature
                unchanged_since = now
            elif (now - unchanged_since) * 1_000 >= self.config.quiet_window_ms:
                return True
            await asyncio.sleep(self.config.quiet_poll_interval_ms / 1_000)
        return False

    async def _extract_frames(self, page: Page) -> tuple[_RawFrame, ...]:
        ordered = self._ordered_frames(page.main_frame)
        frame_keys = {id(frame): key for key, frame in ordered}
        frame_ids = {
            id(frame): (
                self.recorder.frame_id(frame)
                if self.recorder is not None
                else f"frame-{hash_text(key)[:24]}"
            )
            for key, frame in ordered
        }
        results: list[_RawFrame] = []
        for frame_key, frame in ordered:
            try:
                value = await frame.evaluate(
                    FRAME_EXTRACTION_SCRIPT,
                    {
                        "maximumElements": self.config.maximum_elements_per_frame,
                        "maximumTextChars": self.config.maximum_text_chars,
                        "redactedNames": list(self.policy.redacted_names),
                    },
                )
                data = value if isinstance(value, dict) else {}
            except PlaywrightError as exc:
                data = {
                    "extractionError": redact_text(str(exc), self.policy.redacted_names)
                }
            parent = frame.parent_frame
            children = tuple(
                frame_ids[id(child)]
                for child in frame.child_frames
                if id(child) in frame_ids
            )
            results.append(
                _RawFrame(
                    frame=frame,
                    frame_key=frame_keys[id(frame)],
                    frame_id=frame_ids[id(frame)],
                    parent_frame_id=frame_ids.get(id(parent)) if parent else None,
                    child_frame_ids=children,
                    data=data,
                )
            )
        return tuple(results)

    @staticmethod
    def _ordered_frames(main_frame: Frame) -> list[tuple[str, Frame]]:
        result: list[tuple[str, Frame]] = []

        def visit(frame: Frame, key: str) -> None:
            result.append((key, frame))
            active_children = [
                child for child in frame.child_frames if not child.is_detached()
            ]
            for index, child in enumerate(active_children):
                visit(child, f"{key}/{index}")

        visit(main_frame, "main")
        return result

    async def _materialize_frame(
        self,
        raw_frame: _RawFrame,
        *,
        main_origin: Optional[str],
    ) -> tuple[FrameSnapshot, tuple[ElementSnapshot, ...], Optional[str]]:
        limitations: list[str] = []
        extraction_error = raw_frame.data.get("extractionError")
        if extraction_error:
            limitations.append(f"frame extraction failed: {extraction_error}")
        if raw_frame.data.get("truncated"):
            limitations.append(
                "element extraction truncated at "
                f"{self.config.maximum_elements_per_frame} records"
            )
        closed_shadow_hosts = _nonnegative_int(raw_frame.data.get("closedShadowHosts"))
        if closed_shadow_hosts:
            limitations.append(
                f"{closed_shadow_hosts} declared closed shadow roots were not inspectable"
            )

        raw_elements = raw_frame.data.get("elements")
        element_values = raw_elements if isinstance(raw_elements, list) else []
        elements = self._normalize_elements(
            element_values,
            frame_id=raw_frame.frame_id,
            frame_key=raw_frame.frame_key,
        )
        active_element_id = self._active_element_id(raw_frame.data, elements)

        dom_artifact = await self._store_dom(raw_frame, limitations)
        elements_artifact = await self._store_element_collection(
            raw_frame.frame_id,
            elements,
            limitations,
        )
        accessibility_artifact = await self._store_accessibility(
            raw_frame.frame_id,
            elements,
            limitations,
        )
        safe_url = redact_url(raw_frame.frame.url, self.policy.redacted_names)
        frame_origin = _origin(raw_frame.frame.url)
        snapshot = FrameSnapshot(
            frame_id=raw_frame.frame_id,
            frame_path=raw_frame.frame_key,
            parent_frame_id=raw_frame.parent_frame_id,
            name=redact_text(raw_frame.frame.name, self.policy.redacted_names) or None,
            url=safe_url or "about:blank",
            is_main=raw_frame.frame.parent_frame is None,
            is_cross_origin=(
                raw_frame.frame.parent_frame is not None
                and frame_origin is not None
                and main_origin is not None
                and frame_origin != main_origin
            ),
            element_ids=tuple(element.element_id for element in elements),
            child_frame_ids=raw_frame.child_frame_ids,
            dom_artifact=dom_artifact,
            elements_artifact=elements_artifact,
            accessibility_artifact=accessibility_artifact,
            limitations=tuple(limitations),
        )
        return snapshot, elements, active_element_id

    def _normalize_elements(
        self,
        raw_elements: list[object],
        *,
        frame_id: str,
        frame_key: str,
    ) -> tuple[ElementSnapshot, ...]:
        mappings = [item for item in raw_elements if isinstance(item, dict)]
        identities: list[tuple[str, tuple[str, ...], str]] = []
        for raw in mappings:
            shadow_path = tuple(_string_list(raw.get("shadowHostPath")))
            dom_path = self._safe_text(_string(raw.get("domPath")))
            identities.append((dom_path, shadow_path, self._element_identity(raw)))

        counts: dict[str, int] = {}
        snapshots: list[ElementSnapshot] = []
        for raw, (dom_path, shadow_path, identity) in zip(mappings, identities):
            base_id = f"element-{hash_text(f'{frame_key}|{identity}')[:32]}"
            occurrence = counts.get(base_id, 0)
            counts[base_id] = occurrence + 1
            element_id = base_id if occurrence == 0 else f"{base_id}-{occurrence + 1}"
            raw_value = raw.get("value")
            value = str(raw_value) if raw_value is not None else None
            attributes = self._attributes(
                raw.get("attributes"),
                current_value=value,
            )
            context_value = raw.get("context")
            context = context_value if isinstance(context_value, dict) else {}
            limitations = (
                ("element text truncated",) if bool(raw.get("textTruncated")) else ()
            )
            snapshots.append(
                ElementSnapshot(
                    element_id=element_id,
                    frame_id=frame_id,
                    tag=_string(raw.get("tag")) or "unknown",
                    role=self._safe_optional(raw.get("role")),
                    accessible_name=self._safe_optional(raw.get("accessibleName")),
                    label=self._safe_optional(raw.get("label")),
                    text=self._safe_optional(raw.get("text")),
                    title=self._safe_optional(raw.get("title")),
                    input_type=self._safe_optional(raw.get("inputType")),
                    attributes=attributes,
                    value_hash=hash_text(value) if value is not None else None,
                    value_redacted=value is not None,
                    interactive=bool(raw.get("interactive")),
                    interaction_signals=tuple(
                        sorted(
                            {
                                self._safe_text(item)
                                for item in _string_list(raw.get("interactionSignals"))
                                if item
                            }
                        )
                    ),
                    visible=bool(raw.get("visible")),
                    enabled=bool(raw.get("enabled")),
                    editable=bool(raw.get("editable")),
                    checked=_optional_bool(raw.get("checked")),
                    selected=_optional_bool(raw.get("selected")),
                    expanded=_optional_bool(raw.get("expanded")),
                    read_only=_optional_bool(raw.get("readOnly")),
                    bounding_box=self._bounding_box(raw.get("boundingBox")),
                    shadow_host_path=tuple(
                        redact_text(item, self.policy.redacted_names)
                        for item in shadow_path
                    ),
                    parent_css_path=self._safe_optional(raw.get("parentDomPath")),
                    context=ElementContext(
                        section_heading=self._safe_optional(
                            context.get("sectionHeading")
                        ),
                        form_id=self._safe_optional(context.get("formId")),
                        table_id=self._safe_optional(context.get("tableId")),
                        row_label=self._safe_optional(context.get("rowLabel")),
                        ancestor_summary=tuple(
                            self._safe_text(item)
                            for item in _string_list(context.get("ancestors"))
                        ),
                    ),
                    options=self._options(
                        raw.get("options"),
                        raw.get("attributes"),
                    ),
                    locators=self._locators(raw, mappings, dom_path),
                    limitations=limitations,
                )
            )
        return tuple(snapshots)

    def _element_identity(self, raw: dict[str, Any]) -> str:
        attributes = raw.get("attributes")
        attrs = attributes if isinstance(attributes, dict) else {}
        for name in ("data-testid", "data-test", "data-cy", "id", "name"):
            value = _string(attrs.get(name))
            if value and _looks_stable(value):
                return f"{name}={value}|{_string(raw.get('tag'))}"
        role = _string(raw.get("role"))
        accessible_name = self._safe_text(_string(raw.get("accessibleName")))
        dom_path = _string(raw.get("domPath"))
        shadow_path = "/".join(_string_list(raw.get("shadowHostPath")))
        return f"{shadow_path}|{dom_path}|{role}|{accessible_name}"

    def _attributes(
        self,
        raw_attributes: object,
        *,
        current_value: Optional[str],
    ) -> tuple[ValueCapture, ...]:
        mapping = raw_attributes if isinstance(raw_attributes, dict) else {}
        captured: list[ValueCapture] = []
        for raw_name in sorted(mapping, key=lambda item: str(item).lower()):
            name = str(raw_name)
            raw = str(mapping[raw_name])
            if name.lower() == "value":
                captured.append(
                    ValueCapture(name=name, value_hash=hash_text(raw), redacted=True)
                )
            elif is_sensitive_name(name, self.policy.redacted_names):
                captured.append(
                    ValueCapture(name=name, value_hash=hash_text(raw), redacted=True)
                )
            else:
                safe = (
                    redact_url(raw, self.policy.redacted_names)
                    if name.lower() in {"action", "formaction", "href", "poster", "src"}
                    else redact_text(raw, self.policy.redacted_names)
                )
                if safe != raw:
                    normalized_safe = safe.lower()
                    retains_marker = (
                        "[redacted]" in normalized_safe
                        or "%5bredacted%5d" in normalized_safe
                    )
                    captured.append(
                        ValueCapture(
                            name=name,
                            safe_value=safe if retains_marker else None,
                            value_hash=hash_text(raw),
                            redacted=True,
                        )
                    )
                else:
                    captured.append(ValueCapture(name=name, value=safe))
        if current_value is not None and not any(
            item.name.lower() == "value" for item in captured
        ):
            captured.append(
                ValueCapture(
                    name="value",
                    value_hash=hash_text(current_value),
                    redacted=True,
                )
            )
        return tuple(captured)

    def _options(
        self,
        value: object,
        raw_attributes: object,
    ) -> tuple[SelectOptionSnapshot, ...]:
        options = value if isinstance(value, list) else []
        attributes = raw_attributes if isinstance(raw_attributes, dict) else {}
        sensitive_control = any(
            is_sensitive_name(_string(attributes.get(name)), self.policy.redacted_names)
            for name in ("id", "name")
        )
        result: list[SelectOptionSnapshot] = []
        for item in options:
            if not isinstance(item, dict):
                continue
            raw_value = _string(item.get("value"))
            safe_value = redact_text(raw_value, self.policy.redacted_names)
            redacted = sensitive_control or safe_value != raw_value
            result.append(
                SelectOptionSnapshot(
                    label=self._safe_text(_string(item.get("label"))),
                    value=None if redacted else safe_value,
                    value_hash=hash_text(raw_value) if redacted else None,
                    selected=bool(item.get("selected")),
                    disabled=bool(item.get("disabled")),
                    redacted=redacted,
                )
            )
        return tuple(result)

    def _locators(
        self,
        raw: dict[str, Any],
        all_elements: list[dict[str, Any]],
        dom_path: str,
    ) -> tuple[LocatorCandidate, ...]:
        attributes_value = raw.get("attributes")
        attributes = attributes_value if isinstance(attributes_value, dict) else {}
        candidates: list[tuple[LocatorStrategy, str, float, int]] = []

        def add(
            strategy: LocatorStrategy,
            value: str,
            confidence: float,
            count: int,
        ) -> None:
            if value:
                candidates.append((strategy, value, confidence, count))

        for name in ("data-testid", "data-test", "data-cy"):
            value = _string(attributes.get(name))
            if value and self._safe_text(value) == value:
                count = sum(
                    1
                    for item in all_elements
                    if isinstance(item.get("attributes"), dict)
                    and _string(item["attributes"].get(name)) == value
                )
                add(
                    LocatorStrategy.TEST_ID,
                    _canonical_json({"attribute": name, "value": value}),
                    0.99,
                    count,
                )
        stable_id = _string(attributes.get("id"))
        if (
            stable_id
            and _looks_stable(stable_id)
            and self._safe_text(stable_id) == stable_id
        ):
            count = sum(
                1
                for item in all_elements
                if isinstance(item.get("attributes"), dict)
                and _string(item["attributes"].get("id")) == stable_id
            )
            add(
                LocatorStrategy.STABLE_ATTRIBUTE,
                _canonical_json({"attribute": "id", "value": stable_id}),
                0.96,
                count,
            )
        role = self._safe_text(_string(raw.get("role")))
        name = self._safe_text(_string(raw.get("accessibleName")))
        if role and name:
            count = sum(
                1
                for item in all_elements
                if self._safe_text(_string(item.get("role"))) == role
                and self._safe_text(_string(item.get("accessibleName"))) == name
            )
            add(
                LocatorStrategy.ROLE,
                _canonical_json({"name": name, "role": role}),
                0.93,
                count,
            )
        label = self._safe_text(_string(raw.get("label")))
        if label:
            count = sum(
                1
                for item in all_elements
                if self._safe_text(_string(item.get("label"))) == label
            )
            add(LocatorStrategy.LABEL, label, 0.90, count)
        for strategy, attribute_name, confidence in (
            (LocatorStrategy.PLACEHOLDER, "placeholder", 0.82),
            (LocatorStrategy.ALT_TEXT, "alt", 0.82),
            (LocatorStrategy.TITLE, "title", 0.76),
        ):
            raw_value = _string(attributes.get(attribute_name))
            safe_value = self._safe_text(raw_value)
            if safe_value and safe_value == raw_value:
                count = sum(
                    1
                    for item in all_elements
                    if isinstance(item.get("attributes"), dict)
                    and _string(item["attributes"].get(attribute_name)) == raw_value
                )
                add(strategy, safe_value, confidence, count)
        stable_name = _string(attributes.get("name"))
        if (
            stable_name
            and _looks_stable(stable_name)
            and self._safe_text(stable_name) == stable_name
        ):
            count = sum(
                1
                for item in all_elements
                if isinstance(item.get("attributes"), dict)
                and _string(item["attributes"].get("name")) == stable_name
                and _string(item.get("tag")) == _string(raw.get("tag"))
            )
            add(
                LocatorStrategy.STABLE_ATTRIBUTE,
                _canonical_json(
                    {
                        "attribute": "name",
                        "tag": _string(raw.get("tag")),
                        "value": stable_name,
                    }
                ),
                0.84,
                count,
            )
        text = self._safe_text(_string(raw.get("text")))
        if bool(raw.get("interactive")) and text and len(text) <= 160:
            count = sum(
                1
                for item in all_elements
                if self._safe_text(_string(item.get("text"))) == text
            )
            add(LocatorStrategy.TEXT, text, 0.65, count)
        add(LocatorStrategy.CSS, dom_path, 0.35, 1)

        candidates.sort(key=lambda item: (-item[2], item[0].value, item[1]))
        primary_index = next(
            (index for index, item in enumerate(candidates) if item[3] == 1),
            None,
        )
        return tuple(
            LocatorCandidate(
                strategy=strategy,
                value=value,
                confidence=confidence,
                unique_match_count=count,
                is_primary=index == primary_index,
            )
            for index, (strategy, value, confidence, count) in enumerate(candidates)
        )

    async def _store_dom(
        self,
        raw_frame: _RawFrame,
        limitations: list[str],
    ) -> Optional[ArtifactReference]:
        if not self.policy.capture_dom:
            return None
        dom = raw_frame.data.get("dom")
        if not isinstance(dom, str):
            limitations.append("DOM serialization unavailable")
            return None
        safe_dom = redact_text(dom, self.policy.redacted_names)
        data = safe_dom.encode("utf-8")
        if not self._within_artifact_limit(data):
            limitations.append("DOM artifact exceeded maximum_artifact_bytes")
            return None
        return await asyncio.to_thread(
            self.store.put_bytes,
            ArtifactKind.DOM,
            data,
            media_type="text/html; charset=utf-8",
            redacted=bool(raw_frame.data.get("domRedacted")) or safe_dom != dom,
        )

    async def _store_element_collection(
        self,
        frame_id: str,
        elements: tuple[ElementSnapshot, ...],
        limitations: list[str],
    ) -> Optional[ArtifactReference]:
        collection = FrameElementCollection(frame_id=frame_id, elements=elements)
        data = _canonical_json(collection.model_dump(mode="json")).encode("utf-8")
        if not self._within_artifact_limit(data):
            limitations.append("element artifact exceeded maximum_artifact_bytes")
            return None
        return await asyncio.to_thread(
            self.store.put_bytes,
            ArtifactKind.ELEMENTS,
            data,
            media_type="application/json",
            redacted=True,
        )

    async def _store_accessibility(
        self,
        frame_id: str,
        elements: tuple[ElementSnapshot, ...],
        limitations: list[str],
    ) -> Optional[ArtifactReference]:
        if not self.policy.capture_accessibility_tree:
            return None
        nodes = [
            {
                "element_id": element.element_id,
                "role": element.role,
                "name": element.accessible_name,
                "label": element.label,
                "text": element.text,
                "visible": element.visible,
                "enabled": element.enabled,
                "bounding_box": (
                    element.bounding_box.model_dump(mode="json")
                    if element.bounding_box is not None
                    else None
                ),
            }
            for element in elements
            if element.role or element.accessible_name or element.label or element.text
        ]
        data = _canonical_json(
            {
                "frame_id": frame_id,
                "kind": "dom_accessibility_approximation",
                "nodes": nodes,
            }
        ).encode("utf-8")
        if not self._within_artifact_limit(data):
            limitations.append("accessibility artifact exceeded maximum_artifact_bytes")
            return None
        limitations.append(
            "accessibility artifact is a browser-independent DOM approximation"
        )
        return await asyncio.to_thread(
            self.store.put_bytes,
            ArtifactKind.ACCESSIBILITY_TREE,
            data,
            media_type="application/json",
            redacted=True,
        )

    async def _capture_screenshot(
        self,
        page: Page,
        limitations: list[str],
    ) -> Optional[ArtifactReference]:
        if not self.policy.capture_screenshots:
            return None
        try:
            data = await page.screenshot(
                full_page=self.config.full_page_screenshot,
                animations="disabled",
                caret="hide",
                scale="css",
                type="png",
            )
        except PlaywrightError as exc:
            limitations.append(
                f"screenshot capture failed: {redact_text(str(exc), self.policy.redacted_names)}"
            )
            return None
        if not self._within_artifact_limit(data):
            limitations.append("screenshot exceeded maximum_artifact_bytes")
            return None
        return await asyncio.to_thread(
            self.store.put_bytes,
            ArtifactKind.SCREENSHOT,
            data,
            media_type="image/png",
            redacted=False,
        )

    async def _capture_storage(
        self,
        page: Page,
        raw_frames: tuple[_RawFrame, ...],
        limitations: list[str],
    ) -> tuple[Optional[ArtifactReference], Optional[str]]:
        if not self.policy.capture_storage_state:
            return None, None
        try:
            state = await page.context.storage_state()
        except PlaywrightError as exc:
            limitations.append(
                f"storage state capture failed: {redact_text(str(exc), self.policy.redacted_names)}"
            )
            return None, None
        cookies = state.get("cookies", []) if isinstance(state, dict) else []
        origins = state.get("origins", []) if isinstance(state, dict) else []
        safe_cookies: list[dict[str, object]] = []
        for cookie in cookies if isinstance(cookies, list) else []:
            if not isinstance(cookie, dict):
                continue
            safe_cookies.append(
                {
                    "name": self._safe_text(_string(cookie.get("name"))),
                    "domain": self._safe_text(_string(cookie.get("domain"))),
                    "path": self._safe_text(_string(cookie.get("path"))),
                    "expires": cookie.get("expires"),
                    "http_only": bool(cookie.get("httpOnly")),
                    "secure": bool(cookie.get("secure")),
                    "same_site": _string(cookie.get("sameSite")),
                    "value_sha256": hash_text(_string(cookie.get("value"))),
                }
            )
        safe_origins: list[dict[str, object]] = []
        for origin in origins if isinstance(origins, list) else []:
            if not isinstance(origin, dict):
                continue
            local_storage = origin.get("localStorage", [])
            safe_origins.append(
                {
                    "origin": redact_url(
                        _string(origin.get("origin")), self.policy.redacted_names
                    ),
                    "local_storage": sorted(
                        (
                            {
                                "name": self._safe_text(_string(item.get("name"))),
                                "value_sha256": hash_text(_string(item.get("value"))),
                            }
                            for item in local_storage
                            if isinstance(item, dict)
                        ),
                        key=lambda item: str(item["name"]),
                    ),
                }
            )
        sessions: list[dict[str, object]] = []
        for raw_frame in raw_frames:
            raw_session = raw_frame.data.get("sessionStorage")
            entries: list[dict[str, str]] = []
            if isinstance(raw_session, list):
                for pair in raw_session:
                    if isinstance(pair, list) and len(pair) == 2:
                        entries.append(
                            {
                                "name": self._safe_text(_string(pair[0])),
                                "value_sha256": hash_text(_string(pair[1])),
                            }
                        )
            sessions.append(
                {
                    "frame_key": raw_frame.frame_key,
                    "origin": _origin(raw_frame.frame.url),
                    "session_storage": sorted(entries, key=lambda item: item["name"]),
                }
            )
        safe_state = {
            "cookies": sorted(
                safe_cookies,
                key=lambda item: (
                    str(item["domain"]),
                    str(item["path"]),
                    str(item["name"]),
                ),
            ),
            "origins": sorted(safe_origins, key=lambda item: str(item["origin"])),
            "sessions": sessions,
        }
        data = _canonical_json(safe_state).encode("utf-8")
        fingerprint = hashlib.sha256(data).hexdigest()
        if not self._within_artifact_limit(data):
            limitations.append("storage artifact exceeded maximum_artifact_bytes")
            return None, fingerprint
        reference = await asyncio.to_thread(
            self.store.put_bytes,
            ArtifactKind.STORAGE_STATE,
            data,
            media_type="application/json",
            redacted=True,
        )
        return reference, fingerprint

    def _fingerprint(
        self,
        *,
        url: str,
        title: str,
        viewport: Viewport,
        scroll: ScrollPosition,
        frames: tuple[_RawFrame, ...],
        elements: list[ElementSnapshot],
        active_element_id: Optional[str],
        storage_fingerprint: Optional[str],
        stable: bool,
    ) -> str:
        frame_keys = {frame.frame_id: frame.frame_key for frame in frames}
        payload = {
            "url": url,
            "title": title,
            "viewport": viewport.model_dump(mode="json"),
            "scroll": scroll.model_dump(mode="json"),
            "frames": [
                {
                    "key": frame.frame_key,
                    "url": redact_url(frame.frame.url, self.policy.redacted_names),
                    "name": self._safe_text(frame.frame.name),
                }
                for frame in frames
            ],
            "elements": [
                {
                    "frame_key": frame_keys.get(element.frame_id, "unknown"),
                    "tag": element.tag,
                    "role": element.role,
                    "name": element.accessible_name,
                    "label": element.label,
                    "text": element.text,
                    "input_type": element.input_type,
                    "attributes": [
                        attribute.model_dump(mode="json")
                        for attribute in element.attributes
                        if attribute.name.lower()
                        in {
                            "aria-busy",
                            "aria-checked",
                            "aria-disabled",
                            "aria-expanded",
                            "aria-hidden",
                            "aria-invalid",
                            "aria-label",
                            "aria-labelledby",
                            "aria-modal",
                            "aria-pressed",
                            "aria-selected",
                            "data-cy",
                            "data-test",
                            "data-testid",
                            "disabled",
                            "href",
                            "id",
                            "name",
                            "open",
                            "readonly",
                            "role",
                            "type",
                        }
                    ],
                    "value_hash": element.value_hash,
                    "interactive": element.interactive,
                    "interaction_signals": element.interaction_signals,
                    "visible": element.visible,
                    "enabled": element.enabled,
                    "editable": element.editable,
                    "checked": element.checked,
                    "selected": element.selected,
                    "expanded": element.expanded,
                    "read_only": element.read_only,
                    "options": [
                        option.model_dump(mode="json") for option in element.options
                    ],
                    "shadow_host_path": element.shadow_host_path,
                }
                for element in elements
            ],
            "active_element_id": active_element_id,
            "storage_fingerprint": storage_fingerprint,
            "stable": stable,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def _active_element_id(
        self,
        data: dict[str, Any],
        elements: tuple[ElementSnapshot, ...],
    ) -> Optional[str]:
        active = data.get("activeElement")
        if not isinstance(active, dict):
            return None
        path = redact_text(_string(active.get("domPath")), self.policy.redacted_names)
        shadow = tuple(
            redact_text(item, self.policy.redacted_names)
            for item in _string_list(active.get("shadowHostPath"))
        )
        for element in elements:
            css_locator = next(
                (
                    locator
                    for locator in element.locators
                    if locator.strategy == LocatorStrategy.CSS
                ),
                None,
            )
            if (
                css_locator is not None
                and css_locator.value == path
                and element.shadow_host_path == shadow
            ):
                return element.element_id
        return None

    def _descriptions(self, raw_frame: _RawFrame, key: str) -> list[str]:
        value = raw_frame.data.get(key)
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            path = self._safe_text(_string(item.get("path")))
            text = self._safe_text(_string(item.get("text")))
            result.append(f"{raw_frame.frame_key}:{path}:{text}")
        return result

    def _viewport(self, data: dict[str, Any]) -> Viewport:
        value = data.get("viewport")
        viewport = value if isinstance(value, dict) else {}
        return Viewport(
            width=max(1, _nonnegative_int(viewport.get("width"))),
            height=max(1, _nonnegative_int(viewport.get("height"))),
            device_scale_factor=max(0.01, _float(viewport.get("deviceScaleFactor"), 1)),
        )

    @staticmethod
    def _scroll(data: dict[str, Any]) -> ScrollPosition:
        value = data.get("scroll")
        scroll = value if isinstance(value, dict) else {}
        maximum_x = max(0.0, _float(scroll.get("maximumX")))
        maximum_y = max(0.0, _float(scroll.get("maximumY")))
        return ScrollPosition(
            x=min(maximum_x, max(0.0, _float(scroll.get("x")))),
            y=min(maximum_y, max(0.0, _float(scroll.get("y")))),
            maximum_x=maximum_x,
            maximum_y=maximum_y,
        )

    @staticmethod
    def _bounding_box(value: object) -> Optional[BoundingBox]:
        if not isinstance(value, dict):
            return None
        return BoundingBox(
            x=_float(value.get("x")),
            y=_float(value.get("y")),
            width=max(0.0, _float(value.get("width"))),
            height=max(0.0, _float(value.get("height"))),
        )

    def _safe_optional(self, value: object) -> Optional[str]:
        safe = self._safe_text(_string(value))
        return safe or None

    def _safe_text(self, value: str) -> str:
        return redact_text(value, self.policy.redacted_names)

    def _within_artifact_limit(self, data: bytes) -> bool:
        return len(data) <= self.store.run.limits.maximum_artifact_bytes


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _string(value: object) -> str:
    return "" if value is None else str(value)


def _string_list(value: object) -> list[str]:
    return [_string(item) for item in value] if isinstance(value, list) else []


def _float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if result != result or result in {float("inf"), float("-inf")}:
        return default
    return result


def _nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _optional_bool(value: object) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _origin(url: str) -> Optional[str]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def _looks_stable(value: str) -> bool:
    return bool(
        value
        and len(value) <= 128
        and re.search(r"[0-9]{6,}", value) is None
        and re.search(r"[a-f0-9]{12,}", value, re.IGNORECASE) is None
        and re.fullmatch(r"[a-f0-9-]{32,}", value, re.IGNORECASE) is None
    )


__all__ = [
    "CapturedState",
    "FRAME_EXTRACTION_SCRIPT",
    "PageStateExtractor",
    "SnapshotConfig",
    "SnapshotExtractionError",
]
