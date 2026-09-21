"""Typed operator guidance and secret-safe click teaching for portal crawls."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from playwright.async_api import BrowserContext
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from scraper.extractor import CapturedState
from scraper.models import ActionKind, ElementSnapshot, LocatorStrategy, utc_now
from scraper.redaction import hash_text, redact_text


MAXIMUM_GUIDE_BYTES = 1_048_576
MAXIMUM_RECORDED_CLICKS = 1_000


class GuidanceError(RuntimeError):
    """Raised when a crawl guide cannot be loaded, recorded, or replayed."""


class CrawlStrategy(str, Enum):
    """How configured guidance and generic discovery are combined."""

    EXHAUSTIVE = "exhaustive"
    GUIDED = "guided"
    HYBRID = "hybrid"


class ParallelSessionMode(str, Enum):
    """Policy for cloning one authenticated session into worker browsers."""

    OFF = "off"
    PROBE = "probe"
    FORCE = "force"


class GuideModel(BaseModel):
    """Strict immutable base for portable guide files."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class GuideTarget(GuideModel):
    """Ordered semantic and structural fallbacks for one browser target."""

    test_id: Optional[str] = Field(default=None, max_length=500)
    test_id_attribute: str = Field(default="data-testid", max_length=100)
    tag: Optional[str] = Field(default=None, max_length=100)
    role: Optional[str] = Field(default=None, max_length=100)
    accessible_name: Optional[str] = Field(default=None, max_length=2_000)
    stable_id: Optional[str] = Field(default=None, max_length=500)
    title: Optional[str] = Field(default=None, max_length=2_000)
    text: Optional[str] = Field(default=None, max_length=2_000)
    css: Optional[str] = Field(default=None, max_length=4_000)
    css_regex: Optional[str] = Field(default=None, max_length=4_000)
    css_classes: tuple[str, ...] = ()
    frame_path: Optional[str] = Field(default=None, max_length=1_000)

    @field_validator("css_classes")
    @classmethod
    def validate_css_classes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        tokens = sorted(
            {
                token.strip()[:200]
                for token in value
                if isinstance(token, str) and token.strip()
            }
        )
        if len(tokens) > 20:
            raise ValueError("guide target allows at most 20 CSS classes")
        return tuple(tokens)

    @model_validator(mode="after")
    def require_locator(self) -> "GuideTarget":
        if not any(
            (
                self.test_id,
                self.role,
                self.accessible_name,
                self.stable_id,
                self.title,
                self.text,
                self.css,
                self.css_regex,
                self.css_classes,
            )
        ):
            raise ValueError("guide target requires at least one locator signal")
        if self.accessible_name and not self.role:
            raise ValueError("accessible_name requires a role")
        return self

    @field_validator("css_regex")
    @classmethod
    def validate_css_regex(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(
                    f"invalid CSS locator regular expression: {exc}"
                ) from exc
        return value


class GuideExpectation(GuideModel):
    """Optional postcondition that prevents silent replay drift."""

    url_pattern: Optional[str] = Field(default=None, max_length=2_000)
    visible_text: Optional[str] = Field(default=None, max_length=2_000)
    modal_open: Optional[bool] = None
    table_present: Optional[bool] = None

    @field_validator("url_pattern")
    @classmethod
    def validate_pattern(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"invalid URL regular expression: {exc}") from exc
        return value


class GuideStep(GuideModel):
    """One explicit action leading toward the promoted exploration root."""

    name: str = Field(min_length=1, max_length=200)
    action: ActionKind = ActionKind.CLICK
    target: GuideTarget
    expect: GuideExpectation = Field(default_factory=GuideExpectation)

    @field_validator("action")
    @classmethod
    def validate_action(cls, value: ActionKind) -> ActionKind:
        supported = {
            ActionKind.CLICK,
            ActionKind.DOUBLE_CLICK,
            ActionKind.HOVER,
            ActionKind.CHECK,
            ActionKind.UNCHECK,
        }
        if value not in supported:
            raise ValueError(
                "guide steps support click, double-click, hover, and toggles"
            )
        return value


class GuideRepeatRule(GuideModel):
    """A row-relative observation demonstrated once and applied to every row."""

    name: str = Field(min_length=1, max_length=200)
    target: GuideTarget
    close_target: GuideTarget
    next_page_target: Optional[GuideTarget] = None
    auto_paginate: bool = True
    require_row_label: bool = True
    expect_dialog_contains_row_label: bool = True
    maximum_rows: int = Field(default=10_000, gt=0)
    maximum_pages: int = Field(default=1_000, gt=0)


class GuideBranchRule(GuideModel):
    """One demonstrated navigation level expanded across sibling controls."""

    name: str = Field(min_length=1, max_length=200)
    target: GuideTarget
    require_row_label: bool = False
    maximum_branches: int = Field(default=1_000, gt=0)


class CrawlGuide(GuideModel):
    """Portable, portal-specific data consumed by the generic crawler."""

    schema_version: str = "1.0"
    source: str = Field(default="manual", pattern=r"^(manual|taught)$")
    created_at: Optional[datetime] = None
    steps: tuple[GuideStep, ...] = ()
    branch_rules: tuple[GuideBranchRule, ...] = ()
    repeat_rules: tuple[GuideRepeatRule, ...] = ()
    promote_final_state_to_root: bool = True
    root_scope_selector: Optional[str] = Field(default="main", max_length=4_000)

    @model_validator(mode="after")
    def validate_names(self) -> "CrawlGuide":
        names = [step.name for step in self.steps]
        names.extend(rule.name for rule in self.branch_rules)
        names.extend(rule.name for rule in self.repeat_rules)
        if len(names) != len(set(names)):
            raise ValueError(
                "guide step, branch-rule, and repeat-rule names must be unique"
            )
        if not self.steps and not self.branch_rules and not self.repeat_rules:
            raise ValueError("crawl guide requires a step, branch rule, or repeat rule")
        if self.branch_rules and not self.repeat_rules:
            raise ValueError("branch rules require a terminal repeat rule")
        return self


class OperatorClick(GuideModel):
    """One secret-safe operator click captured after authentication."""

    sequence: int = Field(ge=0)
    observed_at: datetime = Field(default_factory=utc_now)
    target: GuideTarget
    row_label: Optional[str] = Field(default=None, max_length=2_000)
    table_id: Optional[str] = Field(default=None, max_length=2_000)
    inside_dialog: bool = False


TEACHING_INIT_SCRIPT = r"""
(() => {
    if (window.__czTeachingInstalled) return;
    Object.defineProperty(window, '__czTeachingInstalled', {
        value: true,
        configurable: false,
        enumerable: false,
        writable: false,
    });
    const normalized = (value) => String(value || '').replace(/\s+/g, ' ').trim();
    const stableId = (value) => Boolean(value)
        && value.length <= 200
        && !/[0-9a-f]{12,}/i.test(value)
        && !/\d{7,}/.test(value);
    const cssEscape = (value) => {
        if (window.CSS && typeof window.CSS.escape === 'function') return window.CSS.escape(value);
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
                if (siblings.length > 1) part += `:nth-of-type(${siblings.indexOf(current) + 1})`;
            }
            parts.unshift(part);
            if (!parent || current === document.documentElement) break;
            current = parent;
        }
        return parts.join(' > ');
    };
    const implicitRole = (element) => {
        const tag = element.tagName.toLowerCase();
        const type = normalized(element.getAttribute('type')).toLowerCase();
        if ((tag === 'a' || tag === 'area') && element.hasAttribute('href')) return 'link';
        if (tag === 'button' || tag === 'summary') return 'button';
        if (tag === 'input' && ['button', 'image', 'reset', 'submit'].includes(type)) return 'button';
        if (tag === 'input' && type === 'checkbox') return 'checkbox';
        if (tag === 'input' && type === 'radio') return 'radio';
        return '';
    };
    const accessibleName = (element) => normalized(
        element.getAttribute('aria-label')
        || element.getAttribute('title')
        || (['button', 'a', 'summary'].includes(element.tagName.toLowerCase())
            ? (element.innerText || element.textContent) : '')
    );
    document.addEventListener('pointerdown', (event) => {
        const raw = event.target;
        const element = raw && raw.closest
            ? raw.closest('button, a, input, summary, [role], [tabindex], [onclick]') || raw
            : raw;
        if (!element || element.nodeType !== Node.ELEMENT_NODE) return;
        const row = element.closest ? element.closest('tr') : null;
        const rowCells = row
            ? Array.from(row.querySelectorAll('th[scope=row], th, td'))
            : [];
        const firstCell =
            rowCells.find(
                (cell) => normalized(cell.innerText || cell.textContent)
            ) || null;
        const table = element.closest ? element.closest('table') : null;
        const dialog = element.closest
            ? element.closest('dialog, [role=dialog], .modal, [aria-modal=true]')
            : null;
        const testAttributes = ['data-testid', 'data-test', 'data-cy'];
        let testId = '';
        let testIdAttribute = 'data-testid';
        for (const attribute of testAttributes) {
            if (element.hasAttribute(attribute)) {
                testId = normalized(element.getAttribute(attribute));
                testIdAttribute = attribute;
                break;
            }
        }
        const payload = {
            tag: element.tagName.toLowerCase(),
            role: normalized(element.getAttribute('role')) || implicitRole(element),
            accessibleName: accessibleName(element),
            stableId: stableId(element.id) ? element.id : '',
            title: normalized(element.getAttribute('title')),
            text: normalized(element.innerText || element.textContent).slice(0, 2000),
            css: cssPath(element),
            testId,
            testIdAttribute,
            cssClasses: (element.getAttribute('class') || '')
                .split(/\s+/)
                .filter(Boolean)
                .slice(0, 20),
            rowLabel: normalized(firstCell ? (firstCell.innerText || firstCell.textContent) : ''),
            tableId: normalized(table ? (table.id || '') : ''),
            insideDialog: Boolean(dialog),
        };
        if (typeof window.__czRecordOperatorAction === 'function') {
            void window.__czRecordOperatorAction(payload);
        }
    }, true);
})();
"""


@dataclass
class OperatorActionRecorder:
    """Receive browser click bindings only during an explicit teaching window."""

    redacted_names: tuple[str, ...]
    maximum_clicks: int = MAXIMUM_RECORDED_CLICKS

    def __post_init__(self) -> None:
        if self.maximum_clicks <= 0:
            raise ValueError("maximum_clicks must be positive")
        self._active = False
        self._clicks: list[OperatorClick] = []
        self._lock = asyncio.Lock()

    @property
    def clicks(self) -> tuple[OperatorClick, ...]:
        return tuple(self._clicks)

    async def install(self, context: BrowserContext) -> None:
        """Install the binding and init script before portal navigation."""

        await context.expose_binding("__czRecordOperatorAction", self._receive)
        await context.add_init_script(script=TEACHING_INIT_SCRIPT)

    def activate(self) -> None:
        self._active = True

    def deactivate(self) -> None:
        self._active = False

    async def _receive(self, source: dict[str, Any], payload: object) -> None:
        del source
        if not self._active or not isinstance(payload, dict):
            return
        async with self._lock:
            if not self._active or len(self._clicks) >= self.maximum_clicks:
                return
            click = self._materialize_click(payload, len(self._clicks))
            self._clicks.append(click)

    def _materialize_click(
        self,
        payload: dict[str, Any],
        sequence: int,
    ) -> OperatorClick:
        def safe(name: str, limit: int = 2_000) -> Optional[str]:
            raw = payload.get(name)
            if not isinstance(raw, str) or not raw.strip():
                return None
            return redact_text(raw, self.redacted_names)[:limit].strip() or None

        role = safe("role", 100)
        accessible_name = safe("accessibleName")
        if not role or not accessible_name:
            role = None
            accessible_name = None
        raw_classes = payload.get("cssClasses")
        css_classes = tuple(
            token.strip()[:200]
            for token in (raw_classes if isinstance(raw_classes, list) else [])
            if isinstance(token, str) and token.strip()
        )[:20]
        target = GuideTarget(
            test_id=safe("testId", 500),
            test_id_attribute=safe("testIdAttribute", 100) or "data-testid",
            tag=safe("tag", 100),
            role=role,
            accessible_name=accessible_name,
            stable_id=safe("stableId", 500),
            title=safe("title"),
            text=safe("text"),
            css=safe("css", 4_000),
            css_classes=css_classes,
        )
        return OperatorClick(
            sequence=sequence,
            target=target,
            row_label=safe("rowLabel"),
            table_id=safe("tableId"),
            inside_dialog=bool(payload.get("insideDialog")),
        )


def compile_taught_guide(
    clicks: tuple[OperatorClick, ...],
    *,
    branch_depth: int = 0,
) -> CrawlGuide:
    """Compile navigation clicks and one row/modal demonstration into a guide."""

    if not clicks:
        raise GuidanceError("teaching finished without any recorded clicks")
    if branch_depth < 0:
        raise GuidanceError("teaching branch depth cannot be negative")

    repeated_index: Optional[int] = None
    close_index: Optional[int] = None
    for index, click in enumerate(clicks[:-1]):
        if (
            click.row_label
            and not click.inside_dialog
            and clicks[index + 1].inside_dialog
        ):
            repeated_index = index
            close_index = index + 1
            break

    route_clicks = clicks if repeated_index is None else clicks[:repeated_index]
    if branch_depth and repeated_index is None:
        raise GuidanceError(
            "portal-wide teaching requires a testcase info and dialog-close example"
        )
    if branch_depth > len(route_clicks):
        raise GuidanceError(
            "teaching branch depth exceeds the demonstrated navigation levels"
        )
    explicit_route_clicks = (
        route_clicks[:-branch_depth] if branch_depth else route_clicks
    )
    steps = tuple(
        GuideStep(name=f"taught_step_{index + 1}", target=click.target)
        for index, click in enumerate(explicit_route_clicks)
    )
    branch_rules = tuple(
        GuideBranchRule(
            name=f"taught_branch_{index + 1}",
            target=_generalize_branch_target(click),
            require_row_label=bool(click.row_label),
        )
        for index, click in enumerate(route_clicks[len(explicit_route_clicks) :])
    )
    repeat_rules: tuple[GuideRepeatRule, ...] = ()
    root_scope_selector = "main"
    if repeated_index is not None:
        repeated = clicks[repeated_index]
        close = clicks[close_index] if close_index is not None else None
        if close is None:  # pragma: no cover - inference requires the next click
            raise GuidanceError(
                "row observation teaching requires a dialog close click"
            )
        if close.target.tag in {"a", "area"} or close.target.role == "link":
            raise GuidanceError(
                "dialog close teaching must use the actual button-like close "
                "control, not a link inside the dialog"
            )
        repeat_rules = (
            GuideRepeatRule(
                name="taught_row_observation",
                target=_generalize_repeat_target(
                    repeated.target,
                    repeated.row_label or "",
                ),
                close_target=close.target,
            ),
        )
        if repeated.table_id and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_-]*", repeated.table_id
        ):
            root_scope_selector = f"#{repeated.table_id}"
    return CrawlGuide(
        source="taught",
        created_at=utc_now(),
        steps=steps,
        branch_rules=branch_rules,
        repeat_rules=repeat_rules,
        promote_final_state_to_root=True,
        root_scope_selector=root_scope_selector,
    )


def load_crawl_guide(path: Path) -> tuple[CrawlGuide, str]:
    """Load a bounded JSON guide and return its canonical content digest."""

    source = path.expanduser().resolve()
    if not source.is_file():
        raise GuidanceError("crawl guide path must reference an existing file")
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise GuidanceError(f"failed to read crawl guide: {exc}") from exc
    if len(data) > MAXIMUM_GUIDE_BYTES:
        raise GuidanceError("crawl guide exceeds the one-megabyte size limit")
    try:
        guide = CrawlGuide.model_validate_json(data)
    except ValueError as exc:
        raise GuidanceError(f"invalid crawl guide: {exc}") from exc
    return guide, guide_sha256(guide)


def write_crawl_guide(
    path: Path, guide: CrawlGuide, *, overwrite: bool = False
) -> Path:
    """Atomically write a learned guide without replacing files implicitly."""

    destination = path.expanduser().absolute()
    if destination.is_symlink():
        raise GuidanceError("crawl guide output cannot be a symbolic link")
    if destination.exists() and not overwrite:
        raise GuidanceError("crawl guide output exists; explicit overwrite is required")
    if destination.exists() and not destination.is_file():
        raise GuidanceError("crawl guide output must be a regular file")
    destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    payload = guide.model_dump_json(indent=2).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise GuidanceError(
                    "crawl guide output exists; explicit overwrite is required"
                ) from exc
            temporary.unlink()
        os.chmod(destination, 0o600)
        return destination
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def guide_sha256(guide: CrawlGuide) -> str:
    """Hash canonical guide content for the secret-free behavior manifest."""

    payload = json.dumps(
        guide.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hash_text(payload)


def target_elements(
    capture: CapturedState,
    target: GuideTarget,
    *,
    allow_many: bool,
    require_visible: bool = False,
) -> tuple[ElementSnapshot, ...]:
    """Resolve ordered target fallbacks against immutable element evidence."""

    frame_ids: Optional[set[str]] = None
    if target.frame_path is not None:
        frame_ids = {
            frame.frame_id
            for frame in capture.state.frames
            if frame.frame_path == target.frame_path
        }
    elements = tuple(
        element
        for element in capture.elements
        if frame_ids is None or element.frame_id in frame_ids
    )
    if require_visible:
        # Pages routinely keep several same-shaped dialogs in the DOM with
        # identical close controls; only the dialog that actually opened has
        # a visible one. Visibility is the honest disambiguator for close
        # targets.
        elements = tuple(
            element for element in elements if element.visible and element.enabled
        )
    tiers: list[tuple[ElementSnapshot, ...]] = []
    if target.test_id:
        tiers.append(
            tuple(
                element
                for element in elements
                if _attribute(element, target.test_id_attribute) == target.test_id
            )
        )
    if target.stable_id:
        tiers.append(
            tuple(
                element
                for element in elements
                if _attribute(element, "id") == target.stable_id
            )
        )
    if target.css_classes:
        wanted_classes = set(target.css_classes)
        tiers.append(
            tuple(
                element
                for element in elements
                if wanted_classes <= _element_classes(element)
            )
        )
    if target.role and target.accessible_name:
        tiers.append(
            tuple(
                element
                for element in elements
                if (element.role or "").casefold() == target.role.casefold()
                and (element.accessible_name or "") == target.accessible_name
            )
        )
    elif target.role:
        tiers.append(
            tuple(
                element
                for element in elements
                if (element.role or "").casefold() == target.role.casefold()
                and (target.tag is None or element.tag == target.tag)
                and (target.title is None or element.title == target.title)
                and (target.text is None or element.text == target.text)
            )
        )
    if target.title:
        tiers.append(
            tuple(element for element in elements if element.title == target.title)
        )
    if target.css:
        tiers.append(
            tuple(
                element
                for element in elements
                if any(
                    locator.strategy == LocatorStrategy.CSS
                    and locator.value == target.css
                    for locator in element.locators
                )
            )
        )
    if target.css_regex:
        pattern = re.compile(target.css_regex)
        tiers.append(
            tuple(
                element
                for element in elements
                if any(
                    locator.strategy == LocatorStrategy.CSS
                    and pattern.fullmatch(locator.value)
                    for locator in element.locators
                )
            )
        )
    if target.text:
        tiers.append(
            tuple(element for element in elements if element.text == target.text)
        )

    for matches in tiers:
        if allow_many and matches:
            return matches
        if not allow_many and len(matches) == 1:
            return matches
    return ()


def expectation_errors(
    capture: CapturedState,
    expectation: GuideExpectation,
) -> tuple[str, ...]:
    """Return deterministic postcondition failures for one guide step."""

    errors: list[str] = []
    if expectation.url_pattern and not re.search(
        expectation.url_pattern, capture.state.url
    ):
        errors.append("result URL did not match url_pattern")
    if expectation.visible_text and not any(
        expectation.visible_text in value
        for element in capture.elements
        for value in (
            element.text or "",
            element.accessible_name or "",
            element.title or "",
        )
    ):
        errors.append("visible_text was not found")
    if (
        expectation.modal_open is not None
        and (capture.state.modal_count > 0) != expectation.modal_open
    ):
        errors.append("modal_open expectation did not match")
    if expectation.table_present is not None:
        table_present = any(
            element.tag == "table" or (element.role or "").lower() == "table"
            for element in capture.elements
        )
        if table_present != expectation.table_present:
            errors.append("table_present expectation did not match")
    return tuple(errors)


def _attribute(element: ElementSnapshot, name: str) -> Optional[str]:
    for attribute in element.attributes:
        if attribute.name.casefold() != name.casefold() or attribute.redacted:
            continue
        return attribute.value
    return None


def _element_classes(element: ElementSnapshot) -> set[str]:
    value = _attribute(element, "class")
    if not value:
        return set()
    return {token for token in value.split() if token}


def _generalize_repeat_target(target: GuideTarget, row_label: str) -> GuideTarget:
    """Remove exemplar-specific row tokens and CSS from a repeated control."""

    needle = row_label.casefold()

    def generic(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return None if needle and needle in value.casefold() else value

    test_id = generic(target.test_id)
    stable_id = generic(target.stable_id)
    accessible_name = generic(target.accessible_name)
    title = generic(target.title)
    text = generic(target.text)
    # Class tokens are shared across rows (e.g. a per-table info-button class),
    # so they survive generalization unless the exemplar row label appears in
    # the token itself.
    css_classes = tuple(
        token
        for token in target.css_classes
        if not (needle and needle in token.casefold())
    )
    # A row exemplar's absolute CSS path points only to that row. Repetition is
    # instead grounded by row context plus the remaining semantic signature.
    return GuideTarget(
        test_id=test_id,
        test_id_attribute=target.test_id_attribute,
        tag=target.tag,
        role=target.role,
        accessible_name=accessible_name,
        stable_id=stable_id,
        title=title,
        text=text,
        css=None,
        css_regex=None,
        css_classes=css_classes,
        frame_path=target.frame_path,
    )


def _generalize_branch_target(click: OperatorClick) -> GuideTarget:
    """Generalize one taught branch exemplar to its semantic siblings."""

    if click.row_label:
        return _generalize_repeat_target(click.target, click.row_label)
    css = click.target.css
    if css and re.search(r":nth-(?:of-type|child)\(\d+\)", css):
        # Every positional index is generalized, not only the exemplar's own:
        # sibling pages routinely differ in wrapper or panel counts, so the
        # taught exemplar's intermediate positions are as accidental as its
        # own index. The tag sequence keeps the pattern specific.
        parts = re.split(r"(:nth-(?:of-type|child)\(\d+\))", css)
        css_regex = "".join(
            r":nth-(?:of-type|child)\(\d+\)"
            if re.fullmatch(r":nth-(?:of-type|child)\(\d+\)", part)
            else re.escape(part)
            for part in parts
            if part
        )
        return GuideTarget(
            tag=click.target.tag,
            css_regex=css_regex,
            css_classes=click.target.css_classes,
            frame_path=click.target.frame_path,
        )
    raise GuidanceError(
        "a non-row branch exemplar requires a repeatable sibling CSS position"
    )


__all__ = [
    "CrawlGuide",
    "CrawlStrategy",
    "GuidanceError",
    "GuideBranchRule",
    "GuideExpectation",
    "GuideRepeatRule",
    "GuideStep",
    "GuideTarget",
    "OperatorActionRecorder",
    "OperatorClick",
    "ParallelSessionMode",
    "TEACHING_INIT_SCRIPT",
    "compile_taught_guide",
    "expectation_errors",
    "guide_sha256",
    "load_crawl_guide",
    "target_elements",
    "write_crawl_guide",
]
