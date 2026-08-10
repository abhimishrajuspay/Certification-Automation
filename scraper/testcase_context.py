"""Incremental, selector-free testcase-context completion tracking."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from scraper.extractor import CapturedState
from scraper.models import ElementSnapshot, TestcaseContextCoverage


_TEST_CASE_ID_KEYS = {
    "case_id",
    "tc",
    "tc_id",
    "test_case",
    "test_case_id",
    "test_id",
    "testcase",
    "testcase_id",
}
_TEST_CASE_SIGNAL_KEYS = {
    "api",
    "api_name",
    "api_type",
    "apiname",
    "apitype",
    "biller_id",
    "billerid",
    "dependency",
    "dependency_case",
    "expected_response_code",
    "rc",
    "response_code",
    "status",
    "test_data",
    "testdata",
}
_TOTAL_TEST_CASE_KEYS = {
    "total_cases",
    "total_tc",
    "total_tcs",
    "total_test_cases",
    "total_testcases",
}
_ORDINAL_KEYS = {"column_1", "index", "no", "number", "row", "serial", "sr_no"}
_IDENTIFIER_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{2,}")
_PAGINATION_TOTAL = re.compile(
    r"\bshowing\s+(?P<start>\d[\d,]*)\s+to\s+(?P<end>\d[\d,]*)\s+"
    r"of\s+(?P<total>\d[\d,]*)\s+"
    r"(?:entries|items|records|results|test\s*cases?|tests?)\b",
    re.IGNORECASE,
)
_PAGINATION_QUERY_KEYS = {
    "cursor",
    "limit",
    "offset",
    "page",
    "pageindex",
    "pagenumber",
    "pagesize",
    "start",
}


@dataclass
class _CaseEvidence:
    fields: dict[str, set[str]] = field(default_factory=dict)


class TestcaseContextTracker:
    """Accumulate table and dialog evidence and expose a safe early-stop gate.

    The tracker intentionally consumes the same semantic evidence as the Phase 7
    normalizer: table headers and row structure, data attributes and links for
    identifier disambiguation, and dialog ancestry for descriptions. It never
    relies on a portal-specific selector or fixed testcase prefix.
    """

    def __init__(self, required_stable_observations: int = 2) -> None:
        if required_stable_observations <= 0:
            raise ValueError("required_stable_observations must be positive")
        self.required_stable_observations = required_stable_observations
        self._observed_fingerprints: set[str] = set()
        self._declared_totals: dict[str, int] = {}
        self._pagination_totals: dict[str, int] = {}
        self._declared_conflicts: set[str] = set()
        self._cases: dict[str, _CaseEvidence] = {}
        self._modal_texts: set[tuple[int, str]] = set()
        self._last_complete_signature: Optional[tuple[object, ...]] = None
        self._stable_observations = 0

    def observe(self, capture: CapturedState) -> TestcaseContextCoverage:
        """Ingest one capture and return current cumulative coverage."""

        if capture.state.fingerprint not in self._observed_fingerprints:
            self._observed_fingerprints.add(capture.state.fingerprint)
            self._collect_tables(capture.elements)
            self._collect_pagination_total(capture)
            if capture.state.modal_count > 0:
                self._collect_modal_text(capture.elements)

        coverage, signature = self._assess()
        if coverage.context_complete:
            if signature == self._last_complete_signature:
                self._stable_observations += 1
            else:
                self._last_complete_signature = signature
                self._stable_observations = 1
        else:
            self._last_complete_signature = None
            self._stable_observations = 0
        return coverage.model_copy(
            update={"stable_observations": self._stable_observations}
        )

    def observe_selected_total(
        self,
        capture: CapturedState,
        control: ElementSnapshot,
    ) -> TestcaseContextCoverage:
        """Collect only the summary-table row selected by the final guide step."""

        row_path = _control_row_path(control)
        if row_path is not None:
            self._collect_tables(
                capture.elements,
                declared_total_rows={row_path},
                collect_test_cases=False,
            )
        return self.snapshot()

    def snapshot(self) -> TestcaseContextCoverage:
        """Return current progress without counting another observation."""

        coverage, _ = self._assess()
        return coverage.model_copy(
            update={"stable_observations": self._stable_observations}
        )

    def _collect_tables(
        self,
        elements: tuple[ElementSnapshot, ...],
        *,
        declared_total_rows: Optional[set[str]] = None,
        collect_test_cases: bool = True,
    ) -> None:
        headers_by_root: dict[tuple[str, str], list[ElementSnapshot]] = defaultdict(
            list
        )
        cells_by_row: dict[tuple[str, str, str], list[ElementSnapshot]] = defaultdict(
            list
        )
        for element in elements:
            root = _table_root(element)
            if root is None:
                continue
            table_key = (element.frame_id, root)
            role = (element.role or "").lower()
            if element.tag == "th" or role == "columnheader":
                headers_by_root[table_key].append(element)
            elif element.tag == "td" or role == "cell":
                row_path = element.parent_css_path
                if row_path:
                    cells_by_row[(element.frame_id, root, row_path)].append(element)

        for (frame_id, root), header_elements in headers_by_root.items():
            headers = tuple(
                _element_text(element) or f"Column {index}"
                for index, element in enumerate(header_elements, start=1)
            )
            keys = _field_keys(headers)
            header_keys = set(keys)
            for (row_frame_id, row_root, row_path), cells in cells_by_row.items():
                if row_frame_id != frame_id or row_root != root or not cells:
                    continue
                fields = tuple(
                    (
                        keys[index] if index < len(keys) else f"column_{index + 1}",
                        _element_text(cell),
                    )
                    for index, cell in enumerate(cells)
                )
                if declared_total_rows is None or row_path in declared_total_rows:
                    self._collect_declared_total(fields, row_path)
                if not collect_test_cases:
                    continue
                if not (header_keys & _TEST_CASE_ID_KEYS):
                    continue
                if len(header_keys & _TEST_CASE_SIGNAL_KEYS) < 1:
                    continue
                controls = tuple(
                    element
                    for element in elements
                    if element.frame_id == frame_id
                    and element.interactive
                    and _belongs_to_row(element, row_path)
                )
                test_case_id = _test_case_id(fields, controls)
                if test_case_id is None:
                    continue
                case = self._cases.setdefault(test_case_id, _CaseEvidence())
                for key, value in fields:
                    normalized_value = (
                        test_case_id if key in _TEST_CASE_ID_KEYS else value
                    )
                    case.fields.setdefault(key, set()).add(normalized_value)

    def _collect_declared_total(
        self,
        fields: tuple[tuple[str, str], ...],
        row_path: str,
    ) -> None:
        total_field = next(
            ((key, value) for key, value in fields if key in _TOTAL_TEST_CASE_KEYS),
            None,
        )
        if total_field is None or not total_field[1].isdigit():
            return
        identity = next(
            (
                value
                for key, value in fields
                if key not in _TOTAL_TEST_CASE_KEYS | _ORDINAL_KEYS
                and value
                and not value.isdigit()
            ),
            row_path,
        )
        value = int(total_field[1])
        previous = self._declared_totals.get(identity)
        if previous is None:
            self._declared_totals[identity] = value
        elif previous != value:
            self._declared_conflicts.add(
                f"{identity}: conflicting totals {previous} and {value}"
            )

    def _collect_pagination_total(self, capture: CapturedState) -> None:
        totals = {
            int(match.group("total").replace(",", ""))
            for element in capture.elements
            if element.visible
            for text in (_element_text(element),)
            if 0 < len(text) <= 200
            for match in (_PAGINATION_TOTAL.search(text),)
            if match is not None
        }
        if not totals:
            return
        identity = _logical_pagination_url(capture.state.url)
        if len(totals) > 1:
            self._declared_conflicts.add(
                f"{identity}: conflicting pagination totals {sorted(totals)}"
            )
            return
        value = next(iter(totals))
        previous = self._pagination_totals.get(identity)
        if previous is None:
            self._pagination_totals[identity] = value
        elif previous != value:
            self._declared_conflicts.add(
                f"{identity}: conflicting pagination totals {previous} and {value}"
            )

    def _collect_modal_text(self, elements: tuple[ElementSnapshot, ...]) -> None:
        for element in elements:
            if not element.visible or not _inside_dialog(element):
                continue
            text = _element_text(element)
            if len(text) < 3:
                continue
            self._modal_texts.add((_modal_text_priority(element), text))

    def _assess(self) -> tuple[TestcaseContextCoverage, tuple[object, ...]]:
        missing: list[str] = []
        conflicts: list[str] = []
        selected_descriptions: list[tuple[str, str]] = []
        for test_case_id, case in sorted(self._cases.items()):
            case_conflicted = any(len(values) > 1 for values in case.fields.values())
            matching = [
                (priority, text)
                for priority, text in self._modal_texts
                if _contains_identifier(text, test_case_id)
            ]
            best_priority = min(
                (priority for priority, _ in matching),
                default=None,
            )
            descriptions = sorted(
                {text for priority, text in matching if priority == best_priority}
            )
            if not descriptions:
                missing.append(test_case_id)
            else:
                description = sorted(
                    descriptions,
                    key=lambda item: (-len(item), item),
                )[0]
                selected_descriptions.append((test_case_id, description))
                if len(descriptions) > 1:
                    case_conflicted = True
            if case_conflicted:
                conflicts.append(test_case_id)

        declared_total = (
            sum(self._declared_totals.values())
            if self._declared_totals
            else (
                sum(self._pagination_totals.values())
                if self._pagination_totals
                else None
            )
        )
        context_complete = bool(
            declared_total is not None
            and declared_total == len(self._cases)
            and not missing
            and not conflicts
            and not self._declared_conflicts
        )
        coverage = TestcaseContextCoverage(
            declared_test_cases=declared_total,
            test_cases_discovered=len(self._cases),
            descriptions_captured=len(self._cases) - len(missing),
            missing_description_ids=tuple(missing),
            conflicting_test_case_ids=tuple(conflicts),
            declared_total_conflicts=tuple(sorted(self._declared_conflicts)),
            stable_observations=0,
            required_stable_observations=self.required_stable_observations,
            context_complete=context_complete,
        )
        signature: tuple[object, ...] = (
            declared_total,
            tuple(sorted(self._cases)),
            tuple(selected_descriptions),
        )
        return coverage, signature


def _table_root(element: ElementSnapshot) -> Optional[str]:
    path = element.parent_css_path
    if path:
        positions = [
            path.find(marker)
            for marker in (" > thead", " > tbody", " > tfoot")
            if marker in path
        ]
        if positions:
            return path[: min(positions)]
    if element.context.table_id:
        return f"table#{element.context.table_id}"
    return None


def _belongs_to_row(element: ElementSnapshot, row_path: str) -> bool:
    path = element.parent_css_path or ""
    return path == row_path or path.startswith(f"{row_path} >")


def _control_row_path(element: ElementSnapshot) -> Optional[str]:
    path = element.parent_css_path or ""
    positions = [path.find(marker) for marker in (" > td", " > th") if marker in path]
    return path[: min(positions)] if positions else None


def _logical_pagination_url(url: str) -> str:
    parsed = urlsplit(url)
    query = [
        (name, value)
        for name, value in parse_qsl(parsed.query, keep_blank_values=True)
        if _normalize_key(name).replace("_", "") not in _PAGINATION_QUERY_KEYS
        and _normalize_key(name) not in _PAGINATION_QUERY_KEYS
    ]
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query, doseq=True), "")
    )


def _field_keys(headers: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    counts: dict[str, int] = defaultdict(int)
    for index, header in enumerate(headers, start=1):
        base = _normalize_key(header) or f"column_{index}"
        counts[base] += 1
        result.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return tuple(result)


def _normalize_key(value: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", value.lower())
    key = "_".join(tokens)
    if key and key[0].isdigit():
        key = f"column_{key}"
    return key


def _element_text(element: ElementSnapshot) -> str:
    value = element.text or element.accessible_name or element.label or ""
    return " ".join(value.split())


def _attribute(element: ElementSnapshot, name: str) -> Optional[str]:
    for attribute in element.attributes:
        if attribute.name.lower() == name.lower():
            return (
                attribute.value if attribute.value is not None else attribute.safe_value
            )
    return None


def _inside_dialog(element: ElementSnapshot) -> bool:
    if element.context.inside_dialog:
        return True
    if (element.role or "").lower() in {"alertdialog", "dialog"}:
        return True
    return any(
        "role=dialog" in ancestor.lower() or "role=alertdialog" in ancestor.lower()
        for ancestor in element.context.ancestor_summary
    )


def _modal_text_priority(element: ElementSnapshot) -> int:
    semantic_values = {
        value.casefold()
        for name in ("class", "id")
        for value in (_attribute(element, name),)
        if value
    }
    if any("body" in value for value in semantic_values):
        return 0
    if (element.role or "").lower() == "document":
        return 1
    if (element.role or "").lower() in {"alertdialog", "dialog"}:
        return 3
    return 2


def _test_case_id(
    fields: tuple[tuple[str, str], ...],
    controls: tuple[ElementSnapshot, ...],
) -> Optional[str]:
    field = next((item for item in fields if item[0] in _TEST_CASE_ID_KEYS), None)
    if field is None or not field[1]:
        return None
    raw = field[1].strip()
    candidates: set[str] = set()
    for control in controls:
        candidates.update(
            value
            for attribute in control.attributes
            if attribute.name.lower().startswith("data-")
            for value in (
                attribute.value
                if attribute.value is not None
                else attribute.safe_value,
            )
            if value
        )
        label = control.accessible_name or control.label or control.text
        if label:
            candidates.add(label.strip())
        href = _attribute(control, "href")
        if href:
            candidates.update(
                value for _, value in parse_qsl(urlsplit(href).query) if value
            )
    contained = [
        candidate
        for candidate in candidates
        if len(candidate) >= 3 and candidate.casefold() in raw.casefold()
    ]
    if contained:
        return sorted(contained, key=lambda item: (-len(item), item))[0]
    tokens = [
        token
        for token in _IDENTIFIER_TOKEN.findall(raw)
        if token.casefold() not in {"info", "test", "details"}
    ]
    return sorted(tokens, key=lambda item: (-len(item), item))[0] if tokens else raw


def _contains_identifier(text: str, identifier: str) -> bool:
    identifier_characters = r"A-Za-z0-9_.-"
    pattern = re.compile(
        rf"(?<![{identifier_characters}]){re.escape(identifier)}"
        rf"(?![{identifier_characters}])",
        flags=re.IGNORECASE,
    )
    return pattern.search(text) is not None


__all__ = ["TestcaseContextTracker"]
