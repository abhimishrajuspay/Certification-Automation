"""Deterministically normalize crawl evidence into compact portal knowledge."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from scraper.artifact_store import ArtifactStore, ArtifactStoreError
from scraper.models import (
    ActionCandidate,
    ActionStatus,
    CoverageReport,
    CrawlCompletionGoal,
    ElementSnapshot,
    FrameElementCollection,
    FrameSnapshot,
    InteractionTransition,
    NetworkExchange,
    ScrapeRunStatus,
    StateSnapshot,
)

from knowledge.models import (
    EvidencePointer,
    KnowledgeCoverage,
    KnowledgeField,
    NetworkObservation,
    NormalizedControl,
    NormalizedTable,
    NormalizedTableRow,
    PortalKnowledge,
    PortalRoute,
    TestCaseKnowledge,
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
_DEPENDENCY_KEYS = {
    "dependency",
    "dependency_case",
    "dependency_case_id",
    "depends_on",
    "parent_case",
    "prerequisite",
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


class KnowledgeBuildError(RuntimeError):
    """Raised when evidence cannot safely produce normalized knowledge."""


@dataclass(frozen=True)
class KnowledgeBuildConfig:
    """Controls safety and output breadth without portal-specific selectors."""

    allow_incomplete: bool = False
    verify_integrity: bool = True
    include_network: bool = True
    maximum_network_samples: int = 10

    def __post_init__(self) -> None:
        if self.maximum_network_samples <= 0:
            raise ValueError("maximum_network_samples must be positive")


@dataclass
class _RowAccumulator:
    fields: tuple[KnowledgeField, ...]
    controls: dict[str, NormalizedControl] = field(default_factory=dict)
    evidence: dict[tuple[object, ...], EvidencePointer] = field(default_factory=dict)


@dataclass
class _TableAccumulator:
    table_id: str
    logical_url: str
    frame_path: str
    headers: tuple[str, ...]
    rows: dict[str, _RowAccumulator] = field(default_factory=dict)
    evidence: dict[tuple[object, ...], EvidencePointer] = field(default_factory=dict)


@dataclass
class _CaseAccumulator:
    fields: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    controls: dict[str, NormalizedControl] = field(default_factory=dict)
    evidence: dict[tuple[object, ...], EvidencePointer] = field(default_factory=dict)


@dataclass(frozen=True)
class _ModalText:
    text: str
    evidence: EvidencePointer
    priority: int


@dataclass
class _NetworkAccumulator:
    resource_types: set[str] = field(default_factory=set)
    response_statuses: set[int] = field(default_factory=set)
    occurrences: int = 0
    request_body_artifact_ids: set[str] = field(default_factory=set)
    response_body_artifact_ids: set[str] = field(default_factory=set)
    exchange_ids: set[str] = field(default_factory=set)


class PortalKnowledgeBuilder:
    """Build a deterministic, citation-rich package from one crawl store."""

    def __init__(
        self,
        store: ArtifactStore,
        config: Optional[KnowledgeBuildConfig] = None,
    ) -> None:
        self.store = store
        self.config = config or KnowledgeBuildConfig()

    def build(self) -> PortalKnowledge:
        """Normalize the source run after enforcing its completion boundary."""

        run = self.store.reload_manifest()
        coverage_records = tuple(self.store.iter_records(CoverageReport))
        source_coverage = coverage_records[-1] if coverage_records else None
        source_bounded_complete = bool(
            run.status == ScrapeRunStatus.COMPLETED
            and source_coverage is not None
            and source_coverage.bounded_complete
        )
        source_goal_complete = bool(
            run.status == ScrapeRunStatus.COMPLETED
            and source_coverage is not None
            and source_coverage.configured_goal_complete
        )
        source_acceptable = source_bounded_complete or bool(
            source_goal_complete
            and source_coverage is not None
            and source_coverage.completion_goal == CrawlCompletionGoal.TESTCASE_CONTEXT
        )
        if not source_acceptable and not self.config.allow_incomplete:
            raise KnowledgeBuildError(
                "normalization requires a completed crawl goal; use allow_incomplete "
                "only for diagnostic output"
            )
        if self.config.verify_integrity:
            integrity = self.store.verify_integrity(strict=source_acceptable)
            if not integrity.valid:
                raise KnowledgeBuildError(
                    "crawl evidence failed integrity validation: "
                    + "; ".join(integrity.issues)
                )

        states = self._unique_states()
        if not states:
            raise KnowledgeBuildError("crawl contains no state evidence")

        actions = {
            (action.state_id, action.element_id): action
            for action in self.store.iter_records(ActionCandidate)
        }
        transition_status = {
            transition.action_id: transition.status
            for transition in self.store.iter_records(InteractionTransition)
        }

        artifact_cache: dict[str, FrameElementCollection] = {}
        tables: dict[str, _TableAccumulator] = {}
        modal_texts: list[_ModalText] = []
        for state in states:
            for frame in sorted(state.frames, key=lambda item: item.frame_path):
                elements = self._frame_elements(frame, artifact_cache)
                self._collect_tables(
                    state,
                    frame,
                    elements,
                    actions,
                    transition_status,
                    tables,
                )
                if state.modal_count > 0:
                    modal_texts.extend(self._modal_text(state, frame, elements))

        normalized_tables = self._finalize_tables(tables)
        test_cases = self._build_test_cases(normalized_tables, modal_texts)
        declared_total, declared_limitations = self._declared_testcase_total(
            normalized_tables
        )
        if (
            source_coverage is not None
            and source_coverage.completion_goal == CrawlCompletionGoal.TESTCASE_CONTEXT
            and source_coverage.testcase_context is not None
            and source_coverage.testcase_context.declared_test_cases is not None
        ):
            declared_total = source_coverage.testcase_context.declared_test_cases
        routes = self._routes(states)
        network = self._network() if self.config.include_network else ()

        limitations = list(source_coverage.limitations if source_coverage else ())
        limitations.extend(declared_limitations)
        if not source_acceptable:
            limitations.append("knowledge was generated from incomplete crawl evidence")
        missing = tuple(
            case.test_case_id for case in test_cases if case.description is None
        )
        conflicts = tuple(case.test_case_id for case in test_cases if case.conflicts)
        testcase_context_complete = bool(
            declared_total is not None
            and declared_total == len(test_cases)
            and not missing
            and not conflicts
        )
        if (
            source_coverage is not None
            and source_coverage.completion_goal == CrawlCompletionGoal.TESTCASE_CONTEXT
            and source_goal_complete
            and not testcase_context_complete
            and not self.config.allow_incomplete
        ):
            raise KnowledgeBuildError(
                "crawl reported a complete testcase-context goal, but normalized "
                "evidence does not satisfy that gate"
            )
        if missing:
            limitations.append(
                f"{len(missing)} normalized testcases lack captured detail descriptions"
            )
        if declared_total is not None and declared_total != len(test_cases):
            limitations.append(
                "declared testcase total does not match normalized testcase count: "
                f"{declared_total} != {len(test_cases)}"
            )
        if conflicts:
            limitations.append(
                f"{len(conflicts)} testcases contain conflicting evidence"
            )

        normalized_at = max(state.captured_at for state in states)
        started_at = run.started_at or min(state.captured_at for state in states)
        knowledge_coverage = KnowledgeCoverage(
            source_status=run.status,
            source_bounded_complete=source_bounded_complete,
            testcase_context_complete=testcase_context_complete,
            states_examined=len(states),
            routes_normalized=len(routes),
            tables_normalized=len(normalized_tables),
            declared_test_cases=declared_total,
            test_cases_normalized=len(test_cases),
            descriptions_captured=len(test_cases) - len(missing),
            missing_description_ids=missing,
            conflicting_test_case_ids=conflicts,
            limitations=tuple(sorted(set(limitations))),
        )
        return PortalKnowledge(
            source_run_id=run.run_id,
            source_root_url=run.root_url,
            source_started_at=started_at,
            source_ended_at=run.ended_at,
            normalized_at=normalized_at,
            routes=routes,
            tables=normalized_tables,
            test_cases=test_cases,
            network=network,
            coverage=knowledge_coverage,
        )

    def _unique_states(self) -> tuple[StateSnapshot, ...]:
        states: dict[str, StateSnapshot] = {}
        for state in self.store.iter_records(StateSnapshot):
            current = states.get(state.state_id)
            if current is None or state.sequence < current.sequence:
                states[state.state_id] = state
        return tuple(
            sorted(states.values(), key=lambda item: (item.sequence, item.state_id))
        )

    def _frame_elements(
        self,
        frame: FrameSnapshot,
        cache: dict[str, FrameElementCollection],
    ) -> tuple[ElementSnapshot, ...]:
        reference = frame.elements_artifact
        if reference is None:
            return ()
        collection = cache.get(reference.artifact_id)
        if collection is None:
            try:
                collection = FrameElementCollection.model_validate_json(
                    self.store.read_bytes(reference)
                )
            except (ArtifactStoreError, ValueError) as exc:
                raise KnowledgeBuildError(
                    f"invalid frame element artifact {reference.artifact_id}: {exc}"
                ) from exc
            cache[reference.artifact_id] = collection
        return collection.elements

    def _collect_tables(
        self,
        state: StateSnapshot,
        frame: FrameSnapshot,
        elements: tuple[ElementSnapshot, ...],
        actions: dict[tuple[str, str], ActionCandidate],
        transition_status: dict[str, ActionStatus],
        tables: dict[str, _TableAccumulator],
    ) -> None:
        headers_by_root: dict[str, list[ElementSnapshot]] = defaultdict(list)
        cells_by_row: dict[tuple[str, str], list[ElementSnapshot]] = defaultdict(list)
        for element in elements:
            root = _table_root(element)
            if root is None:
                continue
            role = (element.role or "").lower()
            if element.tag == "th" or role == "columnheader":
                headers_by_root[root].append(element)
            elif element.tag == "td" or role == "cell":
                row_path = element.parent_css_path
                if row_path:
                    cells_by_row[(root, row_path)].append(element)

        for root, header_elements in headers_by_root.items():
            headers = tuple(
                _element_text(element) or f"Column {index}"
                for index, element in enumerate(header_elements, start=1)
            )
            keys = _field_keys(headers)
            logical_url = _logical_url(state.url)
            table_id = _hash_json(
                {
                    "logical_url": logical_url,
                    "frame_path": frame.frame_path,
                    "root": root,
                    "headers": keys,
                }
            )
            table = tables.setdefault(
                table_id,
                _TableAccumulator(
                    table_id=table_id,
                    logical_url=logical_url,
                    frame_path=frame.frame_path,
                    headers=headers,
                ),
            )
            table_pointer = _evidence_pointer(
                state,
                frame,
                tuple(element.element_id for element in header_elements),
            )
            table.evidence[_evidence_key(table_pointer)] = table_pointer

            for (row_root, row_path), cells in cells_by_row.items():
                if row_root != root or not cells:
                    continue
                row_fields = tuple(
                    KnowledgeField(
                        key=keys[index] if index < len(keys) else f"column_{index + 1}",
                        label=(
                            headers[index]
                            if index < len(headers)
                            else f"Column {index + 1}"
                        ),
                        value=_element_text(cell),
                    )
                    for index, cell in enumerate(cells)
                )
                row_controls = tuple(
                    self._control(state, element, actions, transition_status)
                    for element in elements
                    if element.interactive and _belongs_to_row(element, row_path)
                )
                row_id = _hash_json(
                    {
                        "table_id": table_id,
                        "fields": [
                            {"key": item.key, "value": item.value}
                            for item in row_fields
                        ],
                    }
                )
                row = table.rows.setdefault(row_id, _RowAccumulator(fields=row_fields))
                for control in row_controls:
                    row.controls.setdefault(control.element_id, control)
                pointer = _evidence_pointer(
                    state,
                    frame,
                    tuple(
                        dict.fromkeys(
                            [
                                *(cell.element_id for cell in cells),
                                *(control.element_id for control in row_controls),
                            ]
                        )
                    ),
                )
                row.evidence[_evidence_key(pointer)] = pointer

    def _control(
        self,
        state: StateSnapshot,
        element: ElementSnapshot,
        actions: dict[tuple[str, str], ActionCandidate],
        transition_status: dict[str, ActionStatus],
    ) -> NormalizedControl:
        candidate = actions.get((state.state_id, element.element_id))
        identifiers = tuple(
            sorted(
                {
                    value
                    for attribute in element.attributes
                    if attribute.name.lower().startswith("data-")
                    for value in (
                        _captured_value(attribute.value, attribute.safe_value),
                    )
                    if value
                }
            )
        )
        href = _attribute(element, "href")
        return NormalizedControl(
            element_id=element.element_id,
            role=element.role,
            label=element.accessible_name or element.label or element.text,
            title=element.title,
            href=href,
            identifiers=identifiers,
            action_kind=candidate.kind if candidate else None,
            risk=candidate.risk if candidate else None,
            action_status=(
                transition_status.get(candidate.action_id, candidate.status)
                if candidate
                else None
            ),
            policy_rule=candidate.policy_rule if candidate else None,
        )

    def _modal_text(
        self,
        state: StateSnapshot,
        frame: FrameSnapshot,
        elements: tuple[ElementSnapshot, ...],
    ) -> tuple[_ModalText, ...]:
        candidates: dict[str, _ModalText] = {}
        for element in elements:
            if not element.visible or not _inside_dialog(element):
                continue
            text = _element_text(element)
            if not text or len(text) < 3:
                continue
            pointer = _evidence_pointer(state, frame, (element.element_id,))
            candidates.setdefault(
                text,
                _ModalText(
                    text=text,
                    evidence=pointer,
                    priority=_modal_text_priority(element),
                ),
            )
        if not candidates:
            return ()
        return tuple(
            sorted(
                candidates.values(),
                key=lambda item: (item.priority, -len(item.text), item.text),
            )
        )

    @staticmethod
    def _finalize_tables(
        tables: dict[str, _TableAccumulator],
    ) -> tuple[NormalizedTable, ...]:
        result: list[NormalizedTable] = []
        for table in sorted(tables.values(), key=lambda item: item.table_id):
            rows = tuple(
                NormalizedTableRow(
                    row_id=row_id,
                    fields=row.fields,
                    controls=tuple(
                        sorted(row.controls.values(), key=lambda item: item.element_id)
                    ),
                    evidence=tuple(
                        sorted(row.evidence.values(), key=_evidence_sort_key)
                    ),
                )
                for row_id, row in sorted(table.rows.items())
            )
            if not rows:
                continue
            result.append(
                NormalizedTable(
                    table_id=table.table_id,
                    logical_url=table.logical_url,
                    frame_path=table.frame_path,
                    headers=table.headers,
                    rows=rows,
                    evidence=tuple(
                        sorted(table.evidence.values(), key=_evidence_sort_key)
                    ),
                )
            )
        return tuple(result)

    def _build_test_cases(
        self,
        tables: tuple[NormalizedTable, ...],
        modal_texts: list[_ModalText],
    ) -> tuple[TestCaseKnowledge, ...]:
        cases: dict[str, _CaseAccumulator] = {}
        for table in tables:
            header_keys = {_normalize_key(header) for header in table.headers}
            if not (header_keys & _TEST_CASE_ID_KEYS):
                continue
            if len(header_keys & _TEST_CASE_SIGNAL_KEYS) < 1:
                continue
            for row in table.rows:
                test_case_id = _test_case_id(row)
                if not test_case_id:
                    continue
                case = cases.setdefault(test_case_id, _CaseAccumulator())
                for item in row.fields:
                    field_values = case.fields.setdefault(
                        item.key,
                        {"labels": set(), "values": set()},
                    )
                    field_values["labels"].add(item.label)
                    field_values["values"].add(
                        test_case_id if item.key in _TEST_CASE_ID_KEYS else item.value
                    )
                for control in row.controls:
                    case.controls.setdefault(control.element_id, control)
                for pointer in row.evidence:
                    case.evidence[_evidence_key(pointer)] = pointer

        known_ids = tuple(sorted(cases))
        result: list[TestCaseKnowledge] = []
        for test_case_id in known_ids:
            case = cases[test_case_id]
            conflicts: list[str] = []
            fields: list[KnowledgeField] = []
            for key, data in sorted(case.fields.items()):
                values = sorted(data["values"])
                labels = sorted(data["labels"])
                if len(values) > 1:
                    conflicts.append(
                        f"field {key!r} has conflicting values: {values!r}"
                    )
                fields.append(
                    KnowledgeField(
                        key=key,
                        label=labels[0],
                        value=_preferred_value(values),
                    )
                )

            dependencies = _dependencies(test_case_id, tuple(fields), known_ids)
            matching_descriptions = [
                item
                for item in modal_texts
                if _contains_identifier(item.text, test_case_id)
            ]
            best_priority = min(
                (item.priority for item in matching_descriptions),
                default=None,
            )
            preferred_descriptions = [
                item for item in matching_descriptions if item.priority == best_priority
            ]
            description_values = sorted({item.text for item in preferred_descriptions})
            description = (
                sorted(description_values, key=lambda item: (-len(item), item))[0]
                if description_values
                else None
            )
            if len(description_values) > 1:
                conflicts.append(
                    f"description has {len(description_values)} distinct captured values"
                )
            description_evidence = tuple(
                sorted(
                    {
                        _evidence_key(item.evidence): item.evidence
                        for item in preferred_descriptions
                        if item.text == description
                    }.values(),
                    key=_evidence_sort_key,
                )
            )
            result.append(
                TestCaseKnowledge(
                    test_case_id=test_case_id,
                    fields=tuple(fields),
                    dependency_case_ids=dependencies,
                    description=description,
                    controls=tuple(
                        sorted(case.controls.values(), key=lambda item: item.element_id)
                    ),
                    evidence=tuple(
                        sorted(case.evidence.values(), key=_evidence_sort_key)
                    ),
                    description_evidence=description_evidence,
                    conflicts=tuple(conflicts),
                )
            )
        return tuple(result)

    @staticmethod
    def _declared_testcase_total(
        tables: tuple[NormalizedTable, ...],
    ) -> tuple[Optional[int], tuple[str, ...]]:
        totals: dict[str, int] = {}
        limitations: list[str] = []
        for table in tables:
            for row in table.rows:
                total_field = next(
                    (
                        field
                        for field in row.fields
                        if field.key in _TOTAL_TEST_CASE_KEYS
                    ),
                    None,
                )
                if total_field is None or not total_field.value.isdigit():
                    continue
                identity = next(
                    (
                        field.value
                        for field in row.fields
                        if field.key not in _TOTAL_TEST_CASE_KEYS | _ORDINAL_KEYS
                        and field.value
                        and not field.value.isdigit()
                    ),
                    row.row_id,
                )
                value = int(total_field.value)
                previous = totals.get(identity)
                if previous is not None and previous != value:
                    limitations.append(
                        f"declared testcase total for {identity!r} conflicts: "
                        f"{previous} != {value}"
                    )
                else:
                    totals[identity] = value
        return (sum(totals.values()), tuple(limitations)) if totals else (None, ())

    @staticmethod
    def _routes(states: tuple[StateSnapshot, ...]) -> tuple[PortalRoute, ...]:
        grouped: dict[str, dict[str, set[str]]] = {}
        for state in states:
            route = grouped.setdefault(state.url, {"titles": set(), "states": set()})
            if state.title:
                route["titles"].add(state.title)
            route["states"].add(state.state_id)
        return tuple(
            PortalRoute(
                url=url,
                titles=tuple(sorted(values["titles"])),
                state_ids=tuple(sorted(values["states"])),
            )
            for url, values in sorted(grouped.items())
        )

    def _network(self) -> tuple[NetworkObservation, ...]:
        grouped: dict[tuple[str, str], _NetworkAccumulator] = {}
        for exchange in self.store.iter_records(NetworkExchange):
            key = (exchange.method.upper(), exchange.url)
            item = grouped.setdefault(key, _NetworkAccumulator())
            item.occurrences += 1
            item.exchange_ids.add(exchange.exchange_id)
            if exchange.resource_type:
                item.resource_types.add(exchange.resource_type)
            if exchange.response_status is not None:
                item.response_statuses.add(exchange.response_status)
            if exchange.request_body is not None:
                item.request_body_artifact_ids.add(exchange.request_body.artifact_id)
            if exchange.response_body is not None:
                item.response_body_artifact_ids.add(exchange.response_body.artifact_id)
        return tuple(
            NetworkObservation(
                method=method,
                url=url,
                resource_types=tuple(sorted(item.resource_types)),
                response_statuses=tuple(sorted(item.response_statuses)),
                occurrences=item.occurrences,
                request_body_artifact_ids=tuple(sorted(item.request_body_artifact_ids)),
                response_body_artifact_ids=tuple(
                    sorted(item.response_body_artifact_ids)
                ),
                sample_exchange_ids=tuple(sorted(item.exchange_ids))[
                    : self.config.maximum_network_samples
                ],
            )
            for (method, url), item in sorted(grouped.items())
        )


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


def _logical_url(url: str) -> str:
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


def _element_text(element: ElementSnapshot) -> str:
    value = element.text or element.accessible_name or element.label or ""
    return " ".join(value.split())


def _attribute(element: ElementSnapshot, name: str) -> Optional[str]:
    for attribute in element.attributes:
        if attribute.name.lower() == name.lower():
            return _captured_value(attribute.value, attribute.safe_value) or None
    return None


def _captured_value(value: Optional[str], safe_value: Optional[str]) -> str:
    return value if value is not None else safe_value or ""


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


def _evidence_pointer(
    state: StateSnapshot,
    frame: FrameSnapshot,
    element_ids: tuple[str, ...],
) -> EvidencePointer:
    artifacts = (
        (frame.elements_artifact.artifact_id,)
        if frame.elements_artifact is not None
        else ()
    )
    return EvidencePointer(
        state_id=state.state_id,
        state_sequence=state.sequence,
        url=state.url,
        frame_id=frame.frame_id,
        element_ids=tuple(dict.fromkeys(element_ids)),
        artifact_ids=artifacts,
    )


def _evidence_key(pointer: EvidencePointer) -> tuple[object, ...]:
    return (
        pointer.state_id,
        pointer.frame_id,
        pointer.element_ids,
        pointer.artifact_ids,
    )


def _evidence_sort_key(pointer: EvidencePointer) -> tuple[object, ...]:
    return (
        pointer.state_sequence,
        pointer.state_id,
        pointer.frame_id,
        pointer.element_ids,
    )


def _test_case_id(row: NormalizedTableRow) -> Optional[str]:
    field = next((item for item in row.fields if item.key in _TEST_CASE_ID_KEYS), None)
    if field is None or not field.value:
        return None
    raw = field.value.strip()
    candidates: set[str] = set()
    for control in row.controls:
        candidates.update(value for value in control.identifiers if value)
        if control.label:
            candidates.add(control.label.strip())
        if control.href:
            candidates.update(
                value for _, value in parse_qsl(urlsplit(control.href).query) if value
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


def _dependencies(
    test_case_id: str,
    fields: tuple[KnowledgeField, ...],
    known_ids: tuple[str, ...],
) -> tuple[str, ...]:
    text = " ".join(
        field.value for field in fields if field.key in _DEPENDENCY_KEYS and field.value
    )
    if not text:
        return ()
    matches = {
        candidate
        for candidate in known_ids
        if candidate != test_case_id and _contains_identifier(text, candidate)
    }
    if not matches:
        matches.update(
            token
            for token in _IDENTIFIER_TOKEN.findall(text)
            if token != test_case_id and not token.isdigit()
        )
    return tuple(sorted(matches))


def _contains_identifier(text: str, identifier: str) -> bool:
    identifier_characters = r"A-Za-z0-9_.-"
    pattern = re.compile(
        rf"(?<![{identifier_characters}]){re.escape(identifier)}"
        rf"(?![{identifier_characters}])",
        flags=re.IGNORECASE,
    )
    return pattern.search(text) is not None


def _preferred_value(values: list[str]) -> str:
    nonempty = [value for value in values if value]
    return nonempty[0] if nonempty else values[0] if values else ""


def _hash_json(value: object) -> str:
    data = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "KnowledgeBuildConfig",
    "KnowledgeBuildError",
    "PortalKnowledgeBuilder",
]
