"""Deterministically render validated Phase 9 specifications as Postman v2.1."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qsl, quote
from uuid import UUID, uuid5

from synthesis.models import (
    AssertionOperator,
    AssertionSource,
    GeneratedValueKind,
    RequestBodyMode,
    ResponseAssertion,
    SynthesisDisposition,
    SynthesisExportManifest,
    SynthesisPackage,
    TemplateVariableBinding,
    TemplateVariableSource,
    TestCaseExecutionSpec,
)

from postman.models import (
    PostmanBody,
    PostmanCollection,
    PostmanCoverage,
    PostmanEnvironment,
    PostmanEnvironmentValue,
    PostmanEvent,
    PostmanFormParameter,
    PostmanHeader,
    PostmanInfo,
    PostmanItem,
    PostmanPackage,
    PostmanQueryParameter,
    PostmanRequest,
    PostmanScript,
    PostmanUrl,
    PostmanVariable,
    RenderedTestCase,
    SkippedTestCase,
)


_UUID_NAMESPACE = UUID("eb190026-2f5f-4f3c-85b8-167b654ab51c")


class PostmanBuildError(RuntimeError):
    """Raised when Phase 9 output cannot be rendered without guessing."""


@dataclass(frozen=True)
class PostmanBuildConfig:
    base_url_variable: str = "base_url"
    allow_partial: bool = False

    def __post_init__(self) -> None:
        if (
            not self.base_url_variable
            or not self.base_url_variable.replace("_", "a").isalnum()
        ):
            raise ValueError("base_url_variable must contain letters, digits, or _")


@dataclass(frozen=True)
class LoadedSynthesis:
    path: Path
    sha256: str
    synthesis: SynthesisPackage


@dataclass(frozen=True)
class PostmanBuildResult:
    collection: PostmanCollection
    environment: PostmanEnvironment
    report: PostmanPackage


class PostmanBuilder:
    """Render only fully validated ready specifications in dependency order."""

    def __init__(
        self,
        synthesis: SynthesisPackage,
        synthesis_sha256: str,
        *,
        config: Optional[PostmanBuildConfig] = None,
    ) -> None:
        self.synthesis = synthesis
        self.synthesis_sha256 = synthesis_sha256
        self.config = config or PostmanBuildConfig()
        self.specifications = {
            spec.test_case_id: spec for spec in synthesis.specifications
        }

    def build(self) -> PostmanBuildResult:
        if (
            not self.synthesis.coverage.execution_ready
            and not self.config.allow_partial
        ):
            raise PostmanBuildError(
                "Postman generation requires execution_ready Phase 9 output; use "
                "allow_partial only to inspect a collection containing the safe subset"
            )

        ordered_specs, skip_reasons = self._resolve_renderable_specs()
        if skip_reasons and not self.config.allow_partial:
            first = next(iter(skip_reasons.items()))
            raise PostmanBuildError(
                f"cannot render all testcases: {first[0]}: {first[1]}"
            )

        generated_at = datetime.now(timezone.utc)
        collection_id = str(
            uuid5(
                _UUID_NAMESPACE,
                f"collection:{self.synthesis.source_run_id}:{self.synthesis_sha256}",
            )
        )
        environment_id = str(
            uuid5(
                _UUID_NAMESPACE,
                f"environment:{self.synthesis.source_run_id}:{self.synthesis_sha256}",
            )
        )
        dependency_names = self._dependency_variable_names(ordered_specs)
        extractions = self._dependency_extractions(ordered_specs, dependency_names)
        status_names = {
            spec.test_case_id: _status_variable(spec.test_case_id)
            for spec in ordered_specs
        }
        runtime_variable_names = tuple(status_names.values()) + tuple(
            dependency_names.values()
        )
        items = tuple(
            self._render_item(
                spec,
                status_names=status_names,
                dependency_names=dependency_names,
                extractions=extractions.get(spec.test_case_id, ()),
                initialize_run=index == 0,
                runtime_variable_names=runtime_variable_names,
            )
            for index, spec in enumerate(ordered_specs)
        )
        collection_variables = [
            PostmanVariable(
                key="__cz_run_id",
                value="",
                description="Per-iteration marker initialized by the first request",
            ),
            *(
                PostmanVariable(
                    key=status_names[spec.test_case_id],
                    value="",
                    description=f"Execution status for {spec.test_case_id}",
                )
                for spec in ordered_specs
            ),
        ]
        collection_variables.extend(
            PostmanVariable(
                key=postman_name,
                value="",
                description=(
                    f"Dependency output {binding.name} for {spec.test_case_id}"
                ),
            )
            for spec in ordered_specs
            for binding in spec.variable_bindings
            if binding.source == TemplateVariableSource.DEPENDENCY
            for postman_name in (dependency_names[(spec.test_case_id, binding.name)],)
        )
        collection = PostmanCollection(
            info=PostmanInfo(
                name=f"CZ Certification - {self.synthesis.source_run_id}",
                postman_id=collection_id,
                description=(
                    "Generated deterministically from cited Phase 9 execution "
                    f"specifications for {self.synthesis.source_run_id}."
                ),
            ),
            item=items,
            variable=tuple(collection_variables),
        )
        environment = PostmanEnvironment(
            id=environment_id,
            name=f"CZ Certification - {self.synthesis.source_run_id}",
            values=self._environment_values(ordered_specs),
            exported_at=generated_at,
        )

        rendered_records = tuple(
            RenderedTestCase(
                test_case_id=spec.test_case_id,
                item_id=items[index].id,
                order=index,
            )
            for index, spec in enumerate(ordered_specs)
        )
        skipped_records = self._ordered_skipped(skip_reasons)
        ready_source_cases = sum(
            spec.disposition == SynthesisDisposition.READY
            for spec in self.synthesis.specifications
        )
        total = self.synthesis.coverage.test_cases
        limitations: list[str] = []
        if not self.synthesis.coverage.synthesis_complete:
            limitations.append("source Phase 9 synthesis is incomplete")
        if not self.synthesis.coverage.execution_ready:
            limitations.append("source Phase 9 package requires review or is blocked")
        if skipped_records:
            limitations.append(
                f"{len(skipped_records)} testcases were omitted from the collection"
            )
        generation_complete = bool(
            self.synthesis.coverage.synthesis_complete
            and self.synthesis.coverage.execution_ready
            and len(rendered_records) == total
            and not skipped_records
        )
        report = PostmanPackage(
            source_run_id=self.synthesis.source_run_id,
            source_synthesis_sha256=self.synthesis_sha256,
            generated_at=generated_at,
            collection_id=collection_id,
            environment_id=environment_id,
            rendered_test_cases=rendered_records,
            skipped_test_cases=skipped_records,
            coverage=PostmanCoverage(
                source_synthesis_complete=(self.synthesis.coverage.synthesis_complete),
                source_execution_ready=self.synthesis.coverage.execution_ready,
                test_cases=total,
                ready_source_cases=ready_source_cases,
                rendered=len(rendered_records),
                skipped=len(skipped_records),
                generation_complete=generation_complete,
                limitations=tuple(limitations),
            ),
        )
        return PostmanBuildResult(
            collection=collection,
            environment=environment,
            report=report,
        )

    def _resolve_renderable_specs(
        self,
    ) -> tuple[tuple[TestCaseExecutionSpec, ...], dict[str, str]]:
        skip_reasons: dict[str, str] = {
            test_case_id: "model synthesis did not produce a specification"
            for test_case_id in self.synthesis.coverage.failed_test_case_ids
        }
        candidates: dict[str, TestCaseExecutionSpec] = {}
        for spec in self.synthesis.specifications:
            if spec.disposition == SynthesisDisposition.READY:
                candidates[spec.test_case_id] = spec
            else:
                skip_reasons[spec.test_case_id] = (
                    f"Phase 9 disposition is {spec.disposition.value}"
                )

        changed = True
        while changed:
            changed = False
            for case_id, spec in tuple(candidates.items()):
                unavailable = [
                    dependency
                    for dependency in spec.dependency_case_ids
                    if dependency not in candidates
                ]
                if unavailable:
                    candidates.pop(case_id)
                    skip_reasons[case_id] = (
                        "dependency is unavailable for execution: "
                        + ", ".join(unavailable)
                    )
                    changed = True

        ordered, cyclic = _topological_order(candidates, self.synthesis.specifications)
        for case_id in cyclic:
            skip_reasons[case_id] = (
                "dependency graph contains a cycle or cyclic ancestor"
            )
        return tuple(candidates[case_id] for case_id in ordered), skip_reasons

    @staticmethod
    def _dependency_variable_names(
        specifications: tuple[TestCaseExecutionSpec, ...],
    ) -> dict[tuple[str, str], str]:
        return {
            (spec.test_case_id, binding.name): (
                "__cz_dep_"
                + hashlib.sha256(
                    f"{spec.test_case_id}:{binding.name}".encode("utf-8")
                ).hexdigest()[:16]
            )
            for spec in specifications
            for binding in spec.variable_bindings
            if binding.source == TemplateVariableSource.DEPENDENCY
        }

    @staticmethod
    def _dependency_extractions(
        specifications: tuple[TestCaseExecutionSpec, ...],
        dependency_names: dict[tuple[str, str], str],
    ) -> dict[str, tuple[tuple[str, TemplateVariableBinding, str], ...]]:
        by_parent: dict[str, list[tuple[str, TemplateVariableBinding, str]]] = (
            defaultdict(list)
        )
        rendered_ids = {spec.test_case_id for spec in specifications}
        for spec in specifications:
            for binding in spec.variable_bindings:
                if binding.source != TemplateVariableSource.DEPENDENCY:
                    continue
                if binding.dependency_case_id not in rendered_ids:
                    raise PostmanBuildError(
                        f"{spec.test_case_id} depends on an unrendered variable source"
                    )
                by_parent[binding.dependency_case_id].append(
                    (
                        dependency_names[(spec.test_case_id, binding.name)],
                        binding,
                        spec.test_case_id,
                    )
                )
        return {
            parent: tuple(
                sorted(values, key=lambda item: (item[2], item[1].name, item[0]))
            )
            for parent, values in by_parent.items()
        }

    def _render_item(
        self,
        spec: TestCaseExecutionSpec,
        *,
        status_names: dict[str, str],
        dependency_names: dict[tuple[str, str], str],
        extractions: tuple[tuple[str, TemplateVariableBinding, str], ...],
        initialize_run: bool,
        runtime_variable_names: tuple[str, ...],
    ) -> PostmanItem:
        if spec.request is None:
            raise PostmanBuildError(
                f"ready testcase {spec.test_case_id} has no request"
            )
        item_id = str(
            uuid5(
                _UUID_NAMESPACE,
                f"item:{self.synthesis_sha256}:{spec.test_case_id}",
            )
        )

        def render_value(value: str) -> str:
            return _replace_dependency_variables(
                spec,
                value,
                dependency_names,
            )

        headers = [
            PostmanHeader(
                key=header.name,
                value=render_value(header.value),
                description=header.description,
                disabled=not header.enabled,
            )
            for header in spec.request.headers
        ]
        body = _render_body(spec, render_value)
        if (
            body is not None
            and spec.request.body.content_type
            and not any(header.key.casefold() == "content-type" for header in headers)
        ):
            headers.append(
                PostmanHeader(
                    key="Content-Type",
                    value=spec.request.body.content_type,
                    description="From the validated Phase 9 body specification",
                )
            )
        query = tuple(
            PostmanQueryParameter(
                key=parameter.name,
                value=render_value(parameter.value),
                description=parameter.description,
                disabled=not parameter.enabled,
            )
            for parameter in spec.request.query_parameters
        )
        rendered_path = render_value(spec.request.path)
        raw = f"{{{{{self.config.base_url_variable}}}}}{rendered_path}"
        enabled_query = [parameter for parameter in query if not parameter.disabled]
        if enabled_query:
            raw += "?" + "&".join(
                f"{quote(parameter.key, safe='[]')}="
                f"{quote(parameter.value, safe='{}[]:/,')}"
                for parameter in enabled_query
            )
        request = PostmanRequest(
            method=spec.request.method.value,
            header=tuple(headers),
            body=body,
            url=PostmanUrl(
                raw=raw,
                host=(f"{{{{{self.config.base_url_variable}}}}}",),
                path=tuple(
                    segment
                    for segment in rendered_path.lstrip("/").split("/")
                    if segment
                ),
                query=query,
            ),
            description=(
                f"{spec.rationale}\n\nEvidence snippets: "
                + ", ".join(spec.evidence_snippet_ids)
            ),
        )
        prerequest = _prerequest_script(
            spec,
            status_names=status_names,
            own_status=status_names[spec.test_case_id],
            initialize_run=initialize_run,
            runtime_variable_names=runtime_variable_names,
        )
        tests = _test_script(
            spec,
            own_status=status_names[spec.test_case_id],
            extractions=extractions,
            dependency_names=dependency_names,
        )
        return PostmanItem(
            name=f"{spec.test_case_id} - {spec.title}",
            id=item_id,
            event=(
                PostmanEvent(
                    listen="prerequest",
                    script=PostmanScript(exec=tuple(prerequest)),
                ),
                PostmanEvent(
                    listen="test",
                    script=PostmanScript(exec=tuple(tests)),
                ),
            ),
            request=request,
        )

    def _environment_values(
        self,
        specifications: tuple[TestCaseExecutionSpec, ...],
    ) -> tuple[PostmanEnvironmentValue, ...]:
        bindings: dict[str, TemplateVariableBinding] = {}
        for spec in specifications:
            for binding in spec.variable_bindings:
                if binding.source != TemplateVariableSource.ENVIRONMENT:
                    continue
                if binding.name == self.config.base_url_variable:
                    raise PostmanBuildError(
                        f"environment binding {binding.name} conflicts with base URL"
                    )
                existing = bindings.get(binding.name)
                if existing is not None and existing.sensitive != binding.sensitive:
                    raise PostmanBuildError(
                        f"environment binding {binding.name} has conflicting sensitivity"
                    )
                bindings.setdefault(binding.name, binding)
        values = [
            PostmanEnvironmentValue(
                key=self.config.base_url_variable,
                value="",
                type="default",
            )
        ]
        values.extend(
            PostmanEnvironmentValue(
                key=name,
                value="",
                type="secret" if binding.sensitive else "default",
            )
            for name, binding in sorted(bindings.items())
        )
        return tuple(values)

    def _ordered_skipped(
        self,
        reasons: dict[str, str],
    ) -> tuple[SkippedTestCase, ...]:
        source_order = [
            spec.test_case_id for spec in self.synthesis.specifications
        ] + list(self.synthesis.coverage.failed_test_case_ids)
        return tuple(
            SkippedTestCase(test_case_id=case_id, reason=reasons[case_id])
            for case_id in source_order
            if case_id in reasons
        )


def load_synthesis(
    source: Path,
    *,
    verify_manifest: bool = True,
) -> LoadedSynthesis:
    expanded = source.expanduser().resolve()
    path = expanded / "synthesis.json" if expanded.is_dir() else expanded
    if not path.is_file():
        raise PostmanBuildError(f"synthesis file does not exist: {path}")
    try:
        data = path.read_bytes()
        synthesis = SynthesisPackage.model_validate_json(data)
    except (OSError, ValueError) as exc:
        raise PostmanBuildError(f"failed to load synthesis package: {exc}") from exc
    digest = hashlib.sha256(data).hexdigest()

    manifest_path = path.parent / "manifest.json"
    if verify_manifest and manifest_path.is_file():
        try:
            manifest = SynthesisExportManifest.model_validate_json(
                manifest_path.read_bytes()
            )
        except (OSError, ValueError) as exc:
            raise PostmanBuildError(f"invalid synthesis manifest: {exc}") from exc
        expected = next(
            (item.sha256 for item in manifest.files if item.name == path.name),
            None,
        )
        if expected is None or expected != digest:
            raise PostmanBuildError(
                "synthesis digest does not match its export manifest"
            )
        if manifest.source_run_id != synthesis.source_run_id:
            raise PostmanBuildError(
                "synthesis run ID does not match its export manifest"
            )
    return LoadedSynthesis(path=path, sha256=digest, synthesis=synthesis)


def _topological_order(
    candidates: dict[str, TestCaseExecutionSpec],
    source_order: tuple[TestCaseExecutionSpec, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    positions = {spec.test_case_id: index for index, spec in enumerate(source_order)}
    indegree = {case_id: 0 for case_id in candidates}
    children: dict[str, list[str]] = defaultdict(list)
    for case_id, spec in candidates.items():
        for dependency in spec.dependency_case_ids:
            if dependency in candidates:
                indegree[case_id] += 1
                children[dependency].append(case_id)
    queue = [
        (positions[case_id], case_id)
        for case_id, value in indegree.items()
        if value == 0
    ]
    heapq.heapify(queue)
    ordered: list[str] = []
    while queue:
        _, case_id = heapq.heappop(queue)
        ordered.append(case_id)
        for child in sorted(children[case_id], key=lambda item: positions[item]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(queue, (positions[child], child))
    cyclic = tuple(case_id for case_id in candidates if case_id not in set(ordered))
    return tuple(ordered), cyclic


def _replace_dependency_variables(
    spec: TestCaseExecutionSpec,
    value: str,
    names: dict[tuple[str, str], str],
) -> str:
    rendered = value
    for binding in spec.variable_bindings:
        if binding.source == TemplateVariableSource.DEPENDENCY:
            rendered = rendered.replace(
                f"{{{{{binding.name}}}}}",
                f"{{{{{names[(spec.test_case_id, binding.name)]}}}}}",
            )
    return rendered


def _render_body(
    spec: TestCaseExecutionSpec,
    render_value: Callable[[str], str],
) -> Optional[PostmanBody]:
    if spec.request is None or spec.request.body.mode == RequestBodyMode.NONE:
        return None
    template = spec.request.body.template
    if template is None:
        raise PostmanBuildError(f"{spec.test_case_id} request body has no template")
    rendered = render_value(template)
    if spec.request.body.mode == RequestBodyMode.FORM_URLENCODED:
        parameters = tuple(
            PostmanFormParameter(key=key, value=value)
            for key, value in parse_qsl(rendered, keep_blank_values=True)
        )
        if not parameters:
            raise PostmanBuildError(
                f"{spec.test_case_id} form body has no key/value parameters"
            )
        return PostmanBody(mode="urlencoded", urlencoded=parameters)
    language = {
        RequestBodyMode.JSON: "json",
        RequestBodyMode.XML: "xml",
        RequestBodyMode.RAW: "text",
    }[spec.request.body.mode]
    return PostmanBody(
        mode="raw",
        raw=rendered,
        options={"raw": {"language": language}},
    )


def _prerequest_script(
    spec: TestCaseExecutionSpec,
    *,
    status_names: dict[str, str],
    own_status: str,
    initialize_run: bool,
    runtime_variable_names: tuple[str, ...],
) -> list[str]:
    dependency_statuses = [
        status_names[case_id] for case_id in spec.dependency_case_ids
    ]
    lines: list[str] = []
    if initialize_run:
        lines.extend(
            [
                'pm.collectionVariables.set("__cz_run_id", '
                'pm.variables.replaceIn("{{$guid}}"));',
                f"const __czRuntimeVariables = {_js(runtime_variable_names)};",
                "__czRuntimeVariables.forEach((key) => "
                'pm.collectionVariables.set(key, ""));',
            ]
        )
    lines.extend(
        [
            f"const __czDependencyStatuses = {_js(dependency_statuses)};",
            "const __czUnmet = __czDependencyStatuses.filter((key) => "
            'pm.collectionVariables.get(key) !== "passed");',
            "if (__czUnmet.length > 0) {",
            '  console.warn("Skipping testcase because dependencies did not pass", '
            "__czUnmet);",
            "  pm.execution.skipRequest();",
            "} else {",
            f'  pm.collectionVariables.set({_js(own_status)}, "running");',
        ]
    )
    for binding in spec.variable_bindings:
        if binding.source in {
            TemplateVariableSource.PORTAL_FIELD,
            TemplateVariableSource.EVIDENCE_LITERAL,
        }:
            lines.append(
                f"  pm.variables.set({_js(binding.name)}, {_js(binding.value)});"
            )
        elif binding.source == TemplateVariableSource.GENERATED:
            dynamic = {
                GeneratedValueKind.UUID: "{{$guid}}",
                GeneratedValueKind.TIMESTAMP: "{{$timestamp}}",
                GeneratedValueKind.RANDOM_ALPHANUMERIC: "{{$randomAlphaNumeric}}",
            }[binding.generator]
            lines.append(
                f"  pm.variables.set({_js(binding.name)}, "
                f"pm.variables.replaceIn({_js(dynamic)}));"
            )
    lines.append("}")
    return lines


def _test_script(
    spec: TestCaseExecutionSpec,
    *,
    own_status: str,
    extractions: tuple[tuple[str, TemplateVariableBinding, str], ...],
    dependency_names: dict[tuple[str, str], str],
) -> list[str]:
    needs_json = any(
        assertion.source == AssertionSource.JSON_PATH for assertion in spec.assertions
    ) or any(
        binding.extraction_source == AssertionSource.JSON_PATH
        for _, binding, _ in extractions
    )
    needs_xml = any(
        assertion.source == AssertionSource.XPATH for assertion in spec.assertions
    ) or any(
        binding.extraction_source == AssertionSource.XPATH
        for _, binding, _ in extractions
    )
    lines = ["let __czPassed = true;"]
    if needs_json:
        lines.extend(
            [
                "let __czJson;",
                "function __czJsonValue(path) {",
                "  if (__czJson === undefined) { __czJson = pm.response.json(); }",
                '  const normalized = path === "$" ? "" : '
                'path.replace(/^\\$\\.?/, "");',
                "  return normalized ? _.get(__czJson, normalized) : __czJson;",
                "}",
            ]
        )
    if needs_xml:
        lines.extend(
            [
                "let __czXml;",
                "function __czXmlValue(path) {",
                "  if (__czXml === undefined) { __czXml = xml2Json(pm.response.text()); }",
                '  const segments = path.split("/").filter(Boolean);',
                "  return segments.reduce((current, key) => "
                "current == null ? undefined : current[key], __czXml);",
                "}",
            ]
        )
    for index, assertion in enumerate(spec.assertions):
        rendered_expected = (
            _replace_dependency_variables(
                spec,
                assertion.expected,
                dependency_names,
            )
            if assertion.expected is not None
            else None
        )
        lines.extend(
            _assertion_script(
                index,
                assertion,
                expected=rendered_expected,
            )
        )
    for index, (variable_name, binding, child_id) in enumerate(extractions):
        lines.extend(
            _extraction_script(
                index,
                variable_name,
                binding,
                child_id,
            )
        )
    lines.append(
        f"pm.collectionVariables.set({_js(own_status)}, "
        '__czPassed ? "passed" : "failed");'
    )
    return lines


def _assertion_script(
    index: int,
    assertion: ResponseAssertion,
    *,
    expected: Optional[str],
) -> list[str]:
    source = assertion.source
    target = assertion.target
    if source == AssertionSource.HTTP_STATUS:
        actual = "pm.response.code"
        numeric = True
    elif source == AssertionSource.RESPONSE_HEADER:
        actual = f"pm.response.headers.get({_js(target)})"
        numeric = False
    elif source == AssertionSource.JSON_PATH:
        actual = f"__czJsonValue({_js(target)})"
        numeric = False
    elif source == AssertionSource.XPATH:
        actual = f"__czXmlValue({_js(target)})"
        numeric = False
    elif source == AssertionSource.RESPONSE_BODY:
        actual = "pm.response.text()"
        numeric = False
    elif source == AssertionSource.RESPONSE_TIME:
        actual = "pm.response.responseTime"
        numeric = True
    else:
        raise PostmanBuildError(f"unsupported assertion source: {source}")
    lines = [
        f"pm.test({_js(assertion.description)}, function () {{",
        "  try {",
        f"    const __czActual{index} = {actual};",
    ]
    expected_name = f"__czExpected{index}"
    if expected is not None:
        lines.append(
            f"    const {expected_name} = pm.variables.replaceIn({_js(expected)});"
        )
    lines.append(
        "    "
        + _expectation(
            f"__czActual{index}",
            expected_name,
            assertion.operator,
            numeric=numeric,
        )
    )
    lines.extend(
        [
            "  } catch (error) {",
            "    __czPassed = false;",
            "    throw error;",
            "  }",
            "});",
        ]
    )
    return lines


def _expectation(
    actual: str,
    expected: str,
    operator: AssertionOperator,
    *,
    numeric: bool,
) -> str:
    left = f"Number({actual})" if numeric else f"String({actual})"
    right = f"Number({expected})" if numeric else f"String({expected})"
    if operator == AssertionOperator.EQUALS:
        return f"pm.expect({left}).to.eql({right});"
    if operator == AssertionOperator.NOT_EQUALS:
        return f"pm.expect({left}).to.not.eql({right});"
    if operator == AssertionOperator.CONTAINS:
        return f"pm.expect(String({actual})).to.include(String({expected}));"
    if operator == AssertionOperator.NOT_CONTAINS:
        return f"pm.expect(String({actual})).to.not.include(String({expected}));"
    if operator == AssertionOperator.EXISTS:
        return f"pm.expect({actual}).to.not.eql(undefined);"
    if operator == AssertionOperator.NOT_EXISTS:
        return f"pm.expect({actual}).to.eql(undefined);"
    if operator == AssertionOperator.MATCHES:
        return f"pm.expect(String({actual})).to.match(new RegExp({expected}));"
    if operator == AssertionOperator.LESS_THAN:
        return f"pm.expect(Number({actual})).to.be.below(Number({expected}));"
    if operator == AssertionOperator.GREATER_THAN:
        return f"pm.expect(Number({actual})).to.be.above(Number({expected}));"
    raise PostmanBuildError(f"unsupported assertion operator: {operator}")


def _extraction_script(
    index: int,
    variable_name: str,
    binding: TemplateVariableBinding,
    child_id: str,
) -> list[str]:
    source = binding.extraction_source
    if source == AssertionSource.JSON_PATH:
        actual = f"__czJsonValue({_js(binding.source_key)})"
    elif source == AssertionSource.XPATH:
        actual = f"__czXmlValue({_js(binding.source_key)})"
    elif source == AssertionSource.RESPONSE_HEADER:
        actual = f"pm.response.headers.get({_js(binding.source_key)})"
    elif source == AssertionSource.RESPONSE_BODY:
        actual = "pm.response.text()"
    else:
        raise PostmanBuildError(
            f"unsupported dependency extraction source for {binding.name}"
        )
    return [
        f"pm.test({_js(f'Capture {binding.name} for {child_id}')}, function () {{",
        "  try {",
        f"    const __czExtracted{index} = {actual};",
        f"    pm.expect(__czExtracted{index}).to.not.eql(undefined);",
        f"    pm.collectionVariables.set({_js(variable_name)}, "
        f"String(__czExtracted{index}));",
        "  } catch (error) {",
        "    __czPassed = false;",
        "    throw error;",
        "  }",
        "});",
    ]


def _status_variable(test_case_id: str) -> str:
    return (
        "__cz_status_" + hashlib.sha256(test_case_id.encode("utf-8")).hexdigest()[:16]
    )


def _js(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


__all__ = [
    "LoadedSynthesis",
    "PostmanBuildConfig",
    "PostmanBuildError",
    "PostmanBuildResult",
    "PostmanBuilder",
    "load_synthesis",
]
