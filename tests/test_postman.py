from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from postman.builder import (
    PostmanBuildConfig,
    PostmanBuildError,
    PostmanBuilder,
    load_synthesis,
)
from postman.cli import main as postman_main
from postman.exporter import export_postman
from synthesis.exporter import export_synthesis
from synthesis.models import (
    AssertionOperator,
    AssertionSource,
    Confidence,
    GeneratedValueKind,
    HTTPMethod,
    HTTPRequestSpec,
    LLMProviderSummary,
    NamedValue,
    RequestBodyMode,
    RequestBodySpec,
    ResponseAssertion,
    SynthesisCoverage,
    SynthesisDisposition,
    SynthesisPackage,
    SignatureAlgorithm,
    SignatureBinding,
    TemplateVariableBinding,
    TemplateVariableSource,
    TestCaseExecutionSpec as ExecutionSpec,
)


NOW = datetime(2026, 8, 7, tzinfo=timezone.utc)


def _specifications() -> tuple[ExecutionSpec, ExecutionSpec]:
    parent = ExecutionSpec(
        test_case_id="TC_01",
        title="Create a bill",
        disposition=SynthesisDisposition.READY,
        request=HTTPRequestSpec(
            method=HTTPMethod.POST,
            path="/bills",
            headers=(NamedValue(name="x-api-key", value="{{VENDOR_KEY}}"),),
            body=RequestBodySpec(
                mode=RequestBodyMode.JSON,
                content_type="application/json",
                template='{"requestId":"{{REQUEST_ID}}"}',
            ),
        ),
        assertions=(
            ResponseAssertion(
                source=AssertionSource.HTTP_STATUS,
                operator=AssertionOperator.EQUALS,
                expected="200",
                description="Bill request succeeds",
            ),
        ),
        variable_bindings=(
            TemplateVariableBinding(
                name="VENDOR_KEY",
                source=TemplateVariableSource.ENVIRONMENT,
                sensitive=True,
                description="Vendor API credential",
            ),
            TemplateVariableBinding(
                name="REQUEST_ID",
                source=TemplateVariableSource.GENERATED,
                generator=GeneratedValueKind.UUID,
                description="Unique request ID",
            ),
        ),
        evidence_snippet_ids=("a" * 64,),
        confidence=Confidence.HIGH,
        rationale="Cited schema supplies the create request.",
        human_review_required=False,
    )
    child = ExecutionSpec(
        test_case_id="TC_02",
        title="Fetch the created bill",
        dependency_case_ids=("TC_01",),
        disposition=SynthesisDisposition.READY,
        request=HTTPRequestSpec(
            method=HTTPMethod.GET,
            path="/bills/{{BILL_ID}}",
            headers=(NamedValue(name="x-api-key", value="{{VENDOR_KEY}}"),),
        ),
        assertions=(
            ResponseAssertion(
                source=AssertionSource.JSON_PATH,
                operator=AssertionOperator.EQUALS,
                target="$.status",
                expected="SUCCESS",
                description="Fetched bill is successful",
            ),
        ),
        variable_bindings=(
            TemplateVariableBinding(
                name="VENDOR_KEY",
                source=TemplateVariableSource.ENVIRONMENT,
                sensitive=True,
                description="Vendor API credential",
            ),
            TemplateVariableBinding(
                name="BILL_ID",
                source=TemplateVariableSource.DEPENDENCY,
                source_key="$.billId",
                dependency_case_id="TC_01",
                extraction_source=AssertionSource.JSON_PATH,
                description="Bill ID returned by TC_01",
            ),
        ),
        evidence_snippet_ids=("b" * 64,),
        confidence=Confidence.HIGH,
        rationale="Cited schema supplies the fetch request.",
        human_review_required=False,
    )
    return parent, child


def _package(
    specifications: tuple[ExecutionSpec, ...] | None = None,
    *,
    synthesis_complete: bool = True,
    execution_ready: bool = True,
) -> SynthesisPackage:
    specs = specifications or _specifications()
    ready = sum(spec.disposition == SynthesisDisposition.READY for spec in specs)
    needs_review = sum(
        spec.disposition == SynthesisDisposition.NEEDS_REVIEW for spec in specs
    )
    blocked = sum(spec.disposition == SynthesisDisposition.BLOCKED for spec in specs)
    return SynthesisPackage(
        source_run_id="fixture-run",
        source_grounding_sha256="c" * 64,
        synthesized_at=NOW,
        provider=LLMProviderSummary(
            endpoint="https://llm.example.test/v1/chat/completions",
            model="fixture-model",
            response_format="json_schema",
            prompt_sha256="d" * 64,
            responses_received=0,
            valid_responses=0,
            prompt_tokens=0,
            completion_tokens=0,
        ),
        calls=(),
        specifications=specs,
        coverage=SynthesisCoverage(
            source_grounding_complete=True,
            test_cases=len(specs),
            synthesized=len(specs),
            ready=ready,
            needs_review=needs_review,
            blocked=blocked,
            synthesis_complete=synthesis_complete,
            execution_ready=execution_ready,
        ),
    )


def test_renderer_builds_dependency_ordered_secret_safe_collection(
    tmp_path: Path,
) -> None:
    package = _package()
    first = PostmanBuilder(package, "e" * 64).build()
    second = PostmanBuilder(package, "e" * 64).build()

    assert first.collection == second.collection
    assert [item.name.split(" - ")[0] for item in first.collection.item] == [
        "TC_01",
        "TC_02",
    ]
    assert first.report.coverage.generation_complete is True
    environment = {item.key: item for item in first.environment.values}
    assert environment["base_url"].value == ""
    assert environment["VENDOR_KEY"].type == "secret"
    assert environment["VENDOR_KEY"].value == ""

    parent_tests = "\n".join(first.collection.item[0].event[1].script.exec)
    parent_prerequest = "\n".join(first.collection.item[0].event[0].script.exec)
    child_prerequest = "\n".join(first.collection.item[1].event[0].script.exec)
    assert "Capture BILL_ID for TC_02" in parent_tests
    assert "__czRuntimeVariables.forEach" in parent_prerequest
    assert "pm.execution.skipRequest()" in child_prerequest
    assert "__cz_dep_" in first.collection.item[1].request.url.raw
    assert "{{BILL_ID}}" not in first.collection.item[1].request.url.raw

    exported = export_postman(first, tmp_path / "postman")
    collection_json = json.loads(exported.collection_path.read_text())
    environment_json = json.loads(exported.environment_path.read_text())
    assert collection_json["info"]["schema"].endswith(
        "/collection/json/v2.1.0/draft-07/collection.json"
    )
    assert environment_json["_postman_variable_scope"] == "environment"
    assert "top-secret" not in exported.environment_path.read_text()
    for item in exported.manifest.files:
        data = (exported.output_directory / item.name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item.sha256


def test_signature_recipe_renders_deterministic_hmac_prerequest() -> None:
    spec = ExecutionSpec(
        test_case_id="TC_SIG",
        title="Signed call",
        disposition=SynthesisDisposition.READY,
        request=HTTPRequestSpec(
            method=HTTPMethod.POST,
            path="/api/{{API_VERSION}}/merchants/vpas/validity",
            headers=(
                NamedValue(name="Content-Type", value="application/json"),
                NamedValue(name="x-merchant-id", value="{{MERCHANT_ID}}"),
                NamedValue(name="x-merchant-channel-id", value="{{CHANNEL_ID}}"),
                NamedValue(
                    name="x-merchant-signature", value="{{X_MERCHANT_SIGNATURE}}"
                ),
            ),
            body=RequestBodySpec(
                mode=RequestBodyMode.JSON,
                content_type="application/json",
                template='{"customerVpa":"{{CUSTOMER_VPA}}"}',
            ),
            signature=SignatureBinding(
                algorithm=SignatureAlgorithm.MERCHANT_HMAC_SHA256,
                header_name="x-merchant-signature",
                key_binding_name="API_KEY",
                component_binding_names=("MERCHANT_ID", "CHANNEL_ID"),
            ),
        ),
        variable_bindings=(
            TemplateVariableBinding(
                name="API_KEY",
                source=TemplateVariableSource.ENVIRONMENT,
                sensitive=True,
                description="Signing key",
            ),
            TemplateVariableBinding(
                name="X_MERCHANT_SIGNATURE",
                source=TemplateVariableSource.ENVIRONMENT,
                sensitive=True,
                description="Computed at request time by the signature recipe",
            ),
            TemplateVariableBinding(
                name="MERCHANT_ID",
                source=TemplateVariableSource.ENVIRONMENT,
                description="Merchant id",
            ),
            TemplateVariableBinding(
                name="CHANNEL_ID",
                source=TemplateVariableSource.ENVIRONMENT,
                description="Channel id",
            ),
            TemplateVariableBinding(
                name="API_VERSION",
                source=TemplateVariableSource.EVIDENCE_LITERAL,
                value="1",
                description="Evidence-cited api version",
            ),
            TemplateVariableBinding(
                name="CUSTOMER_VPA",
                source=TemplateVariableSource.ENVIRONMENT,
                description="Cert VPA",
            ),
        ),
        assertions=(
            ResponseAssertion(
                source=AssertionSource.HTTP_STATUS,
                operator=AssertionOperator.EQUALS,
                expected="200",
                description="Signed request succeeds",
            ),
        ),
        evidence_snippet_ids=("b" * 64,),
        confidence=Confidence.HIGH,
        rationale="S2S spec documents the HMAC recipe.",
        human_review_required=False,
    )
    package = PostmanBuilder(_package((spec,)), "e" * 64).build()
    prerequest = "\n".join(package.collection.item[0].event[0].script.exec)
    assert "CryptoJS.HmacSHA256(" in prerequest
    assert '(pm.variables.get("MERCHANT_ID") || "")' in prerequest
    assert '(pm.variables.get("CHANNEL_ID") || "")' in prerequest
    assert 'pm.variables.replaceIn(pm.request.body.raw || "")' in prerequest
    assert "CryptoJS.enc.Hex" in prerequest
    assert 'pm.request.headers.upsert({ key: "x-merchant-signature"' in prerequest
    # The signature env placeholder stays empty and secret-free
    keys = {value.key for value in package.environment.values}
    assert "API_KEY" in keys


def test_signature_recipe_rejects_undeclared_or_nonsensitive_key() -> None:
    base = ExecutionSpec(
        test_case_id="TC_SIG2",
        title="Signed call",
        disposition=SynthesisDisposition.READY,
        request=HTTPRequestSpec(
            method=HTTPMethod.POST,
            path="/signed",
            headers=(NamedValue(name="x-sign", value="{{X_SIGN}}"),),
            body=RequestBodySpec(
                mode=RequestBodyMode.JSON,
                content_type="application/json",
                template="{}",
            ),
            signature=SignatureBinding(
                algorithm=SignatureAlgorithm.MERCHANT_HMAC_SHA256,
                header_name="x-sign",
                key_binding_name="API_KEY",
                component_binding_names=("MERCHANT_ID",),
            ),
        ),
        variable_bindings=(
            TemplateVariableBinding(
                name="MERCHANT_ID",
                source=TemplateVariableSource.ENVIRONMENT,
                description="Merchant id",
            ),
            TemplateVariableBinding(
                name="X_SIGN",
                source=TemplateVariableSource.ENVIRONMENT,
                sensitive=True,
                description="Computed at request time by the signature recipe",
            ),
            # Deliberately NOT marked sensitive: the signature recipe must
            # reject a key binding that is not a sensitive environment value.
            TemplateVariableBinding(
                name="API_KEY",
                source=TemplateVariableSource.ENVIRONMENT,
                description="Signing key (wrongly non-secret)",
            ),
        ),
        evidence_snippet_ids=("b" * 64,),
        confidence=Confidence.HIGH,
        assertions=(
            ResponseAssertion(
                source=AssertionSource.HTTP_STATUS,
                operator=AssertionOperator.EQUALS,
                expected="200",
                description="ok",
            ),
        ),
        rationale="x",
        human_review_required=False,
    )
    with pytest.raises(PostmanBuildError, match="sensitive environment"):
        PostmanBuilder(_package((base,)), "e" * 64).build()


def test_include_needs_review_renders_marked_drafts_and_skips_requestless() -> None:
    parent, child = _specifications()
    parent_data = parent.model_dump(mode="json")
    parent_data.update(
        {
            "disposition": SynthesisDisposition.NEEDS_REVIEW,
            "human_review_required": True,
            "unresolved_requirements": ["confirm the endpoint"],
        }
    )
    needs_review = ExecutionSpec.model_validate(parent_data)
    package = _package(
        (needs_review, child),
        synthesis_complete=True,
        execution_ready=False,
    )

    draft = PostmanBuilder(
        package,
        "e" * 64,
        config=PostmanBuildConfig(include_needs_review=True),
    ).build()

    assert draft.report.coverage.rendered == 2
    assert draft.report.coverage.rendered_needs_review == 1
    assert draft.report.coverage.generation_complete is False
    assert any("DRAFT" in line for line in draft.report.coverage.limitations)
    parent_item = next(
        item
        for item in draft.collection.item
        if item.id == draft.report.rendered_test_cases[0].item_id
    )
    assert parent_item.name.startswith("[REVIEW REQUIRED] ")
    assert "DRAFT — REVIEW REQUIRED" in (parent_item.request.description or "")
    assert "confirm the endpoint" in (parent_item.request.description or "")
    child_item = next(
        item
        for item in draft.collection.item
        if not item.name.startswith("[REVIEW REQUIRED]")
    )
    assert "[REVIEW]" not in child_item.name

    # A needs_review spec without a request is skipped honestly, never rendered.
    no_request_data = needs_review.model_dump(mode="json")
    no_request_data["request"] = None
    requestless = ExecutionSpec.model_validate(no_request_data)
    blocked_package = _package(
        (requestless, child),
        synthesis_complete=True,
        execution_ready=False,
    )
    empty = PostmanBuilder(
        blocked_package,
        "e" * 64,
        config=PostmanBuildConfig(include_needs_review=True),
    ).build()
    assert empty.report.coverage.rendered == 0
    assert empty.report.coverage.skipped == 2
    skip_reasons = {
        item.test_case_id: item.reason for item in empty.report.skipped_test_cases
    }
    assert "no request to render" in skip_reasons[parent.test_case_id]
    # A READY child whose draft parent could not render is evicted, never orphaned.
    assert "dependency is unavailable" in skip_reasons[child.test_case_id]


def test_partial_render_skips_nonready_cases_and_their_descendants() -> None:
    parent, child = _specifications()
    parent_data = parent.model_dump(mode="json")
    parent_data.update(
        {
            "disposition": SynthesisDisposition.NEEDS_REVIEW,
            "human_review_required": True,
            "unresolved_requirements": ["confirm the endpoint"],
        }
    )
    needs_review = ExecutionSpec.model_validate(parent_data)
    package = _package(
        (needs_review, child),
        synthesis_complete=True,
        execution_ready=False,
    )

    with pytest.raises(PostmanBuildError, match="execution_ready"):
        PostmanBuilder(package, "e" * 64).build()
    partial = PostmanBuilder(
        package,
        "e" * 64,
        config=PostmanBuildConfig(allow_partial=True),
    ).build()

    assert partial.report.coverage.rendered == 0
    assert partial.report.coverage.skipped == 2
    reasons = {
        item.test_case_id: item.reason for item in partial.report.skipped_test_cases
    }
    assert "needs_review" in reasons["TC_01"]
    assert "dependency is unavailable" in reasons["TC_02"]


def test_dependency_cycle_is_never_silently_ordered() -> None:
    first, second = _specifications()
    first_data = first.model_dump(mode="json")
    first_data["dependency_case_ids"] = ["TC_02"]
    cyclic_first = ExecutionSpec.model_validate(first_data)
    package = _package((cyclic_first, second))

    with pytest.raises(PostmanBuildError, match="dependency graph"):
        PostmanBuilder(package, "e" * 64).build()
    partial = PostmanBuilder(
        package,
        "e" * 64,
        config=PostmanBuildConfig(allow_partial=True),
    ).build()
    assert partial.report.coverage.rendered == 0
    assert all("cycle" in item.reason for item in partial.report.skipped_test_cases)


def test_renderer_maps_form_query_xml_header_body_and_time_assertions() -> None:
    spec = ExecutionSpec(
        test_case_id="TC_FORM",
        title="Submit encoded data",
        disposition=SynthesisDisposition.READY,
        request=HTTPRequestSpec(
            method=HTTPMethod.POST,
            path="/submit",
            query_parameters=(NamedValue(name="mode", value="certification"),),
            body=RequestBodySpec(
                mode=RequestBodyMode.FORM_URLENCODED,
                content_type="application/x-www-form-urlencoded",
                template="biller={{BILLER_ID}}",
            ),
        ),
        assertions=(
            ResponseAssertion(
                source=AssertionSource.RESPONSE_HEADER,
                operator=AssertionOperator.EXISTS,
                target="x-request-id",
                description="Request ID header exists",
            ),
            ResponseAssertion(
                source=AssertionSource.RESPONSE_BODY,
                operator=AssertionOperator.CONTAINS,
                target="body",
                expected="accepted",
                description="Body acknowledges acceptance",
            ),
            ResponseAssertion(
                source=AssertionSource.RESPONSE_TIME,
                operator=AssertionOperator.LESS_THAN,
                target="milliseconds",
                expected="5000",
                description="Response is timely",
            ),
            ResponseAssertion(
                source=AssertionSource.XPATH,
                operator=AssertionOperator.EQUALS,
                target="/Response/code",
                expected="000",
                description="XML response code succeeds",
            ),
        ),
        variable_bindings=(
            TemplateVariableBinding(
                name="BILLER_ID",
                source=TemplateVariableSource.PORTAL_FIELD,
                value="BILLER-1",
                source_key="biller_id",
                description="Portal-declared biller ID",
            ),
        ),
        evidence_snippet_ids=("f" * 64,),
        confidence=Confidence.HIGH,
        rationale="Cited form contract and response rules.",
        human_review_required=False,
    )
    result = PostmanBuilder(_package((spec,)), "e" * 64).build()
    item = result.collection.item[0]
    assert item.request.body is not None
    assert item.request.body.mode == "urlencoded"
    assert item.request.body.urlencoded[0].value == "{{BILLER_ID}}"
    assert item.request.url.raw.endswith("?mode=certification")
    script = "\n".join(item.event[1].script.exec)
    assert "pm.response.headers.get" in script
    assert "pm.response.responseTime" in script
    assert "xml2Json" in script
    assert "pm.response.text()" in script


def test_synthesis_loader_and_postman_plan_verify_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    synthesis = export_synthesis(_package(), tmp_path / "synthesis")

    loaded = load_synthesis(synthesis.output_directory)
    assert loaded.synthesis.source_run_id == "fixture-run"
    assert (
        postman_main(
            [
                "--synthesis",
                str(synthesis.output_directory),
                "--plan-only",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["rendered"] == 2
    assert plan["generation_complete"] is True

    synthesis.synthesis_path.write_bytes(synthesis.synthesis_path.read_bytes() + b"\n")
    with pytest.raises(PostmanBuildError, match="does not match"):
        load_synthesis(synthesis.output_directory)
