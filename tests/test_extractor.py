"""Pure normalization tests for deterministic state extraction."""

from pathlib import Path

from scraper.artifact_store import ArtifactStore
from scraper.extractor import PageStateExtractor, SnapshotConfig
from scraper.models import LocatorStrategy, ScrapeRun


def make_extractor(tmp_path: Path) -> PageStateExtractor:
    run = ScrapeRun(
        run_id="extractor-run",
        root_url="https://portal.test",
        allowed_origins=("https://portal.test",),
    )
    return PageStateExtractor(ArtifactStore.create(tmp_path, run))


def test_snapshot_config_rejects_unbounded_or_invalid_values() -> None:
    for kwargs in (
        {"maximum_elements_per_frame": 0},
        {"maximum_text_chars": 0},
        {"quiet_window_ms": -1},
        {"quiet_timeout_ms": 0},
        {"quiet_poll_interval_ms": 0},
    ):
        try:
            SnapshotConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"SnapshotConfig accepted invalid values: {kwargs}")


def test_element_normalization_is_stable_unique_and_secret_safe(
    tmp_path: Path,
) -> None:
    extractor = make_extractor(tmp_path)
    raw = [
        {
            "domPath": "html > body > input",
            "shadowHostPath": [],
            "tag": "input",
            "role": "textbox",
            "accessibleName": "Customer number",
            "label": "Customer number",
            "text": "",
            "title": "",
            "inputType": "text",
            "attributes": {
                "data-testid": "customer-number",
                "href": "/next?token=url-secret",
                "name": "customerNumber",
                "value": "attribute-secret",
            },
            "value": "current-secret",
            "visible": True,
            "enabled": True,
            "editable": True,
            "checked": None,
            "selected": None,
            "expanded": None,
            "readOnly": False,
            "boundingBox": {"x": 10, "y": 20, "width": 200, "height": 30},
            "options": [],
            "context": {"ancestors": ["form#customer"]},
            "interactive": True,
            "interactionSignals": ["native-control", "role:textbox"],
        },
        {
            "domPath": "html > body > input:nth-of-type(2)",
            "shadowHostPath": [],
            "tag": "input",
            "role": "textbox",
            "accessibleName": "Customer number",
            "label": "Customer number",
            "text": "",
            "title": "",
            "inputType": "text",
            "attributes": {"data-testid": "customer-number"},
            "value": "second-secret",
            "visible": False,
            "enabled": False,
            "editable": False,
            "boundingBox": {"x": 0, "y": 0, "width": 0, "height": 0},
            "options": [],
            "context": {"ancestors": []},
            "interactive": True,
        },
    ]

    first = extractor._normalize_elements(  # noqa: SLF001 - pure boundary test
        raw,
        frame_id="frame-main",
        frame_key="main",
    )
    second = extractor._normalize_elements(  # noqa: SLF001 - pure boundary test
        raw,
        frame_id="frame-main",
        frame_key="main",
    )

    assert first == second
    assert first[0].element_id != first[1].element_id
    assert first[0].value_redacted is True
    assert first[0].interactive is True
    assert first[0].interaction_signals == ("native-control", "role:textbox")
    assert first[0].value_hash is not None
    assert "current-secret" not in first[0].model_dump_json()
    assert "attribute-secret" not in first[0].model_dump_json()
    assert "url-secret" not in first[0].model_dump_json()

    primary = next(locator for locator in first[0].locators if locator.is_primary)
    assert primary.strategy == LocatorStrategy.STABLE_ATTRIBUTE
    test_id = next(
        locator
        for locator in first[0].locators
        if locator.strategy == LocatorStrategy.TEST_ID
    )
    assert test_id.unique_match_count == 2


def test_sensitive_select_options_are_hashed(tmp_path: Path) -> None:
    extractor = make_extractor(tmp_path)
    raw = [
        {
            "domPath": "#credential",
            "shadowHostPath": [],
            "tag": "select",
            "role": "combobox",
            "attributes": {"id": "credential", "name": "accessToken"},
            "value": "secret-option",
            "visible": True,
            "enabled": True,
            "editable": True,
            "boundingBox": {"x": 0, "y": 0, "width": 100, "height": 20},
            "options": [
                {
                    "label": "Credential A",
                    "value": "secret-option",
                    "selected": True,
                    "disabled": False,
                }
            ],
            "context": {},
            "interactive": True,
        }
    ]

    element = extractor._normalize_elements(  # noqa: SLF001 - pure boundary test
        raw,
        frame_id="frame-main",
        frame_key="main",
    )[0]

    assert element.options[0].redacted is True
    assert element.options[0].value is None
    assert element.options[0].value_hash is not None
    assert "secret-option" not in element.model_dump_json()


def test_safe_href_encoding_does_not_create_invalid_redacted_capture(
    tmp_path: Path,
) -> None:
    extractor = make_extractor(tmp_path)

    attributes = extractor._attributes(  # noqa: SLF001 - pure boundary test
        {
            "href": "/ResetPassword?message=hello%20world&next=%2FHome",
        },
        current_value=None,
    )

    href = next(attribute for attribute in attributes if attribute.name == "href")
    assert href.redacted is False
    assert href.value == "/ResetPassword?message=hello%20world&next=%2FHome"
    assert href.safe_value is None
