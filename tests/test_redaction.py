"""Secret redaction tests for browser and network evidence."""

import base64
import json

from scraper.redaction import (
    REDACTED,
    capture_mapping,
    redact_body,
    redact_har,
    redact_text,
    redact_url,
)


REDACTED_NAMES = (
    "authorization",
    "cookie",
    "password",
    "token",
    "secret",
)


def test_mapping_redacts_sensitive_headers_but_preserves_hashes() -> None:
    values = capture_mapping(
        {
            "Content-Type": "application/json",
            "Authorization": "Bearer top-secret",
        },
        REDACTED_NAMES,
    )
    by_name = {value.name.lower(): value for value in values}

    assert by_name["content-type"].value == "application/json"
    assert by_name["authorization"].redacted is True
    assert by_name["authorization"].value is None
    assert by_name["authorization"].value_hash is not None


def test_url_redacts_sensitive_query_and_fragment_values() -> None:
    result = redact_url(
        "https://portal.test/api?token=secret&visible=yes#password=hidden",
        REDACTED_NAMES,
    )

    assert "secret" not in result
    assert "hidden" not in result
    assert "visible=yes" in result
    assert "%5BREDACTED%5D" in result


def test_text_redacts_bearer_assignments_and_xml_elements() -> None:
    text = (
        'Authorization: Bearer abc.def password="hunter2" '
        "<token>xml-secret</token> visible=ok"
    )

    result = redact_text(text, REDACTED_NAMES)

    assert "abc.def" not in result
    assert "hunter2" not in result
    assert "xml-secret" not in result
    assert "visible=ok" in result


def test_structured_body_redaction_is_canonical() -> None:
    body, changed = redact_body(
        b'{"visible":1,"nested":{"apiToken":"secret"}}',
        "application/json; charset=utf-8",
        REDACTED_NAMES,
    )

    assert changed is True
    assert body == b'{"nested":{"apiToken":"[REDACTED]"},"visible":1}'

    form, form_changed = redact_body(
        b"username=alice&password=secret",
        "application/x-www-form-urlencoded",
        REDACTED_NAMES,
    )
    assert form_changed is True
    assert form == b"username=alice&password=%5BREDACTED%5D"


def test_har_redaction_understands_name_value_entries_and_bodies() -> None:
    har = {
        "log": {
            "entries": [
                {
                    "request": {
                        "url": "https://portal.test/api?token=url-secret",
                        "headers": [
                            {"name": "Authorization", "value": "Bearer header-secret"}
                        ],
                        "postData": {
                            "mimeType": "application/json",
                            "text": '{"password":"body-secret","visible":true}',
                        },
                    }
                }
            ]
        }
    }

    data, changed = redact_har(json.dumps(har).encode(), REDACTED_NAMES)
    decoded = data.decode()

    assert changed is True
    assert "url-secret" not in decoded
    assert "header-secret" not in decoded
    assert "body-secret" not in decoded
    assert REDACTED in decoded


def test_har_redaction_decodes_and_reencodes_base64_bodies() -> None:
    encoded_body = base64.b64encode(b'{"token":"base64-secret","visible":true}').decode(
        "ascii"
    )
    har = {
        "log": {
            "entries": [
                {
                    "response": {
                        "content": {
                            "mimeType": "application/json",
                            "encoding": "base64",
                            "text": encoded_body,
                        }
                    }
                }
            ]
        }
    }

    data, changed = redact_har(json.dumps(har).encode(), REDACTED_NAMES)
    content = json.loads(data)["log"]["entries"][0]["response"]["content"]
    decoded_body = base64.b64decode(content["text"])

    assert changed is True
    assert decoded_body == b'{"token":"[REDACTED]","visible":true}'
