"""Deterministic secret redaction for recorded browser evidence."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from scraper.models import ValueCapture


REDACTED = "[REDACTED]"


def hash_text(value: str) -> str:
    """Return a stable SHA-256 digest without retaining the original value."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def is_sensitive_name(name: str, redacted_names: Sequence[str]) -> bool:
    """Return whether a field name matches a configured sensitive marker."""

    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    return any(
        re.sub(r"[^a-z0-9]", "", marker.lower()) in normalized
        for marker in redacted_names
        if marker
    )


def capture_mapping(
    values: Mapping[str, object],
    redacted_names: Sequence[str],
) -> tuple[ValueCapture, ...]:
    """Convert a mapping into deterministic, safely redacted value records."""

    captured: list[ValueCapture] = []
    for name in sorted(values, key=str.lower):
        value = str(values[name])
        if is_sensitive_name(name, redacted_names):
            captured.append(
                ValueCapture(
                    name=name,
                    value_hash=hash_text(value),
                    redacted=True,
                )
            )
        else:
            captured.append(ValueCapture(name=name, value=value))
    return tuple(captured)


def redact_url(url: str, redacted_names: Sequence[str]) -> str:
    """Redact sensitive query and fragment values while preserving URL shape."""

    parts = urlsplit(url)
    query = _redact_pairs(
        parse_qsl(parts.query, keep_blank_values=True), redacted_names
    )
    fragment = parts.fragment
    if fragment:
        fragment_pairs = parse_qsl(fragment, keep_blank_values=True)
        if fragment_pairs:
            fragment = urlencode(
                _redact_pairs(fragment_pairs, redacted_names), doseq=True
            )
        else:
            fragment = redact_text(fragment, redacted_names)
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(query, doseq=True),
            fragment,
        )
    )


def redact_text(text: str, redacted_names: Sequence[str]) -> str:
    """Redact bearer credentials and common name/value secret patterns."""

    redacted = re.sub(
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+",
        f"Bearer {REDACTED}",
        text,
    )
    for marker in sorted(set(redacted_names), key=len, reverse=True):
        if not marker:
            continue
        escaped = re.escape(marker)
        redacted = re.sub(
            rf'(?i)(["\']?{escaped}["\']?\s*[:=]\s*)'
            rf'(["\'])(.*?)(\2)',
            rf"\1\2{REDACTED}\2",
            redacted,
        )
        redacted = re.sub(
            rf"(?i)(\b{escaped}\b\s*[:=]\s*)([^\s,;&\"'<>\)]+)",
            rf"\1{REDACTED}",
            redacted,
        )
        redacted = re.sub(
            rf"(?is)(<{escaped}(?:\s[^>]*)?>)(.*?)(</{escaped}\s*>)",
            rf"\1{REDACTED}\3",
            redacted,
        )
    return redacted


def redact_body(
    data: bytes,
    media_type: str,
    redacted_names: Sequence[str],
) -> tuple[bytes, bool]:
    """Redact structured text bodies and report whether content changed."""

    base_media_type = media_type.split(";", 1)[0].strip().lower()
    if base_media_type.endswith("+json") or base_media_type == "application/json":
        try:
            parsed = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _redact_text_bytes(data, redacted_names)
        value, changed = _redact_json_value(parsed, redacted_names)
        if not changed:
            return data, False
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8"),
            True,
        )
    if base_media_type == "application/x-www-form-urlencoded":
        text = data.decode("utf-8", errors="replace")
        pairs = parse_qsl(text, keep_blank_values=True)
        redacted_pairs = _redact_pairs(pairs, redacted_names)
        encoded = urlencode(redacted_pairs, doseq=True).encode("utf-8")
        return encoded, encoded != data
    if (
        base_media_type.startswith("text/")
        or base_media_type.endswith("+xml")
        or base_media_type in {"application/xml", "application/javascript"}
    ):
        return _redact_text_bytes(data, redacted_names)
    return data, False


def redact_har(data: bytes, redacted_names: Sequence[str]) -> tuple[bytes, bool]:
    """Redact secrets in a Playwright HAR JSON artifact."""

    try:
        value = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _redact_text_bytes(data, redacted_names)
    redacted, changed = _redact_har_value(value, redacted_names)
    if not changed:
        return data, False
    return (
        json.dumps(
            redacted,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8"),
        True,
    )


def _redact_pairs(
    pairs: Sequence[tuple[str, str]],
    redacted_names: Sequence[str],
) -> list[tuple[str, str]]:
    return [
        (name, REDACTED if is_sensitive_name(name, redacted_names) else value)
        for name, value in pairs
    ]


def _redact_text_bytes(
    data: bytes,
    redacted_names: Sequence[str],
) -> tuple[bytes, bool]:
    text = data.decode("utf-8", errors="replace")
    redacted = redact_text(text, redacted_names)
    encoded = redacted.encode("utf-8")
    return encoded, encoded != data


def _redact_json_value(
    value: object,
    redacted_names: Sequence[str],
) -> tuple[object, bool]:
    if isinstance(value, dict):
        result: dict[str, object] = {}
        changed = False
        for key, item in value.items():
            name = str(key)
            if is_sensitive_name(name, redacted_names):
                result[name] = REDACTED
                changed = True
            else:
                result[name], item_changed = _redact_json_value(item, redacted_names)
                changed = changed or item_changed
        return result, changed
    if isinstance(value, list):
        result_list: list[object] = []
        changed = False
        for item in value:
            redacted_item, item_changed = _redact_json_value(item, redacted_names)
            result_list.append(redacted_item)
            changed = changed or item_changed
        return result_list, changed
    return value, False


def _redact_har_value(
    value: object,
    redacted_names: Sequence[str],
    parent_key: Optional[str] = None,
) -> tuple[object, bool]:
    if isinstance(value, dict):
        result: dict[str, object] = {}
        changed = False
        named_value = value.get("name")
        for key, item in value.items():
            if (
                key == "value"
                and isinstance(named_value, str)
                and is_sensitive_name(named_value, redacted_names)
            ):
                result[key] = REDACTED
                changed = True
            elif key == "url" and isinstance(item, str):
                result[key] = redact_url(item, redacted_names)
                changed = changed or result[key] != item
            elif is_sensitive_name(key, redacted_names):
                result[key] = REDACTED
                changed = True
            elif (
                key == "text"
                and isinstance(item, str)
                and parent_key
                in {
                    "postData",
                    "content",
                }
            ):
                mime_type = str(value.get("mimeType", "text/plain"))
                encoding = str(value.get("encoding", "")).lower()
                if encoding == "base64":
                    try:
                        raw_body = base64.b64decode(item, validate=True)
                    except (binascii.Error, ValueError):
                        result[key] = REDACTED
                        changed = True
                        continue
                    body, body_changed = redact_body(
                        raw_body,
                        mime_type,
                        redacted_names,
                    )
                    result[key] = base64.b64encode(body).decode("ascii")
                else:
                    body, body_changed = redact_body(
                        item.encode("utf-8"),
                        mime_type,
                        redacted_names,
                    )
                    result[key] = body.decode("utf-8", errors="replace")
                changed = changed or body_changed
            else:
                result[key], item_changed = _redact_har_value(
                    item,
                    redacted_names,
                    parent_key=key,
                )
                changed = changed or item_changed
        return result, changed
    if isinstance(value, list):
        result_list: list[object] = []
        changed = False
        for item in value:
            redacted_item, item_changed = _redact_har_value(
                item,
                redacted_names,
                parent_key=parent_key,
            )
            result_list.append(redacted_item)
            changed = changed or item_changed
        return result_list, changed
    return value, False


__all__ = [
    "REDACTED",
    "capture_mapping",
    "hash_text",
    "is_sensitive_name",
    "redact_body",
    "redact_har",
    "redact_text",
    "redact_url",
]
