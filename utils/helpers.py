"""Utility helper functions."""

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Optional


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename.

    Args:
        name: String to sanitize.

    Returns:
        Sanitized filename string.
    """
    sanitized = re.sub(r'[^\w\s-]', '', name)
    sanitized = re.sub(r'[-\s]+', '-', sanitized)
    return sanitized.strip('-')


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format datetime as ISO timestamp string.

    Args:
        dt: Datetime object (defaults to now).

    Returns:
        Formatted timestamp string.
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d_%H-%M-%S")


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to maximum length.

    Args:
        text: String to truncate.
        max_length: Maximum length.
        suffix: Suffix to append if truncated.

    Returns:
        Truncated string.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def parse_xml_payload(xml_string: str) -> Optional[dict]:
    """Parse XML payload to extract key fields.

    Args:
        xml_string: XML string.

    Returns:
        Dictionary of extracted fields or None.
    """
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_string)
        result = {
            "root_tag": root.tag,
            "namespaces": dict(root.attrib),
        }

        # Extract correlation IDs
        for elem in root.iter():
            if "correlation" in elem.tag.lower() or "transaction" in elem.tag.lower():
                if elem.text:
                    result["correlation_id"] = elem.text.strip()
                    break

        return result
    except Exception:
        return None


async def wait_with_timeout(
    coroutine,
    timeout: float,
    default=None,
) -> any:
    """Wait for coroutine with timeout.

    Args:
        coroutine: Async coroutine to await.
        timeout: Maximum wait time in seconds.
        default: Default value to return on timeout.

    Returns:
        Coroutine result or default value.
    """
    try:
        return await asyncio.wait_for(coroutine, timeout=timeout)
    except asyncio.TimeoutError:
        return default


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, creating if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_json_loads(text: str) -> Optional[dict]:
    """Safely parse JSON string.

    Args:
        text: JSON string.

    Returns:
        Parsed dict or None on failure.
    """
    try:
        import json
        return json.loads(text)
    except Exception:
        return None
