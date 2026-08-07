"""Minimal async client for read-only MCP search-tool grounding."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from grounding.models import (
    GroundingSnippet,
    GroundingSourceKind,
    MCPCallCitation,
    MCPDocumentReference,
)
from scraper.models import CapturePolicy
from scraper.redaction import redact_text


DEFAULT_READ_ONLY_TOOLS = ("search_docs", "search_documents")


class MCPError(RuntimeError):
    """Base failure raised by the MCP grounding client."""


class MCPTransportError(MCPError):
    """Raised for bounded HTTP, decoding, or response-size failures."""


class MCPRetryableTransportError(MCPTransportError):
    """Raised for transient connection/server failures that may be retried."""


class MCPProtocolError(MCPError):
    """Raised when an endpoint violates the expected JSON-RPC contract."""


@dataclass(frozen=True)
class MCPClientConfig:
    """Transport limits and strict tool allowlist."""

    endpoint: str
    timeout_seconds: float = 20.0
    maximum_response_bytes: int = 4_000_000
    maximum_attempts: int = 3
    retry_backoff_seconds: float = 0.5
    allowed_tools: tuple[str, ...] = DEFAULT_READ_ONLY_TOOLS
    redacted_names: tuple[str, ...] = CapturePolicy().redacted_names

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("MCP endpoint must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "MCP endpoint cannot contain credentials, query, or fragment"
            )
        if (
            self.timeout_seconds <= 0
            or self.maximum_response_bytes <= 0
            or self.maximum_attempts <= 0
            or self.retry_backoff_seconds < 0
        ):
            raise ValueError("MCP transport limits must be positive")
        if not self.allowed_tools or len(self.allowed_tools) != len(
            set(self.allowed_tools)
        ):
            raise ValueError("MCP allowed_tools must be non-empty and unique")


@dataclass(frozen=True)
class MCPServerIdentity:
    protocol_version: str
    name: str
    version: str


@dataclass(frozen=True)
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, object]


@dataclass(frozen=True)
class MCPToolResult:
    tool_name: str
    query: str
    arguments_sha256: str
    response_sha256: str
    retrieved_at: datetime
    text_blocks: tuple[str, ...]


class MCPTransport(Protocol):
    """Injectable JSON transport used by production and fixture tests."""

    async def post_json(
        self,
        endpoint: str,
        payload: dict[str, object],
        *,
        timeout_seconds: float,
        maximum_response_bytes: int,
    ) -> dict[str, object]: ...


class UrllibMCPTransport:
    """Small stdlib transport; blocking I/O is isolated in a worker thread."""

    async def post_json(
        self,
        endpoint: str,
        payload: dict[str, object],
        *,
        timeout_seconds: float,
        maximum_response_bytes: int,
    ) -> dict[str, object]:
        return await asyncio.to_thread(
            self._post_json,
            endpoint,
            payload,
            timeout_seconds,
            maximum_response_bytes,
        )

    @staticmethod
    def _post_json(
        endpoint: str,
        payload: dict[str, object],
        timeout_seconds: float,
        maximum_response_bytes: int,
    ) -> dict[str, object]:
        data = _canonical_json(payload)
        request = Request(
            endpoint,
            data=data,
            method="POST",
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
        )
        try:
            # Do not silently send integration context through process-global
            # proxy configuration. The configured endpoint is contacted directly.
            opener = build_opener(ProxyHandler({}))
            with opener.open(request, timeout=timeout_seconds) as response:
                final_url = response.geturl()
                if _origin(final_url) != _origin(endpoint):
                    raise MCPTransportError("MCP endpoint redirected across origins")
                body = response.read(maximum_response_bytes + 1)
                if len(body) > maximum_response_bytes:
                    raise MCPTransportError(
                        "MCP response exceeded configured byte limit"
                    )
                content_type = response.headers.get("Content-Type", "")
        except HTTPError as exc:
            body = exc.read(1_024).decode("utf-8", errors="replace")
            message = (
                f"MCP HTTP {exc.code}: "
                f"{redact_text(body, CapturePolicy().redacted_names)}"
            )
            error_type = (
                MCPRetryableTransportError if exc.code >= 500 else MCPTransportError
            )
            raise error_type(message) from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise MCPRetryableTransportError(f"MCP request failed: {exc}") from exc

        text = body.decode("utf-8", errors="replace")
        try:
            if "text/event-stream" in content_type.casefold():
                return _json_from_event_stream(text)
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MCPTransportError("MCP response was not valid JSON") from exc
        if not isinstance(value, dict):
            raise MCPTransportError("MCP response root must be an object")
        return value


class MCPClient:
    """JSON-RPC MCP client that can invoke only configured read-only tools."""

    def __init__(
        self,
        config: MCPClientConfig,
        transport: Optional[MCPTransport] = None,
    ) -> None:
        self.config = config
        self.transport = transport or UrllibMCPTransport()
        self.identity: Optional[MCPServerIdentity] = None
        self.tools: tuple[MCPTool, ...] = ()
        self._request_id = 0
        self._id_lock = threading.Lock()

    async def connect(self) -> tuple[MCPServerIdentity, tuple[MCPTool, ...]]:
        """Initialize the endpoint and discover its callable tools."""

        initialize = await self._request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {
                    "name": "cz-certification-automation",
                    "version": "0.8.0",
                },
            },
        )
        protocol = initialize.get("protocolVersion")
        server_info = initialize.get("serverInfo")
        if not isinstance(protocol, str) or not isinstance(server_info, dict):
            raise MCPProtocolError("MCP initialize result lacks server identity")
        name = server_info.get("name")
        version = server_info.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise MCPProtocolError("MCP serverInfo name/version must be strings")
        identity = MCPServerIdentity(protocol, name, version)

        listed = await self._request("tools/list", {})
        raw_tools = listed.get("tools")
        if not isinstance(raw_tools, list):
            raise MCPProtocolError("MCP tools/list result lacks a tools array")
        tools: list[MCPTool] = []
        for raw_tool in raw_tools:
            if not isinstance(raw_tool, dict):
                raise MCPProtocolError("MCP tool definition must be an object")
            tool_name = raw_tool.get("name")
            schema = raw_tool.get("inputSchema", {})
            if not isinstance(tool_name, str) or not isinstance(schema, dict):
                raise MCPProtocolError("MCP tool definition is invalid")
            description = raw_tool.get("description", "")
            tools.append(
                MCPTool(
                    name=tool_name,
                    description=description if isinstance(description, str) else "",
                    input_schema=schema,
                )
            )
        self.identity = identity
        self.tools = tuple(sorted(tools, key=lambda item: item.name))
        return identity, self.tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> MCPToolResult:
        """Invoke one discovered allowlisted tool and retain response digests."""

        if self.identity is None:
            raise MCPProtocolError("MCP client must connect before tool calls")
        if name not in self.config.allowed_tools:
            raise MCPProtocolError(f"MCP tool is not allowlisted: {name}")
        discovered = {tool.name for tool in self.tools}
        if name not in discovered:
            raise MCPProtocolError(f"MCP tool was not advertised: {name}")
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise MCPProtocolError("automatic MCP search arguments require a query")

        result = await self._request(
            "tools/call", {"name": name, "arguments": arguments}
        )
        if result.get("isError") is True:
            raise MCPProtocolError(f"MCP tool reported an error: {name}")
        content = result.get("content")
        if not isinstance(content, list):
            raise MCPProtocolError("MCP tool result lacks a content array")
        text_blocks = tuple(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            and block["text"].strip()
        )
        return MCPToolResult(
            tool_name=name,
            query=query,
            arguments_sha256=hashlib.sha256(_canonical_json(arguments)).hexdigest(),
            response_sha256=hashlib.sha256(_canonical_json(result)).hexdigest(),
            retrieved_at=datetime.now(timezone.utc),
            text_blocks=text_blocks,
        )

    def to_snippets(
        self,
        result: MCPToolResult,
        *,
        maximum_content_characters: int = 10_000,
    ) -> tuple[GroundingSnippet, ...]:
        """Split ranked search output into bounded cited MCP snippets."""

        if self.identity is None:
            raise MCPProtocolError("MCP client has no server identity")
        snippets: list[GroundingSnippet] = []
        rank = 0
        for block in result.text_blocks:
            for segment in _result_segments(block):
                safe_content = redact_text(segment.strip(), self.config.redacted_names)
                if not safe_content:
                    continue
                rank += 1
                safe_content = safe_content[:maximum_content_characters]
                documents = _document_references(safe_content)
                citation = MCPCallCitation(
                    endpoint=self.config.endpoint,
                    server_name=self.identity.name,
                    server_version=self.identity.version,
                    protocol_version=self.identity.protocol_version,
                    tool_name=result.tool_name,
                    query=result.query,
                    arguments_sha256=result.arguments_sha256,
                    response_sha256=result.response_sha256,
                    retrieved_at=result.retrieved_at,
                    documents=documents,
                )
                title = _snippet_title(result.tool_name, documents, rank)
                snippet_id = _hash_json(
                    {
                        "source": GroundingSourceKind.MCP.value,
                        "tool": result.tool_name,
                        "arguments_sha256": result.arguments_sha256,
                        "response_sha256": result.response_sha256,
                        "documents": [
                            item.model_dump(mode="json") for item in documents
                        ],
                        "content": safe_content,
                    }
                )
                snippets.append(
                    GroundingSnippet(
                        snippet_id=snippet_id,
                        source_kind=GroundingSourceKind.MCP,
                        title=title,
                        content=safe_content,
                        relevance_score=max(1, 1_000 - rank),
                        mcp=citation,
                    )
                )
        return tuple(snippets)

    async def _request(
        self,
        method: str,
        params: dict[str, object],
    ) -> dict[str, object]:
        with self._id_lock:
            self._request_id += 1
            request_id = self._request_id
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        response: dict[str, object] | None = None
        for attempt in range(1, self.config.maximum_attempts + 1):
            try:
                response = await self.transport.post_json(
                    self.config.endpoint,
                    payload,
                    timeout_seconds=self.config.timeout_seconds,
                    maximum_response_bytes=self.config.maximum_response_bytes,
                )
                break
            except MCPRetryableTransportError:
                if attempt == self.config.maximum_attempts:
                    raise
                await asyncio.sleep(self.config.retry_backoff_seconds * attempt)
        if response is None:  # Defensive; the loop either returns or raises.
            raise MCPTransportError("MCP request produced no response")
        if response.get("jsonrpc") != "2.0":
            raise MCPProtocolError("MCP response has an invalid JSON-RPC version")
        if response.get("id") != request_id:
            raise MCPProtocolError("MCP response ID does not match request")
        error = response.get("error")
        if error is not None:
            safe_error = redact_text(str(error), self.config.redacted_names)
            raise MCPProtocolError(f"MCP JSON-RPC error: {safe_error}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise MCPProtocolError("MCP response lacks an object result")
        return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_from_event_stream(text: str) -> dict[str, object]:
    for line in reversed(text.splitlines()):
        if not line.startswith("data:"):
            continue
        try:
            value = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("jsonrpc") == "2.0":
            return value
    raise MCPTransportError("MCP event stream contained no JSON-RPC response")


def _result_segments(text: str) -> tuple[str, ...]:
    matches = list(re.finditer(r"(?m)^###\s+Result\s+\d+\s*$", text))
    if not matches:
        return (text,)
    return tuple(
        text[match.start() : matches[index + 1].start()]
        if index + 1 < len(matches)
        else text[match.start() :]
        for index, match in enumerate(matches)
    )


def _document_references(content: str) -> tuple[MCPDocumentReference, ...]:
    source = _markdown_value(content, "Source")
    document_id = _markdown_value(content, "Document ID")
    chunk = _markdown_value(content, "Chunk")
    if not any((source, document_id, chunk)):
        return ()
    return (
        MCPDocumentReference(
            source=source,
            document_id=document_id,
            chunk_id=chunk,
        ),
    )


def _markdown_value(content: str, label: str) -> Optional[str]:
    match = re.search(
        rf"(?mi)^\*\*{re.escape(label)}:\*\*\s*`?([^`\n]+)`?\s*$",
        content,
    )
    return match.group(1).strip() if match else None


def _snippet_title(
    tool_name: str,
    documents: tuple[MCPDocumentReference, ...],
    rank: int,
) -> str:
    if documents:
        document = documents[0]
        identity = document.document_id or document.source or f"result-{rank}"
        suffix = f" chunk {document.chunk_id}" if document.chunk_id else ""
        return f"{identity}{suffix}"
    return f"{tool_name} result {rank}"


def _origin(url: str) -> tuple[str, str, Optional[int]]:
    parsed = urlsplit(url)
    return parsed.scheme.casefold(), (parsed.hostname or "").casefold(), parsed.port


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


__all__ = [
    "DEFAULT_READ_ONLY_TOOLS",
    "MCPClient",
    "MCPClientConfig",
    "MCPError",
    "MCPProtocolError",
    "MCPRetryableTransportError",
    "MCPServerIdentity",
    "MCPTool",
    "MCPToolResult",
    "MCPTransport",
    "MCPTransportError",
    "UrllibMCPTransport",
]
