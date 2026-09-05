"""Async, bounded OpenAI-compatible client for a LiteLLM proxy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional, Protocol, Type
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from pydantic import BaseModel, SecretStr

from synthesis.models import TokenUsage


LOGGER = logging.getLogger("cz.synthesis.transport")


class LiteLLMError(RuntimeError):
    """Base failure raised by the constrained LiteLLM client."""


class LiteLLMTransportError(LiteLLMError):
    """Raised for bounded HTTP and response-decoding failures."""


class LiteLLMRetryableError(LiteLLMTransportError):
    """Raised for a transient failure that may safely be retried."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LiteLLMProtocolError(LiteLLMError):
    """Raised when a successful response violates the chat-completion contract."""


@dataclass(frozen=True)
class LiteLLMConfig:
    """Non-secret model settings and secret API-key boundary."""

    endpoint: str
    model: str
    api_key: Optional[SecretStr] = None
    timeout_seconds: float = 90.0
    maximum_response_bytes: int = 8_000_000
    maximum_transport_attempts: int = 3
    retry_backoff_seconds: float = 15.0
    maximum_output_tokens: int = 16_000
    response_format: str = "json_schema"
    progress_heartbeat_seconds: float = 15.0

    def __post_init__(self) -> None:
        normalized = normalize_litellm_endpoint(self.endpoint)
        object.__setattr__(self, "endpoint", normalized)
        if not self.model.strip():
            raise ValueError("LiteLLM model must be non-empty")
        if self.response_format not in {"json_schema", "json_object"}:
            raise ValueError("response_format must be json_schema or json_object")
        if (
            self.timeout_seconds <= 0
            or self.maximum_response_bytes <= 0
            or self.maximum_transport_attempts <= 0
            or self.retry_backoff_seconds < 0
            or self.maximum_output_tokens <= 0
            or self.progress_heartbeat_seconds <= 0
        ):
            raise ValueError("LiteLLM transport and output limits must be positive")


@dataclass(frozen=True)
class LiteLLMTransportResponse:
    payload: dict[str, object]
    response_sha256: str


@dataclass(frozen=True)
class LiteLLMCompletion:
    content: str
    request_sha256: str
    response_sha256: str
    usage: TokenUsage


class LiteLLMTransport(Protocol):
    async def post_json(
        self,
        endpoint: str,
        payload: bytes,
        *,
        api_key: Optional[SecretStr],
        timeout_seconds: float,
        maximum_response_bytes: int,
    ) -> LiteLLMTransportResponse: ...


class SynthesisLLM(Protocol):
    """Injectable structured-generation boundary used by the builder."""

    @property
    def config(self) -> LiteLLMConfig: ...

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        schema_name: str,
        maximum_output_tokens: Optional[int] = None,
    ) -> LiteLLMCompletion: ...


class UrllibLiteLLMTransport:
    """Stdlib HTTP transport with blocking work isolated in a worker thread."""

    async def post_json(
        self,
        endpoint: str,
        payload: bytes,
        *,
        api_key: Optional[SecretStr],
        timeout_seconds: float,
        maximum_response_bytes: int,
    ) -> LiteLLMTransportResponse:
        return await asyncio.to_thread(
            self._post_json,
            endpoint,
            payload,
            api_key,
            timeout_seconds,
            maximum_response_bytes,
        )

    @staticmethod
    def _post_json(
        endpoint: str,
        payload: bytes,
        api_key: Optional[SecretStr],
        timeout_seconds: float,
        maximum_response_bytes: int,
    ) -> LiteLLMTransportResponse:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key.get_secret_value()}"
        request = Request(endpoint, data=payload, method="POST", headers=headers)
        try:
            # Integration prompts must not silently traverse process-global proxies.
            opener = build_opener(ProxyHandler({}), _NoRedirectHandler())
            with opener.open(request, timeout=timeout_seconds) as response:
                if _origin(response.geturl()) != _origin(endpoint):
                    raise LiteLLMTransportError(
                        "LiteLLM endpoint redirected across origins"
                    )
                body = response.read(maximum_response_bytes + 1)
                if len(body) > maximum_response_bytes:
                    raise LiteLLMTransportError(
                        "LiteLLM response exceeded the configured byte limit"
                    )
        except HTTPError as exc:
            message = f"LiteLLM HTTP {exc.code}"
            error_type = (
                LiteLLMRetryableError
                if exc.code == 408 or exc.code == 429 or exc.code >= 500
                else LiteLLMTransportError
            )
            if error_type is LiteLLMRetryableError:
                raise error_type(
                    message,
                    retry_after_seconds=_retry_after_seconds(
                        exc.headers.get("Retry-After") if exc.headers else None
                    ),
                ) from exc
            raise error_type(message) from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise LiteLLMRetryableError(f"LiteLLM request failed: {exc}") from exc

        try:
            value = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LiteLLMTransportError("LiteLLM response was not valid JSON") from exc
        if not isinstance(value, dict):
            raise LiteLLMTransportError("LiteLLM response root must be an object")
        return LiteLLMTransportResponse(
            payload=value,
            response_sha256=hashlib.sha256(body).hexdigest(),
        )


class LiteLLMClient:
    """Strict chat-completion client supporting JSON Schema or JSON object mode."""

    def __init__(
        self,
        config: LiteLLMConfig,
        transport: Optional[LiteLLMTransport] = None,
    ) -> None:
        self._config = config
        self.transport = transport or UrllibLiteLLMTransport()

    @property
    def config(self) -> LiteLLMConfig:
        return self._config

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        schema_name: str,
        maximum_output_tokens: Optional[int] = None,
    ) -> LiteLLMCompletion:
        effective_output_tokens = (
            self.config.maximum_output_tokens
            if maximum_output_tokens is None
            else maximum_output_tokens
        )
        if effective_output_tokens <= 0:
            raise ValueError("per-request maximum output tokens must be positive")
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": effective_output_tokens,
        }
        if self.config.response_format == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": _strict_json_schema(response_model.model_json_schema()),
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}
        request_data = _canonical_json(payload)
        request_sha256 = hashlib.sha256(request_data).hexdigest()

        response: Optional[LiteLLMTransportResponse] = None
        content = ""
        for attempt in range(1, self.config.maximum_transport_attempts + 1):
            started_at = time.monotonic()
            LOGGER.info(
                "model request started request=%s attempt=%d/%d model=%s "
                "timeout_seconds=%.1f max_output_tokens=%d",
                request_sha256[:12],
                attempt,
                self.config.maximum_transport_attempts,
                self.config.model,
                self.config.timeout_seconds,
                effective_output_tokens,
            )
            try:
                response = await self._post_with_heartbeat(
                    request_data,
                    request_sha256=request_sha256,
                    attempt=attempt,
                    started_at=started_at,
                )
                content = _completion_content(response.payload)
                break
            except LiteLLMRetryableError as exc:
                elapsed = time.monotonic() - started_at
                if attempt >= self.config.maximum_transport_attempts:
                    LOGGER.error(
                        "model request failed request=%s attempt=%d/%d "
                        "elapsed_seconds=%.1f retryable=true error=%s",
                        request_sha256[:12],
                        attempt,
                        self.config.maximum_transport_attempts,
                        elapsed,
                        exc,
                    )
                    raise
                delay = self.config.retry_backoff_seconds * (2 ** (attempt - 1))
                if exc.retry_after_seconds is not None:
                    delay = max(delay, exc.retry_after_seconds)
                LOGGER.warning(
                    "model request retrying request=%s attempt=%d/%d "
                    "elapsed_seconds=%.1f next_attempt_in_seconds=%.1f error=%s",
                    request_sha256[:12],
                    attempt,
                    self.config.maximum_transport_attempts,
                    elapsed,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
            except LiteLLMError as exc:
                LOGGER.error(
                    "model request failed request=%s attempt=%d/%d "
                    "elapsed_seconds=%.1f retryable=false error=%s",
                    request_sha256[:12],
                    attempt,
                    self.config.maximum_transport_attempts,
                    time.monotonic() - started_at,
                    exc,
                )
                raise
        if response is None:  # Defensive; retry loop either returns or raises.
            raise LiteLLMTransportError("LiteLLM transport produced no response")

        usage = _usage(response.payload)
        LOGGER.info(
            "model response received request=%s response=%s content_characters=%d "
            "prompt_tokens=%d completion_tokens=%d total_tokens=%d",
            request_sha256[:12],
            response.response_sha256[:12],
            len(content),
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )
        return LiteLLMCompletion(
            content=content,
            request_sha256=request_sha256,
            response_sha256=response.response_sha256,
            usage=usage,
        )

    async def _post_with_heartbeat(
        self,
        request_data: bytes,
        *,
        request_sha256: str,
        attempt: int,
        started_at: float,
    ) -> LiteLLMTransportResponse:
        task = asyncio.create_task(
            self.transport.post_json(
                self.config.endpoint,
                request_data,
                api_key=self.config.api_key,
                timeout_seconds=self.config.timeout_seconds,
                maximum_response_bytes=self.config.maximum_response_bytes,
            )
        )
        try:
            while not task.done():
                done, _ = await asyncio.wait(
                    {task},
                    timeout=self.config.progress_heartbeat_seconds,
                )
                if not done:
                    LOGGER.info(
                        "model request waiting request=%s attempt=%d/%d "
                        "elapsed_seconds=%.1f",
                        request_sha256[:12],
                        attempt,
                        self.config.maximum_transport_attempts,
                        time.monotonic() - started_at,
                    )
            return await task
        except BaseException:
            if not task.done():
                task.cancel()
            raise


def normalize_litellm_endpoint(value: str) -> str:
    """Normalize a LiteLLM base URL or a chat-completions endpoint."""

    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("LiteLLM URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("LiteLLM URL cannot contain credentials, query, or fragment")
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        normalized_path = path
    elif path.endswith("/v1"):
        normalized_path = f"{path}/chat/completions"
    elif not path:
        normalized_path = "/v1/chat/completions"
    else:
        normalized_path = f"{path}/v1/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, normalized_path, "", ""))


class _NoRedirectHandler(HTTPRedirectHandler):
    """Prevent credentials and prompts from following any HTTP redirect."""

    def redirect_request(
        self,
        request: Request,
        file_pointer: object,
        code: int,
        message: str,
        headers: object,
        new_url: str,
    ) -> None:
        del request, file_pointer, code, message, headers, new_url
        return None


def _completion_content(payload: dict[str, object]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LiteLLMProtocolError("LiteLLM response lacks choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise LiteLLMProtocolError("LiteLLM response lacks an assistant message")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if content is None or isinstance(content, str):
        raise LiteLLMRetryableError("LiteLLM assistant content is empty")
    if isinstance(content, list):
        blocks = [
            block.get("text")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        combined = "\n".join(blocks).strip()
        if combined:
            return combined
    raise LiteLLMProtocolError("LiteLLM assistant content type is unsupported")


def _usage(payload: dict[str, object]) -> TokenUsage:
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        return TokenUsage()
    prompt = _nonnegative_int(raw.get("prompt_tokens"))
    completion = _nonnegative_int(raw.get("completion_tokens"))
    reported_total = _nonnegative_int(raw.get("total_tokens"))
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=max(reported_total, prompt + completion),
    )


def _nonnegative_int(value: object) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _origin(value: str) -> tuple[str, str, Optional[int]]:
    parsed = urlsplit(value)
    return parsed.scheme.casefold(), (parsed.hostname or "").casefold(), parsed.port


def _retry_after_seconds(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    stripped = value.strip()
    try:
        return max(0.0, float(stripped))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(stripped)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _strict_json_schema(value: object) -> object:
    """Make every object property explicit for strict structured-output APIs."""

    if isinstance(value, list):
        return [_strict_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    unsupported_constraints = {
        "default",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "maxItems",
        "maxLength",
        "maxProperties",
        "maximum",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "multipleOf",
        "pattern",
        "patternProperties",
        "propertyNames",
        "unevaluatedProperties",
        "uniqueItems",
    }
    result = {
        key: _strict_json_schema(item)
        for key, item in value.items()
        if key not in unsupported_constraints
    }
    properties = result.get("properties")
    if isinstance(properties, dict):
        result["additionalProperties"] = False
        result["required"] = list(properties)
    return result


__all__ = [
    "LiteLLMClient",
    "LiteLLMCompletion",
    "LiteLLMConfig",
    "LiteLLMError",
    "LiteLLMProtocolError",
    "LiteLLMRetryableError",
    "LiteLLMTransport",
    "LiteLLMTransportError",
    "LiteLLMTransportResponse",
    "SynthesisLLM",
    "UrllibLiteLLMTransport",
    "normalize_litellm_endpoint",
]
