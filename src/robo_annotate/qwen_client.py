"""Bounded async client for structured, multimodal Qwen requests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Awaitable, Callable, Sequence
from numbers import Real
from typing import Any

from pydantic import BaseModel, ValidationError

from robo_annotate.video import FrameSample, as_data_url


Send = Callable[..., Awaitable[Any]]
Sleep = Callable[[float], Awaitable[None]]

_MAX_ERROR_EXCERPT = 256
_OOM_MARKERS = (
    "out of memory",
    "cuda oom",
    "worker died",
    "worker exited",
    "engine dead",
)


class ModelCallError(RuntimeError):
    """A model request failed without yielding a valid typed result."""

    def __init__(self, message: str, *, attempt_count: int, excerpt: str = "") -> None:
        self.attempt_count = attempt_count
        self.excerpt = excerpt
        detail = f"{message} after {attempt_count} attempt(s)"
        if excerpt:
            detail += f": {excerpt}"
        super().__init__(detail)


class ModelOutOfMemory(ModelCallError):
    """The model worker exhausted memory or died, allowing caller degradation."""


class InvalidModelResponse(ModelCallError):
    """The endpoint returned no strictly valid instance of the requested schema."""


class _StrictJSONError(ValueError):
    pass


class QwenClient:
    """Call an OpenAI-compatible Qwen endpoint with a finite total call budget.

    ``max_attempts`` limits every outbound call made by one :meth:`complete`,
    including transient retries and the optional single format-repair call.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        endpoint: str | None = None,
        api_key: str = "local",
        model: str = "Qwen/Qwen3.8-27B",
        timeout: float = 120.0,
        max_attempts: int = 4,
        retry_seconds: float | Sequence[float] = (1.0, 2.0, 4.0),
        use_vllm_guided_json: bool = False,
        send: Send | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        delays = _normalize_retry_seconds(retry_seconds)
        if endpoint is not None and base_url is not None and str(endpoint) != str(base_url):
            raise ValueError("endpoint and base_url must not disagree")

        self._model = model
        self._max_attempts = max_attempts
        self._retry_seconds = delays
        self._guided_json = use_vllm_guided_json
        self._sleep = sleep
        self._owned_client: Any | None = None
        if send is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError("openai is required when no send callable is injected") from exc
            client = AsyncOpenAI(
                base_url=str(endpoint or base_url or "http://127.0.0.1:8000/v1"),
                api_key=api_key,
                timeout=timeout,
            )
            self._owned_client = client
            self._send = client.chat.completions.create
        else:
            self._send = send

    async def aclose(self) -> None:
        """Close the internally-created AsyncOpenAI client exactly once."""
        client, self._owned_client = self._owned_client, None
        if client is not None:
            await client.close()

    async def complete[T: BaseModel](
        self,
        prompt: str,
        frames: list[FrameSample],
        response_type: type[T],
    ) -> T:
        """Return a strictly validated response, or raise a typed bounded error."""
        schema = response_type.model_json_schema()
        request = self._request(prompt, frames, response_type, schema)
        attempt_count = 0
        transient_count = 0
        repair_used = False
        original_json: Any = _NOT_PARSED
        last_invalid = ""

        while attempt_count < self._max_attempts:
            attempt_count += 1
            content, failure, excerpt = await self._send_once(request)
            if failure is not None:
                if failure == "transient" and attempt_count < self._max_attempts:
                    await self._sleep(self._delay(transient_count))
                    transient_count += 1
                    continue
                if failure == "oom":
                    raise ModelOutOfMemory(
                        "model worker ran out of memory or exited",
                        attempt_count=attempt_count,
                        excerpt=excerpt,
                    )
                if failure == "malformed":
                    raise InvalidModelResponse(
                        "model returned a malformed response envelope",
                        attempt_count=attempt_count,
                        excerpt=excerpt,
                    )
                raise ModelCallError(
                    "model request failed",
                    attempt_count=attempt_count,
                    excerpt=excerpt,
                )
            if content is None:
                raise InvalidModelResponse(
                    "model returned a malformed response envelope",
                    attempt_count=attempt_count,
                    excerpt="<missing sanitized response state>",
                )

            result, category = _parse_typed_response(
                content,
                response_type,
                repair_used=repair_used,
                semantic_baseline=original_json,
            )
            if result is not None:
                return result
            last_invalid = _redacted_response_excerpt(content)
            if not repair_used and attempt_count < self._max_attempts:
                original_json = _recover_semantic_baseline(content)
                repair_used = True
                request = self._repair_request(content, response_type, schema)
                continue
            raise InvalidModelResponse(
                f"model response failed strict {category}",
                attempt_count=attempt_count,
                excerpt=last_invalid,
            ) from None

        # The loop exits only if future changes add a zero-cost branch.
        raise InvalidModelResponse(
            "model response was invalid",
            attempt_count=attempt_count,
            excerpt=last_invalid,
        )

    def _request(
        self,
        prompt: str,
        frames: list[FrameSample],
        response_type: type[BaseModel],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": as_data_url(sample)}}
            for sample in frames
        )
        request: dict[str, Any] = {
            "model": self._model,
            "temperature": 0,
            "messages": [{"role": "user", "content": content}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_type.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if self._guided_json:
            request["extra_body"] = {"guided_json": schema}
        return request

    def _repair_request(
        self,
        invalid_response: str,
        response_type: type[BaseModel],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        repair_prompt = (
            "Repair only the JSON representation/format of the response below. "
            "Do not add, remove, infer, or change any semantic value. Return JSON only.\n"
            f"Response type: {response_type.__name__}\n"
            f"JSON schema: {json.dumps(schema, ensure_ascii=True, sort_keys=True)}\n"
            "BEGIN_UNTRUSTED_INVALID_RESPONSE_JSON_STRING\n"
            f"{json.dumps(invalid_response, ensure_ascii=True)}\n"
            "END_UNTRUSTED_INVALID_RESPONSE_JSON_STRING"
        )
        return self._request(repair_prompt, [], response_type, schema)

    async def _send_once(
        self, request: dict[str, Any]
    ) -> tuple[str | None, str | None, str]:
        """Return only content or sanitized primitive failure state."""
        try:
            response = await self._send(**request)
        except Exception as exc:
            try:
                detail = _exception_detail(exc)
            except Exception:
                detail = "unprintable transport exception"
            try:
                status_code = _status_code(exc)
            except Exception:
                status_code = None
            if self._is_oom(detail):
                failure = "oom"
            elif self._exception_is_transient(exc, status_code):
                failure = "transient"
            else:
                failure = "fatal"
            return None, failure, _redacted_diagnostic_excerpt(detail, "transport")
        return self._extract_content_result(response)

    def _extract_content_result(
        self, response: Any
    ) -> tuple[str | None, str | None, str]:
        try:
            if isinstance(response, str):
                content = response
            else:
                status_code = _status_code(response)
                if status_code is not None and status_code >= 400:
                    detail = _response_text(response)
                    if self._is_oom(detail):
                        failure = "oom"
                    elif status_code == 429 or 500 <= status_code <= 599:
                        failure = "transient"
                    else:
                        failure = "fatal"
                    return (
                        None,
                        failure,
                        _redacted_diagnostic_excerpt(detail, f"HTTP {status_code}"),
                    )
                content = response.choices[0].message.content
        except Exception as exc:
            try:
                detail = _exception_detail(exc)
            except Exception:
                detail = "unprintable response accessor exception"
            return None, "malformed", _redacted_diagnostic_excerpt(detail, "envelope")
        if not isinstance(content, str) or not content.strip():
            return (
                None,
                "malformed",
                _redacted_diagnostic_excerpt(
                    "missing or non-string choices[0].message.content", "envelope"
                ),
            )
        return content, None, ""

    def _exception_is_transient(
        self, exc: Exception, status_code: int | None
    ) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        if exc.__class__.__name__ in {"APITimeoutError", "APIConnectionError"}:
            return True
        return status_code == 429 or (
            status_code is not None and 500 <= status_code <= 599
        )

    def _is_oom(self, text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in _OOM_MARKERS)

    def _delay(self, transient_index: int) -> float:
        return self._retry_seconds[min(transient_index, len(self._retry_seconds) - 1)]

_NOT_PARSED = object()


def _status_code(value: object) -> int | None:
    direct = getattr(value, "status_code", None)
    if isinstance(direct, int):
        return direct
    response = getattr(value, "response", None)
    nested = getattr(response, "status_code", None)
    return nested if isinstance(nested, int) else None


def _response_text(response: object) -> str:
    text = getattr(response, "text", "")
    return text if isinstance(text, str) else str(text)


def _exception_detail(exc: Exception) -> str:
    detail = str(exc)
    response = getattr(exc, "response", None)
    response_text = _response_text(response) if response is not None else ""
    if response_text and response_text not in detail:
        return f"{detail}: {response_text}"
    return detail


def _recover_semantic_baseline(content: str) -> Any:
    """Conservatively recover one JSON object without changing its values."""
    try:
        return _strict_json_loads(content)
    except (json.JSONDecodeError, _StrictJSONError):
        pass

    candidates: list[Any] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(content):
        if start is None:
            if character == "{":
                start = index
                depth = 1
                in_string = False
                escaped = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                try:
                    candidate = _strict_json_loads(content[start : index + 1])
                except (json.JSONDecodeError, _StrictJSONError):
                    pass
                else:
                    if isinstance(candidate, dict):
                        candidates.append(candidate)
                start = None

    return candidates[0] if len(candidates) == 1 else _NOT_PARSED


def _strict_json_loads(content: str) -> Any:
    return json.loads(
        content,
        parse_constant=_reject_json_constant,
        parse_float=_parse_finite_json_float,
        object_pairs_hook=_reject_duplicate_object_pairs,
    )


def _reject_json_constant(constant: str) -> Any:
    raise _StrictJSONError(f"non-standard JSON numeric constant: {constant}")


def _parse_finite_json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise _StrictJSONError(f"JSON float is outside the finite range: {token}")
    return value


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJSONError("duplicate JSON object key")
        result[key] = value
    return result


def _parse_typed_response[T: BaseModel](
    content: str,
    response_type: type[T],
    *,
    repair_used: bool,
    semantic_baseline: Any,
) -> tuple[T | None, str]:
    """Parse without allowing raw parser/validation exceptions to escape."""
    try:
        payload = _strict_json_loads(content)
        if repair_used:
            if semantic_baseline is _NOT_PARSED:
                raise ValueError("original response has no recoverable semantic JSON baseline")
            if not _json_semantically_equal(payload, semantic_baseline):
                raise ValueError("format repair changed semantic JSON values")
        return response_type.model_validate(payload), ""
    except (json.JSONDecodeError, _StrictJSONError):
        return None, "JSON parsing"
    except (ValidationError, ValueError):
        return None, "schema validation"


def _redacted_response_excerpt(content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"<model response redacted; chars={len(content)}; sha256={digest}>"


def _redacted_diagnostic_excerpt(detail: str, category: str) -> str:
    digest = hashlib.sha256(detail.encode("utf-8", errors="replace")).hexdigest()[:16]
    excerpt = f"<{category} diagnostic redacted; chars={len(detail)}; sha256={digest}>"
    return excerpt[:_MAX_ERROR_EXCERPT]


def _normalize_retry_seconds(value: float | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, Real) and not isinstance(value, bool):
        raw_delays: tuple[object, ...] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw_delays = tuple(value)
    else:
        raise ValueError("retry_seconds must be a real number or a sequence of real numbers")
    if not raw_delays:
        raise ValueError("retry_seconds must contain at least one delay")
    delays: list[float] = []
    for delay in raw_delays:
        if isinstance(delay, bool) or not isinstance(delay, Real):
            raise ValueError("retry_seconds must contain only real numbers")
        try:
            converted = float(delay)
        except (OverflowError, ValueError) as exc:
            raise ValueError("retry_seconds values must fit in a finite float") from exc
        if not math.isfinite(converted) or converted < 0:
            raise ValueError("retry_seconds must contain finite nonnegative delays")
        delays.append(converted)
    return tuple(delays)


def _json_semantically_equal(left: Any, right: Any) -> bool:
    """Compare JSON values while preserving every scalar's exact JSON type."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_semantically_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_semantically_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return bool(left == right)
