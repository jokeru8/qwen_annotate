"""Bounded async client for structured, multimodal Qwen requests."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from qwen_annotate.video import FrameSample, as_data_url


T = TypeVar("T", bound=BaseModel)
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


class _ResponseStatusError(RuntimeError):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        super().__init__(body)


class _MalformedResponse(ValueError):
    pass


class _StrictJSONError(ValueError):
    pass


class QwenClient(Generic[T]):
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
        if isinstance(retry_seconds, (int, float)) and not isinstance(retry_seconds, bool):
            delays = (float(retry_seconds),)
        else:
            delays = tuple(float(delay) for delay in retry_seconds)
        if not delays or any(delay < 0 for delay in delays):
            raise ValueError("retry_seconds must contain nonnegative delays")
        if endpoint is not None and base_url is not None and str(endpoint) != str(base_url):
            raise ValueError("endpoint and base_url must not disagree")

        self._api_key = api_key
        self._model = model
        self._max_attempts = max_attempts
        self._retry_seconds = delays
        self._guided_json = use_vllm_guided_json
        self._sleep = sleep
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
            self._send = client.chat.completions.create
        else:
            self._send = send

    async def complete(
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
            try:
                raw_response = await self._send(**request)
                content = self._extract_content(raw_response)
            except _MalformedResponse as exc:
                raise InvalidModelResponse(
                    "model returned a malformed response envelope",
                    attempt_count=attempt_count,
                    excerpt=self._safe_excerpt(exc),
                ) from exc
            except Exception as exc:
                full_detail = _exception_detail(exc)
                if self._is_oom(full_detail):
                    raise ModelOutOfMemory(
                        "model worker ran out of memory or exited",
                        attempt_count=attempt_count,
                        excerpt=self._safe_excerpt(full_detail),
                    ) from None
                excerpt = self._safe_excerpt(full_detail)
                if self._is_transient(exc) and attempt_count < self._max_attempts:
                    await self._sleep(self._delay(transient_count))
                    transient_count += 1
                    continue
                raise ModelCallError(
                    "model request failed",
                    attempt_count=attempt_count,
                    excerpt=excerpt,
                ) from None

            try:
                payload = _strict_json_loads(content)
                if repair_used:
                    if original_json is _NOT_PARSED:
                        raise ValueError("original response has no recoverable semantic JSON baseline")
                    if not _json_semantically_equal(payload, original_json):
                        raise ValueError("format repair changed semantic JSON values")
                return response_type.model_validate(payload)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_invalid = self._safe_excerpt(content)
                if not repair_used and attempt_count < self._max_attempts:
                    original_json = _recover_semantic_baseline(content)
                    repair_used = True
                    request = self._repair_request(content, response_type, schema)
                    continue
                category = (
                    "JSON parsing"
                    if isinstance(exc, (json.JSONDecodeError, _StrictJSONError))
                    else "schema validation"
                )
                raise InvalidModelResponse(
                    f"model response failed strict {category}",
                    attempt_count=attempt_count,
                    excerpt=last_invalid,
                ) from exc

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
        response_type: type[T],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": as_data_url(sample)}}
            for sample in frames
        )
        request: dict[str, Any] = {
            "model": self._model,
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
        response_type: type[T],
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

    def _extract_content(self, response: Any) -> str:
        if isinstance(response, str):
            content = response
        else:
            status_code = _status_code(response)
            if status_code is not None and status_code >= 400:
                raise _ResponseStatusError(status_code, _response_text(response))
            try:
                content = response.choices[0].message.content
            except (AttributeError, IndexError, KeyError, TypeError) as exc:
                raise _MalformedResponse("response is missing choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise _MalformedResponse("choices[0].message.content must be a nonempty string")
        return content

    def _is_transient(self, exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        if exc.__class__.__name__ in {"APITimeoutError", "APIConnectionError"}:
            return True
        status = _status_code(exc)
        return status == 429 or (status is not None and 500 <= status <= 599)

    def _is_oom(self, text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in _OOM_MARKERS)

    def _delay(self, transient_index: int) -> float:
        return self._retry_seconds[min(transient_index, len(self._retry_seconds) - 1)]

    def _safe_excerpt(self, value: object) -> str:
        text = str(value).replace(self._api_key, "[REDACTED]") if self._api_key else str(value)
        text = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
        text = re.sub(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+", r"\1[REDACTED]", text)
        text = " ".join(text.split())
        if len(text) > _MAX_ERROR_EXCERPT:
            return text[: _MAX_ERROR_EXCERPT - 3] + "..."
        return text


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
    return json.loads(content, parse_constant=_reject_json_constant)


def _reject_json_constant(constant: str) -> Any:
    raise _StrictJSONError(f"non-standard JSON numeric constant: {constant}")


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
