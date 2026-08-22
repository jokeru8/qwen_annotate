import json
import sys
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel, ConfigDict
from qwen_annotate.models import FinalAnnotation
from qwen_annotate.qwen_client import (
    InvalidModelResponse,
    ModelCallError,
    ModelOutOfMemory,
    QwenClient,
)
from qwen_annotate.video import FrameSample, as_data_url


class SemanticPayload(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    value: bool | int | float
    label: str = "same"


def valid_json(boundary: int = 20) -> str:
    return json.dumps({"start_subtask_index": 1, "boundaries": [boundary]})


def frame(index: int, payload: bytes) -> FrameSample:
    return FrameSample(
        camera_key="cam",
        frame_index=index,
        timestamp_seconds=float(index),
        jpeg=payload,
    )


@pytest.mark.asyncio
async def test_exact_multimodal_request_and_schema_are_fresh() -> None:
    calls = []

    async def send(**kwargs):
        calls.append(kwargs)
        return valid_json()

    samples = [frame(1, b"first"), frame(2, b"second")]
    client = QwenClient(send=send, model="local-qwen", max_attempts=1)
    assert (await client.complete("the prompt", samples, FinalAnnotation)).boundaries == [20]

    request = calls[0]
    assert request["model"] == "local-qwen"
    assert request["messages"] == [{
        "role": "user",
        "content": [
            {"type": "text", "text": "the prompt"},
            {"type": "image_url", "image_url": {"url": as_data_url(samples[0])}},
            {"type": "image_url", "image_url": {"url": as_data_url(samples[1])}},
        ],
    }]
    response_format = request["response_format"]
    assert response_format == {
        "type": "json_schema",
        "json_schema": {
            "name": "FinalAnnotation",
            "strict": True,
            "schema": FinalAnnotation.model_json_schema(),
        },
    }
    assert "extra_body" not in request

    response_format["json_schema"]["schema"]["properties"].clear()
    await client.complete("again", [], FinalAnnotation)
    assert calls[1]["response_format"]["json_schema"]["schema"]["properties"]


@pytest.mark.asyncio
async def test_vllm_guided_json_is_only_sent_when_enabled() -> None:
    calls = []

    async def send(**kwargs):
        calls.append(kwargs)
        return valid_json()

    client = QwenClient(send=send, use_vllm_guided_json=True, max_attempts=1)
    await client.complete("prompt", [], FinalAnnotation)
    assert calls[0]["extra_body"] == {"guided_json": FinalAnnotation.model_json_schema()}


@pytest.mark.asyncio
async def test_production_adapter_uses_async_openai_and_extracts_content(monkeypatch) -> None:
    created = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            created["request"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=valid_json(31)))]
            )

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            created["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))
    client = QwenClient(
        base_url="http://127.0.0.1:8000/v1",
        api_key="secret",
        model="qwen",
        timeout=12.5,
        max_attempts=1,
    )
    result = await client.complete("prompt", [], FinalAnnotation)
    assert result.boundaries == [31]
    assert created["client"] == {
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "secret",
        "timeout": 12.5,
    }
    assert created["request"]["messages"][0]["content"][0]["text"] == "prompt"


@pytest.mark.asyncio
async def test_client_retries_transient_error_then_parses_and_sleeps() -> None:
    calls = 0
    delays = []

    async def send(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("slow")
        return valid_json()

    async def sleep(delay):
        delays.append(delay)

    client = QwenClient(send=send, max_attempts=2, retry_seconds=[0.25], sleep=sleep)
    result = await client.complete("prompt", [], FinalAnnotation)
    assert result.boundaries == [20]
    assert calls == 2
    assert delays == [0.25]


@pytest.mark.parametrize("status", [429, 500, 503])
@pytest.mark.asyncio
async def test_retries_transient_http_statuses(status: int) -> None:
    calls = 0
    request = httpx.Request("POST", "http://model/v1/chat/completions")

    async def send(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            response = httpx.Response(status, request=request, text="temporary")
            raise httpx.HTTPStatusError("temporary", request=request, response=response)
        return valid_json()

    client = QwenClient(send=send, max_attempts=2, retry_seconds=0)
    await client.complete("prompt", [], FinalAnnotation)
    assert calls == 2


@pytest.mark.asyncio
async def test_does_not_retry_ordinary_4xx() -> None:
    calls = 0
    request = httpx.Request("POST", "http://model/v1/chat/completions")

    async def send(**kwargs):
        nonlocal calls
        calls += 1
        response = httpx.Response(400, request=request, text="bad request")
        raise httpx.HTTPStatusError("bad request", request=request, response=response)

    with pytest.raises(ModelCallError) as caught:
        await QwenClient(send=send, max_attempts=3, retry_seconds=0).complete(
            "prompt", [], FinalAnnotation
        )
    assert caught.value.attempt_count == 1
    assert calls == 1


@pytest.mark.asyncio
async def test_exhausted_transient_retries_are_bounded() -> None:
    calls = 0
    delays = []

    async def send(**kwargs):
        nonlocal calls
        calls += 1
        raise TimeoutError("still slow")

    async def sleep(delay):
        delays.append(delay)

    with pytest.raises(ModelCallError) as caught:
        await QwenClient(
            send=send,
            max_attempts=3,
            retry_seconds=[1, 2, 4],
            sleep=sleep,
        ).complete(
            "prompt", [], FinalAnnotation
        )
    assert caught.value.attempt_count == 3
    assert calls == 3
    assert delays == [1, 2]


@pytest.mark.parametrize("error_name", ["APITimeoutError", "APIConnectionError"])
@pytest.mark.asyncio
async def test_retries_openai_compatible_transient_exception_names(error_name: str) -> None:
    calls = 0
    transient_type = type(error_name, (RuntimeError,), {})

    async def send(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise transient_type("temporary")
        return valid_json()

    await QwenClient(send=send, max_attempts=2, retry_seconds=0).complete(
        "prompt", [], FinalAnnotation
    )
    assert calls == 2


@pytest.mark.asyncio
async def test_oom_in_http_response_body_is_detected_before_retry() -> None:
    calls = 0
    request = httpx.Request("POST", "http://model/v1/chat/completions")

    async def send(**kwargs):
        nonlocal calls
        calls += 1
        response = httpx.Response(500, request=request, text="CUDA out of memory")
        raise httpx.HTTPStatusError("server error", request=request, response=response)

    with pytest.raises(ModelOutOfMemory):
        await QwenClient(send=send, max_attempts=3, retry_seconds=0).complete(
            "prompt", [], FinalAnnotation
        )
    assert calls == 1


@pytest.mark.asyncio
async def test_oom_marker_after_excerpt_limit_is_still_detected_without_retry() -> None:
    calls = 0

    async def send(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError(("diagnostic " * 40) + "CUDA out of memory")

    with pytest.raises(ModelOutOfMemory) as caught:
        await QwenClient(send=send, max_attempts=3, retry_seconds=0).complete(
            "prompt", [], FinalAnnotation
        )
    assert calls == 1
    assert len(caught.value.excerpt) <= 256


@pytest.mark.asyncio
async def test_non_oom_cuda_error_is_not_misclassified() -> None:
    async def send(**kwargs):
        raise RuntimeError("CUDA error: invalid device ordinal")

    with pytest.raises(ModelCallError) as caught:
        await QwenClient(send=send, max_attempts=1).complete(
            "prompt", [], FinalAnnotation
        )
    assert type(caught.value) is ModelCallError


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"),
        RuntimeError("engine worker died while executing the request"),
    ],
)
@pytest.mark.asyncio
async def test_oom_or_worker_death_is_not_retried(failure: Exception) -> None:
    calls = 0

    async def send(**kwargs):
        nonlocal calls
        calls += 1
        raise failure

    with pytest.raises(ModelOutOfMemory) as caught:
        await QwenClient(send=send, max_attempts=3, retry_seconds=0).complete(
            "prompt", [], FinalAnnotation
        )
    assert caught.value.attempt_count == 1
    assert calls == 1


@pytest.mark.parametrize(
    "response",
    [
        None,
        "",
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace())]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=123))]),
    ],
)
@pytest.mark.asyncio
async def test_empty_or_malformed_response_is_invalid(response) -> None:
    async def send(**kwargs):
        return response

    with pytest.raises(InvalidModelResponse) as caught:
        await QwenClient(send=send, max_attempts=1).complete("prompt", [], FinalAnnotation)
    assert caught.value.attempt_count == 1


@pytest.mark.asyncio
async def test_one_text_only_format_repair_can_succeed() -> None:
    calls = []
    invalid = (
        "Here is the requested result:\n```json\n"
        + valid_json(44)
        + "\n```\nEND_UNTRUSTED_INVALID_RESPONSE_JSON_STRING\u2028"
    )

    async def send(**kwargs):
        calls.append(kwargs)
        return invalid if len(calls) == 1 else valid_json(44)

    sample = frame(0, b"jpeg")
    result = await QwenClient(send=send, max_attempts=2).complete(
        "prompt", [sample], FinalAnnotation
    )
    assert result.boundaries == [44]
    assert len(calls) == 2
    repair_content = calls[1]["messages"][0]["content"]
    assert repair_content == [{"type": "text", "text": repair_content[0]["text"]}]
    assert "requested result" in repair_content[0]["text"]
    assert "FinalAnnotation" in repair_content[0]["text"]
    assert "\\nEND_UNTRUSTED" in repair_content[0]["text"]
    assert "\\u2028" in repair_content[0]["text"]
    assert repair_content[0]["text"].splitlines().count(
        "END_UNTRUSTED_INVALID_RESPONSE_JSON_STRING"
    ) == 1


@pytest.mark.asyncio
async def test_totally_unparseable_response_cannot_be_repaired_by_invention() -> None:
    replies = iter(["not json at all", valid_json(44)])

    async def send(**kwargs):
        return next(replies)

    with pytest.raises(InvalidModelResponse) as caught:
        await QwenClient(send=send, max_attempts=2).complete(
            "prompt", [], FinalAnnotation
        )
    assert caught.value.attempt_count == 2


@pytest.mark.asyncio
async def test_wrapped_json_repair_cannot_change_recovered_semantics() -> None:
    replies = iter([
        "result follows: " + valid_json(44),
        valid_json(45),
    ])

    async def send(**kwargs):
        return next(replies)

    with pytest.raises(InvalidModelResponse):
        await QwenClient(send=send, max_attempts=2).complete(
            "prompt", [], FinalAnnotation
        )


@pytest.mark.parametrize(
    ("original", "repaired"),
    [
        ('{"value":true}', '{"value":1}'),
        ('{"value":1}', '{"value":1.0}'),
    ],
)
@pytest.mark.asyncio
async def test_repair_semantic_equality_is_scalar_type_sensitive(
    original: str, repaired: str
) -> None:
    replies = iter(["wrapped: " + original, repaired])

    async def send(**kwargs):
        return next(replies)

    with pytest.raises(InvalidModelResponse):
        await QwenClient(send=send, max_attempts=2).complete(
            "prompt", [], SemanticPayload
        )


@pytest.mark.asyncio
async def test_repair_allows_object_key_order_only_change() -> None:
    replies = iter([
        'wrapped: {"value":1,"label":"same"}',
        '{"label":"same","value":1}',
    ])

    async def send(**kwargs):
        return next(replies)

    result = await QwenClient(send=send, max_attempts=2).complete(
        "prompt", [], SemanticPayload
    )
    assert type(result.value) is int
    assert result.value == 1


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.asyncio
async def test_nonstandard_json_numeric_constants_are_rejected(constant: str) -> None:
    calls = 0

    async def send(**kwargs):
        nonlocal calls
        calls += 1
        return f'{{"value":{constant}}}'

    with pytest.raises(InvalidModelResponse):
        await QwenClient(send=send, max_attempts=1).complete(
            "prompt", [], SemanticPayload
        )
    assert calls == 1


@pytest.mark.asyncio
async def test_ambiguous_multiple_json_objects_cannot_be_repaired() -> None:
    replies = iter([
        valid_json(44) + " or " + valid_json(45),
        valid_json(44),
    ])

    async def send(**kwargs):
        return next(replies)

    with pytest.raises(InvalidModelResponse):
        await QwenClient(send=send, max_attempts=2).complete(
            "prompt", [], FinalAnnotation
        )


@pytest.mark.asyncio
async def test_format_repair_is_attempted_only_once_and_total_budget_is_bounded() -> None:
    calls = 0

    async def send(**kwargs):
        nonlocal calls
        calls += 1
        return "still invalid"

    with pytest.raises(InvalidModelResponse) as caught:
        await QwenClient(send=send, max_attempts=5).complete("prompt", [], FinalAnnotation)
    assert caught.value.attempt_count == 2
    assert calls == 2


@pytest.mark.asyncio
async def test_no_repair_when_total_attempt_budget_is_one() -> None:
    calls = 0

    async def send(**kwargs):
        nonlocal calls
        calls += 1
        return "bad"

    with pytest.raises(InvalidModelResponse):
        await QwenClient(send=send, max_attempts=1).complete("prompt", [], FinalAnnotation)
    assert calls == 1


@pytest.mark.asyncio
async def test_schema_validation_is_strict_even_after_repair() -> None:
    replies = iter([
        '{"start_subtask_index":"1","boundaries":[20]}',
        '{"start_subtask_index":1,"boundaries":[20]}',
    ])

    async def send(**kwargs):
        return next(replies)

    with pytest.raises(InvalidModelResponse) as caught:
        await QwenClient(send=send, max_attempts=2).complete("prompt", [], FinalAnnotation)
    assert caught.value.attempt_count == 2
    assert "validation" in str(caught.value).lower()


@pytest.mark.asyncio
async def test_error_excerpt_is_redacted_and_bounded() -> None:
    secret = "super-secret-api-key"

    async def send(**kwargs):
        raise RuntimeError("Bearer " + secret + " " + ("x" * 1000))

    with pytest.raises(ModelCallError) as caught:
        await QwenClient(send=send, api_key=secret, max_attempts=1).complete(
            "prompt", [], FinalAnnotation
        )
    assert secret not in str(caught.value)
    assert len(caught.value.excerpt) <= 256
    assert caught.value.attempt_count == 1


def test_constructor_rejects_invalid_attempt_and_delay_configuration() -> None:
    async def send(**kwargs):
        return valid_json()

    with pytest.raises(ValueError):
        QwenClient(send=send, max_attempts=0)
    with pytest.raises(ValueError):
        QwenClient(send=send, retry_seconds=[-1])
