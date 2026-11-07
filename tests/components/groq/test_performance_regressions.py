"""Offline regressions for bounded processing, attempt accounting and ownership."""

import asyncio
from contextlib import aclosing
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from homeassistant.exceptions import HomeAssistantError

from custom_components.groq import api
from custom_components.groq.api import (
    GroqApiClient,
    SpeechRequest,
    TextGenerationRequest,
)
from custom_components.groq.chat import _async_chat_log_messages
from custom_components.groq.conversation import GroqConversationEntity
from custom_components.groq.errors import GroqApiError, GroqRateLimitExceeded
from custom_components.groq.model_registry import GroqModelRegistry
from custom_components.groq.streaming import ChatStream

from .test_transport_architecture import Hass, Response, Session


class DiscardedValue:
    def __str__(self):
        raise AssertionError("Discarded tool results must not be serialized")


@pytest.mark.asyncio
@pytest.mark.parametrize("objects", [False, True])
async def test_history_selects_before_serializing_tools_and_leaves_log_unchanged(
    objects,
):
    items = [
        {"role": "system", "content": "Keep this instruction"},
        {"role": "user", "content": "old"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "old", "tool_name": "Read", "tool_args": {"discard": object()}}
            ],
        },
        {
            "role": "tool_result",
            "tool_call_id": "old",
            "tool_name": "Read",
            "result": {"data": DiscardedValue()},
        },
        *[
            {"role": "user" if i % 2 == 0 else "assistant", "content": str(i)}
            for i in range(14)
        ],
        {"role": "user", "content": "current"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "new", "tool_name": "Read", "tool_args": {"n": 1}}],
        },
        {
            "role": "tool_result",
            "tool_call_id": "new",
            "tool_name": "Read",
            "result": {"data": {"value": 2}, "error": True},
        },
    ]
    if objects:
        items = [SimpleNamespace(**item) for item in items]
    snapshot = deepcopy(items)
    result = await _async_chat_log_messages(
        Hass(),
        GroqModelRegistry(),
        "openai/gpt-oss-20b",
        SimpleNamespace(content=items),
        "current",
    )
    assert result[0] == {"role": "system", "content": "Keep this instruction"}
    assert len(result) == 13
    assert result[1] == {"role": "assistant", "content": "5"}
    assert result[-2]["tool_calls"][0]["function"]["arguments"] == '{"n":1}'
    assert json.loads(result[-1]["content"]) == {"data": {"value": 2}, "error": True}
    # Compare JSON-compatible retained inputs, including the original tool_args.
    assert items[4:] == snapshot[4:]


def test_stream_joins_fragmented_interleaved_names_once_without_mutating_results():
    def event(calls, finish=None):
        return {"choices": [{"delta": {"tool_calls": calls}, "finish_reason": finish}]}

    stream = ChatStream("fake")
    for index, call_id in ((1, "second"), (0, "first")):
        stream.add(
            event(
                [
                    {
                        "index": index,
                        "id": call_id,
                        "function": {"name": "Re", "arguments": '{"value":"'},
                    }
                ]
            )
        )
    for _ in range(1000):
        stream.add(
            event(
                [
                    {"index": index, "function": {"arguments": "x" * 16}}
                    for index in (0, 1)
                ]
            )
        )
    for index in (0, 1):
        stream.add(
            event(
                [
                    {
                        "index": index,
                        "function": {"name": "ad", "arguments": '"}'},
                    }
                ]
            )
        )
    stream.add(event([], "tool_calls"))
    first = stream.result()
    assert stream.result() == first
    calls = first["choices"][0]["message"]["tool_calls"]
    assert [call["id"] for call in calls] == ["first", "second"]
    assert all(call["function"]["name"] == "Read" for call in calls)
    assert all(
        json.loads(call["function"]["arguments"]) == {"value": "x" * 16000}
        for call in calls
    )
    calls[0]["function"]["arguments"] = "changed"
    fresh_calls = stream.result()["choices"][0]["message"]["tool_calls"]
    assert json.loads(fresh_calls[0]["function"]["arguments"]) == {"value": "x" * 16000}


class FlakySession:
    def __init__(self, status=200):
        self.calls = 0
        self.response = Response(
            status=status, body=b"audio", headers={"content-type": "audio/wav"}
        )

    def request(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise aiohttp.ClientError("offline transient failure")
        return self.response


def speech_client(monkeypatch, *, limit=10, protect=True, session=None):
    session = session or FlakySession()
    client = GroqApiClient(Hass(), api_key="fake", session=session)
    monkeypatch.setattr(
        client,
        "_free_tier_limits",
        lambda _: {
            "requests_per_minute": limit,
            "requests_per_day": 100,
            "tokens_per_minute": 10000,
            "tokens_per_day": 100000,
        },
    )
    request = SpeechRequest(
        text="fake",
        model="fake",
        voice="fake",
        service_id="test",
        protect_free_tier=protect,
    )
    return client, request, session


@pytest.mark.asyncio
@pytest.mark.parametrize("protect", [False, True])
async def test_audio_retry_counts_each_attempt_and_cache_hits_cost_nothing(
    monkeypatch, protect
):
    client, request, session = speech_client(monkeypatch, protect=protect)
    sleep = AsyncMock()
    monkeypatch.setattr(api.asyncio, "sleep", sleep)
    assert await client.async_synthesize_speech(request) == b"audio"
    assert await client.async_synthesize_speech(request) == b"audio"
    assert session.calls == 2
    sleep.assert_awaited_once_with(1)
    if protect:
        assert len(client._tts_usage_state(request).request_timestamps) == 2
    else:
        assert client._tts_usage == {}


@pytest.mark.asyncio
async def test_retry_stops_at_local_limit_without_charging_blocked_attempt(monkeypatch):
    client, request, session = speech_client(monkeypatch, limit=1)
    monkeypatch.setattr(api.asyncio, "sleep", AsyncMock())
    with pytest.raises(GroqApiError, match="requests per minute"):
        await client.async_synthesize_speech(request)
    assert session.calls == 1
    assert len(client._tts_usage_state(request).request_timestamps) == 1


@pytest.mark.asyncio
async def test_retry_rechecks_cooldown_established_during_backoff(monkeypatch):
    client, request, session = speech_client(monkeypatch)

    async def backoff(_):
        client._rate_limiter.update_from_headers("test", {"retry-after": "60"})

    monkeypatch.setattr(api.asyncio, "sleep", backoff)
    with pytest.raises(GroqRateLimitExceeded):
        await client.async_synthesize_speech(request)
    assert session.calls == 1
    assert len(client._tts_usage_state(request).request_timestamps) == 1


@pytest.mark.asyncio
async def test_cancellation_during_audio_backoff_does_not_retry(monkeypatch):
    client, request, session = speech_client(monkeypatch)
    sleeping = asyncio.Event()

    async def backoff(_):
        sleeping.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(api.asyncio, "sleep", backoff)
    task = asyncio.create_task(client.async_synthesize_speech(request))
    await sleeping.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert session.calls == 1
    assert len(client._tts_usage_state(request).request_timestamps) == 1
    assert not client._speech_inflight


@pytest.mark.asyncio
async def test_coalesced_speech_waiters_share_one_quota_reservation(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    class WaitingResponse(Response):
        async def iter_chunked(self, _):
            entered.set()
            await release.wait()
            yield self.body

    response = WaitingResponse(body=b"audio", headers={"content-type": "audio/wav"})
    session = Session(response)
    client, request, _ = speech_client(monkeypatch, session=session)
    first = asyncio.create_task(client.async_synthesize_speech(request))
    await entered.wait()
    second = asyncio.create_task(client.async_synthesize_speech(request))
    release.set()
    assert await asyncio.gather(first, second) == [b"audio", b"audio"]
    assert len(session.calls) == 1
    assert len(client._tts_usage_state(request).request_timestamps) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["async_stream_chat", "async_stream_text"])
@pytest.mark.parametrize("end", ["close", "complete", "consumer_error", "cancel_read"])
async def test_public_streams_release_response_before_returning(method, end):
    waiting = asyncio.Event()

    class StreamResponse(Response):
        async def iter_chunked(self, _):
            yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            if end == "cancel_read":
                waiting.set()
                await asyncio.Event().wait()
            yield b"data: [DONE]\n\n"

    response = StreamResponse()
    client = GroqApiClient(Hass(), api_key="fake", session=Session(response))
    stream = getattr(client, method)(TextGenerationRequest(prompt="fake", model="fake"))
    assert await anext(stream) == "hello"
    if end == "complete":
        async for _ in stream:
            pass
    elif end == "consumer_error":
        with pytest.raises(ValueError, match="downstream"):
            async with aclosing(stream):
                raise ValueError("downstream")
    elif end == "cancel_read":
        task = asyncio.create_task(anext(stream))
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await stream.aclose()
    assert response.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ["break", "error", "cancel", "invalid_tool"])
async def test_assist_owns_stream_when_ha_consumer_stops_early(end):
    response = Response(
        body=(
            b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call",'
            b'"function":{"name":"Unexposed","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}\n\n'
            b"data: [DONE]\n\n"
        )
    )
    client = GroqApiClient(Hass(), api_key="fake", session=Session(response))
    consumer_closed = False
    processing = asyncio.Event()

    class ChatLog:
        async def async_add_delta_content_stream(self, _, stream):
            nonlocal consumer_closed
            try:
                assert await anext(stream) == {"role": "assistant"}
                assert await anext(stream) == {"content": "hello"}
                if end == "error":
                    raise ValueError("downstream")
                if end == "cancel":
                    processing.set()
                    await asyncio.Event().wait()
                if end == "invalid_tool":
                    await anext(stream)
                return
                yield  # This adapter is an async generator, like HA's implementation.
            finally:
                consumer_closed = True

    request = TextGenerationRequest(prompt="fake", model="fake")
    coroutine = GroqConversationEntity._async_stream_message(
        SimpleNamespace(_client=client),
        SimpleNamespace(agent_id="fake"),
        ChatLog(),
        request,
    )
    if end == "error":
        with pytest.raises(ValueError, match="downstream"):
            await coroutine
    elif end == "invalid_tool":
        with pytest.raises(HomeAssistantError, match="unexposed"):
            await coroutine
    elif end == "cancel":
        task = asyncio.create_task(coroutine)
        await processing.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert await coroutine == ""
    assert response.closed
    assert consumer_closed
