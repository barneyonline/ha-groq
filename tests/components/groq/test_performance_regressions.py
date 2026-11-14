"""Offline regressions for bounded processing, attempt accounting and ownership."""

import asyncio
from contextlib import aclosing
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from homeassistant.components.conversation import SystemContent
from homeassistant.exceptions import HomeAssistantError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.groq import api, audio_files, chat, runtime, tts
from custom_components.groq.api import (
    GroqApiClient,
    SpeechRequest,
    TextGenerationRequest,
)
from custom_components.groq.chat import (
    ChatHistoryCache,
    _assistant_native,
    _async_chat_log_messages,
    _result_tool_calls,
)
from custom_components.groq.conversation import GroqConversationEntity
from custom_components.groq.errors import GroqApiError, GroqRateLimitExceeded
from custom_components.groq.model_registry import GroqModelRegistry
from custom_components.groq.streaming import ChatStream
from custom_components.groq.text_generation import request_context_window_error

from .test_transport_architecture import Hass, Response, Session


@pytest.mark.asyncio
async def test_stream_framing_does_not_rescan_pending_bytes(monkeypatch):
    searched = 0

    class CountingBuffer(bytearray):
        def find(self, value, start=0):
            nonlocal searched
            searched += len(self) - start
            return super().find(value, start)

    class FragmentedResponse(Response):
        async def iter_chunked(self, _size):
            for _ in range(128):
                yield b"x" * 4096
            yield b"\nlast"

    monkeypatch.setattr(api, "bytearray", CountingBuffer, raising=False)
    size = 128 * 4096
    lines = [
        line
        async for line in api._iter_limited_response_lines(
            FragmentedResponse(), size + 5, "stream"
        )
    ]
    assert lines == [b"x" * size + b"\n", b"last"]
    assert searched <= size + 10


@pytest.mark.asyncio
async def test_cached_speech_bypasses_saturated_worker_pool(monkeypatch):
    monkeypatch.setattr(api, "MAX_SPEECH_INFLIGHT", 1)
    client, request, session = speech_client(monkeypatch)
    monkeypatch.setattr(api.asyncio, "sleep", AsyncMock())
    assert await client.async_synthesize_speech(request) == b"audio"
    worker_started, release = asyncio.Event(), asyncio.Event()

    async def synthesize(_):
        worker_started.set()
        await release.wait()
        return b"fresh"

    monkeypatch.setattr(client, "_async_synthesize_speech", synthesize)
    worker = asyncio.create_task(
        client.async_synthesize_speech(replace(request, text="uncached"))
    )
    try:
        await asyncio.wait_for(worker_started.wait(), 1)
        assert (
            await asyncio.wait_for(client.async_synthesize_speech(request), 0.1)
            == b"audio"
        )
        assert not release.is_set()
        assert session.calls == 2
    finally:
        release.set()
        await worker
        await client.async_shutdown()


@pytest.mark.asyncio
async def test_audio_timeout_after_response_starts_does_not_retry(monkeypatch, caplog):
    class InterruptedResponse(Response):
        async def iter_chunked(self, _size):
            yield b"partial audio"
            raise TimeoutError

    response = InterruptedResponse(headers={"content-type": "audio/wav"})
    session = Session(response)
    client, request, _ = speech_client(monkeypatch, session=session)
    backoff = AsyncMock()
    monkeypatch.setattr(api.asyncio, "sleep", backoff)
    with pytest.raises(GroqApiError, match="Timed out"):
        await client.async_synthesize_speech(request)
    assert response.closed
    assert len(session.calls) == 1
    assert len(client._tts_usage_state(request).request_timestamps) == 1
    backoff.assert_not_awaited()
    assert (
        "Timed out calling Groq API; the next request will check connectivity again"
        in caplog.text
    )


@pytest.mark.asyncio
async def test_concurrent_cold_ffmpeg_checks_share_probe(monkeypatch):
    entity = tts.GroqTTSEntity(
        Hass(), SimpleNamespace(data={}, options={}, unique_id="offline"), object()
    )
    started, release = asyncio.Event(), asyncio.Event()

    async def run(_cmd):
        started.set()
        await release.wait()
        return b""

    ffmpeg = AsyncMock(side_effect=run)
    monkeypatch.setattr(entity, "_async_run_ffmpeg", ffmpeg)
    callers = [
        asyncio.create_task(entity._async_check_ffmpeg("wav", True)) for _ in range(8)
    ]
    try:
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.sleep(0)
        callers[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await callers[0]
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(*callers[1:]), 1)
    assert ffmpeg.await_count == 2
    await entity._async_check_ffmpeg("wav", True)
    assert ffmpeg.await_count == 2


@pytest.mark.asyncio
async def test_ffmpeg_probe_recovers_after_last_waiter_cancels(monkeypatch):
    entity = tts.GroqTTSEntity(
        Hass(), SimpleNamespace(data={}, options={}, unique_id="offline"), object()
    )
    started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = 0

    async def ffmpeg(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
        return b""

    monkeypatch.setattr(entity, "_async_run_ffmpeg", ffmpeg)
    first = asyncio.create_task(entity._async_check_ffmpeg("wav", False))
    await asyncio.wait_for(started.wait(), 1)
    first.cancel()
    await asyncio.wait_for(cleaning.wait(), 1)
    replacement = asyncio.create_task(entity._async_check_ffmpeg("wav", False))
    await asyncio.sleep(0)
    assert calls == 1
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.wait_for(replacement, 1)
    assert calls == 3
    assert not entity._ffmpeg_probes


@pytest.mark.asyncio
async def test_failed_ffmpeg_probe_can_retry_and_profiles_are_independent(monkeypatch):
    entity = tts.GroqTTSEntity(
        Hass(), SimpleNamespace(data={}, options={}, unique_id="offline"), object()
    )
    ffmpeg = AsyncMock(side_effect=[HomeAssistantError("failed"), b"", b"", b"", b""])
    monkeypatch.setattr(entity, "_async_run_ffmpeg", ffmpeg)
    with pytest.raises(HomeAssistantError, match="failed"):
        await entity._async_check_ffmpeg("wav", False)
    await asyncio.gather(
        entity._async_check_ffmpeg("wav", False),
        entity._async_check_ffmpeg("mp3", False),
    )
    assert ffmpeg.await_count == 5
    assert entity._ffmpeg_capabilities == {("wav", False, None), ("mp3", False, None)}


@pytest.mark.asyncio
async def test_speech_queue_rejects_excess_and_reclaims_cancelled_waiter(monkeypatch):
    monkeypatch.setattr(api, "MAX_SPEECH_INFLIGHT", 1)
    monkeypatch.setattr(api, "MAX_SPEECH_QUEUED", 1)
    client, request, _ = speech_client(monkeypatch)
    started, release = asyncio.Event(), asyncio.Event()

    async def synthesize(_):
        started.set()
        await release.wait()
        return b"audio"

    monkeypatch.setattr(client, "_async_synthesize_speech", synthesize)
    first = asyncio.create_task(client.async_synthesize_speech(request))
    await asyncio.wait_for(started.wait(), 1)
    queued = asyncio.create_task(
        client.async_synthesize_speech(replace(request, text="queued"))
    )
    await asyncio.sleep(0)
    with pytest.raises(GroqApiError, match="queue is full"):
        await client.async_synthesize_speech(replace(request, text="overflow"))
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert set(client._speech_inflight) == {request}
    release.set()
    assert await first == b"audio"


@pytest.mark.asyncio
async def test_identical_queued_speech_shares_one_admission(monkeypatch):
    monkeypatch.setattr(api, "MAX_SPEECH_INFLIGHT", 1)
    monkeypatch.setattr(api, "MAX_SPEECH_QUEUED", 2)
    client, request, _ = speech_client(monkeypatch)
    started, release = asyncio.Event(), asyncio.Event()
    shared = replace(request, text="shared", cache_max=0)
    independent = replace(request, text="independent")
    calls = []

    async def synthesize(current):
        calls.append(current.text)
        if current == request:
            started.set()
            await release.wait()
        return current.text.encode()

    monkeypatch.setattr(client, "_async_synthesize_speech", synthesize)
    first = asyncio.create_task(client.async_synthesize_speech(request))
    await asyncio.wait_for(started.wait(), 1)
    duplicates = [
        asyncio.create_task(client.async_synthesize_speech(shared)) for _ in range(4)
    ]
    other = asyncio.create_task(client.async_synthesize_speech(independent))
    try:
        await asyncio.sleep(0)
        duplicates[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await duplicates[0]
    finally:
        release.set()
        results = await asyncio.gather(
            first, *duplicates[1:], other, return_exceptions=True
        )
        await client.async_shutdown()
    assert results == [
        request.text.encode(),
        b"shared",
        b"shared",
        b"shared",
        b"independent",
    ]
    assert calls == [request.text, "shared", "independent"]


@pytest.mark.asyncio
async def test_speech_queued_before_cache_fill_reuses_completed_result(monkeypatch):
    monkeypatch.setattr(api, "MAX_SPEECH_INFLIGHT", 1)
    client, request, session = speech_client(monkeypatch)
    started, release = asyncio.Event(), asyncio.Event()

    async def audio(*_args, **_kwargs):
        started.set()
        await release.wait()
        return b"audio"

    fetch = AsyncMock(side_effect=audio)
    monkeypatch.setattr(client, "_request_audio", fetch)
    first = asyncio.create_task(client.async_synthesize_speech(request))
    await asyncio.wait_for(started.wait(), 1)
    # Different synthesis settings give a distinct flight and the same cache key.
    queued = asyncio.create_task(
        client.async_synthesize_speech(replace(request, protect_free_tier=False))
    )
    await asyncio.sleep(0)
    assert not queued.done()
    release.set()
    assert await asyncio.gather(first, queued) == [b"audio", b"audio"]
    fetch.assert_awaited_once()
    assert session.calls == 0


@pytest.mark.asyncio
async def test_audio_spool_budget_released_after_overflow_and_success(monkeypatch):
    monkeypatch.setattr(audio_files, "MAX_AUDIO_SPOOL_BYTES", 8)
    budget = audio_files.AudioSpoolBudget()

    async def chunks(*values):
        for value in values:
            yield value

    async with audio_files.async_audio_chunk_paths(
        Hass(), chunks(b"123456"), budget
    ) as first:
        assert budget.used == 6
        with pytest.raises(GroqApiError, match="temporary audio storage"):
            async with audio_files.async_audio_chunk_paths(
                Hass(), chunks(b"1", b"23"), budget
            ):
                pytest.fail("the second announcement must exceed the shared budget")
        assert budget.used == 6
        assert Path(first[0]).read_bytes() == b"123456"
    assert budget.used == 0
    assert not Path(first[0]).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("removed_externally", [False, True])
async def test_audio_spool_stays_charged_until_failed_removal_succeeds(
    hass, monkeypatch, caplog, removed_externally
):
    monkeypatch.setattr(audio_files, "MAX_AUDIO_SPOOL_BYTES", 8)
    entry = MockConfigEntry(domain="groq", data={"api_key": "fixture"})
    entry.add_to_hass(hass)
    client = runtime.build_runtime(hass, entry).client
    budget = client.audio_spool_budget
    real_remove = audio_files.shutil.rmtree
    failed_path = None

    def remove(directory, ignore_errors=False):
        if Path(directory) == failed_path:
            if ignore_errors:
                return
            raise PermissionError("fixture denied removal")
        real_remove(directory, ignore_errors=ignore_errors)

    monkeypatch.setattr(audio_files.shutil, "rmtree", remove)

    async def chunks(value):
        yield value

    try:
        async with audio_files.async_audio_chunk_paths(
            hass, chunks(b"123456"), budget
        ) as paths:
            failed_path = Path(paths[0]).parent
        assert failed_path.exists()
        assert budget.used == 6
        assert "fixture denied removal" in caplog.text
        await client.async_shutdown()
        budget = runtime.build_runtime(hass, entry).client.audio_spool_budget
        with pytest.raises(GroqApiError, match="temporary audio storage"):
            async with audio_files.async_audio_chunk_paths(
                hass, chunks(b"123"), budget
            ):
                pytest.fail("files left by failed cleanup still occupy the budget")
        assert budget.used == 6
        old_path, failed_path = failed_path, None
        if removed_externally:
            real_remove(old_path)
        async with audio_files.async_audio_chunk_paths(
            hass, chunks(b"12345678"), budget
        ):
            assert not old_path.exists()
            assert budget.used == 8
        assert budget.used == 0
    finally:
        if failed_path is not None:
            real_remove(failed_path, ignore_errors=True)


def test_tts_stops_consuming_chunks_once_limit_is_exceeded(monkeypatch):
    def chunks(*_):
        yield from ("chunk" for _ in range(tts.MAX_LONG_TTS_CHUNKS + 1))
        pytest.fail("splitter consumed more than the first excess chunk")

    monkeypatch.setattr(tts, "_split_tts_text", chunks)
    with pytest.raises(ValueError, match="too many"):
        tts._tts_input_chunks("oversized", "")


@pytest.mark.parametrize("enable_long_tts", [False, True])
@pytest.mark.parametrize("word", ["é", "a "])
def test_tts_rejects_large_text_before_splitting(monkeypatch, enable_long_tts, word):
    def unexpected_split(*_):
        pytest.fail("a prefix already exceeds the total chunk capacity")

    monkeypatch.setattr(tts, "_split_tts_text", unexpected_split)
    with pytest.raises(ValueError, match="Groq Orpheus TTS"):
        tts._tts_input_chunks(word * 100_000, "", enable_long_tts=enable_long_tts)


def test_tts_length_precheck_accepts_collapsible_whitespace():
    assert tts._tts_input_chunks("a" + " " * 500 + "b", "", enable_long_tts=False) == [
        "a b"
    ]


@pytest.mark.asyncio
async def test_tool_iterations_reuse_serialization_and_token_estimates(monkeypatch):
    result = {"data": "é" * 2048}
    content = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "one", "tool_name": "Read", "tool_args": {"n": 1}}],
        },
        {
            "role": "tool_result",
            "tool_call_id": "one",
            "tool_name": "Read",
            "result": result,
        },
    ]
    log = SimpleNamespace(content=content)
    registry = GroqModelRegistry()
    cache = ChatHistoryCache()
    calls = []
    real_dumps = json.dumps

    def dumps(value, **kwargs):
        if isinstance(value, str) and len(value) >= 1024:
            calls.append(value)
        return real_dumps(value, **kwargs)

    monkeypatch.setattr(chat.json, "dumps", dumps)
    for _ in range(2):
        messages = await _async_chat_log_messages(
            Hass(), registry, "openai/gpt-oss-20b", log, "question", history_cache=cache
        )
        request = TextGenerationRequest(
            prompt="question", model="openai/gpt-oss-20b", messages=messages
        )
        assert (
            request_context_window_error(
                registry, request, string_estimates=cache.string_estimates
            )
            is None
        )
        assert json.loads(messages[3]["content"])["data"] == result["data"]
        messages[0]["content"] = "changed request"
        assert content[0]["content"] == "instructions"
    assert len(calls) == 1
    assert len(cache.tool_payloads) == 2


@pytest.mark.asyncio
async def test_tool_iteration_refreshes_replaced_home_assistant_system_prompt():
    content = [
        SystemContent(content="old instructions"),
        SystemContent(content="another instruction"),
        {"role": "user", "content": "question"},
    ]
    log = SimpleNamespace(content=content)
    cache = ChatHistoryCache()
    registry = GroqModelRegistry()
    for prompt in ("old instructions", "replacement", "", "restored"):
        content[0] = SystemContent(content=prompt)
        messages = await _async_chat_log_messages(
            Hass(), registry, "fake", log, "question", history_cache=cache
        )
        assert [message["content"] for message in messages] == [
            *([prompt] if prompt else []),
            "another instruction",
            "question",
        ]
    content[:] = [
        SystemContent(content="reset"),
        {"role": "user", "content": "question"},
    ]
    messages = await _async_chat_log_messages(
        Hass(), registry, "fake", log, "question", history_cache=cache
    )
    assert [message["content"] for message in messages] == ["reset", "question"]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [4, 2048])
async def test_json_uses_executor_for_large_bodies_and_keeps_session_serializer(
    monkeypatch, size
):
    calls = []
    payload = {"value": "é" * size}

    monkeypatch.setattr(api, "JSON_EXECUTOR_THRESHOLD", 1024)

    def serialize(value):
        calls.append(value)
        return json.dumps(value, ensure_ascii=False)

    def request(_session, _method, _url, **kwargs):
        if size >= 1024:
            assert kwargs["headers"]["Content-Type"] == "application/json"
            assert json.loads(kwargs["data"]) == payload
            assert "json" not in kwargs
        else:
            assert kwargs["json"] == payload
        return Response(body=json.dumps(payload).encode())

    monkeypatch.setattr(aiohttp.ClientSession, "request", request)
    async with aiohttp.ClientSession(json_serialize=serialize) as session:
        hass = Hass()
        executor = AsyncMock(wraps=hass.async_add_executor_job)
        monkeypatch.setattr(hass, "async_add_executor_job", executor)
        client = GroqApiClient(hass, api_key="fake", session=session)
        assert (
            await client._request_json("POST", "/test", json_payload=payload) == payload
        )
        assert calls == ([payload] if size >= 1024 else [])
        assert executor.await_count == (2 if size >= 1024 else 0)


@pytest.mark.asyncio
async def test_large_json_preserves_aiohttp_bytes_serializer(hass, monkeypatch, caplog):
    monkeypatch.setattr(api, "JSON_EXECUTOR_THRESHOLD", 64)
    payload = {"value": "é" * 256}
    bodies = []
    locale = ContextVar("locale", default="en")
    locale.set("fr")

    async def capture(request, _handler):
        assert request.body is not None
        bodies.append(await request.body.as_bytes())
        raise aiohttp.ClientConnectionError("offline capture")

    def serialize(value):
        return json.dumps(
            {**value, "locale": locale.get()}, ensure_ascii=False, separators=(",", ":")
        ).encode()

    async with aiohttp.ClientSession(
        json_serialize_bytes=serialize, middlewares=(capture,)
    ) as session:
        url = "http://localhost/test"
        with pytest.raises(aiohttp.ClientConnectionError, match="offline capture"):
            async with session.post(url, json=payload):
                pytest.fail("the middleware must stop the request before network I/O")
        client = GroqApiClient(
            hass, api_key="fake", base_url="http://localhost", session=session
        )
        with pytest.raises(GroqApiError, match="Network error"):
            await client._request_json("POST", "/test", json_payload=payload)
    assert bodies == [serialize(payload), serialize(payload)]
    assert (
        "Network error calling Groq API; the next request will check connectivity again"
        in caplog.text
    )


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
    monkeypatch.setattr("random.uniform", lambda *_: 0.25)
    monkeypatch.setattr(api.asyncio, "sleep", sleep)
    assert await client.async_synthesize_speech(request) == b"audio"
    assert await client.async_synthesize_speech(request) == b"audio"
    assert session.calls == 2
    sleep.assert_awaited_once_with(1.25)
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


@pytest.mark.asyncio
async def test_discarded_history_does_not_build_tool_descriptors(monkeypatch):
    from custom_components.groq import chat

    build_call = chat._tool_call_message

    def inspect_call(call, **kwargs):
        assert call["id"] != "discarded"
        return build_call(call, **kwargs)

    monkeypatch.setattr(chat, "_tool_call_message", inspect_call)
    content = [
        {"role": "system", "content": "first instruction"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "discarded", "tool_name": "Read", "tool_args": {}}],
        },
        {"role": "system", "content": "second instruction"},
        *({"role": "user", "content": str(i)} for i in range(12)),
    ]
    messages = await _async_chat_log_messages(
        Hass(), GroqModelRegistry(), "fake", SimpleNamespace(content=content), "11"
    )
    assert [message["content"] for message in messages] == [
        "first instruction",
        "second instruction",
        *(str(i) for i in range(12)),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["async_stream_text", "async_stream_chat"])
async def test_stream_validates_arguments_once_and_only_chat_collects_text(
    monkeypatch, method
):
    arguments = '{"entity":"light.a"}'
    parse = json.loads
    parsed = []
    collected = []
    result = ChatStream.result

    def count_parses(data, *args, **kwargs):
        if data == arguments:
            parsed.append(data)
        return parse(data, *args, **kwargs)

    def inspect_result(stream, *args, **kwargs):
        collected.extend(stream.content)
        return result(stream, *args, **kwargs)

    monkeypatch.setattr(json, "loads", count_parses)
    monkeypatch.setattr(ChatStream, "result", inspect_result)
    tool_delta = {
        "index": 0,
        "id": "call",
        "function": {"name": "Read", "arguments": arguments},
    }
    events = [
        {"choices": [{"delta": {"content": "Checking", "tool_calls": [tool_delta]}}]},
        {"choices": [{"finish_reason": "tool_calls"}], "usage": {"total_tokens": 5}},
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    response = Response(body=(body + "data: [DONE]\n\n").encode())
    client = GroqApiClient(Hass(), api_key="fake", session=Session(response))
    request = TextGenerationRequest(prompt="fake", model="fake", service_id="test")
    chunks = [chunk async for chunk in getattr(client, method)(request)]
    assert parsed == [arguments]
    assert chunks[0] == "Checking"
    assert client.usage.values["test"]["requests"] == 1
    if method == "async_stream_text":
        assert chunks == ["Checking"]
        assert collected == []
    else:
        assert collected == ["Checking"]
        assert _result_tool_calls(chunks[-1])[0].tool_args == {"entity": "light.a"}
        assert chunks[-1].tool_calls[0]["function"]["parsed_arguments"] == {
            "entity": "light.a"
        }
        assert (
            "parsed_arguments"
            not in chunks[-1].raw["choices"][0]["message"]["tool_calls"][0]["function"]
        )
    assert parsed == [arguments] * (2 if method == "async_stream_chat" else 1)


@pytest.mark.asyncio
async def test_simultaneous_model_refreshes_share_fetch_and_survive_one_cancellation():
    client = GroqApiClient(Hass(), api_key="fake")
    entered, release = asyncio.Event(), asyncio.Event()

    async def request(_method, path, **_kwargs):
        if path == "/models":
            return {"data": [{"id": "custom-model"}]}
        assert path == "/models/custom-model"
        entered.set()
        await release.wait()
        return {"id": "custom-model", "context_window": 100}

    client._request_json = AsyncMock(side_effect=request)
    first = asyncio.create_task(client.async_list_models())
    await entered.wait()
    second = asyncio.create_task(client.async_list_models())
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    assert (await second)[0].context_window == 100
    assert client._request_json.await_count == 2
    assert (await client.async_list_models())[0].context_window == 100
    assert client._request_json.await_count == 4


@pytest.mark.asyncio
async def test_simultaneous_audio_retries_are_spread(monkeypatch):
    from dataclasses import replace

    class ConcurrentFailures:
        def __init__(self):
            self.attempts = {}

        def request(self, *_args, json, **_kwargs):
            text = json["input"]
            self.attempts[text] = self.attempts.get(text, 0) + 1
            if self.attempts[text] == 1:
                raise aiohttp.ClientError("offline transient failure")
            return Response(body=b"audio", headers={"content-type": "audio/wav"})

    session = ConcurrentFailures()
    client, request, _ = speech_client(monkeypatch, session=session)
    delays = []
    both_waiting = asyncio.Event()
    sleep = asyncio.sleep

    async def backoff(delay):
        delays.append(delay)
        if len(delays) == 2:
            both_waiting.set()
        await both_waiting.wait()
        await sleep(0)

    monkeypatch.setattr(api.asyncio, "sleep", backoff)
    monkeypatch.setattr("random.uniform", lambda *_: 0.1 if not delays else 0.4)
    await asyncio.gather(
        client.async_synthesize_speech(request),
        client.async_synthesize_speech(replace(request, text="other")),
    )
    assert sorted(delays) == [1.1, 1.4]
    assert session.attempts == {"fake": 2, "other": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ["last_waiter", "shutdown", "caller_during_shutdown"])
async def test_model_refresh_cancellation_drains_owned_work(monkeypatch, end):
    client = GroqApiClient(Hass(), api_key="fake")
    entered, draining, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def fetch(*, hydrate):
        assert hydrate
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            draining.set()
            await release.wait()

    monkeypatch.setattr(client, "_async_list_models", fetch)
    refresh = asyncio.create_task(client.async_list_models())
    await entered.wait()
    if end != "last_waiter":
        shutdown = asyncio.create_task(client.async_shutdown())
        await draining.wait()
        with pytest.raises(GroqApiError, match="unloaded"):
            await client.async_list_models()
        assert not shutdown.done()
        if end == "caller_during_shutdown":
            refresh.cancel()
            with pytest.raises(asyncio.CancelledError):
                await refresh
            assert not shutdown.done()
        release.set()
        await shutdown
    else:
        refresh.cancel()
        await draining.wait()
        replacement = AsyncMock(return_value=[])
        monkeypatch.setattr(client, "_async_list_models", replacement)
        next_refresh = asyncio.create_task(client.async_list_models())
        await asyncio.sleep(0)
        replacement.assert_not_awaited()
        release.set()
        assert await next_refresh == []
        replacement.assert_awaited_once_with(hydrate=True)
    if end == "shutdown":
        with pytest.raises(GroqApiError, match="unloaded"):
            await refresh
    else:
        with pytest.raises(asyncio.CancelledError):
            await refresh
    assert client._model_refresh is None


@pytest.mark.asyncio
async def test_repeated_cancellation_of_last_refresh_waiter_drains_detail(monkeypatch):
    client = GroqApiClient(Hass(), api_key="fake")
    entered, cleaning, release, finished = (asyncio.Event() for _ in range(4))

    async def request(_method, path, **_kwargs):
        if path == "/models":
            return {"data": [{"id": "custom-model"}]}
        assert path == "/models/custom-model"
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            finished.set()

    monkeypatch.setattr(client, "_request_json", request)
    async with asyncio.timeout(3):
        refresh = asyncio.create_task(client.async_list_models())
        try:
            await entered.wait()
            refresh.cancel()
            await cleaning.wait()
            assert refresh.cancel()
            await asyncio.sleep(0)
            assert not refresh.done()
            assert not finished.is_set()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await refresh
            try:
                assert finished.is_set()
            finally:
                await client.async_shutdown()
        assert client._model_refresh is None


@pytest.mark.asyncio
async def test_shutdown_waits_for_already_cancelling_model_refresh(monkeypatch):
    client = GroqApiClient(Hass(), api_key="fake")
    entered, draining, release, cleaned = (asyncio.Event() for _ in range(4))

    async def request(_method, path, **_kwargs):
        if path == "/models":
            return {"data": [{"id": "custom-model"}]}
        assert path == "/models/custom-model"
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            draining.set()
            await release.wait()
            cleaned.set()

    monkeypatch.setattr(client, "_request_json", request)
    async with asyncio.timeout(3):
        refresh = asyncio.create_task(client.async_list_models())
        await entered.wait()
        refresh.cancel()
        await draining.wait()
        shutdown = asyncio.create_task(client.async_shutdown())
        try:
            await asyncio.sleep(0)
            assert not shutdown.done()
        finally:
            release.set()
            await shutdown
            with pytest.raises(asyncio.CancelledError):
                await refresh
        assert cleaned.is_set()
        assert client._model_refresh is None


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["shutdown", "replace"])
async def test_model_refresh_drains_every_detail_request(monkeypatch, action):
    client = GroqApiClient(Hass(), api_key="fake")
    started = asyncio.Event()
    slow_cleaning = asyncio.Event()
    release_slow = asyncio.Event()
    slow_finished = asyncio.Event()
    detail_tasks = {}
    catalog_requests = 0

    async def request(_method, path, **_kwargs):
        nonlocal catalog_requests
        if path == "/models":
            catalog_requests += 1
            if catalog_requests > 1:
                assert slow_finished.is_set()
                return {"data": []}
            return {"data": [{"id": "fast"}, {"id": "slow"}]}

        detail_tasks[path] = asyncio.current_task()
        if len(detail_tasks) == 2:
            started.set()
        try:
            await asyncio.Event().wait()
        finally:
            if path == "/models/slow":
                slow_cleaning.set()
                await release_slow.wait()
                slow_finished.set()

    monkeypatch.setattr(client, "_request_json", request)
    async with asyncio.timeout(3):
        refresh = asyncio.create_task(client.async_list_models())
        await started.wait()
        if action == "shutdown":
            operation = asyncio.create_task(client.async_shutdown())
        else:
            refresh.cancel()
            operation = asyncio.create_task(client.async_list_models())
        try:
            await slow_cleaning.wait()
            with pytest.raises(asyncio.CancelledError):
                await detail_tasks["/models/fast"]
            # Let the cancellation propagate through the worker and its waiters.
            for _ in range(10):
                await asyncio.sleep(0)
            assert not operation.done()
            assert client._model_refresh is not None
        finally:
            release_slow.set()
            await asyncio.gather(operation, refresh, return_exceptions=True)
        assert slow_finished.is_set()
        assert client._model_refresh is None
        if action == "shutdown":
            assert operation.result() is None
            with pytest.raises(GroqApiError, match="unloaded"):
                refresh.result()
        else:
            assert operation.result() == []
            with pytest.raises(asyncio.CancelledError):
                refresh.result()


@pytest.mark.asyncio
async def test_model_refresh_propagates_unexpected_detail_failure(monkeypatch):
    client = GroqApiClient(Hass(), api_key="fake")
    sibling_started = asyncio.Event()
    sibling_cleaning = asyncio.Event()
    release_sibling = asyncio.Event()
    sibling_finished = asyncio.Event()

    async def request(_method, path, **_kwargs):
        if path == "/models":
            return {"data": [{"id": "failing"}, {"id": "stalled"}]}
        if path == "/models/failing":
            await sibling_started.wait()
            raise RuntimeError("unexpected detail failure")
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_cleaning.set()
            await release_sibling.wait()
            sibling_finished.set()

    monkeypatch.setattr(client, "_request_json", request)
    refresh = asyncio.create_task(client.async_list_models())
    try:
        await asyncio.wait_for(sibling_cleaning.wait(), 2)
        assert not refresh.done()
    finally:
        release_sibling.set()
        if not sibling_cleaning.is_set():
            refresh.cancel()
        await asyncio.wait_for(asyncio.gather(refresh, return_exceptions=True), 2)
    with pytest.raises(RuntimeError, match="unexpected detail failure"):
        refresh.result()
    assert sibling_finished.is_set()
    assert client._model_refresh is None


@pytest.mark.asyncio
async def test_cancelled_shutdown_preserves_model_detail_cleanup(monkeypatch):
    client = GroqApiClient(Hass(), api_key="fake")
    entered, cleaning, release, finished = (asyncio.Event() for _ in range(4))

    async def request(_method, path, **_kwargs):
        if path == "/models":
            return {"data": [{"id": "custom-model"}]}
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            finished.set()

    monkeypatch.setattr(client, "_request_json", request)
    async with asyncio.timeout(3):
        refresh = asyncio.create_task(client.async_list_models())
        await entered.wait()
        shutdown = asyncio.create_task(client.async_shutdown())
        try:
            await cleaning.wait()
            for _ in range(2):
                shutdown.cancel()
                await asyncio.sleep(0)
            assert not shutdown.done()
            assert client._model_refresh is not None
        finally:
            release.set()
            await asyncio.gather(shutdown, refresh, return_exceptions=True)
        assert finished.is_set()
        assert client._model_refresh is None
        with pytest.raises(asyncio.CancelledError):
            shutdown.result()
        with pytest.raises(GroqApiError, match="unloaded"):
            refresh.result()


@pytest.mark.asyncio
@pytest.mark.parametrize("other_waiter", [False, True])
async def test_model_refresh_failure_is_retrieved_when_waiter_cancels(
    monkeypatch, other_waiter
):
    import gc
    import weakref

    client = GroqApiClient(Hass(), api_key="fake")
    loop = asyncio.get_running_loop()
    reported = []
    handler = loop.get_exception_handler()
    release = asyncio.Event()

    async def fetch(*, hydrate):
        assert hydrate
        await release.wait()
        waiter.cancel()
        raise GroqApiError("offline refresh failure")

    monkeypatch.setattr(client, "_async_list_models", fetch)
    loop.set_exception_handler(lambda _, context: reported.append(context["message"]))
    try:
        waiter = asyncio.create_task(client.async_list_models())
        await asyncio.sleep(0)
        worker_ref = weakref.ref(client._model_refresh.task)
        if other_waiter:
            remaining = asyncio.create_task(client.async_list_models())
            await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        if other_waiter:
            with pytest.raises(GroqApiError, match="offline refresh failure"):
                await remaining
            del remaining
        assert client._model_refresh is None
        if not other_waiter:
            # Prove the task was collected so an unretrieved failure would report.
            gc.collect()
            await asyncio.sleep(0)
            assert worker_ref() is None
        assert reported == []
    finally:
        loop.set_exception_handler(handler)


@pytest.mark.asyncio
async def test_history_counts_only_messages_with_valid_content():
    messages = await _async_chat_log_messages(
        Hass(),
        GroqModelRegistry(),
        "fake",
        SimpleNamespace(
            content=[
                {"role": "user", "content": "old user"},
                {"role": "tool_result", "result": {"data": "missing identity"}},
                {"role": "assistant", "tool_calls": [{"tool_args": {}}]},
                {"role": "assistant", "content": "reply"},
            ]
        ),
        "current user",
    )
    assert [message["content"] for message in messages] == [
        "old user",
        "reply",
        "current user",
    ]


def test_normalized_tool_arguments_reject_json_encoded_strings():
    client = GroqApiClient(Hass(), api_key="fake")
    call = {
        "id": "call",
        "function": {"name": "Read", "arguments": json.dumps("{}")},
    }
    result = client._chat_result({"choices": [{"message": {"tool_calls": [call]}}]})
    with pytest.raises(HomeAssistantError, match="invalid tool arguments"):
        _result_tool_calls(result)


@pytest.mark.parametrize("wire_format", [True, False])
def test_tool_execution_cannot_mutate_native_trace_arguments(monkeypatch, wire_format):
    from custom_components.groq import chat

    def unexpected_copy(_value):
        pytest.fail("Wire JSON should be decoded instead of deep-copied")

    client = GroqApiClient(Hass(), api_key="fake")
    original = {"area": "Kitchen", "options": {"targets": ["light.kitchen"]}}
    encoded = json.dumps(original)
    arguments = encoded if wire_format else deepcopy(original)
    call = {"id": "call", "function": {"name": "Read", "arguments": arguments}}
    result = client._chat_result({"choices": [{"message": {"tool_calls": [call]}}]})
    trace = _assistant_native(result)
    if wire_format:
        monkeypatch.setattr(chat, "deepcopy", unexpected_copy)

    (tool_input,) = _result_tool_calls(result)
    tool_input.tool_args["area"] = "kitchen_id"
    tool_input.tool_args["options"]["targets"].append("light.hall")

    function = trace["tool_calls"][0]["function"]
    if wire_format:
        assert function["arguments"] == encoded
        assert function["parsed_arguments"] == original
    else:
        assert function["arguments"] == original


@pytest.mark.parametrize("wire_format", [True, False])
def test_deeply_nested_tool_arguments_keep_execution_and_trace_isolated(wire_format):
    client = GroqApiClient(Hass(), api_key="fake")
    depth = 510
    encoded = '{"options":' + "[" * depth + '{"area":"Kitchen"}' + "]" * depth + "}"
    arguments = encoded if wire_format else json.loads(encoded)
    call = {"id": "call", "function": {"name": "Read", "arguments": arguments}}
    result = client._chat_result({"choices": [{"message": {"tool_calls": [call]}}]})
    (tool_input,) = _result_tool_calls(result)
    function = _assistant_native(result)["tool_calls"][0]["function"]
    source = function["parsed_arguments"] if wire_format else function["arguments"]
    original, execution = source["options"], tool_input.tool_args["options"]
    for _ in range(depth):
        original, execution = original[0], execution[0]
    assert execution == {"area": "Kitchen"}
    execution["area"] = "kitchen_id"
    assert original == {"area": "Kitchen"}
