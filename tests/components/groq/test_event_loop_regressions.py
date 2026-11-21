"""Regression coverage for cooperative stream processing and large JSON decode."""

import asyncio
import json
from contextlib import aclosing
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from homeassistant.components import conversation
from homeassistant.components.ai_task import GenDataTask
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.chat_session import async_get_chat_session

from custom_components.groq import ai_task, api
from custom_components.groq.errors import GroqResponseError

from .test_foundation import DummyEntry
from .test_tool_turn_lifecycle import ToolSession, fixture_tool_api
from .test_transport_architecture import Response, Session


class BufferedResponse(Response):
    """Yield immediately available slices as an aiohttp response can."""

    max_fragment_size: int | None = None

    async def iter_chunked(self, size):
        if self.max_fragment_size is not None:
            size = min(size, self.max_fragment_size)
        for start in range(0, len(self.body), size):
            yield self.body[start : start + size]


@pytest.mark.asyncio
@pytest.mark.parametrize(("line_size", "fragment_size"), [(1025, 1), (131073, 16384)])
async def test_fragmented_stream_yields_while_framing_first_line(
    line_size, fragment_size
):
    data = b"x" * (line_size - 1) + b"\n"
    observed = []
    received_chunks = 0

    class FragmentedResponse(Response):
        async def iter_chunked(self, _size):
            nonlocal received_chunks
            for offset in range(0, len(data), fragment_size):
                received_chunks += 1
                yield data[offset : offset + fragment_size]

    asyncio.get_running_loop().call_soon(lambda: observed.append(received_chunks))
    async with aclosing(
        api._iter_limited_response_lines(FragmentedResponse(), line_size, "stream")
    ) as lines:
        assert await anext(lines) == data
        assert observed and 0 < observed[0] < received_chunks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("count", "content_size", "fragment_size"),
    [(512, 1, None), (16, 16384, None), (16, 128, 1)],
)
async def test_buffered_stream_yields_before_draining(
    hass, count, content_size, fragment_size
):
    chunks = [str(i) + "x" * content_size for i in range(count)]
    events = [
        f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
        for chunk in chunks
    ]
    response = BufferedResponse(body=("".join(events) + "data: [DONE]\n\n").encode())
    response.max_fragment_size = fragment_size
    client = api.GroqApiClient(hass, api_key="fake", session=Session(response))
    received = []
    counts = []
    asyncio.get_running_loop().call_soon(lambda: counts.append(len(received)))
    async with aclosing(
        client.async_stream_text(api.TextGenerationRequest(prompt="test", model="test"))
    ) as stream:
        async for chunk in stream:
            received.append(chunk)

    assert received == chunks
    assert counts and 0 < counts[0] < len(chunks)
    assert response.closed


@pytest.mark.asyncio
async def test_buffered_stream_cancellation_at_checkpoint_closes_response(hass):
    event = b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
    response = BufferedResponse(body=event * 512 + b"data: [DONE]\n\n")
    client = api.GroqApiClient(hass, api_key="fake", session=Session(response))

    async def consume():
        task = asyncio.current_task()
        asyncio.get_running_loop().call_soon(task.cancel)
        async with aclosing(
            client.async_stream_text(
                api.TextGenerationRequest(prompt="test", model="test")
            )
        ) as stream:
            async for _ in stream:
                pass

    task = asyncio.create_task(consume())
    with pytest.raises(asyncio.CancelledError):
        await task
    assert response.closed


class RecordingExecutor:
    """Record and execute the jobs the integration submits to Home Assistant."""

    def __init__(self):
        self.calls = []

    async def async_add_executor_job(self, target, *args):
        self.calls.append((target, args))
        return await asyncio.to_thread(target, *args)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["valid", "malformed", "wrong_shape"])
async def test_structured_api_offloads_large_inner_json(hass, monkeypatch, outcome):
    content = '["' + "x" * 65536 + ("" if outcome == "malformed" else '"]')
    body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    response = BufferedResponse(body=body)
    loop = asyncio.get_running_loop()
    checkpoint = Event()
    executor = RecordingExecutor()

    def decode_after_checkpoint(value):
        # A ready loop callback can run while the worker job is pending.
        assert checkpoint.wait(3)
        return json.loads(value)

    async def schedule_job(target, *args):
        if target is json.loads:
            loop.call_soon(checkpoint.set)
            target = decode_after_checkpoint
        return await executor.async_add_executor_job(target, *args)

    monkeypatch.setattr(hass, "async_add_executor_job", schedule_job)
    client = api.GroqApiClient(hass, api_key="fake", session=Session(response))
    request = api.StructuredGenerationRequest(
        prompt="test",
        model="test",
        schema={"type": "object"} if outcome == "wrong_shape" else None,
    )
    if outcome == "malformed":
        with pytest.raises(GroqResponseError, match="structured response"):
            await client.async_generate_structured(request)
    elif outcome == "wrong_shape":
        with pytest.raises(HomeAssistantError, match="requested structure") as error:
            await client.async_generate_structured(request)
        assert error.value.translation_key == "structured_response_invalid"
    else:
        assert (await client.async_generate_structured(request))["data"] == [
            "x" * 65536
        ]
    assert response.closed
    assert (decode_after_checkpoint, (content,)) in executor.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("use_tools", [False, True])
@pytest.mark.parametrize("valid", [True, False])
async def test_ai_task_offloads_large_fenced_json(hass, monkeypatch, use_tools, valid):
    schema = {"type": "string"}
    content = '```json\n"' + "x" * 65536
    content += ('"' if valid else "") + "\n```"
    tools_called = []

    async def run_tool(tool_input):
        tools_called.append(tool_input.id)
        return llm.ToolResult(data={"ready": True})

    final_round = 2 if use_tools else 1
    session = ToolSession(int(use_tools), final_round=final_round, final_text=content)
    client = api.GroqApiClient(hass, api_key="fake", session=session)
    model = "openai/gpt-oss-20b" if use_tools else "llama-3.1-8b-instant"
    entity = ai_task.GroqAITaskEntity(
        hass,
        DummyEntry(),
        {"model": model, "schema": schema, "structured_outputs": True},
        client,
    )
    entity.platform = SimpleNamespace(domain="groq")
    entity.entity_id = "ai_task.fixture"
    monkeypatch.setattr(entity, "async_write_ha_state", Mock())
    executor = RecordingExecutor()
    monkeypatch.setattr(hass, "async_add_executor_job", executor.async_add_executor_job)
    task = GenDataTask(
        name="fixture",
        instructions="Return text",
        llm_api=fixture_tool_api(hass, "large_json", run_tool) if use_tools else None,
    )
    with async_get_chat_session(hass, context=None) as chat_session:
        if valid:
            result = await entity.internal_async_generate_data(chat_session, task)
            assert result.data == "x" * 65536
            with conversation.async_get_chat_log(hass, chat_session) as log:
                assert log.content[-1].content == content
        else:
            with pytest.raises(HomeAssistantError, match="requested structure"):
                await entity.internal_async_generate_data(chat_session, task)

    assert (ai_task._decode_json_text, (content,)) in executor.calls
    assert len(session.requests) == final_round
    assert all(response.closed for response in session.responses)
    if use_tools:
        assert tools_called == ["1-0"]
        assert session.requests[-1]["messages"][-1]["role"] == "tool"
    else:
        assert not tools_called
