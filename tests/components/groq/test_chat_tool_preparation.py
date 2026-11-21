"""Check the cost and wire semantics of preparing tool history."""

import asyncio
import json
from contextvars import ContextVar
from threading import Event, get_ident
from unittest.mock import AsyncMock, Mock

import pytest
from homeassistant.components import conversation
from homeassistant.components.ai_task import GenDataTask
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from custom_components.groq import chat
from custom_components.groq.ai_task import GroqAITaskEntity
from custom_components.groq.api import TextGenerationRequest
from custom_components.groq.model_registry import GroqModelRegistry
from custom_components.groq.text_generation import request_context_window_error

from .test_foundation import DummyEntry
from .test_transport_architecture import Hass

_MODEL = "openai/gpt-oss-20b"


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [False, True])
async def test_no_tool_ai_task_reuses_large_history_estimate(
    hass, monkeypatch, structured
):
    plain = "x" * (1024 * 1024)
    log = conversation.ChatLog(
        hass,
        "fixture",
        content=[
            conversation.SystemContent(content="System prompt."),
            conversation.UserContent(content="Read"),
            conversation.AssistantContent(
                agent_id="groq",
                tool_calls=[
                    llm.ToolInput(tool_name="FakeRead", tool_args={}, id="read")
                ],
            ),
            conversation.ToolResultContent(
                agent_id="groq",
                tool_name="FakeRead",
                tool_call_id="read",
                result=llm.ToolResult(data={"value": plain}),
            ),
            conversation.UserContent(content="Summarize"),
        ],
    )
    provider = AsyncMock()
    entity = GroqAITaskEntity(
        hass,
        DummyEntry(),
        {
            "model": _MODEL,
            "structured_outputs": structured,
            "schema": {"type": "object"},
        },
        provider,
    )
    dumps = json.dumps
    string_encodes = []

    def track_large_strings(value, *args, **kwargs):
        if isinstance(value, str) and len(value) > len(plain):
            string_encodes.append(get_ident())
        return dumps(value, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", track_large_strings)
    with pytest.raises(HomeAssistantError, match="context window"):
        await entity._async_generate_data(
            GenDataTask(name="summary", instructions="Summarize"), log
        )

    assert not string_encodes
    provider.async_generate_text.assert_not_awaited()
    provider.async_generate_structured.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("large", [False, True])
async def test_chat_tool_payload_preparation_keeps_wire_and_guard_semantics(
    hass, monkeypatch, large
):
    """Move only expensive encoding off the loop and reuse it within a turn."""
    row = {"state": "on", "label": 'é💡"\\\n' + "x" * 96, "count": 10, "ratio": 1.5}
    result = llm.ToolResult(data={"rows": [row] * (32_768 if large else 16)})
    log = conversation.ChatLog(
        hass,
        "fixture",
        content=[
            conversation.UserContent(content="read"),
            conversation.AssistantContent(
                agent_id="groq",
                tool_calls=[
                    llm.ToolInput(
                        tool_name="FakeRead", tool_args={"light": "one"}, id="read"
                    )
                ],
            ),
            conversation.ToolResultContent(
                agent_id="groq",
                tool_name="FakeRead",
                tool_call_id="read",
                result=result,
            ),
        ],
    )
    registry = GroqModelRegistry()
    cache = chat.ChatHistoryCache()
    executor = Mock(wraps=hass.async_add_executor_job)
    monkeypatch.setattr(hass, "async_add_executor_job", executor)
    loop_thread = get_ident()
    started = asyncio.Event()
    release = Event()
    loop = asyncio.get_running_loop()
    result_threads = []
    original_dumps = json.dumps

    def observed_dumps(value, **kwargs):
        if isinstance(value, dict) and value.get("data") is result.data:
            result_threads.append(get_ident())
            loop.call_soon_threadsafe(started.set)
            if large and get_ident() != loop_thread:
                assert release.wait(5)
        return original_dumps(value, **kwargs)

    monkeypatch.setattr(chat.json, "dumps", observed_dumps)
    expected = original_dumps(
        {"data": result.data, "error": False}, separators=(",", ":"), default=str
    )
    for attempt in range(2):
        pending = asyncio.create_task(
            chat._async_chat_log_messages(
                hass, registry, _MODEL, log, "read", history_cache=cache
            )
        )
        if large and attempt == 0:
            try:
                await asyncio.wait_for(started.wait(), 2)
                assert not pending.done()
            finally:
                release.set()
        messages = await pending
        assert messages[-1]["content"] == expected
        assert json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]) == {
            "light": "one"
        }
        request = TextGenerationRequest(prompt="read", model=_MODEL, messages=messages)
        cached = request_context_window_error(
            registry, request, string_estimates=cache.string_estimates
        )
        assert cached == request_context_window_error(registry, request)
        assert (cached is not None) is large

    assert executor.call_count == int(large)
    assert len(result_threads) == 1
    assert (result_threads[0] != loop_thread) is large


@pytest.mark.asyncio
async def test_tool_result_str_fallback_keeps_context_in_executor():
    """A small custom value can return an arbitrarily large fallback string."""
    text = ContextVar("text", default="context was lost")

    class Export:
        def __str__(self):
            return text.get()

    cache = chat.ChatHistoryCache()
    value = "x" * (256 * 1024)
    token = text.set(value)
    try:
        wire = await cache.async_serialize_tool(
            Hass(), llm.ToolResult(data={"export": Export()}), result=True
        )
    finally:
        text.reset(token)

    assert json.loads(wire) == {"data": {"export": value}, "error": False}
    assert cache.string_estimates[id(wire)][0] is wire
