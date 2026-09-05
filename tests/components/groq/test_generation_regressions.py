"""Behavioral regressions for generation policy and Home Assistant boundaries."""

from __future__ import annotations

from types import SimpleNamespace
from threading import get_ident
from unittest.mock import AsyncMock, Mock

import pytest
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.components.ai_task import GenDataTask
from homeassistant.core import Context, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.chat_session import async_get_chat_session

from custom_components.groq import services
from custom_components.groq.ai_task import GroqAITaskEntity
from custom_components.groq.api import (
    GroqApiClient,
    StructuredGenerationRequest,
    TextGenerationRequest,
)
from custom_components.groq.conversation import GroqConversationEntity
from custom_components.groq.chat import _async_chat_log_messages, _result_tool_calls
from custom_components.groq.feature_registry import GroqFeature, GroqFeatureRegistry
from custom_components.groq.model_registry import GroqModelRegistry
from custom_components.groq.prompt_cache import GroqPromptCache
from custom_components.groq.structured import validate_json_schema_data
from custom_components.groq.text_generation import request_context_window_error

from .test_foundation import (
    DummyEntry,
    DummyHass,
    DummyResponse,
    DummySession,
    DummyTextClient,
)

SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


@pytest.mark.asyncio
async def test_inline_image_validation_runs_outside_event_loop(hass, monkeypatch):
    loop_thread = get_ident()
    original = services._validate_image_url

    def validate(url):
        assert get_ident() != loop_thread
        return original(url)

    monkeypatch.setattr(services, "_validate_image_url", validate)
    url = "data:image/png;base64,aW1hZ2U="
    assert (
        await services._image_url_from_call(
            hass, SimpleNamespace(data={"image_url": url})
        )
        == url
    )


def test_inline_images_are_not_counted_as_base64_text_tokens():
    """Image transport must not reject otherwise small prompts locally."""
    registry = GroqModelRegistry()
    request = TextGenerationRequest(
        prompt="Describe",
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + "A" * 600_000},
                    },
                ],
            }
        ],
    )
    assert request_context_window_error(registry, request) is None
    assert (
        request_context_window_error(
            registry,
            TextGenerationRequest(prompt="x" * 600_000, model="llama-3.1-8b-instant"),
        )
        is not None
    )


@pytest.mark.parametrize("arguments", ["{", "[]", "null", "1", [], None, 123])
def test_malformed_tool_arguments_never_become_default_operations(arguments):
    """A malformed optional-argument call must never turn into executable {}."""
    with pytest.raises(HomeAssistantError):
        _result_tool_calls(
            SimpleNamespace(
                tool_calls=[
                    {
                        "id": "one",
                        "function": {"name": "Operate", "arguments": arguments},
                    }
                ]
            )
        )


@pytest.mark.parametrize("call_ids", [["one", "one"], [None], [""], [123]])
def test_invalid_tool_ids_are_rejected(call_ids):
    with pytest.raises(HomeAssistantError):
        _result_tool_calls(
            SimpleNamespace(
                tool_calls=[
                    {"id": call_id, "function": {"name": "Operate", "arguments": "{}"}}
                    for call_id in call_ids
                ]
            )
        )


def test_same_name_tool_calls_remain_independent():
    calls = _result_tool_calls(
        SimpleNamespace(
            tool_calls=[
                {"id": call_id, "function": {"name": "Operate", "arguments": "{}"}}
                for call_id in ("one", "two")
            ]
        )
    )
    assert [(call.id, call.tool_args) for call in calls] == [("one", {}), ("two", {})]


@pytest.mark.asyncio
async def test_discarded_images_are_not_read_and_current_image_is_reused(
    tmp_path, monkeypatch
):
    """Only retained history causes I/O, even through multiple tool iterations."""
    current = tmp_path / "current.png"
    current.write_bytes(b"image")
    attachments = [SimpleNamespace(mime_type="image/png", path=current)]
    log = SimpleNamespace(
        content=[
            {
                "role": "user",
                "content": "old",
                "attachments": [
                    SimpleNamespace(
                        mime_type="image/png", path=tmp_path / "deleted.png"
                    )
                ],
            }
        ]
        + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": str(i)}
            for i in range(14)
        ]
        + [
            {
                "role": "user",
                "content": "Describe",
                "attachments": attachments,
            }
        ]
    )
    hass = DummyHass()
    read = AsyncMock(wraps=hass.async_add_executor_job)
    monkeypatch.setattr(hass, "async_add_executor_job", read)
    cache = {}
    for _ in range(2):
        messages = await _async_chat_log_messages(
            hass,
            GroqModelRegistry(),
            "meta-llama/llama-4-scout-17b-16e-instruct",
            log,
            "Describe",
            attachments,
            cache,
        )
        assert messages[-1]["content"][1]["image_url"]["url"].endswith("aW1hZ2U=")
    assert read.await_count == 1


@pytest.mark.asyncio
async def test_large_tool_batch_preserves_current_request():
    log = SimpleNamespace(
        content=[
            {"role": "user", "content": "Operate"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": str(i), "tool_name": "Operate", "tool_args": {}}
                    for i in range(14)
                ],
            },
        ]
        + [
            {
                "role": "tool_result",
                "tool_call_id": str(i),
                "tool_name": "Operate",
                "tool_result": {"done": True},
            }
            for i in range(14)
        ]
    )
    messages = await _async_chat_log_messages(
        DummyHass(), GroqModelRegistry(), "openai/gpt-oss-20b", log, "Operate"
    )
    assert messages[0] == {"role": "user", "content": "Operate"}
    assert len(messages) == 16
    assert len(messages[1]["tool_calls"]) == 14


@pytest.mark.asyncio
async def test_registered_service_preserves_defaults_and_cache_opt_out(
    hass, monkeypatch
):
    """Exercise actual HA service schema injection, not only the handler."""
    entry = DummyEntry()
    client = DummyTextClient('{"summary":"ok"}')
    service_data = {
        "unique_id": "text-service",
        "name": "Text",
        "model": "openai/gpt-oss-20b",
        "structured_outputs": True,
        "schema": SCHEMA,
        "schema_name": "configured",
        "strict": True,
        "prompt_caching": False,
    }
    runtime = SimpleNamespace(
        client=client,
        model_registry=GroqModelRegistry(),
        feature_registry=GroqFeatureRegistry(
            [GroqFeature.TEXT_GENERATION, GroqFeature.PROMPT_CACHING]
        ),
        prompt_cache=GroqPromptCache(),
        services_by_type={"text_generation": (service_data,)},
    )
    monkeypatch.setattr(
        services, "_runtime_from_call", AsyncMock(return_value=(entry, runtime))
    )
    key = Mock(wraps=services._cache_key)
    monkeypatch.setattr(services, "_cache_key", key)
    hass.services.async_register(
        "groq",
        "generate_text",
        services._handle_generate_text(hass),
        schema=services.GENERATE_TEXT_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    data = {"service_id": "text-service", "prompt": "Summarize"}
    for _ in range(2):
        await hass.services.async_call(
            "groq", "generate_text", data, blocking=True, return_response=True
        )
    assert len(client.requests) == 2
    assert key.call_count == 0
    assert client.requests[0].strict is True
    assert client.requests[0].schema_name == "configured"
    await hass.services.async_call(
        "groq",
        "generate_text",
        {**data, "strict": False, "schema_name": "explicit"},
        blocking=True,
        return_response=True,
    )
    assert client.requests[-1].strict is False
    assert client.requests[-1].schema_name == "explicit"
    service_data["prompt_caching"] = True
    await hass.services.async_call(
        "groq", "generate_text", data, blocking=True, return_response=True
    )
    cached = await hass.services.async_call(
        "groq", "generate_text", data, blocking=True, return_response=True
    )
    assert len(client.requests) == 4
    assert cached["cached"] is True


@pytest.mark.asyncio
async def test_reasoning_false_override_uses_effective_options(hass, monkeypatch):
    entry = DummyEntry()
    client = SimpleNamespace(
        async_generate_text=AsyncMock(
            return_value=SimpleNamespace(
                text="ok",
                reasoning=None,
                executed_tools=None,
                usage_breakdown=None,
                model="llama-3.1-8b-instant",
                usage={},
            )
        )
    )
    data = {
        "unique_id": "text",
        "model": "llama-3.1-8b-instant",
        "include_reasoning": True,
    }
    runtime = SimpleNamespace(
        client=client,
        model_registry=GroqModelRegistry(),
        feature_registry=GroqFeatureRegistry([GroqFeature.TEXT_GENERATION]),
        services_by_type={"text_generation": (data,)},
    )
    monkeypatch.setattr(
        services, "_runtime_from_call", AsyncMock(return_value=(entry, runtime))
    )
    hass.services.async_register(
        "groq",
        "generate_text",
        services._handle_generate_text(hass),
        schema=services.GENERATE_TEXT_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    await hass.services.async_call(
        "groq",
        "generate_text",
        {"service_id": "text", "prompt": "Hi", "include_reasoning": False},
        blocking=True,
        return_response=True,
    )
    assert client.async_generate_text.call_args.args[0].include_reasoning is None
    with pytest.raises(HomeAssistantError):
        await hass.services.async_call(
            "groq",
            "generate_text",
            {"service_id": "text", "prompt": "Hi"},
            blocking=True,
            return_response=True,
        )


class PromptAPI(llm.API):
    """An API can supply useful context without exposing any callable tools."""

    async def async_get_api_instance(self, llm_context):
        return llm.APIInstance(
            api=self,
            api_prompt="Remember that the room is occupied.",
            llm_context=llm_context,
            tools=[],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, schema, response",
    [
        ("openai/gpt-oss-20b", SCHEMA, '{"summary":"ok"}'),
        ("llama-3.1-8b-instant", SCHEMA, '```json\n{"summary":"ok"}\n```'),
        ("llama-3.1-8b-instant", None, "plain text"),
    ],
)
async def test_real_ai_task_chat_log_keeps_api_prompt_and_final_content(
    hass, monkeypatch, model, schema, response
):
    entry = DummyEntry()
    client = DummyTextClient(response)
    entity = GroqAITaskEntity(
        hass,
        entry,
        {
            "model": model,
            "unique_id": "task",
            "structured_outputs": schema is not None,
            "schema": schema,
            "system_prompt": "Be concise.",
        },
        client,
    )
    entity.platform = SimpleNamespace(domain="groq")
    entity.entity_id = "ai_task.groq"
    monkeypatch.setattr(entity, "async_write_ha_state", Mock())
    task = GenDataTask(
        name="summary",
        instructions="Summarize",
        llm_api=PromptAPI(hass=hass, id="context_only", name="Context only"),
    )
    with async_get_chat_session(hass) as session:
        result = await entity.internal_async_generate_data(session, task)
        with conversation.async_get_chat_log(hass, session) as chat_log:
            assistants = [item for item in chat_log.content if item.role == "assistant"]
            assert len(assistants) == 1
            assert assistants[0].content == response
    request = client.requests[0]
    assert any(
        "room is occupied" in message.get("content", "")
        for message in request.messages
        if message["role"] == "system"
    )
    assert request.system_prompt == "Be concise."
    assert (
        len([message for message in request.messages if message["role"] == "user"]) == 1
    )
    assert result.data == ({"summary": "ok"} if schema else "plain text")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["openai/gpt-oss-20b", "llama-3.1-8b-instant"])
async def test_ai_task_service_schema_rejects_wrong_shape(hass, model):
    entity = GroqAITaskEntity(
        hass,
        DummyEntry(),
        {"model": model, "structured_outputs": True, "schema": SCHEMA},
        GroqApiClient(
            hass,
            api_key="fake",
            session=DummySession(
                DummyResponse(
                    200,
                    {"content-type": "application/json"},
                    {"choices": [{"message": {"content": '{"wrong":true}'}}]},
                )
            ),
        ),
    )
    log = conversation.ChatLog(hass, "test")
    with pytest.raises(HomeAssistantError, match="requested structure"):
        await entity._async_generate_data(
            GenDataTask(name="summary", instructions="Summarize"), log
        )
    assert all(content.role != "assistant" for content in log.content)


@pytest.mark.asyncio
async def test_direct_structured_api_rejects_wrong_shape_before_response(hass):
    client = GroqApiClient(
        hass,
        api_key="fake",
        session=DummySession(
            DummyResponse(
                200,
                {"content-type": "application/json"},
                {"choices": [{"message": {"content": '{"wrong":true}'}}]},
            )
        ),
    )
    with pytest.raises(HomeAssistantError, match="requested structure"):
        await client.async_generate_structured(
            StructuredGenerationRequest(
                prompt="Summarize", model="openai/gpt-oss-20b", schema=SCHEMA
            )
        )


def test_json_schema_does_not_retrieve_remote_references():
    with pytest.raises(HomeAssistantError, match="requested structure"):
        validate_json_schema_data({}, {"$ref": "https://example.invalid/schema.json"})


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image", "audio"])
async def test_service_media_reader_rejects_growth_after_stat(
    tmp_path, monkeypatch, kind
):
    path = tmp_path / ("media.png" if kind == "image" else "media.wav")
    path.write_bytes(b"small")
    hass = DummyHass()
    hass.config = SimpleNamespace(is_allowed_path=lambda _path: True)

    def grew(_path, _limit):
        raise ValueError("grew")

    monkeypatch.setattr(services, "read_bounded_file", grew)
    read = (
        services._image_from_local_path
        if kind == "image"
        else services._audio_from_local_path
    )
    with pytest.raises(HomeAssistantError, match="too large"):
        await read(hass, str(path))


@pytest.mark.asyncio
async def test_streaming_assist_uses_real_completed_chat_content(hass):
    entity = GroqConversationEntity(
        hass,
        DummyEntry(),
        {"model": "llama-3.1-8b-instant", "system_prompt": "Reply plainly."},
        DummyTextClient("unused", ["Hello", " world"]),
    )
    log = conversation.ChatLog(hass, "stream")
    log.async_add_user_content(conversation.UserContent("Hello"))
    result = await entity._async_handle_message(
        conversation.ConversationInput(
            text="Hello",
            context=Context(),
            conversation_id="stream",
            device_id=None,
            satellite_id=None,
            language="en",
            agent_id="conversation.groq",
        ),
        log,
    )
    assert result.response.speech["plain"]["speech"] == "Hello world"
    assert log.content[-1].content == "Hello world"


@pytest.mark.asyncio
async def test_sanitized_provider_json_failure_still_uses_validated_fallback(hass):
    class SequenceSession:
        def __init__(self):
            self.responses = [
                DummyResponse(
                    400,
                    {"content-type": "application/json"},
                    {
                        "error": {
                            "code": "json_validate_failed",
                            "message": "Echoed sensitive prompt",
                        }
                    },
                ),
                DummyResponse(
                    200,
                    {"content-type": "application/json"},
                    {"choices": [{"message": {"content": '{"summary":"ok"}'}}]},
                ),
            ]
            self.calls = []

        def request(self, *args, **kwargs):
            self.calls.append(kwargs)
            return self.responses.pop(0)

    session = SequenceSession()
    client = GroqApiClient(hass, api_key="fake", session=session)
    entity = GroqAITaskEntity(
        hass, DummyEntry(), {"model": "openai/gpt-oss-20b"}, client
    )
    task = GenDataTask(
        name="summary",
        instructions="Summarize",
        structure=vol.Schema({vol.Required("summary"): str}),
    )
    result = await entity._async_generate_data(
        task, conversation.ChatLog(hass, "retry")
    )
    assert result.data == {"summary": "ok"}
    assert len(session.calls) == 2
    assert session.calls[0]["json"]["response_format"]["type"] == "json_schema"
    assert "response_format" not in session.calls[1]["json"]
