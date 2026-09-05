"""Speech metadata, translation, usage, search and streamed control contracts."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.core import Context, SupportsResponse
from homeassistant.exceptions import HomeAssistantError

from custom_components.groq import api, flow_schemas, sensor, services
from custom_components.groq.api import (
    GroqApiClient,
    TextGenerationRequest,
    StructuredGenerationRequest,
)
from custom_components.groq.ai_task import GroqAITaskEntity
from custom_components.groq.citations import extract_citations
from custom_components.groq.conversation import GroqConversationEntity
from custom_components.groq.feature_registry import GroqFeature, GroqFeatureRegistry
from custom_components.groq.model_registry import GroqModelRegistry
from custom_components.groq.streaming import ChatStream
from custom_components.groq.usage import GroqUsage
from .test_foundation import DummyEntry, DummyHass
from .test_transport_architecture import Response, Session

MODEL = "openai/gpt-oss-20b"


def event(delta=None, finish=None, **extra):
    return {"choices": [{"delta": delta or {}, "finish_reason": finish}], **extra}


def stream_client(hass, events):
    body = (
        b"".join(b"data: " + json.dumps(item).encode() + b"\n" for item in events)
        + b"data: [DONE]\n"
    )
    response = Response(body=body)
    return GroqApiClient(hass, api_key="fake", session=Session(response))


@pytest.mark.asyncio
async def test_transcription_metadata_multipart_and_legacy_text():
    metadata = {
        "text": "Hello",
        "duration": 2.0,
        "language": "english",
        "words": [{"word": "Hello", "start": 0, "end": 1}],
        "segments": [{"text": "Hello", "no_speech_prob": 0.01}],
        "private": "omit",
    }
    session = Session(Response(body=json.dumps(metadata).encode()))
    client = GroqApiClient(DummyHass(), api_key="fake", session=session)
    result = await client.async_transcribe_audio_result(
        audio=b"audio",
        filename="a.wav",
        model="whisper-large-v3-turbo",
        language="en-AU",
        prompt="James",
        response_format="verbose_json",
        timestamp_granularities=["word", "segment", "word"],
    )
    assert result == {k: v for k, v in metadata.items() if k != "private"}
    fields = session.calls[0][1]["data"]._fields
    assert [
        (header["name"], value)
        for header, _, value in fields
        if header["name"] != "file"
    ] == [
        ("model", "whisper-large-v3-turbo"),
        ("response_format", "verbose_json"),
        ("language", "en"),
        ("prompt", "James"),
        ("timestamp_granularities[]", "word"),
        ("timestamp_granularities[]", "segment"),
    ]
    assert (
        await client.async_transcribe_audio(
            audio=b"audio", filename="a.wav", model="whisper-large-v3"
        )
        == "Hello"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"response_format": "text"},
        {"timestamp_granularities": ["word"]},
        {"response_format": "verbose_json", "timestamp_granularities": ["invalid"]},
        {"translate": True, "model": "whisper-large-v3-turbo"},
        {"translate": True, "response_format": "verbose_json"},
    ],
)
async def test_invalid_audio_options_fail_before_network(kwargs):
    session = Session(Response())
    client = GroqApiClient(DummyHass(), api_key="fake", session=session)
    with pytest.raises(HomeAssistantError):
        await client.async_transcribe_audio_result(
            **{
                "audio": b"a",
                "filename": "a.wav",
                "model": "whisper-large-v3",
                **kwargs,
            }
        )
    assert session.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "translate, extra",
    [
        (True, {}),
        (
            False,
            {"response_format": "verbose_json", "timestamp_granularities": ["segment"]},
        ),
        (False, {}),
    ],
)
async def test_registered_audio_actions(hass, monkeypatch, translate, extra):
    client = GroqApiClient(
        hass,
        api_key="fake",
        session=Session(Response(body=b'{"text":"Hello","segments":[]}')),
    )
    data = {
        "unique_id": "speech",
        "model": "whisper-large-v3-turbo",
        "language": "fr",
        "protect_free_tier": False,
    }
    runtime = SimpleNamespace(
        client=client,
        model_registry=GroqModelRegistry(),
        feature_registry=GroqFeatureRegistry([GroqFeature.SPEECH_TO_TEXT]),
        services_by_type={"speech_to_text": (data,)},
    )
    monkeypatch.setattr(
        services, "_runtime_from_call", AsyncMock(return_value=(DummyEntry(), runtime))
    )
    source = AsyncMock(return_value=(b"audio", "a.wav"))
    monkeypatch.setattr(services, "_audio_from_call", source)
    action = "translate_audio" if translate else "transcribe_audio"
    hass.services.async_register(
        "groq",
        action,
        services._handle_transcribe_audio(hass, translate=translate),
        schema=(
            services.TRANSLATE_AUDIO_SCHEMA
            if translate
            else services.TRANSCRIBE_AUDIO_SCHEMA
        ),
        supports_response=SupportsResponse.ONLY,
    )
    result = await hass.services.async_call(
        "groq",
        action,
        {"service_id": "speech", **extra},
        blocking=True,
        return_response=True,
    )
    assert result["text"] == "Hello"
    assert result["model"] == (
        "whisper-large-v3" if translate else "whisper-large-v3-turbo"
    )
    assert result["language"] == ("en" if translate else "fr")
    args, kwargs = client._session.calls[0]
    assert args[1].endswith(
        "/audio/translations" if translate else "/audio/transcriptions"
    )
    names = [header["name"] for header, _, _ in kwargs["data"]._fields]
    assert ("language" in names) is not translate
    source.assert_awaited_once()
    if translate:
        with pytest.raises(HomeAssistantError):
            await hass.services.async_call(
                "groq",
                action,
                {"service_id": "speech", "model": "whisper-large-v3-turbo"},
                blocking=True,
                return_response=True,
            )
    else:
        with pytest.raises(HomeAssistantError):
            await hass.services.async_call(
                "groq",
                action,
                {"service_id": "speech", "timestamp_granularities": ["word"]},
                blocking=True,
                return_response=True,
            )
    assert len(client._session.calls) == 1


def test_usage_replaces_missing_metrics_and_unsubscribes():
    usage = GroqUsage()
    update = Mock()
    unsubscribe = usage.subscribe(update)
    usage.record(None, {})
    assert not usage.values
    usage.record(
        "text",
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "total_time": 0.2,
            "prompt_tokens_details": {"cached_tokens": 75},
            "private": "prompt",
        },
    )
    assert usage.values["text"]["cache_hit_rate"] == 75
    assert "private" not in usage.values["text"]
    usage.record("other", {})
    usage.record(
        "text",
        {
            "total_tokens": float("nan"),
            "total_time": -1,
            "completion_tokens": True,
            "prompt_tokens": "100",
        },
    )
    assert usage.values["text"] == {
        "requests": 2,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "response_time": None,
        "cached_tokens": None,
        "cache_hit_rate": None,
    }
    unsubscribe()
    usage.record(
        "text", {"prompt_tokens": 2, "prompt_tokens_details": {"cached_tokens": 3}}
    )
    assert usage.values["text"]["cache_hit_rate"] is None
    assert update.call_count == 3
    usage.clear()
    assert not usage.values


@pytest.mark.asyncio
async def test_sensors_subentry_setup_and_push_lifecycle(hass, monkeypatch):
    usage = GroqUsage()
    service = {
        "unique_id": "text",
        "model": MODEL,
        "name": "Assistant",
        "subentry_id": "sub",
    }
    runtime = SimpleNamespace(
        client=SimpleNamespace(usage=usage),
        services_by_type={"text_generation": (service,)},
    )
    monkeypatch.setattr(sensor, "async_get_runtime", AsyncMock(return_value=runtime))
    add = Mock()
    await sensor.async_setup_entry(hass, DummyEntry(), add)
    entities = add.call_args.args[0]
    assert len(entities) == 7
    callbacks = []
    for entity in entities:
        entity.hass = hass
        assert entity.native_value is None
        assert not entity.entity_registry_enabled_default
        assert entity.device_info["identifiers"] == {("groq", "text")}
        monkeypatch.setattr(entity, "async_on_remove", callbacks.append)
        monkeypatch.setattr(entity, "async_write_ha_state", Mock())
        await entity.async_added_to_hass()
    usage.record("text", {"total_tokens": 42})
    assert (
        next(
            e for e in entities if e.entity_description.key == "total_tokens"
        ).native_value
        == 42
    )
    for entity, unsubscribe in zip(entities, callbacks):
        entity.async_write_ha_state.assert_called_once()
        unsubscribe()
    usage.record("text", {})
    assert all(e.async_write_ha_state.call_count == 1 for e in entities)


def test_citations_are_safe_unique_and_bounded():
    message = {
        "annotations": [
            {
                "type": "url_citation",
                "url_citation": {"url": "https://example.com/a", "title": "Source"},
            },
            None,
            {"url_citation": []},
            {"url": "javascript:alert(1)"},
            {"url": "https://user:pass@example.com"},
            {"url": "https://["},
            {"url": "https://exa mple.com"},
            {"url": 1},
        ],
        "citations": ["https://example.com/a", "https://example.org"],
        "executed_tools": [
            None,
            {
                "search_results": {
                    "results": [{"url": "https://example.net", "title": "x" * 600}]
                }
            },
            {"search_results": "bad"},
            {
                "browser_results": [
                    {"url": "https://example.com/a", "title": "Duplicate"},
                    {"url": "https://example.edu", "title": "Browsed source"},
                    {"url": "javascript:alert(1)"},
                ]
            },
            {"browser_results": None},
            {"browser_results": "bad"},
        ],
    }
    result = extract_citations(message)
    assert [item["url"] for item in result] == [
        "https://example.com/a",
        "https://example.org",
        "https://example.net",
        "https://example.edu",
    ]
    assert len(result[2]["title"]) == 512
    assert result[-1]["title"] == "Browsed source"
    assert (
        len(
            extract_citations(
                {"citations": [f"https://example.com/{i}" for i in range(100)]}
            )
        )
        == 50
    )


@pytest.mark.asyncio
async def test_stream_preserves_usage_reasoning_sources_and_interleaved_tools(hass):
    events = [
        event(
            {
                "content": "Checking",
                "reasoning": "Reason",
                "tool_calls": [
                    {
                        "index": 1,
                        "id": "second",
                        "function": {"name": "Operate", "arguments": '{"entity":'},
                    },
                    {
                        "index": 0,
                        "id": "first",
                        "function": {
                            "name": "Operate",
                            "arguments": '{"entity":"light.a"}',
                        },
                    },
                ],
            }
        ),
        event(
            {
                "tool_calls": [{"index": 1, "function": {"arguments": '"light.b"}'}}],
                "annotations": [
                    {"url_citation": {"url": "https://example.com", "title": "Source"}}
                ],
            },
            "tool_calls",
        ),
        {
            "choices": [],
            "model": "openai/gpt-oss-120b",
            "x_groq": {
                "usage": {"total_tokens": 5},
                "usage_breakdown": {
                    "models": [{"model": MODEL, "usage": {"total_tokens": 5}}]
                },
            },
        },
    ]
    client = stream_client(hass, events)
    chunks = [
        chunk
        async for chunk in client.async_stream_chat(
            TextGenerationRequest(prompt="p", model=MODEL, service_id="text")
        )
    ]
    assert chunks[0] == "Checking"
    result = chunks[-1]
    assert result.model == "openai/gpt-oss-120b"
    assert result.reasoning == "Reason"
    assert [c["id"] for c in result.tool_calls] == ["first", "second"]
    assert json.loads(result.tool_calls[1]["function"]["arguments"]) == {
        "entity": "light.b"
    }
    assert result.citations == [{"url": "https://example.com", "title": "Source"}]
    assert result.usage == {"total_tokens": 5}
    assert result.usage_breakdown == {
        "models": [{"model": MODEL, "usage": {"total_tokens": 5}}]
    }
    assert client.usage.values["text"]["requests"] == 1


@pytest.mark.parametrize(
    "delta",
    [
        None,
        {"index": -1},
        {"index": True},
        {"index": 64},
        {"index": 0, "type": "remote"},
        {"index": 0, "id": 1},
        {"index": 0, "function": []},
        {"index": 0, "function": {"arguments": {}}},
    ],
)
def test_stream_rejects_malformed_tool_deltas(delta):
    stream = ChatStream(MODEL)
    with pytest.raises(HomeAssistantError):
        stream.add(event({"tool_calls": [delta]}))


@pytest.mark.parametrize(
    "arguments, call_id, name, finish",
    [
        ("{", "call", "Operate", "tool_calls"),
        ("[]", "call", "Operate", "tool_calls"),
        ("{}", "", "Operate", "tool_calls"),
        ("{}", "call", "", "tool_calls"),
        ("{}", "call", "Operate", "length"),
        ("{}", "call", "Operate", None),
    ],
)
def test_stream_rejects_incomplete_operations(arguments, call_id, name, finish):
    stream = ChatStream(MODEL)
    stream.add(
        event(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": call_id,
                        "function": {"name": name, "arguments": arguments},
                    }
                ]
            },
            finish,
        )
    )
    with pytest.raises(HomeAssistantError):
        stream.result()


def test_stream_rejects_reused_ids_and_post_completion_content():
    stream = ChatStream(MODEL)
    call = {
        "index": 0,
        "id": "call",
        "function": {"name": "Operate", "arguments": "{}"},
    }
    stream.add(event({"tool_calls": [call]}))
    with pytest.raises(HomeAssistantError):
        stream.add(event({"tool_calls": [{"index": 0, "id": "changed"}]}))
    stream.add(event({"tool_calls": [{**call, "index": 1}]}, "tool_calls"))
    with pytest.raises(HomeAssistantError):
        stream.result()
    with pytest.raises(HomeAssistantError):
        stream.add(event({"content": "after"}))


@pytest.mark.parametrize("browser_search", [False, True])
@pytest.mark.parametrize("body", [["invalid"], "invalid", 1, True])
def test_request_body_selector_values_return_validation_errors(browser_search, body):
    schema = flow_schemas.text_generation_advanced_schema({"model": MODEL})
    data = schema({"browser_search": browser_search, "request_body_options": body})
    assert flow_schemas.validate_text_generation_input({**data, "model": MODEL}) == {
        "request_body_options": "invalid_request_body_options"
    }


def test_browser_search_payload_and_capability_validation():
    tools = [{"type": "function", "function": {"name": "Operate"}}]
    request = TextGenerationRequest(
        prompt="p", model=MODEL, browser_search=True, tools=tools
    )
    payload = api.build_text_generation_payload(request)
    assert payload["tools"] == [*tools, {"type": "browser_search"}]
    assert len(tools) == 1
    assert (
        "browser_search"
        in flow_schemas.text_generation_advanced_schema({"model": MODEL}).schema
    )
    assert (
        "browser_search"
        not in flow_schemas.text_generation_advanced_schema(
            {"model": "llama-3.1-8b-instant"}
        ).schema
    )
    assert "browser search" in flow_schemas.text_generation_model_capability_summary(
        MODEL
    )
    assert not flow_schemas.validate_text_generation_input(
        {"model": MODEL, "browser_search": True}
    )
    assert (
        flow_schemas.validate_text_generation_input(
            {"model": MODEL, "browser_search": True, "structured_outputs": True}
        )["browser_search"]
        == "browser_search_structured_output"
    )
    assert (
        flow_schemas.validate_text_generation_input(
            {"model": "llama-3.1-8b-instant", "browser_search": True}
        )["browser_search"]
        == "unsupported_browser_search_model"
    )
    assert "browser_search" not in flow_schemas.sanitize_text_generation_service_data(
        {"model": "llama-3.1-8b-instant", "browser_search": True}
    )


@pytest.mark.parametrize(
    "generation_request",
    [
        TextGenerationRequest(
            prompt="p", model="llama-3.1-8b-instant", browser_search=True
        ),
        StructuredGenerationRequest(prompt="p", model=MODEL, browser_search=True),
        TextGenerationRequest(
            prompt="p",
            model=MODEL,
            browser_search=True,
            extra_body={"response_format": {"type": "json_object"}},
        ),
    ],
)
def test_invalid_search_never_builds_a_payload(generation_request):
    with pytest.raises(HomeAssistantError):
        api.build_text_generation_payload(generation_request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad", [None, "malformed", "unexposed", "truncated", "cancelled"]
)
async def test_real_chat_log_streams_and_executes_only_completed_exposed_tools(
    hass, monkeypatch, bad
):
    called = []

    async def operate(tool):
        called.append(tool.tool_args)
        return {"success": True}

    log = conversation.ChatLog(hass, "stream-tools")
    log.async_add_user_content(conversation.UserContent("Turn on the light"))
    log.llm_api = SimpleNamespace(
        tools=[
            SimpleNamespace(
                name="Operate",
                description="Operate a light",
                parameters=vol.Schema({vol.Required("entity_id"): str}),
            )
        ],
        custom_serializer=None,
        async_call_tool=operate,
    )
    monkeypatch.setattr(log, "async_provide_llm_data", AsyncMock())
    first = [
        event(
            {
                "content": "Checking. ",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "one",
                        "function": {
                            "name": "Unexposed" if bad == "unexposed" else "Operate",
                            "arguments": '{"entity_id":',
                        },
                    }
                ],
            }
        ),
        event(
            {
                "tool_calls": [
                    {"index": 0, "id": None, "type": None, "function": None},
                    {
                        "index": 0,
                        "function": {"name": None, "arguments": None},
                    },
                ]
            }
        ),
        event(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": None,
                        "function": {
                            "name": None,
                            "arguments": (
                                "bad" if bad == "malformed" else '"light.kitchen"}'
                            ),
                        },
                    }
                ]
            },
            "length" if bad == "truncated" else "tool_calls",
        ),
    ]
    second = [
        event(
            {
                "content": "Done",
                "annotations": [
                    {"url_citation": {"url": "https://example.com", "title": "Source"}}
                ],
            },
            "stop",
            usage={"total_tokens": 8},
        )
    ]
    clients = [stream_client(hass, first), stream_client(hass, second)]
    client = clients[0]
    first_response = client._session.response
    second_response = clients[1]._session.response

    class SequenceSession:
        def __init__(self):
            self.responses = iter([first_response, second_response])
            self.calls = []

        def request(self, *args, **kwargs):
            self.calls.append(kwargs["json"])
            return next(self.responses)

    client._session = SequenceSession()
    entity = GroqConversationEntity(
        hass,
        DummyEntry(),
        {"model": MODEL, "unique_id": "text", "browser_search": True},
        client,
    )
    user = conversation.ConversationInput(
        text="Turn on the light",
        context=Context(),
        conversation_id="stream-tools",
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id="conversation.groq",
    )
    if bad == "cancelled":

        async def cancelled(_size):
            yield b'data: {"choices":[{"delta":{"content":"Checking"}}]}\n'
            raise asyncio.CancelledError

        monkeypatch.setattr(first_response, "iter_chunked", cancelled)
    if bad:
        with pytest.raises(
            asyncio.CancelledError if bad == "cancelled" else HomeAssistantError
        ):
            await entity._async_handle_message(user, log)
        assert not called
        assert first_response.closed
        return
    result = await entity._async_handle_message(user, log)
    assert called == [{"entity_id": "light.kitchen"}]
    assert result.response.speech["plain"]["speech"] == "Done"
    assert any(item.role == "tool_result" for item in log.content)
    assert log.content[-1].native["citations"][0]["url"] == "https://example.com"
    assert client.usage.values["text"]["requests"] == 2
    assert client.usage.values["text"]["total_tokens"] == 8
    assert all(payload["stream"] for payload in client._session.calls)
    assert client._session.calls[0]["tools"][-1] == {"type": "browser_search"}
    assert any(item["role"] == "tool" for item in client._session.calls[1]["messages"])


@pytest.mark.asyncio
async def test_search_actions_inherit_override_validate_and_do_not_cache(
    hass, monkeypatch
):
    from custom_components.groq.prompt_cache import GroqPromptCache

    payload = {
        "model": MODEL,
        "usage": {"total_tokens": 4},
        "choices": [
            {
                "message": {
                    "content": "Answer",
                    "executed_tools": [
                        {
                            "type": "browser",
                            "index": 0,
                            "arguments": "{}",
                            "browser_results": [
                                {"url": "https://example.com", "title": "Source"}
                            ],
                        }
                    ],
                }
            }
        ],
    }
    client = GroqApiClient(
        hass,
        api_key="fake",
        session=Session(Response(body=json.dumps(payload).encode())),
    )
    data = {
        "unique_id": "text",
        "model": MODEL,
        "browser_search": True,
        "prompt_caching": True,
    }
    runtime = SimpleNamespace(
        client=client,
        model_registry=GroqModelRegistry(),
        feature_registry=GroqFeatureRegistry(
            [GroqFeature.TEXT_GENERATION, GroqFeature.PROMPT_CACHING]
        ),
        services_by_type={"text_generation": (data,)},
        prompt_cache=GroqPromptCache(),
    )
    monkeypatch.setattr(
        services, "_runtime_from_call", AsyncMock(return_value=(DummyEntry(), runtime))
    )
    hass.services.async_register(
        "groq",
        "generate_text",
        services._handle_generate_text(hass),
        schema=services.GENERATE_TEXT_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )

    async def generate(**extra):
        return await hass.services.async_call(
            "groq",
            "generate_text",
            {"service_id": "text", "prompt": "News", **extra},
            blocking=True,
            return_response=True,
        )

    for _ in range(2):
        result = await generate()
        assert result["citations"] == [
            {"url": "https://example.com", "title": "Source"}
        ]
        assert not result["cached"]
    assert len(client._session.calls) == 2
    await generate(browser_search=False)
    result = await generate(browser_search=False)
    assert result["cached"]
    assert len(client._session.calls) == 3
    assert client.usage.values["text"]["requests"] == 3
    for extra in [
        {"schema": {"type": "object"}},
        {"model": "llama-3.1-8b-instant"},
        {"request_body_options": {"response_format": {"type": "json_schema"}}},
    ]:
        with pytest.raises(HomeAssistantError):
            await generate(**extra)
    assert len(client._session.calls) == 3


@pytest.mark.asyncio
async def test_structured_ai_task_rejects_browser_search_before_tools(hass):
    from homeassistant.components.ai_task import GenDataTask

    client = SimpleNamespace(
        async_generate_text=AsyncMock(), async_generate_structured=AsyncMock()
    )
    entity = GroqAITaskEntity(
        hass, DummyEntry(), {"model": MODEL, "browser_search": True}, client
    )
    with pytest.raises(HomeAssistantError, match="Browser search"):
        await entity._async_generate_data(
            GenDataTask(
                name="data",
                instructions="Generate",
                structure=vol.Schema({vol.Required("value"): str}),
            ),
            conversation.ChatLog(hass, "task"),
        )
    client.async_generate_text.assert_not_called()
    client.async_generate_structured.assert_not_called()
