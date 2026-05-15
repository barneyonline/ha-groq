from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import voluptuous as vol
from homeassistant.components import stt
from homeassistant.components.ai_task import GenDataTask
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import Platform
from homeassistant.exceptions import ServiceValidationError

from custom_components.groq.api import (
    GroqApiClient,
    StructuredGenerationRequest,
    TextGenerationRequest,
    VisionRequest,
    build_structured_generation_payload,
    build_text_generation_payload,
    build_vision_payload,
    normalize_base_url,
)
from custom_components.groq.ai_task import GroqAITaskEntity
from custom_components.groq.conversation import (
    GroqConversationEntity,
    _chat_log_messages,
)
from custom_components.groq.const import (
    COMPOUND_MODELS,
    CONF_INCLUDE_REASONING,
    CONF_MODEL,
    CONF_PROMPT_CACHING,
    CONF_REASONING_EFFORT,
    CONF_REASONING_FORMAT,
    CONF_STRUCTURED_OUTPUTS,
    DEFAULT_SYSTEM_PROMPT,
    TEXT_MODELS,
)
from custom_components.groq.feature_registry import (
    CONF_ENABLED_FEATURES,
    GroqFeature,
    GroqFeatureRegistry,
    enabled_features_from_options,
)
from custom_components.groq.flow_schemas import validate_text_generation_input
from custom_components.groq.model_registry import (
    GroqCapability,
    GroqModelRegistry,
    infer_capabilities,
)
from custom_components.groq.prompt_cache import GroqPromptCache
from custom_components.groq.runtime import build_runtime
from custom_components.groq.services import (
    ATTR_CONFIG_ENTRY_ID,
    ATTR_SCHEMA,
    ATTR_SERVICE_ID,
    _handle_analyze_image,
    _handle_generate_text,
    _handle_list_models,
)
from custom_components.groq.stt import GroqSTTEntity


class DummyResponse:
    def __init__(self, status: int, headers: dict[str, str], payload: dict):
        self.status = status
        self.headers = headers
        self._payload = payload

    async def read(self):
        import json

        return json.dumps(self._payload).encode()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummySession:
    def __init__(self, response: DummyResponse):
        self.response = response
        self.calls = []

    def request(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.response


class DummyConfigEntries:
    def __init__(self, entries):
        self._entries = {entry.entry_id: entry for entry in entries}

    def async_entries(self, domain):
        return list(self._entries.values())

    def async_get_entry(self, entry_id):
        return self._entries.get(entry_id)


class DummyHass:
    def __init__(self, entries=()):
        self.data = {}
        self.config_entries = DummyConfigEntries(entries)


class DummyTextClient:
    def __init__(self, text: str, stream_chunks: list[str] | None = None):
        self.text = text
        self.stream_chunks = stream_chunks or []
        self.requests = []

    async def async_generate_text(self, request):
        self.requests.append(request)
        return SimpleNamespace(text=self.text)

    async def async_stream_text(self, request):
        self.requests.append(request)
        for chunk in self.stream_chunks:
            yield chunk

    async def async_generate_structured(self, request):
        self.requests.append(request)
        return {
            "text": self.text,
            "data": json.loads(self.text),
            "model": request.model,
            "usage": {},
            "cached": False,
        }


class DummySTTClient:
    def __init__(self, text: str):
        self.text = text
        self.requests = []

    async def async_transcribe_audio(self, **kwargs):
        self.requests.append(kwargs)
        return self.text


class DummyChatLog:
    conversation_id = "conversation-id"

    def __init__(self):
        self.assistant_content = []
        self.stream_deltas = []

    def async_add_assistant_content_without_tools(self, content):
        self.assistant_content.append(content)

    async def async_add_delta_content_stream(self, agent_id, stream):
        content = ""
        async for delta in stream:
            self.stream_deltas.append(delta)
            content += delta.get("content", "")
        completed = SimpleNamespace(agent_id=agent_id, content=content)
        self.assistant_content.append(completed)
        yield completed


class DummyEntry:
    def __init__(self, entry_id="entry-id", state=ConfigEntryState.LOADED):
        self.entry_id = entry_id
        self.state = state
        self.data = {
            "api_key": "api-key",
            "url": "https://api.groq.com/openai/v1/audio/speech",
        }
        self.options = {}


def service_call(data):
    return SimpleNamespace(data=data)


def test_payload_builders_use_openai_compatible_shapes():
    payload = build_text_generation_payload(
        TextGenerationRequest(
            prompt="Hello",
            system_prompt="Be concise",
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=40,
            top_p=0.9,
            stop=["END"],
            seed=7,
            service_tier="flex",
            reasoning_effort="low",
            reasoning_format="parsed",
            extra_body={
                "citation_options": "disabled",
                "compound_custom": {"tools": {"enabled_tools": ["web_search"]}},
                "disable_tool_validation": True,
                "documents": [{"text": "static context"}],
                "exclude_domains": ["old.example"],
                "frequency_penalty": 0,
                "function_call": "auto",
                "functions": [{"name": "legacy_function"}],
                "include_domains": ["example.com"],
                "logit_bias": {"1": -100},
                "logprobs": True,
                "max_tokens": 40,
                "metadata": {"source": "home_assistant"},
                "n": 1,
                "parallel_tool_calls": False,
                "presence_penalty": 0,
                "response_format": {"type": "json_object"},
                "search_settings": {"include_domains": ["example.com"]},
                "store": False,
                "stream_options": {"include_usage": True},
                "tool_choice": "auto",
                "tools": [{"type": "function", "function": {"name": "tool"}}],
                "top_logprobs": 2,
                "user": "home-assistant",
                "messages": [{"role": "user", "content": "override"}],
                "model": "override-model",
                "stream": True,
            },
        )
    )

    assert payload == {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hello"},
        ],
        "temperature": 0.2,
        "max_completion_tokens": 40,
        "top_p": 0.9,
        "stop": ["END"],
        "seed": 7,
        "service_tier": "flex",
        "reasoning_effort": "low",
        "reasoning_format": "parsed",
        "citation_options": "disabled",
        "compound_custom": {"tools": {"enabled_tools": ["web_search"]}},
        "disable_tool_validation": True,
        "documents": [{"text": "static context"}],
        "exclude_domains": ["old.example"],
        "frequency_penalty": 0,
        "function_call": "auto",
        "functions": [{"name": "legacy_function"}],
        "include_domains": ["example.com"],
        "logit_bias": {"1": -100},
        "logprobs": True,
        "max_tokens": 40,
        "metadata": {"source": "home_assistant"},
        "n": 1,
        "parallel_tool_calls": False,
        "presence_penalty": 0,
        "response_format": {"type": "json_object"},
        "search_settings": {"include_domains": ["example.com"]},
        "store": False,
        "stream_options": {"include_usage": True},
        "tool_choice": "auto",
        "tools": [{"type": "function", "function": {"name": "tool"}}],
        "top_logprobs": 2,
        "user": "home-assistant",
    }

    structured = build_structured_generation_payload(
        StructuredGenerationRequest(
            prompt="Return a status",
            model="openai/gpt-oss-20b",
            schema={"type": "object"},
            strict=True,
        )
    )

    assert structured["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": {"type": "object"},
            "strict": True,
        },
    }

    vision = build_vision_payload(
        VisionRequest(
            prompt="Describe",
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            image_url="data:image/png;base64,YWJj",
        )
    )

    assert vision["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,YWJj"},
                },
            ],
        }
    ]


def test_normalize_base_url_strips_legacy_speech_endpoint():
    assert (
        normalize_base_url("https://api.groq.com/openai/v1/audio/speech")
        == "https://api.groq.com/openai/v1"
    )


@pytest.mark.asyncio
async def test_api_client_generates_text_and_sends_auth_header():
    session = DummySession(
        DummyResponse(
            200,
            {"content-type": "application/json"},
            {
                "model": "llama-3.1-8b-instant",
                "choices": [{"message": {"content": "done"}}],
                "usage": {"total_tokens": 3},
            },
        )
    )
    client = GroqApiClient(
        DummyHass(),
        api_key="api-key",
        base_url="https://api.groq.com/openai/v1",
        session=session,
    )

    result = await client.async_generate_text(
        TextGenerationRequest(
            prompt="Go",
            model="llama-3.1-8b-instant",
            api_key="request-api-key",
        )
    )

    assert result.text == "done"
    assert result.usage == {"total_tokens": 3}
    call = session.calls[0]
    assert call["args"][:2] == (
        "POST",
        "https://api.groq.com/openai/v1/chat/completions",
    )
    assert call["kwargs"]["headers"]["Authorization"] == "Bearer request-api-key"


@pytest.mark.asyncio
async def test_api_client_extracts_compound_response_metadata():
    session = DummySession(
        DummyResponse(
            200,
            {"content-type": "application/json"},
            {
                "model": "groq/compound",
                "choices": [
                    {
                        "message": {
                            "content": "Compound answer",
                            "executed_tools": [
                                {
                                    "type": "search",
                                    "name": "web_search",
                                }
                            ],
                        }
                    }
                ],
                "usage": {"total_tokens": 10},
                "usage_breakdown": {
                    "models": [
                        {
                            "model": "openai/gpt-oss-120b",
                            "usage": {"total_tokens": 8},
                        }
                    ]
                },
            },
        )
    )
    client = GroqApiClient(
        DummyHass(),
        api_key="api-key",
        base_url="https://api.groq.com/openai/v1",
        session=session,
    )

    result = await client.async_generate_text(
        TextGenerationRequest(prompt="Search", model="groq/compound")
    )

    assert result.text == "Compound answer"
    assert result.executed_tools == [{"type": "search", "name": "web_search"}]
    assert result.usage_breakdown == {
        "models": [
            {
                "model": "openai/gpt-oss-120b",
                "usage": {"total_tokens": 8},
            }
        ]
    }


@pytest.mark.asyncio
async def test_api_client_lists_models():
    session = DummySession(
        DummyResponse(
            200,
            {"content-type": "application/json"},
            {
                "data": [
                    {
                        "id": "llama-3.1-8b-instant",
                        "owned_by": "Meta",
                        "active": True,
                    }
                ]
            },
        )
    )
    client = GroqApiClient(DummyHass(), api_key=None, session=session)

    models = await client.async_list_models()

    assert models[0].model_id == "llama-3.1-8b-instant"
    assert GroqCapability.TEXT_GENERATION in models[0].capabilities


def test_feature_registry_supports_enabled_options():
    enabled = enabled_features_from_options(
        {
            CONF_ENABLED_FEATURES: [
                "text_generation",
                "image_recognition",
                "prompt_caching",
                "unknown",
            ]
        }
    )
    registry = GroqFeatureRegistry(enabled)

    assert registry.is_enabled(GroqFeature.TEXT_GENERATION)
    assert registry.is_enabled(GroqFeature.VISION)
    assert registry.enabled_services() == {
        "analyze_image",
        "extract_text_from_image",
        "generate_structured",
        "generate_text",
        "clear_cache",
    }
    assert set(registry.enabled_platforms()) == {
        Platform.AI_TASK,
        Platform.CONVERSATION,
    }


def test_feature_registry_defaults_legacy_entries_to_tts_only():
    registry = GroqFeatureRegistry(enabled_features_from_options({}))

    assert registry.is_enabled(GroqFeature.TEXT_TO_SPEECH)
    assert not registry.is_enabled(GroqFeature.TEXT_GENERATION)


def test_runtime_enables_platforms_from_service_subentries():
    entry = DummyEntry()
    entry.subentries = {
        "text-service": SimpleNamespace(
            data={
                "service_type": "text_generation",
                "model": "openai/gpt-oss-20b",
                "structured_outputs": True,
                "prompt_caching": True,
                "reasoning_effort": "low",
            }
        )
    }

    runtime = build_runtime(DummyHass(), entry)

    assert runtime.feature_registry.is_enabled(GroqFeature.TEXT_GENERATION)
    assert runtime.feature_registry.is_enabled(GroqFeature.STRUCTURED_OUTPUTS)
    assert runtime.feature_registry.is_enabled(GroqFeature.PROMPT_CACHING)
    assert runtime.feature_registry.is_enabled(GroqFeature.REASONING)
    assert Platform.CONVERSATION in runtime.feature_registry.enabled_platforms()
    assert Platform.AI_TASK in runtime.feature_registry.enabled_platforms()


def test_runtime_account_only_entry_has_no_enabled_platforms():
    entry = DummyEntry()
    entry.data = {"api_key": "api-key", "name": "Groq"}

    runtime = build_runtime(DummyHass(), entry)

    assert runtime.feature_registry.enabled_features == frozenset()
    assert runtime.feature_registry.enabled_platforms() == []


def test_runtime_enables_stt_platform_from_speech_service_subentry():
    entry = DummyEntry()
    entry.subentries = {
        "stt-service": SimpleNamespace(
            data={
                "service_type": "speech_to_text",
                "model": "whisper-large-v3",
            }
        )
    }

    runtime = build_runtime(DummyHass(), entry)

    assert runtime.feature_registry.is_enabled(GroqFeature.SPEECH_TO_TEXT)
    assert Platform.STT in runtime.feature_registry.enabled_platforms()
    assert "transcribe_audio" in runtime.feature_registry.enabled_services()


def test_runtime_enables_image_actions_from_image_service_subentry():
    entry = DummyEntry()
    entry.subentries = {
        "image-service": SimpleNamespace(
            data={
                "service_type": "image_recognition",
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            }
        )
    }

    runtime = build_runtime(DummyHass(), entry)

    assert runtime.feature_registry.is_enabled(GroqFeature.VISION)
    assert "analyze_image" in runtime.feature_registry.enabled_services()
    assert "extract_text_from_image" in runtime.feature_registry.enabled_services()


def test_runtime_ignores_prompt_caching_for_unsupported_text_models():
    entry = DummyEntry()
    entry.subentries = {
        "text-service": SimpleNamespace(
            data={
                "service_type": "text_generation",
                "model": "llama-3.1-8b-instant",
                "prompt_caching": True,
            }
        )
    }

    runtime = build_runtime(DummyHass(), entry)

    assert runtime.feature_registry.is_enabled(GroqFeature.TEXT_GENERATION)
    assert not runtime.feature_registry.is_enabled(GroqFeature.PROMPT_CACHING)


def test_model_registry_infers_capabilities():
    assert set(COMPOUND_MODELS) <= set(TEXT_MODELS)
    assert infer_capabilities("whisper-large-v3") == frozenset(
        {GroqCapability.SPEECH_TO_TEXT}
    )
    assert GroqCapability.VISION in infer_capabilities(
        "meta-llama/llama-4-scout-17b-16e-instruct"
    )
    assert GroqModelRegistry().supports(
        "llama-3.1-8b-instant", GroqFeature.TEXT_GENERATION
    )
    assert GroqModelRegistry().supports("groq/compound", GroqFeature.TEXT_GENERATION)
    assert GroqCapability.COMPOUND in infer_capabilities("groq/compound-mini")
    compound = GroqModelRegistry().get("groq/compound")
    assert compound is not None
    assert compound.context_window == 131072
    assert compound.max_completion_tokens == 8192
    assert GroqModelRegistry().supports(
        "openai/gpt-oss-20b", GroqFeature.PROMPT_CACHING
    )
    assert GroqModelRegistry().supports("openai/gpt-oss-20b", GroqFeature.REASONING)
    assert not GroqModelRegistry().supports(
        "llama-3.1-8b-instant", GroqFeature.STRUCTURED_OUTPUTS
    )
    assert not GroqModelRegistry().supports(
        "custom/text-model", GroqFeature.STRUCTURED_OUTPUTS
    )
    assert GroqModelRegistry().supports("qwen/qwen3-32b", GroqFeature.REASONING)
    assert not GroqModelRegistry().supports(
        "llama-3.1-8b-instant", GroqFeature.REASONING
    )


def test_text_generation_config_flow_rejects_reasoning_for_unsupported_models():
    errors = validate_text_generation_input(
        {
            CONF_MODEL: "llama-3.1-8b-instant",
            CONF_REASONING_EFFORT: "low",
        }
    )

    assert errors == {CONF_MODEL: "unsupported_reasoning_model"}


def test_text_generation_config_flow_accepts_reasoning_models():
    errors = validate_text_generation_input(
        {
            CONF_MODEL: "openai/gpt-oss-20b",
            CONF_REASONING_EFFORT: "medium",
            CONF_REASONING_FORMAT: "parsed",
            CONF_INCLUDE_REASONING: False,
        }
    )

    assert errors == {}


def test_text_generation_config_flow_rejects_prompt_caching_for_unsupported_models():
    errors = validate_text_generation_input(
        {
            CONF_MODEL: "llama-3.1-8b-instant",
            CONF_PROMPT_CACHING: True,
        }
    )

    assert errors == {CONF_MODEL: "unsupported_prompt_caching_model"}


def test_text_generation_config_flow_accepts_prompt_caching_models():
    errors = validate_text_generation_input(
        {
            CONF_MODEL: "openai/gpt-oss-120b",
            CONF_PROMPT_CACHING: True,
        }
    )

    assert errors == {}


def test_text_generation_config_flow_rejects_structured_outputs_for_unsupported_models():
    errors = validate_text_generation_input(
        {
            CONF_MODEL: "groq/compound-mini",
            CONF_STRUCTURED_OUTPUTS: True,
        }
    )

    assert errors == {CONF_MODEL: "unsupported_structured_outputs_model"}


def test_text_generation_config_flow_accepts_structured_output_models():
    errors = validate_text_generation_input(
        {
            CONF_MODEL: "openai/gpt-oss-20b",
            CONF_STRUCTURED_OUTPUTS: True,
        }
    )

    assert errors == {}


def test_prompt_cache_evicts_lru_and_clears():
    cache = GroqPromptCache(max_size=1, default_ttl=None)

    cache.set("a", {"text": "A"})
    cache.set("b", {"text": "B"})

    assert cache.get("a") is None
    assert cache.get("b") == {"text": "B"}
    assert cache.clear() == 1
    assert cache.size == 0


@pytest.mark.asyncio
async def test_generate_text_service_uses_cache():
    entry = DummyEntry()
    entry.options = {CONF_ENABLED_FEATURES: ["text_generation", "prompt_caching"]}
    hass = DummyHass([entry])
    session = DummySession(
        DummyResponse(
            200,
            {"content-type": "application/json"},
            {
                "model": "openai/gpt-oss-20b",
                "choices": [{"message": {"content": "cached answer"}}],
                "usage": {"total_tokens": 4},
            },
        )
    )
    handler = _handle_generate_text(hass)

    with patch(
        "custom_components.groq.api.async_get_clientsession",
        return_value=session,
    ):
        first = await handler(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    "prompt": "Hi",
                    "model": "openai/gpt-oss-20b",
                    "top_p": 0.8,
                    "stop": "DONE",
                    "seed": 123,
                    "service_tier": "flex",
                    "request_body_options": {
                        "user": "home-assistant",
                        "parallel_tool_calls": False,
                    },
                }
            )
        )
        second = await handler(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    "prompt": "Hi",
                    "model": "openai/gpt-oss-20b",
                    "top_p": 0.8,
                    "stop": "DONE",
                    "seed": 123,
                    "service_tier": "flex",
                    "request_body_options": {
                        "user": "home-assistant",
                        "parallel_tool_calls": False,
                    },
                }
            )
        )

    assert first["text"] == "cached answer"
    assert first["cached"] is False
    assert second["cached"] is True
    assert len(session.calls) == 1
    request_body = session.calls[0]["kwargs"]["json"]
    messages = request_body["messages"]
    assert messages[0] == {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    assert request_body["top_p"] == 0.8
    assert request_body["stop"] == "DONE"
    assert request_body["seed"] == 123
    assert request_body["service_tier"] == "flex"
    assert request_body["user"] == "home-assistant"
    assert request_body["parallel_tool_calls"] is False


@pytest.mark.asyncio
async def test_generate_text_service_rejects_excessive_completion_tokens():
    entry = DummyEntry()
    entry.options = {CONF_ENABLED_FEATURES: ["text_generation"]}
    hass = DummyHass([entry])
    handler = _handle_generate_text(hass)

    with pytest.raises(ServiceValidationError, match="65,536 completion tokens"):
        await handler(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    "prompt": "Hi",
                    "model": "openai/gpt-oss-20b",
                    "max_tokens": 65537,
                }
            )
        )

    with pytest.raises(ServiceValidationError, match="8,192 completion tokens"):
        await handler(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    "prompt": "Hi",
                    "model": "groq/compound",
                    "request_body_options": {"max_completion_tokens": "8193"},
                }
            )
        )


@pytest.mark.asyncio
async def test_generate_text_service_uses_selected_subentry_defaults():
    entry = DummyEntry()
    entry.subentries = {
        "text-service": SimpleNamespace(
            subentry_id="text-service",
            data={
                "service_type": "text_generation",
                "name": "Usage tracked text",
                "model": "openai/gpt-oss-20b",
                "system_prompt": "Use Home Assistant context.",
                "temperature": 0.1,
            },
        )
    }
    hass = DummyHass([entry])
    session = DummySession(
        DummyResponse(
            200,
            {"content-type": "application/json"},
            {
                "model": "openai/gpt-oss-20b",
                "choices": [{"message": {"content": "subentry answer"}}],
                "usage": {"total_tokens": 4},
            },
        )
    )
    handler = _handle_generate_text(hass)

    with patch(
        "custom_components.groq.api.async_get_clientsession",
        return_value=session,
    ):
        response = await handler(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    ATTR_SERVICE_ID: "text-service",
                    "prompt": "Hi",
                }
            )
        )

    assert response["text"] == "subentry answer"
    call = session.calls[0]
    assert call["kwargs"]["headers"]["Authorization"] == "Bearer api-key"
    request_body = call["kwargs"]["json"]
    assert request_body["model"] == "openai/gpt-oss-20b"
    assert request_body["messages"][0] == {
        "role": "system",
        "content": "Use Home Assistant context.",
    }
    assert request_body["temperature"] == 0.1


@pytest.mark.asyncio
async def test_generate_text_service_requires_service_id_for_multiple_subentries():
    entry = DummyEntry()
    entry.subentries = {
        "text-one": SimpleNamespace(
            subentry_id="text-one",
            data={
                "service_type": "text_generation",
                "name": "Text one",
                "model": "llama-3.1-8b-instant",
            },
        ),
        "text-two": SimpleNamespace(
            subentry_id="text-two",
            data={
                "service_type": "text_generation",
                "name": "Text two",
                "model": "openai/gpt-oss-20b",
            },
        ),
    }
    hass = DummyHass([entry])
    handler = _handle_generate_text(hass)

    with pytest.raises(ServiceValidationError, match="service_id"):
        await handler(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    "prompt": "Hi",
                }
            )
        )


@pytest.mark.asyncio
async def test_generate_text_service_does_not_cache_unsupported_models():
    entry = DummyEntry()
    entry.options = {CONF_ENABLED_FEATURES: ["text_generation", "prompt_caching"]}
    hass = DummyHass([entry])
    session = DummySession(
        DummyResponse(
            200,
            {"content-type": "application/json"},
            {
                "model": "llama-3.1-8b-instant",
                "choices": [{"message": {"content": "uncached answer"}}],
                "usage": {"total_tokens": 4},
            },
        )
    )
    handler = _handle_generate_text(hass)

    with patch(
        "custom_components.groq.api.async_get_clientsession",
        return_value=session,
    ):
        first = await handler(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    "prompt": "Hi",
                    "model": "llama-3.1-8b-instant",
                }
            )
        )
        second = await handler(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    "prompt": "Hi",
                    "model": "llama-3.1-8b-instant",
                }
            )
        )

    assert first["cached"] is False
    assert second["cached"] is False
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_generate_text_service_supports_structured_outputs():
    entry = DummyEntry()
    entry.options = {CONF_ENABLED_FEATURES: ["text_generation"]}
    hass = DummyHass([entry])
    session = DummySession(
        DummyResponse(
            200,
            {"content-type": "application/json"},
            {
                "model": "llama-3.1-8b-instant",
                "choices": [{"message": {"content": '{"summary": "Done"}'}}],
                "usage": {"total_tokens": 6},
            },
        )
    )
    handler = _handle_generate_text(hass)

    with patch(
        "custom_components.groq.api.async_get_clientsession",
        return_value=session,
    ):
        response = await handler(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    "prompt": "Summarize the home",
                    ATTR_SCHEMA: {
                        "type": "object",
                        "properties": {"summary": {"type": "string"}},
                        "required": ["summary"],
                    },
                    "schema_name": "home_summary",
                    "strict": False,
                }
            )
        )

    assert response["data"] == {"summary": "Done"}
    response_format = session.calls[0]["kwargs"]["json"]["response_format"]
    assert response_format == {
        "type": "json_schema",
        "json_schema": {
            "name": "home_summary",
            "schema": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
            "strict": False,
        },
    }


@pytest.mark.asyncio
async def test_generate_text_service_rejects_structured_outputs_for_unsupported_model():
    entry = DummyEntry()
    entry.options = {CONF_ENABLED_FEATURES: ["text_generation"]}
    hass = DummyHass([entry])
    handler = _handle_generate_text(hass)

    with pytest.raises(ServiceValidationError, match="structured_outputs"):
        await handler(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    "prompt": "Summarize the home",
                    "model": "custom/text-model",
                    ATTR_SCHEMA: {
                        "type": "object",
                        "properties": {"summary": {"type": "string"}},
                    },
                }
            )
        )


@pytest.mark.asyncio
async def test_generate_text_service_rejects_reasoning_for_unsupported_model():
    entry = DummyEntry()
    entry.options = {CONF_ENABLED_FEATURES: ["text_generation"]}
    hass = DummyHass([entry])
    handler = _handle_generate_text(hass)

    with pytest.raises(ServiceValidationError, match="reasoning"):
        await handler(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    "prompt": "Think carefully",
                    "model": "llama-3.1-8b-instant",
                    "reasoning_effort": "low",
                }
            )
        )


@pytest.mark.asyncio
async def test_generate_text_service_sends_reasoning_for_supported_model():
    entry = DummyEntry()
    entry.options = {CONF_ENABLED_FEATURES: ["text_generation"]}
    hass = DummyHass([entry])
    session = DummySession(
        DummyResponse(
            200,
            {"content-type": "application/json"},
            {
                "model": "openai/gpt-oss-20b",
                "choices": [
                    {
                        "message": {
                            "content": "Reasoned answer",
                            "reasoning": "Reasoning trace",
                        }
                    }
                ],
                "usage": {"total_tokens": 8},
            },
        )
    )
    handler = _handle_generate_text(hass)

    with patch(
        "custom_components.groq.api.async_get_clientsession",
        return_value=session,
    ):
        response = await handler(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    "prompt": "Think carefully",
                    "model": "openai/gpt-oss-20b",
                    "reasoning_effort": "medium",
                    "reasoning_format": "parsed",
                }
            )
        )

    assert response["text"] == "Reasoned answer"
    assert response["reasoning"] == "Reasoning trace"
    request_body = session.calls[0]["kwargs"]["json"]
    assert request_body["reasoning_effort"] == "medium"
    assert request_body["reasoning_format"] == "parsed"


@pytest.mark.asyncio
async def test_generate_text_service_supports_compound_models():
    entry = DummyEntry()
    entry.options = {CONF_ENABLED_FEATURES: ["text_generation"]}
    hass = DummyHass([entry])
    session = DummySession(
        DummyResponse(
            200,
            {"content-type": "application/json"},
            {
                "model": "groq/compound-mini",
                "choices": [
                    {
                        "message": {
                            "content": "Compound answer",
                            "executed_tools": [
                                {"type": "search", "name": "web_search"}
                            ],
                        }
                    }
                ],
                "usage": {"total_tokens": 12},
                "usage_breakdown": {
                    "models": [
                        {
                            "model": "llama-3.3-70b-versatile",
                            "usage": {"total_tokens": 4},
                        }
                    ]
                },
            },
        )
    )
    handler = _handle_generate_text(hass)

    with patch(
        "custom_components.groq.api.async_get_clientsession",
        return_value=session,
    ):
        response = await handler(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    "prompt": "Use Compound",
                    "model": "groq/compound-mini",
                    "request_body_options": {
                        "search_settings": {"include_domains": ["example.com"]},
                    },
                }
            )
        )

    assert response["text"] == "Compound answer"
    assert response["executed_tools"] == [{"type": "search", "name": "web_search"}]
    assert response["usage_breakdown"] == {
        "models": [
            {
                "model": "llama-3.3-70b-versatile",
                "usage": {"total_tokens": 4},
            }
        ]
    }
    request_body = session.calls[0]["kwargs"]["json"]
    assert request_body["model"] == "groq/compound-mini"
    assert request_body["search_settings"] == {"include_domains": ["example.com"]}


@pytest.mark.asyncio
async def test_analyze_image_service_uses_image_subentry_defaults():
    entry = DummyEntry()
    entry.subentries = {
        "vision-service": SimpleNamespace(
            subentry_id="vision-service",
            data={
                "service_type": "image_recognition",
                "name": "Vision usage key",
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "system_prompt": "Describe only visible objects.",
            },
        )
    }
    hass = DummyHass([entry])
    session = DummySession(
        DummyResponse(
            200,
            {"content-type": "application/json"},
            {
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "choices": [{"message": {"content": "A kitchen counter."}}],
                "usage": {"total_tokens": 9},
            },
        )
    )
    handler = _handle_analyze_image(hass)

    with patch(
        "custom_components.groq.api.async_get_clientsession",
        return_value=session,
    ):
        response = await handler(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    ATTR_SERVICE_ID: "vision-service",
                    "prompt": "What is shown?",
                    "image_url": "data:image/png;base64,YWJj",
                }
            )
        )

    assert response["text"] == "A kitchen counter."
    call = session.calls[0]
    assert call["kwargs"]["headers"]["Authorization"] == "Bearer api-key"
    request_body = call["kwargs"]["json"]
    assert request_body["model"] == "meta-llama/llama-4-scout-17b-16e-instruct"
    assert request_body["messages"][0] == {
        "role": "system",
        "content": "Describe only visible objects.",
    }


@pytest.mark.asyncio
async def test_conversation_entity_generates_assist_response():
    entry = DummyEntry()
    service_data = {
        "unique_id": "assist-service",
        "name": "Groq Assist",
        "model": "llama-3.1-8b-instant",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "temperature": 0.2,
        "stream": False,
    }
    client = DummyTextClient("Turned on the lights.")
    entity = GroqConversationEntity(DummyHass(), entry, service_data, client)
    chat_log = DummyChatLog()
    chat_log.content = [
        {"role": "user", "content": "What is the kitchen status?"},
        SimpleNamespace(role="assistant", content="The kitchen lights are off."),
    ]

    result = await entity._async_handle_message(
        SimpleNamespace(
            text="Turn on the kitchen lights",
            language="en",
            agent_id="conversation.groq_assist",
            extra_system_prompt="Prefer brief replies.",
        ),
        chat_log,
    )

    assert result.conversation_id == "conversation-id"
    assert chat_log.assistant_content[0].content == "Turned on the lights."
    request = client.requests[0]
    assert request.model == "llama-3.1-8b-instant"
    assert request.prompt == "Turn on the kitchen lights"
    assert request.messages == [
        {"role": "user", "content": "What is the kitchen status?"},
        {"role": "assistant", "content": "The kitchen lights are off."},
        {"role": "user", "content": "Turn on the kitchen lights"},
    ]
    assert DEFAULT_SYSTEM_PROMPT in request.system_prompt
    assert "Prefer brief replies." in request.system_prompt
    assert request.temperature == 0.2


@pytest.mark.asyncio
async def test_conversation_entity_limits_assist_history():
    entry = DummyEntry()
    service_data = {
        "unique_id": "assist-service",
        "name": "Groq Assist",
        "model": "llama-3.1-8b-instant",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "stream": False,
    }
    client = DummyTextClient("Done.")
    entity = GroqConversationEntity(DummyHass(), entry, service_data, client)
    chat_log = DummyChatLog()
    chat_log.content = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"turn {index}"}
        for index in range(30)
    ]

    await entity._async_handle_message(
        SimpleNamespace(
            text="latest request",
            language="en",
            agent_id="conversation.groq_assist",
            extra_system_prompt=None,
        ),
        chat_log,
    )

    request = client.requests[0]
    assert len(request.messages) == 13
    assert request.messages[0] == {"role": "user", "content": "turn 18"}
    assert request.messages[-1] == {"role": "user", "content": "latest request"}


def test_chat_log_messages_handles_role_fallbacks():
    class AssistantFallback:
        content = "Fallback assistant reply"

    class UserFallback:
        text = "Fallback user request"

    chat_log = SimpleNamespace(
        content=[
            AssistantFallback(),
            UserFallback(),
            SimpleNamespace(content="ignored"),
        ]
    )

    assert _chat_log_messages(chat_log, "current request") == [
        {"role": "assistant", "content": "Fallback assistant reply"},
        {"role": "user", "content": "Fallback user request"},
        {"role": "user", "content": "current request"},
    ]


@pytest.mark.asyncio
async def test_conversation_entity_streams_assist_response():
    entry = DummyEntry()
    service_data = {
        "unique_id": "assist-service",
        "name": "Groq Assist",
        "model": "llama-3.1-8b-instant",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "stream": True,
    }
    client = DummyTextClient("", stream_chunks=["Turned ", "on ", "the lights."])
    entity = GroqConversationEntity(DummyHass(), entry, service_data, client)
    chat_log = DummyChatLog()

    result = await entity._async_handle_message(
        SimpleNamespace(
            text="Turn on the kitchen lights",
            language="en",
            agent_id="conversation.groq_assist",
            extra_system_prompt=None,
        ),
        chat_log,
    )

    assert result.conversation_id == "conversation-id"
    assert chat_log.assistant_content[0].content == "Turned on the lights."
    assert chat_log.stream_deltas == [
        {"role": "assistant"},
        {"content": "Turned "},
        {"content": "on "},
        {"content": "the lights."},
    ]
    assert client.requests[0].model == "llama-3.1-8b-instant"


@pytest.mark.asyncio
async def test_stt_entity_transcribes_with_service_defaults():
    entry = DummyEntry()
    service_data = {
        "unique_id": "stt-service",
        "name": "Groq STT",
        "model": "whisper-large-v3",
        "language": "en",
    }
    client = DummySTTClient("turn on the kitchen lights")
    entity = GroqSTTEntity(entry, service_data, client)

    async def audio_stream():
        yield b"audio"

    result = await entity.async_process_audio_stream(
        stt.SpeechMetadata(
            language="en-US",
            format=stt.AudioFormats.OGG,
            codec=stt.AudioCodecs.OPUS,
            bit_rate=stt.AudioBitRates.BITRATE_16,
            sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
            channel=stt.AudioChannels.CHANNEL_MONO,
        ),
        audio_stream(),
    )

    assert result.result == stt.SpeechResultState.SUCCESS
    assert result.text == "turn on the kitchen lights"
    assert "api_key" not in client.requests[0]
    assert client.requests[0]["model"] == "whisper-large-v3"
    assert client.requests[0]["language"] == "en"


@pytest.mark.asyncio
async def test_ai_task_entity_generates_and_validates_structured_data():
    entry = DummyEntry()
    service_data = {
        "unique_id": "task-service",
        "name": "Groq Data Tasks",
        "model": "openai/gpt-oss-20b",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    }
    client = DummyTextClient('{"summary": "Garage door is open"}')
    entity = GroqAITaskEntity(DummyHass(), entry, service_data, client)
    task = GenDataTask(
        name="summary",
        instructions="Summarize the current home state",
        structure=vol.Schema({vol.Required("summary"): str}),
    )

    result = await entity._async_generate_data(task, DummyChatLog())

    assert result.conversation_id == "conversation-id"
    assert result.data == {"summary": "Garage door is open"}
    request = client.requests[0]
    assert isinstance(request, StructuredGenerationRequest)
    assert request.model == "openai/gpt-oss-20b"
    assert request.prompt == "Summarize the current home state"
    assert request.strict is True
    assert request.schema == {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "additionalProperties": False,
        "required": ["summary"],
    }
    assert request.system_prompt == DEFAULT_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_list_models_service_returns_registry_without_refresh():
    entry = DummyEntry()
    hass = DummyHass([entry])
    handler = _handle_list_models(hass)

    response = await handler(
        service_call({ATTR_CONFIG_ENTRY_ID: "entry-id", "refresh": False})
    )

    assert any(model["id"] == "llama-3.1-8b-instant" for model in response["models"])


@pytest.mark.asyncio
async def test_service_requires_entry_id_when_multiple_entries():
    hass = DummyHass([DummyEntry("one"), DummyEntry("two")])
    handler = _handle_list_models(hass)

    with pytest.raises(ServiceValidationError):
        await handler(service_call({"refresh": False}))
