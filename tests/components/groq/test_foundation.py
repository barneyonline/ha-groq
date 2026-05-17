from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import voluptuous as vol
from homeassistant.components import conversation, stt
from homeassistant.components.ai_task import AITaskEntityFeature, GenDataTask
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_LLM_HASS_API, Platform
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import intent, llm

from custom_components.groq.attachments import (
    async_attachment_content_parts,
    attachment_mime_type,
    attachment_path,
)
from custom_components.groq.api import (
    GroqApiClient,
    StructuredGenerationRequest,
    TextGenerationRequest,
    VisionRequest,
    build_structured_generation_payload,
    build_text_generation_payload,
    build_vision_payload,
    extract_tool_calls,
    normalize_base_url,
)
from custom_components.groq.ai_task import GroqAITaskEntity
from custom_components.groq.conversation import (
    GroqConversationEntity,
    MAX_HISTORY_MESSAGES,
    _async_chat_log_messages,
    _chat_log_messages,
    _chat_log_tools,
    _result_tool_calls,
    _tool_call_id,
    _tool_call_message,
    _tool_result_message,
)
from custom_components.groq.const import (
    COMPOUND_MODELS,
    CONF_COMPOUND_BUILTIN_TOOLS,
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
    model_from_api,
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

    async def async_add_executor_job(self, func, *args):
        return func(*args)


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


class DummyToolTextClient:
    def __init__(self):
        self.requests = []

    async def async_generate_text(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            return SimpleNamespace(
                text="",
                model=request.model,
                usage={"total_tokens": 12},
                usage_breakdown={"models": []},
                reasoning="Need current state.",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "GetState",
                            "arguments": '{"entity_id":"light.kitchen"}',
                        },
                    }
                ],
                executed_tools=None,
                raw={},
            )
        return SimpleNamespace(
            text="The kitchen light is on.",
            model=request.model,
            usage={"total_tokens": 7},
            usage_breakdown=None,
            reasoning=None,
            tool_calls=None,
            executed_tools=None,
            raw={},
        )


class DummyRealToolTextClient(DummyToolTextClient):
    async def async_generate_text(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            return SimpleNamespace(
                text="",
                model=request.model,
                usage={},
                usage_breakdown=None,
                reasoning=None,
                tool_calls=[
                    {
                        "id": "call_real",
                        "type": "function",
                        "function": {
                            "name": "GetState",
                            "arguments": '{"entity_id":"light.kitchen"}',
                        },
                    }
                ],
                executed_tools=None,
                raw={},
            )
        return SimpleNamespace(
            text="The kitchen light is on.",
            model=request.model,
            usage={},
            usage_breakdown=None,
            reasoning=None,
            tool_calls=None,
            executed_tools=None,
            raw={},
        )


class DummyStructuredToolTextClient(DummyToolTextClient):
    def __init__(self, final_text='{"summary":"The kitchen light is on."}'):
        super().__init__()
        self.final_text = final_text

    async def async_generate_text(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            return SimpleNamespace(
                text="",
                model=request.model,
                usage={},
                usage_breakdown=None,
                reasoning=None,
                tool_calls=[
                    {
                        "id": "call_structured",
                        "type": "function",
                        "function": {
                            "name": "GetState",
                            "arguments": '{"entity_id":"light.kitchen"}',
                        },
                    }
                ],
                executed_tools=None,
                raw={},
            )
        return SimpleNamespace(
            text=self.final_text,
            model=request.model,
            usage={},
            usage_breakdown=None,
            reasoning=None,
            tool_calls=None,
            executed_tools=None,
            raw={},
        )


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


class DummyToolChatLog(DummyChatLog):
    def __init__(self):
        super().__init__()
        self.content = [SimpleNamespace(role="system", content="")]
        self.llm_api = None
        self.provided_llm_data = None

    @property
    def unresponded_tool_results(self):
        return self.content[-1].role == "tool_result"

    @property
    def continue_conversation(self):
        return False

    def async_add_assistant_content_without_tools(self, content):
        super().async_add_assistant_content_without_tools(content)
        self.content.append(content)

    async def async_provide_llm_data(
        self,
        llm_context,
        user_llm_hass_api,
        user_llm_prompt,
        user_extra_system_prompt,
    ):
        self.provided_llm_data = {
            "context": llm_context,
            "api": user_llm_hass_api,
            "prompt": user_llm_prompt,
            "extra": user_extra_system_prompt,
        }
        self.content[0] = SimpleNamespace(
            role="system",
            content=f"{user_llm_prompt}\n{user_extra_system_prompt}",
        )
        self.llm_api = SimpleNamespace(
            custom_serializer=None,
            tools=[
                SimpleNamespace(
                    name="GetState",
                    description="Get an entity state",
                    parameters=vol.Schema({vol.Required("entity_id"): str}),
                )
            ],
        )

    async def async_add_assistant_content(self, content):
        self.assistant_content.append(content)
        self.content.append(content)
        if not content.tool_calls:
            return
        for tool_call in content.tool_calls:
            tool_result = SimpleNamespace(
                role="tool_result",
                agent_id=content.agent_id,
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                tool_result={"state": "on"},
            )
            self.content.append(tool_result)
            yield tool_result


class DummyStateTool(llm.Tool):
    name = "GetState"
    description = "Get an entity state"
    parameters = vol.Schema({vol.Required("entity_id"): str})

    async def async_call(self, hass, tool_input, llm_context):
        return {"entity_id": tool_input.tool_args["entity_id"], "state": "on"}


class DummyToolAPI(llm.API):
    async def async_get_api_instance(self, llm_context):
        return llm.APIInstance(
            api=self,
            api_prompt="Use GetState for current entity state.",
            llm_context=llm_context,
            tools=[DummyStateTool()],
        )


async def async_dummy_tool_chat_log() -> DummyToolChatLog:
    """Return a dummy chat log with Home Assistant tools populated."""
    chat_log = DummyToolChatLog()
    await chat_log.async_provide_llm_data(
        SimpleNamespace(platform="groq"),
        user_llm_hass_api=None,
        user_llm_prompt=DEFAULT_SYSTEM_PROMPT,
        user_extra_system_prompt=None,
    )
    return chat_log


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


@pytest.mark.asyncio
async def test_attachment_helpers_handle_dicts_and_guardrails(tmp_path):
    image_path = tmp_path / "snapshot.png"
    image_path.write_bytes(b"image")

    assert attachment_mime_type({"media_content_type": "image/png"}) == "image/png"
    assert attachment_path({"path": image_path}) == image_path
    assert attachment_path({}) is None
    assert (
        await async_attachment_content_parts(DummyHass(), [], text="Describe") is None
    )

    with pytest.raises(HomeAssistantError, match="resolve to files"):
        await async_attachment_content_parts(
            DummyHass(),
            [SimpleNamespace(mime_type="image/png")],
            text="Describe",
        )

    with pytest.raises(HomeAssistantError, match="must be a file"):
        await async_attachment_content_parts(
            DummyHass(),
            [{"mime_type": "image/png", "path": tmp_path}],
            text="Describe",
        )


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
            compound_builtin_tools=["visit_website", "web_search"],
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
        "compound_custom": {
            "tools": {"enabled_tools": ["web_search", "visit_website"]}
        },
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

    tool_payload = build_text_generation_payload(
        TextGenerationRequest(
            prompt="",
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_state",
                                "arguments": '{"entity_id":"light.kitchen"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": '{"state":"on"}',
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_state",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            tool_choice="auto",
        )
    )

    assert tool_payload["messages"][0]["tool_calls"][0]["id"] == "call_1"
    assert tool_payload["messages"][1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"state":"on"}',
    }
    assert tool_payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_state",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert tool_payload["tool_choice"] == "auto"


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
    assert "Groq-Model-Version" not in call["kwargs"]["headers"]


@pytest.mark.asyncio
async def test_api_client_sends_latest_header_for_latest_compound_tools():
    session = DummySession(
        DummyResponse(
            200,
            {"content-type": "application/json"},
            {
                "model": "groq/compound",
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

    await client.async_generate_text(
        TextGenerationRequest(
            prompt="Go",
            model="groq/compound",
            compound_builtin_tools=["web_search", "visit_website"],
        )
    )

    assert session.calls[0]["kwargs"]["headers"]["Groq-Model-Version"] == "latest"


@pytest.mark.asyncio
async def test_api_client_sends_latest_header_for_raw_compound_tools():
    session = DummySession(
        DummyResponse(
            200,
            {"content-type": "application/json"},
            {
                "model": "groq/compound",
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

    await client.async_generate_text(
        TextGenerationRequest(
            prompt="Go",
            model="groq/compound",
            extra_body={
                "compound_custom": {
                    "tools": {"enabled_tools": ["web_search", "wolfram_alpha"]}
                }
            },
        )
    )

    assert session.calls[0]["kwargs"]["headers"]["Groq-Model-Version"] == "latest"


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
async def test_api_client_extracts_openai_tool_calls_and_reasoning():
    session = DummySession(
        DummyResponse(
            200,
            {"content-type": "application/json"},
            {
                "model": "openai/gpt-oss-20b",
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning": "Need current entity state.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_state",
                                        "arguments": '{"entity_id":"light.kitchen"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
        )
    )
    client = GroqApiClient(DummyHass(), api_key="api-key", session=session)

    result = await client.async_generate_text(
        TextGenerationRequest(prompt="Kitchen?", model="openai/gpt-oss-20b")
    )

    assert result.text == ""
    assert result.reasoning == "Need current entity state."
    assert result.tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_state",
                "arguments": '{"entity_id":"light.kitchen"}',
                "parsed_arguments": {"entity_id": "light.kitchen"},
            },
        }
    ]
    assert extract_tool_calls(
        {"choices": [{"message": {"tool_calls": [{"function": {"arguments": "{"}}]}}]}
    ) == [{"function": {"arguments": "{"}}]
    assert extract_tool_calls({"choices": []}) is None
    assert (
        extract_tool_calls({"choices": [{"message": {"tool_calls": ["bad"]}}]}) is None
    )


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
    assert GroqCapability.TEXT_TO_SPEECH in infer_capabilities(
        "canopylabs/orpheus-custom"
    )
    assert infer_capabilities("playai-tts") == frozenset()
    assert not GroqModelRegistry(
        [model_from_api({"id": "playai-tts"})],
        include_built_ins=False,
    ).models_for_feature(GroqFeature.TEXT_GENERATION)
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
    assert GroqModelRegistry().supports(
        "openai/gpt-oss-20b", GroqCapability.TOOL_CALLING
    )
    assert not GroqModelRegistry().supports(
        "groq/compound", GroqCapability.TOOL_CALLING
    )
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


def test_text_generation_config_flow_rejects_llm_tools_for_unsupported_models():
    errors = validate_text_generation_input(
        {
            CONF_MODEL: "groq/compound-mini",
            CONF_LLM_HASS_API: ["assist"],
        }
    )

    assert errors == {CONF_LLM_HASS_API: "unsupported_tool_calling_model"}


def test_text_generation_config_flow_validates_compound_builtin_tools():
    errors = validate_text_generation_input(
        {
            CONF_MODEL: "llama-3.1-8b-instant",
            CONF_COMPOUND_BUILTIN_TOOLS: ["web_search"],
        }
    )

    assert errors == {
        CONF_COMPOUND_BUILTIN_TOOLS: "unsupported_compound_builtin_tools_model"
    }

    errors = validate_text_generation_input(
        {
            CONF_MODEL: "groq/compound",
            CONF_COMPOUND_BUILTIN_TOOLS: ["web_search", "bad_tool"],
        }
    )

    assert errors == {CONF_COMPOUND_BUILTIN_TOOLS: "invalid_compound_builtin_tools"}

    errors = validate_text_generation_input(
        {
            CONF_MODEL: "groq/compound",
            CONF_COMPOUND_BUILTIN_TOOLS: ["browser_automation"],
        }
    )

    assert errors == {CONF_COMPOUND_BUILTIN_TOOLS: "invalid_compound_builtin_tools"}

    errors = validate_text_generation_input(
        {
            CONF_MODEL: "groq/compound",
            CONF_COMPOUND_BUILTIN_TOOLS: [
                "web_search",
                "browser_automation",
                "code_interpreter",
            ],
        }
    )

    assert errors == {}

    errors = validate_text_generation_input(
        {
            CONF_MODEL: "groq/compound",
            "request_body_options": {
                "compound_custom": {
                    "tools": {"enabled_tools": ["web_search", "bad_tool"]}
                }
            },
        }
    )

    assert errors == {"request_body_options": "invalid_compound_builtin_tools"}


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


def test_tool_message_helpers_handle_dicts_and_invalid_values():
    assert _tool_call_id({"id": "call_123"}) == "call_123"
    assert _tool_call_id({}) == "tool_call"
    assert _tool_call_message(
        {
            "id": "call_1",
            "tool_name": "GetState",
            "tool_args": {"entity_id": "light.kitchen"},
        }
    ) == {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "GetState",
            "arguments": '{"entity_id":"light.kitchen"}',
        },
    }
    assert _tool_call_message({"tool_name": "GetState", "tool_args": None}) is None
    assert _tool_result_message({"tool_call_id": "call_1"}) is None


def test_chat_log_messages_includes_system_tool_and_assistant_tool_calls():
    chat_log = SimpleNamespace(
        content=[
            {"role": "system", "content": "System prompt"},
            {
                "role": "tool_result",
                "tool_call_id": "call_1",
                "tool_name": "GetState",
                "tool_result": {"state": "on"},
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "tool_name": "GetState",
                        "tool_args": {"entity_id": "light.kitchen"},
                    }
                ],
            },
            {"content": "ignored"},
        ]
    )

    assert _chat_log_messages(chat_log, "current request") == [
        {"role": "system", "content": "System prompt"},
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "GetState",
            "content": '{"state":"on"}',
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "GetState",
                        "arguments": '{"entity_id":"light.kitchen"}',
                    },
                }
            ],
        },
        {"role": "user", "content": "current request"},
    ]


@pytest.mark.asyncio
async def test_async_chat_log_messages_drops_orphan_tool_results():
    messages = [
        {"role": "user", "content": f"turn {index}"}
        for index in range(MAX_HISTORY_MESSAGES - 1)
    ]
    messages.extend(
        [
            {
                "role": "tool_result",
                "tool_call_id": "orphaned",
                "tool_name": "GetState",
                "tool_result": {"state": "on"},
            },
            {"content": "ignored"},
            {"role": "user", "content": "current request"},
        ]
    )

    result = await _async_chat_log_messages(
        DummyHass(),
        GroqModelRegistry(),
        "openai/gpt-oss-20b",
        SimpleNamespace(content=messages),
        "current request",
    )

    assert not any(message["role"] == "tool" for message in result)
    assert result[-1] == {"role": "user", "content": "current request"}


def test_tool_conversion_handles_empty_api_and_raw_result_shapes():
    assert _chat_log_tools(SimpleNamespace(llm_api=SimpleNamespace(tools=[]))) is None

    tool_inputs = _result_tool_calls(
        SimpleNamespace(
            tool_calls=None,
            raw={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                "bad",
                                {"function": None},
                                {"function": {"arguments": "{}"}},
                                {
                                    "id": "bad_json",
                                    "function": {
                                        "name": "BadJson",
                                        "arguments": "{",
                                    },
                                },
                                {
                                    "id": "dict_args",
                                    "function": {
                                        "name": "DictArgs",
                                        "arguments": {"entity_id": "light.kitchen"},
                                    },
                                },
                                {
                                    "id": "other_args",
                                    "function": {
                                        "name": "OtherArgs",
                                        "arguments": 123,
                                    },
                                },
                            ]
                        }
                    }
                ]
            },
        )
    )

    assert [(item.tool_name, item.tool_args, item.id) for item in tool_inputs] == [
        ("BadJson", {}, "bad_json"),
        ("DictArgs", {"entity_id": "light.kitchen"}, "dict_args"),
        ("OtherArgs", {}, "other_args"),
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
async def test_conversation_entity_sends_image_attachments_to_vision_models(tmp_path):
    entry = DummyEntry()
    service_data = {
        "unique_id": "assist-service",
        "name": "Groq Vision Assist",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "stream": False,
    }
    client = DummyTextClient("The image shows the garage.")
    entity = GroqConversationEntity(DummyHass(), entry, service_data, client)
    image_path = tmp_path / "snapshot.png"
    image_path.write_bytes(b"image bytes")
    chat_log = DummyChatLog()
    chat_log.content = [
        SimpleNamespace(
            role="user",
            content="What is in this image?",
            attachments=[
                SimpleNamespace(
                    mime_type="image/png",
                    path=image_path,
                )
            ],
        )
    ]

    await entity._async_handle_message(
        SimpleNamespace(
            text="What is in this image?",
            language="en",
            agent_id="conversation.groq_assist",
            extra_system_prompt=None,
        ),
        chat_log,
    )

    request = client.requests[0]
    assert request.messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aW1hZ2UgYnl0ZXM="},
                },
            ],
        }
    ]

    text_entity = GroqConversationEntity(
        DummyHass(),
        entry,
        {"model": "llama-3.1-8b-instant"},
        DummyTextClient("No vision"),
    )
    with pytest.raises(HomeAssistantError, match="vision-capable model"):
        await text_entity._async_handle_message(
            SimpleNamespace(
                text="What is in this image?",
                language="en",
                agent_id="conversation.groq_assist",
                extra_system_prompt=None,
            ),
            chat_log,
        )


@pytest.mark.asyncio
async def test_conversation_entity_prefers_current_input_attachments(tmp_path):
    entry = DummyEntry()
    service_data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "stream": False,
    }
    client = DummyTextClient("The image shows the driveway.")
    entity = GroqConversationEntity(DummyHass(), entry, service_data, client)
    image_path = tmp_path / "current.png"
    image_path.write_bytes(b"current image")
    chat_log = DummyChatLog()
    chat_log.content = [{"role": "user", "content": "Describe this"}]

    await entity._async_handle_message(
        SimpleNamespace(
            text="Describe this",
            language="en",
            agent_id="conversation.groq_assist",
            extra_system_prompt=None,
            attachments=[
                SimpleNamespace(
                    mime_type="image/png",
                    path=image_path,
                )
            ],
        ),
        chat_log,
    )

    assert client.requests[0].messages[-1] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,Y3VycmVudCBpbWFnZQ=="},
            },
        ],
    }


@pytest.mark.asyncio
async def test_conversation_entity_keeps_repeated_prompt_current_attachment(tmp_path):
    entry = DummyEntry()
    service_data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "stream": False,
    }
    client = DummyTextClient("The image shows the driveway.")
    entity = GroqConversationEntity(DummyHass(), entry, service_data, client)
    old_image_path = tmp_path / "old.png"
    old_image_path.write_bytes(b"old image")
    current_image_path = tmp_path / "current.png"
    current_image_path.write_bytes(b"current image")
    chat_log = DummyChatLog()
    chat_log.content = [
        SimpleNamespace(
            role="user",
            content="Describe this",
            attachments=[
                SimpleNamespace(
                    mime_type="image/png",
                    path=old_image_path,
                )
            ],
        )
    ]

    await entity._async_handle_message(
        SimpleNamespace(
            text="Describe this",
            language="en",
            agent_id="conversation.groq_assist",
            extra_system_prompt=None,
            attachments=[
                SimpleNamespace(
                    mime_type="image/png",
                    path=current_image_path,
                )
            ],
        ),
        chat_log,
    )

    assert client.requests[0].messages[-1] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,Y3VycmVudCBpbWFnZQ=="},
            },
        ],
    }
    assert client.requests[0].messages[-2]["content"][1]["image_url"] == {
        "url": "data:image/png;base64,b2xkIGltYWdl"
    }


@pytest.mark.asyncio
async def test_conversation_entity_uses_home_assistant_llm_tools():
    entry = DummyEntry()
    service_data = {
        "unique_id": "assist-service",
        "name": "Groq Assist",
        "model": "openai/gpt-oss-20b",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "stream": True,
        CONF_LLM_HASS_API: ["assist"],
    }
    client = DummyToolTextClient()
    entity = GroqConversationEntity(DummyHass(), entry, service_data, client)
    chat_log = DummyToolChatLog()

    result = await entity._async_handle_message(
        SimpleNamespace(
            text="Is the kitchen light on?",
            language="en",
            agent_id="conversation.groq_assist",
            extra_system_prompt="Prefer brief replies.",
            as_llm_context=lambda domain: SimpleNamespace(platform=domain),
        ),
        chat_log,
    )

    assert result.conversation_id == "conversation-id"
    assert result.continue_conversation is False
    assert chat_log.provided_llm_data["api"] == ["assist"]
    assert DEFAULT_SYSTEM_PROMPT in chat_log.provided_llm_data["prompt"]
    assert chat_log.provided_llm_data["extra"] == "Prefer brief replies."
    first_request = client.requests[0]
    assert first_request.system_prompt is None
    assert first_request.tools[0]["function"]["name"] == "GetState"
    assert first_request.tool_choice == "auto"
    assert chat_log.assistant_content[0].thinking_content == "Need current state."
    assert chat_log.assistant_content[0].native == {
        "model": "openai/gpt-oss-20b",
        "usage": {"total_tokens": 12},
        "usage_breakdown": {"models": []},
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "GetState",
                    "arguments": '{"entity_id":"light.kitchen"}',
                },
            }
        ],
    }
    assert chat_log.assistant_content[0].tool_calls[0].tool_args == {
        "entity_id": "light.kitchen"
    }
    second_request = client.requests[1]
    assert second_request.messages[-2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "GetState",
        "content": '{"state":"on"}',
    }


@pytest.mark.asyncio
async def test_conversation_entity_returns_converse_error_from_llm_setup():
    class ErrorChatLog(DummyChatLog):
        async def async_provide_llm_data(self, *args):
            response = intent.IntentResponse(language="en")
            response.async_set_error("unknown", "Tool setup failed")
            raise conversation.ConverseError(
                "Tool setup failed", self.conversation_id, response
            )

    entry = DummyEntry()
    service_data = {
        "model": "openai/gpt-oss-20b",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    }
    entity = GroqConversationEntity(
        DummyHass(), entry, service_data, DummyTextClient("unused")
    )

    result = await entity._async_handle_message(
        SimpleNamespace(
            text="Hello",
            language="en",
            agent_id="conversation.groq_assist",
            extra_system_prompt=None,
            as_llm_context=lambda domain: SimpleNamespace(platform=domain),
        ),
        ErrorChatLog(),
    )

    assert result.conversation_id == "conversation-id"
    assert result.response.speech["plain"]["speech"] == "Tool setup failed"


@pytest.mark.asyncio
async def test_conversation_entity_rejects_tools_for_non_tool_model():
    entry = DummyEntry()
    service_data = {
        "model": "no-tools",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    }
    entity = GroqConversationEntity(
        DummyHass(),
        entry,
        service_data,
        DummyTextClient("unused"),
        GroqModelRegistry([], include_built_ins=False),
    )

    with pytest.raises(HomeAssistantError, match="tool calls"):
        await entity._async_handle_message(
            SimpleNamespace(
                text="Use tools",
                language="en",
                agent_id="conversation.groq_assist",
                extra_system_prompt=None,
                as_llm_context=lambda domain: SimpleNamespace(platform=domain),
            ),
            DummyToolChatLog(),
        )


@pytest.mark.asyncio
async def test_conversation_entity_uses_tool_result_content_as_reply():
    class ReplyingToolChatLog(DummyChatLog):
        def __init__(self):
            super().__init__()
            self.llm_api = SimpleNamespace(
                custom_serializer=None,
                tools=[
                    SimpleNamespace(
                        name="GetState",
                        description="Get an entity state",
                        parameters=vol.Schema({vol.Required("entity_id"): str}),
                    )
                ],
            )

        @property
        def unresponded_tool_results(self):
            return False

        async def async_add_assistant_content(self, content):
            self.assistant_content.append(content)
            yield SimpleNamespace(content="Tool supplied final text.")

    entry = DummyEntry()
    service_data = {
        "model": "openai/gpt-oss-20b",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    }
    entity = GroqConversationEntity(
        DummyHass(), entry, service_data, DummyToolTextClient()
    )

    result = await entity._async_handle_message(
        SimpleNamespace(
            text="Use tools",
            language="en",
            agent_id="conversation.groq_assist",
            extra_system_prompt=None,
        ),
        ReplyingToolChatLog(),
    )

    assert result.response.speech["plain"]["speech"] == "Tool supplied final text."


@pytest.mark.asyncio
async def test_conversation_entity_limits_unresolved_tool_iterations():
    class AlwaysToolClient:
        def __init__(self):
            self.requests = []

        async def async_generate_text(self, request):
            self.requests.append(request)
            return SimpleNamespace(
                text="",
                model=request.model,
                usage={},
                usage_breakdown=None,
                reasoning=None,
                tool_calls=[
                    {
                        "id": "call_loop",
                        "type": "function",
                        "function": {
                            "name": "GetState",
                            "arguments": '{"entity_id":"light.kitchen"}',
                        },
                    }
                ],
                executed_tools=None,
                raw={},
            )

    class StuckToolChatLog(DummyChatLog):
        def __init__(self):
            super().__init__()
            self.llm_api = SimpleNamespace(
                custom_serializer=None,
                tools=[
                    SimpleNamespace(
                        name="GetState",
                        description="Get an entity state",
                        parameters=vol.Schema({vol.Required("entity_id"): str}),
                    )
                ],
            )

        @property
        def unresponded_tool_results(self):
            return True

        async def async_add_assistant_content(self, content):
            self.assistant_content.append(content)
            yield SimpleNamespace(content=None)

    entry = DummyEntry()
    service_data = {
        "model": "openai/gpt-oss-20b",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    }
    client = AlwaysToolClient()
    entity = GroqConversationEntity(DummyHass(), entry, service_data, client)

    with pytest.raises(HomeAssistantError, match="tool-call limit"):
        await entity._async_handle_message(
            SimpleNamespace(
                text="Use tools",
                language="en",
                agent_id="conversation.groq_assist",
                extra_system_prompt=None,
            ),
            StuckToolChatLog(),
        )

    assert len(client.requests) == 10


@pytest.mark.asyncio
async def test_conversation_entity_uses_real_chat_log_llm_prompt():
    entry = DummyEntry()
    service_data = {
        "unique_id": "assist-service",
        "name": "Groq Assist",
        "model": "openai/gpt-oss-20b",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "stream": False,
    }
    client = DummyTextClient("Done.")
    entity = GroqConversationEntity(DummyHass(), entry, service_data, client)
    chat_log = DummyToolChatLog()

    result = await entity._async_handle_message(
        SimpleNamespace(
            text="Summarize the house",
            language="en",
            agent_id="conversation.groq_assist",
            extra_system_prompt="Use one sentence.",
            as_llm_context=lambda domain: llm.LLMContext(
                platform=domain,
                context=None,
                language="en",
                assistant=None,
                device_id=None,
            ),
        ),
        chat_log,
    )

    assert result.conversation_id == "conversation-id"
    assert client.requests[0].system_prompt is None
    assert DEFAULT_SYSTEM_PROMPT in client.requests[0].messages[0]["content"]
    assert "Use one sentence." in client.requests[0].messages[0]["content"]


@pytest.mark.asyncio
async def test_conversation_entity_uses_real_chat_log_tool_execution():
    entry = DummyEntry()
    service_data = {
        "unique_id": "assist-service",
        "name": "Groq Assist",
        "model": "openai/gpt-oss-20b",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "stream": False,
        CONF_LLM_HASS_API: ["test_tools"],
    }
    client = DummyRealToolTextClient()
    entity = GroqConversationEntity(DummyHass(), entry, service_data, client)
    chat_log = DummyToolChatLog()

    result = await entity._async_handle_message(
        SimpleNamespace(
            text="Is the kitchen light on?",
            language="en",
            agent_id="conversation.groq_assist",
            extra_system_prompt=None,
            as_llm_context=lambda domain: SimpleNamespace(platform=domain),
        ),
        chat_log,
    )

    assert result.conversation_id == "conversation-id"
    assert result.response.speech["plain"]["speech"] == "The kitchen light is on."
    assert client.requests[1].messages[-2] == {
        "role": "tool",
        "tool_call_id": "call_real",
        "name": "GetState",
        "content": '{"state":"on"}',
    }


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


def test_ai_task_text_generation_request_rejects_invalid_system_prompt():
    entry = DummyEntry()
    entity = GroqAITaskEntity(
        DummyHass(),
        entry,
        {"model": "openai/gpt-oss-20b", "system_prompt": DEFAULT_SYSTEM_PROMPT},
        DummyTextClient("{}"),
    )

    with pytest.raises(TypeError, match="system_prompt"):
        entity._text_generation_request("Generate data", system_prompt=object())


@pytest.mark.asyncio
async def test_ai_task_entity_sends_image_attachments_to_vision_models(tmp_path):
    entry = DummyEntry()
    service_data = {
        "unique_id": "vision-task-service",
        "name": "Groq Vision Tasks",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    }
    client = DummyTextClient("Garage door is open")
    entity = GroqAITaskEntity(DummyHass(), entry, service_data, client)
    image_path = tmp_path / "snapshot.png"
    image_path.write_bytes(b"image bytes")
    task = GenDataTask(
        name="camera_summary",
        instructions="Summarize this camera image",
        attachments=[
            SimpleNamespace(
                mime_type="image/png",
                path=image_path,
            )
        ],
    )

    result = await entity._async_generate_data(task, DummyChatLog())

    assert result.data == "Garage door is open"
    assert AITaskEntityFeature.SUPPORT_ATTACHMENTS in entity.supported_features
    request = client.requests[0]
    assert isinstance(request, TextGenerationRequest)
    assert request.messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize this camera image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aW1hZ2UgYnl0ZXM="},
                },
            ],
        }
    ]

    text_entity = GroqAITaskEntity(
        DummyHass(),
        entry,
        {"model": "llama-3.1-8b-instant"},
        DummyTextClient("No vision"),
    )
    assert AITaskEntityFeature.SUPPORT_ATTACHMENTS not in text_entity.supported_features
    with pytest.raises(HomeAssistantError, match="vision-capable model"):
        await text_entity._async_generate_data(task, DummyChatLog())


@pytest.mark.asyncio
async def test_ai_task_image_attachment_guardrails(tmp_path):
    entry = DummyEntry()
    service_data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    }
    entity = GroqAITaskEntity(DummyHass(), entry, service_data, DummyTextClient("ok"))
    text_path = tmp_path / "note.txt"
    text_path.write_text("not an image")
    task = GenDataTask(
        name="bad_attachment",
        instructions="Describe",
        attachments=[SimpleNamespace(mime_type="text/plain", path=text_path)],
    )
    with pytest.raises(HomeAssistantError, match="image files"):
        await entity._async_generate_data(task, DummyChatLog())

    missing_task = GenDataTask(
        name="missing_attachment",
        instructions="Describe",
        attachments=[
            SimpleNamespace(mime_type="image/png", path=tmp_path / "missing.png")
        ],
    )
    with pytest.raises(HomeAssistantError, match="does not exist"):
        await entity._async_generate_data(missing_task, DummyChatLog())

    oversized_path = tmp_path / "oversized.png"
    oversized_path.write_bytes(b"12345")
    oversized_task = GenDataTask(
        name="oversized_attachment",
        instructions="Describe",
        attachments=[SimpleNamespace(mime_type="image/png", path=oversized_path)],
    )
    with (
        patch("custom_components.groq.attachments.MAX_IMAGE_ATTACHMENT_BYTES", 4),
        pytest.raises(HomeAssistantError, match="exceeds the 10 MB"),
    ):
        await entity._async_generate_data(oversized_task, DummyChatLog())


@pytest.mark.asyncio
async def test_ai_task_entity_uses_task_llm_api_tools():
    entry = DummyEntry()
    service_data = {
        "unique_id": "task-service",
        "name": "Groq AI Tasks",
        "model": "openai/gpt-oss-20b",
    }
    client = DummyRealToolTextClient()
    entity = GroqAITaskEntity(DummyHass(), entry, service_data, client)
    task = GenDataTask(
        name="state_summary",
        instructions="Is the kitchen light on?",
    )
    chat_log = await async_dummy_tool_chat_log()

    result = await entity._async_generate_data(task, chat_log)

    assert result.conversation_id == "conversation-id"
    assert result.data == "The kitchen light is on."
    assert client.requests[0].tools[0]["function"]["name"] == "GetState"
    assert client.requests[0].tool_choice == "auto"
    assert {
        "role": "tool",
        "tool_call_id": "call_real",
        "name": "GetState",
        "content": '{"state":"on"}',
    } in client.requests[1].messages


@pytest.mark.asyncio
async def test_ai_task_entity_uses_tools_with_structure_and_image(tmp_path):
    entry = DummyEntry()
    service_data = {
        "unique_id": "task-service",
        "name": "Groq AI Tasks",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    }
    client = DummyStructuredToolTextClient()
    entity = GroqAITaskEntity(DummyHass(), entry, service_data, client)
    image_path = tmp_path / "snapshot.png"
    image_path.write_bytes(b"image bytes")
    attachments = [
        SimpleNamespace(
            media_content_id="media-source://camera/snapshot",
            mime_type="image/png",
            path=image_path,
        )
    ]
    task = GenDataTask(
        name="state_summary",
        instructions="Summarize this camera image and kitchen light state.",
        structure=vol.Schema({vol.Required("summary"): str}),
        attachments=attachments,
    )
    chat_log = await async_dummy_tool_chat_log()

    result = await entity._async_generate_data(task, chat_log)

    assert result.data == {"summary": "The kitchen light is on."}
    assert (
        "Return only a valid JSON object" in client.requests[0].messages[0]["content"]
    )
    assert "summary" in client.requests[0].messages[0]["content"]
    for request in client.requests:
        image_messages = [
            message
            for message in request.messages
            if message["role"] == "user" and isinstance(message.get("content"), list)
        ]
        assert len(image_messages) == 1
        assert image_messages[0]["content"] == [
            {"type": "text", "text": task.instructions},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aW1hZ2UgYnl0ZXM="},
            },
        ]


@pytest.mark.asyncio
async def test_ai_task_entity_validates_service_schema_with_tools():
    entry = DummyEntry()
    service_data = {
        "unique_id": "task-service",
        "name": "Groq AI Tasks",
        "model": "openai/gpt-oss-20b",
        "structured_outputs": True,
        "schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
    }
    client = DummyStructuredToolTextClient('{"summary":"The kitchen light is on."}')
    entity = GroqAITaskEntity(DummyHass(), entry, service_data, client)
    task = GenDataTask(
        name="state_summary",
        instructions="Summarize the kitchen light state.",
    )
    chat_log = await async_dummy_tool_chat_log()

    result = await entity._async_generate_data(task, chat_log)

    assert result.data == {"summary": "The kitchen light is on."}
    assert (
        "Return only a valid JSON object" in client.requests[0].messages[0]["content"]
    )
    assert '"required":["summary"]' in client.requests[0].messages[0]["content"]


@pytest.mark.asyncio
async def test_ai_task_entity_rejects_invalid_service_schema_tool_result():
    entry = DummyEntry()
    service_data = {
        "unique_id": "task-service",
        "name": "Groq AI Tasks",
        "model": "openai/gpt-oss-20b",
        "structured_outputs": True,
        "schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
    }
    client = DummyStructuredToolTextClient('{"wrong":"shape"}')
    entity = GroqAITaskEntity(DummyHass(), entry, service_data, client)
    task = GenDataTask(
        name="state_summary",
        instructions="Summarize the kitchen light state.",
    )
    chat_log = await async_dummy_tool_chat_log()

    with pytest.raises(HomeAssistantError, match="requested structure"):
        await entity._async_generate_data(task, chat_log)


@pytest.mark.asyncio
async def test_ai_task_entity_rejects_invalid_service_schema_with_tools():
    entry = DummyEntry()
    service_data = {
        "unique_id": "task-service",
        "name": "Groq AI Tasks",
        "model": "openai/gpt-oss-20b",
        "structured_outputs": True,
        "schema": {
            "type": "not-a-json-schema-type",
            "properties": {"summary": {"type": "string"}},
        },
    }
    client = DummyStructuredToolTextClient('{"summary":"The kitchen light is on."}')
    entity = GroqAITaskEntity(DummyHass(), entry, service_data, client)
    task = GenDataTask(
        name="state_summary",
        instructions="Summarize the kitchen light state.",
    )
    chat_log = await async_dummy_tool_chat_log()

    with pytest.raises(HomeAssistantError, match="requested structure"):
        await entity._async_generate_data(task, chat_log)


@pytest.mark.asyncio
async def test_ai_task_entity_rejects_unresolved_service_schema_ref_with_tools():
    entry = DummyEntry()
    service_data = {
        "unique_id": "task-service",
        "name": "Groq AI Tasks",
        "model": "openai/gpt-oss-20b",
        "structured_outputs": True,
        "schema": {
            "type": "object",
            "properties": {"summary": {"$ref": "#/$defs/missing"}},
        },
    }
    client = DummyStructuredToolTextClient('{"summary":"The kitchen light is on."}')
    entity = GroqAITaskEntity(DummyHass(), entry, service_data, client)
    task = GenDataTask(
        name="state_summary",
        instructions="Summarize the kitchen light state.",
    )
    chat_log = await async_dummy_tool_chat_log()

    with pytest.raises(HomeAssistantError, match="requested structure"):
        await entity._async_generate_data(task, chat_log)


@pytest.mark.asyncio
async def test_ai_task_entity_rejects_malformed_service_schema_tool_result():
    entry = DummyEntry()
    service_data = {
        "unique_id": "task-service",
        "name": "Groq AI Tasks",
        "model": "openai/gpt-oss-20b",
        "structured_outputs": True,
        "schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    }
    client = DummyStructuredToolTextClient("not json")
    entity = GroqAITaskEntity(DummyHass(), entry, service_data, client)
    task = GenDataTask(
        name="state_summary",
        instructions="Summarize the kitchen light state.",
    )
    chat_log = await async_dummy_tool_chat_log()

    with pytest.raises(HomeAssistantError, match="requested structure"):
        await entity._async_generate_data(task, chat_log)


@pytest.mark.asyncio
async def test_ai_task_tool_request_inserts_system_instruction_without_existing_system():
    entity = GroqAITaskEntity(
        DummyHass(),
        DummyEntry(),
        {"model": "openai/gpt-oss-20b"},
        DummyTextClient("ok"),
    )
    request = await entity._async_tool_generation_request(
        SimpleNamespace(attachments=None),
        SimpleNamespace(content=[], conversation_id="conversation-id"),
        "Return data",
        [{"type": "function", "function": {"name": "GetState"}}],
        "Return JSON.",
    )

    assert request.messages[0] == {"role": "system", "content": "Return JSON."}


@pytest.mark.asyncio
async def test_ai_task_tool_generation_requires_tool_capable_model():
    entity = GroqAITaskEntity(
        DummyHass(),
        DummyEntry(),
        {"model": "whisper-large-v3"},
        DummyTextClient("ok"),
    )

    with pytest.raises(HomeAssistantError, match="tool calls"):
        await entity._async_generate_text_with_tools(
            SimpleNamespace(attachments=None),
            DummyChatLog(),
            "Return data",
            [{"type": "function", "function": {"name": "GetState"}}],
        )


@pytest.mark.asyncio
async def test_ai_task_tool_generation_raises_after_tool_limit():
    class LoopingToolClient:
        def __init__(self):
            self.requests = []

        async def async_generate_text(self, request):
            self.requests.append(request)
            return SimpleNamespace(
                text="",
                model=request.model,
                usage={},
                usage_breakdown=None,
                reasoning=None,
                tool_calls=[
                    {
                        "id": f"call_{len(self.requests)}",
                        "type": "function",
                        "function": {
                            "name": "GetState",
                            "arguments": '{"entity_id":"light.kitchen"}',
                        },
                    }
                ],
                executed_tools=None,
                raw={},
            )

    entity = GroqAITaskEntity(
        DummyHass(),
        DummyEntry(),
        {"model": "openai/gpt-oss-20b"},
        LoopingToolClient(),
    )
    chat_log = DummyToolChatLog()

    with pytest.raises(HomeAssistantError, match="tool-call limit"):
        await entity._async_generate_text_with_tools(
            SimpleNamespace(attachments=None),
            chat_log,
            "Return data",
            [{"type": "function", "function": {"name": "GetState"}}],
        )


@pytest.mark.asyncio
async def test_ai_task_messages_ignores_unreadable_attachment_content():
    entity = GroqAITaskEntity(
        DummyHass(),
        DummyEntry(),
        {"model": "meta-llama/llama-4-scout-17b-16e-instruct"},
        DummyTextClient("ok"),
    )
    task = GenDataTask(
        name="camera_summary",
        instructions="Describe",
        attachments=[SimpleNamespace(mime_type="image/png", path="/tmp/missing.png")],
    )

    with patch(
        "custom_components.groq.ai_task.async_attachment_content_parts",
        return_value=None,
    ):
        assert await entity._async_task_messages(task, task.instructions) is None


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
