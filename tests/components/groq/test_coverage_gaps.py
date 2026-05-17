from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp
import pytest
import voluptuous as vol
import yaml
from homeassistant import data_entry_flow
from homeassistant.components import stt
from homeassistant.components.media_source import Unresolvable
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import SupportsResponse
from homeassistant.exceptions import (
    ConfigEntryAuthFailed,
    ConfigEntryNotReady,
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.helpers.service import SERVICE_DESCRIPTION_CACHE

import custom_components.groq as integration
from custom_components.groq import (
    ai_task as ai_task_module,
    api as api_module,
    config_flow,
    conversation as conversation_module,
    model_registry as model_registry_module,
    repairs as repairs_module,
    text_generation as text_generation_module,
)
from custom_components.groq.ai_task import (
    GroqAITaskEntity,
    _strip_json_fence,
    _structure_description,
)
from custom_components.groq.api import (
    ChatCompletionResult,
    GroqApiClient,
    SpeechRequest,
    StructuredGenerationRequest,
    TextGenerationRequest,
    VisionRequest,
    build_structured_generation_payload,
    build_text_generation_payload,
    build_vision_payload,
    extract_chat_reasoning,
    extract_chat_text,
    extract_executed_tools,
)
from custom_components.groq.const import (
    CONF_API_KEY,
    CONF_ENABLED_FEATURES,
    CONF_MODEL,
    CONF_NAME,
    CONF_SERVICE_TYPE,
    CONF_VOICE,
    DEFAULT_STT_LANGUAGE,
    DOMAIN,
    FEATURE_IMAGE_RECOGNITION,
    FEATURE_SPEECH_TO_TEXT,
    FEATURE_TEXT_GENERATION,
    FEATURE_TEXT_TO_SPEECH,
    UNIQUE_ID,
    enabled_features_from_entry,
    normalize_enabled_features,
    stt_language_default,
    voice_options_for_model,
)
from custom_components.groq.errors import GroqApiError, GroqRateLimitExceeded
from custom_components.groq.feature_registry import (
    GroqFeature,
    GroqFeatureRegistry,
    coerce_feature,
)
from custom_components.groq.flow_schemas import (
    _model_default,
    _requested_max_completion_tokens,
    clean_service_input,
    image_recognition_schema,
    sanitize_text_generation_service_data,
    service_type_schema,
    speech_to_text_schema,
    text_generation_advanced_schema,
    text_generation_basic_schema,
    text_to_speech_schema,
    validate_text_generation_input,
    validate_user_input,
)
from custom_components.groq.model_registry import (
    GroqCapability,
    GroqModel,
    GroqModelRegistry,
    infer_capabilities,
    model_from_api,
)
from custom_components.groq.prompt_cache import GroqPromptCache
from custom_components.groq.rate_limit import (
    GroqRateLimitInfo,
    GroqRateLimiter,
    _duration_seconds,
    _guard_delay_seconds,
)
from custom_components.groq.runtime import (
    async_get_runtime,
    async_hydrate_runtime_model_registry,
    build_runtime,
)
from custom_components.groq.services import (
    ATTR_CAMERA_ENTITY_ID,
    ATTR_AUDIO_FILE,
    ATTR_AUDIO_PATH,
    ATTR_CONFIG_ENTRY_ID,
    ATTR_IMAGE_FILE,
    ATTR_IMAGE_PATH,
    ATTR_IMAGE_URL,
    ATTR_PROMPT,
    ATTR_REASONING_EFFORT,
    ATTR_REFRESH,
    ATTR_REQUEST_BODY_OPTIONS,
    ATTR_SCHEMA,
    ATTR_SERVICE_ID,
    SERVICE_ANALYZE_IMAGE,
    SERVICE_CLEAR_CACHE,
    SERVICE_EXTRACT_TEXT_FROM_IMAGE,
    SERVICE_GENERATE_STRUCTURED,
    SERVICE_GENERATE_TEXT,
    SERVICE_LIST_MODELS,
    SERVICE_TRANSCRIBE_AUDIO,
    _cache_get,
    _cache_key,
    _cache_set,
    _coerce_completion_tokens,
    _entry_from_call,
    _entry_from_service_id,
    _ensure_completion_token_limit,
    _audio_from_call,
    _audio_from_local_path,
    _audio_from_media_source,
    _handle_analyze_image,
    _handle_clear_cache,
    _handle_extract_text_from_image,
    _handle_generate_structured,
    _handle_generate_text,
    _handle_list_models,
    _handle_transcribe_audio,
    _image_data_url,
    _image_from_camera_target,
    _image_from_local_path,
    _image_from_media_source,
    _image_url_from_call,
    _apply_service_options,
    _request_options,
    _runtime_from_call,
    _service_options,
    _service_from_call,
    _service_subentries,
    async_register_services,
    async_unregister_services,
    async_update_service_descriptions,
)
from custom_components.groq.stt import GroqSTTEntity, async_setup_entry as stt_setup
from custom_components.groq.subentries import service_data_by_type
from custom_components.groq.text_generation import (
    entry_value,
    is_prompt_caching_model,
    is_reasoning_model,
    selector_to_json_schema,
    service_include_reasoning,
    service_request_body_options,
    service_schema,
    service_schema_name,
    service_stop,
    service_unique_id,
    voluptuous_schema_to_json_schema,
)
from custom_components.groq.tts import GroqTTSEntity


class DummyEntry:
    def __init__(self, entry_id: str = "entry-id", *, state=ConfigEntryState.LOADED):
        self.entry_id = entry_id
        self.domain = "groq"
        self.state = state
        self.unique_id = "entry-uid"
        self.data = {"api_key": "entry-key", "name": "Groq"}
        self.options = {}
        self.subentries = {}
        self.reloads = []

    def add_update_listener(self, listener):
        self.listener = listener
        return "unsub"

    def async_on_unload(self, unsub):
        self.unsub = unsub


class DummyConfigEntries:
    def __init__(self, entries=()):
        self.entries = {entry.entry_id: entry for entry in entries}
        self.forwarded = []
        self.unloaded = []
        self.updated = []
        self.reloaded = []

    def async_entries(self, domain):
        return list(self.entries.values())

    def async_get_entry(self, entry_id):
        return self.entries.get(entry_id)

    async def async_forward_entry_setups(self, entry, platforms):
        self.forwarded.append((entry, platforms))

    async def async_unload_platforms(self, entry, platforms):
        self.unloaded.append((entry, platforms))
        return True

    async def async_reload(self, entry_id):
        self.reloaded.append(entry_id)

    def async_update_entry(self, entry, **kwargs):
        self.updated.append((entry, kwargs))


class DummyServices:
    def __init__(self):
        self.registered = []
        self.removed = []

    def async_register(self, domain, service, handler, **kwargs):
        self.registered.append((domain, service, handler, kwargs))

    def async_remove(self, domain, service):
        self.removed.append((domain, service))

    def supports_response(self, domain, service):
        return SupportsResponse.ONLY


class DummyHass:
    def __init__(self, entries=(), *, services=None):
        self.data = {}
        self.config_entries = DummyConfigEntries(entries)
        if services is not None:
            self.services = services

    async def async_add_executor_job(self, func, *args):
        return func(*args)


class JsonResponse:
    def __init__(self, status: int, payload, headers: dict[str, str] | None = None):
        self.status = status
        self._payload = payload
        self.headers = headers or {"content-type": "application/json"}

    async def read(self):
        if isinstance(self._payload, bytes):
            return self._payload
        return json.dumps(self._payload).encode()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class StreamContent:
    def __init__(self, lines):
        self._lines = [line.encode() for line in lines]

    def __aiter__(self):
        self._iter = iter(self._lines)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class StreamResponse(JsonResponse):
    def __init__(self, status: int, lines, headers: dict[str, str] | None = None):
        super().__init__(status, {}, headers or {})
        self.content = StreamContent(lines)


class DummySession:
    def __init__(self, responses):
        if not isinstance(responses, list):
            responses = [responses]
        self.responses = list(responses)
        self.calls = []

    def request(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.responses.pop(0)


class PostResponse(JsonResponse):
    def __init__(self, status: int, payload, headers: dict[str, str] | None = None):
        super().__init__(status, payload, headers or {"content-type": "audio/wav"})


class DummyPostSession:
    def __init__(self, responses):
        if not isinstance(responses, list):
            responses = [responses]
        self.responses = list(responses)
        self.calls = []

    def request(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def test_services_yaml_guides_action_inputs():
    """Ensure service action metadata uses constrained selectors where possible."""
    services = yaml.safe_load(
        (Path(__file__).parents[3] / "custom_components/groq/services.yaml").read_text()
    )

    for service_id, service in services.items():
        assert service["name"] and "_" not in service["name"], service_id
        assert service["description"], service_id
        for field_id, field in service["fields"].items():
            assert field["name"], (service_id, field_id)
            assert field["description"], (service_id, field_id)
            assert field["selector"], (service_id, field_id)

    for service_id in ("generate_text", "generate_structured"):
        fields = services[service_id]["fields"]
        assert fields["config_entry_id"]["selector"] == {
            "config_entry": {"integration": "groq"}
        }
        assert "fills this dropdown" in fields["service_id"]["description"]
        assert fields["service_id"]["selector"] == {
            "select": {"custom_value": True, "options": []}
        }
        assert "select" in fields["model"]["selector"]
        assert fields["model"]["selector"]["select"]["custom_value"] is True
        assert "newer Groq model ID" in fields["model"]["description"]
        assert "YAML automations" in fields["stop"]["description"]
        assert fields["service_tier"]["selector"]["select"]["options"] == [
            {"label": "Auto", "value": "auto"},
            {"label": "On Demand", "value": "on_demand"},
            {"label": "Flex", "value": "flex"},
            {"label": "Performance", "value": "performance"},
        ]
        assert "Free-form" in fields["request_body_options"]["description"]
        assert fields["request_body_options"]["selector"] == {"object": None}
    assert services["generate_structured"]["name"] == "Generate Text Output"
    assert services["generate_structured"]["fields"]["schema"]["required"] is False
    assert (
        "Leave empty for plain text"
        in services["generate_structured"]["fields"]["schema"]["description"]
    )
    assert any(
        option["value"] == "llama-3.1-8b-instant"
        for option in services["generate_structured"]["fields"]["model"]["selector"][
            "select"
        ]["options"]
    )

    for service_id in ("analyze_image", "extract_text_from_image"):
        fields = services[service_id]["fields"]
        assert "target" not in services[service_id]
        assert "fills this dropdown" in fields["service_id"]["description"]
        assert fields["service_id"]["selector"] == {
            "select": {"custom_value": True, "options": []}
        }
        assert fields["camera_entity_id"]["selector"] == {
            "entity": {"domain": "camera"}
        }
        assert fields["image_file"]["selector"] == {"media": {"accept": ["image/*"]}}
        assert "allowlist_external_dirs" in fields["image_path"]["description"]
        assert fields["image_url"]["selector"] == {"text": {"type": "url"}}
        assert fields["model"]["selector"]["select"]["custom_value"] is True

    stt_fields = services["transcribe_audio"]["fields"]
    assert stt_fields["service_id"]["selector"] == {
        "select": {"custom_value": True, "options": []}
    }
    assert stt_fields["audio_file"]["selector"] == {"media": {"accept": ["audio/*"]}}
    assert "allowlist_external_dirs" in stt_fields["audio_path"]["description"]
    assert stt_fields["model"]["selector"]["select"]["custom_value"] is True
    assert stt_fields["language"]["selector"]["select"]["custom_value"] is True

    assert services["clear_cache"]["fields"]["config_entry_id"]["selector"] == {
        "config_entry": {"integration": "groq"}
    }
    assert services["list_models"]["fields"]["refresh"]["selector"] == {"boolean": None}


@pytest.mark.asyncio
async def test_service_descriptions_include_dynamic_groq_service_options(monkeypatch):
    """Ensure action UI selectors list configured Groq service subentries."""
    entry = DummyEntry("entry-id")
    entry.title = "Primary Groq"
    entry.subentries = {
        "text": SimpleNamespace(
            subentry_id="text-id",
            data={
                CONF_SERVICE_TYPE: FEATURE_TEXT_GENERATION,
                CONF_NAME: "Assistant",
                CONF_MODEL: "openai/gpt-oss-20b",
            },
        ),
        "vision": SimpleNamespace(
            subentry_id="vision-id",
            data={
                CONF_SERVICE_TYPE: FEATURE_IMAGE_RECOGNITION,
                CONF_NAME: "Driveway Vision",
                CONF_MODEL: "meta-llama/llama-4-scout-17b-16e-instruct",
            },
        ),
        "stt": SimpleNamespace(
            subentry_id="stt-id",
            data={
                CONF_SERVICE_TYPE: FEATURE_SPEECH_TO_TEXT,
                CONF_NAME: "Voice Notes",
                CONF_MODEL: "whisper-large-v3-turbo",
            },
        ),
    }
    second = DummyEntry("second-entry")
    second.title = "Second Groq"
    second.subentries = {
        "text": SimpleNamespace(
            subentry_id="second-text-id",
            data={
                CONF_SERVICE_TYPE: FEATURE_TEXT_GENERATION,
                CONF_NAME: "Notifications",
            },
        )
    }
    hass = DummyHass([entry, second], services=DummyServices())

    await async_update_service_descriptions(hass)

    cache = hass.data[SERVICE_DESCRIPTION_CACHE]
    text_selector = cache[(DOMAIN, SERVICE_GENERATE_TEXT)]["fields"][ATTR_SERVICE_ID][
        "selector"
    ]["select"]
    assert text_selector["custom_value"] is True
    assert text_selector["options"] == [
        {
            "label": "Primary Groq - Assistant (openai/gpt-oss-20b)",
            "value": "text-id",
        },
        {"label": "Second Groq - Notifications", "value": "second-text-id"},
    ]
    image_selector = cache[(DOMAIN, SERVICE_ANALYZE_IMAGE)]["fields"][ATTR_SERVICE_ID][
        "selector"
    ]["select"]
    assert image_selector["options"] == [
        {
            "label": "Primary Groq - Driveway Vision (meta-llama/llama-4-scout-17b-16e-instruct)",
            "value": "vision-id",
        }
    ]
    stt_selector = cache[(DOMAIN, SERVICE_TRANSCRIBE_AUDIO)]["fields"][ATTR_SERVICE_ID][
        "selector"
    ]["select"]
    assert stt_selector["options"] == [
        {
            "label": "Primary Groq - Voice Notes (whisper-large-v3-turbo)",
            "value": "stt-id",
        }
    ]

    await async_update_service_descriptions(hass, exclude_entry_id="entry-id")
    text_selector = cache[(DOMAIN, SERVICE_GENERATE_TEXT)]["fields"][ATTR_SERVICE_ID][
        "selector"
    ]["select"]
    assert text_selector["options"] == [
        {"label": "Second Groq - Notifications", "value": "second-text-id"}
    ]

    assert _service_options(hass, FEATURE_IMAGE_RECOGNITION) == [
        {
            "label": "Primary Groq - Driveway Vision (meta-llama/llama-4-scout-17b-16e-instruct)",
            "value": "vision-id",
        }
    ]
    assert _service_options(hass, FEATURE_SPEECH_TO_TEXT) == [
        {
            "label": "Primary Groq - Voice Notes (whisper-large-v3-turbo)",
            "value": "stt-id",
        }
    ]
    entry.state = ConfigEntryState.SETUP_IN_PROGRESS
    assert _service_options(hass, FEATURE_IMAGE_RECOGNITION) == [
        {
            "label": "Primary Groq - Driveway Vision (meta-llama/llama-4-scout-17b-16e-instruct)",
            "value": "vision-id",
        }
    ]
    with pytest.raises(ServiceValidationError, match="No loaded Groq") as err:
        _entry_from_call(DummyHass([entry]), service_call({}))
    assert err.value.translation_domain == DOMAIN
    assert err.value.translation_key == "no_loaded_config_entry"
    entry.state = ConfigEntryState.LOADED
    assert (
        _entry_from_call(
            hass,
            service_call({ATTR_SERVICE_ID: "second-text-id"}),
            FEATURE_TEXT_GENERATION,
        )
        is second
    )
    assert _entry_from_service_id(hass, FEATURE_TEXT_GENERATION, "missing-id") is None
    duplicate = DummyEntry("duplicate-entry")
    duplicate.subentries = {
        "text": SimpleNamespace(
            subentry_id="text-id",
            data={
                CONF_SERVICE_TYPE: FEATURE_TEXT_GENERATION,
                CONF_NAME: "Assistant",
            },
        )
    }
    with pytest.raises(
        ServiceValidationError, match="Multiple Groq services match"
    ) as err:
        _entry_from_service_id(
            DummyHass([entry, duplicate]),
            FEATURE_TEXT_GENERATION,
            "text-id",
        )
    assert err.value.translation_domain == DOMAIN
    assert err.value.translation_key == "multiple_services_match"
    assert _apply_service_options({"fields": {}}, []) == {"fields": {}}

    # The updater exits quietly when service descriptions are unavailable or
    # the loaded metadata is not shaped like services.yaml.
    await async_update_service_descriptions(SimpleNamespace())
    await async_update_service_descriptions(
        SimpleNamespace(
            data={},
            config_entries=DummyConfigEntries(),
            services=DummyServices(),
        )
    )
    monkeypatch.setattr(
        "custom_components.groq.services.load_yaml",
        lambda path: None,
    )
    await async_update_service_descriptions(hass)
    monkeypatch.setattr(
        "custom_components.groq.services.load_yaml",
        lambda path: {SERVICE_GENERATE_TEXT: None},
    )
    await async_update_service_descriptions(hass)


def test_free_tier_toggle_strings_are_user_facing():
    """Ensure config flow toggle labels never fall back to raw field keys."""
    component_path = Path(__file__).parents[3] / "custom_components/groq"
    for filename in ("strings.json", "translations/en.json"):
        translations = json.loads((component_path / filename).read_text())
        subentries = translations["config_subentries"]
        for service_key in (
            "text_generation",
            "speech_to_text",
            "text_to_speech",
            "image_recognition",
        ):
            step = subentries[service_key]["step"][service_key]
            assert step["data"]["protect_free_tier"] == "Free-Tier Protection"
            assert (
                "Home Assistant pauses only this service"
                in step["data_description"]["protect_free_tier"]
            )


class DummyClient:
    def __init__(self):
        self.requests = []
        self.text = "plain text"
        self.structured = {
            "text": '{"ok": true}',
            "data": {"ok": True},
            "cached": False,
        }
        self.image_text = "image text"
        self.models = []

    async def async_generate_text(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            text=self.text,
            model=request.model,
            usage={},
            reasoning="because",
            executed_tools=None,
            usage_breakdown=None,
        )

    async def async_generate_structured(self, request):
        self.requests.append(request)
        return self.structured

    async def async_analyze_image(self, request):
        self.requests.append(request)
        return SimpleNamespace(text=self.image_text, model=request.model, usage={})

    async def async_list_models(self):
        return self.models

    async def async_transcribe_audio(self, **kwargs):
        self.requests.append(kwargs)
        return "transcribed"


def service_call(data):
    return SimpleNamespace(data=data)


def test_api_payload_and_extractors_cover_optional_shapes():
    result = ChatCompletionResult(text="content", model=None, usage={}, raw={})
    assert result.content == "content"
    assert TextGenerationRequest(prompt="p", model="m").prompt == "p"
    assert (
        build_text_generation_payload(
            TextGenerationRequest(
                prompt="p",
                model="m",
                include_reasoning=True,
                stream=True,
                extra_body={"metadata": {"a": 1}, "drop": None},
            )
        )["include_reasoning"]
        is True
    )
    assert build_structured_generation_payload(
        StructuredGenerationRequest(prompt="p", model="m")
    )["response_format"] == {"type": "json_object"}
    assert build_vision_payload(
        VisionRequest(prompt="p", model="m", image_url="url", system_prompt="s")
    )["messages"][0] == {"role": "system", "content": "s"}

    payload = {"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]}
    assert extract_chat_text(payload) == "a\nb"
    assert (
        extract_chat_reasoning(
            {"choices": [{"message": {"reasoning_content": "trace"}}]}
        )
        == "trace"
    )
    assert extract_chat_reasoning({"choices": []}) is None
    assert extract_chat_reasoning({"choices": ["bad"]}) is None
    assert extract_executed_tools({"choices": []}) is None
    assert (
        extract_executed_tools({"choices": [{"message": {"executed_tools": "bad"}}]})
        is None
    )
    assert extract_executed_tools(
        {"choices": [{"message": {"executed_tools": [{"ok": True}, "bad"]}}]}
    ) == [{"ok": True}]
    assert result.text == "content"

    for bad_payload, match in (
        ({}, "choices"),
        ({"choices": [{}]}, "message"),
        ({"choices": [{"message": {"content": []}}]}, "text content"),
    ):
        with pytest.raises(GroqApiError, match=match):
            extract_chat_text(bad_payload)


@pytest.mark.asyncio
async def test_api_client_covers_json_stream_and_error_paths():
    client = GroqApiClient(
        DummyHass(),
        api_key="entry-key",
        base_url="https://api.groq.com/openai/v1",
        session=DummySession(
            [
                JsonResponse(
                    200, {"choices": [{"message": {"content": '{"ok": true}'}}]}
                ),
                JsonResponse(200, {"choices": [{"message": {"content": "bad json"}}]}),
                JsonResponse(200, {"choices": [{"message": {"content": "img"}}]}),
                JsonResponse(200, {"text": "hello"}),
                JsonResponse(200, {"data": "bad"}),
                JsonResponse(200, []),
                JsonResponse(500, {"error": {"message": "server", "type": "bad"}}),
                JsonResponse(500, {"error": "plain"}),
                JsonResponse(429, {"error": "rate"}, {"retry-after": "2"}),
                JsonResponse(401, {"error": "auth"}),
            ]
        ),
    )

    assert client.base_url == "https://api.groq.com/openai/v1"
    structured = await client.async_generate_structured(
        StructuredGenerationRequest(prompt="p", model="openai/gpt-oss-20b")
    )
    assert structured["data"] == {"ok": True}
    with pytest.raises(GroqApiError, match="valid JSON"):
        await client.async_generate_structured(
            StructuredGenerationRequest(prompt="p", model="openai/gpt-oss-20b")
        )
    assert (
        await client.async_analyze_image(VisionRequest(prompt="p", model="m"))
    ).text == "img"
    assert (
        await client.async_transcribe_audio(
            audio=b"a", filename="a.wav", model="m", language="en-US", prompt="hint"
        )
    ) == "hello"
    with pytest.raises(GroqApiError, match="data list"):
        await client.async_list_models()
    with pytest.raises(GroqApiError, match="non-object JSON"):
        await client.async_generate_text(TextGenerationRequest(prompt="p", model="m"))
    with pytest.raises(GroqApiError, match="server"):
        await client.async_generate_text(TextGenerationRequest(prompt="p", model="m"))
    with pytest.raises(GroqApiError, match="plain"):
        await client.async_generate_text(TextGenerationRequest(prompt="p", model="m"))
    with pytest.raises(GroqRateLimitExceeded):
        await client.async_generate_text(TextGenerationRequest(prompt="p", model="m"))
    with pytest.raises(ConfigEntryAuthFailed):
        await client.async_generate_text(TextGenerationRequest(prompt="p", model="m"))

    non_json_auth_client = GroqApiClient(
        DummyHass(),
        api_key="entry-key",
        session=DummySession(JsonResponse(401, b"<html>")),
    )
    with pytest.raises(ConfigEntryAuthFailed):
        await non_json_auth_client.async_generate_text(
            TextGenerationRequest(prompt="p", model="m")
        )

    stream_client = GroqApiClient(
        DummyHass(),
        api_key=None,
        session=DummySession(
            StreamResponse(
                200,
                [
                    "",
                    "event: ignored",
                    'data: {"choices": []}',
                    'data: {"choices": [{}]}',
                    'data: {"choices": [{"delta": {}}]}',
                    'data: {"choices": [{"delta": {"content": "Hi"}}]}',
                    "data: [DONE]",
                ],
            )
        ),
    )
    chunks = [
        chunk
        async for chunk in stream_client.async_stream_text(
            TextGenerationRequest(prompt="p", model="m", api_key="request-key")
        )
    ]
    assert chunks == ["Hi"]

    with pytest.raises(GroqApiError, match="invalid JSON"):
        GroqApiClient._decode_json(b"{")
    assert GroqApiClient._try_decode_json(b"<html>") is None
    assert isinstance(GroqApiClient._api_error(500, []), GroqApiError)


@pytest.mark.asyncio
async def test_api_client_hydrates_model_limits():
    client = GroqApiClient(
        DummyHass(),
        api_key="key",
        session=DummySession(
            [
                JsonResponse(
                    200,
                    {
                        "data": [
                            {"id": "custom-model"},
                            {"id": "openai/gpt-oss-20b"},
                        ]
                    },
                ),
                JsonResponse(
                    200,
                    {
                        "id": "custom-model",
                        "context_window": 100,
                        "max_completion_tokens": 40,
                    },
                ),
                JsonResponse(
                    200,
                    {
                        "id": "openai/gpt-oss-20b",
                        "context_window": 200,
                        "max_completion_tokens": 80,
                    },
                ),
            ]
        ),
    )

    models = await client.async_list_models()

    assert models[0].context_window == 100
    assert models[0].max_completion_tokens == 40
    assert models[1].context_window == 200
    assert models[1].max_completion_tokens == 80
    assert client._session.calls[1]["args"][1].endswith("/models/custom-model")
    assert client._session.calls[2]["args"][1].endswith("/models/openai%2Fgpt-oss-20b")
    assert (
        await client._async_hydrate_models(
            [GroqModel("ready-model", context_window=10, max_completion_tokens=5)]
        )
    )[0].model_id == "ready-model"
    assert await client._async_hydrate_models([]) == []


@pytest.mark.asyncio
async def test_api_client_keeps_list_model_when_detail_fails():
    client = GroqApiClient(
        DummyHass(),
        api_key="key",
        session=DummySession(
            [
                JsonResponse(
                    200,
                    {"data": [{"id": "custom-model"}, {"id": "list-detail"}]},
                ),
                JsonResponse(500, {"error": "boom"}),
                JsonResponse(200, []),
            ]
        ),
    )

    models = await client.async_list_models()

    assert models[0].model_id == "custom-model"
    assert models[0].max_completion_tokens is None
    assert models[1].model_id == "list-detail"


@pytest.mark.asyncio
async def test_api_stream_error_paths():
    for response, error in (
        (StreamResponse(401, []), ConfigEntryAuthFailed),
        (JsonResponse(500, {"error": "bad"}), GroqApiError),
        (
            JsonResponse(429, {"error": "rate"}, {"retry-after": "1"}),
            GroqRateLimitExceeded,
        ),
        (StreamResponse(200, ["data: {"]), GroqApiError),
    ):
        client = GroqApiClient(
            DummyHass(), api_key=None, session=DummySession(response)
        )
        with pytest.raises(error):
            async for _event in client._request_stream(
                "POST", "/chat/completions", json_payload={}
            ):
                pass


@pytest.mark.asyncio
async def test_api_client_network_error_paths():
    class FailingSession:
        def request(self, *args, **kwargs):
            raise aiohttp.ClientError("boom")

    client = GroqApiClient(DummyHass(), api_key=None, session=FailingSession())
    with pytest.raises(GroqApiError, match="Network error"):
        await client._request_json("GET", "/models")
    with pytest.raises(GroqApiError, match="Network error"):
        async for _event in client._request_stream(
            "POST", "/chat/completions", json_payload={}
        ):
            pass


@pytest.mark.asyncio
async def test_api_client_tracks_availability_and_logs_recovery(caplog):
    client = GroqApiClient(
        DummyHass(),
        api_key="key",
        session=DummySession(
            [
                JsonResponse(500, {"error": {"message": "server"}}),
                JsonResponse(500, {"error": {"message": "server"}}),
                JsonResponse(200, {"data": []}),
            ]
        ),
    )

    with pytest.raises(GroqApiError):
        await client.async_list_models()
    assert client.available is False

    with pytest.raises(GroqApiError):
        await client.async_list_models()
    assert caplog.text.count("Groq API returned HTTP 500") == 1

    assert await client.async_list_models() == []
    assert client.available is True
    assert "Groq API is reachable again" in caplog.text


@pytest.mark.asyncio
async def test_api_client_creates_repair_for_model_access_error(monkeypatch):
    issues = []
    monkeypatch.setattr(
        "custom_components.groq.api.async_create_model_access_issue",
        lambda hass, model: issues.append((hass, model)),
    )
    client = GroqApiClient(
        DummyHass(),
        api_key="key",
        session=DummySession(
            JsonResponse(
                400,
                {"error": {"message": "The model custom/missing is not available"}},
            )
        ),
    )

    with pytest.raises(GroqApiError):
        await client.async_generate_text(
            TextGenerationRequest(prompt="hello", model="custom/missing")
        )

    assert issues == [(client._hass, "custom/missing")]


def test_api_client_internal_availability_and_model_access_branches(monkeypatch):
    client = GroqApiClient(DummyHass(), api_key="key", session=DummySession([]))
    client._handle_http_unavailable(408, {"error": "timeout"})
    assert client.available is False
    assert client._unavailable_reason == "Groq API request timed out"

    issues = []
    monkeypatch.setattr(
        "custom_components.groq.api.async_create_model_access_issue",
        lambda hass, model: issues.append((hass, model)),
    )
    client._create_model_access_issue(
        400,
        {"error": {"message": "model is not available"}},
        {"model": 123},
    )
    assert issues == []

    assert api_module._payload_mentions_model_access([]) is False
    assert (
        api_module._payload_mentions_model_access(
            {"error": "Model custom/missing does not exist"}
        )
        is True
    )


@pytest.mark.asyncio
async def test_api_client_cancel_and_transcription_shape_paths():
    class CancelSession:
        def request(self, *args, **kwargs):
            raise asyncio.CancelledError

    client = GroqApiClient(DummyHass(), api_key=None, session=CancelSession())
    with pytest.raises(asyncio.CancelledError):
        await client._request_json("GET", "/models")
    with pytest.raises(asyncio.CancelledError):
        async for _event in client._request_stream(
            "POST", "/chat/completions", json_payload={}
        ):
            pass

    transcribe_client = GroqApiClient(
        DummyHass(),
        api_key=None,
        session=DummySession(JsonResponse(200, {"not_text": True})),
    )
    with pytest.raises(GroqApiError, match="transcription response"):
        await transcribe_client.async_transcribe_audio(
            audio=b"a",
            filename="a.wav",
            model="whisper-large-v3",
        )

    stream_client = GroqApiClient(
        DummyHass(),
        api_key=None,
        session=DummySession(
            StreamResponse(
                200,
                [
                    'data: {"choices": ["bad"]}',
                    'data: {"choices": [{"delta": {"content": "ok"}}]}',
                    "data: [DONE]",
                ],
            )
        ),
    )
    chunks = [
        chunk
        async for chunk in stream_client.async_stream_text(
            TextGenerationRequest(prompt="p", model="m")
        )
    ]
    assert chunks == ["ok"]


def test_const_errors_features_cache_and_rate_limit_helpers():
    assert normalize_enabled_features("text_generation") == ["text_generation"]
    assert normalize_enabled_features(1) == list(("text_to_speech",))
    entry = DummyEntry()
    entry.options = {CONF_ENABLED_FEATURES: ["speech_to_text"]}
    assert enabled_features_from_entry(entry) == ["speech_to_text"]
    entry.options = {}
    entry.data = {CONF_ENABLED_FEATURES: ["image_recognition"]}
    assert enabled_features_from_entry(entry) == ["image_recognition"]
    entry.data = {}
    entry.subentries = {
        "vision": SimpleNamespace(data={CONF_SERVICE_TYPE: FEATURE_IMAGE_RECOGNITION})
    }
    assert enabled_features_from_entry(entry) == [FEATURE_IMAGE_RECOGNITION]
    assert (
        _requested_max_completion_tokens(
            {"request_body_options": {"max_completion_tokens": ["not", "scalar"]}}
        )
        == []
    )
    entry.data = {"url": "u", "model": "m", "voice": "v"}
    entry.subentries = {}
    assert enabled_features_from_entry(entry) == [FEATURE_TEXT_TO_SPEECH]

    err = GroqApiError("bad", status=500, error_type="server", payload={"x": 1})
    assert err.status == 500
    assert err.error_type == "server"
    assert err.payload == {"x": 1}
    rate = GroqRateLimitExceeded(
        "limited",
        retry_after="1",
        reset_requests="2",
        reset_tokens="3",
        payload={"error": "rate"},
    )
    assert rate.status == 429
    assert rate.retry_after == "1"
    assert rate.reset_requests == "2"
    assert rate.reset_tokens == "3"

    assert coerce_feature(GroqFeature.TEXT_GENERATION) == GroqFeature.TEXT_GENERATION
    with pytest.raises(Exception):
        GroqFeatureRegistry([]).ensure_enabled(GroqFeature.TEXT_GENERATION)

    cache = GroqPromptCache(max_size=0)
    cache.set("x", {"value": 1})
    assert cache.get("x") is None
    expired = GroqPromptCache(max_size=2, default_ttl=0)
    expired.set("x", {"value": 1})
    assert expired.get("x") is None

    info = GroqRateLimitInfo.from_headers(
        {
            "Retry-After": "1",
            "x-ratelimit-limit-requests": "10",
            "x-ratelimit-limit-tokens": "20",
            "x-ratelimit-remaining-requests": "9",
            "x-ratelimit-remaining-tokens": "19",
            "x-ratelimit-reset-requests": "1s",
            "x-ratelimit-reset-tokens": "2s",
        }
    )
    assert info.as_dict()["limit_requests"] == "10"
    assert GroqRateLimiter.from_headers({"retry-after": "1"}).retry_after == "1"
    assert "retry after 1 seconds" in info.error_message()
    with pytest.raises(GroqRateLimitExceeded):
        GroqRateLimiter.raise_for_headers({"retry-after": "1"}, {"error": "rate"})
    limiter = GroqRateLimiter()
    limiter.raise_if_blocked(None)
    limiter.raise_if_blocked("service")
    limiter.update_from_headers(None, {"retry-after": "1"})
    limiter.update_from_headers("service", {})
    limiter.update_from_headers("service", {"retry-after": "2"})
    with pytest.raises(GroqRateLimitExceeded, match="free-tier guard"):
        limiter.raise_if_blocked("service")
    limiter._blocked_until["expired"] = 0
    limiter.raise_if_blocked("expired")
    assert "expired" not in limiter._blocked_until
    assert _guard_delay_seconds(GroqRateLimitInfo(retry_after="1.5")) == 2
    assert (
        _guard_delay_seconds(
            GroqRateLimitInfo(
                remaining_requests="0",
                reset_requests="2m",
                reset_tokens="1s",
            )
        )
        == 120
    )
    assert _guard_delay_seconds(GroqRateLimitInfo(remaining_tokens="0")) == 60
    assert _guard_delay_seconds(GroqRateLimitInfo(remaining_requests="1")) is None
    assert _duration_seconds(None) is None
    assert _duration_seconds("250ms") == 1
    assert _duration_seconds("2h") == 7200
    assert _duration_seconds("bad") is None
    assert _duration_seconds("badms") is None


def test_repair_issue_helpers_create_sanitized_issues(monkeypatch):
    created = []
    deleted = []
    monkeypatch.setattr(
        repairs_module.ir,
        "async_create_issue",
        lambda *args, **kwargs: created.append((args, kwargs)),
    )
    monkeypatch.setattr(
        repairs_module.ir,
        "async_delete_issue",
        lambda *args, **kwargs: deleted.append((args, kwargs)),
    )
    entry = DummyEntry()

    repairs_module.async_create_ffmpeg_missing_issue(DummyHass(), entry)
    repairs_module.async_create_model_access_issue(
        DummyHass(),
        "m" * 200,
        service_id="svc",
    )
    repairs_module.async_create_model_configuration_issue(
        DummyHass(),
        entry,
        {UNIQUE_ID: "service-uid"},
        "vision-model",
        "image recognition",
    )
    repairs_module.async_delete_ffmpeg_missing_issue(
        DummyHass(),
        entry,
        {UNIQUE_ID: "service-uid"},
    )

    assert [kwargs["translation_key"] for _args, kwargs in created] == [
        repairs_module.ISSUE_FFMPEG_MISSING,
        repairs_module.ISSUE_MODEL_ACCESS,
        repairs_module.ISSUE_MODEL_CONFIGURATION,
    ]
    assert created[0][1]["translation_placeholders"] == {"service_name": "Groq"}
    assert len(created[1][1]["translation_placeholders"]["model"]) == 128
    assert created[2][1]["translation_placeholders"] == {
        "service_name": "service-uid",
        "model": "vision-model",
        "feature": "image recognition",
    }
    assert deleted


@pytest.mark.asyncio
async def test_config_flow_fetch_models_uses_lightweight_list(monkeypatch):
    session = DummySession(
        [
            JsonResponse(
                200,
                {
                    "data": [
                        {"id": "custom/model"},
                        {"id": "other-model"},
                        {"id": "bad-detail"},
                        {"id": "openai/gpt-oss-20b"},
                    ]
                },
            ),
            JsonResponse(
                200,
                {
                    "id": "custom/model",
                    "context_window": 100,
                    "max_completion_tokens": 40,
                },
            ),
            JsonResponse(
                200,
                {
                    "id": "other-model",
                    "context_window": 90,
                    "max_completion_tokens": 30,
                },
            ),
            JsonResponse(500, {"error": "boom"}),
            JsonResponse(
                200,
                {
                    "id": "openai/gpt-oss-20b",
                    "context_window": 200,
                    "max_completion_tokens": 80,
                },
            ),
        ]
    )
    monkeypatch.setattr(config_flow, "async_get_clientsession", lambda _hass: session)

    models = await config_flow.async_fetch_available_models(DummyHass(), "key")

    assert models[0].completion_token_limit is None
    assert models[1].completion_token_limit is None
    assert models[2].model_id == "bad-detail"
    assert models[3].completion_token_limit == 65536
    assert len(session.calls) == 1
    assert session.calls[0]["kwargs"]["timeout"].total == 10


@pytest.mark.asyncio
async def test_config_flow_fetch_models_keeps_model_when_detail_errors(monkeypatch):
    class ErrorSession:
        def __init__(self):
            self.calls = 0

        def request(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return JsonResponse(200, {"data": [{"id": "custom-model"}]})
            raise aiohttp.ClientError("boom")

    session = ErrorSession()
    monkeypatch.setattr(config_flow, "async_get_clientsession", lambda _hass: session)

    models = await config_flow.async_fetch_available_models(DummyHass(), "key")

    assert models[0].model_id == "custom-model"
    assert models[0].max_completion_tokens is None


@pytest.mark.asyncio
async def test_config_flow_dynamic_model_and_locale_fallback_branches(monkeypatch):
    async def empty_models(_hass, _api_key):
        return []

    monkeypatch.setattr(config_flow, "async_fetch_available_models", empty_models)
    assert (
        await config_flow.async_get_model_registry(DummyHass(), "key")
    ).models_for_feature(GroqFeature.TEXT_GENERATION)

    async def broken_models(_hass, _api_key):
        raise RuntimeError("models unavailable")

    monkeypatch.setattr(config_flow, "async_fetch_available_models", broken_models)
    assert (
        await config_flow.async_get_model_registry(DummyHass(), "key")
    ).models_for_feature(GroqFeature.TEXT_GENERATION)

    async def value_error_models(_hass, _api_key):
        raise ValueError("invalid_auth")

    monkeypatch.setattr(config_flow, "async_fetch_available_models", value_error_models)
    with pytest.raises(ValueError):
        await config_flow.async_get_model_registry(DummyHass(), "key")

    async def type_error_models(_hass, _api_key):
        raise TypeError("bad payload")

    monkeypatch.setattr(config_flow, "async_fetch_available_models", type_error_models)
    assert await config_flow.async_validate_api_key(DummyHass(), "key") == "unknown"

    async def client_error_models(_hass, _api_key):
        raise aiohttp.ClientError("network down")

    monkeypatch.setattr(
        config_flow, "async_fetch_available_models", client_error_models
    )
    assert (
        await config_flow.async_validate_api_key(DummyHass(), "key") == "cannot_connect"
    )

    assert voice_options_for_model(None)
    assert "aisha" in voice_options_for_model("custom-arabic-orpheus")
    assert "troy" in voice_options_for_model("custom-orpheus")
    assert voice_options_for_model("custom-model")
    assert stt_language_default(None) == DEFAULT_STT_LANGUAGE
    assert stt_language_default("en_AU") == "en"
    assert stt_language_default("es-MX") == "es-ES"
    assert stt_language_default("xx-YY") == DEFAULT_STT_LANGUAGE

    flow = config_flow.GroqServiceSubentryFlow()
    flow.hass = DummyHass()
    monkeypatch.setattr(
        flow,
        "async_show_form",
        lambda **kwargs: {"type": "form", **kwargs},
    )
    model_registry_calls = 0

    async def counted_model_registry(_hass, _api_key):
        nonlocal model_registry_calls
        model_registry_calls += 1
        return GroqModelRegistry()

    monkeypatch.setattr(config_flow, "async_get_model_registry", counted_model_registry)
    result = await flow.async_step_text_to_speech(
        {
            CONF_NAME: "TTS",
            CONF_MODEL: "canopylabs/orpheus-v1-english",
            CONF_VOICE: "aisha",
        }
    )
    assert result["type"] == "form"
    assert result["errors"] == {CONF_VOICE: "invalid_voice"}
    await flow._model_registry()
    assert model_registry_calls == 1

    async def raise_value_registry(_hass, _api_key):
        raise ValueError("invalid_auth")

    monkeypatch.setattr(config_flow, "async_get_model_registry", raise_value_registry)
    failed_flow = config_flow.GroqServiceSubentryFlow()
    failed_flow.hass = DummyHass()
    registry = await failed_flow._model_registry()
    assert registry.models_for_feature(GroqFeature.TEXT_GENERATION)


def test_flow_schema_and_text_generation_helpers_cover_branches():
    assert _model_default({}, CONF_MODEL, "fallback", []) == "fallback"
    assert _model_default({}, CONF_MODEL, "fallback", ["first"]) == "first"
    assert service_type_schema()({"service_type": FEATURE_TEXT_GENERATION})
    assert (
        speech_to_text_schema()({"name": "STT", "model": "whisper-large-v3"})[
            "language"
        ]
        == "en-US"
    )
    assert (
        speech_to_text_schema(default_language="fr-FR")(
            {"name": "STT", "model": "whisper-large-v3"}
        )["language"]
        == "fr-FR"
    )
    assert text_to_speech_schema()(
        {
            "name": "TTS",
            "model": "canopylabs/orpheus-v1-english",
            "voice": "troy",
        }
    )
    assert image_recognition_schema()(
        {
            "name": "Vision",
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        }
    )
    assert text_generation_basic_schema()(
        {"name": "Text", "model": "llama-3.1-8b-instant"}
    )
    registry = GroqModelRegistry()
    assert text_generation_advanced_schema(
        {"model": "groq/compound"},
        registry,
    )
    assert validate_text_generation_input(
        {"model": "groq/compound", "max_tokens": 8193},
        registry,
    ) == {"max_tokens": "max_completion_tokens_exceeded"}
    assert validate_text_generation_input(
        {
            "model": "groq/compound",
            "request_body_options": {"max_completion_tokens": 8193},
        },
        registry,
    ) == {"request_body_options": "max_completion_tokens_exceeded"}
    assert validate_text_generation_input(
        {
            "model": "llama-3.1-8b-instant",
            "request_body_options": {
                "response_format": {"type": "json_schema", "json_schema": {}}
            },
        },
        registry,
    ) == {"request_body_options": "unsupported_structured_outputs_model"}
    assert validate_text_generation_input(
        {
            "model": "llama-3.1-8b-instant",
            "request_body_options": {"reasoning_effort": "low"},
        },
        registry,
    ) == {"request_body_options": "unsupported_reasoning_model"}
    assert validate_text_generation_input(
        {
            "model": "groq/compound",
            "request_body_options": {"model": "override-model"},
        },
        registry,
    ) == {"request_body_options": "reserved_request_body_option"}
    assert validate_text_generation_input(
        {
            "model": "openai/gpt-oss-20b",
            "request_body_options": {"tools": []},
        },
        registry,
    ) == {"request_body_options": "reserved_request_body_option"}
    assert "integration-managed fields" in (
        text_generation_module.request_body_options_error_message(
            registry,
            "groq/compound",
            {"tool_choice": "auto"},
        )
        or ""
    )
    assert (
        validate_text_generation_input(
            {
                "model": "llama-3.1-8b-instant",
                "request_body_options": {"response_format": {"type": "text"}},
            },
            registry,
        )
        == {}
    )
    assert (
        text_generation_module.request_body_options_validation_error(
            registry,
            "llama-3.1-8b-instant",
            {"response_format": None},
        )
        is None
    )
    assert (
        text_generation_module.request_body_options_validation_error(
            registry,
            "llama-3.1-8b-instant",
            {"response_format": "json_schema"},
        )
        == "unsupported_structured_outputs_model"
    )
    assert (
        validate_text_generation_input(
            {
                "model": "groq/compound",
                "request_body_options": {"max_tokens": "not-a-number"},
            },
            registry,
        )
        == {}
    )
    sanitized = sanitize_text_generation_service_data(
        {
            "model": "groq/compound",
            "max_tokens": 9000,
            "reasoning_effort": "low",
            "prompt_caching": True,
            "structured_outputs": True,
            "schema_name": "response",
            "schema": {"type": "object"},
            "strict": True,
            "request_body_options": {
                "max_completion_tokens": 9000,
                "max_tokens": "not-a-number",
                "response_format": {"type": "json_object"},
                "reasoning_effort": "low",
                "user": "ha",
            },
        },
        registry,
    )
    assert sanitized["max_tokens"] == 8192
    assert sanitized["request_body_options"] == {
        "max_completion_tokens": 8192,
        "user": "ha",
    }
    for key in (
        "reasoning_effort",
        "prompt_caching",
        "structured_outputs",
        "schema_name",
        "schema",
        "strict",
    ):
        assert key not in sanitized
    assert (
        sanitize_text_generation_service_data(
            {"model": "unknown-model", "max_tokens": 10},
            registry,
        )["max_tokens"]
        == 10
    )
    invalid_tokens = sanitize_text_generation_service_data(
        {"model": "groq/compound", "max_tokens": "not-a-number"},
        registry,
    )
    assert "max_tokens" not in invalid_tokens
    assert sanitize_text_generation_service_data(
        {
            "model": "llama-3.1-8b-instant",
            "request_body_options": {"response_format": None},
        },
        registry,
    )["request_body_options"] == {"response_format": None}
    assert "request_body_options" not in sanitize_text_generation_service_data(
        {
            "model": "llama-3.1-8b-instant",
            "request_body_options": {"reasoning_effort": "low"},
        },
        registry,
    )
    assert validate_text_generation_input({"reasoning_effort": "low"}, registry) == {}
    assert _coerce_completion_tokens("not-a-number") is None
    _ensure_completion_token_limit(
        SimpleNamespace(model_registry=registry),
        TextGenerationRequest(prompt="p", model="unknown-model", max_tokens=999999),
    )
    tiny_registry = GroqModelRegistry(
        [
            GroqModel(
                "tiny-text",
                context_window=40,
                max_completion_tokens=10,
                capabilities=frozenset({GroqCapability.TEXT_GENERATION}),
            )
        ],
        include_built_ins=False,
    )
    assert (
        text_generation_module.request_context_window_error(
            tiny_registry,
            TextGenerationRequest(prompt="short", model="tiny-text", max_tokens=1),
        )
        is None
    )
    assert "40 token context window" in (
        text_generation_module.request_context_window_error(
            tiny_registry,
            TextGenerationRequest(
                prompt="x" * 95,
                model="tiny-text",
                max_tokens=10,
                extra_body={"metadata": {"source": "test"}},
            ),
        )
        or ""
    )
    assert (
        text_generation_module.request_context_window_error(
            GroqModelRegistry(include_built_ins=False),
            TextGenerationRequest(prompt="p", model="unknown-model"),
        )
        is None
    )
    assert text_generation_module._payload_token_upper_bound({"bad": object()}) == 0
    assert text_generation_module._payload_token_upper_bound(None) == 0
    assert (
        text_generation_module.request_context_window_error(
            tiny_registry,
            TextGenerationRequest(
                prompt="short",
                model="tiny-text",
                max_tokens=object(),
            ),
        )
        is None
    )
    assert clean_service_input(
        {
            "api_key": "",
            "advanced_options": False,
            "reasoning_effort": "",
            "request_body_options": {},
            "schema": {},
            "include_reasoning": False,
        }
    ) == {"advanced_options": False}
    with pytest.raises(ValueError, match="invalid"):
        asyncio.run(validate_user_input({"api_key": "key", "enabled_features": 1}))
    asyncio.run(
        validate_user_input(
            {"api_key": "key", "enabled_features": FEATURE_TEXT_GENERATION}
        )
    )

    entry = DummyEntry()
    entry.unique_id = None
    service_data = {
        "stop": "A\nB",
        "include_reasoning": True,
        "request_body_options": {"user": "ha"},
        "schema": {"type": "object"},
        "schema_name": "bad name!",
        "temperature": None,
        "max_tokens": "",
        "top_p": "",
        "seed": "",
        "service_tier": "",
        "reasoning_effort": "low",
        "reasoning_format": "parsed",
        "stream": True,
        "prompt_caching": True,
        "structured_outputs": True,
        "strict": True,
    }
    assert entry_value(entry, {}, "missing", "fallback") == "fallback"
    assert text_generation_module.text_generation_service_data(entry) == []
    assert text_generation_module.service_name(entry, {}) == "Groq"
    assert text_generation_module.service_model(entry, {}) == "openai/gpt-oss-20b"
    assert text_generation_module.service_system_prompt(entry, {"system_prompt": ""})
    assert text_generation_module.service_temperature(entry, service_data) is None
    assert text_generation_module.service_max_tokens(entry, service_data) is None
    assert text_generation_module.service_max_tokens(entry, {"max_tokens": "10"}) == 10
    assert (
        text_generation_module.service_max_tokens(
            entry,
            {"model": "groq/compound", "max_tokens": "9000"},
            GroqModelRegistry(),
        )
        == 8192
    )
    assert text_generation_module.service_top_p(entry, service_data) is None
    assert text_generation_module.service_top_p(entry, {"top_p": "0.5"}) == 0.5
    assert service_stop(entry, service_data) == ["A", "B"]
    assert service_stop(entry, {"stop": ["A", "", "B"]}) == ["A", "B"]
    assert service_stop(entry, {"stop": "A"}) == "A"
    assert service_stop(entry, {"stop": ""}) is None
    assert text_generation_module.service_seed(entry, service_data) is None
    assert text_generation_module.service_seed(entry, {"seed": "123"}) == 123
    assert text_generation_module.service_service_tier(entry, service_data) is None
    assert text_generation_module.service_reasoning_effort(entry, service_data) == "low"
    assert (
        text_generation_module.service_reasoning_format(entry, service_data) == "parsed"
    )
    assert service_include_reasoning(entry, service_data) is True
    assert text_generation_module.service_stream(entry, service_data) is True
    assert text_generation_module.service_stream(entry, {}) is True
    assert text_generation_module.service_prompt_caching(entry, service_data) is True
    assert service_request_body_options(entry, service_data) == {"user": "ha"}
    assert service_request_body_options(
        entry,
        {
            "model": "groq/compound",
            "request_body_options": {
                "max_completion_tokens": "9000",
                "max_tokens": "bad",
            },
        },
        GroqModelRegistry(),
    ) == {"max_completion_tokens": 8192, "max_tokens": "bad"}
    assert service_request_body_options(
        entry,
        {
            "model": "unknown-model",
            "request_body_options": {"max_completion_tokens": 9000},
        },
        GroqModelRegistry(),
    ) == {"max_completion_tokens": 9000}
    assert service_request_body_options(
        entry,
        {
            "model": "groq/compound",
            "request_body_options": {"max_completion_tokens": ""},
        },
        GroqModelRegistry(),
    ) == {"max_completion_tokens": ""}
    assert service_request_body_options(entry, {}) is None
    assert service_schema(entry, service_data) == {"type": "object"}
    assert service_schema(entry, {}) is None
    assert service_schema_name(entry, service_data) == "bad_name"
    assert (
        text_generation_module.service_structured_outputs(entry, service_data) is True
    )
    assert text_generation_module.service_strict(entry, service_data) is True
    assert service_unique_id(entry, {}) == "entry-id"
    assert service_unique_id(DummyEntry(), {UNIQUE_ID: "service-id"}) == "service-id"
    assert is_reasoning_model("openai/gpt-oss-20b")
    assert is_prompt_caching_model("openai/gpt-oss-20b")

    assert selector_to_json_schema(SimpleNamespace(selector_type="text")) == {
        "type": "string"
    }
    assert selector_to_json_schema(SimpleNamespace(selector_type="boolean")) == {
        "type": "boolean"
    }
    assert selector_to_json_schema(
        SimpleNamespace(selector_type="number", config={"min": 1, "max": 5})
    ) == {"type": "number", "minimum": 1, "maximum": 5}
    assert selector_to_json_schema(
        SimpleNamespace(
            selector_type="select",
            config={"options": [{"value": "a"}, "b"], "multiple": True},
        )
    ) == {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}}
    assert selector_to_json_schema(
        SimpleNamespace(selector_type="select", config={"options": ["a"]})
    ) == {"type": "string", "enum": ["a"]}
    assert selector_to_json_schema(SimpleNamespace(selector_type="object")) == {
        "type": "object"
    }
    assert selector_to_json_schema(SimpleNamespace(selector_type="unknown")) == {}
    assert selector_to_json_schema(str) == {"type": "string"}
    assert selector_to_json_schema(bool) == {"type": "boolean"}
    assert selector_to_json_schema(int) == {"type": "integer"}
    assert selector_to_json_schema(float) == {"type": "number"}
    assert selector_to_json_schema(vol.Schema({vol.Required("a"): str})) == {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": False,
        "required": ["a"],
    }
    assert (
        voluptuous_schema_to_json_schema(
            vol.Schema({vol.Required("a", description="A value"): str})
        )["properties"]["a"]["description"]
        == "A value"
    )
    assert voluptuous_schema_to_json_schema(vol.Schema(str)) == {}


def test_model_registry_branches():
    infer_capabilities.cache_clear()
    assert infer_capabilities("custom-whisper") == frozenset(
        {GroqCapability.SPEECH_TO_TEXT}
    )
    assert infer_capabilities("custom-orpheus") == frozenset(
        {GroqCapability.TEXT_TO_SPEECH}
    )
    assert infer_capabilities("custom-tts") == frozenset()
    assert GroqCapability.VISION in infer_capabilities("custom-vision-model")
    assert GroqCapability.COMPOUND in infer_capabilities("groq/compound-custom")
    with (
        patch.object(model_registry_module, "REASONING_MODELS", {"custom-reason"}),
        patch.object(model_registry_module, "PROMPT_CACHING_MODELS", {"custom-cache"}),
        patch.object(
            model_registry_module,
            "STRUCTURED_OUTPUTS_MODELS",
            {"custom-structured"},
        ),
    ):
        infer_capabilities.cache_clear()
        assert GroqCapability.REASONING in infer_capabilities("custom-reason")
        assert GroqCapability.PROMPT_CACHING in infer_capabilities("custom-cache")
        assert GroqCapability.STRUCTURED_OUTPUTS in infer_capabilities(
            "custom-structured"
        )
    infer_capabilities.cache_clear()
    model = model_from_api(
        {
            "id": "custom-model",
            "active": False,
            "owned_by": "me",
            "context_window": 10,
            "max_completion_tokens": 5,
        }
    )
    registry = GroqModelRegistry([model])
    assert registry.get("custom-model") is model
    assert registry.completion_token_limit("custom-model") == 5
    assert registry.context_window("custom-model") == 10
    assert registry.context_window("missing-model") is None
    assert registry.completion_token_limit("missing-model") is None
    assert (
        model_registry_module.BUILT_IN_MODELS["qwen/qwen3-32b"].completion_token_limit
        == 40960
    )
    assert (
        model_registry_module.BUILT_IN_MODELS[
            "meta-llama/llama-4-maverick-17b-128e-instruct"
        ].as_dict()["completion_token_limit"]
        == 8192
    )
    assert registry.all_models()[0].model_id
    assert registry.models_for_feature(GroqFeature.TEXT_GENERATION)
    discovered_only = GroqModelRegistry([model], include_built_ins=False)
    assert discovered_only.models_for_feature(GroqFeature.TEXT_GENERATION) == []
    with patch.dict(
        model_registry_module.FEATURE_CAPABILITIES,
        {GroqFeature.TEXT_GENERATION: frozenset()},
    ):
        assert registry.models_for_feature(GroqFeature.TEXT_GENERATION)
        assert registry.supports("custom-model", GroqFeature.TEXT_GENERATION)
    assert registry.supports("custom-vision-model", GroqCapability.VISION)
    assert registry.supports("whatever", GroqFeature.PROMPT_CACHING) is False


@pytest.mark.asyncio
async def test_runtime_and_integration_lifecycle_branches(monkeypatch):
    async def fake_list_models(self, **_kwargs):
        return []

    monkeypatch.setattr(GroqApiClient, "async_list_models", fake_list_models)
    services = DummyServices()
    entry = DummyEntry()
    entry.subentries = {
        "unsupported": SimpleNamespace(data={"service_type": "unknown"}),
        "text": SimpleNamespace(data={"service_type": "text_generation"}),
    }
    hass = DummyHass([entry], services=services)
    assert await integration.async_setup(hass, {}) is True
    assert services.registered
    assert await integration.async_setup_entry(hass, entry) is True
    assert entry.runtime_data is await async_get_runtime(hass, entry)
    assert await integration.async_unload_entry(hass, entry) is True
    assert services.removed == []

    other = DummyEntry("other")
    services2 = DummyServices()
    hass2 = DummyHass([entry, other], services=services2)
    await integration.async_setup(hass2, {})
    await integration.async_setup_entry(hass2, entry)
    await integration.async_unload_entry(hass2, entry)
    assert services2.removed == []

    legacy = DummyEntry()
    legacy.unique_id = None
    legacy.data = {"unique_id": "old", "url": "u", "model": "m", "voice": "v"}
    assert await integration.async_migrate_entry(hass, legacy)
    assert hass.config_entries.updated[-1][1]["unique_id"] == "old"
    assert await integration.async_migrate_entry(hass, entry)
    no_legacy = DummyEntry()
    no_legacy.unique_id = None
    no_legacy.data = {}
    assert await integration.async_migrate_entry(hass, no_legacy)
    assert not integration._has_other_loaded_entries(
        SimpleNamespace(config_entries=SimpleNamespace()),
        entry,
    )
    assert not integration._has_other_loaded_entries(DummyHass([entry]), entry)
    assert not integration._has_other_loaded_entries(
        DummyHass([entry, DummyEntry("unloaded", state=ConfigEntryState.NOT_LOADED)]),
        entry,
    )
    assert integration._has_other_loaded_entries(
        DummyHass([entry, DummyEntry("loaded")]),
        entry,
    )

    no_runtime_entry = DummyEntry()
    no_runtime_entry.data = {"url": "u", "model": "m", "voice": "v"}
    runtime = build_runtime(DummyHass(), no_runtime_entry)
    assert runtime.feature_registry.is_enabled(GroqFeature.TEXT_TO_SPEECH)

    class RuntimeSetterRaises(DummyEntry):
        @property
        def runtime_data(self):
            return None

        @runtime_data.setter
        def runtime_data(self, value):
            raise AttributeError

    assert isinstance(
        await async_get_runtime(DummyHass(), RuntimeSetterRaises()),
        type(runtime),
    )

    async def setup_unavailable(self, **_kwargs):
        raise GroqApiError("Network error calling Groq API: down")

    unavailable_entry = DummyEntry("unavailable")
    monkeypatch.setattr(GroqApiClient, "async_list_models", setup_unavailable)
    with pytest.raises(ConfigEntryNotReady):
        await integration.async_setup_entry(
            DummyHass([unavailable_entry]), unavailable_entry
        )


@pytest.mark.asyncio
async def test_runtime_model_registry_hydration_branches():
    entry = DummyEntry()
    runtime = build_runtime(DummyHass(), entry)

    async def dynamic_models(**_kwargs):
        return [
            GroqModel(
                model_id="custom/dynamic",
                context_window=2048,
                max_completion_tokens=1024,
            )
        ]

    runtime.client.async_list_models = dynamic_models
    await async_hydrate_runtime_model_registry(entry, runtime)
    assert runtime.model_registry.context_window("custom/dynamic") == 2048

    no_key_entry = DummyEntry()
    no_key_entry.data = {}
    no_key_runtime = build_runtime(DummyHass(), no_key_entry)

    async def should_not_run(**_kwargs):
        raise AssertionError("model hydration should be skipped without an API key")

    no_key_runtime.client.async_list_models = should_not_run
    await async_hydrate_runtime_model_registry(no_key_entry, no_key_runtime)

    async def cannot_connect(**_kwargs):
        raise GroqApiError("down")

    runtime.client.async_list_models = cannot_connect
    await async_hydrate_runtime_model_registry(entry, runtime)
    with pytest.raises(ConfigEntryNotReady):
        await async_hydrate_runtime_model_registry(
            entry,
            runtime,
            raise_not_ready=True,
        )

    async def timed_out(**_kwargs):
        raise TimeoutError("slow")

    runtime.client.async_list_models = timed_out
    await async_hydrate_runtime_model_registry(entry, runtime)

    async def invalid_auth(**_kwargs):
        raise ConfigEntryAuthFailed("bad key")

    runtime.client.async_list_models = invalid_auth
    with pytest.raises(ConfigEntryAuthFailed):
        await async_hydrate_runtime_model_registry(entry, runtime)


@pytest.mark.asyncio
async def test_config_flow_remaining_paths(monkeypatch):
    assert config_flow.generate_entry_id()
    assert isinstance(
        config_flow.GroqConfigFlow.async_get_options_flow(DummyEntry()),
        config_flow.GroqOptionsFlow,
    )

    duplicate_entry = DummyEntry("duplicate")
    duplicate_entry.unique_id = "legacy-id"
    duplicate_entry.data = {CONF_API_KEY: "duplicate-key"}
    duplicate_flow = config_flow.GroqConfigFlow()
    duplicate_flow.hass = SimpleNamespace(
        config_entries=DummyConfigEntries([duplicate_entry])
    )
    monkeypatch.setattr(
        duplicate_flow,
        "async_show_form",
        lambda **kwargs: {"type": "form", **kwargs},
    )
    monkeypatch.setattr(
        duplicate_flow,
        "async_create_entry",
        lambda **kwargs: {"type": "create_entry", **kwargs},
    )
    monkeypatch.setattr(
        duplicate_flow,
        "async_set_unique_id",
        lambda unique_id: asyncio.sleep(0),
    )
    monkeypatch.setattr(duplicate_flow, "_abort_if_unique_id_configured", lambda: None)
    monkeypatch.setattr(
        config_flow,
        "async_validate_api_key",
        lambda hass, api_key: asyncio.sleep(0, result=None),
    )
    duplicate_result = await duplicate_flow.async_step_user(
        {CONF_API_KEY: "duplicate-key", CONF_NAME: "Groq"}
    )
    assert duplicate_result["errors"] == {"base": "duplicate_api_key"}

    flow = config_flow.GroqConfigFlow()
    flow.hass = SimpleNamespace(config_entries=DummyConfigEntries([]))
    flow.context = {"entry_id": "missing"}
    monkeypatch.setattr(
        flow, "async_abort", lambda **kwargs: {"type": "abort", **kwargs}
    )
    monkeypatch.setattr(
        flow,
        "async_show_form",
        lambda **kwargs: {"type": "form", **kwargs},
    )
    monkeypatch.setattr(flow, "async_set_unique_id", lambda unique_id: asyncio.sleep(0))
    monkeypatch.setattr(flow, "_abort_if_unique_id_configured", lambda: None)
    monkeypatch.setattr(
        flow,
        "async_create_entry",
        lambda **kwargs: {"type": "create_entry", **kwargs},
    )
    monkeypatch.setattr(
        config_flow,
        "async_validate_api_key",
        lambda hass, api_key: asyncio.sleep(0, result=None),
    )

    result = await flow.async_step_user({CONF_API_KEY: "key", CONF_NAME: "Name"})
    assert result["type"] == "create_entry"

    async def invalid_features(_user_input):
        raise ValueError("Enabled features are invalid")

    monkeypatch.setattr(config_flow, "validate_user_input", invalid_features)
    result = await flow.async_step_user({CONF_API_KEY: "key"})
    assert result["errors"][CONF_ENABLED_FEATURES] == "invalid_enabled_features"

    async def unknown_value_error(_user_input):
        raise ValueError("Something else")

    monkeypatch.setattr(config_flow, "validate_user_input", unknown_value_error)
    result = await flow.async_step_user({CONF_API_KEY: "key"})
    assert result["errors"]["base"] == "unknown_error"

    async def unexpected_error(_user_input):
        raise RuntimeError("boom")

    monkeypatch.setattr(config_flow, "validate_user_input", unexpected_error)
    result = await flow.async_step_user({CONF_API_KEY: "key"})
    assert result["errors"]["base"] == "unknown_error"

    async def aborting_validate(_user_input):
        return None

    monkeypatch.setattr(config_flow, "validate_user_input", aborting_validate)
    monkeypatch.setattr(
        flow,
        "_abort_if_unique_id_configured",
        lambda: (_ for _ in ()).throw(data_entry_flow.AbortFlow("already_configured")),
    )
    result = await flow.async_step_user({CONF_API_KEY: "key"})
    assert result == {"type": "abort", "reason": "already_configured"}

    assert await flow.async_step_reauth_confirm({"api_key": "new"}) == {
        "type": "abort",
        "reason": "unknown",
    }
    assert (await flow.async_step_reauth_confirm({}))["errors"][
        CONF_API_KEY
    ] == "required"

    reauth_entry = DummyEntry()
    flow._reauth_entry = reauth_entry
    monkeypatch.setattr(
        flow,
        "async_update_reload_and_abort",
        lambda entry, **kwargs: {"entry": entry, **kwargs},
    )
    updated = await flow.async_step_reauth_confirm({CONF_API_KEY: "updated"})
    assert updated["data"][CONF_API_KEY] == "updated"
    assert updated["unique_id"] == reauth_entry.unique_id

    options_flow = config_flow.GroqOptionsFlow()
    monkeypatch.setattr(
        options_flow,
        "async_show_form",
        lambda **kwargs: {"type": "form", **kwargs},
    )
    monkeypatch.setattr(
        options_flow,
        "async_create_entry",
        lambda **kwargs: {"type": "create_entry", **kwargs},
    )
    assert (await options_flow.async_step_init())["type"] == "form"
    assert (await options_flow.async_step_init({CONF_API_KEY: ""}))["data"] == {}

    monkeypatch.setattr(
        config_flow,
        "async_validate_api_key",
        lambda hass, api_key: asyncio.sleep(0, result="unknown"),
    )
    assert (await options_flow.async_step_init({CONF_API_KEY: "bad"}))["errors"] == {
        "base": "unknown"
    }

    tuple_handler_entry = DummyEntry("tuple-entry")
    tuple_options_flow = config_flow.GroqOptionsFlow()
    tuple_options_flow.hass = SimpleNamespace(
        config_entries=DummyConfigEntries([tuple_handler_entry])
    )
    tuple_options_flow.handler = (tuple_handler_entry.entry_id, FEATURE_TEXT_GENERATION)
    assert tuple_options_flow._current_entry() is tuple_handler_entry

    subflow = config_flow.GroqServiceSubentryFlow()
    assert await subflow.async_step_user() == subflow.async_show_form(
        step_id="init", data_schema=config_flow.service_type_schema()
    )
    subflow.hass = SimpleNamespace()
    subflow._get_entry = lambda: SimpleNamespace(
        data={CONF_API_KEY: "data-key"},
        options={CONF_API_KEY: "options-key"},
    )
    assert subflow._account_api_key() == "options-key"
    bare_subflow = object.__new__(config_flow.GroqServiceSubentryFlow)
    bare_subflow._service_type = None
    assert bare_subflow._configured_service_type is None
    with pytest.raises(ValueError):
        subflow._existing_service_type()
    subflow._service_type = FEATURE_TEXT_GENERATION
    assert subflow._configured_service_type == FEATURE_TEXT_GENERATION

    class TypeErrorSubentryFlow(config_flow.GroqServiceSubentryFlow):
        @property
        def _subentry_type(self):
            raise TypeError

    type_error_subflow = TypeErrorSubentryFlow()
    type_error_subflow.handler = object()
    assert type_error_subflow._configured_service_type is None

    class UnsupportedSubentryFlow(config_flow.GroqServiceSubentryFlow):
        @property
        def _subentry_type(self):
            return "unsupported"

    unsupported_subflow = UnsupportedSubentryFlow()
    unsupported_subflow.handler = object()
    assert unsupported_subflow._configured_service_type is None

    class TypedSubentryFlow(config_flow.GroqServiceSubentryFlow):
        @property
        def _subentry_type(self):
            return FEATURE_IMAGE_RECOGNITION

    typed_subflow = TypedSubentryFlow()
    typed_subflow.handler = object()
    assert typed_subflow._configured_service_type == FEATURE_IMAGE_RECOGNITION

    async def fake_text_step(user_input=None):
        return {"step": "text", "user_input": user_input}

    typed_subflow.async_step_text_generation = fake_text_step
    typed_subflow._service_type = FEATURE_TEXT_GENERATION
    assert await typed_subflow.async_step_user({"x": 1}) == {
        "step": "text",
        "user_input": {"x": 1},
    }

    reconfigure_flow = config_flow.GroqServiceSubentryFlow()
    reconfigure_flow.async_step_text_generation = fake_text_step
    monkeypatch.setattr(
        reconfigure_flow,
        "_existing_service_type",
        lambda: FEATURE_TEXT_GENERATION,
    )
    assert await reconfigure_flow.async_step_reconfigure({"y": 2}) == {
        "step": "text",
        "user_input": {"y": 2},
    }

    class ExistingTypeFlow(config_flow.GroqServiceSubentryFlow):
        @property
        def _is_reconfigure(self):
            return True

        def _get_reconfigure_subentry(self):
            return SimpleNamespace(data={CONF_SERVICE_TYPE: FEATURE_TEXT_TO_SPEECH})

    assert ExistingTypeFlow()._existing_service_type() == FEATURE_TEXT_TO_SPEECH

    class ConfiguredTypeFlow(config_flow.GroqServiceSubentryFlow):
        @property
        def _is_reconfigure(self):
            return True

        def _get_reconfigure_subentry(self):
            return SimpleNamespace(data={})

    configured_type_flow = ConfiguredTypeFlow()
    configured_type_flow._service_type = FEATURE_IMAGE_RECOGNITION
    assert configured_type_flow._existing_service_type() == FEATURE_IMAGE_RECOGNITION

    class ReconfigureTextFlow(config_flow.GroqServiceSubentryFlow):
        @property
        def _is_reconfigure(self):
            return True

        def _existing_service_data(self):
            return {CONF_NAME: "Existing"}

        def _create_service_entry(self, service_type, user_input):
            return {"service_type": service_type, "data": user_input}

    reconfigure_text_flow = ReconfigureTextFlow()
    merged = await reconfigure_text_flow.async_step_text_generation(
        {CONF_NAME: "Text", CONF_MODEL: "llama-3.1-8b-instant"}
    )
    assert merged["data"][CONF_NAME] == "Text"

    form_flow = config_flow.GroqServiceSubentryFlow()
    monkeypatch.setattr(
        form_flow,
        "async_show_form",
        lambda **kwargs: {"type": "form", **kwargs},
    )
    monkeypatch.setattr(
        form_flow,
        "_create_service_entry",
        lambda service_type, user_input: {
            "type": "create",
            "service_type": service_type,
            "data": user_input,
        },
    )
    assert (await form_flow.async_step_init())[("step_id")] == "init"
    assert (
        await form_flow.async_step_init({CONF_SERVICE_TYPE: FEATURE_TEXT_TO_SPEECH})
    )["step_id"] == FEATURE_TEXT_TO_SPEECH
    assert (await form_flow.async_step_text_generation())[
        "step_id"
    ] == FEATURE_TEXT_GENERATION
    assert (
        await form_flow.async_step_text_generation(
            {
                CONF_NAME: "Text",
                CONF_MODEL: "llama-3.1-8b-instant",
                "reasoning_effort": "low",
            }
        )
    )["errors"][CONF_MODEL]
    advanced = await form_flow.async_step_text_generation(
        {
            CONF_NAME: "Text",
            CONF_MODEL: "openai/gpt-oss-20b",
            "advanced_options": True,
        }
    )
    assert advanced["step_id"] == "text_generation_advanced"
    assert (await form_flow.async_step_text_generation_advanced())[
        "step_id"
    ] == "text_generation_advanced"
    assert (
        await form_flow.async_step_text_generation_advanced(
            {CONF_MODEL: "llama-3.1-8b-instant", "reasoning_effort": "low"}
        )
    )["errors"]["base"]
    created = await form_flow.async_step_text_generation_advanced(
        {CONF_MODEL: "openai/gpt-oss-20b"}
    )
    assert created["service_type"] == FEATURE_TEXT_GENERATION
    assert (await form_flow.async_step_speech_to_text())["step_id"] == "speech_to_text"
    assert (await form_flow.async_step_speech_to_text({CONF_NAME: "STT"}))[
        "service_type"
    ] == "speech_to_text"
    assert (
        await form_flow.async_step_text_to_speech(
            {
                CONF_NAME: "TTS",
                CONF_MODEL: "canopylabs/orpheus-v1-english",
                "voice": "troy",
            }
        )
    )["service_type"] == FEATURE_TEXT_TO_SPEECH
    assert (await form_flow.async_step_image_recognition())[
        "step_id"
    ] == FEATURE_IMAGE_RECOGNITION
    assert (await form_flow.async_step_image_recognition({CONF_NAME: "Vision"}))[
        "service_type"
    ] == FEATURE_IMAGE_RECOGNITION


@pytest.mark.asyncio
async def test_services_handlers_and_registration_cover_remaining_paths(
    tmp_path,
    monkeypatch,
):
    entry = DummyEntry()
    client = DummyClient()
    runtime = build_runtime(DummyHass(), entry)
    runtime.client = client
    runtime.feature_registry = GroqFeatureRegistry(
        [
            GroqFeature.TEXT_GENERATION,
            GroqFeature.SPEECH_TO_TEXT,
            GroqFeature.VISION,
            GroqFeature.PROMPT_CACHING,
        ]
    )
    runtime.services_by_type = {
        "text_generation": (
            {
                "unique_id": "text-id",
                "name": "Text Service",
                "service_type": "text_generation",
                "model": "openai/gpt-oss-20b",
                "structured_outputs": True,
                "schema": {"type": "object"},
            },
        ),
        "image_recognition": (
            {
                "unique_id": "vision-id",
                "name": "Vision Service",
                "service_type": "image_recognition",
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            },
        ),
        "speech_to_text": (
            {
                "unique_id": "stt-id",
                "name": "Voice Notes",
                "service_type": "speech_to_text",
                "model": "whisper-large-v3-turbo",
                "language": "en-US",
            },
        ),
    }
    entry.runtime_data = runtime
    hass = DummyHass([entry], services=DummyServices())
    hass.config = SimpleNamespace(is_allowed_path=lambda _path: True)

    assert (
        _entry_from_call(hass, service_call({ATTR_CONFIG_ENTRY_ID: "entry-id"}))
        is entry
    )
    with pytest.raises(Exception):
        _entry_from_call(hass, service_call({ATTR_CONFIG_ENTRY_ID: "missing"}))
    with pytest.raises(Exception):
        _entry_from_call(DummyHass([]), service_call({}))
    with pytest.raises(Exception):
        _entry_from_call(DummyHass([entry, DummyEntry("other")]), service_call({}))
    assert _entry_from_call(DummyHass([entry]), service_call({})) is entry
    assert await _runtime_from_call(
        hass, service_call({ATTR_CONFIG_ENTRY_ID: "entry-id"})
    )
    assert _service_subentries(entry, None, "text_generation") == []
    assert (
        _service_from_call(entry, runtime, service_call({}), "text_generation")[
            "unique_id"
        ]
        == "text-id"
    )
    assert (
        _service_from_call(
            entry,
            runtime,
            service_call({ATTR_SERVICE_ID: "Text Service"}),
            "text_generation",
        )["unique_id"]
        == "text-id"
    )
    with pytest.raises(Exception):
        _service_from_call(
            entry,
            runtime,
            service_call({ATTR_SERVICE_ID: "missing"}),
            "text_generation",
        )
    assert _request_options({"include_reasoning": True})["include_reasoning"] is True

    key = _cache_key("ns", {"b": 2, "a": 1})
    _cache_set(runtime, "openai/gpt-oss-20b", key, {"text": "cached"})
    assert _cache_get(runtime, "openai/gpt-oss-20b", key)["cached"] is True
    assert _cache_get(runtime, "llama-3.1-8b-instant", "missing") is None

    structured = await _handle_generate_structured(hass)(
        service_call(
            {
                ATTR_CONFIG_ENTRY_ID: "entry-id",
                ATTR_SERVICE_ID: "text-id",
                ATTR_PROMPT: "p",
                ATTR_SCHEMA: {"type": "object"},
                ATTR_REASONING_EFFORT: "low",
            }
        )
    )
    assert structured["data"] == {"ok": True}
    cached_structured = await _handle_generate_structured(hass)(
        service_call(
            {
                ATTR_CONFIG_ENTRY_ID: "entry-id",
                ATTR_SERVICE_ID: "text-id",
                ATTR_PROMPT: "p",
                ATTR_SCHEMA: {"type": "object"},
                ATTR_REASONING_EFFORT: "low",
            }
        )
    )
    assert cached_structured["cached"] is True
    plain_text_output = await _handle_generate_structured(hass)(
        service_call(
            {
                ATTR_CONFIG_ENTRY_ID: "entry-id",
                ATTR_SERVICE_ID: "text-id",
                ATTR_PROMPT: "plain",
                CONF_MODEL: "llama-3.1-8b-instant",
            }
        )
    )
    assert plain_text_output["text"] == "plain text"
    with pytest.raises(ServiceValidationError, match="structured_outputs"):
        await _handle_generate_structured(hass)(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    ATTR_SERVICE_ID: "text-id",
                    ATTR_PROMPT: "bad",
                    ATTR_SCHEMA: {"type": "object"},
                    CONF_MODEL: "llama-3.1-8b-instant",
                }
            )
        )
    with pytest.raises(ServiceValidationError, match="response_format"):
        await _handle_generate_structured(hass)(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    ATTR_SERVICE_ID: "text-id",
                    ATTR_PROMPT: "bad",
                    CONF_MODEL: "llama-3.1-8b-instant",
                    ATTR_REQUEST_BODY_OPTIONS: {
                        "response_format": {"type": "json_schema"}
                    },
                }
            )
        )
    with pytest.raises(ServiceValidationError, match="reasoning options"):
        await _handle_generate_structured(hass)(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    ATTR_SERVICE_ID: "text-id",
                    ATTR_PROMPT: "bad",
                    CONF_MODEL: "llama-3.1-8b-instant",
                    ATTR_REQUEST_BODY_OPTIONS: {"reasoning_effort": "low"},
                }
            )
        )
    model_configuration_issues = []
    monkeypatch.setattr(
        "custom_components.groq.services.async_create_model_configuration_issue",
        lambda hass, entry, service_data, model, feature: model_configuration_issues.append(
            (entry.entry_id, service_data["unique_id"], model, feature)
        ),
    )
    runtime.services_by_type["text_generation"][0]["model"] = "whisper-large-v3"
    with pytest.raises(ServiceValidationError, match="text_generation"):
        await _handle_generate_text(hass)(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    ATTR_SERVICE_ID: "text-id",
                    ATTR_PROMPT: "bad",
                }
            )
        )
    assert model_configuration_issues == [
        ("entry-id", "text-id", "whisper-large-v3", "text_generation")
    ]
    runtime.services_by_type["text_generation"][0]["model"] = "openai/gpt-oss-20b"
    original_registry = runtime.model_registry
    runtime.model_registry = GroqModelRegistry(
        [
            GroqModel(
                "tiny-text",
                context_window=20,
                capabilities=frozenset({GroqCapability.TEXT_GENERATION}),
            )
        ],
        include_built_ins=False,
    )
    with pytest.raises(ServiceValidationError, match="context window"):
        await _handle_generate_structured(hass)(
            service_call(
                {
                    ATTR_CONFIG_ENTRY_ID: "entry-id",
                    ATTR_SERVICE_ID: "text-id",
                    ATTR_PROMPT: "x" * 30,
                    CONF_MODEL: "tiny-text",
                }
            )
        )
    runtime.model_registry = original_registry
    generated = await _handle_generate_text(hass)(
        service_call(
            {
                ATTR_CONFIG_ENTRY_ID: "entry-id",
                ATTR_SERVICE_ID: "text-id",
                ATTR_PROMPT: "p",
            }
        )
    )
    assert generated["data"] == {"ok": True}

    runtime.model_registry.update(
        [
            GroqModel(
                "vision-cache",
                capabilities=frozenset(
                    {GroqCapability.VISION, GroqCapability.PROMPT_CACHING}
                ),
            )
        ]
    )
    runtime.services_by_type["image_recognition"][0]["model"] = "vision-cache"
    analyzed = await _handle_analyze_image(hass)(
        service_call(
            {
                ATTR_CONFIG_ENTRY_ID: "entry-id",
                ATTR_SERVICE_ID: "vision-id",
                ATTR_PROMPT: "p",
                ATTR_IMAGE_URL: "https://example.test/image.jpg",
            }
        )
    )
    assert analyzed["text"] == "image text"
    cached_analyzed = await _handle_analyze_image(hass)(
        service_call(
            {
                ATTR_CONFIG_ENTRY_ID: "entry-id",
                ATTR_SERVICE_ID: "vision-id",
                ATTR_PROMPT: "p",
                ATTR_IMAGE_URL: "https://example.test/image.jpg",
            }
        )
    )
    assert cached_analyzed["cached"] is True
    ocr = await _handle_extract_text_from_image(hass)(
        service_call(
            {
                ATTR_CONFIG_ENTRY_ID: "entry-id",
                ATTR_SERVICE_ID: "vision-id",
                ATTR_PROMPT: "p",
                ATTR_IMAGE_URL: "https://example.test/image.jpg",
            }
        )
    )
    assert ocr["text"] == "image text"
    cached_ocr = await _handle_extract_text_from_image(hass)(
        service_call(
            {
                ATTR_CONFIG_ENTRY_ID: "entry-id",
                ATTR_SERVICE_ID: "vision-id",
                ATTR_PROMPT: "p",
                ATTR_IMAGE_URL: "https://example.test/image.jpg",
            }
        )
    )
    assert cached_ocr["cached"] is True
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"RIFF audio")
    transcription = await _handle_transcribe_audio(hass)(
        service_call(
            {
                ATTR_CONFIG_ENTRY_ID: "entry-id",
                ATTR_SERVICE_ID: "stt-id",
                ATTR_AUDIO_PATH: str(audio_path),
                ATTR_PROMPT: "Home Assistant device names",
            }
        )
    )
    assert transcription == {
        "text": "transcribed",
        "model": "whisper-large-v3-turbo",
        "language": "en-US",
        "filename": "voice.wav",
    }
    assert (
        await _handle_clear_cache(hass)(
            service_call({ATTR_CONFIG_ENTRY_ID: "entry-id"})
        )
    )["cleared"] >= 0
    client.models = [GroqModel("new-model")]
    listed = await _handle_list_models(hass)(
        service_call({ATTR_CONFIG_ENTRY_ID: "entry-id", ATTR_REFRESH: True})
    )
    assert any(model["id"] == "new-model" for model in listed["models"])

    await async_register_services(hass)
    await async_register_services(hass)
    assert len(hass.services.registered) == 7
    await async_unregister_services(hass)
    assert {service for _, service in hass.services.removed} == {
        SERVICE_GENERATE_TEXT,
        SERVICE_GENERATE_STRUCTURED,
        SERVICE_ANALYZE_IMAGE,
        SERVICE_EXTRACT_TEXT_FROM_IMAGE,
        SERVICE_TRANSCRIBE_AUDIO,
        SERVICE_CLEAR_CACHE,
        SERVICE_LIST_MODELS,
    }
    await async_unregister_services(hass)

    class RuntimeSetterRaises(DummyEntry):
        @property
        def runtime_data(self):
            return None

        @runtime_data.setter
        def runtime_data(self, value):
            raise AttributeError

    entry_without_runtime_setter = RuntimeSetterRaises()
    assert await _runtime_from_call(
        DummyHass([entry_without_runtime_setter]),
        service_call({ATTR_CONFIG_ENTRY_ID: "entry-id"}),
    )

    bad_subentry_entry = DummyEntry()
    bad_subentry_entry.subentries = {"bad": SimpleNamespace(data={})}
    assert service_data_by_type(bad_subentry_entry) == {}


@pytest.mark.asyncio
async def test_image_source_resolution_paths(tmp_path, monkeypatch):
    """Cover camera, media-source, local-path, and fallback image inputs."""

    class ImageHass(DummyHass):
        def __init__(self, *, allowed: bool = True):
            super().__init__()
            self.config = SimpleNamespace(is_allowed_path=lambda _path: allowed)

        async def async_add_executor_job(self, func, *args):
            return func(*args)

    hass = ImageHass()
    image_path = tmp_path / "snapshot.jpg"
    image_path.write_bytes(b"image")
    text_path = tmp_path / "not-image.txt"
    text_path.write_text("not image")

    assert _image_data_url(b"abc", None) == "data:image/jpeg;base64,YWJj"
    assert await _image_from_camera_target(hass, service_call({})) is None
    assert (
        await _image_from_camera_target(
            hass, service_call({"entity_id": "light.kitchen"})
        )
        is None
    )
    with pytest.raises(ServiceValidationError, match="only one camera"):
        await _image_from_camera_target(
            hass,
            service_call({"entity_id": ["camera.front", "camera.back"]}),
        )

    async def fake_camera_image(_hass, _entity_id):
        return SimpleNamespace(content=b"camera", content_type="image/png")

    monkeypatch.setattr(
        "custom_components.groq.services.camera.async_get_image",
        fake_camera_image,
    )
    assert (
        await _image_from_camera_target(
            hass, service_call({ATTR_CAMERA_ENTITY_ID: "camera.front"})
        )
        == "data:image/png;base64,Y2FtZXJh"
    )
    assert (
        await _image_from_camera_target(
            hass, service_call({"entity_id": "camera.front"})
        )
        == "data:image/png;base64,Y2FtZXJh"
    )

    async def fake_target_entities(_call):
        return {"camera.area"}

    monkeypatch.setattr(
        "custom_components.groq.services.service_helper.async_extract_entity_ids",
        fake_target_entities,
    )
    assert (
        await _image_from_camera_target(hass, service_call({"area_id": "front_yard"}))
        == "data:image/png;base64,Y2FtZXJh"
    )

    async def failing_camera_image(_hass, _entity_id):
        raise RuntimeError("camera unavailable")

    monkeypatch.setattr(
        "custom_components.groq.services.camera.async_get_image",
        failing_camera_image,
    )
    with pytest.raises(ServiceValidationError, match="Could not capture"):
        await _image_from_camera_target(hass, service_call({"entity_id": "camera.bad"}))

    assert await _image_from_local_path(hass, str(image_path)) == (
        "data:image/jpeg;base64,aW1hZ2U="
    )
    with pytest.raises(ServiceValidationError, match="allowlist"):
        await _image_from_local_path(ImageHass(allowed=False), str(image_path))
    with pytest.raises(ServiceValidationError, match="not found"):
        await _image_from_local_path(hass, str(tmp_path / "missing.jpg"))
    with pytest.raises(ServiceValidationError, match="not an image"):
        await _image_from_local_path(hass, str(text_path))

    assert await _image_from_media_source(hass, str(image_path)) == (
        "data:image/jpeg;base64,aW1hZ2U="
    )

    async def fake_media_path(_hass, _media_id, _target):
        return SimpleNamespace(path=image_path, mime_type="image/jpeg", url="unused")

    monkeypatch.setattr(
        "custom_components.groq.services.async_resolve_media",
        fake_media_path,
    )
    assert await _image_from_media_source(
        hass, "media-source://media/snapshot.jpg"
    ) == ("data:image/jpeg;base64,aW1hZ2U=")

    async def fake_media_url(_hass, _media_id, _target):
        return SimpleNamespace(path=None, mime_type="image/png", url="https://image")

    monkeypatch.setattr(
        "custom_components.groq.services.async_resolve_media",
        fake_media_url,
    )
    assert (
        await _image_from_media_source(hass, "media-source://media/remote.png")
        == "https://image"
    )

    async def fake_unresolvable_media(_hass, _media_id, _target):
        raise Unresolvable("missing")

    monkeypatch.setattr(
        "custom_components.groq.services.async_resolve_media",
        fake_unresolvable_media,
    )
    with pytest.raises(ServiceValidationError, match="Could not resolve"):
        await _image_from_media_source(hass, "media-source://media/missing.jpg")

    async def fake_non_image_media(_hass, _media_id, _target):
        return SimpleNamespace(path=None, mime_type="text/plain", url="https://text")

    monkeypatch.setattr(
        "custom_components.groq.services.async_resolve_media",
        fake_non_image_media,
    )
    with pytest.raises(ServiceValidationError, match="not an image"):
        await _image_from_media_source(hass, "media-source://media/file.txt")

    monkeypatch.setattr(
        "custom_components.groq.services.camera.async_get_image",
        fake_camera_image,
    )
    assert (
        await _image_url_from_call(
            hass, service_call({"entity_id": "camera.front", ATTR_IMAGE_URL: "url"})
        )
        == "data:image/png;base64,Y2FtZXJh"
    )
    assert (
        await _image_url_from_call(
            hass,
            service_call(
                {
                    ATTR_CAMERA_ENTITY_ID: "camera.front",
                    ATTR_IMAGE_FILE: str(image_path),
                }
            ),
        )
        == "data:image/png;base64,Y2FtZXJh"
    )
    assert (
        await _image_url_from_call(
            hass, service_call({ATTR_IMAGE_FILE: str(image_path)})
        )
        == "data:image/jpeg;base64,aW1hZ2U="
    )
    assert (
        await _image_url_from_call(
            hass, service_call({ATTR_IMAGE_PATH: str(image_path)})
        )
        == "data:image/jpeg;base64,aW1hZ2U="
    )
    assert (
        await _image_url_from_call(
            hass, service_call({ATTR_IMAGE_URL: "https://example/image.jpg"})
        )
        == "https://example/image.jpg"
    )
    assert (
        await _image_url_from_call(
            hass, service_call({ATTR_IMAGE_URL: "data:image/png;base64,YWJj"})
        )
        == "data:image/png;base64,YWJj"
    )
    with pytest.raises(ServiceValidationError, match="Image URL"):
        await _image_url_from_call(
            hass, service_call({ATTR_IMAGE_URL: "file:///tmp/snapshot.jpg"})
        )
    with pytest.raises(ServiceValidationError, match="Image URL"):
        await _image_url_from_call(
            hass, service_call({ATTR_IMAGE_URL: "data:image/png;base64,abc"})
        )
    with pytest.raises(ServiceValidationError, match="Image URL"):
        await _image_url_from_call(
            hass, service_call({ATTR_IMAGE_URL: "data:image/png;base64"})
        )
    assert (
        await _image_url_from_call(
            hass, service_call({ATTR_IMAGE_URL: "data:image/png,raw"})
        )
        == "data:image/png,raw"
    )
    monkeypatch.setattr("custom_components.groq.services.MAX_IMAGE_BYTES", 4)
    with patch(
        "custom_components.groq.services.b64decode",
        side_effect=AssertionError("oversized image should not be decoded"),
    ):
        with pytest.raises(ServiceValidationError, match="too large"):
            await _image_url_from_call(
                hass,
                service_call({ATTR_IMAGE_URL: "data:image/png;base64,YWJjZGU="}),
            )
    with pytest.raises(ServiceValidationError, match="too large"):
        await _image_from_local_path(hass, str(image_path))
    with pytest.raises(ServiceValidationError, match="Select a camera entity"):
        await _image_url_from_call(hass, service_call({}))


@pytest.mark.asyncio
async def test_audio_source_resolution_paths(tmp_path, monkeypatch):
    """Cover media-source, local-path, and fallback audio inputs."""

    class AudioHass(DummyHass):
        def __init__(self, *, allowed: bool = True):
            super().__init__()
            self.config = SimpleNamespace(is_allowed_path=lambda _path: allowed)

        async def async_add_executor_job(self, func, *args):
            return func(*args)

    hass = AudioHass()
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"audio")
    text_path = tmp_path / "not-audio.txt"
    text_path.write_text("not audio")

    assert await _audio_from_local_path(hass, str(audio_path)) == (
        b"audio",
        "voice.wav",
    )
    with pytest.raises(ServiceValidationError, match="allowlist"):
        await _audio_from_local_path(AudioHass(allowed=False), str(audio_path))
    with pytest.raises(ServiceValidationError, match="not found"):
        await _audio_from_local_path(hass, str(tmp_path / "missing.wav"))
    with pytest.raises(ServiceValidationError, match="not audio"):
        await _audio_from_local_path(hass, str(text_path))

    assert await _audio_from_media_source(hass, str(audio_path)) == (
        b"audio",
        "voice.wav",
    )

    async def fake_media_path(_hass, _media_id, _target):
        return SimpleNamespace(path=audio_path, mime_type="audio/wav", url="unused")

    monkeypatch.setattr(
        "custom_components.groq.services.async_resolve_media",
        fake_media_path,
    )
    assert await _audio_from_media_source(hass, "media-source://media/voice.wav") == (
        b"audio",
        "voice.wav",
    )

    async def fake_media_url(_hass, _media_id, _target):
        return SimpleNamespace(path=None, mime_type="audio/wav", url="https://audio")

    monkeypatch.setattr(
        "custom_components.groq.services.async_resolve_media",
        fake_media_url,
    )
    with pytest.raises(ServiceValidationError, match="local file"):
        await _audio_from_media_source(hass, "media-source://media/remote.wav")

    async def fake_unresolvable_media(_hass, _media_id, _target):
        raise Unresolvable("missing")

    monkeypatch.setattr(
        "custom_components.groq.services.async_resolve_media",
        fake_unresolvable_media,
    )
    with pytest.raises(ServiceValidationError, match="Could not resolve"):
        await _audio_from_media_source(hass, "media-source://media/missing.wav")

    async def fake_non_audio_media(_hass, _media_id, _target):
        return SimpleNamespace(path=audio_path, mime_type="text/plain", url="unused")

    monkeypatch.setattr(
        "custom_components.groq.services.async_resolve_media",
        fake_non_audio_media,
    )
    with pytest.raises(ServiceValidationError, match="not audio"):
        await _audio_from_media_source(hass, "media-source://media/file.txt")

    assert await _audio_from_call(
        hass, service_call({ATTR_AUDIO_FILE: str(audio_path)})
    ) == (b"audio", "voice.wav")
    assert await _audio_from_call(
        hass, service_call({ATTR_AUDIO_PATH: str(audio_path)})
    ) == (b"audio", "voice.wav")
    monkeypatch.setattr("custom_components.groq.services.MAX_AUDIO_BYTES", 4)
    audio_path.write_bytes(b"audio!")
    with pytest.raises(ServiceValidationError, match="too large"):
        await _audio_from_local_path(hass, str(audio_path))
    with pytest.raises(ServiceValidationError, match="Select an audio file"):
        await _audio_from_call(hass, service_call({}))


@pytest.mark.asyncio
async def test_stt_setup_properties_wav_error_and_empty_results(monkeypatch):
    entry = DummyEntry()
    entry.subentries = {
        "skip": SimpleNamespace(data={"service_type": "other"}),
        "stt": SimpleNamespace(
            subentry_id="stt-id",
            data={"service_type": "speech_to_text", "model": "whisper-large-v3"},
        ),
    }
    runtime = build_runtime(DummyHass(), entry)
    runtime.client = DummyClient()
    entry.runtime_data = runtime
    added = []
    subentry_ids = []

    def add_entities(entities, **kwargs):
        added.extend(entities)
        subentry_ids.append(kwargs.get("config_subentry_id"))

    await stt_setup(DummyHass(), entry, add_entities)
    assert len(added) == 1
    assert subentry_ids == ["stt-id"]
    entity = added[0]
    assert "en-US" in entity.supported_languages
    assert stt.AudioFormats.WAV in entity.supported_formats
    assert stt.AudioCodecs.PCM in entity.supported_codecs
    assert stt.AudioBitRates.BITRATE_16 in entity.supported_bit_rates
    assert stt.AudioSampleRates.SAMPLERATE_16000 in entity.supported_sample_rates
    assert stt.AudioChannels.CHANNEL_MONO in entity.supported_channels
    assert entity.device_info["model"] == "whisper-large-v3"
    assert entity.device_info["name"] == "Groq Speech-to-Text"
    assert entity.has_entity_name is True
    assert entity.translation_key == "speech_to_text"

    async def stream():
        yield b"\x00\x00"

    result = await entity.async_process_audio_stream(
        stt.SpeechMetadata(
            language="en-US",
            format=stt.AudioFormats.WAV,
            codec=stt.AudioCodecs.PCM,
            bit_rate=stt.AudioBitRates.BITRATE_16,
            sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
            channel=stt.AudioChannels.CHANNEL_MONO,
        ),
        stream(),
    )
    assert result.result == stt.SpeechResultState.SUCCESS

    class EmptyClient(DummyClient):
        async def async_transcribe_audio(self, **kwargs):
            return ""

    class ErrorClient(DummyClient):
        async def async_transcribe_audio(self, **kwargs):
            raise GroqApiError("bad")

    class AuthClient(DummyClient):
        async def async_transcribe_audio(self, **kwargs):
            raise ConfigEntryAuthFailed("bad credentials")

    empty = GroqSTTEntity(entry, {"model": "whisper-large-v3"}, EmptyClient())
    assert (
        await empty.async_process_audio_stream(
            stt.SpeechMetadata(
                language="en-US",
                format=stt.AudioFormats.OGG,
                codec=stt.AudioCodecs.OPUS,
                bit_rate=stt.AudioBitRates.BITRATE_16,
                sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
                channel=stt.AudioChannels.CHANNEL_MONO,
            ),
            stream(),
        )
    ).result == stt.SpeechResultState.ERROR
    error = GroqSTTEntity(entry, {"model": "whisper-large-v3"}, ErrorClient())
    assert (
        await error.async_process_audio_stream(
            stt.SpeechMetadata(
                language="en-US",
                format=stt.AudioFormats.OGG,
                codec=stt.AudioCodecs.OPUS,
                bit_rate=stt.AudioBitRates.BITRATE_16,
                sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
                channel=stt.AudioChannels.CHANNEL_MONO,
            ),
            stream(),
        )
    ).result == stt.SpeechResultState.ERROR
    auth = GroqSTTEntity(entry, {"model": "whisper-large-v3"}, AuthClient())
    with pytest.raises(ConfigEntryAuthFailed):
        await auth.async_process_audio_stream(
            stt.SpeechMetadata(
                language="en-US",
                format=stt.AudioFormats.OGG,
                codec=stt.AudioCodecs.OPUS,
                bit_rate=stt.AudioBitRates.BITRATE_16,
                sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
                channel=stt.AudioChannels.CHANNEL_MONO,
            ),
            stream(),
        )

    async def oversized_stream():
        yield b"12345"

    monkeypatch.setattr("custom_components.groq.stt.MAX_STT_AUDIO_BYTES", 4)
    assert (
        await entity.async_process_audio_stream(
            stt.SpeechMetadata(
                language="en-US",
                format=stt.AudioFormats.OGG,
                codec=stt.AudioCodecs.OPUS,
                bit_rate=stt.AudioBitRates.BITRATE_16,
                sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
                channel=stt.AudioChannels.CHANNEL_MONO,
            ),
            oversized_stream(),
        )
    ).result == stt.SpeechResultState.ERROR


@pytest.mark.asyncio
async def test_ai_task_helpers_and_fallback_paths():
    assert _strip_json_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert "- name (required)" in _structure_description(
        vol.Schema({vol.Required("name", description="Name"): str})
    )
    assert _structure_description(vol.Schema(str))

    entry = DummyEntry()
    service_data = {
        "unique_id": "ai-id",
        "name": "AI",
        "model": "openai/gpt-oss-20b",
        "structured_outputs": True,
        "schema": {"type": "object"},
    }
    client = DummyClient()
    entity = GroqAITaskEntity(DummyHass(), entry, service_data, client)
    assert entity.device_info["name"] == "AI"
    assert entity.has_entity_name is True
    assert entity.translation_key == "data_generation_tasks"
    task = SimpleNamespace(
        name="task",
        instructions="Return data",
        structure=None,
    )
    result = await entity._async_generate_data(
        task, SimpleNamespace(conversation_id="c")
    )
    assert result.data == {"ok": True}

    plain_client = DummyClient()
    plain_client.text = '```json\n{"name":"Kitchen"}\n```'
    plain_entity = GroqAITaskEntity(
        DummyHass(), entry, {"model": "llama-3.1-8b-instant"}, plain_client
    )
    task = SimpleNamespace(
        name="task",
        instructions="Return data",
        structure=vol.Schema({vol.Required("name"): str}),
    )
    # Force fallback by disabling service-level structured outputs and monkeypatching
    # the imported converter to return no schema for this specific branch.
    with patch(
        "custom_components.groq.ai_task.voluptuous_schema_to_json_schema",
        return_value={},
    ):
        result = await plain_entity._async_generate_data(
            task, SimpleNamespace(conversation_id="c")
        )
    assert result.data == {"name": "Kitchen"}

    plain_client.text = "not json"
    with patch(
        "custom_components.groq.ai_task.voluptuous_schema_to_json_schema",
        return_value={},
    ):
        with pytest.raises(HomeAssistantError):
            await plain_entity._async_generate_data(
                task, SimpleNamespace(conversation_id="c")
            )

    bad_structured = DummyClient()
    bad_structured.structured = {"text": "{}", "data": {}, "cached": False}
    bad_entity = GroqAITaskEntity(
        DummyHass(), entry, {"model": "openai/gpt-oss-20b"}, bad_structured
    )
    with pytest.raises(HomeAssistantError):
        await bad_entity._async_generate_data(
            task, SimpleNamespace(conversation_id="c")
        )

    class FailingStructuredClient(DummyClient):
        async def async_generate_structured(self, request):
            self.requests.append(request)
            raise GroqApiError(
                "Failed to validate JSON",
                status=400,
                payload={"failed_generation": "{}"},
            )

    retry_client = FailingStructuredClient()
    retry_client.text = '{"name":"Fallback"}'
    retry_entity = GroqAITaskEntity(
        DummyHass(), entry, {"model": "openai/gpt-oss-20b"}, retry_client
    )
    result = await retry_entity._async_generate_data(
        task, SimpleNamespace(conversation_id="c")
    )
    assert result.data == {"name": "Fallback"}
    assert isinstance(retry_client.requests[0], StructuredGenerationRequest)
    assert isinstance(retry_client.requests[1], TextGenerationRequest)

    class FatalStructuredClient(DummyClient):
        async def async_generate_structured(self, request):
            self.requests.append(request)
            raise GroqApiError("server error", status=500)

    fatal_entity = GroqAITaskEntity(
        DummyHass(), entry, {"model": "openai/gpt-oss-20b"}, FatalStructuredClient()
    )
    with pytest.raises(GroqApiError):
        await fatal_entity._async_generate_data(
            task, SimpleNamespace(conversation_id="c")
        )
    tiny_registry = GroqModelRegistry(
        [
            GroqModel(
                "tiny-text",
                context_window=20,
                capabilities=frozenset(
                    {
                        GroqCapability.TEXT_GENERATION,
                        GroqCapability.STRUCTURED_OUTPUTS,
                    }
                ),
            )
        ],
        include_built_ins=False,
    )
    tiny_entity = GroqAITaskEntity(
        DummyHass(),
        entry,
        {"model": "tiny-text"},
        DummyClient(),
        tiny_registry,
    )
    with pytest.raises(HomeAssistantError, match="context window"):
        await tiny_entity._async_generate_data(
            SimpleNamespace(name="task", instructions="x" * 30, structure=None),
            SimpleNamespace(conversation_id="c"),
        )
    tiny_structured_entity = GroqAITaskEntity(
        DummyHass(),
        entry,
        {"model": "tiny-text"},
        DummyClient(),
        tiny_registry,
    )
    with pytest.raises(HomeAssistantError, match="context window"):
        await tiny_structured_entity._async_generate_data(
            task,
            SimpleNamespace(conversation_id="c"),
        )
    body_error_entity = GroqAITaskEntity(
        DummyHass(),
        entry,
        {
            "model": "llama-3.1-8b-instant",
            "request_body_options": {"reasoning_effort": "low"},
        },
        DummyClient(),
    )
    with pytest.raises(HomeAssistantError, match="reasoning options"):
        await body_error_entity._async_generate_data(
            SimpleNamespace(name="task", instructions="Return data", structure=None),
            SimpleNamespace(conversation_id="c"),
        )
    structured_body_error_entity = GroqAITaskEntity(
        DummyHass(),
        entry,
        {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "structured_outputs": True,
            "schema": {"type": "object"},
            "request_body_options": {"reasoning_effort": "low"},
        },
        DummyClient(),
    )
    with pytest.raises(HomeAssistantError, match="reasoning options"):
        await structured_body_error_entity._async_generate_data(
            SimpleNamespace(name="task", instructions="Return data", structure=None),
            SimpleNamespace(conversation_id="c"),
        )


@pytest.mark.asyncio
async def test_ai_task_and_conversation_setup_properties():
    entry = DummyEntry()
    entry.subentries = {
        "text": SimpleNamespace(
            subentry_id="text-id",
            data={
                CONF_SERVICE_TYPE: FEATURE_TEXT_GENERATION,
                CONF_NAME: "Assistant",
                CONF_MODEL: "openai/gpt-oss-20b",
            },
        ),
        "unsupported-text": SimpleNamespace(
            subentry_id="unsupported-text-id",
            data={
                CONF_SERVICE_TYPE: FEATURE_TEXT_GENERATION,
                CONF_NAME: "Unsupported Text",
                CONF_MODEL: "whisper-large-v3",
            },
        ),
    }
    runtime = build_runtime(DummyHass(), entry)
    runtime.client = DummyClient()
    entry.runtime_data = runtime

    ai_entities = []
    ai_subentry_ids = []

    def add_ai_entities(entities, **kwargs):
        ai_entities.extend(entities)
        ai_subentry_ids.append(kwargs.get("config_subentry_id"))

    await ai_task_module.async_setup_entry(
        DummyHass(),
        entry,
        add_ai_entities,
    )
    assert len(ai_entities) == 1
    assert ai_subentry_ids == ["text-id"]

    entry.subentries["text"].data[CONF_MODEL] = "llama-3.1-8b-instant"
    ai_entities.clear()
    ai_subentry_ids.clear()
    runtime = build_runtime(DummyHass(), entry)
    runtime.client = DummyClient()
    entry.runtime_data = runtime
    await ai_task_module.async_setup_entry(
        DummyHass(),
        entry,
        add_ai_entities,
    )
    assert len(ai_entities) == 1
    assert ai_subentry_ids == ["text-id"]

    del entry.subentries["unsupported-text"]
    conversation_entities = []
    conversation_subentry_ids = []

    def add_conversation_entities(entities, **kwargs):
        conversation_entities.extend(entities)
        conversation_subentry_ids.append(kwargs.get("config_subentry_id"))

    await conversation_module.async_setup_entry(
        DummyHass(),
        entry,
        add_conversation_entities,
    )
    assert len(conversation_entities) == 1
    assert conversation_subentry_ids == ["text-id"]
    entity = conversation_entities[0]
    assert entity.supported_languages == "*"
    assert entity.device_info["name"] == "Assistant"
    assert entity.has_entity_name is True
    assert entity.translation_key == "assist"
    tiny_registry = GroqModelRegistry(
        [
            GroqModel(
                "tiny-text",
                context_window=20,
                capabilities=frozenset({GroqCapability.TEXT_GENERATION}),
            )
        ],
        include_built_ins=False,
    )
    tiny_conversation = conversation_module.GroqConversationEntity(
        DummyHass(),
        entry,
        {"model": "tiny-text"},
        DummyClient(),
        tiny_registry,
    )
    with pytest.raises(HomeAssistantError, match="context window"):
        await tiny_conversation._async_handle_message(
            SimpleNamespace(
                text="x" * 30,
                language="en",
                agent_id="conversation.groq_assist",
                extra_system_prompt=None,
            ),
            SimpleNamespace(conversation_id="c"),
        )
    body_error_conversation = conversation_module.GroqConversationEntity(
        DummyHass(),
        entry,
        {
            "model": "llama-3.1-8b-instant",
            "request_body_options": {
                "response_format": {"type": "json_object"},
            },
        },
        DummyClient(),
    )
    with pytest.raises(HomeAssistantError, match="response_format"):
        await body_error_conversation._async_handle_message(
            SimpleNamespace(
                text="hello",
                language="en",
                agent_id="conversation.groq_assist",
                extra_system_prompt=None,
            ),
            SimpleNamespace(conversation_id="c"),
        )


@pytest.mark.asyncio
async def test_tts_entity_and_api_remaining_paths(monkeypatch):
    entry = DummyEntry()
    entry.data = {"unique_id": "entry-uid", "model": "m", "voice": "v", "url": "url"}
    client = SimpleNamespace(calls=[])

    async def async_synthesize_speech(request):
        client.calls.append(request)
        return b"audio"

    client.async_synthesize_speech = async_synthesize_speech
    entity = GroqTTSEntity(
        DummyHass(), entry, client, {"name": "tts", "unique_id": "tts-id"}
    )
    assert entity.default_language == "en"
    assert entity.supported_languages == ["ar", "en"]
    assert entity.device_info["identifiers"] == {("groq", "tts-id")}
    assert entity.device_info["name"] == "tts"
    assert entity.has_entity_name is True
    assert entity.translation_key == "text_to_speech"
    fmt, audio = await entity.async_get_tts_audio(
        "hello",
        "en",
        {"vocal_directions": "warm", "normalize_audio": False},
    )
    assert fmt == "wav"
    assert audio == b"audio"
    assert client.calls[0].text == "[warm] hello"
    assert await entity.async_get_tts_audio("x" * 201, "en") == (None, None)

    async def cancelled_tts(request):
        raise asyncio.CancelledError

    cancel_client = SimpleNamespace(async_synthesize_speech=cancelled_tts)
    cancel_entity = GroqTTSEntity(DummyHass(), entry, cancel_client, {})
    assert await cancel_entity.async_get_tts_audio("hello", "en") == (None, None)

    async def missing_ffmpeg(*args, **kwargs):
        raise FileNotFoundError

    ffmpeg_issues = []
    monkeypatch.setattr(
        "custom_components.groq.tts.async_create_ffmpeg_missing_issue",
        lambda hass, entry, service_data: ffmpeg_issues.append(
            (entry.entry_id, service_data.get("unique_id"))
        ),
    )
    monkeypatch.setattr(
        "custom_components.groq.tts.asyncio.create_subprocess_exec",
        missing_ffmpeg,
    )
    assert await entity.async_get_tts_audio(
        "hello",
        "en",
        {"normalize_audio": True},
    ) == (None, None)
    assert ffmpeg_issues == [("entry-id", "tts-id")]

    api_client = GroqApiClient(DummyHass(), api_key="key")
    assert api_client.available is True
    assert api_module._payload_mentions_model_access([]) is False
    assert (
        api_module._payload_mentions_model_access(
            {"error": "Model custom/missing not found"}
        )
        is True
    )
    assert api_client._estimate_tts_token_usage("") == 1
    assert api_client._free_tier_limits("missing") is None
    request = SpeechRequest(
        text="text",
        model="canopylabs/orpheus-v1-english",
        voice="voice",
        protect_free_tier=False,
    )
    assert api_client._check_local_tts_free_tier_limit(request) == 4
    guarded_request = SpeechRequest(
        text="text",
        model="canopylabs/orpheus-v1-english",
        voice="voice",
    )
    state = api_client._tts_usage_state(guarded_request)
    state.request_timestamps.extend([1.0, 100000.0])
    state.token_timestamps.extend([(1.0, 1), (100000.0, 2)])
    api_client._prune_local_tts_usage(state, 100000.0)
    assert list(state.request_timestamps) == [100000.0]

    monkeypatch.setattr(
        api_client,
        "_free_tier_limits",
        lambda model: {
            "requests_per_minute": 0,
            "requests_per_day": 10,
            "tokens_per_minute": 10,
            "tokens_per_day": 10,
        },
    )
    with pytest.raises(GroqApiError):
        api_client._check_local_tts_free_tier_limit(guarded_request)

    for limits in (
        {
            "requests_per_minute": 10,
            "requests_per_day": 0,
            "tokens_per_minute": 10,
            "tokens_per_day": 10,
        },
        {
            "requests_per_minute": 10,
            "requests_per_day": 10,
            "tokens_per_minute": 1,
            "tokens_per_day": 10,
        },
        {
            "requests_per_minute": 10,
            "requests_per_day": 10,
            "tokens_per_minute": 10,
            "tokens_per_day": 1,
        },
    ):
        guarded = GroqApiClient(DummyHass(), api_key="key")
        monkeypatch.setattr(
            guarded, "_free_tier_limits", lambda model, limits=limits: limits
        )
        with pytest.raises(GroqApiError):
            guarded._check_local_tts_free_tier_limit(guarded_request)

    tts_http = GroqApiClient(
        DummyHass(),
        api_key="key",
        session=DummyPostSession(PostResponse(200, b"audio")),
    )
    speech_request = SpeechRequest(
        text="hello",
        model="model",
        voice="voice",
        protect_free_tier=False,
    )
    assert await tts_http.async_synthesize_speech(speech_request) == b"audio"
    assert await tts_http.async_synthesize_speech(speech_request) == b"audio"

    error_cases = [
        (
            PostResponse(401, {"error": "auth"}, {"content-type": "application/json"}),
            ConfigEntryAuthFailed,
        ),
        (
            PostResponse(
                429,
                {"error": "rate"},
                {"retry-after": "1", "content-type": "application/json"},
            ),
            GroqRateLimitExceeded,
        ),
        (
            PostResponse(
                500, {"error": {"message": "bad"}}, {"content-type": "application/json"}
            ),
            HomeAssistantError,
        ),
        (
            PostResponse(500, b"{", {"content-type": "application/json"}),
            HomeAssistantError,
        ),
        (
            PostResponse(500, b"bad", {"content-type": "text/plain"}),
            HomeAssistantError,
        ),
        (
            PostResponse(
                200,
                {"error": {"message": "embedded"}},
                {"content-type": "application/json"},
            ),
            HomeAssistantError,
        ),
        (
            PostResponse(200, {"ok": True}, {"content-type": "application/json"}),
            HomeAssistantError,
        ),
        (
            PostResponse(200, b"{", {"content-type": "application/json"}),
            HomeAssistantError,
        ),
        (
            PostResponse(200, b"text", {"content-type": "text/plain"}),
            HomeAssistantError,
        ),
    ]
    for response, error_type in error_cases:
        failing = GroqApiClient(
            DummyHass(), api_key="key", session=DummyPostSession(response)
        )
        with pytest.raises(error_type):
            await failing.async_synthesize_speech(speech_request)

    model_access_issues = []
    monkeypatch.setattr(
        "custom_components.groq.api.async_create_model_access_issue",
        lambda hass, model: model_access_issues.append((hass, model)),
    )
    model_access = GroqApiClient(
        DummyHass(),
        api_key="key",
        session=DummyPostSession(
            PostResponse(
                400,
                {"error": {"message": "Model model is not available"}},
                {"content-type": "application/json"},
            )
        ),
    )
    with pytest.raises(GroqApiError, match="not available"):
        await model_access.async_synthesize_speech(speech_request)
    assert model_access_issues == [(model_access_issues[0][0], "model")]

    cancelled = GroqApiClient(
        DummyHass(),
        api_key="key",
        session=DummyPostSession(asyncio.CancelledError()),
    )
    with pytest.raises(asyncio.CancelledError):
        await cancelled.async_synthesize_speech(speech_request)
