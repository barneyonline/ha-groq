from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp
import pytest
from homeassistant import data_entry_flow
from homeassistant.const import Platform
from homeassistant.exceptions import HomeAssistantError

import custom_components.groq as integration
from custom_components.groq import config_flow, tts_engine, tts
from custom_components.groq.const import (
    DEFAULT_TTS_URL,
    FEATURE_IMAGE_RECOGNITION,
    FEATURE_SPEECH_TO_TEXT,
    FEATURE_TEXT_GENERATION,
    FEATURE_TEXT_TO_SPEECH,
)
from custom_components.groq.tts_engine import GroqTTSEngine
from custom_components.groq.tts import GroqTTSEntity

ORPHEUS_ENGLISH_MODEL = "canopylabs/orpheus-v1-english"
ORPHEUS_ENGLISH_VOICE = "troy"


class DummyHass:
    pass


class DummyResponse:
    def __init__(self, status: int, headers: dict[str, str], body: bytes):
        self.status = status
        self.headers = headers
        self._body = body

    async def read(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummyCaptureSession:
    def __init__(self):
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return DummyResponse(200, {"content-type": "audio/wav"}, b"RIFF....WAVEfmt ")


class DummyConfigEntry:
    def __init__(
        self,
        data: dict,
        options: dict,
        *,
        unique_id: str | None = "uid",
    ):
        self.data = data
        self.options = options
        self.unique_id = unique_id
        self.entry_id = "entry-id"

    def add_update_listener(self, listener):
        self.listener = listener
        return "unsub"

    def async_on_unload(self, unsub):
        self.unsub = unsub


class DummyEngine:
    class Response:
        content = b"audio-bytes"

    def __init__(self):
        self.calls = []

    async def async_get_tts(
        self, hass, text, voice=None, model=None, response_format=None
    ):
        self.calls.append(
            {
                "text": text,
                "voice": voice,
                "model": model,
                "response_format": response_format,
            }
        )
        return self.Response()

    @staticmethod
    def get_supported_langs():
        return ["ar", "en"]


@pytest.mark.asyncio
async def test_validate_user_input_accepts_account_features_and_requires_api_key():
    await config_flow.validate_user_input(
        {
            "api_key": "api-key",
            "enabled_features": ["text_to_speech"],
        }
    )

    with pytest.raises(ValueError, match="API key is required"):
        await config_flow.validate_user_input({"enabled_features": ["text_to_speech"]})


@pytest.mark.asyncio
async def test_validate_user_input_rejects_unknown_enabled_features():
    with pytest.raises(ValueError, match="Enabled features are invalid"):
        await config_flow.validate_user_input(
            {
                "api_key": "api-key",
                "enabled_features": ["text_to_speech", "unknown"],
            }
        )


class DummyGetResponse:
    status = 200

    async def json(self):
        return {"data": [{"id": "model-a"}, {"name": "model-b"}, "model-c", {}]}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummyGetSession:
    def __init__(self):
        self.calls = []

    def get(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return DummyGetResponse()


@pytest.mark.asyncio
async def test_fetch_available_extracts_models_and_auth_header():
    session = DummyGetSession()

    with patch.object(config_flow, "async_get_clientsession", return_value=session):
        models = await config_flow.fetch_available(
            DummyHass(), "https://api.groq.com/openai/v1/models", "api-key"
        )

    assert models == ["model-a", "model-b", "model-c"]
    assert session.calls[0]["kwargs"]["headers"] == {"Authorization": "Bearer api-key"}


@pytest.mark.asyncio
async def test_fetch_available_returns_empty_on_client_error():
    class ErrorSession:
        def get(self, *args, **kwargs):
            raise aiohttp.ClientError("boom")

    with patch.object(
        config_flow, "async_get_clientsession", return_value=ErrorSession()
    ):
        assert (
            await config_flow.fetch_available(DummyHass(), "https://example.com") == []
        )


@pytest.mark.asyncio
async def test_get_dynamic_options_filters_discovered_models(monkeypatch):
    async def fake_fetch_available(hass, endpoint, api_key=None):
        return ["llama-3.3-70b-versatile", "canopylabs/orpheus-custom"]

    monkeypatch.setattr(config_flow, "fetch_available", fake_fetch_available)

    models, voices = await config_flow.get_dynamic_options(DummyHass(), "api-key")

    assert "canopylabs/orpheus-custom" in models
    assert "llama-3.3-70b-versatile" not in models
    assert ORPHEUS_ENGLISH_VOICE in voices


@pytest.mark.asyncio
async def test_async_get_tts_uses_cache_and_evicts_lru():
    session = DummyCaptureSession()
    engine = GroqTTSEngine(
        "api-key",
        ORPHEUS_ENGLISH_VOICE,
        ORPHEUS_ENGLISH_MODEL,
        "https://api.groq.com/openai/v1/audio/speech",
        cache_max=1,
    )

    with patch.object(tts_engine, "async_get_clientsession", return_value=session):
        first = await engine.async_get_tts(DummyHass(), "hello")
        cached = await engine.async_get_tts(DummyHass(), "hello")
        second = await engine.async_get_tts(DummyHass(), "new")

    assert first.content == cached.content == second.content
    assert len(session.calls) == 2
    assert list(engine._cache) == [
        (ORPHEUS_ENGLISH_MODEL, ORPHEUS_ENGLISH_VOICE, "wav", "new")
    ]
    assert engine.close() is None
    assert engine.get_supported_langs() == ["ar", "en"]


class Dummy500JsonSession:
    def post(self, *args, **kwargs):
        return DummyResponse(
            500,
            {"content-type": "application/json"},
            b'{"error": {"message": "bad model"}}',
        )


@pytest.mark.asyncio
async def test_async_get_tts_raises_json_http_error():
    engine = GroqTTSEngine(None, "voice", "model", "http://example.com")
    with patch.object(
        tts_engine, "async_get_clientsession", return_value=Dummy500JsonSession()
    ):
        with pytest.raises(HomeAssistantError, match="HTTP 500"):
            await engine.async_get_tts(DummyHass(), "hello")


def test_tts_entity_properties_use_options_over_data():
    data = {
        "url": "http://example.com",
        "model": "data-model",
        "voice": "data-voice",
    }
    options = {"model": ORPHEUS_ENGLISH_MODEL, "voice": ORPHEUS_ENGLISH_VOICE}
    entity = GroqTTSEntity(
        DummyHass(), DummyConfigEntry(data, options, unique_id=None), DummyEngine()
    )

    assert entity.unique_id == "http://example.com_data-model"
    assert entity.default_language == "en"
    assert entity.supported_options == [
        "input",
        "model",
        "normalize_audio",
        "response_format",
        "voice",
        "vocal_directions",
    ]
    assert entity.default_options["voice"] == ORPHEUS_ENGLISH_VOICE
    assert entity.default_options["model"] == ORPHEUS_ENGLISH_MODEL
    assert entity.default_options["response_format"] == "wav"
    assert entity.default_options["vocal_directions"] == ""
    assert entity.supported_languages == ["ar", "en"]
    assert entity.device_info["model"] == ORPHEUS_ENGLISH_MODEL
    assert entity.name == ORPHEUS_ENGLISH_MODEL.upper()


class DummyProc:
    def __init__(self, returncode: int):
        self.returncode = returncode

    async def communicate(self, input=None):  # noqa: A002
        if self.returncode == 0:
            return b"processed-audio", b""
        return b"", b"ffmpeg error"


@pytest.mark.asyncio
async def test_tts_normalize_runs_ffmpeg_and_returns_mp3(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), DummyEngine())
    commands = []

    async def fake_exec(*args, **kwargs):  # noqa: ANN001
        commands.append(args)
        return DummyProc(returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    ext, payload = await entity.async_get_tts_audio(
        "Hello", "en", options={"normalize_audio": True}
    )

    assert ext == "mp3"
    assert payload == b"processed-audio"
    assert "-af" in commands[0]


@pytest.mark.asyncio
async def test_tts_service_options_override_groq_speech_payload():
    data = {
        "url": "http://example.com",
        "model": "data-model",
        "voice": "data-voice",
        "response_format": "wav",
        "unique_id": "uid",
    }
    engine = DummyEngine()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), engine)

    ext, payload = await entity.async_get_tts_audio(
        "service message",
        "en",
        options={
            "input": "override input",
            "model": ORPHEUS_ENGLISH_MODEL,
            "voice": ORPHEUS_ENGLISH_VOICE,
            "response_format": "wav",
            "vocal_directions": "cheerful",
        },
    )

    assert ext == "wav"
    assert payload == b"audio-bytes"
    assert engine.calls == [
        {
            "text": "[cheerful] override input",
            "voice": ORPHEUS_ENGLISH_VOICE,
            "model": ORPHEUS_ENGLISH_MODEL,
            "response_format": "wav",
        }
    ]


@pytest.mark.asyncio
async def test_tts_async_setup_entry_builds_engine_with_options():
    data = {
        "api_key": "data-key",
        "url": "data-url",
        "model": "data-model",
        "voice": "data-voice",
    }
    options = {
        "api_key": "option-key",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "url": "option-url",
        "cache_size": 12,
        "protect_free_tier": False,
    }
    added = []

    await tts.async_setup_entry(
        DummyHass(),
        DummyConfigEntry(data, options),
        lambda entities: added.extend(entities),
    )

    assert len(added) == 1
    engine = added[0]._engine
    assert engine._api_key == "option-key"
    assert engine._url == "option-url"
    assert engine._cache_max == 12
    assert engine._protect_free_tier is False
    assert engine._response_format == "wav"


@pytest.mark.asyncio
async def test_tts_async_setup_entry_skips_account_only_entries():
    added = []

    await tts.async_setup_entry(
        DummyHass(),
        DummyConfigEntry({"api_key": "data-key"}, {}),
        lambda entities: added.extend(entities),
    )

    assert added == []


@pytest.mark.asyncio
async def test_tts_async_setup_entry_builds_entities_from_subentries():
    entry = DummyConfigEntry({"api_key": "data-key"}, {})
    entry.subentries = {
        "subentry-id": SimpleNamespace(
            subentry_id="subentry-id",
            data={
                "service_type": "text_to_speech",
                "name": "Kitchen TTS",
                "model": ORPHEUS_ENGLISH_MODEL,
                "voice": ORPHEUS_ENGLISH_VOICE,
                "response_format": "wav",
                "vocal_directions": "warm",
            },
        )
    }
    added = []

    await tts.async_setup_entry(
        DummyHass(),
        entry,
        lambda entities: added.extend(entities),
    )

    assert len(added) == 1
    assert added[0].unique_id == "subentry-id"
    assert added[0].name == "KITCHEN TTS"
    assert added[0]._engine._url == DEFAULT_TTS_URL
    assert added[0]._engine._model == ORPHEUS_ENGLISH_MODEL
    assert added[0]._engine._voice == ORPHEUS_ENGLISH_VOICE


class DummyConfigEntries:
    def __init__(self):
        self.forwarded = []
        self.unloaded = []
        self.reloaded = []
        self.updated = []
        self.entry = None

    async def async_forward_entry_setups(self, entry, platforms):
        self.forwarded.append((entry, platforms))

    async def async_unload_platforms(self, entry, platforms):
        self.unloaded.append((entry, platforms))
        return True

    async def async_reload(self, entry_id):
        self.reloaded.append(entry_id)

    def async_update_entry(self, entry, **kwargs):
        self.updated.append((entry, kwargs))

    def async_get_entry(self, entry_id):
        return self.entry


@pytest.mark.asyncio
async def test_integration_setup_unload_update_and_migration():
    config_entries = DummyConfigEntries()
    hass = SimpleNamespace(config_entries=config_entries)
    entry = DummyConfigEntry(
        {
            "unique_id": "legacy-id",
            "url": "https://api.groq.com/openai/v1/audio/speech",
            "model": "model",
            "voice": "voice",
        },
        {},
        unique_id=None,
    )

    assert await integration.async_setup_entry(hass, entry) is True
    assert entry.unsub == "unsub"
    assert config_entries.forwarded == [(entry, [Platform.TTS])]

    assert await integration.async_unload_entry(hass, entry) is True
    assert config_entries.unloaded == [(entry, [Platform.TTS])]

    await integration._async_update_listener(hass, entry)
    assert config_entries.reloaded == ["entry-id"]

    assert await integration.async_migrate_entry(hass, entry) is True
    assert config_entries.updated == [
        (
            entry,
            {
                "data": {
                    "url": "https://api.groq.com/openai/v1/audio/speech",
                    "model": "model",
                    "voice": "voice",
                },
                "unique_id": "legacy-id",
            },
        )
    ]


def _patch_flow_common(monkeypatch, flow, hass=None):
    flow.hass = hass or DummyHass()

    def show_form(**kwargs):
        return {"type": "form", **kwargs}

    def show_menu(**kwargs):
        return {"type": "menu", **kwargs}

    def create_entry(**kwargs):
        return {"type": "create_entry", **kwargs}

    def update_and_abort(entry, subentry, **kwargs):
        return {
            "type": "abort",
            "reason": "reconfigure_successful",
            "entry": entry,
            "subentry": subentry,
            **kwargs,
        }

    monkeypatch.setattr(flow, "async_show_form", show_form)
    monkeypatch.setattr(flow, "async_show_menu", show_menu)
    monkeypatch.setattr(flow, "async_create_entry", create_entry)
    monkeypatch.setattr(
        flow,
        "async_update_and_abort",
        update_and_abort,
        raising=False,
    )


@pytest.mark.asyncio
async def test_config_flow_user_success_and_error(monkeypatch):
    async def fake_dynamic_options(hass, api_key):
        return [ORPHEUS_ENGLISH_MODEL], [ORPHEUS_ENGLISH_VOICE]

    monkeypatch.setattr(config_flow, "get_dynamic_options", fake_dynamic_options)
    flow = config_flow.GroqConfigFlow()
    _patch_flow_common(monkeypatch, flow)
    unique_ids = []

    async def set_unique_id(unique_id):
        unique_ids.append(unique_id)

    monkeypatch.setattr(flow, "async_set_unique_id", set_unique_id)
    monkeypatch.setattr(flow, "_abort_if_unique_id_configured", lambda: None)

    result = await flow.async_step_user(
        {
            "name": "Groq Text Account",
            "api_key": "api-key",
        }
    )

    assert result["type"] == "create_entry"
    assert result["title"] == "Groq Text Account"
    assert result["data"]["unique_id"] == unique_ids[0]
    assert result["data"]["name"] == "Groq Text Account"
    assert result["data"]["api_key"] == "api-key"
    assert "enabled_features" not in result["data"]
    assert "url" not in result["data"]
    assert "model" not in result["data"]
    assert "voice" not in result["data"]

    error_result = await flow.async_step_user({"enabled_features": ["text_to_speech"]})
    assert error_result["errors"] == {"api_key": "required"}


@pytest.mark.asyncio
async def test_config_flow_user_aborts_duplicate(monkeypatch):
    async def fake_dynamic_options(hass, api_key):
        return [ORPHEUS_ENGLISH_MODEL], [ORPHEUS_ENGLISH_VOICE]

    monkeypatch.setattr(config_flow, "get_dynamic_options", fake_dynamic_options)
    flow = config_flow.GroqConfigFlow()
    _patch_flow_common(monkeypatch, flow)

    async def set_unique_id(unique_id):
        return None

    def abort_duplicate():
        raise data_entry_flow.AbortFlow("already_configured")

    monkeypatch.setattr(flow, "async_set_unique_id", set_unique_id)
    monkeypatch.setattr(flow, "_abort_if_unique_id_configured", abort_duplicate)
    monkeypatch.setattr(
        flow, "async_abort", lambda **kwargs: {"type": "abort", **kwargs}
    )

    result = await flow.async_step_user(
        {
            "api_key": "api-key",
        }
    )

    assert result == {"type": "abort", "reason": "already_configured"}


@pytest.mark.asyncio
async def test_config_flow_user_defaults_account_name(monkeypatch):
    flow = config_flow.GroqConfigFlow()
    _patch_flow_common(monkeypatch, flow)

    async def set_unique_id(unique_id):
        return None

    monkeypatch.setattr(flow, "async_set_unique_id", set_unique_id)
    monkeypatch.setattr(flow, "_abort_if_unique_id_configured", lambda: None)

    result = await flow.async_step_user({"api_key": "api-key"})

    assert result["type"] == "create_entry"
    assert result["title"] == "Groq"
    assert result["data"]["name"] == "Groq"


def test_config_flow_exposes_dedicated_service_subentry_types():
    supported = config_flow.GroqConfigFlow.async_get_supported_subentry_types(
        SimpleNamespace()
    )

    assert supported == {
        FEATURE_TEXT_GENERATION: config_flow.GroqServiceSubentryFlow,
        FEATURE_SPEECH_TO_TEXT: config_flow.GroqServiceSubentryFlow,
        FEATURE_TEXT_TO_SPEECH: config_flow.GroqServiceSubentryFlow,
        FEATURE_IMAGE_RECOGNITION: config_flow.GroqServiceSubentryFlow,
    }
    assert "service" not in supported


@pytest.mark.asyncio
async def test_text_to_speech_subentry_flow_creates_service_with_api_key(monkeypatch):
    flow = config_flow.GroqServiceSubentryFlow()
    flow.handler = ("entry-id", FEATURE_TEXT_TO_SPEECH)

    async def async_add_executor_job(func):
        return func()

    flow.hass = SimpleNamespace(async_add_executor_job=async_add_executor_job)
    _patch_flow_common(monkeypatch, flow, flow.hass)

    form = await flow.async_step_user()
    assert form["type"] == "form"
    assert form["step_id"] == FEATURE_TEXT_TO_SPEECH

    result = await flow.async_step_user(
        {
            "name": "Kitchen TTS",
            "api_key": "tts-api-key",
            "model": ORPHEUS_ENGLISH_MODEL,
            "voice": ORPHEUS_ENGLISH_VOICE,
            "response_format": "wav",
            "vocal_directions": "warm",
            "normalize_audio": False,
            "protect_free_tier": True,
        }
    )

    assert result == {
        "type": "create_entry",
        "title": "Kitchen TTS",
        "data": {
            "name": "Kitchen TTS",
            "api_key": "tts-api-key",
            "model": ORPHEUS_ENGLISH_MODEL,
            "voice": ORPHEUS_ENGLISH_VOICE,
            "response_format": "wav",
            "vocal_directions": "warm",
            "normalize_audio": False,
            "protect_free_tier": True,
            "service_type": FEATURE_TEXT_TO_SPEECH,
        },
    }


@pytest.mark.asyncio
async def test_text_to_speech_subentry_reconfigure_preserves_api_key(monkeypatch):
    flow = config_flow.GroqServiceSubentryFlow()
    flow.handler = ("entry-id", FEATURE_TEXT_TO_SPEECH)
    flow.context = {"source": "reconfigure", "subentry_id": "subentry-id"}
    entry = SimpleNamespace(entry_id="entry-id")
    subentry = SimpleNamespace(
        data={
            "name": "Old TTS",
            "api_key": "old-service-key",
            "model": ORPHEUS_ENGLISH_MODEL,
            "voice": ORPHEUS_ENGLISH_VOICE,
            "response_format": "wav",
            "vocal_directions": "warm",
            "normalize_audio": True,
            "protect_free_tier": True,
            "service_type": FEATURE_TEXT_TO_SPEECH,
        }
    )
    _patch_flow_common(monkeypatch, flow)
    monkeypatch.setattr(flow, "_get_entry", lambda: entry)
    monkeypatch.setattr(flow, "_get_reconfigure_subentry", lambda: subentry)

    form = await flow.async_step_reconfigure()
    assert form["type"] == "form"
    assert form["step_id"] == FEATURE_TEXT_TO_SPEECH

    result = await flow.async_step_reconfigure(
        {
            "name": "Updated TTS",
            "model": ORPHEUS_ENGLISH_MODEL,
            "voice": ORPHEUS_ENGLISH_VOICE,
            "response_format": "wav",
            "vocal_directions": "",
            "normalize_audio": False,
            "protect_free_tier": True,
        }
    )

    assert result["type"] == "abort"
    assert result["reason"] == "reconfigure_successful"
    assert result["title"] == "Updated TTS"
    assert result["data"]["api_key"] == "old-service-key"
    assert result["data"]["name"] == "Updated TTS"
    assert result["data"]["vocal_directions"] == ""
    assert result["data"]["normalize_audio"] is False
    assert result["data"]["service_type"] == FEATURE_TEXT_TO_SPEECH


@pytest.mark.asyncio
async def test_text_generation_subentry_flow_uses_advanced_step(monkeypatch):
    flow = config_flow.GroqServiceSubentryFlow()
    _patch_flow_common(monkeypatch, flow)

    basic = await flow.async_step_text_generation()
    assert basic["type"] == "form"
    assert basic["step_id"] == "text_generation"

    advanced = await flow.async_step_text_generation(
        {
            "name": "Text API Key",
            "api_key": "text-api-key",
            "model": "openai/gpt-oss-20b",
            "system_prompt": "You are helpful.",
            "temperature": 0.2,
            "advanced_options": True,
        }
    )

    assert advanced["type"] == "form"
    assert advanced["step_id"] == "text_generation_advanced"

    result = await flow.async_step_text_generation_advanced(
        {
            "max_tokens": 256,
            "top_p": 0.8,
            "service_tier": "flex",
            "reasoning_effort": "medium",
            "prompt_caching": True,
            "request_body_options": {
                "citation_options": "disabled",
                "search_settings": {"include_domains": ["example.com"]},
            },
        }
    )

    assert result == {
        "type": "create_entry",
        "title": "Text API Key",
        "data": {
            "name": "Text API Key",
            "api_key": "text-api-key",
            "model": "openai/gpt-oss-20b",
            "system_prompt": "You are helpful.",
            "temperature": 0.2,
            "max_tokens": 256,
            "top_p": 0.8,
            "service_tier": "flex",
            "reasoning_effort": "medium",
            "prompt_caching": True,
            "request_body_options": {
                "citation_options": "disabled",
                "search_settings": {"include_domains": ["example.com"]},
            },
            "service_type": "text_generation",
        },
    }


@pytest.mark.asyncio
async def test_text_generation_reconfigure_keeps_advanced_defaults_until_edited(
    monkeypatch,
):
    flow = config_flow.GroqServiceSubentryFlow()
    flow.handler = ("entry-id", FEATURE_TEXT_GENERATION)
    flow.context = {"source": "reconfigure", "subentry_id": "subentry-id"}
    entry = SimpleNamespace(entry_id="entry-id")
    subentry = SimpleNamespace(
        data={
            "name": "Existing Text",
            "api_key": "old-text-key",
            "model": "openai/gpt-oss-20b",
            "system_prompt": "Existing prompt.",
            "temperature": 0.3,
            "max_tokens": 512,
            "top_p": 0.7,
            "service_tier": "flex",
            "prompt_caching": True,
            "request_body_options": {"citation_options": "disabled"},
            "service_type": FEATURE_TEXT_GENERATION,
        }
    )
    _patch_flow_common(monkeypatch, flow)
    monkeypatch.setattr(flow, "_get_entry", lambda: entry)
    monkeypatch.setattr(flow, "_get_reconfigure_subentry", lambda: subentry)

    advanced = await flow.async_step_reconfigure(
        {
            "name": "Updated Text",
            "model": "openai/gpt-oss-20b",
            "system_prompt": "Updated prompt.",
            "temperature": 0.4,
            "advanced_options": True,
        }
    )

    assert advanced["type"] == "form"
    assert advanced["step_id"] == "text_generation_advanced"
    assert flow._pending_service_data["max_tokens"] == 512
    assert flow._pending_service_data["request_body_options"] == {
        "citation_options": "disabled"
    }

    result = await flow.async_step_text_generation_advanced(
        {
            "max_tokens": 256,
            "top_p": 0.9,
            "service_tier": "",
            "prompt_caching": False,
            "request_body_options": {},
        }
    )

    assert result["type"] == "abort"
    assert result["reason"] == "reconfigure_successful"
    assert result["title"] == "Updated Text"
    assert result["data"]["api_key"] == "old-text-key"
    assert result["data"]["max_tokens"] == 256
    assert result["data"]["top_p"] == 0.9
    assert "service_tier" not in result["data"]
    assert "prompt_caching" in result["data"]
    assert "request_body_options" not in result["data"]
    assert result["data"]["service_type"] == FEATURE_TEXT_GENERATION


@pytest.mark.asyncio
async def test_config_flow_reauth_confirm(monkeypatch):
    config_entries = DummyConfigEntries()
    reauth_entry = DummyConfigEntry({"api_key": "old", "model": "model"}, {})
    config_entries.entry = reauth_entry
    flow = config_flow.GroqConfigFlow()
    flow.hass = SimpleNamespace(config_entries=config_entries)
    flow.context = {"entry_id": "entry-id"}
    _patch_flow_common(monkeypatch, flow, flow.hass)
    monkeypatch.setattr(
        flow,
        "async_update_reload_and_abort",
        lambda entry, **kwargs: {"type": "abort", "entry": entry, **kwargs},
    )

    missing = await flow.async_step_reauth_confirm({})
    assert missing["errors"] == {"api_key": "required"}

    result = await flow.async_step_reauth({"ignored": True})
    assert result["type"] == "form"

    result = await flow.async_step_reauth_confirm({"api_key": "new"})
    assert result["entry"] is reauth_entry
    assert result["data_updates"]["api_key"] == "new"


@pytest.mark.asyncio
async def test_options_flow_shows_schema_and_saves(monkeypatch):
    async def fake_dynamic_options(hass, api_key):
        return [ORPHEUS_ENGLISH_MODEL], [ORPHEUS_ENGLISH_VOICE]

    monkeypatch.setattr(config_flow, "get_dynamic_options", fake_dynamic_options)
    flow = config_flow.GroqOptionsFlow()
    entry = DummyConfigEntry(
        {"api_key": "data-key", "model": ORPHEUS_ENGLISH_MODEL}, {}
    )
    flow.handler = entry.entry_id

    async def async_add_executor_job(func):
        return func()

    flow.hass = SimpleNamespace(
        async_add_executor_job=async_add_executor_job,
        config_entries=SimpleNamespace(
            async_get_known_entry=lambda entry_id: entry,
        ),
    )
    _patch_flow_common(monkeypatch, flow, flow.hass)

    form = await flow.async_step_init()
    assert form["type"] == "form"
    assert form["step_id"] == "init"

    saved = await flow.async_step_init(
        {
            "voice": ORPHEUS_ENGLISH_VOICE,
            "enabled_features": ["text_to_speech", "image_recognition"],
        }
    )
    assert saved == {
        "type": "create_entry",
        "title": "",
        "data": {
            "voice": ORPHEUS_ENGLISH_VOICE,
            "enabled_features": ["text_to_speech", "image_recognition"],
        },
    }
