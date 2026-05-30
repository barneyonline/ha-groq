from __future__ import annotations

import asyncio
from collections import OrderedDict
import json
import struct
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp
import pytest
from homeassistant import data_entry_flow
from homeassistant.const import CONF_LLM_HASS_API, Platform

import custom_components.groq as integration
from custom_components.groq import config_flow, tts
from custom_components.groq.api import GroqApiClient, SpeechRequest
from custom_components.groq.const import (
    FEATURE_IMAGE_RECOGNITION,
    FEATURE_SPEECH_TO_TEXT,
    FEATURE_TEXT_GENERATION,
    FEATURE_TEXT_TO_SPEECH,
    RESPONSE_FORMATS,
)
from custom_components.groq.errors import GroqApiError
from custom_components.groq.model_registry import model_from_api
from custom_components.groq.tts import FFMPEG_OUTPUT_ARGS, GroqTTSEntity

ORPHEUS_ENGLISH_MODEL = "canopylabs/orpheus-v1-english"
ORPHEUS_ENGLISH_VOICE = "troy"
ORPHEUS_ARABIC_MODEL = "canopylabs/orpheus-arabic-saudi"
ORPHEUS_ARABIC_VOICE = "aisha"
PCM_WAV_BYTES = (
    b"RIFF"
    + struct.pack("<I", 40)
    + b"WAVEfmt "
    + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
    + b"data"
    + struct.pack("<I", 4)
    + b"\0\0\0\0"
)


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

    def request(self, *args, **kwargs):
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


class DummyClient:
    def __init__(self):
        self.calls = []

    async def async_synthesize_speech(self, request):
        self.calls.append(
            {
                "text": request.text,
                "voice": request.voice,
                "model": request.model,
                "response_format": request.response_format,
                "cache_max": request.cache_max,
                "protect_free_tier": request.protect_free_tier,
            }
        )
        return PCM_WAV_BYTES


def _selector_config(schema, field):
    """Return the selector config for a voluptuous schema field."""
    for key, value in schema.schema.items():
        if getattr(key, "schema", key) == field:
            return value.config
    raise AssertionError(f"{field} not found in schema")


def test_new_account_unique_id_uses_groq_prefix():
    assert config_flow._new_account_unique_id().startswith("groq_")


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
    def __init__(self, status: int = 200, payload=None):
        self.status = status
        self.headers = {"content-type": "application/json"}
        self._payload = payload or {
            "data": [{"id": "model-a"}, {"name": "model-b"}, "model-c", {}]
        }

    async def json(self):
        return self._payload

    async def read(self):
        return json.dumps(self._payload).encode()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummyGetSession:
    def __init__(self, response: DummyGetResponse | None = None):
        self.calls = []
        self._response = response or DummyGetResponse()

    def get(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self._response

    def request(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self._response


@pytest.mark.asyncio
async def test_fetch_available_extracts_models_and_auth_header():
    session = DummyGetSession()

    with patch.object(config_flow, "async_get_clientsession", return_value=session):
        models = await config_flow.fetch_available(
            DummyHass(), "https://api.groq.com/openai/v1/models", "api-key"
        )

    assert models == ["model-a", "model-b", "model-c"]
    assert session.calls[0]["kwargs"]["headers"]["Authorization"] == "Bearer api-key"


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
async def test_async_validate_api_key_accepts_valid_key():
    session = DummyGetSession()

    with patch.object(config_flow, "async_get_clientsession", return_value=session):
        assert await config_flow.async_validate_api_key(DummyHass(), "api-key") is None

    assert session.calls[0]["kwargs"]["headers"]["Authorization"] == "Bearer api-key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (DummyGetResponse(status=401), "invalid_auth"),
        (DummyGetResponse(status=403), "invalid_auth"),
        (DummyGetResponse(status=500), "unknown"),
        (DummyGetResponse(payload={"models": []}), "unknown"),
    ],
)
async def test_async_validate_api_key_maps_error_responses(response, expected):
    with patch.object(
        config_flow,
        "async_get_clientsession",
        return_value=DummyGetSession(response),
    ):
        assert (
            await config_flow.async_validate_api_key(DummyHass(), "api-key") == expected
        )


@pytest.mark.asyncio
async def test_async_validate_api_key_maps_connection_errors():
    class ErrorSession:
        def get(self, *args, **kwargs):
            raise aiohttp.ClientError("boom")

        def request(self, *args, **kwargs):
            raise aiohttp.ClientError("boom")

    with patch.object(
        config_flow,
        "async_get_clientsession",
        return_value=ErrorSession(),
    ):
        assert (
            await config_flow.async_validate_api_key(DummyHass(), "api-key")
            == "cannot_connect"
        )


@pytest.mark.asyncio
async def test_get_dynamic_options_filters_discovered_models(monkeypatch):
    async def fake_fetch_available_models(hass, api_key):
        return [
            model_from_api({"id": "llama-3.3-70b-versatile"}),
            model_from_api({"id": "playai-tts"}),
            model_from_api({"id": "canopylabs/orpheus-custom"}),
        ]

    monkeypatch.setattr(
        config_flow,
        "async_fetch_available_models",
        fake_fetch_available_models,
    )

    models, voices = await config_flow.get_dynamic_options(DummyHass(), "api-key")

    assert "canopylabs/orpheus-custom" in models
    assert "playai-tts" not in models
    assert "llama-3.3-70b-versatile" not in models
    assert ORPHEUS_ENGLISH_VOICE in voices


@pytest.mark.asyncio
async def test_async_get_tts_uses_cache_and_evicts_lru():
    session = DummyCaptureSession()
    client = GroqApiClient(DummyHass(), api_key="api-key", session=session)

    first = await client.async_synthesize_speech(
        SpeechRequest(
            text="hello",
            model=ORPHEUS_ENGLISH_MODEL,
            voice=ORPHEUS_ENGLISH_VOICE,
            cache_max=1,
        )
    )
    cached = await client.async_synthesize_speech(
        SpeechRequest(
            text="hello",
            model=ORPHEUS_ENGLISH_MODEL,
            voice=ORPHEUS_ENGLISH_VOICE,
            cache_max=1,
        )
    )
    second = await client.async_synthesize_speech(
        SpeechRequest(
            text="new",
            model=ORPHEUS_ENGLISH_MODEL,
            voice=ORPHEUS_ENGLISH_VOICE,
            cache_max=1,
        )
    )

    assert first == cached == second
    assert len(session.calls) == 2
    cache = client._speech_caches[f"{ORPHEUS_ENGLISH_MODEL}:{ORPHEUS_ENGLISH_VOICE}"]
    assert list(cache) == [(ORPHEUS_ENGLISH_MODEL, ORPHEUS_ENGLISH_VOICE, "wav", "new")]


class Dummy500JsonSession:
    def request(self, *args, **kwargs):
        return DummyResponse(
            500,
            {"content-type": "application/json"},
            b'{"error": {"message": "bad model"}}',
        )


@pytest.mark.asyncio
async def test_async_get_tts_raises_json_http_error():
    client = GroqApiClient(DummyHass(), api_key=None, session=Dummy500JsonSession())
    with pytest.raises(GroqApiError, match="HTTP 500"):
        await client.async_synthesize_speech(
            SpeechRequest(text="hello", model="model", voice="voice")
        )


def test_tts_entity_properties_use_options_over_data():
    data = {
        "url": "http://example.com",
        "model": "data-model",
        "voice": "data-voice",
        "normalize_audio": "yes",
        "response_format": " MP3 ",
    }
    options = {
        "model": ORPHEUS_ENGLISH_MODEL,
        "normalize_audio": True,
        "voice": ORPHEUS_ENGLISH_VOICE,
    }
    entity = GroqTTSEntity(
        DummyHass(), DummyConfigEntry(data, options, unique_id=None), DummyClient()
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
    assert "enable_long_tts" not in entity.default_options
    assert entity.default_options["voice"] == ORPHEUS_ENGLISH_VOICE
    assert entity.default_options["model"] == ORPHEUS_ENGLISH_MODEL
    assert entity.default_options["normalize_audio"] is True
    assert entity.default_options["response_format"] == "mp3"
    assert entity.default_options["vocal_directions"] == ""
    assert entity.supported_languages == ["ar", "en"]
    assert entity.device_info["model"] == ORPHEUS_ENGLISH_MODEL
    assert entity.device_info["name"] == ORPHEUS_ENGLISH_MODEL
    assert entity.has_entity_name is True
    assert entity.translation_key == "text_to_speech"


def test_tts_entity_default_options_fall_back_from_invalid_stored_values():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "normalize_audio": ["true"],
        "response_format": ["mp3"],
    }
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), DummyClient())

    assert entity.default_options["normalize_audio"] is False
    assert entity.default_options["response_format"] == "wav"


def test_tts_supported_formats_match_conversion_formats():
    assert set(RESPONSE_FORMATS) == set(FFMPEG_OUTPUT_ARGS)


class DummyProc:
    def __init__(self, returncode: int):
        self.returncode = returncode

    async def communicate(self, input=None):  # noqa: A002
        if self.returncode == 0:
            return b"processed-audio", b""
        return b"", b"ffmpeg error"


class CancelledProc:
    returncode = None

    def __init__(self):
        self.killed = False
        self.waited = False

    async def communicate(self, input=None):  # noqa: A002
        raise asyncio.CancelledError

    def kill(self):
        self.killed = True

    async def wait(self):
        self.waited = True
        self.returncode = -9


@pytest.mark.asyncio
async def test_tts_normalize_runs_ffmpeg_and_keeps_selected_format(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), DummyClient())
    commands = []

    async def fake_exec(*args, **kwargs):  # noqa: ANN001
        commands.append(args)
        return DummyProc(returncode=0)

    monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    ext, payload = await entity.async_get_tts_audio(
        "Hello", "en", options={"normalize_audio": True}
    )

    assert ext == "wav"
    assert payload == b"processed-audio"
    assert commands[0] == ("ffmpeg", "-version")
    assert "-f" in commands[1]
    assert commands[1][commands[1].index("-f") + 1] == "lavfi"
    assert "-af" in commands[2]
    assert commands[2][commands[2].index("-f") + 1] == "wav"


@pytest.mark.asyncio
async def test_tts_cancelled_ffmpeg_process_is_stopped(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), DummyClient())
    process = CancelledProc()

    async def fake_exec(*args, **kwargs):  # noqa: ANN001
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    assert await entity.async_get_tts_audio(
        "Hello", "en", options={"normalize_audio": True}
    ) == (None, None)
    assert process.killed is True
    assert process.waited is True


@pytest.mark.asyncio
async def test_tts_converts_to_selected_playback_format(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)
    commands = []

    async def fake_exec(*args, **kwargs):  # noqa: ANN001
        commands.append(args)
        return DummyProc(returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    ext, payload = await entity.async_get_tts_audio(
        "Hello", "en", options={"response_format": "flac"}
    )

    assert ext == "flac"
    assert payload == b"processed-audio"
    assert client.calls[0]["response_format"] == "wav"
    assert commands[0] == ("ffmpeg", "-version")
    assert "-f" in commands[1]
    assert commands[1][commands[1].index("-f") + 1] == "lavfi"
    assert "-af" not in commands[2]
    assert commands[2][commands[2].index("-f") + 1] == "flac"


@pytest.mark.asyncio
async def test_tts_normalizes_service_response_format_text(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)

    async def fake_exec(*args, **kwargs):  # noqa: ANN001
        return DummyProc(returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    assert await entity.async_get_tts_audio(
        "Hello", "en", options={"response_format": " MP3 "}
    ) == ("mp3", b"processed-audio")


@pytest.mark.asyncio
async def test_tts_rejects_invalid_service_response_format_without_groq_call():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)

    assert await entity.async_get_tts_audio(
        "Hello", "en", options={"response_format": ["mp3"]}
    ) == (None, None)
    assert client.calls == []


@pytest.mark.asyncio
async def test_tts_rejects_falsey_non_string_response_format_without_groq_call():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)

    assert await entity.async_get_tts_audio(
        "Hello", "en", options={"response_format": 0}
    ) == (None, None)
    assert client.calls == []


@pytest.mark.asyncio
async def test_tts_rejects_unsupported_service_response_format_without_groq_call():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)

    assert await entity.async_get_tts_audio(
        "Hello", "en", options={"response_format": "ogg"}
    ) == (None, None)
    assert client.calls == []


@pytest.mark.asyncio
async def test_tts_normalizes_false_string_normalize_option_without_ffmpeg(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)
    ffmpeg_calls = []

    async def fake_exec(*args, **kwargs):  # noqa: ANN001
        ffmpeg_calls.append(args)
        return DummyProc(returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    assert await entity.async_get_tts_audio(
        "Hello", "en", options={"normalize_audio": "false"}
    ) == ("wav", PCM_WAV_BYTES)
    assert ffmpeg_calls == []


@pytest.mark.asyncio
async def test_tts_empty_normalize_option_skips_ffmpeg(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)
    ffmpeg_calls = []

    async def fake_exec(*args, **kwargs):  # noqa: ANN001
        ffmpeg_calls.append(args)
        return DummyProc(returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    assert await entity.async_get_tts_audio(
        "Hello", "en", options={"normalize_audio": ""}
    ) == ("wav", PCM_WAV_BYTES)
    assert ffmpeg_calls == []


@pytest.mark.asyncio
async def test_tts_normalizes_true_string_normalize_option(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)

    async def fake_exec(*args, **kwargs):  # noqa: ANN001
        return DummyProc(returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    assert await entity.async_get_tts_audio(
        "Hello", "en", options={"normalize_audio": "yes"}
    ) == ("wav", b"processed-audio")


@pytest.mark.asyncio
async def test_tts_rejects_invalid_normalize_option_without_groq_call():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)

    assert await entity.async_get_tts_audio(
        "Hello", "en", options={"normalize_audio": ["true"]}
    ) == (None, None)
    assert client.calls == []


@pytest.mark.asyncio
async def test_tts_unknown_error_returns_none():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }

    class BrokenClient:
        async def async_synthesize_speech(self, request):  # noqa: ANN001
            raise RuntimeError("boom")

    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), BrokenClient())

    assert await entity.async_get_tts_audio("Hello", "en") == (None, None)


@pytest.mark.asyncio
async def test_tts_raw_wav_override_does_not_clear_configured_ffmpeg_issue(
    monkeypatch,
):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
        "response_format": "mp3",
    }
    deleted_issues = []
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), DummyClient())
    monkeypatch.setattr(
        tts,
        "async_delete_ffmpeg_missing_issue",
        lambda hass, entry, service_data: deleted_issues.append(entry.entry_id),
    )

    assert await entity.async_get_tts_audio(
        "Hello", "en", options={"response_format": "wav"}
    ) == ("wav", PCM_WAV_BYTES)
    assert deleted_issues == []


@pytest.mark.asyncio
async def test_tts_raw_wav_override_keeps_issue_for_invalid_configured_format(
    monkeypatch,
):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
        "response_format": ["mp3"],
    }
    deleted_issues = []
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), DummyClient())
    monkeypatch.setattr(
        tts,
        "async_delete_ffmpeg_missing_issue",
        lambda hass, entry, service_data: deleted_issues.append(entry.entry_id),
    )

    assert await entity.async_get_tts_audio(
        "Hello", "en", options={"response_format": "wav"}
    ) == ("wav", PCM_WAV_BYTES)
    assert deleted_issues == []


@pytest.mark.asyncio
async def test_tts_raw_wav_override_keeps_issue_for_invalid_configured_normalize(
    monkeypatch,
):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
        "normalize_audio": ["true"],
    }
    deleted_issues = []
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), DummyClient())
    monkeypatch.setattr(
        tts,
        "async_delete_ffmpeg_missing_issue",
        lambda hass, entry, service_data: deleted_issues.append(entry.entry_id),
    )

    assert await entity.async_get_tts_audio(
        "Hello", "en", options={"response_format": "wav", "normalize_audio": False}
    ) == ("wav", PCM_WAV_BYTES)
    assert deleted_issues == []


def test_tts_raw_wav_override_keeps_issue_for_invalid_configured_long_tts(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
        "enable_long_tts": ["true"],
    }
    deleted_issues = []
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), DummyClient())
    monkeypatch.setattr(
        tts,
        "async_delete_ffmpeg_missing_issue",
        lambda hass, entry, service_data: deleted_issues.append(entry.entry_id),
    )

    assert entity._configured_audio_needs_ffmpeg() is True
    assert deleted_issues == []


@pytest.mark.asyncio
async def test_tts_raw_wav_default_clears_ffmpeg_issue(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    deleted_issues = []
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), DummyClient())
    monkeypatch.setattr(
        tts,
        "async_delete_ffmpeg_missing_issue",
        lambda hass, entry, service_data: deleted_issues.append(entry.entry_id),
    )

    assert await entity.async_get_tts_audio("Hello", "en") == ("wav", PCM_WAV_BYTES)
    assert deleted_issues == ["entry-id"]


@pytest.mark.asyncio
async def test_tts_ffmpeg_preflight_is_cached_for_conversion(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)
    commands = []

    async def fake_exec(*args, **kwargs):  # noqa: ANN001
        commands.append(args)
        return DummyProc(returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    assert await entity.async_get_tts_audio(
        "First", "en", options={"response_format": "mp3"}
    ) == ("mp3", b"processed-audio")
    assert await entity.async_get_tts_audio(
        "Second", "en", options={"response_format": "mp3"}
    ) == ("mp3", b"processed-audio")

    assert commands.count(("ffmpeg", "-version")) == 1
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_tts_normalized_preflight_is_separate_from_conversion_cache(
    monkeypatch,
):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)
    commands = []

    async def fail_loudnorm_preflight(*args, **kwargs):  # noqa: ANN001
        commands.append(args)
        is_loudnorm_preflight = "lavfi" in args and "-af" in args
        return DummyProc(returncode=1 if is_loudnorm_preflight else 0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_loudnorm_preflight)

    assert await entity.async_get_tts_audio(
        "Plain", "en", options={"response_format": "mp3"}
    ) == ("mp3", b"processed-audio")
    assert await entity.async_get_tts_audio(
        "Normalized",
        "en",
        options={"response_format": "mp3", "normalize_audio": True},
    ) == (None, None)

    assert commands.count(("ffmpeg", "-version")) == 2
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_tts_conversion_unsupported_output_skips_groq_call(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)
    ffmpeg_issues = []
    commands = []

    async def unsupported_output(*args, **kwargs):  # noqa: ANN001
        commands.append(args)
        return DummyProc(returncode=0 if args == ("ffmpeg", "-version") else 1)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unsupported_output)
    monkeypatch.setattr(
        tts,
        "async_create_ffmpeg_missing_issue",
        lambda hass, entry, service_data: ffmpeg_issues.append(
            (entry.entry_id, service_data.get("unique_id"))
        ),
    )

    ext, payload = await entity.async_get_tts_audio(
        "Hello", "en", options={"response_format": "mp3"}
    )

    assert ext is None
    assert payload is None
    assert client.calls == []
    assert commands[0] == ("ffmpeg", "-version")
    assert "-f" in commands[1]
    assert commands[1][commands[1].index("-f") + 1] == "lavfi"
    assert ffmpeg_issues == [("entry-id", None)]


@pytest.mark.asyncio
async def test_tts_conversion_missing_ffmpeg_skips_groq_call(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)
    ffmpeg_issues = []

    async def missing_ffmpeg(*args, **kwargs):  # noqa: ANN001
        raise FileNotFoundError

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing_ffmpeg)
    monkeypatch.setattr(
        tts,
        "async_create_ffmpeg_missing_issue",
        lambda hass, entry, service_data: ffmpeg_issues.append(
            (entry.entry_id, service_data.get("unique_id"))
        ),
    )

    ext, payload = await entity.async_get_tts_audio(
        "Hello", "en", options={"response_format": "mp3"}
    )

    assert ext is None
    assert payload is None
    assert client.calls == []
    assert ffmpeg_issues == [("entry-id", None)]


@pytest.mark.asyncio
async def test_tts_conversion_ffmpeg_start_error_skips_groq_call(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)
    ffmpeg_issues = []

    async def ffmpeg_start_error(*args, **kwargs):  # noqa: ANN001
        raise PermissionError("not executable")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", ffmpeg_start_error)
    monkeypatch.setattr(
        tts,
        "async_create_ffmpeg_missing_issue",
        lambda hass, entry, service_data: ffmpeg_issues.append(
            (entry.entry_id, service_data.get("unique_id"))
        ),
    )

    assert await entity.async_get_tts_audio(
        "Hello", "en", options={"response_format": "mp3"}
    ) == (None, None)
    assert client.calls == []
    assert ffmpeg_issues == [("entry-id", None)]


@pytest.mark.asyncio
async def test_tts_conversion_failure_invalidates_ffmpeg_cache(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)
    ffmpeg_issues = []
    commands = []

    async def fail_after_preflight(*args, **kwargs):  # noqa: ANN001
        commands.append(args)
        is_preflight = (
            args == ("ffmpeg", "-version") or "-f" in args and "lavfi" in args
        )
        return DummyProc(returncode=0 if is_preflight else 1)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_after_preflight)
    monkeypatch.setattr(
        tts,
        "async_create_ffmpeg_missing_issue",
        lambda hass, entry, service_data: ffmpeg_issues.append(
            (entry.entry_id, service_data.get("unique_id"))
        ),
    )

    assert await entity.async_get_tts_audio(
        "Hello", "en", options={"response_format": "mp3"}
    ) == (None, None)

    assert len(client.calls) == 1
    assert ("mp3", False) not in entity._ffmpeg_capabilities
    assert commands.count(("ffmpeg", "-version")) == 1
    assert ffmpeg_issues == [("entry-id", None)]


@pytest.mark.asyncio
async def test_tts_normalize_splits_and_stitches_long_announcements(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(
        DummyHass(),
        DummyConfigEntry(
            data,
            {"enable_long_tts": True, "normalize_audio": True},
        ),
        client,
    )
    commands = []
    first_sentence = f"{'A' * 198}."
    second_sentence = f"{'B' * 40}."

    async def fake_exec(*args, **kwargs):  # noqa: ANN001
        commands.append(args)
        return DummyProc(returncode=0)

    monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    ext, payload = await entity.async_get_tts_audio(
        f"{first_sentence} {second_sentence}",
        "en",
        options=None,
    )

    assert ext == "wav"
    assert payload == b"processed-audio"
    assert [call["text"] for call in client.calls] == [
        first_sentence,
        second_sentence,
    ]
    stitch_command = next(
        command for command in commands if "-filter_complex" in command
    )
    assert any("concat=n=2:v=0:a=1" in arg for arg in stitch_command)
    assert any("loudnorm=I=-16:TP=-1:LRA=5" in arg for arg in stitch_command)


@pytest.mark.asyncio
async def test_tts_long_announcements_can_stitch_without_normalization(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(
        DummyHass(),
        DummyConfigEntry(data, {"enable_long_tts": True}),
        client,
    )
    commands = []
    first_sentence = f"{'A' * 198}."
    second_sentence = f"{'B' * 40}."

    async def fake_exec(*args, **kwargs):  # noqa: ANN001
        commands.append(args)
        return DummyProc(returncode=0)

    monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    ext, payload = await entity.async_get_tts_audio(
        f"{first_sentence} {second_sentence}",
        "en",
        options=None,
    )

    assert ext == "wav"
    assert payload == b"processed-audio"
    assert [call["text"] for call in client.calls] == [
        first_sentence,
        second_sentence,
    ]
    stitch_command = next(
        command for command in commands if "-filter_complex" in command
    )
    assert any("concat=n=2:v=0:a=1" in arg for arg in stitch_command)
    assert not any("loudnorm" in arg for arg in stitch_command)


@pytest.mark.asyncio
async def test_tts_long_cached_chunks_bypass_batch_free_tier_guard(monkeypatch):
    data = {
        "url": "https://api.groq.com/openai/v1/audio/speech",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    first_sentence = f"{'A' * 198}."
    second_sentence = f"{'B' * 40}."
    client = GroqApiClient(DummyHass(), api_key="api-key")
    monkeypatch.setattr(
        client,
        "_free_tier_limits",
        lambda model: {
            "requests_per_minute": 1,
            "requests_per_day": 100,
            "tokens_per_minute": 1000,
            "tokens_per_day": 1000,
        },
    )
    namespace = f"{ORPHEUS_ENGLISH_MODEL}:{ORPHEUS_ENGLISH_VOICE}"
    cache = client._speech_caches.setdefault(namespace, OrderedDict())
    cache[(ORPHEUS_ENGLISH_MODEL, ORPHEUS_ENGLISH_VOICE, "wav", first_sentence)] = (
        b"chunk-one"
    )
    cache[(ORPHEUS_ENGLISH_MODEL, ORPHEUS_ENGLISH_VOICE, "wav", second_sentence)] = (
        b"chunk-two"
    )
    client._record_local_tts_usage(
        SpeechRequest(
            text="existing",
            model=ORPHEUS_ENGLISH_MODEL,
            voice=ORPHEUS_ENGLISH_VOICE,
        ),
        1,
    )
    entity = GroqTTSEntity(
        DummyHass(),
        DummyConfigEntry(data, {"enable_long_tts": True}),
        client,
    )

    async def fake_exec(*args, **kwargs):  # noqa: ANN001
        return DummyProc(returncode=0)

    monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    ext, payload = await entity.async_get_tts_audio(
        f"{first_sentence} {second_sentence}",
        "en",
        options=None,
    )

    assert ext == "wav"
    assert payload == b"processed-audio"


@pytest.mark.asyncio
async def test_tts_long_batch_guard_blocks_eviction_misses_before_api(monkeypatch):
    data = {
        "url": "https://api.groq.com/openai/v1/audio/speech",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    first_sentence = f"{'A' * 198}."
    second_sentence = f"{'B' * 198}."
    third_sentence = f"{'C' * 40}."

    class RecordingClient(GroqApiClient):
        def __init__(self):
            super().__init__(DummyHass(), api_key="api-key")
            self.calls = []

        async def _request_audio(
            self,
            *args,
            **kwargs,
        ):
            self.calls.append(kwargs)
            return PCM_WAV_BYTES

    client = RecordingClient()
    monkeypatch.setattr(
        client,
        "_free_tier_limits",
        lambda model: {
            "requests_per_minute": 2,
            "requests_per_day": 100,
            "tokens_per_minute": 1000,
            "tokens_per_day": 1000,
        },
    )
    namespace = f"{ORPHEUS_ENGLISH_MODEL}:{ORPHEUS_ENGLISH_VOICE}"
    cache = client._speech_caches.setdefault(namespace, OrderedDict())
    for text in (first_sentence, third_sentence):
        cache[(ORPHEUS_ENGLISH_MODEL, ORPHEUS_ENGLISH_VOICE, "wav", text)] = b"cached"
    client._record_local_tts_usage(
        SpeechRequest(
            text="existing",
            model=ORPHEUS_ENGLISH_MODEL,
            voice=ORPHEUS_ENGLISH_VOICE,
        ),
        1,
    )
    entity = GroqTTSEntity(
        DummyHass(),
        DummyConfigEntry(data, {"enable_long_tts": True, "cache_size": 2}),
        client,
    )
    monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    ext, payload = await entity.async_get_tts_audio(
        f"{first_sentence} {second_sentence} {third_sentence}",
        "en",
        options=None,
    )

    assert ext is None
    assert payload is None
    assert client.calls == []


@pytest.mark.asyncio
async def test_tts_long_stitching_temp_file_work_uses_executor(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    calls = []

    async def async_add_executor_job(func, *args):
        calls.append(func.__name__)
        return func(*args)

    hass = SimpleNamespace(async_add_executor_job=async_add_executor_job)
    entity = GroqTTSEntity(
        hass,
        DummyConfigEntry(data, {"enable_long_tts": True}),
        DummyClient(),
    )

    async def fake_exec(*args, **kwargs):  # noqa: ANN001
        return DummyProc(returncode=0)

    monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    ext, payload = await entity.async_get_tts_audio(
        f"{'A' * 198}. {'B' * 40}.",
        "en",
        options=None,
    )

    assert ext == "wav"
    assert payload == b"processed-audio"
    assert "mkdtemp" in calls
    assert "_write_audio_chunks" in calls
    assert "rmtree" in calls


@pytest.mark.asyncio
async def test_tts_service_options_override_groq_speech_payload():
    data = {
        "url": "http://example.com",
        "model": "data-model",
        "voice": "data-voice",
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)

    ext, payload = await entity.async_get_tts_audio(
        "service message",
        "en",
        options={
            "input": "override input",
            "model": ORPHEUS_ENGLISH_MODEL,
            "voice": ORPHEUS_ENGLISH_VOICE,
            "vocal_directions": "cheerful",
        },
    )

    assert ext == "wav"
    assert payload == PCM_WAV_BYTES
    assert client.calls == [
        {
            "text": "[cheerful] override input",
            "voice": ORPHEUS_ENGLISH_VOICE,
            "model": ORPHEUS_ENGLISH_MODEL,
            "response_format": "wav",
            "cache_max": 256,
            "protect_free_tier": True,
        }
    ]


@pytest.mark.asyncio
async def test_tts_uses_account_level_protect_free_tier_option():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(
        DummyHass(),
        DummyConfigEntry(data, {"protect_free_tier": False}),
        client,
    )

    assert await entity.async_get_tts_audio("Hello", "en") == (
        "wav",
        PCM_WAV_BYTES,
    )
    assert client.calls[0]["protect_free_tier"] is False


@pytest.mark.asyncio
async def test_tts_normalizes_protect_free_tier_option_strings():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(
        DummyHass(),
        DummyConfigEntry(data, {"protect_free_tier": "false"}),
        client,
    )

    assert await entity.async_get_tts_audio("Hello", "en") == (
        "wav",
        PCM_WAV_BYTES,
    )
    assert client.calls[0]["protect_free_tier"] is False


@pytest.mark.asyncio
async def test_tts_invalid_protect_free_tier_defaults_to_enabled():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(
        DummyHass(),
        DummyConfigEntry(data, {"protect_free_tier": ["false"]}),
        client,
    )

    assert await entity.async_get_tts_audio("Hello", "en") == (
        "wav",
        PCM_WAV_BYTES,
    )
    assert client.calls[0]["protect_free_tier"] is True


@pytest.mark.asyncio
async def test_tts_async_setup_entry_uses_runtime_client_with_options():
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
    client = added[0]._client
    assert client._api_key == "option-key"
    assert client.base_url == "option-url"


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
                "vocal_directions": "warm",
                "protect_free_tier": False,
            },
        )
    }
    added = []
    subentry_ids = []

    def add_entities(entities, **kwargs):
        added.extend(entities)
        subentry_ids.append(kwargs.get("config_subentry_id"))

    await tts.async_setup_entry(
        DummyHass(),
        entry,
        add_entities,
    )

    assert len(added) == 1
    assert subentry_ids == ["subentry-id"]
    assert added[0].unique_id == "subentry-id"
    assert added[0].has_entity_name is True
    assert added[0].translation_key == "text_to_speech"
    assert added[0].device_info["name"] == "Kitchen TTS"
    assert added[0]._client.base_url == "https://api.groq.com/openai/v1"


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

    async def validate_api_key(_hass, _api_key):
        return None

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
    monkeypatch.setattr(config_flow, "async_validate_api_key", validate_api_key)


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("validation_error", "expected_errors"),
    [
        ("invalid_auth", {"api_key": "invalid_auth"}),
        ("cannot_connect", {"base": "cannot_connect"}),
        ("unknown", {"base": "unknown"}),
    ],
)
async def test_config_flow_user_shows_api_key_validation_errors(
    monkeypatch,
    validation_error,
    expected_errors,
):
    flow = config_flow.GroqConfigFlow()
    _patch_flow_common(monkeypatch, flow)

    async def set_unique_id(unique_id):
        return None

    async def validate_api_key(_hass, _api_key):
        return validation_error

    monkeypatch.setattr(flow, "async_set_unique_id", set_unique_id)
    monkeypatch.setattr(flow, "_abort_if_unique_id_configured", lambda: None)
    monkeypatch.setattr(config_flow, "async_validate_api_key", validate_api_key)

    result = await flow.async_step_user({"api_key": "api-key"})

    assert result["type"] == "form"
    assert result["errors"] == expected_errors


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
async def test_speech_to_text_subentry_flow_defaults_language_from_hass(monkeypatch):
    flow = config_flow.GroqServiceSubentryFlow()
    flow.handler = ("entry-id", FEATURE_SPEECH_TO_TEXT)
    hass = SimpleNamespace(config=SimpleNamespace(language="fr-FR"))
    _patch_flow_common(monkeypatch, flow, hass)

    form = await flow.async_step_user()

    assert form["type"] == "form"
    assert form["step_id"] == FEATURE_SPEECH_TO_TEXT
    assert (
        form["data_schema"](
            {
                "name": "Speech-to-Text",
                "model": "whisper-large-v3",
            }
        )["language"]
        == "fr-FR"
    )


@pytest.mark.asyncio
async def test_text_to_speech_subentry_flow_creates_service(monkeypatch):
    flow = config_flow.GroqServiceSubentryFlow()
    flow.handler = ("entry-id", FEATURE_TEXT_TO_SPEECH)

    async def async_add_executor_job(func, *args):
        return func(*args)

    flow.hass = SimpleNamespace(async_add_executor_job=async_add_executor_job)
    _patch_flow_common(monkeypatch, flow, flow.hass)
    monkeypatch.setattr(config_flow.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    form = await flow.async_step_user()
    assert form["type"] == "form"
    assert form["step_id"] == FEATURE_TEXT_TO_SPEECH

    result = await flow.async_step_user(
        {
            "name": "Kitchen TTS",
            "model": ORPHEUS_ENGLISH_MODEL,
            "voice": ORPHEUS_ENGLISH_VOICE,
            "response_format": "mp3",
            "vocal_directions": "warm",
            "normalize_audio": False,
            "enable_long_tts": False,
        }
    )

    assert result == {
        "type": "create_entry",
        "title": "Kitchen TTS",
        "data": {
            "name": "Kitchen TTS",
            "model": ORPHEUS_ENGLISH_MODEL,
            "voice": ORPHEUS_ENGLISH_VOICE,
            "response_format": "mp3",
            "vocal_directions": "warm",
            "normalize_audio": False,
            "enable_long_tts": False,
            "service_type": FEATURE_TEXT_TO_SPEECH,
        },
    }


@pytest.mark.asyncio
async def test_text_to_speech_subentry_flow_disables_ffmpeg_options_when_missing(
    monkeypatch,
):
    flow = config_flow.GroqServiceSubentryFlow()
    flow.handler = ("entry-id", FEATURE_TEXT_TO_SPEECH)
    _patch_flow_common(monkeypatch, flow)
    monkeypatch.setattr(config_flow.shutil, "which", lambda name: None)

    form = await flow.async_step_user()
    defaults = form["data_schema"](
        {
            "name": "Kitchen TTS",
            "model": ORPHEUS_ENGLISH_MODEL,
            "voice": ORPHEUS_ENGLISH_VOICE,
        }
    )
    assert defaults["normalize_audio"] is False
    assert defaults["enable_long_tts"] is False
    assert _selector_config(form["data_schema"], "normalize_audio") == {
        "read_only": True
    }
    assert _selector_config(form["data_schema"], "enable_long_tts") == {
        "read_only": True
    }

    result = await flow.async_step_user(
        {
            "name": "Kitchen TTS",
            "model": ORPHEUS_ENGLISH_MODEL,
            "voice": ORPHEUS_ENGLISH_VOICE,
            "normalize_audio": True,
            "enable_long_tts": True,
        }
    )

    assert result["type"] == "form"
    assert result["errors"] == {
        "normalize_audio": "ffmpeg_required",
        "enable_long_tts": "ffmpeg_required",
    }
    corrected = result["data_schema"](
        {
            "name": "Kitchen TTS",
            "model": ORPHEUS_ENGLISH_MODEL,
            "voice": ORPHEUS_ENGLISH_VOICE,
        }
    )
    assert corrected["normalize_audio"] is False
    assert corrected["enable_long_tts"] is False


@pytest.mark.asyncio
async def test_text_to_speech_subentry_flow_disables_converted_format_when_ffmpeg_missing(
    monkeypatch,
):
    flow = config_flow.GroqServiceSubentryFlow()
    flow.handler = ("entry-id", FEATURE_TEXT_TO_SPEECH)
    _patch_flow_common(monkeypatch, flow)
    monkeypatch.setattr(config_flow.shutil, "which", lambda name: None)

    result = await flow.async_step_user(
        {
            "name": "Kitchen TTS",
            "model": ORPHEUS_ENGLISH_MODEL,
            "voice": ORPHEUS_ENGLISH_VOICE,
            "response_format": "mp3",
        }
    )

    assert result["type"] == "form"
    assert result["errors"] == {"response_format": "ffmpeg_required"}
    assert (
        result["data_schema"](
            {
                "name": "Kitchen TTS",
                "model": ORPHEUS_ENGLISH_MODEL,
                "voice": ORPHEUS_ENGLISH_VOICE,
            }
        )["response_format"]
        == "wav"
    )


@pytest.mark.asyncio
async def test_text_to_speech_subentry_flow_clears_voice_when_model_changes(
    monkeypatch,
):
    flow = config_flow.GroqServiceSubentryFlow()
    flow.handler = ("entry-id", FEATURE_TEXT_TO_SPEECH)
    _patch_flow_common(monkeypatch, flow)
    monkeypatch.setattr(config_flow.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    result = await flow.async_step_user(
        {
            "name": "Kitchen TTS",
            "model": ORPHEUS_ARABIC_MODEL,
            "voice": ORPHEUS_ENGLISH_VOICE,
        }
    )

    assert result["type"] == "form"
    assert result["step_id"] == FEATURE_TEXT_TO_SPEECH
    assert result["errors"] == {"voice": "select_voice_for_model"}
    assert result["data_schema"](
        {
            "name": "Kitchen TTS",
            "model": ORPHEUS_ARABIC_MODEL,
            "voice": ORPHEUS_ARABIC_VOICE,
        }
    )

    created = await flow.async_step_user(
        {
            "name": "Kitchen TTS",
            "model": ORPHEUS_ARABIC_MODEL,
            "voice": ORPHEUS_ARABIC_VOICE,
        }
    )

    assert created["type"] == "create_entry"
    assert created["data"]["voice"] == ORPHEUS_ARABIC_VOICE


@pytest.mark.asyncio
async def test_text_to_speech_subentry_flow_rejects_invalid_response_format(
    monkeypatch,
):
    flow = config_flow.GroqServiceSubentryFlow()
    flow.handler = ("entry-id", FEATURE_TEXT_TO_SPEECH)
    _patch_flow_common(monkeypatch, flow)
    monkeypatch.setattr(config_flow.shutil, "which", lambda name: None)

    result = await flow.async_step_user(
        {
            "name": "Kitchen TTS",
            "model": ORPHEUS_ENGLISH_MODEL,
            "voice": ORPHEUS_ENGLISH_VOICE,
            "response_format": "ogg",
        }
    )

    assert result["type"] == "form"
    assert result["step_id"] == FEATURE_TEXT_TO_SPEECH
    assert result["errors"] == {"response_format": "invalid_response_format"}
    assert (
        result["data_schema"](
            {
                "name": "Kitchen TTS",
                "model": ORPHEUS_ENGLISH_MODEL,
                "voice": ORPHEUS_ENGLISH_VOICE,
            }
        )["response_format"]
        == "wav"
    )


@pytest.mark.asyncio
async def test_text_to_speech_subentry_reconfigure_replaces_service_data(monkeypatch):
    flow = config_flow.GroqServiceSubentryFlow()
    flow.handler = ("entry-id", FEATURE_TEXT_TO_SPEECH)
    flow.context = {"source": "reconfigure", "subentry_id": "subentry-id"}
    entry = SimpleNamespace(entry_id="entry-id")
    subentry = SimpleNamespace(
        data={
            "name": "Old TTS",
            "model": ORPHEUS_ENGLISH_MODEL,
            "voice": ORPHEUS_ENGLISH_VOICE,
            "response_format": "flac",
            "vocal_directions": "warm",
            "normalize_audio": True,
            "enable_long_tts": True,
            "service_type": FEATURE_TEXT_TO_SPEECH,
        }
    )
    _patch_flow_common(monkeypatch, flow)
    monkeypatch.setattr(flow, "_get_entry", lambda: entry)
    monkeypatch.setattr(flow, "_get_reconfigure_subentry", lambda: subentry)
    monkeypatch.setattr(config_flow.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    form = await flow.async_step_reconfigure()
    assert form["type"] == "form"
    assert form["step_id"] == FEATURE_TEXT_TO_SPEECH

    result = await flow.async_step_reconfigure(
        {
            "name": "Updated TTS",
            "model": ORPHEUS_ENGLISH_MODEL,
            "voice": ORPHEUS_ENGLISH_VOICE,
            "response_format": "mp3",
            "vocal_directions": "",
            "normalize_audio": False,
            "enable_long_tts": False,
        }
    )

    assert result["type"] == "abort"
    assert result["reason"] == "reconfigure_successful"
    assert result["title"] == "Updated TTS"
    assert result["data"]["name"] == "Updated TTS"
    assert result["data"]["response_format"] == "mp3"
    assert result["data"]["vocal_directions"] == ""
    assert result["data"]["normalize_audio"] is False
    assert result["data"]["enable_long_tts"] is False
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
            "name": "Text Service",
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
        "title": "Text Service",
        "data": {
            "name": "Text Service",
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
async def test_text_generation_subentry_flow_stores_llm_hass_api(monkeypatch):
    flow = config_flow.GroqServiceSubentryFlow()
    _patch_flow_common(monkeypatch, flow)
    monkeypatch.setattr(
        config_flow.llm,
        "async_get_apis",
        lambda _hass: [SimpleNamespace(id="assist", name="Assist")],
    )

    form = await flow.async_step_text_generation()
    assert form["type"] == "form"
    assert form["data_schema"](
        {
            "name": "Text Service",
            "model": "llama-3.1-8b-instant",
            CONF_LLM_HASS_API: ["assist"],
        }
    )[CONF_LLM_HASS_API] == ["assist"]

    result = await flow.async_step_text_generation(
        {
            "name": "Text Service",
            "model": "llama-3.1-8b-instant",
            CONF_LLM_HASS_API: ["assist"],
        }
    )

    assert result["type"] == "create_entry"
    assert result["data"][CONF_LLM_HASS_API] == ["assist"]


@pytest.mark.asyncio
async def test_text_generation_subentry_flow_omits_empty_llm_hass_api(monkeypatch):
    flow = config_flow.GroqServiceSubentryFlow()
    _patch_flow_common(monkeypatch, flow)

    result = await flow.async_step_text_generation(
        {
            "name": "Text Service",
            "model": "llama-3.1-8b-instant",
            CONF_LLM_HASS_API: [],
        }
    )

    assert result["type"] == "create_entry"
    assert CONF_LLM_HASS_API not in result["data"]


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
    assert result["data"]["max_tokens"] == 256
    assert result["data"]["top_p"] == 0.9
    assert "service_tier" not in result["data"]
    assert "prompt_caching" in result["data"]
    assert "request_body_options" not in result["data"]
    assert result["data"]["service_type"] == FEATURE_TEXT_GENERATION


@pytest.mark.asyncio
async def test_text_generation_reconfigure_strips_unsupported_hidden_options(
    monkeypatch,
):
    flow = config_flow.GroqServiceSubentryFlow()
    flow.handler = ("entry-id", FEATURE_TEXT_GENERATION)
    flow.context = {"source": "reconfigure", "subentry_id": "subentry-id"}
    entry = SimpleNamespace(entry_id="entry-id")
    subentry = SimpleNamespace(
        data={
            "name": "Existing Text",
            "model": "openai/gpt-oss-20b",
            "system_prompt": "Existing prompt.",
            "temperature": 0.3,
            "reasoning_effort": "medium",
            "prompt_caching": True,
            "structured_outputs": True,
            "schema_name": "response",
            "schema": {"type": "object"},
            "strict": True,
            "request_body_options": {
                "response_format": {"type": "json_schema", "json_schema": {}},
                "reasoning_effort": "low",
                "user": "home-assistant",
            },
            "service_type": FEATURE_TEXT_GENERATION,
        }
    )
    _patch_flow_common(monkeypatch, flow)
    monkeypatch.setattr(flow, "_get_entry", lambda: entry)
    monkeypatch.setattr(flow, "_get_reconfigure_subentry", lambda: subentry)

    result = await flow.async_step_reconfigure(
        {
            "name": "Updated Text",
            "model": "llama-3.1-8b-instant",
            "system_prompt": "Updated prompt.",
            "temperature": 0.4,
        }
    )

    assert result["type"] == "abort"
    assert result["data"]["model"] == "llama-3.1-8b-instant"
    for key in (
        "reasoning_effort",
        "prompt_caching",
        "structured_outputs",
        "schema_name",
        "schema",
        "strict",
    ):
        assert key not in result["data"]
    assert result["data"]["request_body_options"] == {"user": "home-assistant"}
    assert result["data"]["service_type"] == FEATURE_TEXT_GENERATION


@pytest.mark.asyncio
async def test_text_generation_reconfigure_reports_remaining_hidden_errors(
    monkeypatch,
):
    flow = config_flow.GroqServiceSubentryFlow()
    flow.handler = ("entry-id", FEATURE_TEXT_GENERATION)
    flow.context = {"source": "reconfigure", "subentry_id": "subentry-id"}
    entry = SimpleNamespace(entry_id="entry-id")
    subentry = SimpleNamespace(
        data={
            "name": "Existing Text",
            "model": "openai/gpt-oss-20b",
            "request_body_options": {"model": "override-model"},
            "service_type": FEATURE_TEXT_GENERATION,
        }
    )
    _patch_flow_common(monkeypatch, flow)
    monkeypatch.setattr(flow, "_get_entry", lambda: entry)
    monkeypatch.setattr(flow, "_get_reconfigure_subentry", lambda: subentry)

    result = await flow.async_step_reconfigure(
        {
            "name": "Updated Text",
            "model": "openai/gpt-oss-20b",
            "system_prompt": "Updated prompt.",
            "temperature": 0.4,
        }
    )

    assert result["type"] == "form"
    assert result["step_id"] == FEATURE_TEXT_GENERATION
    assert result["errors"] == {
        "base": "reserved_request_body_option",
    }


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
    assert result["data"]["api_key"] == "new"
    assert result["options"] == {}
    assert result["unique_id"] == reauth_entry.unique_id

    async def invalid_api_key(_hass, _api_key):
        return "invalid_auth"

    monkeypatch.setattr(config_flow, "async_validate_api_key", invalid_api_key)
    invalid = await flow.async_step_reauth_confirm({"api_key": "bad"})
    assert invalid["type"] == "form"
    assert invalid["errors"] == {"api_key": "invalid_auth"}

    duplicate_key = "duplicate"
    other_entry = DummyConfigEntry(
        {"api_key": duplicate_key},
        {},
        unique_id="other",
    )
    other_entry.entry_id = "other-entry"

    class DuplicateConfigEntries(DummyConfigEntries):
        def async_entries(self, _domain):
            return [reauth_entry, other_entry]

    flow.hass = SimpleNamespace(config_entries=DuplicateConfigEntries())
    flow._reauth_entry = reauth_entry
    monkeypatch.setattr(
        config_flow,
        "async_validate_api_key",
        lambda _hass, _api_key: asyncio.sleep(0, result=None),
    )
    duplicate = await flow.async_step_reauth_confirm({"api_key": duplicate_key})
    assert duplicate["type"] == "form"
    assert duplicate["errors"] == {"base": "duplicate_api_key"}


@pytest.mark.asyncio
async def test_config_flow_reconfigure_updates_account(monkeypatch):
    entry = DummyConfigEntry({"api_key": "old", "name": "Old Groq"}, {})
    flow = config_flow.GroqConfigFlow()
    _patch_flow_common(monkeypatch, flow)
    monkeypatch.setattr(flow, "_get_reconfigure_entry", lambda: entry)
    monkeypatch.setattr(
        flow,
        "async_update_reload_and_abort",
        lambda entry, **kwargs: {"type": "abort", "entry": entry, **kwargs},
    )

    form = await flow.async_step_reconfigure()
    assert form["type"] == "form"
    assert form["step_id"] == "reconfigure"

    result = await flow.async_step_reconfigure(
        {"name": "Production Groq", "api_key": "new-key"}
    )

    assert result["type"] == "abort"
    assert result["entry"] is entry
    assert result["title"] == "Production Groq"
    assert result["data"]["name"] == "Production Groq"
    assert result["data"]["api_key"] == "new-key"
    assert result["options"] == {}
    assert result["unique_id"] == entry.unique_id
    assert result["reason"] == "reconfigure_successful"

    other = DummyConfigEntry({"api_key": "duplicate"}, {})
    other.entry_id = "other-entry"
    other.unique_id = "other"

    class DuplicateConfigEntries(DummyConfigEntries):
        def async_entries(self, _domain):
            return [entry, other]

    flow.hass = SimpleNamespace(config_entries=DuplicateConfigEntries())
    duplicate = await flow.async_step_reconfigure(
        {"name": "Production Groq", "api_key": "duplicate"}
    )
    assert duplicate["type"] == "form"
    assert duplicate["errors"] == {"base": "duplicate_api_key"}


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

    updated = []
    flow.hass = SimpleNamespace(
        async_add_executor_job=async_add_executor_job,
        config_entries=SimpleNamespace(
            async_get_known_entry=lambda entry_id: entry,
            async_update_entry=lambda entry, **kwargs: updated.append((entry, kwargs)),
        ),
    )
    _patch_flow_common(monkeypatch, flow, flow.hass)

    form = await flow.async_step_init()
    assert form["type"] == "form"
    assert form["step_id"] == "init"

    saved = await flow.async_step_init({"api_key": "new-key"})
    assert saved == {
        "type": "create_entry",
        "title": "",
        "data": {},
    }
    assert updated[0][0] is entry
    assert updated[0][1]["data"]["api_key"] == "new-key"
    assert updated[0][1]["options"] == {}
    assert updated[0][1]["unique_id"] == entry.unique_id

    async def cannot_connect(_hass, _api_key):
        return "cannot_connect"

    monkeypatch.setattr(config_flow, "async_validate_api_key", cannot_connect)
    failed = await flow.async_step_init({"api_key": "broken-key"})
    assert failed["type"] == "form"
    assert failed["errors"] == {"base": "cannot_connect"}


@pytest.mark.asyncio
async def test_options_flow_rejects_duplicate_api_key(monkeypatch):
    duplicate_key = "duplicate"
    current_entry = DummyConfigEntry({"api_key": "current"}, {}, unique_id="current")
    other_entry = DummyConfigEntry(
        {"api_key": duplicate_key},
        {},
        unique_id="other",
    )
    other_entry.entry_id = "other-entry"

    class DuplicateConfigEntries:
        def async_entries(self, _domain):
            return [current_entry, other_entry]

    flow = config_flow.GroqOptionsFlow()
    flow.hass = SimpleNamespace(config_entries=DuplicateConfigEntries())
    flow.handler = current_entry.entry_id
    monkeypatch.setattr(
        flow,
        "async_show_form",
        lambda **kwargs: {"type": "form", **kwargs},
    )
    monkeypatch.setattr(
        flow,
        "async_create_entry",
        lambda **kwargs: {"type": "create_entry", **kwargs},
    )
    monkeypatch.setattr(
        config_flow,
        "async_validate_api_key",
        lambda _hass, _api_key: asyncio.sleep(0, result=None),
    )

    result = await flow.async_step_init({"api_key": duplicate_key})

    assert result["type"] == "form"
    assert result["errors"] == {"base": "duplicate_api_key"}
