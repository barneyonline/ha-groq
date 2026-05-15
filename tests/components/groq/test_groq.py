import asyncio
import logging
import pytest
from unittest.mock import patch

import aiohttp
from homeassistant.exceptions import ConfigEntryAuthFailed, HomeAssistantError

from custom_components.groq import config_flow, tts_engine
from custom_components.groq.const import normalize_enabled_features
from custom_components.groq.tts_engine import GroqRateLimitError, GroqTTSEngine
from custom_components.groq.tts import GroqTTSEntity

validate_user_input = config_flow.validate_user_input
get_model_options = config_flow.get_model_options

ORPHEUS_ENGLISH_MODEL = "canopylabs/orpheus-v1-english"
ORPHEUS_ENGLISH_VOICE = "troy"


@pytest.mark.asyncio
async def test_validate_user_input_missing_api_key():
    with pytest.raises(ValueError):
        await validate_user_input({})


@pytest.mark.asyncio
async def test_validate_user_input_accepts_account_level_setup():
    await validate_user_input(
        {
            "api_key": "api-key",
            "enabled_features": ["text_generation", "text_to_speech"],
        }
    )


def test_get_model_options_filters_to_orpheus_tts_models():
    opts = get_model_options(
        [
            "llama-3.3-70b-versatile",
            "whisper-large-v3",
            "playai-tts",
            "canopylabs/orpheus-v1-english",
            "canopylabs/orpheus-arabic-saudi",
        ]
    )

    assert opts == [
        "canopylabs/orpheus-arabic-saudi",
        "canopylabs/orpheus-v1-english",
    ]


def test_normalize_enabled_features_defaults_and_preserves_explicit_empty():
    assert normalize_enabled_features(None) == ["text_to_speech"]
    assert normalize_enabled_features([]) == []
    assert normalize_enabled_features(
        ["prompt_caching", "unknown", "text_to_speech", "image_recognition"]
    ) == ["text_to_speech", "image_recognition"]


class DummySession:
    def post(self, *args, **kwargs):
        raise aiohttp.ClientError("boom")


class DummyHass:
    pass


@pytest.mark.asyncio
async def test_async_get_tts_network_error():
    engine = GroqTTSEngine(None, "voice", "model", "http://example.com")

    with patch.object(
        tts_engine, "async_get_clientsession", return_value=DummySession()
    ):
        with pytest.raises(HomeAssistantError):
            await engine.async_get_tts(DummyHass(), "hi")


class DummyResponse:
    def __init__(self, status: int, headers: dict, body: bytes):
        self.status = status
        self.headers = headers
        self._body = body

    async def read(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummyOkJsonSession:
    def post(self, *args, **kwargs):
        headers = {"content-type": "application/json"}
        body = b'{"ok": true}'
        return DummyResponse(200, headers, body)


@pytest.mark.asyncio
async def test_async_get_tts_non_audio_2xx():
    engine = GroqTTSEngine(None, "voice", "model", "http://example.com")
    with patch.object(
        tts_engine, "async_get_clientsession", return_value=DummyOkJsonSession()
    ):
        with pytest.raises(HomeAssistantError):
            await engine.async_get_tts(DummyHass(), "hello")


class Dummy401Session:
    def post(self, *args, **kwargs):
        headers = {"content-type": "text/plain"}
        body = b"unauthorized"
        return DummyResponse(401, headers, body)


@pytest.mark.asyncio
async def test_async_get_tts_raises_config_entry_auth_failed_on_401():
    engine = GroqTTSEngine(None, "voice", "model", "http://example.com")
    with patch.object(
        tts_engine, "async_get_clientsession", return_value=Dummy401Session()
    ):
        with pytest.raises(ConfigEntryAuthFailed):
            await engine.async_get_tts(DummyHass(), "hello")


class Dummy429Session:
    def __init__(self):
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        headers = {
            "content-type": "application/json",
            "retry-after": "12",
            "x-ratelimit-reset-requests": "2m59.56s",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-remaining-tokens": "1199",
        }
        body = b'{"error": {"message": "rate limit exceeded"}}'
        return DummyResponse(429, headers, body)


@pytest.mark.asyncio
async def test_async_get_tts_raises_rate_limit_error_on_429():
    session = Dummy429Session()
    engine = GroqTTSEngine(
        None,
        ORPHEUS_ENGLISH_VOICE,
        ORPHEUS_ENGLISH_MODEL,
        "http://example.com",
    )
    with patch.object(tts_engine, "async_get_clientsession", return_value=session):
        with pytest.raises(GroqRateLimitError, match="retry after 12 seconds"):
            await engine.async_get_tts(DummyHass(), "hello")

    assert session.calls == 1


class DummyCaptureSession:
    def __init__(self):
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        headers = {"content-type": "audio/wav"}
        body = b"RIFF....WAVEfmt "
        return DummyResponse(200, headers, body)


@pytest.mark.asyncio
async def test_async_get_tts_posts_orpheus_wav_payload():
    session = DummyCaptureSession()
    engine = GroqTTSEngine(
        "api-key",
        ORPHEUS_ENGLISH_VOICE,
        ORPHEUS_ENGLISH_MODEL,
        "https://api.groq.com/openai/v1/audio/speech",
    )

    with patch.object(tts_engine, "async_get_clientsession", return_value=session):
        response = await engine.async_get_tts(DummyHass(), "hello")

    assert response.content == b"RIFF....WAVEfmt "
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["args"] == ("https://api.groq.com/openai/v1/audio/speech",)
    assert call["kwargs"]["json"] == {
        "model": ORPHEUS_ENGLISH_MODEL,
        "input": "hello",
        "voice": ORPHEUS_ENGLISH_VOICE,
        "response_format": "wav",
    }
    assert call["kwargs"]["headers"]["Authorization"] == "Bearer api-key"


@pytest.mark.asyncio
async def test_async_get_tts_accepts_model_voice_and_response_format_overrides():
    session = DummyCaptureSession()
    engine = GroqTTSEngine(
        "api-key",
        "default-voice",
        "default-model",
        "https://api.groq.com/openai/v1/audio/speech",
    )

    with patch.object(tts_engine, "async_get_clientsession", return_value=session):
        await engine.async_get_tts(
            DummyHass(),
            "hello",
            voice=ORPHEUS_ENGLISH_VOICE,
            model=ORPHEUS_ENGLISH_MODEL,
            response_format="wav",
        )

    assert session.calls[0]["kwargs"]["json"] == {
        "model": ORPHEUS_ENGLISH_MODEL,
        "input": "hello",
        "voice": ORPHEUS_ENGLISH_VOICE,
        "response_format": "wav",
    }


@pytest.mark.asyncio
async def test_async_get_tts_local_free_tier_guard_blocks_eleventh_minute_request():
    session = DummyCaptureSession()
    engine = GroqTTSEngine(
        "api-key",
        ORPHEUS_ENGLISH_VOICE,
        ORPHEUS_ENGLISH_MODEL,
        "https://api.groq.com/openai/v1/audio/speech",
    )

    with patch.object(tts_engine, "async_get_clientsession", return_value=session):
        for index in range(10):
            await engine.async_get_tts(DummyHass(), f"hello {index}")
        with pytest.raises(GroqRateLimitError, match="requests per minute"):
            await engine.async_get_tts(DummyHass(), "hello blocked")

    assert len(session.calls) == 10


@pytest.mark.asyncio
async def test_async_get_tts_free_tier_guard_ignores_cache_hits():
    session = DummyCaptureSession()
    engine = GroqTTSEngine(
        "api-key",
        ORPHEUS_ENGLISH_VOICE,
        ORPHEUS_ENGLISH_MODEL,
        "https://api.groq.com/openai/v1/audio/speech",
    )

    with patch.object(tts_engine, "async_get_clientsession", return_value=session):
        for _ in range(20):
            await engine.async_get_tts(DummyHass(), "same message")

    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_async_get_tts_cache_hit_log_redacts_text(caplog):
    session = DummyCaptureSession()
    engine = GroqTTSEngine(
        "api-key",
        ORPHEUS_ENGLISH_VOICE,
        ORPHEUS_ENGLISH_MODEL,
        "https://api.groq.com/openai/v1/audio/speech",
    )

    with patch.object(tts_engine, "async_get_clientsession", return_value=session):
        await engine.async_get_tts(DummyHass(), "private spoken message")
        with caplog.at_level(logging.DEBUG, logger="custom_components.groq.tts_engine"):
            await engine.async_get_tts(DummyHass(), "private spoken message")

    assert "private spoken message" not in caplog.text
    assert "text_hash=" in caplog.text


def test_local_free_tier_guard_can_be_disabled():
    engine = GroqTTSEngine(
        "api-key",
        ORPHEUS_ENGLISH_VOICE,
        ORPHEUS_ENGLISH_MODEL,
        "https://api.groq.com/openai/v1/audio/speech",
        protect_free_tier=False,
    )

    for _ in range(20):
        engine._record_local_usage(200, now=1)

    assert engine._check_local_free_tier_limit("hello", now=1) == 5


class DummyEngine:
    class _Resp:
        def __init__(self, content: bytes):
            self.content = content

    async def async_get_tts(
        self, hass, text, voice=None, model=None, response_format=None
    ):
        return self._Resp(b"audio-bytes")


class DummyConfigEntry:
    def __init__(self, data: dict, options: dict):
        self.data = data
        self.options = options
        self.unique_id = data.get("unique_id")


@pytest.mark.asyncio
async def test_tts_returns_raw_wav_without_processing():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), DummyEngine())

    ext, payload = await entity.async_get_tts_audio("Hello", "en", options=None)

    assert ext == "wav"
    assert payload == b"audio-bytes"


@pytest.mark.asyncio
async def test_tts_rejects_orpheus_input_over_200_chars():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), DummyEngine())

    ext, payload = await entity.async_get_tts_audio("x" * 201, "en", options=None)

    assert ext is None
    assert payload is None


class DummyProc:
    def __init__(self, returncode: int):
        self.returncode = returncode

    async def communicate(self, input=None):  # noqa: A002
        return b"", b"ffmpeg error"


@pytest.mark.asyncio
async def test_tts_ffmpeg_failure_returns_none(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    options = {"normalize_audio": True}
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, options), DummyEngine())

    async def fake_exec(*args, **kwargs):  # noqa: ANN001, D401
        return DummyProc(returncode=1)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    ext, payload = await entity.async_get_tts_audio("Hello", "en", options=None)
    assert ext is None and payload is None
