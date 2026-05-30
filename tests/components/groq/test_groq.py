import asyncio
from collections import OrderedDict
import logging
import struct
from types import SimpleNamespace

import aiohttp
import pytest
from homeassistant.exceptions import ConfigEntryAuthFailed

from custom_components.groq import api, config_flow, tts
from custom_components.groq.api import GroqApiClient, SpeechRequest
from custom_components.groq.const import normalize_enabled_features
from custom_components.groq.errors import GroqApiError, GroqRateLimitExceeded
from custom_components.groq.tts import (
    GroqTTSEntity,
    _split_overlong_tts_segment,
    _split_tts_text,
    _tts_input_chunks,
)

validate_user_input = config_flow.validate_user_input
get_model_options = config_flow.get_model_options

ORPHEUS_ENGLISH_MODEL = "canopylabs/orpheus-v1-english"
ORPHEUS_ENGLISH_VOICE = "troy"
PCM_WAV_BYTES = (
    b"RIFF"
    + struct.pack("<I", 40)
    + b"WAVEfmt "
    + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
    + b"data"
    + struct.pack("<I", 4)
    + b"\0\0\0\0"
)
FLOAT_WAV_BYTES = (
    b"RIFF"
    + struct.pack("<I", 40)
    + b"WAVEfmt "
    + struct.pack("<IHHIIHH", 16, 3, 1, 24000, 96000, 4, 32)
    + b"data"
    + struct.pack("<I", 4)
    + b"\0\0\0\0"
)
FORMAT_ONLY_WAV_BYTES = (
    b"RIFF"
    + struct.pack("<I", 36)
    + b"WAVEfmt "
    + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
)
TRUNCATED_CHUNK_WAV_BYTES = b"RIFF" + struct.pack("<I", 16) + b"WAVEdata\x04\0\0\0\0"
SHORT_FORMAT_WAV_BYTES = (
    b"RIFF" + struct.pack("<I", 16) + b"WAVEfmt " + struct.pack("<I", 4) + b"\0" * 4
)


async def _async_return(value):
    return value


def test_tts_wav_compatibility_parser_rejects_malformed_chunks():
    assert tts._audio_needs_compatibility_transcode(TRUNCATED_CHUNK_WAV_BYTES) is True
    assert tts._audio_needs_compatibility_transcode(SHORT_FORMAT_WAV_BYTES) is True


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
    def request(self, *args, **kwargs):
        raise aiohttp.ClientError("boom")


class DummyTimeoutResponse:
    async def __aenter__(self):
        raise asyncio.TimeoutError

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummyTimeoutSession:
    def __init__(self):
        self.calls = []

    def request(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return DummyTimeoutResponse()


class DummyHass:
    pass


@pytest.mark.asyncio
async def test_api_clientsession_helper_preloads_with_executor(monkeypatch):
    from custom_components.groq import api

    calls = []

    def factory(hass):
        return ("session", hass)

    def load_factory():
        calls.append("load")
        return factory

    async def async_add_executor_job(func, *args):
        calls.append("executor")
        return func(*args)

    monkeypatch.setattr(api, "_CLIENTSESSION_FACTORY", None)
    monkeypatch.setattr(api, "_load_clientsession_factory", load_factory)

    hass = SimpleNamespace(async_add_executor_job=async_add_executor_job)
    await api.async_preload_clientsession_helper(hass)

    assert api.async_get_clientsession("hass") == ("session", "hass")
    assert calls == ["executor", "load"]


@pytest.mark.asyncio
async def test_api_clientsession_helper_falls_back_without_executor(monkeypatch):
    from custom_components.groq import api

    calls = []

    def factory(hass):
        return ("session", hass)

    def load_factory():
        calls.append("load")
        return factory

    monkeypatch.setattr(api, "_CLIENTSESSION_FACTORY", None)
    monkeypatch.setattr(api, "_load_clientsession_factory", load_factory)

    await api.async_preload_clientsession_helper(SimpleNamespace())

    assert api.async_get_clientsession("hass") == ("session", "hass")
    assert calls == ["load"]


def test_api_clientsession_helper_loads_on_direct_use(monkeypatch):
    from custom_components.groq import api

    calls = []

    def factory(hass):
        return ("session", hass)

    def load_factory():
        calls.append("load")
        return factory

    monkeypatch.setattr(api, "_CLIENTSESSION_FACTORY", None)
    monkeypatch.setattr(api, "_load_clientsession_factory", load_factory)

    assert api.async_get_clientsession("hass") == ("session", "hass")
    assert calls == ["load"]


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
    def request(self, *args, **kwargs):
        headers = {"content-type": "application/json"}
        body = b'{"ok": true}'
        return DummyResponse(200, headers, body)


@pytest.mark.asyncio
async def test_synthesize_speech_non_audio_2xx():
    client = GroqApiClient(DummyHass(), api_key=None, session=DummyOkJsonSession())
    with pytest.raises(GroqApiError):
        await client.async_synthesize_speech(
            SpeechRequest(text="hello", model="model", voice="voice")
        )


class Dummy401Session:
    def request(self, *args, **kwargs):
        headers = {"content-type": "text/plain"}
        body = b"unauthorized"
        return DummyResponse(401, headers, body)


@pytest.mark.asyncio
async def test_synthesize_speech_raises_config_entry_auth_failed_on_401():
    client = GroqApiClient(DummyHass(), api_key=None, session=Dummy401Session())
    with pytest.raises(ConfigEntryAuthFailed):
        await client.async_synthesize_speech(
            SpeechRequest(text="hello", model="model", voice="voice")
        )


class Dummy429Session:
    def __init__(self):
        self.calls = 0

    def request(self, *args, **kwargs):
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
async def test_synthesize_speech_raises_rate_limit_error_on_429():
    session = Dummy429Session()
    client = GroqApiClient(DummyHass(), api_key=None, session=session)
    with pytest.raises(GroqRateLimitExceeded, match="retry after 12 seconds"):
        await client.async_synthesize_speech(
            SpeechRequest(
                text="hello",
                model="custom-tts",
                voice=ORPHEUS_ENGLISH_VOICE,
            )
        )

    assert session.calls == 1


class DummyCaptureSession:
    def __init__(self):
        self.calls = []

    def request(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        headers = {"content-type": "audio/wav"}
        body = b"RIFF....WAVEfmt "
        return DummyResponse(200, headers, body)


class DummyFlakyAudioSession:
    def __init__(self):
        self.calls = []

    def request(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if len(self.calls) == 1:
            raise aiohttp.ClientError("temporary network error")
        headers = {"content-type": "audio/wav"}
        body = b"RIFF....WAVEfmt "
        return DummyResponse(200, headers, body)


class DummyFailingAudioSession:
    def __init__(self, error: BaseException):
        self.error = error
        self.calls = []

    def request(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        raise self.error


@pytest.mark.asyncio
async def test_synthesize_speech_retries_transient_audio_network_error(monkeypatch):
    session = DummyFlakyAudioSession()
    client = GroqApiClient(DummyHass(), api_key="api-key", session=session)

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(api.asyncio, "sleep", no_sleep)

    response = await client.async_synthesize_speech(
        SpeechRequest(
            text="hello",
            model=ORPHEUS_ENGLISH_MODEL,
            voice=ORPHEUS_ENGLISH_VOICE,
        )
    )

    assert response == b"RIFF....WAVEfmt "
    assert len(session.calls) == 2
    assert client.available is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "message"),
    [
        (aiohttp.ClientError("network down"), "Network error calling Groq API"),
        (TimeoutError("slow"), "Timed out calling Groq API"),
    ],
)
async def test_synthesize_speech_final_audio_network_error_marks_unavailable(
    monkeypatch, error, message
):
    session = DummyFailingAudioSession(error)
    client = GroqApiClient(DummyHass(), api_key="api-key", session=session)

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(api.asyncio, "sleep", no_sleep)

    with pytest.raises(GroqApiError, match=message):
        await client.async_synthesize_speech(
            SpeechRequest(
                text="hello",
                model=ORPHEUS_ENGLISH_MODEL,
                voice=ORPHEUS_ENGLISH_VOICE,
            )
        )

    assert len(session.calls) == 2
    assert client.available is False


@pytest.mark.asyncio
async def test_synthesize_speech_posts_orpheus_wav_payload():
    session = DummyCaptureSession()
    client = GroqApiClient(DummyHass(), api_key="api-key", session=session)

    response = await client.async_synthesize_speech(
        SpeechRequest(
            text="hello",
            model=ORPHEUS_ENGLISH_MODEL,
            voice=ORPHEUS_ENGLISH_VOICE,
        )
    )
    assert response == b"RIFF....WAVEfmt "
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["args"] == ("POST", "https://api.groq.com/openai/v1/audio/speech")
    assert call["kwargs"]["json"] == {
        "model": ORPHEUS_ENGLISH_MODEL,
        "input": "hello",
        "voice": ORPHEUS_ENGLISH_VOICE,
        "response_format": "wav",
    }
    assert call["kwargs"]["headers"]["Authorization"] == "Bearer api-key"


@pytest.mark.asyncio
async def test_synthesize_speech_accepts_model_voice_and_response_format():
    session = DummyCaptureSession()
    client = GroqApiClient(DummyHass(), api_key="api-key", session=session)

    await client.async_synthesize_speech(
        SpeechRequest(
            text="hello",
            model=ORPHEUS_ENGLISH_MODEL,
            voice=ORPHEUS_ENGLISH_VOICE,
            response_format="wav",
        )
    )

    assert session.calls[0]["kwargs"]["json"] == {
        "model": ORPHEUS_ENGLISH_MODEL,
        "input": "hello",
        "voice": ORPHEUS_ENGLISH_VOICE,
        "response_format": "wav",
    }


@pytest.mark.asyncio
async def test_synthesize_speech_local_free_tier_guard_blocks_eleventh_request():
    session = DummyCaptureSession()
    client = GroqApiClient(DummyHass(), api_key="api-key", session=session)
    for index in range(10):
        await client.async_synthesize_speech(
            SpeechRequest(
                text=f"hello {index}",
                model=ORPHEUS_ENGLISH_MODEL,
                voice=ORPHEUS_ENGLISH_VOICE,
            )
        )
    with pytest.raises(GroqApiError, match="requests per minute"):
        await client.async_synthesize_speech(
            SpeechRequest(
                text="hello blocked",
                model=ORPHEUS_ENGLISH_MODEL,
                voice=ORPHEUS_ENGLISH_VOICE,
            )
        )

    assert len(session.calls) == 10


@pytest.mark.asyncio
async def test_synthesize_speech_free_tier_guard_ignores_cache_hits():
    session = DummyCaptureSession()
    client = GroqApiClient(DummyHass(), api_key="api-key", session=session)
    for _ in range(20):
        await client.async_synthesize_speech(
            SpeechRequest(
                text="same message",
                model=ORPHEUS_ENGLISH_MODEL,
                voice=ORPHEUS_ENGLISH_VOICE,
            )
        )

    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_synthesize_speech_cache_can_be_disabled():
    session = DummyCaptureSession()
    client = GroqApiClient(DummyHass(), api_key="api-key", session=session)
    request = SpeechRequest(
        text="same message",
        model=ORPHEUS_ENGLISH_MODEL,
        voice=ORPHEUS_ENGLISH_VOICE,
        cache_max=0,
    )

    await client.async_synthesize_speech(request)
    await client.async_synthesize_speech(request)

    assert len(session.calls) == 2
    assert client._speech_caches == {}


@pytest.mark.asyncio
async def test_synthesize_speech_header_guard_does_not_record_local_usage():
    session = DummyCaptureSession()
    client = GroqApiClient(DummyHass(), api_key="api-key", session=session)
    client._rate_limiter.update_from_headers("tts-service", {"retry-after": "60"})

    with pytest.raises(GroqRateLimitExceeded, match="retry after"):
        await client.async_synthesize_speech(
            SpeechRequest(
                text="blocked before send",
                model=ORPHEUS_ENGLISH_MODEL,
                voice=ORPHEUS_ENGLISH_VOICE,
                service_id="tts-service",
            )
        )

    assert session.calls == []
    assert client._tts_usage == {}


@pytest.mark.asyncio
async def test_synthesize_speech_cache_hit_log_redacts_text(caplog):
    session = DummyCaptureSession()
    client = GroqApiClient(DummyHass(), api_key="api-key", session=session)
    request = SpeechRequest(
        text="private spoken message",
        model=ORPHEUS_ENGLISH_MODEL,
        voice=ORPHEUS_ENGLISH_VOICE,
    )
    await client.async_synthesize_speech(request)
    with caplog.at_level(logging.DEBUG, logger="custom_components.groq.api"):
        await client.async_synthesize_speech(request)

    assert "private spoken message" not in caplog.text
    assert "text_hash=" in caplog.text


def test_local_free_tier_guard_can_be_disabled():
    client = GroqApiClient(DummyHass(), api_key="api-key")
    request = SpeechRequest(
        text="hello",
        model=ORPHEUS_ENGLISH_MODEL,
        voice=ORPHEUS_ENGLISH_VOICE,
        protect_free_tier=False,
    )
    assert client._check_local_tts_free_tier_limit(request, now=1) == 5


def test_tts_local_usage_counters_prune_minute_and_day_windows():
    client = GroqApiClient(DummyHass(), api_key="api-key")
    request = SpeechRequest(
        text="hello",
        model=ORPHEUS_ENGLISH_MODEL,
        voice=ORPHEUS_ENGLISH_VOICE,
    )
    state = client._tts_usage_state(request)

    client._record_local_tts_usage(request, 3, now=100.0)
    client._record_local_tts_usage(request, 5, now=150.0)

    client._prune_local_tts_usage(state, 161.0)
    assert len(state.minute_request_timestamps) == 1
    assert list(state.minute_request_timestamps) == [150.0]
    assert list(state.minute_token_timestamps) == [(150.0, 5)]

    after_daily_window = 150.0 + api.RATE_LIMIT_DAY_SECONDS + 1
    client._prune_local_tts_usage(state, after_daily_window)
    assert state.daily_token_total == 0
    assert state.minute_token_total == 0


def test_tts_local_usage_counters_drive_token_limit_checks(monkeypatch):
    client = GroqApiClient(DummyHass(), api_key="api-key")
    request = SpeechRequest(
        text="hi",
        model=ORPHEUS_ENGLISH_MODEL,
        voice=ORPHEUS_ENGLISH_VOICE,
    )
    monkeypatch.setattr(
        client,
        "_free_tier_limits",
        lambda model: {
            "requests_per_minute": 100,
            "requests_per_day": 100,
            "tokens_per_minute": 10,
            "tokens_per_day": 100,
        },
    )

    client._record_local_tts_usage(request, 7, now=100.0)
    client._record_local_tts_usage(request, 2, now=150.0)

    with pytest.raises(GroqApiError, match="tokens per minute"):
        client._check_local_tts_free_tier_limit(request, now=150.0)

    assert client._check_local_tts_free_tier_limit(request, now=161.0) == 2


def test_tts_batch_free_tier_guard_blocks_partial_long_tts_batches(monkeypatch):
    client = GroqApiClient(DummyHass(), api_key="api-key")
    request = SpeechRequest(
        text="existing",
        model=ORPHEUS_ENGLISH_MODEL,
        voice=ORPHEUS_ENGLISH_VOICE,
    )
    monkeypatch.setattr(
        client,
        "_free_tier_limits",
        lambda model: {
            "requests_per_minute": 3,
            "requests_per_day": 100,
            "tokens_per_minute": 1000,
            "tokens_per_day": 1000,
        },
    )

    client._record_local_tts_usage(request, 1, now=100.0)
    client._record_local_tts_usage(request, 1, now=100.0)

    with pytest.raises(GroqApiError, match="batch usage"):
        client._check_local_tts_free_tier_batch(
            [
                SpeechRequest(
                    text="hello",
                    model=ORPHEUS_ENGLISH_MODEL,
                    voice=ORPHEUS_ENGLISH_VOICE,
                ),
                SpeechRequest(
                    text="again",
                    model=ORPHEUS_ENGLISH_MODEL,
                    voice=ORPHEUS_ENGLISH_VOICE,
                ),
            ],
            now=100.0,
        )

    assert client._check_local_tts_free_tier_batch(
        [
            SpeechRequest(
                text="hello",
                model=ORPHEUS_ENGLISH_MODEL,
                voice=ORPHEUS_ENGLISH_VOICE,
            )
        ],
        now=100.0,
    ) == [5]


def test_tts_batch_free_tier_guard_ignores_cached_chunks(monkeypatch):
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
    cached_texts = ["cached one", "cached two"]
    namespace = f"{ORPHEUS_ENGLISH_MODEL}:{ORPHEUS_ENGLISH_VOICE}"
    cache = client._speech_caches.setdefault(namespace, OrderedDict())
    for text in cached_texts:
        cache[(ORPHEUS_ENGLISH_MODEL, ORPHEUS_ENGLISH_VOICE, "wav", text)] = b"cached"
    request = SpeechRequest(
        text="existing",
        model=ORPHEUS_ENGLISH_MODEL,
        voice=ORPHEUS_ENGLISH_VOICE,
    )
    client._record_local_tts_usage(request, 1, now=100.0)

    assert client._check_local_tts_free_tier_batch(
        [
            SpeechRequest(
                text=text,
                model=ORPHEUS_ENGLISH_MODEL,
                voice=ORPHEUS_ENGLISH_VOICE,
            )
            for text in cached_texts
        ],
        now=100.0,
    ) == [10, 10]

    with pytest.raises(GroqApiError, match="batch usage"):
        client._check_local_tts_free_tier_batch(
            [
                *[
                    SpeechRequest(
                        text=text,
                        model=ORPHEUS_ENGLISH_MODEL,
                        voice=ORPHEUS_ENGLISH_VOICE,
                    )
                    for text in cached_texts
                ],
                SpeechRequest(
                    text="uncached",
                    model=ORPHEUS_ENGLISH_MODEL,
                    voice=ORPHEUS_ENGLISH_VOICE,
                ),
            ],
            now=100.0,
        )


def test_tts_batch_free_tier_guard_simulates_cache_evictions(monkeypatch):
    client = GroqApiClient(DummyHass(), api_key="api-key")
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
    first_text = "cached first"
    second_text = "uncached second"
    third_text = "cached third"
    namespace = f"{ORPHEUS_ENGLISH_MODEL}:{ORPHEUS_ENGLISH_VOICE}"
    cache = client._speech_caches.setdefault(namespace, OrderedDict())
    for text in (first_text, third_text):
        cache[(ORPHEUS_ENGLISH_MODEL, ORPHEUS_ENGLISH_VOICE, "wav", text)] = b"cached"
    request = SpeechRequest(
        text="existing",
        model=ORPHEUS_ENGLISH_MODEL,
        voice=ORPHEUS_ENGLISH_VOICE,
    )
    client._record_local_tts_usage(request, 1, now=100.0)

    with pytest.raises(GroqApiError, match="batch usage"):
        client._check_local_tts_free_tier_batch(
            [
                SpeechRequest(
                    text=first_text,
                    model=ORPHEUS_ENGLISH_MODEL,
                    voice=ORPHEUS_ENGLISH_VOICE,
                    cache_max=2,
                ),
                SpeechRequest(
                    text=second_text,
                    model=ORPHEUS_ENGLISH_MODEL,
                    voice=ORPHEUS_ENGLISH_VOICE,
                    cache_max=2,
                ),
                SpeechRequest(
                    text=third_text,
                    model=ORPHEUS_ENGLISH_MODEL,
                    voice=ORPHEUS_ENGLISH_VOICE,
                    cache_max=2,
                ),
            ],
            now=100.0,
        )

    assert list(client._speech_caches[namespace]) == [
        (ORPHEUS_ENGLISH_MODEL, ORPHEUS_ENGLISH_VOICE, "wav", first_text),
        (ORPHEUS_ENGLISH_MODEL, ORPHEUS_ENGLISH_VOICE, "wav", third_text),
    ]


def test_tts_batch_free_tier_guard_returns_when_model_has_no_limits(monkeypatch):
    client = GroqApiClient(DummyHass(), api_key="api-key")
    monkeypatch.setattr(client, "_free_tier_limits", lambda model: None)

    assert client._check_local_tts_free_tier_batch(
        [
            SpeechRequest(
                text="hello",
                model=ORPHEUS_ENGLISH_MODEL,
                voice=ORPHEUS_ENGLISH_VOICE,
            )
        ],
        now=100.0,
    ) == [5]


def test_tts_batch_free_tier_guard_skips_unprotected_requests(monkeypatch):
    client = GroqApiClient(DummyHass(), api_key="api-key")
    monkeypatch.setattr(
        client,
        "_free_tier_limits",
        lambda model: {
            "requests_per_minute": 0,
            "requests_per_day": 0,
            "tokens_per_minute": 0,
            "tokens_per_day": 0,
        },
    )

    assert client._check_local_tts_free_tier_batch(
        [
            SpeechRequest(
                text="hello",
                model=ORPHEUS_ENGLISH_MODEL,
                voice=ORPHEUS_ENGLISH_VOICE,
                protect_free_tier=False,
            )
        ],
        now=100.0,
    ) == [5]


def test_tts_batch_free_tier_guard_blocks_daily_requests(monkeypatch):
    client = GroqApiClient(DummyHass(), api_key="api-key")
    request = SpeechRequest(
        text="existing",
        model=ORPHEUS_ENGLISH_MODEL,
        voice=ORPHEUS_ENGLISH_VOICE,
    )
    monkeypatch.setattr(
        client,
        "_free_tier_limits",
        lambda model: {
            "requests_per_minute": 100,
            "requests_per_day": 1,
            "tokens_per_minute": 1000,
            "tokens_per_day": 1000,
        },
    )
    client._record_local_tts_usage(request, 1, now=100.0)

    with pytest.raises(GroqApiError, match="requests per day"):
        client._check_local_tts_free_tier_batch([request], now=100.0)


def test_tts_batch_free_tier_guard_blocks_minute_tokens(monkeypatch):
    client = GroqApiClient(DummyHass(), api_key="api-key")
    request = SpeechRequest(
        text="hello",
        model=ORPHEUS_ENGLISH_MODEL,
        voice=ORPHEUS_ENGLISH_VOICE,
    )
    monkeypatch.setattr(
        client,
        "_free_tier_limits",
        lambda model: {
            "requests_per_minute": 100,
            "requests_per_day": 100,
            "tokens_per_minute": 6,
            "tokens_per_day": 1000,
        },
    )
    client._record_local_tts_usage(request, 2, now=100.0)

    with pytest.raises(GroqApiError, match="tokens per minute"):
        client._check_local_tts_free_tier_batch([request], now=100.0)


def test_tts_batch_free_tier_guard_blocks_daily_tokens(monkeypatch):
    client = GroqApiClient(DummyHass(), api_key="api-key")
    request = SpeechRequest(
        text="hello",
        model=ORPHEUS_ENGLISH_MODEL,
        voice=ORPHEUS_ENGLISH_VOICE,
    )
    monkeypatch.setattr(
        client,
        "_free_tier_limits",
        lambda model: {
            "requests_per_minute": 100,
            "requests_per_day": 100,
            "tokens_per_minute": 1000,
            "tokens_per_day": 6,
        },
    )
    client._record_local_tts_usage(request, 2, now=100.0)

    with pytest.raises(GroqApiError, match="tokens per day"):
        client._check_local_tts_free_tier_batch([request], now=100.0)


def test_tts_split_helpers_handle_overlong_words_and_blank_segments(monkeypatch):
    assert _split_overlong_tts_segment("aa bbb cc", 5) == ["aa", "bbb", "cc"]
    assert _split_overlong_tts_segment("aa bbbbbbb", 5) == ["aa", "bbbbb", "bb"]
    assert _split_overlong_tts_segment("   ", 2) == ["  ", " "]
    assert _split_tts_text("   ", 5) == [""]

    class Boundary:
        def split(self, text):
            return ["", "Hello.", "World."]

    monkeypatch.setattr(tts, "_SENTENCE_BOUNDARY", Boundary())
    assert _split_tts_text("ignored", 20) == ["Hello. World."]


def test_tts_input_chunks_rejects_vocal_directions_that_fill_limit():
    with pytest.raises(ValueError, match="Vocal directions leave no room"):
        _tts_input_chunks("hello", "[too long]", max_chars=10)


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


class DummyBatchGuardClient(DummyClient):
    def _check_local_tts_free_tier_batch(self, requests):
        raise GroqApiError(
            "batch would exceed local free-tier usage",
            status=429,
            error_type="rate_limit_exceeded",
        )


class ExplodingClient(DummyClient):
    async def async_synthesize_speech(self, request):
        raise RuntimeError("boom")


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
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), DummyClient())

    ext, payload = await entity.async_get_tts_audio("Hello", "en", options=None)

    assert ext == "wav"
    assert payload == PCM_WAV_BYTES


@pytest.mark.asyncio
async def test_tts_rewrites_non_pcm_wav_for_playback_compatibility():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)
    client.async_synthesize_speech = lambda request: _async_return(FLOAT_WAV_BYTES)
    ffmpeg_inputs = []

    async def skip_ffmpeg_check(_output_format, _normalize_audio):
        return None

    async def rewrite_wav(_cmd, input_bytes=None, *, create_repair=True):
        ffmpeg_inputs.append((input_bytes, create_repair))
        return b"RIFF-compatible-wav"

    entity._async_check_ffmpeg = skip_ffmpeg_check
    entity._async_run_ffmpeg = rewrite_wav

    assert await entity.async_get_tts_audio("Hello", "en") == (
        "wav",
        b"RIFF-compatible-wav",
    )
    assert ffmpeg_inputs == [(FLOAT_WAV_BYTES, False)]


@pytest.mark.asyncio
async def test_tts_rewrites_non_wav_payload_served_as_wav():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)
    client.async_synthesize_speech = lambda request: _async_return(b"audio-bytes")
    ffmpeg_inputs = []

    async def skip_ffmpeg_check(_output_format, _normalize_audio):
        return None

    async def rewrite_wav(_cmd, input_bytes=None, *, create_repair=True):
        ffmpeg_inputs.append((input_bytes, create_repair))
        return b"RIFF-compatible-wav"

    entity._async_check_ffmpeg = skip_ffmpeg_check
    entity._async_run_ffmpeg = rewrite_wav

    assert await entity.async_get_tts_audio("Hello", "en") == (
        "wav",
        b"RIFF-compatible-wav",
    )
    assert ffmpeg_inputs == [(b"audio-bytes", False)]


@pytest.mark.asyncio
async def test_tts_rewrites_wav_without_data_chunk():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)
    client.async_synthesize_speech = lambda request: _async_return(
        FORMAT_ONLY_WAV_BYTES
    )
    ffmpeg_inputs = []

    async def skip_ffmpeg_check(_output_format, _normalize_audio):
        return None

    async def rewrite_wav(_cmd, input_bytes=None, *, create_repair=True):
        ffmpeg_inputs.append((input_bytes, create_repair))
        return b"RIFF-compatible-wav"

    entity._async_check_ffmpeg = skip_ffmpeg_check
    entity._async_run_ffmpeg = rewrite_wav

    assert await entity.async_get_tts_audio("Hello", "en") == (
        "wav",
        b"RIFF-compatible-wav",
    )
    assert ffmpeg_inputs == [(FORMAT_ONLY_WAV_BYTES, False)]


@pytest.mark.asyncio
async def test_tts_unknown_client_error_returns_none():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), ExplodingClient())

    ext, payload = await entity.async_get_tts_audio("Hello", "en", options=None)

    assert ext is None
    assert payload is None


@pytest.mark.asyncio
async def test_tts_rejects_whitespace_only_input_before_api():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)

    ext, payload = await entity.async_get_tts_audio(" " * 201, "en", options=None)

    assert ext is None
    assert payload is None
    assert client.calls == []


@pytest.mark.asyncio
async def test_tts_rejects_orpheus_input_over_200_chars_without_normalization():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)

    ext, payload = await entity.async_get_tts_audio("x" * 201, "en", options=None)

    assert ext is None
    assert payload is None
    assert client.calls == []


@pytest.mark.asyncio
async def test_tts_rejects_long_input_when_long_tts_disabled_with_normalization():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)

    ext, payload = await entity.async_get_tts_audio(
        "x" * 201,
        "en",
        options={"normalize_audio": True},
    )

    assert ext is None
    assert payload is None
    assert client.calls == []


@pytest.mark.asyncio
async def test_tts_rejects_per_call_long_tts_override_when_config_disabled():
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyClient()
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, {}), client)

    ext, payload = await entity.async_get_tts_audio(
        "x" * 201,
        "en",
        options={"enable_long_tts": True},
    )

    assert ext is None
    assert payload is None
    assert client.calls == []


@pytest.mark.asyncio
async def test_tts_checks_ffmpeg_before_sending_long_tts_chunks(monkeypatch):
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

    async def missing_ffmpeg(*args, **kwargs):  # noqa: ANN001
        raise FileNotFoundError

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing_ffmpeg)

    ext, payload = await entity.async_get_tts_audio(
        f"{'A' * 198}. {'B' * 40}.",
        "en",
        options=None,
    )

    assert ext is None
    assert payload is None
    assert client.calls == []


@pytest.mark.asyncio
async def test_tts_checks_batch_free_tier_guard_before_long_tts_api(monkeypatch):
    data = {
        "url": "http://example.com",
        "model": ORPHEUS_ENGLISH_MODEL,
        "voice": ORPHEUS_ENGLISH_VOICE,
        "unique_id": "uid",
    }
    client = DummyBatchGuardClient()
    entity = GroqTTSEntity(
        DummyHass(),
        DummyConfigEntry(data, {"enable_long_tts": True}),
        client,
    )
    monkeypatch.setattr(
        "custom_components.groq.tts.shutil.which",
        lambda name: "/usr/bin/ffmpeg",
    )

    ext, payload = await entity.async_get_tts_audio(
        f"{'A' * 198}. {'B' * 40}.",
        "en",
        options=None,
    )

    assert ext is None
    assert payload is None
    assert client.calls == []


@pytest.mark.asyncio
async def test_tts_rejects_too_many_long_tts_chunks_before_api(monkeypatch):
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
    monkeypatch.setattr(
        "custom_components.groq.tts.shutil.which",
        lambda name: "/usr/bin/ffmpeg",
    )

    ext, payload = await entity.async_get_tts_audio(
        " ".join(f"{chr(65 + index) * 199}." for index in range(11)),
        "en",
        options=None,
    )

    assert ext is None
    assert payload is None
    assert client.calls == []


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
    entity = GroqTTSEntity(DummyHass(), DummyConfigEntry(data, options), DummyClient())

    async def fake_exec(*args, **kwargs):  # noqa: ANN001, D401
        return DummyProc(returncode=1)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    ext, payload = await entity.async_get_tts_audio("Hello", "en", options=None)
    assert ext is None and payload is None
