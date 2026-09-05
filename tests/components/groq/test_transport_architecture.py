"""Behavioral regressions for transport, cache and media ownership."""

import asyncio
import json
import math
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.exceptions import ConfigEntryAuthFailed, HomeAssistantError

from custom_components.groq import (
    api,
    attachments,
    audio_files,
    flow_schemas,
    prompt_cache,
    tts,
)
from custom_components.groq.api import (
    GroqApiClient,
    SpeechRequest,
)
from custom_components.groq.errors import (
    GroqApiError,
    GroqRateLimitExceeded,
    safe_error_payload,
)
from custom_components.groq.rate_limit import (
    GroqRateLimiter,
    GroqRateLimitInfo,
    _duration_seconds,
    _guard_delay_seconds,
)
from custom_components.groq.tts import _normalize_sample_rate, _normalize_speed


class Hass:
    async def async_add_executor_job(self, func, *args):
        return await asyncio.to_thread(func, *args)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 16000.5])
def test_invalid_numeric_tts_defaults_remain_selector_safe(value):
    assert flow_schemas._sample_rate_default({"sample_rate": value}) is None
    assert flow_schemas._speed_default({"speed": value}) == 1.0


@pytest.mark.asyncio
async def test_ffmpeg_pipe_failure_kills_and_reaps_process():
    process = SimpleNamespace(returncode=None, kill=Mock(), wait=AsyncMock())
    with (
        patch.object(tts.asyncio, "create_subprocess_exec", return_value=process),
        patch.object(
            tts, "async_communicate_audio", side_effect=OSError("broken pipe")
        ),
        pytest.raises(HomeAssistantError, match="pipe I/O"),
    ):
        await tts.GroqTTSEntity._async_run_ffmpeg(
            SimpleNamespace(), ["ffmpeg"], create_repair=False
        )
    process.kill.assert_called_once()
    process.wait.assert_awaited_once()


class Response:
    def __init__(self, status=200, body=b"{}", headers=None):
        self.status = status
        self.body = body
        self.headers = headers or {}
        self.content = self
        self.closed = False

    async def iter_chunked(self, _size):
        yield self.body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True


class Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


async def perform(client, transport, *, service_id="service"):
    kwargs = {"json_payload": {"model": "model"}, "repair_service_id": service_id}
    if transport == "stream":
        return [
            event
            async for event in client._request_stream(
                "POST", "/chat/completions", **kwargs
            )
        ]
    if transport == "audio":
        return await client._request_audio("POST", "/audio/speech", **kwargs)
    return await client._request_json("POST", "/chat/completions", **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["json", "stream", "audio"])
@pytest.mark.parametrize(
    "code", ["model_permission_blocked_org", "model_permission_blocked_project"]
)
async def test_permission_failure_is_model_scoped_not_reauth(transport, code):
    response = Response(
        403,
        json.dumps(
            {
                "error": {
                    "type": "permissions_error",
                    "code": code,
                    "message": "echoed private prompt",
                }
            }
        ).encode(),
    )
    reauth = []
    client = GroqApiClient(
        Hass(),
        api_key="key",
        session=Session(response),
        entry_id="account",
        auth_failure_callback=lambda: reauth.append(True),
    )
    with (
        patch.object(api, "async_create_model_access_issue") as create,
        pytest.raises(GroqApiError) as caught,
    ):
        await perform(client, transport)
    assert caught.value.status == 403
    assert caught.value.error_type == "permissions_error"
    assert "echoed" not in str(caught.value)
    assert caught.value.payload == {
        "error": {"type": "permissions_error", "code": code}
    }
    create.assert_called_once_with(
        client._hass, "model", "service", entry_id="account", reason="permissions"
    )
    assert not reauth
    assert response.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["json", "stream", "audio"])
@pytest.mark.parametrize("status", [401, 403])
async def test_unknown_auth_failure_preserves_reauth(transport, status):
    response = Response(status, b"not json")
    reauth = []
    client = GroqApiClient(
        Hass(),
        api_key="key",
        session=Session(response),
        auth_failure_callback=lambda: reauth.append(True),
    )
    with pytest.raises(ConfigEntryAuthFailed):
        await perform(client, transport)
    assert reauth == [True]
    assert response.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["json", "stream", "audio"])
@pytest.mark.parametrize("body", [b"", b"upstream unavailable", b"[]", b"{bad"])
@pytest.mark.parametrize("status", [429, 503])
async def test_http_error_classification_survives_malformed_body(
    transport, body, status
):
    response = Response(status, body, {"retry-after": "3"})
    client = GroqApiClient(Hass(), api_key="key", session=Session(response))
    with pytest.raises(GroqApiError) as caught:
        await perform(client, transport)
    assert caught.value.status == status
    assert response.closed
    if status == 429:
        assert isinstance(caught.value, GroqRateLimitExceeded)
        assert caught.value.retry_after == "3"
    else:
        assert not client.available


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, message",
    [
        (b"data: \xff\n", "UTF-8"),
        (b'data: {"choices":[{"delta":{"content":"partial"}}]}\n', "before completion"),
        (b'data: {"error":{"message":"private","type":"server_error"}}\n', "HTTP 200"),
    ],
)
async def test_bad_stream_cannot_be_success(body, message):
    response = Response(body=body)
    client = GroqApiClient(Hass(), api_key="key", session=Session(response))
    with pytest.raises(GroqApiError, match=message):
        await perform(client, "stream")
    assert response.closed


@pytest.mark.asyncio
async def test_stream_finish_reason_and_matching_repair_recovery():
    response = Response(body=b'data: {"choices":[{"finish_reason":"stop"}]}\n')
    client = GroqApiClient(
        Hass(), api_key="key", session=Session(response), entry_id="account"
    )
    with patch.object(api, "async_delete_model_access_issue") as clear:
        assert len(await perform(client, "stream")) == 1
    clear.assert_called_once_with(client._hass, "model", "service", entry_id="account")


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2m59.56s", 180),
        ("1h2m3s4ms", 3724),
        ("7.66s", 8),
        ("250ms", 1),
        (" 3.1 ", 4),
        ("0", 1),
        ("-1", None),
        ("inf", None),
        ("nan", None),
        ("infs", None),
        ("nans", None),
        ("2mgarbage3s", None),
        ("", None),
        ("2m3", None),
    ],
)
def test_duration_headers(value, expected):
    assert _duration_seconds(value) == expected


def test_guard_uses_only_exhausted_windows_and_preserves_longer_block(monkeypatch):
    assert (
        _guard_delay_seconds(
            GroqRateLimitInfo(
                remaining_requests="1",
                remaining_tokens="0",
                reset_requests="2h",
                reset_tokens="7.66s",
            )
        )
        == 8
    )
    assert (
        _guard_delay_seconds(
            GroqRateLimitInfo(
                retry_after="bad", remaining_requests="0", reset_requests="2m59.56s"
            )
        )
        == 180
    )
    assert (
        _guard_delay_seconds(
            GroqRateLimitInfo(
                remaining_requests="0", remaining_tokens="0", reset_tokens="3s"
            )
        )
        == 60
    )
    limiter = GroqRateLimiter()
    monkeypatch.setattr("custom_components.groq.rate_limit.time.monotonic", lambda: 0)
    limiter.update_from_headers("service", {"retry-after": "10"})
    limiter.update_from_headers("service", {"retry-after": "2"})
    monkeypatch.setattr("custom_components.groq.rate_limit.time.monotonic", lambda: 1.1)
    with pytest.raises(GroqRateLimitExceeded) as caught:
        limiter.raise_if_blocked("service")
    assert caught.value.retry_after == "9"


def test_error_metadata_never_echoes_provider_content():
    secret = "private input api_key=synthetic"
    for payload in [
        None,
        [],
        {"request": secret},
        {"error": secret},
        {"error": {"message": secret, "type": secret, "code": secret}},
    ]:
        assert safe_error_payload(payload) == {}
        error = GroqApiClient._api_error(500, payload)
        assert secret not in str(error)
        assert error.payload == {}
    assert safe_error_payload({"error": {"type": "a" * 65, "code": "bad\n"}}) == {}


def test_prompt_cache_owns_nested_values_and_enforces_bytes(monkeypatch):
    cache = prompt_cache.GroqPromptCache(max_size=3, max_bytes=70)
    original = {"data": {"items": [1]}}
    cache.set("a", original)
    original["data"]["items"].append(2)
    result = cache.get("a")
    result["data"]["items"].append(3)
    assert cache.get("a") == {"data": {"items": [1]}}
    first_size = cache.size_bytes
    cache.set("a", {"data": "x"})
    assert cache.size_bytes < first_size
    cache.set("b", {"data": "b" * 30})
    cache.set("c", {"data": "c" * 30})
    assert cache.size_bytes <= 70
    assert cache.get("a") is None
    cache.set("c", {"data": "z" * 100})
    assert cache.get("c") is None
    cache.clear()
    assert cache.size_bytes == 0
    disabled = prompt_cache.GroqPromptCache(max_bytes=0)
    disabled.set("a", {"x": 1})
    assert disabled.size == 0


def test_expired_entries_are_removed_before_live_eviction(monkeypatch):
    now = [0]
    monkeypatch.setattr(prompt_cache, "monotonic", lambda: now[0])
    cache = prompt_cache.GroqPromptCache(max_size=2)
    cache.set("live", {"v": 1}, ttl=50)
    cache.set("expired", {"v": 2}, ttl=1)
    now[0] = 2
    cache.set("new", {"v": 3})
    assert cache.get("live") == {"v": 1}
    assert cache.get("expired") is None
    assert cache.size == 2


@pytest.mark.parametrize("value", ["nan", float("nan"), "inf", -float("inf")])
def test_tts_rejects_nonfinite_speed(value):
    with pytest.raises(ValueError):
        _normalize_speed(value)


@pytest.mark.parametrize("value", [24000.5, "24000.5", math.inf, math.nan])
def test_tts_rejects_nonintegral_rate(value):
    with pytest.raises(ValueError):
        _normalize_sample_rate(value)


def test_attachment_read_is_bounded(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"123456")
    assert attachments.read_bounded_file(path, 6) == b"123456"
    with patch.object(Path, "open") as opened:
        opened.return_value.__enter__.return_value.read.return_value = b"123456"
        with pytest.raises(ValueError):
            attachments.read_bounded_file(path, 5)
        opened.return_value.__enter__.return_value.read.assert_called_once_with(6)
    with (
        patch.object(attachments, "read_bounded_file", side_effect=PermissionError),
        pytest.raises(HomeAssistantError),
    ):
        attachments._read_attachment_data_url(path, "image/png")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_during", ["create", "write", "body"])
async def test_temp_audio_files_are_cleaned_after_cancellation(
    tmp_path, monkeypatch, cancel_during
):
    started, release = Event(), Event()
    real_write = Path.write_bytes
    directory = tmp_path / "audio"

    def mkdtemp(**_kwargs):
        directory.mkdir()
        if cancel_during == "create":
            started.set()
            release.wait(3)
        return str(directory)

    def write(path, value):
        if cancel_during == "write":
            started.set()
            release.wait(3)
        return real_write(path, value)

    monkeypatch.setattr(audio_files.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(Path, "write_bytes", write)

    async def use():
        async with audio_files.async_audio_chunk_paths(Hass(), [b"audio"]) as paths:
            assert Path(paths[0]).exists()
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(use())
    assert await asyncio.to_thread(started.wait, 3)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not directory.exists()


def test_audio_creation_failure_cleans_directory(tmp_path, monkeypatch):
    directory = tmp_path / "audio"
    directory.mkdir()
    monkeypatch.setattr(audio_files.tempfile, "mkdtemp", lambda **_: str(directory))
    with patch.object(Path, "write_bytes", side_effect=OSError), pytest.raises(OSError):
        audio_files._prepare_audio_chunks([b"audio"])
    assert not directory.exists()


@pytest.mark.asyncio
async def test_duplicate_speech_is_coalesced_and_one_cancel_preserves_other():
    started, release = asyncio.Event(), asyncio.Event()
    client = GroqApiClient(Hass(), api_key="key")
    request = SpeechRequest(text="hello", model="custom", voice="voice")

    async def produce(_request):
        started.set()
        await release.wait()
        return b"audio"

    with patch.object(
        client, "_async_synthesize_speech", side_effect=produce
    ) as synthesis:
        first = asyncio.create_task(client.async_synthesize_speech(request))
        await started.wait()
        second = asyncio.create_task(client.async_synthesize_speech(request))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        assert await second == b"audio"
    assert synthesis.call_count == 1
    assert not client._speech_inflight


@pytest.mark.asyncio
async def test_shutdown_and_last_waiter_cancel_stop_owned_speech():
    for shutdown in (False, True):
        started = asyncio.Event()
        client = GroqApiClient(Hass(), api_key="key")

        async def produce(_request, started=started):
            started.set()
            await asyncio.Event().wait()

        with patch.object(client, "_async_synthesize_speech", side_effect=produce):
            task = asyncio.create_task(
                client.async_synthesize_speech(
                    SpeechRequest(text="x", model="custom", voice="v")
                )
            )
            await started.wait()
            if shutdown:
                await client.async_shutdown()
            else:
                task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not client._speech_inflight
        await client.async_shutdown()
        with pytest.raises(GroqApiError, match="unloaded"):
            await client.async_synthesize_speech(
                SpeechRequest(text="x", model="custom", voice="v")
            )


@pytest.mark.asyncio
async def test_speech_cache_global_budget_and_credential_namespaces(monkeypatch):
    monkeypatch.setattr(api, "MAX_SPEECH_CACHE_BYTES", 35)
    client = GroqApiClient(Hass(), api_key="key")
    with patch.object(client, "_request_audio", return_value=b"12345") as request_audio:
        first = SpeechRequest(text="a", model="custom", voice="v", service_id="one")
        second = SpeechRequest(text="b", model="custom", voice="v", service_id="two")
        await client.async_synthesize_speech(first)
        await client.async_synthesize_speech(first)
        assert request_audio.call_count == 1
        await client.async_synthesize_speech(second)
        assert client._speech_cache_bytes == 19
        await client.async_synthesize_speech(first)
        assert request_audio.call_count == 3
        assert client._speech_namespace(
            SpeechRequest(text="a", model="custom", voice="v", api_key="override")
        ) != client._speech_namespace(first)
    with patch.object(client, "_request_audio", return_value=b"x" * 36):
        await client.async_synthesize_speech(second)
        assert "two" not in client._speech_caches
    await client.async_shutdown()
    assert client._speech_cache_bytes == 0


def test_credential_cache_identity_is_stable_only_within_one_client():
    first = GroqApiClient(Hass(), api_key="account-key")
    second = GroqApiClient(Hass(), api_key="account-key")
    request = SpeechRequest(
        text="hello",
        model="custom",
        voice="v",
        service_id="same-service",
        api_key="override-key",
    )
    namespace = first._speech_namespace(request)
    assert namespace == first._speech_namespace(request)
    assert namespace != second._speech_namespace(request)
    assert "override-key" not in namespace


@pytest.mark.asyncio
async def test_speech_cache_isolates_credentials_within_the_same_service():
    client = GroqApiClient(Hass(), api_key="account-key")
    with patch.object(
        client,
        "_request_audio",
        side_effect=[
            b"first",
            b"second",
            b"default",
            b"explicit-account",
            b"other-service",
        ],
    ) as upstream:
        for service_id, credential, expected in (
            ("same-service", "first-key", b"first"),
            ("same-service", "second-key", b"second"),
            ("same-service", None, b"default"),
            ("same-service", "", b"default"),
            ("same-service", "account-key", b"explicit-account"),
            ("other-service", "first-key", b"other-service"),
        ):
            request = SpeechRequest(
                text="hello",
                model="custom",
                voice="v",
                service_id=service_id,
                api_key=credential,
            )
            assert await client.async_synthesize_speech(request) == expected
            assert await client.async_synthesize_speech(request) == expected
    assert [call.kwargs["api_key"] for call in upstream.call_args_list] == [
        "first-key",
        "second-key",
        None,
        "account-key",
        "first-key",
    ]
    assert len(client._speech_caches) == 5
    assert all(
        "first-key" not in key and "second-key" not in key
        for key in client._speech_caches
    )


@pytest.mark.asyncio
async def test_batch_preflight_uses_the_synthesis_credential_namespace():
    client = GroqApiClient(Hass(), api_key="account-key")
    requests = [
        SpeechRequest(
            text="hello",
            model="custom",
            voice="v",
            service_id="same-service",
            api_key=key,
        )
        for key in ("first-key", "second-key")
    ]
    limits = {
        "requests_per_minute": 1,
        "requests_per_day": 100,
        "tokens_per_minute": 1000,
        "tokens_per_day": 1000,
    }
    with (
        patch.object(client, "_free_tier_limits", return_value=limits),
        patch.object(client, "_request_audio", return_value=b"audio"),
    ):
        assert await client.async_synthesize_speech(requests[0]) == b"audio"
        assert client.check_tts_batch([requests[0]]) == [5]
        with pytest.raises(GroqApiError, match="batch usage"):
            client.check_tts_batch([requests[1]])


@pytest.mark.asyncio
async def test_new_speech_request_waits_for_cancelled_flight_cleanup():
    started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    client = GroqApiClient(Hass(), api_key="key")
    request = SpeechRequest(text="hello", model="custom", voice="voice")
    calls = 0

    async def produce(_request):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
        return b"fresh audio"

    with patch.object(client, "_async_synthesize_speech", side_effect=produce):
        first = asyncio.create_task(client.async_synthesize_speech(request))
        await started.wait()
        first.cancel()
        await cleaning.wait()
        second = asyncio.create_task(client.async_synthesize_speech(request))
        await asyncio.sleep(0)
        assert not second.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await second == b"fresh audio"
    assert calls == 2
    assert not client._speech_inflight


@pytest.mark.asyncio
@pytest.mark.parametrize("api_key", [None, "override-key"])
async def test_failed_speech_does_not_retain_empty_cache_namespaces(api_key):
    client = GroqApiClient(Hass(), api_key="key")
    with patch.object(client, "_request_audio", side_effect=GroqApiError("offline")):
        for index in range(20):
            with pytest.raises(GroqApiError):
                await client.async_synthesize_speech(
                    SpeechRequest(
                        text="hello",
                        model="custom",
                        voice="v",
                        service_id=str(index),
                        api_key=api_key,
                    )
                )
    assert not client._speech_caches
    assert not client._speech_cache_order


@pytest.mark.asyncio
async def test_temp_cleanup_finishes_despite_repeated_cancellation(
    tmp_path, monkeypatch
):
    started, release = Event(), Event()
    real_remove = audio_files.shutil.rmtree
    paths = []

    def remove(directory, ignore_errors):
        paths.append(directory)
        started.set()
        release.wait(3)
        real_remove(directory, ignore_errors=ignore_errors)

    monkeypatch.setattr(audio_files.shutil, "rmtree", remove)

    async def use():
        async with audio_files.async_audio_chunk_paths(Hass(), [b"audio"]):
            pass

    task = asyncio.create_task(use())
    assert await asyncio.to_thread(started.wait, 3)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not Path(paths[0]).exists()


@pytest.mark.asyncio
async def test_queued_speech_does_not_restart_after_shutdown(monkeypatch):
    monkeypatch.setattr(api, "MAX_SPEECH_INFLIGHT", 1)
    started = asyncio.Event()
    client = GroqApiClient(Hass(), api_key="key")

    async def produce(_):
        started.set()
        await asyncio.Event().wait()

    with patch.object(client, "_async_synthesize_speech", side_effect=produce):
        first = asyncio.create_task(
            client.async_synthesize_speech(
                SpeechRequest(text="a", model="m", voice="v")
            )
        )
        await started.wait()
        second = asyncio.create_task(
            client.async_synthesize_speech(
                SpeechRequest(text="b", model="m", voice="v")
            )
        )
        await asyncio.sleep(0)
        await client.async_shutdown()
        with pytest.raises(asyncio.CancelledError):
            await first
        with pytest.raises(GroqApiError, match="unloaded"):
            await second


@pytest.mark.asyncio
async def test_queued_duplicate_speech_rechecks_existing_flight(monkeypatch):
    monkeypatch.setattr(api, "MAX_SPEECH_INFLIGHT", 2)
    client = GroqApiClient(Hass(), api_key="key")
    initial_release, final_release = asyncio.Event(), asyncio.Event()
    started = []

    async def produce(request):
        started.append(request.text)
        await (initial_release if request.text in ("a", "b") else final_release).wait()
        return b"audio"

    with patch.object(
        client, "_async_synthesize_speech", side_effect=produce
    ) as synthesis:
        initial = [
            asyncio.create_task(
                client.async_synthesize_speech(
                    SpeechRequest(text=t, model="m", voice="v")
                )
            )
            for t in ("a", "b")
        ]
        while len(started) < 2:
            await asyncio.sleep(0)
        queued = [
            asyncio.create_task(
                client.async_synthesize_speech(
                    SpeechRequest(text="c", model="m", voice="v")
                )
            )
            for _ in range(2)
        ]
        await asyncio.sleep(0)
        initial_release.set()
        await asyncio.gather(*initial)
        while len(started) < 3:
            await asyncio.sleep(0)
        final_release.set()
        assert await asyncio.gather(*queued) == [b"audio", b"audio"]
    assert synthesis.call_count == 3


@pytest.mark.asyncio
async def test_ffmpeg_output_cap_and_stderr_drain(monkeypatch):
    monkeypatch.setattr(audio_files, "MAX_FFMPEG_OUTPUT_BYTES", 4)
    monkeypatch.setattr(audio_files, "MAX_FFMPEG_STDERR_BYTES", 3)

    def reader(data):
        stream = asyncio.StreamReader()
        stream.feed_data(data)
        stream.feed_eof()
        return stream

    writer = SimpleNamespace(
        write=lambda _: None, drain=lambda: asyncio.sleep(0), close=lambda: None
    )
    process = SimpleNamespace(
        stdin=writer,
        stdout=reader(b"abcd"),
        stderr=reader(b"abcdef"),
        wait=lambda: asyncio.sleep(0),
    )
    assert await audio_files.async_communicate_audio(process, b"input") == (
        b"abcd",
        b"abc",
    )
    process.stdout = reader(b"abcde")
    process.stderr = reader(b"")
    with pytest.raises(HomeAssistantError, match="byte limit"):
        await audio_files.async_communicate_audio(process, None)

    def broken(_):
        raise BrokenPipeError

    writer.write = broken
    await audio_files._feed_input(writer, b"input")


@pytest.mark.asyncio
async def test_ffmpeg_overflow_cancels_pending_pipe_reader(monkeypatch):
    monkeypatch.setattr(audio_files, "MAX_FFMPEG_OUTPUT_BYTES", 1)
    output = asyncio.StreamReader()
    output.feed_data(b"too much")
    output.feed_eof()
    pending_errors = asyncio.StreamReader()
    writer = SimpleNamespace(
        write=lambda _: None, drain=lambda: asyncio.sleep(0), close=lambda: None
    )
    process = SimpleNamespace(
        stdin=writer,
        stdout=output,
        stderr=pending_errors,
        wait=lambda: asyncio.sleep(0),
    )
    with pytest.raises(HomeAssistantError, match="byte limit"):
        await audio_files.async_communicate_audio(process, b"input")
    assert pending_errors._waiter is None


@pytest.mark.parametrize("format, codec", [("ogg", "libopus"), ("mulaw", "pcm_mulaw")])
def test_ffmpeg_formats_preserve_codec_contract(format, codec):
    from custom_components.groq.tts import _ffmpeg_output_args

    assert codec in _ffmpeg_output_args(format, None)


@pytest.mark.asyncio
async def test_stt_permission_error_has_model_and_account_context():
    response = Response(403, b'{"error":{"code":"model_permission_blocked_project"}}')
    client = GroqApiClient(
        Hass(), api_key="key", session=Session(response), entry_id="account"
    )
    with (
        patch.object(api, "async_create_model_access_issue") as create,
        pytest.raises(GroqApiError),
    ):
        await client.async_transcribe_audio(
            audio=b"audio",
            filename="audio.wav",
            model="whisper",
            service_id="stt-service",
            protect_free_tier=False,
        )
    create.assert_called_once_with(
        client._hass, "whisper", "stt-service", entry_id="account", reason="permissions"
    )


@pytest.mark.asyncio
async def test_stt_success_clears_repair_without_adding_json_to_multipart():
    response = Response(body=b'{"text":"transcribed"}')
    session = Session(response)
    client = GroqApiClient(Hass(), api_key="key", session=session, entry_id="account")
    with patch.object(api, "async_delete_model_access_issue") as clear:
        assert (
            await client.async_transcribe_audio(
                audio=b"audio",
                filename="audio.wav",
                model="whisper",
                service_id="stt-service",
                protect_free_tier=False,
            )
            == "transcribed"
        )
    clear.assert_called_once_with(
        client._hass, "whisper", "stt-service", entry_id="account"
    )
    assert "json" not in session.calls[0][1]
    assert "data" in session.calls[0][1]
