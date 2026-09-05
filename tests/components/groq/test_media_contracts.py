"""Validate audio container contracts at the STT and TTS boundaries."""

import asyncio
import io
import json
import math
import struct
import wave
from unittest.mock import AsyncMock, Mock

import pytest
from homeassistant.components import stt

from custom_components.groq.stt import GroqSTTEntity
from custom_components.groq.tts import GroqTTSEntity

from .test_foundation import DummyEntry

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "rate,channels,bits",
    [(8000, 1, 8), (16000, 1, 16), (22000, 2, 24), (48000, 2, 32)],
)
async def test_stt_wav_preserves_stream_frames_and_metadata(rate, channels, bits):
    frames = bytes(range(24)) * 4
    client = Mock(async_transcribe_audio=AsyncMock(return_value="Recognized"))
    entity = GroqSTTEntity(DummyEntry(), {"model": "whisper-large-v3"}, client)

    async def stream():
        # Transport chunks need not align with PCM frames.
        yield frames[:5]
        yield frames[5:37]
        yield frames[37:]

    result = await entity.async_process_audio_stream(
        stt.SpeechMetadata(
            language="en-US",
            format=stt.AudioFormats.WAV,
            codec=stt.AudioCodecs.PCM,
            bit_rate=stt.AudioBitRates(bits),
            sample_rate=stt.AudioSampleRates(rate),
            channel=stt.AudioChannels(channels),
        ),
        stream(),
    )
    assert result.result is stt.SpeechResultState.SUCCESS
    uploaded = client.async_transcribe_audio.call_args.kwargs
    assert uploaded["filename"] == "audio.wav"
    assert uploaded["language"] == "en-US"
    with wave.open(io.BytesIO(uploaded["audio"]), "rb") as audio:
        assert audio.getframerate() == rate
        assert audio.getnchannels() == channels
        assert audio.getsampwidth() == bits // 8
        assert audio.getnframes() == len(frames) // (channels * bits // 8)
        assert audio.readframes(audio.getnframes()) == frames


async def test_stt_ogg_passes_container_through_unchanged():
    container = b"OggS\x00\x02opaque-opus-container"
    client = Mock(async_transcribe_audio=AsyncMock(return_value="Recognized"))
    entity = GroqSTTEntity(DummyEntry(), {"language": "fr"}, client)

    async def stream():
        yield container[:3]
        yield container[3:]

    await entity.async_process_audio_stream(
        stt.SpeechMetadata(
            language="en-US",
            format=stt.AudioFormats.OGG,
            codec=stt.AudioCodecs.OPUS,
            bit_rate=stt.AudioBitRates.BITRATE_16,
            sample_rate=stt.AudioSampleRates.SAMPLERATE_48000,
            channel=stt.AudioChannels.CHANNEL_STEREO,
        ),
        stream(),
    )
    uploaded = client.async_transcribe_audio.call_args.kwargs
    assert uploaded["audio"] == container
    assert uploaded["filename"] == "audio.ogg"
    assert uploaded["language"] == "fr"


def tone_wav():
    """Generate one second of stereo PCM without a network/media dependency."""
    samples = (
        int(4000 * math.sin(2 * math.pi * 440 * index / 24000))
        for index in range(24000)
    )
    frames = b"".join(struct.pack("<hh", sample, sample) for sample in samples)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(frames)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "output_format,codec,rate,long_text,normalize",
    [
        ("wav", "pcm_s16le", 16000, False, True),
        ("mp3", "mp3", 44100, False, True),
        ("flac", "flac", 24000, False, True),
        ("ogg", "opus", 48000, False, True),
        ("mulaw", "pcm_mulaw", 8000, False, True),
        ("wav", "pcm_s16le", 16000, True, True),
        ("wav", "pcm_s16le", 16000, True, False),
    ],
)
async def test_real_ffmpeg_conversion_and_stitching(
    hass, tmp_path, output_format, codec, rate, long_text, normalize
):
    """Exercise the bundled ffmpeg and inspect its output with ffprobe."""
    client = Mock(
        async_synthesize_speech=AsyncMock(return_value=tone_wav()),
        check_tts_batch=Mock(),
    )
    entity = GroqTTSEntity(
        hass,
        DummyEntry(),
        client,
        service_data={
            "unique_id": "audio-test",
            "model": "canopylabs/orpheus-v1-english",
            "voice": "troy",
            "enable_long_tts": long_text,
            "protect_free_tier": False,
        },
    )
    message = "Hello world. " * 30 if long_text else "Hello world."
    extension, payload = await entity.async_get_tts_audio(
        message,
        "en",
        options={
            "normalize_audio": normalize,
            "response_format": output_format,
            "sample_rate": rate,
        },
    )
    assert extension == output_format
    assert payload
    chunks = client.async_synthesize_speech.await_count
    assert chunks > 1 if long_text else chunks == 1
    if long_text:
        client.check_tts_batch.assert_called_once()
    output = tmp_path / f"speech.{extension}"
    await hass.async_add_executor_job(output.write_bytes, payload)
    input_options = (
        ["-f", "mulaw", "-ar", str(rate)] if output_format == "mulaw" else []
    )
    process = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_frames",
        "-of",
        "json",
        *input_options,
        str(output),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 0, stderr.decode()
    probe = json.loads(stdout)
    audio = probe["streams"][0]
    assert audio["codec_name"] == codec
    assert int(audio["sample_rate"]) == rate
    assert audio["channels"] == 1
    # Piped FLAC has no duration header; decoded samples work for every codec.
    duration = sum(frame["nb_samples"] for frame in probe["frames"]) / rate
    assert duration == pytest.approx(chunks, abs=0.1)
