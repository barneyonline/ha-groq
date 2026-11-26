"""Regression coverage for STT language configuration and Assist requests."""

from pathlib import Path

import pytest
import yaml
from homeassistant.components import stt

from custom_components.groq.api import GroqApiClient
from custom_components.groq.const import STT_LANGUAGE_OPTIONS, stt_language_default
from custom_components.groq.flow_schemas import speech_to_text_schema
from custom_components.groq.stt import GroqSTTEntity

from .test_foundation import DummyEntry, DummyHass
from .test_transport_architecture import Response, Session


@pytest.mark.parametrize(
    ("locale", "label"), [("ro-RO", "Romanian"), ("hu-HU", "Hungarian")]
)
def test_stt_selector_and_action_parity(locale: str, label: str) -> None:
    """The setup form and action offer the same language options."""
    assert {"value": locale, "label": label} in STT_LANGUAGE_OPTIONS
    assert speech_to_text_schema()({"language": locale})["language"] == locale
    services = yaml.safe_load(
        Path("custom_components/groq/services.yaml").read_text(encoding="utf-8")
    )
    selector = services["transcribe_audio"]["fields"]["language"]["selector"]["select"]
    assert selector["options"] == STT_LANGUAGE_OPTIONS
    assert selector["custom_value"] is True


@pytest.mark.parametrize(
    ("locale", "expected"),
    [
        ("ro", "ro-RO"),
        ("ro-RO", "ro-RO"),
        ("ro_RO", "ro-RO"),
        ("hu", "hu-HU"),
        ("hu-HU", "hu-HU"),
        ("hu_HU", "hu-HU"),
    ],
)
def test_stt_locale_defaults(locale: str, expected: str) -> None:
    """Use the closest supported locale for new and reconfigured services."""
    default = stt_language_default(locale)
    assert default == expected
    assert speech_to_text_schema(default_language=default)({})["language"] == expected
    assert (
        speech_to_text_schema({"language": expected}, default_language="en-US")({})[
            "language"
        ]
        == expected
    )


def test_existing_language_defaults_are_preserved():
    assert speech_to_text_schema()({})["language"] == "en-US"
    assert stt_language_default("xx-YY") == "en-US"
    assert (
        speech_to_text_schema({"language": "fr-FR"}, default_language="ro-RO")({})[
            "language"
        ]
        == "fr-FR"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["whisper-large-v3", "whisper-large-v3-turbo"])
@pytest.mark.parametrize("use_configured_language", [False, True])
@pytest.mark.parametrize(("locale", "hint"), [("ro-RO", "ro"), ("hu-HU", "hu")])
async def test_assist_language_and_request(
    model: str, use_configured_language: bool, locale: str, hint: str
) -> None:
    """Assist accepts the locale and the client sends the ISO language hint."""
    session = Session(Response(body=b'{"text":"Transcribed"}'))
    client = GroqApiClient(DummyHass(), api_key="fake", session=session)
    service = {"model": model}
    if use_configured_language:
        service["language"] = locale
    entity = GroqSTTEntity(DummyEntry(), service, client)
    assert locale in entity.supported_languages
    metadata = stt.SpeechMetadata(
        language=locale,
        format=stt.AudioFormats.WAV,
        codec=stt.AudioCodecs.PCM,
        bit_rate=stt.AudioBitRates.BITRATE_16,
        sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
    )
    assert entity.check_metadata(metadata)

    async def stream():
        yield b"\x00\x00" * 160

    result = await entity.async_process_audio_stream(metadata, stream())
    assert result.result is stt.SpeechResultState.SUCCESS
    assert result.text == "Transcribed"
    fields = session.calls[0][1]["data"]._fields
    assert [
        (header["name"], value)
        for header, _, value in fields
        if header["name"] == "language"
    ] == [("language", hint)]
