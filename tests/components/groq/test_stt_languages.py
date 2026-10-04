"""Regression coverage for Romanian STT configuration and Assist requests."""

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


def test_romanian_selector_and_action_parity():
    """Both selectors offer Romanian and accept its locale."""
    assert {"value": "ro-RO", "label": "Romanian"} in STT_LANGUAGE_OPTIONS
    assert speech_to_text_schema()({"language": "ro-RO"})["language"] == "ro-RO"
    services = yaml.safe_load(
        Path("custom_components/groq/services.yaml").read_text(encoding="utf-8")
    )
    selector = services["transcribe_audio"]["fields"]["language"]["selector"]["select"]
    assert selector["options"] == STT_LANGUAGE_OPTIONS
    assert selector["custom_value"] is True


@pytest.mark.parametrize("locale", ["ro", "ro-RO", "ro_RO"])
def test_romanian_locale_defaults(locale):
    """HA Romanian locales select Romanian instead of the English fallback."""
    default = stt_language_default(locale)
    assert default == "ro-RO"
    assert speech_to_text_schema(default_language=default)({})["language"] == "ro-RO"
    assert (
        speech_to_text_schema({"language": "ro-RO"}, default_language="en-US")({})[
            "language"
        ]
        == "ro-RO"
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


@pytest.mark.parametrize("model", ["whisper-large-v3", "whisper-large-v3-turbo"])
@pytest.mark.parametrize("configured_language", [None, "ro-RO"])
async def test_romanian_assist_language_and_request(model, configured_language):
    """Assist accepts Romanian and the real client sends the ISO base hint."""
    session = Session(Response(body=b'{"text":"Salut"}'))
    client = GroqApiClient(DummyHass(), api_key="fake", session=session)
    service = {"model": model}
    if configured_language:
        service["language"] = configured_language
    entity = GroqSTTEntity(DummyEntry(), service, client)
    assert "ro-RO" in entity.supported_languages
    metadata = stt.SpeechMetadata(
        language="ro-RO",
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
    assert result.text == "Salut"
    fields = session.calls[0][1]["data"]._fields
    assert [
        (header["name"], value)
        for header, _, value in fields
        if header["name"] == "language"
    ] == [("language", "ro")]
