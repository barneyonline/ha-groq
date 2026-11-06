"""Speech-to-Text vocabulary and context prompt contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.components import stt
from homeassistant.core import SupportsResponse
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.groq import services
from custom_components.groq.const import MAX_STT_PROMPT_LENGTH
from custom_components.groq.diagnostics import async_get_config_entry_diagnostics
from custom_components.groq.feature_registry import GroqFeature, GroqFeatureRegistry
from custom_components.groq.flow_schemas import (
    clean_service_input,
    speech_to_text_schema,
)
from custom_components.groq.model_registry import GroqModelRegistry
from custom_components.groq.stt import GroqSTTEntity

from .test_foundation import DummyEntry

pytestmark = pytest.mark.usefixtures("enable_custom_integrations")

MODEL = "whisper-large-v3-turbo"
PROMPT = "The conversation is about Home Assistant, Zigbee, and Philips Hue."
COMPONENT = Path(__file__).parents[3] / "custom_components/groq"


def account(hass, *, data=None):
    """Register a real entry with one Speech-to-Text service."""
    entry = MockConfigEntry(
        domain="groq",
        entry_id="account",
        unique_id="account",
        minor_version=2,
        data={"api_key": "test-key", "name": "Test Groq"},
        subentries_data=[
            {
                "subentry_id": "speech",
                "subentry_type": "speech_to_text",
                "title": "Speech",
                "unique_id": None,
                "data": {
                    "service_type": "speech_to_text",
                    "name": "Speech",
                    "model": MODEL,
                    "language": "en-US",
                    **(data or {}),
                },
            }
        ],
    )
    entry.add_to_hass(hass)
    return entry


async def submit(hass, entry, user_input, *, source="reconfigure"):
    """Run a real subentry flow step and return the first and last results."""
    assert await async_setup_component(hass, "homeassistant", {})
    context = {"source": source}
    if source == "reconfigure":
        context["subentry_id"] = "speech"
    with patch(
        "custom_components.groq.config_flow.async_get_model_registry",
        return_value=GroqModelRegistry(),
    ):
        form = await hass.config_entries.subentries.async_init(
            (entry.entry_id, "speech_to_text"), context=context
        )
        result = await hass.config_entries.subentries.async_configure(
            form["flow_id"], {"name": "Speech", "model": MODEL, **user_input}
        )
    return form, result


def prompt_marker(schema):
    """Return the schema key and selector config of the prompt field."""
    for key, value in schema.schema.items():
        if key == "prompt":
            return key, value.config
    raise AssertionError("prompt field missing")


def test_schema_offers_an_optional_multiline_field():
    key, config = prompt_marker(speech_to_text_schema())

    assert config["multiline"] is True
    assert key.description == {"suggested_value": None}
    assert "prompt" not in speech_to_text_schema()({})


def test_schema_suggests_but_does_not_default_the_saved_prompt():
    schema = speech_to_text_schema({"prompt": PROMPT})

    assert prompt_marker(schema)[0].description == {"suggested_value": PROMPT}
    # An omitted field must stay omitted, otherwise it could never be cleared.
    assert "prompt" not in schema({})


@pytest.mark.parametrize(
    "value, expected",
    [
        (f"  {PROMPT}\n", {"prompt": PROMPT}),
        ("Zigbee\nPhilips Hue", {"prompt": "Zigbee\nPhilips Hue"}),
        (" \n ", {}),
        ("", {}),
    ],
)
def test_clean_service_input_normalizes_the_prompt(value, expected):
    assert clean_service_input({"prompt": value}) == expected


async def test_flow_stores_the_prompt(hass):
    entry = account(hass)

    form, result = await submit(hass, entry, {"prompt": f" {PROMPT} "})

    assert form["type"] == "form"
    assert result["reason"] == "reconfigure_successful"
    assert entry.subentries["speech"].data["prompt"] == PROMPT


async def test_flow_clears_the_saved_prompt(hass):
    entry = account(hass, data={"prompt": PROMPT})

    form, result = await submit(hass, entry, {})

    assert prompt_marker(form["data_schema"])[0].description == {
        "suggested_value": PROMPT
    }
    assert result["reason"] == "reconfigure_successful"
    assert "prompt" not in entry.subentries["speech"].data


async def test_flow_accepts_a_prompt_at_the_limit(hass):
    entry = account(hass)
    prompt = "x" * MAX_STT_PROMPT_LENGTH

    _form, result = await submit(hass, entry, {"prompt": prompt})

    assert result["reason"] == "reconfigure_successful"
    assert entry.subentries["speech"].data["prompt"] == prompt


async def test_flow_rejects_a_long_prompt_and_keeps_the_text(hass):
    entry = account(hass, data={"prompt": PROMPT})
    prompt = "x" * (MAX_STT_PROMPT_LENGTH + 1)

    _form, result = await submit(hass, entry, {"prompt": prompt})

    assert result["type"] == "form"
    assert result["errors"] == {"prompt": "prompt_too_long"}
    assert result["description_placeholders"]["max_length"] == str(
        MAX_STT_PROMPT_LENGTH
    )
    assert prompt_marker(result["data_schema"])[0].description == {
        "suggested_value": prompt
    }
    assert entry.subentries["speech"].data["prompt"] == PROMPT


async def test_new_service_starts_without_a_prompt(hass):
    entry = account(hass)

    form, result = await submit(hass, entry, {}, source="user")

    assert prompt_marker(form["data_schema"])[0].description == {
        "suggested_value": None
    }
    assert result["type"] == "create_entry"
    assert "prompt" not in result["data"]


@pytest.mark.parametrize(
    "service_data, expected",
    [({"prompt": PROMPT}, PROMPT), ({}, None)],
)
async def test_assist_pipeline_sends_the_saved_prompt(service_data, expected):
    client = Mock(async_transcribe_audio=AsyncMock(return_value="Recognized"))
    entity = GroqSTTEntity(DummyEntry(), {"model": MODEL, **service_data}, client)

    async def stream():
        yield b"OggS"

    result = await entity.async_process_audio_stream(
        stt.SpeechMetadata(
            language="en-US",
            format=stt.AudioFormats.OGG,
            codec=stt.AudioCodecs.OPUS,
            bit_rate=stt.AudioBitRates.BITRATE_16,
            sample_rate=stt.AudioSampleRates.SAMPLERATE_48000,
            channel=stt.AudioChannels.CHANNEL_MONO,
        ),
        stream(),
    )

    assert result.result is stt.SpeechResultState.SUCCESS
    assert client.async_transcribe_audio.call_args.kwargs["prompt"] == expected


@pytest.mark.parametrize(
    "translate, extra, expected",
    [
        (False, {}, PROMPT),
        (False, {"prompt": "Matter"}, "Matter"),
        (False, {"prompt": ""}, ""),
        # The saved prompt matches the audio language; translation expects English.
        (True, {}, None),
        (True, {"prompt": "Matter"}, "Matter"),
    ],
)
async def test_audio_actions_resolve_the_prompt(
    hass, monkeypatch, translate, extra, expected
):
    client = Mock(
        async_transcribe_audio=AsyncMock(return_value="Hello"),
        async_transcribe_audio_result=AsyncMock(return_value={"text": "Hello"}),
    )
    data = {
        "unique_id": "speech",
        "model": MODEL,
        "language": "fr",
        "prompt": PROMPT,
        "protect_free_tier": False,
    }
    runtime = SimpleNamespace(
        client=client,
        model_registry=GroqModelRegistry(),
        feature_registry=GroqFeatureRegistry([GroqFeature.SPEECH_TO_TEXT]),
        services_by_type={"speech_to_text": (data,)},
    )
    monkeypatch.setattr(
        services, "_runtime_from_call", AsyncMock(return_value=(DummyEntry(), runtime))
    )
    monkeypatch.setattr(
        services, "_audio_from_call", AsyncMock(return_value=(b"audio", "a.wav"))
    )
    action = "translate_audio" if translate else "transcribe_audio"
    hass.services.async_register(
        "groq",
        action,
        services._handle_transcribe_audio(hass, translate=translate),
        schema=(
            services.TRANSLATE_AUDIO_SCHEMA
            if translate
            else services.TRANSCRIBE_AUDIO_SCHEMA
        ),
        supports_response=SupportsResponse.ONLY,
    )

    await hass.services.async_call(
        "groq",
        action,
        {"service_id": "speech", **extra},
        blocking=True,
        return_response=True,
    )

    upload = (
        client.async_transcribe_audio_result
        if translate
        else client.async_transcribe_audio
    )
    assert upload.call_args.kwargs["prompt"] == expected


async def test_diagnostics_do_not_expose_the_prompt(hass):
    entry = account(hass, data={"prompt": PROMPT})

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)

    assert diagnostics["summary"]["services"]["speech_to_text"]
    assert PROMPT not in json.dumps(diagnostics)


@pytest.mark.parametrize(
    "path",
    [COMPONENT / "strings.json", *sorted((COMPONENT / "translations").glob("*.json"))],
    ids=lambda path: path.stem,
)
def test_prompt_strings_are_user_facing(path):
    service = json.loads(path.read_text(encoding="utf-8"))["config_subentries"][
        "speech_to_text"
    ]
    step = service["step"]["speech_to_text"]

    assert step["data"]["prompt"] != "prompt"
    assert step["data_description"]["prompt"]
    assert "{max_length}" in service["error"]["prompt_too_long"]
