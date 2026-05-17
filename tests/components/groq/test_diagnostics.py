from __future__ import annotations

from types import SimpleNamespace

import pytest

from custom_components.groq.diagnostics import (
    async_get_config_entry_diagnostics,
    async_get_device_diagnostics,
)


class DummyEntry:
    data = {
        "api_key": "secret-key",
        "url": "https://api.groq.com/openai/v1/audio/speech",
        "model": "canopylabs/orpheus-v1-english",
        "voice": "autumn",
        "response_format": "wav",
        "vocal_directions": "friendly",
    }
    options = {
        "api_key": "option-secret",
        "enabled_features": ["text_to_speech", "image_recognition"],
        "voice": "troy",
        "normalize_audio": True,
        "enable_long_tts": True,
        "cache_size": 64,
        "protect_free_tier": False,
    }


@pytest.mark.asyncio
async def test_config_entry_diagnostics_redacts_api_keys() -> None:
    diagnostics = await async_get_config_entry_diagnostics(None, DummyEntry())

    assert diagnostics["entry_data"]["api_key"] == "**REDACTED**"
    assert diagnostics["options"]["api_key"] == "**REDACTED**"
    assert diagnostics["summary"] == {
        "enabled_features": ["text_to_speech", "image_recognition"],
        "available_features": [
            "text_generation",
            "speech_to_text",
            "text_to_speech",
            "image_recognition",
        ],
        "text_to_speech_enabled": True,
        "service_counts": {},
        "total_services": 0,
        "services": {},
        "legacy_defaults": {
            "text_to_speech": {
                "endpoint": "https://api.groq.com/openai/v1/audio/speech",
                "model": "canopylabs/orpheus-v1-english",
                "voice": "troy",
                "response_format": "wav",
                "vocal_directions_configured": True,
                "normalize_audio": True,
                "enable_long_tts": True,
                "cache_size": 64,
            }
        },
    }


@pytest.mark.asyncio
async def test_config_entry_diagnostics_defaults_legacy_entry_to_tts() -> None:
    entry = SimpleNamespace(
        data={
            "api_key": "secret-key",
            "url": "https://api.groq.com/openai/v1/audio/speech",
            "model": "canopylabs/orpheus-v1-english",
            "voice": "autumn",
        },
        options={},
    )

    diagnostics = await async_get_config_entry_diagnostics(None, entry)

    assert diagnostics["summary"]["enabled_features"] == ["text_to_speech"]
    assert diagnostics["summary"]["text_to_speech_enabled"] is True
    assert (
        diagnostics["summary"]["legacy_defaults"]["text_to_speech"]["response_format"]
        == "wav"
    )


@pytest.mark.asyncio
async def test_config_entry_diagnostics_account_only_entry_has_no_services() -> None:
    entry = SimpleNamespace(data={"api_key": "secret-key", "name": "Groq"}, options={})

    diagnostics = await async_get_config_entry_diagnostics(None, entry)

    assert diagnostics["summary"]["enabled_features"] == []
    assert diagnostics["summary"]["text_to_speech_enabled"] is False
    assert diagnostics["summary"]["total_services"] == 0


@pytest.mark.asyncio
async def test_config_entry_diagnostics_summarizes_service_subentries() -> None:
    entry = SimpleNamespace(
        data={"api_key": "secret-key", "name": "Groq"},
        options={},
        subentries={
            "text": SimpleNamespace(
                subentry_id="text-id",
                title="Assistant",
                data={
                    "name": "Assistant",
                    "service_type": "text_generation",
                    "model": "openai/gpt-oss-20b",
                    "system_prompt": "secret instructions",
                    "temperature": 0.2,
                    "stream": True,
                    "structured_outputs": True,
                },
            ),
            "tts": SimpleNamespace(
                subentry_id="tts-id",
                title="Voice",
                data={
                    "name": "Voice",
                    "service_type": "text_to_speech",
                    "model": "canopylabs/orpheus-v1-english",
                    "voice": "autumn",
                    "protect_free_tier": True,
                },
            ),
        },
    )

    diagnostics = await async_get_config_entry_diagnostics(None, entry)

    assert diagnostics["summary"]["service_counts"] == {
        "text_generation": 1,
        "text_to_speech": 1,
    }
    assert diagnostics["summary"]["total_services"] == 2
    assert "legacy_defaults" not in diagnostics["summary"]
    text_service = diagnostics["summary"]["services"]["text_generation"][0]
    assert text_service == {
        "subentry_id": "text-id",
        "title": "Assistant",
        "name": "Assistant",
        "model": "openai/gpt-oss-20b",
        "temperature": 0.2,
        "stream": True,
        "structured_outputs": True,
    }
    assert "system_prompt" not in text_service


@pytest.mark.asyncio
async def test_device_diagnostics_adds_device_context() -> None:
    device = SimpleNamespace(
        identifiers={("groq", "abc")},
        name="Groq",
        manufacturer="Groq",
        model="canopylabs/orpheus-v1-english",
    )

    diagnostics = await async_get_device_diagnostics(None, DummyEntry(), device)

    assert diagnostics["entry_data"]["api_key"] == "**REDACTED**"
    assert diagnostics["device"]["identifiers"] == [("groq", "abc")]
    assert diagnostics["device"]["manufacturer"] == "Groq"
