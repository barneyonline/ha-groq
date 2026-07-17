"""Diagnostics support for Groq.

Provides config entry and device diagnostics with sensitive data redacted.
"""

from __future__ import annotations

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceEntry

from .const import (
    CONF_API_KEY,
    CONF_CACHE_SIZE,
    CONF_ENABLE_LONG_TTS,
    CONF_INCLUDE_REASONING,
    CONF_LANGUAGE,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_NAME,
    CONF_NORMALIZE_AUDIO,
    CONF_PROMPT,
    CONF_PROMPT_CACHING,
    CONF_PROTECT_FREE_TIER,
    CONF_REASONING_EFFORT,
    CONF_REASONING_FORMAT,
    CONF_RESPONSE_FORMAT,
    CONF_SAMPLE_RATE,
    CONF_SERVICE_TIER,
    CONF_SERVICE_TYPE,
    CONF_SPEED,
    CONF_STREAM,
    CONF_STRICT,
    CONF_STRUCTURED_OUTPUTS,
    CONF_SUBENTRY_ID,
    CONF_SYSTEM_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_URL,
    CONF_VOCAL_DIRECTIONS,
    CONF_VOICE,
    DEFAULT_CACHE_SIZE,
    DEFAULT_RESPONSE_FORMAT,
    DEFAULT_TTS_URL,
    FEATURE_TEXT_TO_SPEECH,
    SUPPORTED_FEATURES,
    enabled_features_from_entry,
)
from .types import GroqConfigEntry

TO_REDACT = {CONF_API_KEY, CONF_PROMPT, CONF_SYSTEM_PROMPT}

SERVICE_SUMMARY_KEYS = (
    CONF_NAME,
    CONF_MODEL,
    CONF_LANGUAGE,
    CONF_VOICE,
    CONF_RESPONSE_FORMAT,
    CONF_SAMPLE_RATE,
    CONF_SPEED,
    CONF_NORMALIZE_AUDIO,
    CONF_ENABLE_LONG_TTS,
    CONF_CACHE_SIZE,
    CONF_PROTECT_FREE_TIER,
    CONF_TEMPERATURE,
    CONF_MAX_TOKENS,
    CONF_TOP_P,
    CONF_STREAM,
    CONF_SERVICE_TIER,
    CONF_REASONING_EFFORT,
    CONF_REASONING_FORMAT,
    CONF_INCLUDE_REASONING,
    CONF_PROMPT_CACHING,
    CONF_STRUCTURED_OUTPUTS,
    CONF_STRICT,
)


def _entry_value(entry: GroqConfigEntry, key: str, default: Any = None) -> Any:
    """Return effective value, allowing options to override setup data."""
    return entry.options.get(key, entry.data.get(key, default))


def _default_summary(entry: GroqConfigEntry) -> dict[str, Any]:
    """Return legacy account-level defaults without exposing generated content."""
    return {
        "text_to_speech": {
            "endpoint": _entry_value(entry, CONF_URL, DEFAULT_TTS_URL),
            "model": _entry_value(entry, CONF_MODEL),
            "voice": _entry_value(entry, CONF_VOICE),
            "response_format": _entry_value(
                entry, CONF_RESPONSE_FORMAT, DEFAULT_RESPONSE_FORMAT
            ),
            "sample_rate": _entry_value(entry, CONF_SAMPLE_RATE),
            "speed": _entry_value(entry, CONF_SPEED),
            "vocal_directions_configured": bool(
                _entry_value(entry, CONF_VOCAL_DIRECTIONS, "")
            ),
            "normalize_audio": _entry_value(entry, CONF_NORMALIZE_AUDIO, False),
            "enable_long_tts": _entry_value(entry, CONF_ENABLE_LONG_TTS, False),
            "cache_size": _entry_value(entry, CONF_CACHE_SIZE, DEFAULT_CACHE_SIZE),
        }
    }


def _subentry_services_summary(
    entry: GroqConfigEntry,
) -> dict[str, list[dict[str, Any]]]:
    """Return configured service subentries grouped by service type."""
    services: dict[str, list[dict[str, Any]]] = {}
    for subentry in (getattr(entry, "subentries", None) or {}).values():
        data = async_redact_data(dict(getattr(subentry, "data", {})), TO_REDACT)
        service_type = data.get(CONF_SERVICE_TYPE, "unknown")
        service = {
            CONF_SUBENTRY_ID: getattr(subentry, "subentry_id", None),
            "title": getattr(subentry, "title", None),
        }
        for key in SERVICE_SUMMARY_KEYS:
            if key in data:
                service[key] = data[key]
        services.setdefault(service_type, []).append(service)
    return services


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: GroqConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry with secrets redacted."""
    redacted_data = async_redact_data(dict(entry.data), TO_REDACT)
    redacted_options = async_redact_data(dict(entry.options), TO_REDACT)
    enabled_features = enabled_features_from_entry(entry)
    services = _subentry_services_summary(entry)
    service_counts = {
        service_type: len(service_entries)
        for service_type, service_entries in services.items()
    }

    summary = {
        "enabled_features": enabled_features,
        "available_features": list(SUPPORTED_FEATURES),
        "text_to_speech_enabled": FEATURE_TEXT_TO_SPEECH in enabled_features,
        "service_counts": service_counts,
        "total_services": sum(service_counts.values()),
        "services": services,
    }
    if not services:
        summary["legacy_defaults"] = _default_summary(entry)

    return {
        "entry_data": redacted_data,
        "options": redacted_options,
        "summary": summary,
    }


async def async_get_device_diagnostics(
    hass: HomeAssistant, entry: GroqConfigEntry, device: DeviceEntry
) -> dict[str, Any]:
    """Return diagnostics for a device.

    This integration exposes a single device per config entry; include the same redacted
    data with the device identifiers.
    """
    data = await async_get_config_entry_diagnostics(hass, entry)
    data["device"] = {
        "identifiers": list(device.identifiers),
        "name": device.name,
        "manufacturer": device.manufacturer,
        "model": device.model,
    }
    return data
