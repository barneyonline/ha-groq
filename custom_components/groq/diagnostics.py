"""Diagnostics support for Groq.

Provides config entry and device diagnostics with sensitive data redacted.
"""

from __future__ import annotations

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceEntry

from .const import (
    CONF_API_KEY,
    CONF_PROMPT,
    CONF_SYSTEM_PROMPT,
    CONF_MODEL,
    CONF_NORMALIZE_AUDIO,
    CONF_RESPONSE_FORMAT,
    CONF_URL,
    CONF_VOICE,
    CONF_VOCAL_DIRECTIONS,
    CONF_CACHE_SIZE,
    CONF_PROTECT_FREE_TIER,
    DEFAULT_CACHE_SIZE,
    DEFAULT_PROTECT_FREE_TIER,
    DEFAULT_RESPONSE_FORMAT,
    DEFAULT_TTS_URL,
    FEATURE_TEXT_TO_SPEECH,
    SUPPORTED_FEATURES,
    enabled_features_from_entry,
)

TO_REDACT = {CONF_API_KEY, CONF_PROMPT, CONF_SYSTEM_PROMPT}


def _entry_value(entry: ConfigEntry, key: str, default: Any = None) -> Any:
    """Return effective value, allowing options to override setup data."""
    return entry.options.get(key, entry.data.get(key, default))


def _default_summary(entry: ConfigEntry) -> dict[str, Any]:
    """Return configured feature defaults without exposing generated content."""
    return {
        "text_to_speech": {
            "endpoint": _entry_value(entry, CONF_URL, DEFAULT_TTS_URL),
            "model": _entry_value(entry, CONF_MODEL),
            "voice": _entry_value(entry, CONF_VOICE),
            "response_format": _entry_value(
                entry, CONF_RESPONSE_FORMAT, DEFAULT_RESPONSE_FORMAT
            ),
            "vocal_directions_configured": bool(
                _entry_value(entry, CONF_VOCAL_DIRECTIONS, "")
            ),
            "normalize_audio": _entry_value(entry, CONF_NORMALIZE_AUDIO, False),
            "cache_size": _entry_value(entry, CONF_CACHE_SIZE, DEFAULT_CACHE_SIZE),
            "protect_free_tier": _entry_value(
                entry, CONF_PROTECT_FREE_TIER, DEFAULT_PROTECT_FREE_TIER
            ),
        }
    }


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry with secrets redacted."""
    redacted_data = async_redact_data(dict(entry.data), TO_REDACT)
    redacted_options = async_redact_data(dict(entry.options), TO_REDACT)
    enabled_features = enabled_features_from_entry(entry)
    defaults = _default_summary(entry)

    summary = {
        "enabled_features": enabled_features,
        "available_features": list(SUPPORTED_FEATURES),
        "text_to_speech_enabled": FEATURE_TEXT_TO_SPEECH in enabled_features,
        "defaults": defaults,
    }

    return {
        "entry_data": redacted_data,
        "options": redacted_options,
        "summary": summary,
    }


async def async_get_device_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry, device: DeviceEntry
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
