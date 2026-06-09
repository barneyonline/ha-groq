"""Repair issue helpers for the Groq integration."""

from __future__ import annotations

from contextlib import suppress
from hashlib import sha1
from typing import Any

from homeassistant import data_entry_flow
from homeassistant.components.repairs.models import RepairsFlow
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir

from .const import CONF_NAME, DOMAIN, UNIQUE_ID

ISSUE_FFMPEG_MISSING = "ffmpeg_missing"
ISSUE_MODEL_ACCESS = "model_access"
ISSUE_MODEL_CONFIGURATION = "model_configuration"


class GroqRepairsFlow(RepairsFlow):
    """Repairs flow for Groq issues."""

    async def async_step_init(
        self,
        user_input: dict[str, str] | None = None,
    ) -> data_entry_flow.FlowResult:
        """Abort repair flows because Groq currently creates non-fixable issues."""
        return self.async_abort(reason="not_fixable")


async def async_create_fix_flow(
    _hass: HomeAssistant,
    _issue_id: str,
    _data: dict[str, str | int | float | None] | None,
) -> RepairsFlow:
    """Create a repair flow for a Groq issue."""
    return GroqRepairsFlow()


def _safe(value: Any, fallback: str = "unknown") -> str:
    """Return a short sanitized placeholder value."""
    text = str(value or fallback)
    return text[:128]


def _hashed_issue_id(issue_type: str, *parts: Any) -> str:
    """Return a stable issue id without embedding secrets or long user data."""
    encoded = "|".join(str(part or "") for part in parts)
    return f"{issue_type}_{sha1(encoded.encode('utf-8')).hexdigest()[:12]}"


def _service_name(service_data: dict[str, Any] | None) -> str:
    """Return a user-facing service label."""
    if not service_data:
        return "Groq"
    return _safe(service_data.get(CONF_NAME) or service_data.get(UNIQUE_ID), "Groq")


def async_create_ffmpeg_missing_issue(
    hass: HomeAssistant,
    entry: ConfigEntry,
    service_data: dict[str, Any] | None = None,
) -> None:
    """Create a repair issue when ffmpeg is required but missing."""
    with suppress(Exception):
        ir.async_create_issue(
            hass,
            DOMAIN,
            _hashed_issue_id(
                ISSUE_FFMPEG_MISSING,
                entry.entry_id,
                (service_data or {}).get(UNIQUE_ID),
            ),
            is_fixable=False,
            is_persistent=True,
            issue_domain=DOMAIN,
            severity=ir.IssueSeverity.ERROR,
            translation_key=ISSUE_FFMPEG_MISSING,
            translation_placeholders={
                "service_name": _service_name(service_data),
            },
        )


def async_delete_ffmpeg_missing_issue(
    hass: HomeAssistant,
    entry: ConfigEntry,
    service_data: dict[str, Any] | None = None,
) -> None:
    """Delete the ffmpeg repair issue after audio processing succeeds."""
    with suppress(Exception):
        ir.async_delete_issue(
            hass,
            DOMAIN,
            _hashed_issue_id(
                ISSUE_FFMPEG_MISSING,
                entry.entry_id,
                (service_data or {}).get(UNIQUE_ID),
            ),
        )


def async_create_model_access_issue(
    hass: HomeAssistant,
    model: str,
    service_id: str | None = None,
) -> None:
    """Create a repair issue when Groq reports a model access problem."""
    with suppress(Exception):
        ir.async_create_issue(
            hass,
            DOMAIN,
            _hashed_issue_id(ISSUE_MODEL_ACCESS, model, service_id),
            is_fixable=False,
            is_persistent=True,
            issue_domain=DOMAIN,
            severity=ir.IssueSeverity.ERROR,
            translation_key=ISSUE_MODEL_ACCESS,
            translation_placeholders={
                "model": _safe(model),
            },
        )


def async_create_model_configuration_issue(
    hass: HomeAssistant,
    entry: ConfigEntry,
    service_data: dict[str, Any],
    model: str,
    feature: str,
) -> None:
    """Create a repair issue for a configured model that cannot serve a feature."""
    with suppress(Exception):
        ir.async_create_issue(
            hass,
            DOMAIN,
            _hashed_issue_id(
                ISSUE_MODEL_CONFIGURATION,
                entry.entry_id,
                service_data.get(UNIQUE_ID),
                model,
                feature,
            ),
            is_fixable=False,
            is_persistent=True,
            issue_domain=DOMAIN,
            severity=ir.IssueSeverity.ERROR,
            translation_key=ISSUE_MODEL_CONFIGURATION,
            translation_placeholders={
                "service_name": _service_name(service_data),
                "model": _safe(model),
                "feature": _safe(feature),
            },
        )
