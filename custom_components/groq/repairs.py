"""Owned, recoverable repair issues for Groq configuration and model access."""

from __future__ import annotations

import logging
from collections.abc import Callable
from hashlib import sha1
from typing import Any

from homeassistant import data_entry_flow
from homeassistant.components.repairs.models import RepairsFlow
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir

from .const import CONF_MODEL, CONF_NAME, DOMAIN, UNIQUE_ID, entry_value
from .feature_registry import GroqFeature
from .model_registry import GroqModelRegistry
from .subentries import service_data_by_type
from .types import GroqConfigEntry

ISSUE_FFMPEG_MISSING = "ffmpeg_missing"
ISSUE_MODEL_ACCESS = "model_access"
ISSUE_MODEL_CONFIGURATION = "model_configuration"
_ISSUE_TYPES = (ISSUE_FFMPEG_MISSING, ISSUE_MODEL_ACCESS, ISSUE_MODEL_CONFIGURATION)
_LOGGER = logging.getLogger(__name__)


class GroqRepairsFlow(RepairsFlow):
    """Repairs flow for non-fixable Groq issues."""

    async def async_step_init(
        self, user_input: dict[str, str] | None = None
    ) -> data_entry_flow.FlowResult:
        """Direct users to the resolution described by the repair issue."""
        return self.async_abort(reason="not_fixable")


async def async_create_fix_flow(
    _hass: HomeAssistant,
    _issue_id: str,
    _data: dict[str, str | int | float | None] | None,
) -> RepairsFlow:
    """Create a repair flow for a Groq issue."""
    return GroqRepairsFlow()


def _safe(value: Any, fallback: str = "unknown") -> str:
    """Return a bounded placeholder value."""
    return str(value or fallback)[:128]


def _hashed_issue_id(issue_type: str, *parts: Any) -> str:
    """Return a stable id without embedding user data in issue identifiers."""
    encoded = "|".join(str(part or "") for part in parts)
    return f"{issue_type}_{sha1(encoded.encode('utf-8')).hexdigest()[:12]}"


def _service_name(service_data: dict[str, Any] | None) -> str:
    """Return a bounded user-facing service label."""
    if not service_data:
        return "Groq"
    return _safe(service_data.get(CONF_NAME) or service_data.get(UNIQUE_ID), "Groq")


def _update_issue(operation: Callable[[], None]) -> None:
    """Keep optional repair reporting from breaking requests, with visible failures."""
    try:
        operation()
    except Exception:  # Repairs must not replace the original request outcome.
        _LOGGER.exception("Could not update the Groq repair registry")


def _create_issue(
    hass: HomeAssistant,
    issue_type: str,
    issue_id: str,
    placeholders: dict[str, str],
    *,
    entry_id: str | None,
    service_id: str | None = None,
    model: str | None = None,
    feature: str | None = None,
    reason: str | None = None,
) -> None:
    """Create an issue with nonsecret ownership metadata for later reconciliation."""
    _update_issue(
        lambda: ir.async_create_issue(
            hass,
            DOMAIN,
            issue_id,
            is_fixable=False,
            is_persistent=True,
            issue_domain=DOMAIN,
            severity=ir.IssueSeverity.ERROR,
            translation_key=issue_type,
            translation_placeholders=placeholders,
            data={
                "entry_id": entry_id,
                "service_id": service_id,
                "model": model,
                "feature": feature,
                "reason": reason,
            },
        )
    )


def async_create_ffmpeg_missing_issue(
    hass: HomeAssistant,
    entry: GroqConfigEntry,
    service_data: dict[str, Any] | None = None,
) -> None:
    """Create a repair issue when ffmpeg is required but missing."""
    service_id = (service_data or {}).get(UNIQUE_ID)
    _create_issue(
        hass,
        ISSUE_FFMPEG_MISSING,
        _hashed_issue_id(ISSUE_FFMPEG_MISSING, entry.entry_id, service_id),
        {"service_name": _service_name(service_data)},
        entry_id=entry.entry_id,
        service_id=service_id,
    )


def async_delete_ffmpeg_missing_issue(
    hass: HomeAssistant,
    entry: GroqConfigEntry,
    service_data: dict[str, Any] | None = None,
) -> None:
    """Delete the ffmpeg repair issue after audio processing succeeds."""
    _update_issue(
        lambda: ir.async_delete_issue(
            hass,
            DOMAIN,
            _hashed_issue_id(
                ISSUE_FFMPEG_MISSING,
                entry.entry_id,
                (service_data or {}).get(UNIQUE_ID),
            ),
        )
    )


def async_create_model_access_issue(
    hass: HomeAssistant,
    model: str,
    service_id: str | None = None,
    *,
    entry_id: str | None = None,
    reason: str | None = None,
) -> None:
    """Report a model access failure scoped to its account and service."""
    _create_issue(
        hass,
        ISSUE_MODEL_ACCESS,
        _hashed_issue_id(ISSUE_MODEL_ACCESS, entry_id, model, service_id),
        {"model": _safe(model)},
        entry_id=entry_id,
        service_id=service_id,
        model=model,
        reason=reason,
    )


def async_delete_model_access_issue(
    hass: HomeAssistant,
    model: str,
    service_id: str | None = None,
    *,
    entry_id: str | None = None,
) -> None:
    """Clear matching account/service access failure after successful inference."""
    _update_issue(
        lambda: ir.async_delete_issue(
            hass,
            DOMAIN,
            _hashed_issue_id(ISSUE_MODEL_ACCESS, entry_id, model, service_id),
        )
    )


def async_create_model_configuration_issue(
    hass: HomeAssistant,
    entry: GroqConfigEntry,
    service_data: dict[str, Any],
    model: str,
    feature: str,
) -> None:
    """Report a configured model that cannot serve a feature."""
    service_id = service_data.get(UNIQUE_ID)
    _create_issue(
        hass,
        ISSUE_MODEL_CONFIGURATION,
        _hashed_issue_id(
            ISSUE_MODEL_CONFIGURATION, entry.entry_id, service_id, model, feature
        ),
        {
            "service_name": _service_name(service_data),
            "model": _safe(model),
            "feature": _safe(feature),
        },
        entry_id=entry.entry_id,
        service_id=service_id,
        model=model,
        feature=feature,
    )


def async_delete_model_configuration_issue(
    hass: HomeAssistant,
    entry: GroqConfigEntry,
    service_data: dict[str, Any],
    model: str,
    feature: str,
) -> None:
    """Clear the configured-model issue after capability validation succeeds."""
    _update_issue(
        lambda: ir.async_delete_issue(
            hass,
            DOMAIN,
            _hashed_issue_id(
                ISSUE_MODEL_CONFIGURATION,
                entry.entry_id,
                service_data.get(UNIQUE_ID),
                model,
                feature,
            ),
        )
    )


def async_reconcile_entry_issues(
    hass: HomeAssistant,
    entry: GroqConfigEntry,
    *,
    model_registry: GroqModelRegistry | None = None,
    remove_entry: bool = False,
) -> None:
    """Remove obsolete owned issues after setup, configuration changes, or removal."""

    def reconcile() -> None:
        registry = ir.async_get(hass)
        services = {
            data.get(UNIQUE_ID): data
            for group in service_data_by_type(entry).values()
            for data in group
        }
        configured_models = {data.get(CONF_MODEL) for data in services.values()}
        configured_models.add(entry_value(entry, CONF_MODEL))
        for (domain, issue_id), issue in list(registry.issues.items()):
            if domain != DOMAIN or not issue_id.startswith(
                tuple(f"{kind}_" for kind in _ISSUE_TYPES)
            ):
                continue
            data = issue.data or {}
            # Old persistent issues had no ownership metadata and could never be
            # resolved safely. Retire them once; active failures recreate scoped issues.
            if "entry_id" not in data:
                ir.async_delete_issue(hass, DOMAIN, issue_id)
                continue
            if data["entry_id"] != entry.entry_id:
                continue
            service_id = data.get("service_id")
            obsolete = remove_entry or (
                service_id is not None and service_id not in services
            )
            model = data.get("model")
            if model is not None:
                if service_id is not None and service_id in services:
                    obsolete |= services[service_id].get(CONF_MODEL) != model
                else:
                    obsolete |= model not in configured_models
            feature = data.get("feature")
            if (
                model_registry is not None
                and isinstance(model, str)
                and isinstance(feature, str)
            ):
                try:
                    obsolete |= model_registry.supports(model, GroqFeature(feature))
                except ValueError:
                    obsolete = True
            if obsolete:
                ir.async_delete_issue(hass, DOMAIN, issue_id)

    _update_issue(reconcile)
