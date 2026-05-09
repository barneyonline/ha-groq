"""Helpers for Groq config subentry data."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry

from .const import CONF_SERVICE_TYPE, UNIQUE_ID


def service_data_by_type(entry: ConfigEntry) -> dict[str, tuple[dict[str, Any], ...]]:
    """Return configured subentry data grouped by Groq service type."""
    services: dict[str, list[dict[str, Any]]] = {}
    for subentry in (getattr(entry, "subentries", None) or {}).values():
        data = dict(getattr(subentry, "data", {}))
        service_type = data.get(CONF_SERVICE_TYPE)
        if not service_type:
            continue
        data.setdefault(UNIQUE_ID, getattr(subentry, "subentry_id", None))
        if data.get(UNIQUE_ID) is None:
            data.pop(UNIQUE_ID, None)
        services.setdefault(service_type, []).append(data)
    return {
        service_type: tuple(service_data)
        for service_type, service_data in services.items()
    }


def service_data_for_type(
    entry: ConfigEntry,
    service_type: str,
) -> list[dict[str, Any]]:
    """Return configured subentry data for a single Groq service type."""
    return [dict(data) for data in service_data_by_type(entry).get(service_type, ())]
