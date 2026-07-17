"""Shared entity helpers for Groq services."""

from __future__ import annotations

from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo

from .const import DOMAIN


def service_device_info(unique_id: str, model: str, name: str) -> DeviceInfo:
    """Return registry information for a Groq cloud service."""
    return DeviceInfo(
        entry_type=DeviceEntryType.SERVICE,
        identifiers={(DOMAIN, unique_id)},
        manufacturer="Groq",
        model=model,
        name=name,
    )
