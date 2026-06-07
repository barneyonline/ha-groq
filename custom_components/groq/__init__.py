"""Custom integration for Groq."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
import homeassistant.helpers.config_validation as cv

from .const import DOMAIN, UNIQUE_ID
from .runtime import (
    GroqConfigEntry,
    async_hydrate_runtime_model_registry,
    build_runtime,
)
from .api import async_preload_clientsession_helper

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.empty_config_schema(DOMAIN)


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up Groq integration-level actions."""
    if hasattr(hass, "services"):
        from .services import async_register_services

        await async_register_services(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: GroqConfigEntry) -> bool:
    """Set up entities."""
    await async_preload_clientsession_helper(hass)
    runtime = build_runtime(hass, entry)
    await async_hydrate_runtime_model_registry(entry, runtime, raise_not_ready=True)
    entry.runtime_data = runtime

    if hasattr(hass, "services"):
        from .services import async_update_service_descriptions

        await async_update_service_descriptions(hass)
    # Service subentries determine which HA platforms are needed; account-only
    # entries do not create entities until the user adds at least one service.
    await hass.config_entries.async_forward_entry_setups(
        entry, runtime.feature_registry.enabled_platforms()
    )
    return True


async def async_unload_entry(hass: HomeAssistant, entry: GroqConfigEntry) -> bool:
    """Unload a config entry."""
    runtime = getattr(entry, "runtime_data", None)
    platforms = runtime.feature_registry.enabled_platforms() if runtime else []
    unload_ok = await hass.config_entries.async_unload_platforms(entry, platforms)
    if unload_ok:
        if hasattr(hass, "services"):
            from .services import async_update_service_descriptions

            await async_update_service_descriptions(
                hass,
                exclude_entry_id=entry.entry_id,
            )
    return unload_ok


def _has_other_loaded_entries(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Return whether another Groq config entry is still loaded."""
    async_entries = getattr(hass.config_entries, "async_entries", None)
    if async_entries is None:
        return False
    for other_entry in async_entries(DOMAIN):
        if other_entry.entry_id == entry.entry_id:
            continue
        if (
            getattr(other_entry, "state", ConfigEntryState.LOADED)
            == ConfigEntryState.LOADED
        ):
            return True
    return False


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate old entry data to new format.

    - Move legacy UNIQUE_ID stored in data to entry.unique_id
    """
    # If the entry already has a unique_id, nothing to do
    if entry.unique_id:
        return True

    # Migrate legacy unique id
    if isinstance(entry.data, dict) and UNIQUE_ID in entry.data:
        new_data = dict(entry.data)
        unique_id = new_data.pop(UNIQUE_ID)
        _LOGGER.debug("Migrating config entry to set unique_id and clean data")
        hass.config_entries.async_update_entry(
            entry, data=new_data, unique_id=unique_id
        )
        return True

    return True
