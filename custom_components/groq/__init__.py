"""Custom integration for Groq."""

from __future__ import annotations

from typing import Any

import homeassistant.helpers.config_validation as cv
from homeassistant.core import HomeAssistant

from .api import async_preload_clientsession_helper
from .const import DOMAIN, UNIQUE_ID
from .repairs import async_reconcile_entry_issues
from .runtime import (
    async_hydrate_runtime_model_registry,
    build_runtime,
)
from .types import GroqConfigEntry

CONFIG_SCHEMA = cv.empty_config_schema(DOMAIN)


async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool:
    """Set up Groq integration-level actions."""
    from .services import async_register_services

    await async_register_services(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: GroqConfigEntry) -> bool:
    """Set up entities."""
    await async_preload_clientsession_helper(hass)
    runtime = build_runtime(hass, entry)
    await async_hydrate_runtime_model_registry(entry, runtime, raise_not_ready=True)
    entry.runtime_data = runtime
    async_reconcile_entry_issues(hass, entry, model_registry=runtime.model_registry)

    from .services import async_update_service_descriptions

    await async_update_service_descriptions(hass)
    # Service subentries determine which HA platforms are needed; account-only
    # entries do not create entities until the user adds at least one service.
    await hass.config_entries.async_forward_entry_setups(
        entry, runtime.feature_registry.enabled_platforms()
    )
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


async def _async_update_listener(hass: HomeAssistant, entry: GroqConfigEntry) -> None:
    """Reload Groq when account options or service subentries change."""
    async_reconcile_entry_issues(hass, entry)
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: GroqConfigEntry) -> bool:
    """Unload a config entry."""
    runtime = getattr(entry, "runtime_data", None)
    platforms = runtime.feature_registry.enabled_platforms() if runtime else []
    unload_ok = await hass.config_entries.async_unload_platforms(entry, platforms)
    if unload_ok:
        if runtime is not None:
            await runtime.client.async_shutdown()
        from .services import async_update_service_descriptions

        await async_update_service_descriptions(
            hass,
            exclude_entry_id=entry.entry_id,
        )
    return bool(unload_ok)


async def async_migrate_entry(hass: HomeAssistant, entry: GroqConfigEntry) -> bool:
    """Migrate legacy stable IDs using Home Assistant's immutable data mappings."""
    if getattr(entry, "version", 1) > 1:
        return False
    if getattr(entry, "minor_version", 1) >= 2:
        return True
    updates: dict[str, Any] = {"minor_version": 2}
    if not entry.unique_id and UNIQUE_ID in entry.data:
        data = dict(entry.data)
        updates["unique_id"] = data.pop(UNIQUE_ID)
        updates["data"] = data
    hass.config_entries.async_update_entry(entry, **updates)
    return True


async def async_remove_entry(hass: HomeAssistant, entry: GroqConfigEntry) -> None:
    """Remove persistent issues owned by a deleted account."""
    async_reconcile_entry_issues(hass, entry, remove_entry=True)
