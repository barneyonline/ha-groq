"""Runtime data helpers for Groq config entries."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed

from .api import GroqApiClient, normalize_base_url
from .const import (
    CONF_API_KEY,
    CONF_CACHE_SIZE,
    CONF_ENABLED_FEATURES,
    CONF_INCLUDE_REASONING,
    CONF_MODEL,
    CONF_PROMPT_CACHING,
    CONF_REASONING_EFFORT,
    CONF_REASONING_FORMAT,
    CONF_STRUCTURED_OUTPUTS,
    CONF_URL,
    CONF_VOICE,
    DEFAULT_CACHE_SIZE,
    FEATURE_TEXT_TO_SPEECH,
    FEATURE_TEXT_GENERATION,
    PROMPT_CACHING_MODELS,
)
from .errors import GroqApiError, GroqResponseError
from .feature_registry import (
    GroqFeature,
    GroqFeatureRegistry,
    enabled_features_from_options,
)
from .model_registry import GroqModelRegistry
from .prompt_cache import GroqPromptCache
from .rate_limit import GroqRateLimiter
from .subentries import service_data_by_type

CONF_BASE_URL = "base_url"
CONF_PROMPT_CACHE_SIZE = "prompt_cache_size"
CONF_PROMPT_CACHE_TTL = "prompt_cache_ttl"

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GroqRuntimeData:
    """Shared runtime objects for one Groq config entry."""

    client: GroqApiClient
    model_registry: GroqModelRegistry
    feature_registry: GroqFeatureRegistry
    rate_limiter: GroqRateLimiter
    prompt_cache: GroqPromptCache
    services_by_type: dict[str, tuple[dict[str, Any], ...]]


type GroqConfigEntry = ConfigEntry[GroqRuntimeData]


def entry_value(entry: ConfigEntry, key: str, default: Any = None) -> Any:
    """Return an effective config entry value, allowing options to override data."""
    return entry.options.get(key, entry.data.get(key, default))


def _has_legacy_tts_config(entry: ConfigEntry) -> bool:
    """Return whether an entry contains pre-subentry TTS configuration."""
    return all(entry_value(entry, key) for key in (CONF_URL, CONF_MODEL, CONF_VOICE))


def build_runtime(hass: HomeAssistant, entry: ConfigEntry) -> GroqRuntimeData:
    """Create runtime data for a config entry."""
    base_url = entry_value(
        entry,
        CONF_BASE_URL,
        normalize_base_url(entry_value(entry, CONF_URL)),
    )
    cache_size = int(
        entry_value(
            entry,
            CONF_PROMPT_CACHE_SIZE,
            entry_value(entry, CONF_CACHE_SIZE, DEFAULT_CACHE_SIZE),
        )
    )
    cache_ttl = entry_value(entry, CONF_PROMPT_CACHE_TTL, 300)

    configured_features = entry.options.get(
        CONF_ENABLED_FEATURES,
        entry.data.get(CONF_ENABLED_FEATURES),
    )
    if configured_features is None:
        # Entries created before service subentries stored TTS settings directly
        # on the config entry. Keep those working while account-only entries stay
        # platform-free until the user adds a service.
        enabled_features = (
            {GroqFeature(FEATURE_TEXT_TO_SPEECH)}
            if _has_legacy_tts_config(entry)
            else set()
        )
    else:
        enabled_features = set(
            enabled_features_from_options({CONF_ENABLED_FEATURES: configured_features})
        )
    services_by_type = service_data_by_type(entry)
    for service_type, services in services_by_type.items():
        try:
            enabled_features.add(GroqFeature(service_type))
        except ValueError:
            continue
        if service_type == FEATURE_TEXT_GENERATION:
            # Text generation sub-options become runtime features only when a
            # configured service actually enables them.
            for data in services:
                if data.get(CONF_STRUCTURED_OUTPUTS):
                    enabled_features.add(GroqFeature.STRUCTURED_OUTPUTS)
                if (
                    data.get(CONF_PROMPT_CACHING)
                    and data.get(CONF_MODEL) in PROMPT_CACHING_MODELS
                ):
                    enabled_features.add(GroqFeature.PROMPT_CACHING)
                if (
                    data.get(CONF_REASONING_EFFORT)
                    or data.get(CONF_REASONING_FORMAT)
                    or data.get(CONF_INCLUDE_REASONING)
                ):
                    enabled_features.add(GroqFeature.REASONING)

    rate_limiter = GroqRateLimiter()
    return GroqRuntimeData(
        client=GroqApiClient(
            hass,
            api_key=entry_value(entry, CONF_API_KEY),
            base_url=base_url,
            rate_limiter=rate_limiter,
        ),
        model_registry=GroqModelRegistry(),
        feature_registry=GroqFeatureRegistry(enabled_features),
        rate_limiter=rate_limiter,
        prompt_cache=GroqPromptCache(max_size=cache_size, default_ttl=cache_ttl),
        services_by_type=services_by_type,
    )


async def async_hydrate_runtime_model_registry(
    entry: ConfigEntry,
    runtime: GroqRuntimeData,
    *,
    hydrate_details: bool = False,
) -> None:
    """Hydrate runtime model metadata from Groq when credentials are available."""
    if not entry_value(entry, CONF_API_KEY):
        return
    try:
        runtime.model_registry.update(
            await runtime.client.async_list_models(hydrate=hydrate_details)
        )
    except ConfigEntryAuthFailed:
        raise
    except (GroqApiError, GroqResponseError, TimeoutError) as err:
        # Built-in model metadata keeps setup usable when Groq is temporarily
        # unreachable; users can still refresh model data from the integration.
        _LOGGER.debug("Could not hydrate Groq model metadata during setup: %s", err)


async def async_get_runtime(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> GroqRuntimeData:
    """Return typed runtime data for a config entry, creating it if needed."""
    runtime = getattr(entry, "runtime_data", None)
    if isinstance(runtime, GroqRuntimeData):
        return runtime
    runtime = build_runtime(hass, entry)
    try:
        entry.runtime_data = runtime
    except AttributeError:
        pass
    return runtime
