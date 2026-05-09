"""Response services for optional Groq features."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv

from .api import (
    StructuredGenerationRequest,
    TextGenerationRequest,
    VisionRequest,
)
from .const import (
    CONF_API_KEY,
    CONF_INCLUDE_REASONING,
    CONF_MAX_TOKENS,
    CONF_NAME,
    CONF_REASONING_EFFORT,
    CONF_REASONING_FORMAT,
    CONF_REQUEST_BODY_OPTIONS,
    CONF_SCHEMA,
    CONF_SCHEMA_NAME,
    CONF_SEED,
    CONF_SERVICE_TIER,
    CONF_STOP,
    CONF_STRUCTURED_OUTPUTS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_SYSTEM_PROMPT,
    DOMAIN,
    FEATURE_IMAGE_RECOGNITION,
    FEATURE_TEXT_GENERATION,
    UNIQUE_ID,
)
from .feature_registry import GroqFeature
from .model_registry import (
    DEFAULT_STRUCTURED_MODEL,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
)
from .runtime import GroqRuntimeData, async_get_runtime
from .subentries import service_data_for_type

ATTR_CONFIG_ENTRY_ID = "config_entry_id"
ATTR_SERVICE_ID = "service_id"
ATTR_PROMPT = "prompt"
ATTR_MODEL = "model"
ATTR_SYSTEM_PROMPT = "system_prompt"
ATTR_TEMPERATURE = "temperature"
ATTR_MAX_TOKENS = "max_tokens"
ATTR_TOP_P = "top_p"
ATTR_STOP = "stop"
ATTR_SEED = "seed"
ATTR_SERVICE_TIER = "service_tier"
ATTR_REASONING_EFFORT = "reasoning_effort"
ATTR_REASONING_FORMAT = "reasoning_format"
ATTR_INCLUDE_REASONING = "include_reasoning"
ATTR_REQUEST_BODY_OPTIONS = "request_body_options"
ATTR_SCHEMA = "schema"
ATTR_SCHEMA_NAME = "schema_name"
ATTR_STRICT = "strict"
ATTR_IMAGE_URL = "image_url"
ATTR_REFRESH = "refresh"

SERVICE_GENERATE_TEXT = "generate_text"
SERVICE_GENERATE_STRUCTURED = "generate_structured"
SERVICE_ANALYZE_IMAGE = "analyze_image"
SERVICE_EXTRACT_TEXT_FROM_IMAGE = "extract_text_from_image"
SERVICE_CLEAR_CACHE = "clear_cache"
SERVICE_LIST_MODELS = "list_models"

_REGISTERED = "services_registered"

_ENTRY_SELECTOR = vol.Optional(ATTR_CONFIG_ENTRY_ID)
_SERVICE_SELECTOR = vol.Optional(ATTR_SERVICE_ID)
_MODEL_SELECTOR = vol.Optional(ATTR_MODEL)
_TEXT_OPTIONS = {
    _ENTRY_SELECTOR: cv.string,
    _SERVICE_SELECTOR: cv.string,
    _MODEL_SELECTOR: cv.string,
    vol.Required(ATTR_PROMPT): cv.string,
    vol.Optional(ATTR_SYSTEM_PROMPT): cv.string,
    vol.Optional(ATTR_TEMPERATURE): vol.All(vol.Coerce(float), vol.Range(min=0, max=2)),
    vol.Optional(ATTR_MAX_TOKENS): vol.All(vol.Coerce(int), vol.Range(min=1)),
    vol.Optional(ATTR_TOP_P): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
    vol.Optional(ATTR_STOP): vol.Any(cv.string, [cv.string]),
    vol.Optional(ATTR_SEED): vol.All(vol.Coerce(int), vol.Range(min=0)),
    vol.Optional(ATTR_SERVICE_TIER): cv.string,
    vol.Optional(ATTR_REASONING_EFFORT): cv.string,
    vol.Optional(ATTR_REASONING_FORMAT): cv.string,
    vol.Optional(ATTR_INCLUDE_REASONING): cv.boolean,
    vol.Optional(ATTR_REQUEST_BODY_OPTIONS): dict,
}

GENERATE_TEXT_SCHEMA = vol.Schema(
    {
        **_TEXT_OPTIONS,
        vol.Optional(ATTR_SCHEMA): dict,
        vol.Optional(ATTR_SCHEMA_NAME, default="response"): cv.string,
        vol.Optional(ATTR_STRICT, default=False): cv.boolean,
    }
)
GENERATE_STRUCTURED_SCHEMA = vol.Schema(
    {
        **_TEXT_OPTIONS,
        vol.Required(ATTR_SCHEMA): dict,
        vol.Optional(ATTR_SCHEMA_NAME, default="response"): cv.string,
        vol.Optional(ATTR_STRICT, default=False): cv.boolean,
    }
)
VISION_SCHEMA = vol.Schema(
    {
        _ENTRY_SELECTOR: cv.string,
        _SERVICE_SELECTOR: cv.string,
        _MODEL_SELECTOR: cv.string,
        vol.Required(ATTR_PROMPT): cv.string,
        vol.Optional(ATTR_SYSTEM_PROMPT): cv.string,
        vol.Required(ATTR_IMAGE_URL): cv.string,
    }
)
OCR_SCHEMA = vol.Schema(
    {
        _ENTRY_SELECTOR: cv.string,
        _SERVICE_SELECTOR: cv.string,
        _MODEL_SELECTOR: cv.string,
        vol.Optional(
            ATTR_PROMPT,
            default="Extract all visible text from this image. Return only the text.",
        ): cv.string,
        vol.Optional(ATTR_SYSTEM_PROMPT): cv.string,
        vol.Required(ATTR_IMAGE_URL): cv.string,
    }
)
CLEAR_CACHE_SCHEMA = vol.Schema({_ENTRY_SELECTOR: cv.string})
LIST_MODELS_SCHEMA = vol.Schema(
    {
        _ENTRY_SELECTOR: cv.string,
        vol.Optional(ATTR_REFRESH, default=False): cv.boolean,
    }
)


def _cache_key(namespace: str, data: dict[str, Any]) -> str:
    """Return a stable cache key without exposing raw prompt text."""
    encoded = json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
    return f"{namespace}:{sha256(encoded.encode('utf-8')).hexdigest()}"


def _domain_data(hass: HomeAssistant) -> dict[str, Any]:
    """Return integration domain data."""
    return hass.data.setdefault(DOMAIN, {})


def _loaded_entries(hass: HomeAssistant) -> list[ConfigEntry]:
    """Return loaded Groq entries."""
    return [
        entry
        for entry in hass.config_entries.async_entries(DOMAIN)
        if getattr(entry, "state", ConfigEntryState.LOADED) == ConfigEntryState.LOADED
    ]


def _entry_from_call(hass: HomeAssistant, call: ServiceCall) -> ConfigEntry:
    """Resolve a Groq config entry from a service call."""
    entry_id = call.data.get(ATTR_CONFIG_ENTRY_ID)
    if entry_id:
        entry = hass.config_entries.async_get_entry(entry_id)
        if entry is None or getattr(entry, "domain", DOMAIN) not in (DOMAIN, None):
            raise ServiceValidationError(f"Groq config entry not found: {entry_id}")
        return entry

    entries = _loaded_entries(hass)
    if not entries:
        raise ServiceValidationError("No loaded Groq config entry found")
    if len(entries) > 1:
        raise ServiceValidationError(
            "Multiple Groq config entries are loaded; provide config_entry_id"
        )
    return entries[0]


async def _runtime_from_call(
    hass: HomeAssistant,
    call: ServiceCall,
) -> tuple[ConfigEntry, GroqRuntimeData]:
    """Return entry and runtime for a service call."""
    entry = _entry_from_call(hass, call)
    runtime = getattr(entry, "runtime_data", None)
    if runtime is None:
        runtime = await async_get_runtime(hass, entry)
        try:
            entry.runtime_data = runtime
        except AttributeError:
            pass
    return entry, runtime


def _service_subentries(
    entry: ConfigEntry,
    runtime: GroqRuntimeData | None,
    service_type: str,
) -> list[dict[str, Any]]:
    """Return service subentry data for a service type."""
    if runtime is not None:
        return [dict(data) for data in runtime.services_by_type.get(service_type, ())]

    return service_data_for_type(entry, service_type)


def _service_from_call(
    entry: ConfigEntry,
    runtime: GroqRuntimeData,
    call: ServiceCall,
    service_type: str,
) -> dict[str, Any]:
    """Resolve an optional service subentry from a service call."""
    services = _service_subentries(entry, runtime, service_type)
    requested = call.data.get(ATTR_SERVICE_ID)
    if requested:
        for service in services:
            # Accept both the stable subentry id and the user-facing service
            # name so service calls remain practical from automations.
            if requested in (service.get(UNIQUE_ID), service.get(CONF_NAME)):
                return service
        raise ServiceValidationError(
            f"Groq {service_type} service not found: {requested}"
        )
    if len(services) == 1:
        return services[0]
    if len(services) > 1:
        raise ServiceValidationError(
            f"Multiple Groq {service_type} services are configured; provide service_id"
        )
    return {}


def _service_value(
    call: ServiceCall,
    service_data: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Return a service call value, falling back to selected subentry defaults."""
    if key in call.data:
        return call.data[key]
    return service_data.get(key, default)


def _service_api_key(service_data: dict[str, Any]) -> str | None:
    """Return an optional service-specific API key."""
    value = service_data.get(CONF_API_KEY)
    return str(value) if value else None


def _ensure_feature(runtime: GroqRuntimeData, feature: GroqFeature) -> None:
    """Raise if a feature is disabled."""
    runtime.feature_registry.ensure_enabled(feature)


def _ensure_model(runtime: GroqRuntimeData, model: str, feature: GroqFeature) -> None:
    """Raise if a model is not known to support a feature."""
    if not runtime.model_registry.supports(model, feature):
        raise ServiceValidationError(
            f"Groq model {model} is not known to support {feature.value}"
        )


def _reasoning_requested(
    data: dict[str, Any],
    service_data: dict[str, Any] | None = None,
) -> bool:
    """Return whether a service call requested Groq reasoning options."""
    service_data = service_data or {}
    return bool(
        data.get(ATTR_REASONING_EFFORT)
        or data.get(ATTR_REASONING_FORMAT)
        or data.get(ATTR_INCLUDE_REASONING)
        or service_data.get(CONF_REASONING_EFFORT)
        or service_data.get(CONF_REASONING_FORMAT)
        or service_data.get(CONF_INCLUDE_REASONING)
    )


def _request_options(
    data: dict[str, Any],
    service_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return shared chat completion request options."""
    service_data = service_data or {}
    include_reasoning = data.get(
        ATTR_INCLUDE_REASONING,
        service_data.get(CONF_INCLUDE_REASONING),
    )
    return {
        "temperature": data.get(ATTR_TEMPERATURE, service_data.get(CONF_TEMPERATURE)),
        "max_tokens": data.get(ATTR_MAX_TOKENS, service_data.get(CONF_MAX_TOKENS)),
        "top_p": data.get(ATTR_TOP_P, service_data.get(CONF_TOP_P)),
        "stop": data.get(ATTR_STOP, service_data.get(CONF_STOP)),
        "seed": data.get(ATTR_SEED, service_data.get(CONF_SEED)),
        "service_tier": data.get(
            ATTR_SERVICE_TIER,
            service_data.get(CONF_SERVICE_TIER),
        ),
        "reasoning_effort": data.get(
            ATTR_REASONING_EFFORT,
            service_data.get(CONF_REASONING_EFFORT),
        ),
        "reasoning_format": data.get(
            ATTR_REASONING_FORMAT,
            service_data.get(CONF_REASONING_FORMAT),
        ),
        # Only send include_reasoning when enabled. Some Groq models reject a
        # false/null reasoning flag even though they accept omitted options.
        "include_reasoning": True if include_reasoning else None,
        "extra_body": data.get(
            ATTR_REQUEST_BODY_OPTIONS,
            service_data.get(CONF_REQUEST_BODY_OPTIONS),
        ),
    }


def _request_cache_fields(request: TextGenerationRequest) -> dict[str, Any]:
    """Return request values that influence a generated response."""
    return {
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "top_p": request.top_p,
        "stop": request.stop,
        "seed": request.seed,
        "service_tier": request.service_tier,
        "reasoning_effort": request.reasoning_effort,
        "reasoning_format": request.reasoning_format,
        "include_reasoning": request.include_reasoning,
        "extra_body": request.extra_body,
    }


def _prompt_cache_allowed(runtime: GroqRuntimeData, model: str) -> bool:
    """Return whether prompt caching should apply for this model."""
    return runtime.feature_registry.is_enabled(
        GroqFeature.PROMPT_CACHING
    ) and runtime.model_registry.supports(model, GroqFeature.PROMPT_CACHING)


def _cache_get(
    runtime: GroqRuntimeData,
    model: str,
    key: str,
) -> ServiceResponse | None:
    """Return a cached response when prompt caching is enabled."""
    if not _prompt_cache_allowed(runtime, model):
        return None
    cached = runtime.prompt_cache.get(key)
    if cached is None:
        return None
    cached["cached"] = True
    return cached


def _cache_set(
    runtime: GroqRuntimeData,
    model: str,
    key: str,
    response: ServiceResponse,
) -> None:
    """Store a response when prompt caching is enabled."""
    if _prompt_cache_allowed(runtime, model):
        runtime.prompt_cache.set(key, response)


def _handle_generate_text(hass: HomeAssistant):
    """Build the generate_text service handler."""

    async def handler(call: ServiceCall) -> ServiceResponse:
        entry, runtime = await _runtime_from_call(hass, call)
        _ensure_feature(runtime, GroqFeature.TEXT_GENERATION)
        service_data = _service_from_call(entry, runtime, call, FEATURE_TEXT_GENERATION)
        model = _service_value(call, service_data, ATTR_MODEL, DEFAULT_TEXT_MODEL)
        _ensure_model(runtime, model, GroqFeature.TEXT_GENERATION)
        if _reasoning_requested(call.data, service_data):
            _ensure_model(runtime, model, GroqFeature.REASONING)

        schema = call.data.get(ATTR_SCHEMA)
        if schema is None and service_data.get(CONF_STRUCTURED_OUTPUTS):
            schema = service_data.get(CONF_SCHEMA)
        if schema:
            # generate_text doubles as the ergonomic entry point for structured
            # outputs when the selected service has a schema configured.
            _ensure_model(runtime, model, GroqFeature.STRUCTURED_OUTPUTS)
            request = StructuredGenerationRequest(
                prompt=call.data[ATTR_PROMPT],
                model=model,
                system_prompt=_service_value(
                    call,
                    service_data,
                    ATTR_SYSTEM_PROMPT,
                    DEFAULT_SYSTEM_PROMPT,
                ),
                **_request_options(call.data, service_data),
                api_key=_service_api_key(service_data),
                schema=schema,
                schema_name=_service_value(
                    call,
                    service_data,
                    ATTR_SCHEMA_NAME,
                    service_data.get(CONF_SCHEMA_NAME, "response"),
                ),
                strict=_service_value(call, service_data, ATTR_STRICT, False),
            )
        else:
            request = TextGenerationRequest(
                prompt=call.data[ATTR_PROMPT],
                model=model,
                system_prompt=_service_value(
                    call,
                    service_data,
                    ATTR_SYSTEM_PROMPT,
                    DEFAULT_SYSTEM_PROMPT,
                ),
                **_request_options(call.data, service_data),
                api_key=_service_api_key(service_data),
            )
        key = _cache_key(
            "text_generation",
            {
                "service_id": service_data.get(UNIQUE_ID),
                "model": request.model,
                "prompt": request.prompt,
                "system_prompt": request.system_prompt,
                **_request_cache_fields(request),
                "schema": schema,
                "schema_name": getattr(request, "schema_name", "response"),
                "strict": getattr(request, "strict", False),
            },
        )
        if cached := _cache_get(runtime, request.model, key):
            return cached

        if isinstance(request, StructuredGenerationRequest):
            response = await runtime.client.async_generate_structured(request)
        else:
            result = await runtime.client.async_generate_text(request)
            response = {
                "text": result.text,
                "reasoning": result.reasoning,
                "executed_tools": result.executed_tools or [],
                "usage_breakdown": result.usage_breakdown or {},
                "model": result.model,
                "usage": result.usage,
                "cached": False,
            }
        _cache_set(runtime, request.model, key, response)
        return response

    return handler


def _handle_generate_structured(hass: HomeAssistant):
    """Build the generate_structured service handler."""

    async def handler(call: ServiceCall) -> ServiceResponse:
        entry, runtime = await _runtime_from_call(hass, call)
        _ensure_feature(runtime, GroqFeature.STRUCTURED_OUTPUTS)
        service_data = _service_from_call(entry, runtime, call, FEATURE_TEXT_GENERATION)
        model = _service_value(call, service_data, ATTR_MODEL, DEFAULT_STRUCTURED_MODEL)
        _ensure_model(runtime, model, GroqFeature.STRUCTURED_OUTPUTS)
        if _reasoning_requested(call.data, service_data):
            _ensure_model(runtime, model, GroqFeature.REASONING)
        request = StructuredGenerationRequest(
            prompt=call.data[ATTR_PROMPT],
            model=model,
            system_prompt=_service_value(call, service_data, ATTR_SYSTEM_PROMPT),
            **_request_options(call.data, service_data),
            api_key=_service_api_key(service_data),
            schema=call.data[ATTR_SCHEMA],
            schema_name=_service_value(
                call,
                service_data,
                ATTR_SCHEMA_NAME,
                service_data.get(CONF_SCHEMA_NAME, "response"),
            ),
            strict=_service_value(call, service_data, ATTR_STRICT, False),
        )
        key = _cache_key(
            "structured_outputs",
            {
                "service_id": service_data.get(UNIQUE_ID),
                "model": request.model,
                "prompt": request.prompt,
                "system_prompt": request.system_prompt,
                **_request_cache_fields(request),
                "schema": request.schema,
                "schema_name": request.schema_name,
                "strict": request.strict,
            },
        )
        if cached := _cache_get(runtime, request.model, key):
            return cached

        response = await runtime.client.async_generate_structured(request)
        _cache_set(runtime, request.model, key, response)
        return response

    return handler


def _handle_analyze_image(hass: HomeAssistant):
    """Build the analyze_image service handler."""

    async def handler(call: ServiceCall) -> ServiceResponse:
        entry, runtime = await _runtime_from_call(hass, call)
        _ensure_feature(runtime, GroqFeature.VISION)
        service_data = _service_from_call(
            entry, runtime, call, FEATURE_IMAGE_RECOGNITION
        )
        model = _service_value(call, service_data, ATTR_MODEL, DEFAULT_VISION_MODEL)
        _ensure_model(runtime, model, GroqFeature.VISION)
        request = VisionRequest(
            prompt=call.data[ATTR_PROMPT],
            model=model,
            system_prompt=_service_value(call, service_data, ATTR_SYSTEM_PROMPT),
            image_url=call.data[ATTR_IMAGE_URL],
            api_key=_service_api_key(service_data),
        )
        key = _cache_key(
            "vision",
            {
                "service_id": service_data.get(UNIQUE_ID),
                "model": request.model,
                "prompt": request.prompt,
                "system_prompt": request.system_prompt,
                "image_url": request.image_url,
            },
        )
        if cached := _cache_get(runtime, request.model, key):
            return cached

        result = await runtime.client.async_analyze_image(request)
        response = {
            "text": result.text,
            "model": result.model,
            "usage": result.usage,
            "cached": False,
        }
        _cache_set(runtime, request.model, key, response)
        return response

    return handler


def _handle_extract_text_from_image(hass: HomeAssistant):
    """Build the extract_text_from_image service handler."""

    async def handler(call: ServiceCall) -> ServiceResponse:
        entry, runtime = await _runtime_from_call(hass, call)
        _ensure_feature(runtime, GroqFeature.OCR)
        service_data = _service_from_call(
            entry, runtime, call, FEATURE_IMAGE_RECOGNITION
        )
        model = _service_value(call, service_data, ATTR_MODEL, DEFAULT_VISION_MODEL)
        _ensure_model(runtime, model, GroqFeature.OCR)
        request = VisionRequest(
            prompt=call.data[ATTR_PROMPT],
            model=model,
            system_prompt=_service_value(call, service_data, ATTR_SYSTEM_PROMPT),
            image_url=call.data[ATTR_IMAGE_URL],
            api_key=_service_api_key(service_data),
        )
        key = _cache_key(
            "ocr",
            {
                "service_id": service_data.get(UNIQUE_ID),
                "model": request.model,
                "prompt": request.prompt,
                "system_prompt": request.system_prompt,
                "image_url": request.image_url,
            },
        )
        if cached := _cache_get(runtime, request.model, key):
            return cached

        result = await runtime.client.async_analyze_image(request)
        response = {
            "text": result.text,
            "model": result.model,
            "usage": result.usage,
            "cached": False,
        }
        _cache_set(runtime, request.model, key, response)
        return response

    return handler


def _handle_clear_cache(hass: HomeAssistant):
    """Build the clear_cache service handler."""

    async def handler(call: ServiceCall) -> ServiceResponse:
        _entry, runtime = await _runtime_from_call(hass, call)
        _ensure_feature(runtime, GroqFeature.PROMPT_CACHING)
        return {"cleared": runtime.prompt_cache.clear()}

    return handler


def _handle_list_models(hass: HomeAssistant):
    """Build the list_models service handler."""

    async def handler(call: ServiceCall) -> ServiceResponse:
        _entry, runtime = await _runtime_from_call(hass, call)
        if call.data.get(ATTR_REFRESH):
            runtime.model_registry.update(await runtime.client.async_list_models())
        return {
            "models": [
                model.as_dict()
                for model in sorted(
                    runtime.model_registry.models.values(),
                    key=lambda item: item.model_id,
                )
            ]
        }

    return handler


async def async_register_services(hass: HomeAssistant) -> None:
    """Register Groq response services once."""
    data = _domain_data(hass)
    if data.get(_REGISTERED):
        return

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_TEXT,
        _handle_generate_text(hass),
        schema=GENERATE_TEXT_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_STRUCTURED,
        _handle_generate_structured(hass),
        schema=GENERATE_STRUCTURED_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_ANALYZE_IMAGE,
        _handle_analyze_image(hass),
        schema=VISION_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_EXTRACT_TEXT_FROM_IMAGE,
        _handle_extract_text_from_image(hass),
        schema=OCR_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_CLEAR_CACHE,
        _handle_clear_cache(hass),
        schema=CLEAR_CACHE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_LIST_MODELS,
        _handle_list_models(hass),
        schema=LIST_MODELS_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    data[_REGISTERED] = True


async def async_unregister_services(hass: HomeAssistant) -> None:
    """Remove Groq response services."""
    data = _domain_data(hass)
    if not data.pop(_REGISTERED, False):
        return
    for service in (
        SERVICE_GENERATE_TEXT,
        SERVICE_GENERATE_STRUCTURED,
        SERVICE_ANALYZE_IMAGE,
        SERVICE_EXTRACT_TEXT_FROM_IMAGE,
        SERVICE_CLEAR_CACHE,
        SERVICE_LIST_MODELS,
    ):
        hass.services.async_remove(DOMAIN, service)


async_setup_services = async_register_services
async_unload_services = async_unregister_services
