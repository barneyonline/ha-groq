"""Response services for optional Groq features."""

from __future__ import annotations

from base64 import b64decode, b64encode
from binascii import Error as BinasciiError
from collections.abc import Awaitable, Callable
from copy import deepcopy
from hashlib import sha256
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import voluptuous as vol

from homeassistant.components import camera
from homeassistant.components.media_source import (
    Unresolvable,
    async_resolve_media,
    is_media_source_id,
)
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import service as service_helper
from homeassistant.util.yaml.loader import load_yaml

from .api import (
    StructuredGenerationRequest,
    TextGenerationRequest,
    VisionRequest,
)
from .const import (
    CONF_INCLUDE_REASONING,
    CONF_LANGUAGE,
    CONF_MAX_TOKENS,
    CONF_NAME,
    CONF_PROTECT_FREE_TIER,
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
    DEFAULT_PROTECT_FREE_TIER,
    DEFAULT_STT_LANGUAGE,
    DEFAULT_STT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEXT_MODEL,
    DOMAIN,
    FEATURE_IMAGE_RECOGNITION,
    FEATURE_SPEECH_TO_TEXT,
    FEATURE_TEXT_GENERATION,
    UNIQUE_ID,
)
from .errors import translated_service_error
from .feature_registry import GroqFeature
from .model_registry import DEFAULT_VISION_MODEL
from .runtime import GroqRuntimeData, async_get_runtime
from .repairs import async_create_model_configuration_issue
from .subentries import service_data_for_type
from .text_generation import (
    request_body_options_error_message,
    request_context_window_error,
)

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
ATTR_CAMERA_ENTITY_ID = "camera_entity_id"
ATTR_IMAGE_URL = "image_url"
ATTR_IMAGE_FILE = "image_file"
ATTR_IMAGE_PATH = "image_path"
ATTR_AUDIO_FILE = "audio_file"
ATTR_AUDIO_PATH = "audio_path"
ATTR_LANGUAGE = "language"
ATTR_REFRESH = "refresh"

SERVICE_GENERATE_TEXT = "generate_text"
SERVICE_GENERATE_STRUCTURED = "generate_structured"
SERVICE_ANALYZE_IMAGE = "analyze_image"
SERVICE_EXTRACT_TEXT_FROM_IMAGE = "extract_text_from_image"
SERVICE_TRANSCRIBE_AUDIO = "transcribe_audio"
SERVICE_CLEAR_CACHE = "clear_cache"
SERVICE_LIST_MODELS = "list_models"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_AUDIO_BYTES = 25 * 1024 * 1024

_REGISTERED = "services_registered"
_SERVICES_YAML = Path(__file__).with_name("services.yaml")

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
        vol.Optional(ATTR_SCHEMA): dict,
        vol.Optional(ATTR_SCHEMA_NAME, default="response"): cv.string,
        vol.Optional(ATTR_STRICT, default=False): cv.boolean,
    }
)
VISION_SCHEMA = vol.Schema(
    {
        **cv.TARGET_SERVICE_FIELDS,
        _ENTRY_SELECTOR: cv.string,
        _SERVICE_SELECTOR: cv.string,
        _MODEL_SELECTOR: cv.string,
        vol.Required(ATTR_PROMPT): cv.string,
        vol.Optional(ATTR_SYSTEM_PROMPT): cv.string,
        vol.Optional(ATTR_CAMERA_ENTITY_ID): cv.entity_id,
        vol.Optional(ATTR_IMAGE_FILE): cv.string,
        vol.Optional(ATTR_IMAGE_PATH): cv.string,
        vol.Optional(ATTR_IMAGE_URL): cv.string,
    }
)
OCR_SCHEMA = vol.Schema(
    {
        **cv.TARGET_SERVICE_FIELDS,
        _ENTRY_SELECTOR: cv.string,
        _SERVICE_SELECTOR: cv.string,
        _MODEL_SELECTOR: cv.string,
        vol.Optional(
            ATTR_PROMPT,
            default="Extract all visible text from this image. Return only the text.",
        ): cv.string,
        vol.Optional(ATTR_SYSTEM_PROMPT): cv.string,
        vol.Optional(ATTR_CAMERA_ENTITY_ID): cv.entity_id,
        vol.Optional(ATTR_IMAGE_FILE): cv.string,
        vol.Optional(ATTR_IMAGE_PATH): cv.string,
        vol.Optional(ATTR_IMAGE_URL): cv.string,
    }
)
TRANSCRIBE_AUDIO_SCHEMA = vol.Schema(
    {
        _ENTRY_SELECTOR: cv.string,
        _SERVICE_SELECTOR: cv.string,
        _MODEL_SELECTOR: cv.string,
        vol.Optional(ATTR_AUDIO_FILE): cv.string,
        vol.Optional(ATTR_AUDIO_PATH): cv.string,
        vol.Optional(ATTR_LANGUAGE): cv.string,
        vol.Optional(ATTR_PROMPT): cv.string,
    }
)
CLEAR_CACHE_SCHEMA = vol.Schema({_ENTRY_SELECTOR: cv.string})
LIST_MODELS_SCHEMA = vol.Schema(
    {
        _ENTRY_SELECTOR: cv.string,
        vol.Optional(ATTR_REFRESH, default=False): cv.boolean,
    }
)

_SERVICE_FIELD_TYPES = {
    SERVICE_GENERATE_TEXT: FEATURE_TEXT_GENERATION,
    SERVICE_GENERATE_STRUCTURED: FEATURE_TEXT_GENERATION,
    SERVICE_ANALYZE_IMAGE: FEATURE_IMAGE_RECOGNITION,
    SERVICE_EXTRACT_TEXT_FROM_IMAGE: FEATURE_IMAGE_RECOGNITION,
    SERVICE_TRANSCRIBE_AUDIO: FEATURE_SPEECH_TO_TEXT,
}

ServiceHandler = Callable[[ServiceCall], Awaitable[ServiceResponse]]


def _cache_key(namespace: str, data: dict[str, Any]) -> str:
    """Return a stable cache key without exposing raw prompt text."""
    encoded = json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
    return f"{namespace}:{sha256(encoded.encode('utf-8')).hexdigest()}"


def _domain_data(hass: HomeAssistant) -> dict[str, Any]:
    """Return integration domain data."""
    return hass.data.setdefault(DOMAIN, {})


def _service_error(
    translation_key: str,
    fallback_message: str,
    **placeholders: object,
) -> ServiceValidationError:
    """Return a translated service validation error."""
    return translated_service_error(
        fallback_message,
        translation_key,
        **placeholders,
    )


def _entry_state_matches(
    entry: ConfigEntry,
    *,
    include_setup: bool = False,
) -> bool:
    """Return whether an entry should be considered for runtime or UI use."""
    state = getattr(entry, "state", ConfigEntryState.LOADED)
    if state == ConfigEntryState.LOADED:
        return True
    return include_setup and getattr(state, "name", None) == "SETUP_IN_PROGRESS"


def _loaded_entries(
    hass: HomeAssistant,
    *,
    exclude_entry_id: str | None = None,
    include_setup: bool = False,
) -> list[ConfigEntry]:
    """Return Groq entries usable for service calls or UI selectors."""
    return [
        entry
        for entry in hass.config_entries.async_entries(DOMAIN)
        if entry.entry_id != exclude_entry_id
        if _entry_state_matches(entry, include_setup=include_setup)
    ]


def _service_matches(service: dict[str, Any], requested: str) -> bool:
    """Return whether a service subentry matches a requested service selector."""
    return requested in (service.get(UNIQUE_ID), service.get(CONF_NAME))


def _entry_from_service_id(
    hass: HomeAssistant,
    service_type: str,
    requested: str,
) -> ConfigEntry | None:
    """Return the loaded config entry that owns a selected Groq service."""
    matched_entry: ConfigEntry | None = None
    for entry in _loaded_entries(hass):
        runtime = getattr(entry, "runtime_data", None)
        if any(
            _service_matches(service, requested)
            for service in _service_subentries(entry, runtime, service_type)
        ):
            if matched_entry is not None:
                raise _service_error(
                    "multiple_services_match",
                    f"Multiple Groq services match service_id: {requested}",
                    service_id=requested,
                )
            matched_entry = entry
    return matched_entry


def _entry_from_call(
    hass: HomeAssistant,
    call: ServiceCall,
    service_type: str | None = None,
) -> ConfigEntry:
    """Resolve a Groq config entry from a service call."""
    entry_id = call.data.get(ATTR_CONFIG_ENTRY_ID)
    if entry_id:
        entry = hass.config_entries.async_get_entry(entry_id)
        if entry is None or getattr(entry, "domain", DOMAIN) not in (DOMAIN, None):
            raise _service_error(
                "config_entry_not_found",
                f"Groq config entry not found: {entry_id}",
                entry_id=entry_id,
            )
        return entry

    entries = _loaded_entries(hass)
    if not entries:
        raise _service_error(
            "no_loaded_config_entry",
            "No loaded Groq config entry found",
        )
    if service_type and (requested := call.data.get(ATTR_SERVICE_ID)):
        if entry := _entry_from_service_id(hass, service_type, requested):
            return entry
    if len(entries) > 1:
        raise _service_error(
            "multiple_config_entries",
            "Multiple Groq config entries are loaded; provide config_entry_id or service_id",
        )
    return entries[0]


async def _runtime_from_call(
    hass: HomeAssistant,
    call: ServiceCall,
    service_type: str | None = None,
) -> tuple[ConfigEntry, GroqRuntimeData]:
    """Return entry and runtime for a service call."""
    entry = _entry_from_call(hass, call, service_type)
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
            if _service_matches(service, requested):
                return service
        raise _service_error(
            "service_not_found",
            f"Groq {service_type} service not found: {requested}",
            service_type=service_type,
            service_id=requested,
        )
    if len(services) == 1:
        return services[0]
    if len(services) > 1:
        raise _service_error(
            "multiple_services_configured",
            f"Multiple Groq {service_type} services are configured; provide service_id",
            service_type=service_type,
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


def _service_protect_free_tier(service_data: dict[str, Any]) -> bool:
    """Return whether local free-tier protection is enabled for this service."""
    return bool(service_data.get(CONF_PROTECT_FREE_TIER, DEFAULT_PROTECT_FREE_TIER))


def _ensure_feature(runtime: GroqRuntimeData, feature: GroqFeature) -> None:
    """Raise if a feature is disabled."""
    runtime.feature_registry.ensure_enabled(feature)


def _ensure_model(
    runtime: GroqRuntimeData,
    model: str,
    feature: GroqFeature,
    *,
    hass: HomeAssistant | None = None,
    entry: ConfigEntry | None = None,
    service_data: dict[str, Any] | None = None,
) -> None:
    """Raise if a model is not known to support a feature."""
    if not runtime.model_registry.supports(model, feature):
        if (
            hass is not None
            and entry is not None
            and service_data is not None
            and service_data.get(ATTR_MODEL) == model
        ):
            async_create_model_configuration_issue(
                hass,
                entry,
                service_data,
                model,
                feature.value,
            )
        raise _service_error(
            "unsupported_model",
            f"Groq model {model} is not known to support {feature.value}",
            model=model,
            feature=feature.value,
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


def _coerce_completion_tokens(value: Any) -> int | None:
    """Return an integer token count from a request option."""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ensure_completion_token_limit(
    runtime: GroqRuntimeData,
    request: TextGenerationRequest,
) -> None:
    """Raise when a request exceeds the selected model's completion limit."""
    limit = runtime.model_registry.completion_token_limit(request.model)
    if limit is None:
        return
    requested = [_coerce_completion_tokens(request.max_tokens)]
    if isinstance(request.extra_body, dict):
        requested.extend(
            (
                _coerce_completion_tokens(
                    request.extra_body.get("max_completion_tokens")
                ),
                _coerce_completion_tokens(request.extra_body.get("max_tokens")),
            )
        )
    if any(value is not None and value > limit for value in requested):
        limit_text = f"{limit:,}"
        raise _service_error(
            "completion_token_limit",
            f"Groq model {request.model} supports at most "
            f"{limit_text} completion tokens",
            model=request.model,
            limit=limit_text,
        )


def _ensure_request_body_features(
    runtime: GroqRuntimeData,
    request: TextGenerationRequest,
) -> None:
    """Raise when advanced body options require unsupported model features."""
    if error := request_body_options_error_message(
        runtime.model_registry,
        request.model,
        request.extra_body,
    ):
        raise _service_error("request_invalid", error, message=error)


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
    """Return whether local response caching should apply for this model."""
    return runtime.feature_registry.is_enabled(
        GroqFeature.PROMPT_CACHING
    ) and runtime.model_registry.supports(model, GroqFeature.PROMPT_CACHING)


def _cache_get(
    runtime: GroqRuntimeData,
    model: str,
    key: str,
) -> ServiceResponse | None:
    """Return a cached response when local response caching is enabled."""
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
    """Store a response when local response caching is enabled."""
    if _prompt_cache_allowed(runtime, model):
        runtime.prompt_cache.set(key, response)


def _image_data_url(content: bytes, content_type: str | None) -> str:
    """Return an image data URL from binary content."""
    media_type = content_type or "image/jpeg"
    return f"data:{media_type};base64,{b64encode(content).decode('ascii')}"


def _ensure_size_limit(
    size: int,
    limit: int,
    translation_key: str,
    media_type: str,
) -> None:
    """Raise when a local media payload is too large to buffer safely."""
    if size <= limit:
        return
    raise _service_error(
        translation_key,
        f"Selected {media_type} is too large; maximum size is "
        f"{limit // 1024 // 1024} MB",
        limit_mb=limit // 1024 // 1024,
    )


def _validate_image_url(image_url: str) -> str:
    """Return a supported image URL or raise a service validation error."""
    parsed = urlparse(image_url)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return image_url
    if parsed.scheme == "data" and image_url.lower().startswith("data:image/"):
        _validate_data_image_size(image_url)
        return image_url
    raise _service_error(
        "invalid_image_url",
        "Image URL must be an http, https, or image data URL",
        url=image_url,
    )


def _base64_decoded_size_upper_bound(payload: str) -> int:
    """Return the largest decoded byte size a base64 payload can produce."""
    padding = len(payload) - len(payload.rstrip("="))
    return ((len(payload) + 3) // 4) * 3 - min(padding, 2)


def _validate_data_image_size(image_url: str) -> None:
    """Raise when an inline image data URL is too large to process safely."""
    try:
        metadata, payload = image_url.split(",", 1)
    except (BinasciiError, ValueError) as err:
        raise _service_error(
            "invalid_image_url",
            "Image URL must be an http, https, or image data URL",
            url=image_url,
        ) from err
    if ";base64" not in metadata.lower():
        # Percent-encoded data URLs are not worth decoding here; cap the raw
        # payload size so callers cannot bypass the same guard.
        _ensure_size_limit(
            len(payload.encode("utf-8")),
            MAX_IMAGE_BYTES,
            "image_too_large",
            "image",
        )
        return
    _ensure_size_limit(
        _base64_decoded_size_upper_bound(payload),
        MAX_IMAGE_BYTES,
        "image_too_large",
        "image",
    )
    try:
        decoded_size = len(b64decode(payload, validate=True))
    except ValueError as err:
        raise _service_error(
            "invalid_image_url",
            "Image URL must be an http, https, or image data URL",
            url=image_url,
        ) from err
    _ensure_size_limit(decoded_size, MAX_IMAGE_BYTES, "image_too_large", "image")


async def _image_from_camera_target(
    hass: HomeAssistant,
    call: ServiceCall,
) -> str | None:
    """Return a data URL from the targeted camera entity, if one was selected."""
    if camera_entity_id := call.data.get(ATTR_CAMERA_ENTITY_ID):
        entity_ids = cv.ensure_list(camera_entity_id)
    elif direct_entity_id := call.data.get(ATTR_ENTITY_ID):
        entity_ids = cv.ensure_list(direct_entity_id)
    elif any(
        call.data.get(target_key)
        for target_key in ("device_id", "area_id", "floor_id", "label_id")
    ):
        entity_ids = await service_helper.async_extract_entity_ids(call)
    else:
        return None
    camera_entities = sorted(
        entity_id for entity_id in entity_ids if entity_id.startswith("camera.")
    )
    if not camera_entities:
        return None
    if len(camera_entities) > 1:
        raise _service_error("select_one_camera", "Select only one camera entity")
    try:
        image = await camera.async_get_image(hass, camera_entities[0])
    except Exception as err:  # pylint: disable=broad-except
        raise _service_error(
            "camera_capture_failed",
            f"Could not capture image from camera entity {camera_entities[0]}",
            entity_id=camera_entities[0],
        ) from err
    _ensure_size_limit(
        len(image.content),
        MAX_IMAGE_BYTES,
        "image_too_large",
        "image",
    )
    return _image_data_url(image.content, image.content_type)


async def _image_from_local_path(hass: HomeAssistant, image_path: str) -> str:
    """Return a data URL from an allowlisted local image path."""
    allowed = await hass.async_add_executor_job(
        hass.config.is_allowed_path,
        image_path,
    )
    if not allowed:
        raise _service_error(
            "local_image_path_not_allowed",
            "Local image path is not in Home Assistant's allowlist_external_dirs",
        )
    path = Path(image_path)
    if not await hass.async_add_executor_job(path.is_file):
        raise _service_error(
            "local_image_not_found",
            f"Local image file not found: {image_path}",
            path=image_path,
        )
    content_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    if not content_type.startswith("image/"):
        raise _service_error(
            "local_file_not_image",
            f"Local file is not an image: {image_path}",
            path=image_path,
        )
    size = await hass.async_add_executor_job(lambda: path.stat().st_size)
    _ensure_size_limit(size, MAX_IMAGE_BYTES, "image_too_large", "image")
    content = await hass.async_add_executor_job(path.read_bytes)
    return _image_data_url(content, content_type)


async def _image_from_media_source(hass: HomeAssistant, image_file: str) -> str:
    """Return a data URL or URL from a Home Assistant media-source image."""
    if not is_media_source_id(image_file):
        return await _image_from_local_path(hass, image_file)
    try:
        media = await async_resolve_media(hass, image_file, None)
    except Unresolvable as err:
        raise _service_error(
            "image_media_unresolvable",
            f"Could not resolve image file: {image_file}",
            path=image_file,
        ) from err
    if media.path is not None:
        return await _image_from_local_path(hass, str(media.path))
    if media.mime_type and media.mime_type.startswith("image/"):
        return _validate_image_url(media.url)
    raise _service_error(
        "selected_media_not_image",
        f"Selected media is not an image: {image_file}",
        path=image_file,
    )


async def _image_url_from_call(hass: HomeAssistant, call: ServiceCall) -> str:
    """Resolve the image source selected for a vision service call."""
    if camera_image := await _image_from_camera_target(hass, call):
        return camera_image
    if image_file := call.data.get(ATTR_IMAGE_FILE):
        return await _image_from_media_source(hass, image_file)
    if image_path := call.data.get(ATTR_IMAGE_PATH):
        return await _image_from_local_path(hass, image_path)
    if image_url := call.data.get(ATTR_IMAGE_URL):
        return _validate_image_url(image_url)
    raise _service_error(
        "image_source_required",
        "Select a camera entity, image file, local image path, or image URL",
    )


async def _audio_from_local_path(
    hass: HomeAssistant, audio_path: str
) -> tuple[bytes, str]:
    """Return audio bytes and filename from an allowlisted local path."""
    allowed = await hass.async_add_executor_job(
        hass.config.is_allowed_path,
        audio_path,
    )
    if not allowed:
        raise _service_error(
            "local_audio_path_not_allowed",
            "Local audio path is not in Home Assistant's allowlist_external_dirs",
        )
    path = Path(audio_path)
    if not await hass.async_add_executor_job(path.is_file):
        raise _service_error(
            "local_audio_not_found",
            f"Local audio file not found: {audio_path}",
            path=audio_path,
        )
    content_type = mimetypes.guess_type(audio_path)[0] or "audio/wav"
    if not content_type.startswith("audio/"):
        raise _service_error(
            "local_file_not_audio",
            f"Local file is not audio: {audio_path}",
            path=audio_path,
        )
    size = await hass.async_add_executor_job(lambda: path.stat().st_size)
    _ensure_size_limit(size, MAX_AUDIO_BYTES, "audio_too_large", "audio")
    content = await hass.async_add_executor_job(path.read_bytes)
    return content, path.name


async def _audio_from_media_source(
    hass: HomeAssistant,
    audio_file: str,
) -> tuple[bytes, str]:
    """Return audio bytes and filename from Home Assistant media or local path."""
    if not is_media_source_id(audio_file):
        return await _audio_from_local_path(hass, audio_file)
    try:
        media = await async_resolve_media(hass, audio_file, None)
    except Unresolvable as err:
        raise _service_error(
            "audio_media_unresolvable",
            f"Could not resolve audio file: {audio_file}",
            path=audio_file,
        ) from err
    if media.path is None:
        raise _service_error(
            "audio_media_local_required",
            "Selected audio must resolve to a local file",
        )
    if media.mime_type and not media.mime_type.startswith("audio/"):
        raise _service_error(
            "selected_media_not_audio",
            f"Selected media is not audio: {audio_file}",
            path=audio_file,
        )
    return await _audio_from_local_path(hass, str(media.path))


async def _audio_from_call(hass: HomeAssistant, call: ServiceCall) -> tuple[bytes, str]:
    """Resolve the audio source selected for a speech-to-text action."""
    if audio_file := call.data.get(ATTR_AUDIO_FILE):
        return await _audio_from_media_source(hass, audio_file)
    if audio_path := call.data.get(ATTR_AUDIO_PATH):
        return await _audio_from_local_path(hass, audio_path)
    raise _service_error(
        "audio_source_required",
        "Select an audio file or local audio path",
    )


def _handle_generate_text(
    hass: HomeAssistant, *, use_service_schema: bool = True
) -> ServiceHandler:
    """Build the generate_text service handler."""

    async def handler(call: ServiceCall) -> ServiceResponse:
        entry, runtime = await _runtime_from_call(hass, call, FEATURE_TEXT_GENERATION)
        _ensure_feature(runtime, GroqFeature.TEXT_GENERATION)
        service_data = _service_from_call(entry, runtime, call, FEATURE_TEXT_GENERATION)
        model = _service_value(call, service_data, ATTR_MODEL, DEFAULT_TEXT_MODEL)
        _ensure_model(
            runtime,
            model,
            GroqFeature.TEXT_GENERATION,
            hass=hass,
            entry=entry,
            service_data=service_data,
        )
        if _reasoning_requested(call.data, service_data):
            _ensure_model(
                runtime,
                model,
                GroqFeature.REASONING,
                hass=hass,
                entry=entry,
                service_data=service_data,
            )

        schema = call.data.get(ATTR_SCHEMA)
        if (
            use_service_schema
            and schema is None
            and service_data.get(CONF_STRUCTURED_OUTPUTS)
        ):
            schema = service_data.get(CONF_SCHEMA)
        request: TextGenerationRequest
        if schema:
            # generate_text doubles as the ergonomic entry point for structured
            # outputs when the selected service has a schema configured.
            _ensure_model(
                runtime,
                model,
                GroqFeature.STRUCTURED_OUTPUTS,
                hass=hass,
                entry=entry,
                service_data=service_data,
            )
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
                service_id=service_data.get(UNIQUE_ID),
                protect_free_tier=_service_protect_free_tier(service_data),
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
                service_id=service_data.get(UNIQUE_ID),
                protect_free_tier=_service_protect_free_tier(service_data),
            )
        _ensure_completion_token_limit(runtime, request)
        _ensure_request_body_features(runtime, request)
        if error := request_context_window_error(runtime.model_registry, request):
            raise _service_error("request_invalid", error, message=error)
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


def _handle_generate_structured(hass: HomeAssistant) -> ServiceHandler:
    """Build a generic text-output handler kept for service compatibility."""
    return _handle_generate_text(hass, use_service_schema=False)


def _handle_analyze_image(hass: HomeAssistant) -> ServiceHandler:
    """Build the analyze_image service handler."""

    async def handler(call: ServiceCall) -> ServiceResponse:
        entry, runtime = await _runtime_from_call(hass, call, FEATURE_IMAGE_RECOGNITION)
        _ensure_feature(runtime, GroqFeature.VISION)
        service_data = _service_from_call(
            entry, runtime, call, FEATURE_IMAGE_RECOGNITION
        )
        model = _service_value(call, service_data, ATTR_MODEL, DEFAULT_VISION_MODEL)
        _ensure_model(
            runtime,
            model,
            GroqFeature.VISION,
            hass=hass,
            entry=entry,
            service_data=service_data,
        )
        request = VisionRequest(
            prompt=call.data[ATTR_PROMPT],
            model=model,
            system_prompt=_service_value(call, service_data, ATTR_SYSTEM_PROMPT),
            image_url=await _image_url_from_call(hass, call),
            service_id=service_data.get(UNIQUE_ID),
            protect_free_tier=_service_protect_free_tier(service_data),
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


def _handle_extract_text_from_image(hass: HomeAssistant) -> ServiceHandler:
    """Build the extract_text_from_image service handler."""

    async def handler(call: ServiceCall) -> ServiceResponse:
        entry, runtime = await _runtime_from_call(hass, call, FEATURE_IMAGE_RECOGNITION)
        _ensure_feature(runtime, GroqFeature.VISION)
        service_data = _service_from_call(
            entry, runtime, call, FEATURE_IMAGE_RECOGNITION
        )
        model = _service_value(call, service_data, ATTR_MODEL, DEFAULT_VISION_MODEL)
        _ensure_model(
            runtime,
            model,
            GroqFeature.OCR,
            hass=hass,
            entry=entry,
            service_data=service_data,
        )
        request = VisionRequest(
            prompt=call.data[ATTR_PROMPT],
            model=model,
            system_prompt=_service_value(call, service_data, ATTR_SYSTEM_PROMPT),
            image_url=await _image_url_from_call(hass, call),
            service_id=service_data.get(UNIQUE_ID),
            protect_free_tier=_service_protect_free_tier(service_data),
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


def _handle_transcribe_audio(hass: HomeAssistant) -> ServiceHandler:
    """Build the transcribe_audio service handler."""

    async def handler(call: ServiceCall) -> ServiceResponse:
        entry, runtime = await _runtime_from_call(hass, call, FEATURE_SPEECH_TO_TEXT)
        _ensure_feature(runtime, GroqFeature.SPEECH_TO_TEXT)
        service_data = _service_from_call(entry, runtime, call, FEATURE_SPEECH_TO_TEXT)
        model = _service_value(call, service_data, ATTR_MODEL, DEFAULT_STT_MODEL)
        _ensure_model(
            runtime,
            model,
            GroqFeature.SPEECH_TO_TEXT,
            hass=hass,
            entry=entry,
            service_data=service_data,
        )
        audio, filename = await _audio_from_call(hass, call)
        text = await runtime.client.async_transcribe_audio(
            audio=audio,
            filename=filename,
            model=model,
            language=_service_value(
                call,
                service_data,
                ATTR_LANGUAGE,
                service_data.get(CONF_LANGUAGE, DEFAULT_STT_LANGUAGE),
            ),
            prompt=call.data.get(ATTR_PROMPT),
            service_id=service_data.get(UNIQUE_ID),
            protect_free_tier=_service_protect_free_tier(service_data),
        )
        return {
            "text": text,
            "model": model,
            "language": _service_value(
                call,
                service_data,
                ATTR_LANGUAGE,
                service_data.get(CONF_LANGUAGE, DEFAULT_STT_LANGUAGE),
            ),
            "filename": filename,
        }

    return handler


def _entry_label(entry: ConfigEntry) -> str:
    """Return a stable user-facing label for a Groq account."""
    return (
        getattr(entry, "title", None)
        or entry.data.get(CONF_NAME)
        or entry.entry_id
        or "Groq"
    )


def _service_label(entry: ConfigEntry, service_data: dict[str, Any]) -> str:
    """Return a user-facing option label for a configured Groq service."""
    service_name = service_data.get(CONF_NAME) or service_data.get(UNIQUE_ID)
    model = service_data.get(ATTR_MODEL)
    label = f"{_entry_label(entry)} - {service_name}"
    return f"{label} ({model})" if model else label


def _service_options(
    hass: HomeAssistant,
    service_type: str,
    *,
    exclude_entry_id: str | None = None,
) -> list[dict[str, str]]:
    """Return select options for Groq service subentries of a feature type."""
    options: list[dict[str, str]] = []
    for entry in _loaded_entries(
        hass,
        exclude_entry_id=exclude_entry_id,
        include_setup=True,
    ):
        runtime = getattr(entry, "runtime_data", None)
        for service_data in _service_subentries(entry, runtime, service_type):
            if service_id := service_data.get(UNIQUE_ID):
                options.append(
                    {
                        "label": _service_label(entry, service_data),
                        "value": service_id,
                    }
                )
    return sorted(options, key=lambda option: option["label"])


def _apply_service_options(
    description: dict[str, Any],
    options: list[dict[str, str]],
) -> dict[str, Any]:
    """Return a service description with a populated Groq Service dropdown."""
    updated = deepcopy(description)
    field = updated.get("fields", {}).get(ATTR_SERVICE_ID)
    if not field:
        return updated
    field["description"] = (
        "Configured Groq service to use. Choose from the loaded Groq services, "
        "or enter a service ID manually for YAML/backward compatibility. Leave "
        "empty when the selected account has only one matching service."
    )
    field["selector"] = {
        "select": {
            "custom_value": True,
            "options": options,
        }
    }
    return updated


async def async_update_service_descriptions(
    hass: HomeAssistant,
    *,
    exclude_entry_id: str | None = None,
) -> None:
    """Populate Groq action descriptions with currently configured services."""
    if not hasattr(hass, "services") or not hasattr(hass.services, "supports_response"):
        return
    if hasattr(hass, "async_add_executor_job"):
        descriptions = await hass.async_add_executor_job(load_yaml, _SERVICES_YAML)
    else:
        descriptions = load_yaml(_SERVICES_YAML)
    if not isinstance(descriptions, dict):
        return
    for service_name, service_type in _SERVICE_FIELD_TYPES.items():
        description = descriptions.get(service_name)
        if not isinstance(description, dict):
            continue
        service_helper.async_set_service_schema(
            hass,
            DOMAIN,
            service_name,
            _apply_service_options(
                description,
                _service_options(
                    hass,
                    service_type,
                    exclude_entry_id=exclude_entry_id,
                ),
            ),
        )


def _handle_clear_cache(hass: HomeAssistant) -> ServiceHandler:
    """Build the clear_cache service handler."""

    async def handler(call: ServiceCall) -> ServiceResponse:
        _entry, runtime = await _runtime_from_call(hass, call)
        _ensure_feature(runtime, GroqFeature.PROMPT_CACHING)
        return {"cleared": runtime.prompt_cache.clear()}

    return handler


def _handle_list_models(hass: HomeAssistant) -> ServiceHandler:
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
        await async_update_service_descriptions(hass)
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
        SERVICE_TRANSCRIBE_AUDIO,
        _handle_transcribe_audio(hass),
        schema=TRANSCRIBE_AUDIO_SCHEMA,
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
    await async_update_service_descriptions(hass)


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
        SERVICE_TRANSCRIBE_AUDIO,
        SERVICE_CLEAR_CACHE,
        SERVICE_LIST_MODELS,
    ):
        hass.services.async_remove(DOMAIN, service)


async_setup_services = async_register_services
async_unload_services = async_unregister_services
