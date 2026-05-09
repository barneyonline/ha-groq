"""Shared helpers for Groq text generation entities."""

from __future__ import annotations

import re
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry

from .const import (
    CONF_API_KEY,
    CONF_INCLUDE_REASONING,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_NAME,
    CONF_PROMPT_CACHING,
    CONF_REASONING_EFFORT,
    CONF_REASONING_FORMAT,
    CONF_REQUEST_BODY_OPTIONS,
    CONF_SCHEMA,
    CONF_SCHEMA_NAME,
    CONF_SEED,
    CONF_SERVICE_TIER,
    CONF_STOP,
    CONF_STREAM,
    CONF_STRICT,
    CONF_STRUCTURED_OUTPUTS,
    CONF_SYSTEM_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_SYSTEM_PROMPT,
    FEATURE_TEXT_GENERATION,
    PROMPT_CACHING_MODELS,
    REASONING_MODELS,
    TEXT_MODELS,
    UNIQUE_ID,
)
from .subentries import service_data_for_type

_SCHEMA_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def entry_value(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Return the effective value for a text generation service."""
    if key in service_data:
        return service_data[key]
    return config_entry.options.get(key, config_entry.data.get(key, default))


def text_generation_service_data(config_entry: ConfigEntry) -> list[dict[str, Any]]:
    """Return configured text generation service subentries."""
    return service_data_for_type(config_entry, FEATURE_TEXT_GENERATION)


def service_name(config_entry: ConfigEntry, service_data: dict[str, Any]) -> str:
    """Return the user-facing service name."""
    return str(
        entry_value(
            config_entry,
            service_data,
            CONF_NAME,
            "Groq Text Generation",
        )
    )


def service_model(config_entry: ConfigEntry, service_data: dict[str, Any]) -> str:
    """Return the configured text generation model."""
    return str(entry_value(config_entry, service_data, CONF_MODEL, TEXT_MODELS[0]))


def service_api_key(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
) -> str | None:
    """Return the optional service-specific API key."""
    value = service_data.get(CONF_API_KEY)
    if not value:
        return None
    return str(value)


def service_system_prompt(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
) -> str:
    """Return the configured Home Assistant system prompt."""
    return str(
        entry_value(
            config_entry,
            service_data,
            CONF_SYSTEM_PROMPT,
            DEFAULT_SYSTEM_PROMPT,
        )
        or DEFAULT_SYSTEM_PROMPT
    )


def service_temperature(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
) -> float | None:
    """Return the configured text generation temperature."""
    value = entry_value(config_entry, service_data, CONF_TEMPERATURE)
    if value is None:
        return None
    return float(value)


def service_max_tokens(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
) -> int | None:
    """Return the configured max completion token limit."""
    value = entry_value(config_entry, service_data, CONF_MAX_TOKENS)
    if value in (None, ""):
        return None
    return int(value)


def service_top_p(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
) -> float | None:
    """Return the configured top_p nucleus sampling value."""
    value = entry_value(config_entry, service_data, CONF_TOP_P)
    if value in (None, ""):
        return None
    return float(value)


def service_stop(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
) -> str | list[str] | None:
    """Return configured stop sequence data."""
    value = entry_value(config_entry, service_data, CONF_STOP)
    if not value:
        return None
    if isinstance(value, list):
        return [str(item) for item in value if item]
    lines = [line for line in str(value).splitlines() if line]
    if len(lines) > 1:
        return lines
    return lines[0] if lines else None


def service_seed(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
) -> int | None:
    """Return the configured deterministic sampling seed."""
    value = entry_value(config_entry, service_data, CONF_SEED)
    if value in (None, ""):
        return None
    return int(value)


def service_service_tier(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
) -> str | None:
    """Return the configured Groq service tier."""
    value = entry_value(config_entry, service_data, CONF_SERVICE_TIER)
    return str(value) if value else None


def service_reasoning_effort(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
) -> str | None:
    """Return the configured reasoning effort."""
    value = entry_value(config_entry, service_data, CONF_REASONING_EFFORT)
    return str(value) if value else None


def service_reasoning_format(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
) -> str | None:
    """Return the configured reasoning format."""
    value = entry_value(config_entry, service_data, CONF_REASONING_FORMAT)
    return str(value) if value else None


def service_include_reasoning(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
) -> bool | None:
    """Return whether raw reasoning should be included."""
    value = entry_value(config_entry, service_data, CONF_INCLUDE_REASONING)
    return True if value else None


def service_stream(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
) -> bool:
    """Return whether this service should stream Assist responses."""
    return bool(entry_value(config_entry, service_data, CONF_STREAM, False))


def service_prompt_caching(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
) -> bool:
    """Return whether Groq prompt caching is enabled for the model."""
    return bool(entry_value(config_entry, service_data, CONF_PROMPT_CACHING, False))


def service_request_body_options(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
) -> dict[str, Any] | None:
    """Return advanced passthrough chat completion body options."""
    value = entry_value(config_entry, service_data, CONF_REQUEST_BODY_OPTIONS)
    if not value:
        return None
    return dict(value)


def is_reasoning_model(model: str) -> bool:
    """Return whether the selected model supports Groq reasoning options."""
    return model in REASONING_MODELS


def is_prompt_caching_model(model: str) -> bool:
    """Return whether the selected model supports Groq prompt caching."""
    return model in PROMPT_CACHING_MODELS


def service_structured_outputs(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
) -> bool:
    """Return whether the service should request structured outputs."""
    return bool(entry_value(config_entry, service_data, CONF_STRUCTURED_OUTPUTS, False))


def service_schema(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the configured JSON schema for structured outputs."""
    schema = entry_value(config_entry, service_data, CONF_SCHEMA, None)
    if not schema:
        return None
    return dict(schema)


def service_schema_name(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
    default: str = "response",
) -> str:
    """Return a Groq-compatible schema name."""
    value = str(entry_value(config_entry, service_data, CONF_SCHEMA_NAME, default))
    value = _SCHEMA_NAME_RE.sub("_", value).strip("_")
    return value or default


def service_strict(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
) -> bool:
    """Return whether strict structured output mode is enabled."""
    return bool(entry_value(config_entry, service_data, CONF_STRICT, False))


def service_unique_id(config_entry: ConfigEntry, service_data: dict[str, Any]) -> str:
    """Return a stable service unique id."""
    return str(
        service_data.get(UNIQUE_ID)
        or getattr(config_entry, "unique_id", None)
        or config_entry.entry_id
    )


def selector_to_json_schema(validator: Any) -> dict[str, Any]:
    """Convert a Home Assistant selector or simple validator to JSON Schema."""
    selector_type = getattr(validator, "selector_type", None)
    config = getattr(validator, "config", {}) or {}

    # AI tasks can provide voluptuous structures with HA selectors. Convert only
    # the selector shapes this integration can represent safely as JSON Schema.
    if selector_type == "text":
        return {"type": "string"}
    if selector_type == "boolean":
        return {"type": "boolean"}
    if selector_type == "number":
        schema: dict[str, Any] = {"type": "number"}
        if "min" in config:
            schema["minimum"] = config["min"]
        if "max" in config:
            schema["maximum"] = config["max"]
        return schema
    if selector_type == "select":
        options = config.get("options", [])
        values = [
            option.get("value") if isinstance(option, dict) else option
            for option in options
        ]
        item_schema: dict[str, Any] = {"type": "string"}
        if values:
            item_schema["enum"] = values
        if config.get("multiple"):
            return {"type": "array", "items": item_schema}
        return item_schema
    if selector_type == "object":
        return {"type": "object"}

    if validator is str:
        return {"type": "string"}
    if validator is bool:
        return {"type": "boolean"}
    if validator is int:
        return {"type": "integer"}
    if validator is float:
        return {"type": "number"}
    if isinstance(validator, vol.Schema):
        return voluptuous_schema_to_json_schema(validator)
    return {}


def voluptuous_schema_to_json_schema(schema: vol.Schema) -> dict[str, Any]:
    """Convert a simple Home Assistant AI task structure to JSON Schema."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    schema_data = getattr(schema, "schema", {})

    if not isinstance(schema_data, dict):
        return {}

    for marker, validator in schema_data.items():
        name = str(getattr(marker, "schema", marker))
        field_schema = selector_to_json_schema(validator)
        description = getattr(marker, "description", None)
        if description:
            field_schema = {**field_schema, "description": description}
        properties[name] = field_schema
        # Groq strict structured outputs need an explicit required list rather
        # than relying on voluptuous marker objects.
        if isinstance(marker, vol.Required):
            required.append(name)

    json_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        json_schema["required"] = required
    return json_schema
