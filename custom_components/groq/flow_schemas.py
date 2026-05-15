"""Voluptuous schemas used by the Groq config and subentry flows."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.helpers.selector import selector

from .const import (
    CONF_ADVANCED_OPTIONS,
    CONF_API_KEY,
    CONF_ENABLED_FEATURES,
    CONF_INCLUDE_REASONING,
    CONF_LANGUAGE,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_NAME,
    CONF_NORMALIZE_AUDIO,
    CONF_PROMPT_CACHING,
    CONF_PROTECT_FREE_TIER,
    CONF_REASONING_EFFORT,
    CONF_REASONING_FORMAT,
    CONF_REQUEST_BODY_OPTIONS,
    CONF_RESPONSE_FORMAT,
    CONF_SCHEMA,
    CONF_SCHEMA_NAME,
    CONF_SEED,
    CONF_SERVICE_TYPE,
    CONF_SERVICE_TIER,
    CONF_STOP,
    CONF_STREAM,
    CONF_STRICT,
    CONF_STRUCTURED_OUTPUTS,
    CONF_SYSTEM_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_VOCAL_DIRECTIONS,
    CONF_VOICE,
    DEFAULT_MODEL,
    DEFAULT_PROTECT_FREE_TIER,
    DEFAULT_STT_LANGUAGE,
    DEFAULT_STT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEXT_MODEL,
    DEFAULT_TEXT_TEMPERATURE,
    DEFAULT_VISION_MODEL,
    DEFAULT_VOICE,
    FEATURE_LABELS,
    MODELS,
    REASONING_EFFORT_OPTIONS,
    REASONING_FORMAT_OPTIONS,
    SERVICE_TIER_OPTIONS,
    SETUP_FEATURES,
    STT_LANGUAGE_OPTIONS,
    STT_LANGUAGES,
    STT_MODELS,
    SUPPORTED_FEATURES,
    TEXT_MODELS,
    VISION_MODELS,
    VOCAL_DIRECTION_OPTIONS,
    voice_options_for_model,
)
from .feature_registry import GroqFeature
from .model_registry import GroqCapability, GroqModelRegistry
from .text_generation import request_body_options_validation_error


def _model_default(
    values: dict[str, Any],
    key: str,
    default: str,
    options: list[str],
) -> str:
    """Return a selector default that exists in the current option set."""
    if not options:
        return default
    configured = values.get(key)
    if configured in options:
        return configured
    if default in options:
        return default
    return options[0]


def _supports_model_option(
    registry: GroqModelRegistry | None,
    model: str,
    feature: GroqFeature,
) -> bool:
    """Return whether a model supports an optional Groq feature."""
    if not model:
        return True
    active_registry = registry or GroqModelRegistry()
    return active_registry.supports(model, feature)


def _model_completion_token_limit(
    model: str,
    registry: GroqModelRegistry | None = None,
) -> int | None:
    """Return the configured completion-token ceiling for a model."""
    active_registry = registry or GroqModelRegistry()
    return active_registry.completion_token_limit(model)


def _max_tokens_selector_config(
    model: str,
    registry: GroqModelRegistry | None = None,
) -> dict[str, Any]:
    """Return a number selector config capped to the selected model."""
    config: dict[str, Any] = {"min": 1, "step": 1, "mode": "box"}
    if limit := _model_completion_token_limit(model, registry):
        config["max"] = limit
    return config


def _requested_max_completion_tokens(data: dict[str, Any]) -> list[int]:
    """Return max completion token values requested by dedicated or raw options."""
    values = [data.get(CONF_MAX_TOKENS)]
    request_body_options = data.get(CONF_REQUEST_BODY_OPTIONS)
    if isinstance(request_body_options, dict):
        values.extend(
            (
                request_body_options.get("max_completion_tokens"),
                request_body_options.get("max_tokens"),
            )
        )
    tokens: list[int] = []
    for value in values:
        if value in (None, ""):
            continue
        if not isinstance(value, str | int):
            continue
        try:
            tokens.append(int(value))
        except (TypeError, ValueError):
            continue
    return tokens


def _clamp_max_completion_tokens(
    data: dict[str, Any],
    model: str,
    registry: GroqModelRegistry | None = None,
) -> None:
    """Clamp stored token ceilings to the selected model limit in place."""
    limit = _model_completion_token_limit(model, registry)
    if limit is None:
        return
    if data.get(CONF_MAX_TOKENS) not in (None, ""):
        try:
            data[CONF_MAX_TOKENS] = min(int(data[CONF_MAX_TOKENS]), limit)
        except (TypeError, ValueError):
            data.pop(CONF_MAX_TOKENS, None)
    request_body_options = data.get(CONF_REQUEST_BODY_OPTIONS)
    if not isinstance(request_body_options, dict):
        return
    for key in ("max_completion_tokens", "max_tokens"):
        if request_body_options.get(key) in (None, ""):
            continue
        try:
            request_body_options[key] = min(int(request_body_options[key]), limit)
        except (TypeError, ValueError):
            request_body_options.pop(key, None)


def _response_format_requests_structured_outputs(value: Any) -> bool:
    """Return whether a response_format value requests structured output."""
    if value in (None, ""):
        return False
    if isinstance(value, dict):
        value = value.get("type")
    return value in {"json_object", "json_schema"}


def sanitize_text_generation_service_data(
    user_input: dict[str, Any],
    model_registry: GroqModelRegistry | None = None,
) -> dict[str, Any]:
    """Remove hidden model-scoped options that no longer fit the selected model."""
    data = dict(user_input)
    model = str(data.get(CONF_MODEL, ""))
    request_body_options = data.get(CONF_REQUEST_BODY_OPTIONS)
    if isinstance(request_body_options, dict):
        request_body_options = dict(request_body_options)
        data[CONF_REQUEST_BODY_OPTIONS] = request_body_options

    if not _supports_model_option(model_registry, model, GroqFeature.REASONING):
        for key in (
            CONF_REASONING_EFFORT,
            CONF_REASONING_FORMAT,
            CONF_INCLUDE_REASONING,
        ):
            data.pop(key, None)
        if isinstance(request_body_options, dict):
            for key in ("reasoning_effort", "reasoning_format", "include_reasoning"):
                request_body_options.pop(key, None)

    if not _supports_model_option(model_registry, model, GroqFeature.PROMPT_CACHING):
        data.pop(CONF_PROMPT_CACHING, None)

    if not _supports_model_option(
        model_registry,
        model,
        GroqFeature.STRUCTURED_OUTPUTS,
    ):
        for key in (
            CONF_STRUCTURED_OUTPUTS,
            CONF_SCHEMA,
            CONF_SCHEMA_NAME,
            CONF_STRICT,
        ):
            data.pop(key, None)
        if isinstance(request_body_options, dict) and (
            _response_format_requests_structured_outputs(
                request_body_options.get("response_format")
            )
        ):
            request_body_options.pop("response_format", None)

    _clamp_max_completion_tokens(data, model, model_registry)
    if isinstance(request_body_options, dict) and not request_body_options:
        data.pop(CONF_REQUEST_BODY_OPTIONS, None)
    return data


def text_generation_model_capability_summary(
    model: str,
    registry: GroqModelRegistry | None = None,
) -> str:
    """Return a short user-facing capability summary for a text model."""
    active_registry = registry or GroqModelRegistry()
    model_data = active_registry.get(model)
    capabilities = model_data.capabilities if model_data else frozenset()
    limit = _model_completion_token_limit(model, active_registry)
    supported: list[str] = []
    unsupported: list[str] = []
    supported.append("Assist")
    if GroqCapability.STRUCTURED_OUTPUTS in capabilities:
        supported.append("data generation tasks")
        supported.append("structured outputs")
    else:
        unsupported.append("data generation tasks")
    if GroqCapability.REASONING in capabilities:
        supported.append("reasoning")
    if GroqCapability.PROMPT_CACHING in capabilities:
        supported.append("local response caching")
    if GroqCapability.COMPOUND in capabilities:
        supported.append("Groq Compound tools")
    if GroqCapability.VISION in capabilities:
        supported.append("image input")

    summary = f"Supported: {', '.join(supported)}."
    if unsupported:
        summary = f"{summary} Not supported: {', '.join(unsupported)}."
    if model_data and model_data.context_window:
        summary = f"{summary} Context window: {model_data.context_window:,} tokens."
    if limit:
        summary = f"{summary} Max completion: {limit:,} tokens."
    return summary


def text_generation_model_select_options(
    models: list[str],
    registry: GroqModelRegistry | None = None,
) -> list[dict[str, str]]:
    """Return text generation model options with capability labels."""
    return [
        {
            "value": model,
            "label": f"{model} - {text_generation_model_capability_summary(model, registry)}",
        }
        for model in models
    ]


def api_key_selector():
    """Return a password-style selector for Groq API keys."""
    return selector({"text": {"type": "password"}})


def setup_schema() -> vol.Schema:
    """Return the account-level setup schema."""
    return vol.Schema(
        {
            vol.Required(CONF_NAME, default="Groq"): str,
            vol.Required(CONF_API_KEY): api_key_selector(),
        }
    )


def service_type_schema() -> vol.Schema:
    """Return the Groq service type selector schema."""
    return vol.Schema(
        {
            vol.Required(CONF_SERVICE_TYPE): selector(
                {
                    "select": {
                        "options": [
                            {"value": feature, "label": FEATURE_LABELS[feature]}
                            for feature in SETUP_FEATURES
                        ],
                        "mode": "list",
                    }
                }
            )
        }
    )


def _protect_free_tier_field(values: dict[str, Any]) -> tuple[Any, Any]:
    """Return the shared per-service free-tier protection schema field."""
    return (
        vol.Optional(
            CONF_PROTECT_FREE_TIER,
            default=values.get(CONF_PROTECT_FREE_TIER, DEFAULT_PROTECT_FREE_TIER),
        ),
        selector({"boolean": {}}),
    )


def speech_to_text_schema(
    user_input: dict[str, Any] | None = None,
    model_options: list[str] | None = None,
    default_language: str | None = None,
) -> vol.Schema:
    """Return the speech-to-text service schema."""
    values = user_input or {}
    models = model_options or STT_MODELS
    protect_field, protect_selector = _protect_free_tier_field(values)
    language_default = _model_default(
        values,
        CONF_LANGUAGE,
        default_language or DEFAULT_STT_LANGUAGE,
        STT_LANGUAGES,
    )
    return vol.Schema(
        {
            vol.Required(
                CONF_NAME,
                default=values.get(CONF_NAME, "Speech-to-Text"),
            ): str,
            vol.Required(
                CONF_MODEL,
                default=_model_default(values, CONF_MODEL, DEFAULT_STT_MODEL, models),
            ): selector({"select": {"options": models}}),
            vol.Required(
                CONF_LANGUAGE,
                default=language_default,
            ): selector({"select": {"options": STT_LANGUAGE_OPTIONS}}),
            protect_field: protect_selector,
        }
    )


def text_to_speech_schema(
    user_input: dict[str, Any] | None = None,
    model_options: list[str] | None = None,
    voice_options: list[str] | None = None,
    *,
    clear_voice: bool = False,
) -> vol.Schema:
    """Return the text-to-speech service schema."""
    values = user_input or {}
    models = model_options or MODELS
    protect_field, protect_selector = _protect_free_tier_field(values)
    selected_model = _model_default(values, CONF_MODEL, DEFAULT_MODEL, models)
    voices = voice_options or voice_options_for_model(selected_model)
    voice_field = (
        vol.Required(CONF_VOICE)
        if clear_voice
        else vol.Required(
            CONF_VOICE,
            default=_model_default(values, CONF_VOICE, DEFAULT_VOICE, voices),
        )
    )
    return vol.Schema(
        {
            vol.Required(
                CONF_NAME,
                default=values.get(CONF_NAME, "Text-to-Speech"),
            ): str,
            vol.Required(
                CONF_MODEL,
                default=selected_model,
            ): selector({"select": {"options": models}}),
            voice_field: selector({"select": {"options": voices}}),
            vol.Optional(
                CONF_VOCAL_DIRECTIONS,
                default=values.get(CONF_VOCAL_DIRECTIONS, ""),
            ): selector(
                {
                    "select": {
                        "options": VOCAL_DIRECTION_OPTIONS,
                        "custom_value": True,
                        "mode": "dropdown",
                    }
                }
            ),
            vol.Optional(
                CONF_NORMALIZE_AUDIO,
                default=values.get(CONF_NORMALIZE_AUDIO, False),
            ): selector({"boolean": {}}),
            protect_field: protect_selector,
        }
    )


def image_recognition_schema(
    user_input: dict[str, Any] | None = None,
    model_options: list[str] | None = None,
) -> vol.Schema:
    """Return the image recognition service schema."""
    values = user_input or {}
    models = model_options or VISION_MODELS
    protect_field, protect_selector = _protect_free_tier_field(values)
    return vol.Schema(
        {
            vol.Required(
                CONF_NAME,
                default=values.get(CONF_NAME, "Image Recognition"),
            ): str,
            vol.Required(
                CONF_MODEL,
                default=_model_default(
                    values, CONF_MODEL, DEFAULT_VISION_MODEL, models
                ),
            ): selector({"select": {"options": models}}),
            vol.Optional(
                CONF_SYSTEM_PROMPT,
                default=values.get(CONF_SYSTEM_PROMPT, ""),
            ): str,
            protect_field: protect_selector,
        }
    )


def text_generation_basic_schema(
    user_input: dict[str, Any] | None = None,
    model_options: list[str] | None = None,
    model_registry: GroqModelRegistry | None = None,
) -> vol.Schema:
    """Return the basic text generation service schema."""
    values = user_input or {}
    models = model_options or TEXT_MODELS
    protect_field, protect_selector = _protect_free_tier_field(values)
    return vol.Schema(
        {
            vol.Required(
                CONF_NAME,
                default=values.get(CONF_NAME, "Text Generation"),
            ): str,
            vol.Required(
                CONF_MODEL,
                default=_model_default(values, CONF_MODEL, DEFAULT_TEXT_MODEL, models),
            ): selector(
                {
                    "select": {
                        "options": text_generation_model_select_options(
                            models,
                            model_registry,
                        )
                    }
                }
            ),
            vol.Optional(
                CONF_SYSTEM_PROMPT,
                default=values.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT),
            ): selector({"text": {"multiline": True}}),
            vol.Optional(
                CONF_TEMPERATURE,
                default=values.get(CONF_TEMPERATURE, DEFAULT_TEXT_TEMPERATURE),
            ): selector({"number": {"min": 0, "max": 2, "step": 0.1, "mode": "box"}}),
            vol.Optional(
                CONF_ADVANCED_OPTIONS,
                default=values.get(CONF_ADVANCED_OPTIONS, False),
            ): selector({"boolean": {}}),
            protect_field: protect_selector,
        }
    )


def text_generation_advanced_schema(
    user_input: dict[str, Any] | None = None,
    model_registry: GroqModelRegistry | None = None,
) -> vol.Schema:
    """Return advanced text generation request options."""
    values = user_input or {}
    model = str(values.get(CONF_MODEL, ""))
    schema: dict[Any, Any] = {
        vol.Optional(CONF_MAX_TOKENS, default=values.get(CONF_MAX_TOKENS)): (
            selector({"number": _max_tokens_selector_config(model, model_registry)})
        ),
        vol.Optional(CONF_TOP_P, default=values.get(CONF_TOP_P)): selector(
            {"number": {"min": 0, "max": 1, "step": 0.01, "mode": "box"}}
        ),
        vol.Optional(CONF_STOP, default=values.get(CONF_STOP)): selector(
            {"text": {"multiline": True}}
        ),
        vol.Optional(CONF_SEED, default=values.get(CONF_SEED)): selector(
            {"number": {"min": 0, "step": 1, "mode": "box"}}
        ),
        vol.Optional(
            CONF_SERVICE_TIER,
            default=values.get(CONF_SERVICE_TIER, ""),
        ): selector({"select": {"options": SERVICE_TIER_OPTIONS}}),
        vol.Optional(CONF_STREAM, default=values.get(CONF_STREAM, True)): selector(
            {"boolean": {}}
        ),
    }
    if _supports_model_option(model_registry, model, GroqFeature.REASONING):
        schema.update(
            {
                vol.Optional(
                    CONF_REASONING_EFFORT,
                    default=values.get(CONF_REASONING_EFFORT, ""),
                ): selector({"select": {"options": REASONING_EFFORT_OPTIONS}}),
                vol.Optional(
                    CONF_REASONING_FORMAT,
                    default=values.get(CONF_REASONING_FORMAT, ""),
                ): selector({"select": {"options": REASONING_FORMAT_OPTIONS}}),
                vol.Optional(
                    CONF_INCLUDE_REASONING,
                    default=values.get(CONF_INCLUDE_REASONING, False),
                ): selector({"boolean": {}}),
            }
        )
    if _supports_model_option(model_registry, model, GroqFeature.PROMPT_CACHING):
        schema[
            vol.Optional(
                CONF_PROMPT_CACHING,
                default=values.get(CONF_PROMPT_CACHING, False),
            )
        ] = selector({"boolean": {}})
    if _supports_model_option(model_registry, model, GroqFeature.STRUCTURED_OUTPUTS):
        schema.update(
            {
                vol.Optional(
                    CONF_STRUCTURED_OUTPUTS,
                    default=values.get(CONF_STRUCTURED_OUTPUTS, False),
                ): selector({"boolean": {}}),
                vol.Optional(
                    CONF_SCHEMA_NAME,
                    default=values.get(CONF_SCHEMA_NAME, "response"),
                ): str,
                vol.Optional(
                    CONF_SCHEMA, default=values.get(CONF_SCHEMA, {})
                ): selector({"object": {}}),
                vol.Optional(CONF_STRICT, default=values.get(CONF_STRICT, False)): (
                    selector({"boolean": {}})
                ),
            }
        )
    schema[
        vol.Optional(
            CONF_REQUEST_BODY_OPTIONS,
            default=values.get(CONF_REQUEST_BODY_OPTIONS, {}),
        )
    ] = selector({"object": {}})
    return vol.Schema(schema)


def entry_defaults(user_input: dict[str, Any]) -> dict[str, Any]:
    """Return normalized account-level setup data."""
    data = dict(user_input)
    if not data.get(CONF_NAME):
        data[CONF_NAME] = "Groq"
    data.pop(CONF_ENABLED_FEATURES, None)
    return data


async def validate_user_input(user_input: dict[str, Any]) -> None:
    """Validate account-level setup input."""
    if not user_input.get(CONF_API_KEY):
        raise ValueError("API key is required")
    if CONF_ENABLED_FEATURES in user_input:
        enabled_features = user_input[CONF_ENABLED_FEATURES]
        if isinstance(enabled_features, str):
            enabled_features = [enabled_features]
        try:
            invalid_features = set(enabled_features) - set(SUPPORTED_FEATURES)
        except TypeError as err:
            raise ValueError("Enabled features are invalid") from err
        if invalid_features:
            raise ValueError("Enabled features are invalid")


def clean_service_input(user_input: dict[str, Any]) -> dict[str, Any]:
    """Remove blank service fields before storing a subentry."""
    data = dict(user_input)
    data.pop(CONF_API_KEY, None)
    data.pop(CONF_RESPONSE_FORMAT, None)
    # Empty strings come back from optional selectors when the user leaves them
    # blank. Drop those values so service calls can fall back to integration
    # defaults instead of storing meaningless overrides.
    for key in (
        CONF_ADVANCED_OPTIONS,
        CONF_LANGUAGE,
        CONF_REASONING_EFFORT,
        CONF_REASONING_FORMAT,
        CONF_SERVICE_TIER,
        CONF_STOP,
    ):
        if data.get(key) in ("", None):
            data.pop(key, None)
    if not data.get(CONF_REQUEST_BODY_OPTIONS):
        data.pop(CONF_REQUEST_BODY_OPTIONS, None)
    if not data.get(CONF_SCHEMA):
        data.pop(CONF_SCHEMA, None)
    if not data.get(CONF_INCLUDE_REASONING):
        data.pop(CONF_INCLUDE_REASONING, None)
    return data


def validate_text_generation_input(
    user_input: dict[str, Any],
    model_registry: GroqModelRegistry | None = None,
) -> dict[str, str]:
    """Return validation errors for text generation options."""
    errors: dict[str, str] = {}
    model = str(user_input.get(CONF_MODEL, ""))
    # Reasoning, local response caching, and strict structured outputs are model-scoped
    # Groq features. Validate here so unsupported combinations fail in the setup
    # UI instead of later during Assist or service execution.
    has_reasoning_options = any(
        (
            user_input.get(CONF_REASONING_EFFORT),
            user_input.get(CONF_REASONING_FORMAT),
            user_input.get(CONF_INCLUDE_REASONING),
        )
    )
    if has_reasoning_options and not _supports_model_option(
        model_registry, model, GroqFeature.REASONING
    ):
        errors[CONF_MODEL] = "unsupported_reasoning_model"
    if user_input.get(CONF_PROMPT_CACHING) and not _supports_model_option(
        model_registry, model, GroqFeature.PROMPT_CACHING
    ):
        errors[CONF_MODEL] = "unsupported_prompt_caching_model"
    if user_input.get(CONF_STRUCTURED_OUTPUTS) and not _supports_model_option(
        model_registry, model, GroqFeature.STRUCTURED_OUTPUTS
    ):
        errors[CONF_MODEL] = "unsupported_structured_outputs_model"
    active_registry = model_registry or GroqModelRegistry()
    if body_error := request_body_options_validation_error(
        active_registry,
        model,
        user_input.get(CONF_REQUEST_BODY_OPTIONS),
    ):
        errors[CONF_REQUEST_BODY_OPTIONS] = body_error
    limit = _model_completion_token_limit(model, model_registry)
    if limit is not None:
        requested_max_tokens = _requested_max_completion_tokens(
            {CONF_MAX_TOKENS: user_input.get(CONF_MAX_TOKENS)}
        )
        requested_body_tokens = _requested_max_completion_tokens(
            {CONF_REQUEST_BODY_OPTIONS: user_input.get(CONF_REQUEST_BODY_OPTIONS)}
        )
        if any(value > limit for value in requested_max_tokens):
            errors[CONF_MAX_TOKENS] = "max_completion_tokens_exceeded"
        elif any(value > limit for value in requested_body_tokens):
            errors[CONF_REQUEST_BODY_OPTIONS] = "max_completion_tokens_exceeded"
    return errors
