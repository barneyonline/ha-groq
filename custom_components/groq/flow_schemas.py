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
    DEFAULT_RESPONSE_FORMAT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_VOICE,
    FEATURE_LABELS,
    MODELS,
    PROMPT_CACHING_MODELS,
    REASONING_EFFORT_OPTIONS,
    REASONING_FORMAT_OPTIONS,
    REASONING_MODELS,
    RESPONSE_FORMATS,
    SERVICE_TIER_OPTIONS,
    SETUP_FEATURES,
    STT_MODELS,
    STRUCTURED_OUTPUTS_MODELS,
    SUPPORTED_FEATURES,
    TEXT_MODELS,
    VISION_MODELS,
    VOCAL_DIRECTION_OPTIONS,
    VOICES,
)


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


def speech_to_text_schema(user_input: dict[str, Any] | None = None) -> vol.Schema:
    """Return the speech-to-text service schema."""
    values = user_input or {}
    return vol.Schema(
        {
            vol.Required(
                CONF_NAME,
                default=values.get(CONF_NAME, "Speech-to-Text"),
            ): str,
            vol.Optional(CONF_API_KEY): api_key_selector(),
            vol.Required(
                CONF_MODEL,
                default=values.get(CONF_MODEL, STT_MODELS[0]),
            ): selector({"select": {"options": STT_MODELS, "custom_value": True}}),
            vol.Optional(
                CONF_LANGUAGE,
                default=values.get(CONF_LANGUAGE, "en"),
            ): str,
        }
    )


def text_to_speech_schema(user_input: dict[str, Any] | None = None) -> vol.Schema:
    """Return the text-to-speech service schema."""
    values = user_input or {}
    return vol.Schema(
        {
            vol.Required(
                CONF_NAME,
                default=values.get(CONF_NAME, "Text-to-Speech"),
            ): str,
            vol.Optional(CONF_API_KEY): api_key_selector(),
            vol.Required(
                CONF_MODEL,
                default=values.get(CONF_MODEL, DEFAULT_MODEL),
            ): selector({"select": {"options": MODELS, "custom_value": True}}),
            vol.Required(
                CONF_VOICE,
                default=values.get(CONF_VOICE, DEFAULT_VOICE),
            ): selector({"select": {"options": VOICES, "custom_value": True}}),
            vol.Required(
                CONF_RESPONSE_FORMAT,
                default=values.get(CONF_RESPONSE_FORMAT, DEFAULT_RESPONSE_FORMAT),
            ): selector({"select": {"options": RESPONSE_FORMATS}}),
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
            vol.Optional(
                CONF_PROTECT_FREE_TIER,
                default=values.get(CONF_PROTECT_FREE_TIER, DEFAULT_PROTECT_FREE_TIER),
            ): selector({"boolean": {}}),
        }
    )


def image_recognition_schema(user_input: dict[str, Any] | None = None) -> vol.Schema:
    """Return the image recognition service schema."""
    values = user_input or {}
    return vol.Schema(
        {
            vol.Required(
                CONF_NAME,
                default=values.get(CONF_NAME, "Image Recognition"),
            ): str,
            vol.Optional(CONF_API_KEY): api_key_selector(),
            vol.Required(
                CONF_MODEL,
                default=values.get(CONF_MODEL, VISION_MODELS[0]),
            ): selector({"select": {"options": VISION_MODELS, "custom_value": True}}),
            vol.Optional(
                CONF_SYSTEM_PROMPT,
                default=values.get(CONF_SYSTEM_PROMPT, ""),
            ): str,
        }
    )


def text_generation_basic_schema(
    user_input: dict[str, Any] | None = None,
) -> vol.Schema:
    """Return the basic text generation service schema."""
    values = user_input or {}
    return vol.Schema(
        {
            vol.Required(
                CONF_NAME,
                default=values.get(CONF_NAME, "Text Generation"),
            ): str,
            vol.Optional(CONF_API_KEY): api_key_selector(),
            vol.Required(
                CONF_MODEL,
                default=values.get(CONF_MODEL, TEXT_MODELS[0]),
            ): selector({"select": {"options": TEXT_MODELS, "custom_value": True}}),
            vol.Optional(
                CONF_SYSTEM_PROMPT,
                default=values.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT),
            ): selector({"text": {"multiline": True}}),
            vol.Optional(
                CONF_TEMPERATURE,
                default=values.get(CONF_TEMPERATURE, 0.7),
            ): selector({"number": {"min": 0, "max": 2, "step": 0.1, "mode": "box"}}),
            vol.Optional(
                CONF_ADVANCED_OPTIONS,
                default=values.get(CONF_ADVANCED_OPTIONS, False),
            ): selector({"boolean": {}}),
        }
    )


def text_generation_advanced_schema(
    user_input: dict[str, Any] | None = None,
) -> vol.Schema:
    """Return advanced text generation request options."""
    values = user_input or {}
    return vol.Schema(
        {
            vol.Optional(CONF_MAX_TOKENS, default=values.get(CONF_MAX_TOKENS)): (
                selector({"number": {"min": 1, "step": 1, "mode": "box"}})
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
            vol.Optional(
                CONF_PROMPT_CACHING,
                default=values.get(CONF_PROMPT_CACHING, False),
            ): selector({"boolean": {}}),
            vol.Optional(
                CONF_STRUCTURED_OUTPUTS,
                default=values.get(CONF_STRUCTURED_OUTPUTS, False),
            ): selector({"boolean": {}}),
            vol.Optional(
                CONF_SCHEMA_NAME,
                default=values.get(CONF_SCHEMA_NAME, "response"),
            ): str,
            vol.Optional(CONF_SCHEMA, default=values.get(CONF_SCHEMA, {})): selector(
                {"object": {}}
            ),
            vol.Optional(CONF_STRICT, default=values.get(CONF_STRICT, False)): (
                selector({"boolean": {}})
            ),
            vol.Optional(
                CONF_REQUEST_BODY_OPTIONS,
                default=values.get(CONF_REQUEST_BODY_OPTIONS, {}),
            ): selector({"object": {}}),
        }
    )


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
    # Empty strings come back from optional selectors when the user leaves them
    # blank. Drop those values so service calls can fall back to integration
    # defaults instead of storing meaningless overrides.
    for key in (
        CONF_API_KEY,
        CONF_ADVANCED_OPTIONS,
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


def validate_text_generation_input(user_input: dict[str, Any]) -> dict[str, str]:
    """Return validation errors for text generation options."""
    errors: dict[str, str] = {}
    model = str(user_input.get(CONF_MODEL, ""))
    # Reasoning, prompt caching, and strict structured outputs are model-scoped
    # Groq features. Validate here so unsupported combinations fail in the setup
    # UI instead of later during Assist or service execution.
    has_reasoning_options = any(
        (
            user_input.get(CONF_REASONING_EFFORT),
            user_input.get(CONF_REASONING_FORMAT),
            user_input.get(CONF_INCLUDE_REASONING),
        )
    )
    if has_reasoning_options and model not in REASONING_MODELS:
        errors[CONF_MODEL] = "unsupported_reasoning_model"
    if user_input.get(CONF_PROMPT_CACHING) and model not in PROMPT_CACHING_MODELS:
        errors[CONF_MODEL] = "unsupported_prompt_caching_model"
    if (
        user_input.get(CONF_STRUCTURED_OUTPUTS)
        and model not in STRUCTURED_OUTPUTS_MODELS
    ):
        errors[CONF_MODEL] = "unsupported_structured_outputs_model"
    return errors
