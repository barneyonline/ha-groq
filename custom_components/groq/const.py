"""Constants for Groq custom component."""

from __future__ import annotations

from typing import Any

DOMAIN = "groq"
CONF_API_KEY = "api_key"
CONF_MODEL = "model"
CONF_INPUT = "input"
CONF_VOICE = "voice"
CONF_RESPONSE_FORMAT = "response_format"
CONF_VOCAL_DIRECTIONS = "vocal_directions"
CONF_URL = "url"
UNIQUE_ID = "unique_id"

CONF_BASE_URL = "base_url"
CONF_ADVANCED_OPTIONS = "advanced_options"
CONF_ENABLED_FEATURES = "enabled_features"
CONF_ENTRY_ID = "entry_id"
CONF_PROMPT = "prompt"
CONF_SYSTEM_PROMPT = "system_prompt"
CONF_TEMPERATURE = "temperature"
CONF_MAX_TOKENS = "max_tokens"
CONF_TOP_P = "top_p"
CONF_STOP = "stop"
CONF_SEED = "seed"
CONF_SERVICE_TIER = "service_tier"
CONF_REASONING_EFFORT = "reasoning_effort"
CONF_REASONING_FORMAT = "reasoning_format"
CONF_INCLUDE_REASONING = "include_reasoning"
CONF_PROMPT_CACHING = "prompt_caching"
CONF_STREAM = "stream"
CONF_REQUEST_BODY_OPTIONS = "request_body_options"
CONF_SCHEMA = "schema"
CONF_SCHEMA_NAME = "schema_name"
CONF_STRICT = "strict"
CONF_STRUCTURED_OUTPUTS = "structured_outputs"
CONF_IMAGE_URL = "image_url"
CONF_IMAGE_ENTITY_ID = "image_entity_id"
CONF_RESULT_FORMAT = "result_format"
CONF_NAME = "name"
CONF_SERVICE_TYPE = "service_type"
CONF_SUBENTRY_ID = "subentry_id"
CONF_LANGUAGE = "language"

FEATURE_TEXT_GENERATION = "text_generation"
FEATURE_SPEECH_TO_TEXT = "speech_to_text"
FEATURE_TEXT_TO_SPEECH = "text_to_speech"
FEATURE_OCR = "ocr"
FEATURE_IMAGE_RECOGNITION = "image_recognition"
FEATURE_REASONING = "reasoning"
FEATURE_STRUCTURED_OUTPUTS = "structured_outputs"
FEATURE_PROMPT_CACHING = "prompt_caching"

SUPPORTED_FEATURES = (
    FEATURE_TEXT_GENERATION,
    FEATURE_SPEECH_TO_TEXT,
    FEATURE_TEXT_TO_SPEECH,
    FEATURE_IMAGE_RECOGNITION,
)

DEFAULT_ENABLED_FEATURES = (FEATURE_TEXT_TO_SPEECH,)
SETUP_FEATURES = (
    FEATURE_TEXT_GENERATION,
    FEATURE_SPEECH_TO_TEXT,
    FEATURE_TEXT_TO_SPEECH,
    FEATURE_IMAGE_RECOGNITION,
)

FEATURE_LABELS = {
    FEATURE_TEXT_GENERATION: "Text Generation",
    FEATURE_SPEECH_TO_TEXT: "Speech-to-Text",
    FEATURE_TEXT_TO_SPEECH: "Text-to-Speech",
    FEATURE_IMAGE_RECOGNITION: "Image Recognition",
}

FEATURE_SELECT_OPTIONS = [
    {"value": feature, "label": FEATURE_LABELS[feature]}
    for feature in SUPPORTED_FEATURES
]

MODELS = [
    "canopylabs/orpheus-v1-english",
    "canopylabs/orpheus-arabic-saudi",
]
VOICES = [
    "autumn",
    "diana",
    "hannah",
    "austin",
    "daniel",
    "troy",
    "abdullah",
    "fahad",
    "sultan",
    "lulwa",
    "noura",
    "aisha",
]
ENGLISH_ORPHEUS_VOICES = [
    "autumn",
    "diana",
    "hannah",
    "austin",
    "daniel",
    "troy",
]
ARABIC_ORPHEUS_VOICES = [
    "abdullah",
    "fahad",
    "sultan",
    "lulwa",
    "noura",
    "aisha",
]
TTS_VOICES_BY_MODEL = {
    "canopylabs/orpheus-v1-english": ENGLISH_ORPHEUS_VOICES,
    "canopylabs/orpheus-arabic-saudi": ARABIC_ORPHEUS_VOICES,
}
DEFAULT_MODEL = MODELS[0]
DEFAULT_VOICE = VOICES[0]
RESPONSE_FORMATS = ["wav"]
DEFAULT_RESPONSE_FORMAT = RESPONSE_FORMATS[0]
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_TTS_URL = f"{DEFAULT_BASE_URL}/audio/speech"

TEXT_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-safeguard-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "groq/compound",
    "groq/compound-mini",
]
DEFAULT_TEXT_MODEL = "openai/gpt-oss-20b"
DEFAULT_TEXT_TEMPERATURE = 0.2

REASONING_MODELS = {
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-safeguard-20b",
    "qwen/qwen3-32b",
}
PROMPT_CACHING_MODELS = {
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-safeguard-20b",
}
STRUCTURED_OUTPUTS_MODELS = {
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-safeguard-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
}
COMPOUND_MODELS = {
    "groq/compound",
    "groq/compound-mini",
}
REASONING_EFFORT_OPTIONS = [
    {"value": "", "label": "Model default"},
    {"value": "none", "label": "None"},
    {"value": "default", "label": "Default"},
    {"value": "low", "label": "Low"},
    {"value": "medium", "label": "Medium"},
    {"value": "high", "label": "High"},
]
REASONING_FORMAT_OPTIONS = [
    {"value": "", "label": "Model default"},
    {"value": "hidden", "label": "Hidden"},
    {"value": "raw", "label": "Raw"},
    {"value": "parsed", "label": "Parsed"},
]
SERVICE_TIER_OPTIONS = [
    {"value": "", "label": "Groq default"},
    {"value": "auto", "label": "Auto"},
    {"value": "on_demand", "label": "On demand"},
    {"value": "flex", "label": "Flex"},
    {"value": "performance", "label": "Performance"},
]

STT_MODELS = [
    "whisper-large-v3-turbo",
    "whisper-large-v3",
]
DEFAULT_STT_MODEL = "whisper-large-v3-turbo"
STT_LANGUAGE_OPTIONS = [
    {"value": "en-US", "label": "English (United States)"},
    {"value": "en-GB", "label": "English (United Kingdom)"},
    {"value": "en", "label": "English"},
    {"value": "de-DE", "label": "German"},
    {"value": "es-ES", "label": "Spanish"},
    {"value": "fr-FR", "label": "French"},
    {"value": "it-IT", "label": "Italian"},
    {"value": "pt-PT", "label": "Portuguese"},
    {"value": "nl-NL", "label": "Dutch"},
    {"value": "ja-JP", "label": "Japanese"},
    {"value": "ko-KR", "label": "Korean"},
    {"value": "zh-CN", "label": "Chinese"},
]
STT_LANGUAGES = [option["value"] for option in STT_LANGUAGE_OPTIONS]
DEFAULT_STT_LANGUAGE = "en-US"

VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]
DEFAULT_VISION_MODEL = VISION_MODELS[0]

VOCAL_DIRECTION_OPTIONS = [
    {"value": "", "label": "None"},
    {"value": "cheerful", "label": "Cheerful"},
    {"value": "friendly", "label": "Friendly"},
    {"value": "warm", "label": "Warm"},
    {"value": "professional", "label": "Professional"},
    {"value": "calm", "label": "Calm"},
    {"value": "excited", "label": "Excited"},
    {"value": "whisper", "label": "Whisper"},
    {"value": "dramatic", "label": "Dramatic"},
]

CONF_NORMALIZE_AUDIO = "normalize_audio"
CONF_CACHE_SIZE = "cache_size"
CONF_PROTECT_FREE_TIER = "protect_free_tier"
DEFAULT_CACHE_SIZE = 256
DEFAULT_PROTECT_FREE_TIER = True
DEFAULT_SYSTEM_PROMPT = (
    "You are a voice assistant for Home Assistant. Answer in plain text. "
    "Keep it simple and to the point."
)

GROQ_FREE_TIER_LIMITS = {
    "canopylabs/orpheus-v1-english": {
        "requests_per_minute": 10,
        "requests_per_day": 100,
        "tokens_per_minute": 1200,
        "tokens_per_day": 3600,
    },
    "canopylabs/orpheus-arabic-saudi": {
        "requests_per_minute": 10,
        "requests_per_day": 100,
        "tokens_per_minute": 1200,
        "tokens_per_day": 3600,
    },
}


def normalize_enabled_features(
    enabled_features: Any,
    *,
    default: tuple[str, ...] = DEFAULT_ENABLED_FEATURES,
) -> list[str]:
    """Return known feature ids in stable order.

    Missing values are treated as the migration/default case. An explicit empty
    list remains empty so users can disable all optional feature surfaces.
    """
    if enabled_features is None:
        return list(default)

    if isinstance(enabled_features, str):
        requested = {enabled_features}
    else:
        try:
            requested = set(enabled_features)
        except TypeError:
            return list(default)

    return [feature for feature in SUPPORTED_FEATURES if feature in requested]


def enabled_features_from_entry(entry: Any) -> list[str]:
    """Return effective enabled features from config entry options/data."""
    if CONF_ENABLED_FEATURES in entry.options:
        return normalize_enabled_features(entry.options.get(CONF_ENABLED_FEATURES))
    if CONF_ENABLED_FEATURES in entry.data:
        return normalize_enabled_features(entry.data.get(CONF_ENABLED_FEATURES))

    enabled = {
        data.get(CONF_SERVICE_TYPE)
        for subentry in (getattr(entry, "subentries", None) or {}).values()
        if isinstance((data := getattr(subentry, "data", {})), dict)
    }
    if enabled:
        return [feature for feature in SUPPORTED_FEATURES if feature in enabled]

    if all(entry.data.get(key) for key in (CONF_URL, CONF_MODEL, CONF_VOICE)):
        return [FEATURE_TEXT_TO_SPEECH]
    return []


def voice_options_for_model(model: str | None) -> list[str]:
    """Return valid Orpheus voices for a TTS model."""
    if not model:
        return list(VOICES)
    if model in TTS_VOICES_BY_MODEL:
        return list(TTS_VOICES_BY_MODEL[model])
    model_id = model.lower()
    if "arabic" in model_id or "saudi" in model_id:
        return list(ARABIC_ORPHEUS_VOICES)
    if "orpheus" in model_id:
        return list(ENGLISH_ORPHEUS_VOICES)
    return list(VOICES)


def stt_language_default(language: str | None) -> str:
    """Return the closest configured STT language for a Home Assistant locale."""
    if not language:
        return DEFAULT_STT_LANGUAGE
    locale = language.replace("_", "-")
    if locale in STT_LANGUAGES:
        return locale
    base_language = locale.split("-", 1)[0]
    if base_language in STT_LANGUAGES:
        return base_language
    for supported_language in STT_LANGUAGES:
        if supported_language.split("-", 1)[0] == base_language:
            return supported_language
    return DEFAULT_STT_LANGUAGE
