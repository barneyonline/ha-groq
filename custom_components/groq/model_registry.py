"""Model and capability registry for Groq features."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
import re
from typing import Any, Iterable

from .const import PROMPT_CACHING_MODELS, REASONING_MODELS, STRUCTURED_OUTPUTS_MODELS
from .feature_registry import GroqFeature


class GroqCapability(StrEnum):
    """Capability ids used to match models to features."""

    TEXT = "text_generation"
    TEXT_GENERATION = "text_generation"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    VISION = "vision"
    OCR = "vision"
    REASONING = "reasoning"
    STRUCTURED_OUTPUTS = "structured_outputs"
    PROMPT_CACHING = "prompt_caching"
    COMPOUND = "compound"
    TOOL_CALLING = "tool_calling"


@dataclass(frozen=True, slots=True)
class GroqModel:
    """Groq model metadata relevant to the integration."""

    model_id: str
    active: bool = True
    owned_by: str | None = None
    context_window: int | None = None
    max_completion_tokens: int | None = None
    capabilities: frozenset[GroqCapability] = frozenset()

    @property
    def completion_token_limit(self) -> int | None:
        """Return the safest known upper bound for generated tokens."""
        limits = [
            value
            for value in (self.context_window, self.max_completion_tokens)
            if value is not None
        ]
        return min(limits) if limits else None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "id": self.model_id,
            "active": self.active,
            "owned_by": self.owned_by,
            "context_window": self.context_window,
            "max_completion_tokens": self.max_completion_tokens,
            "completion_token_limit": self.completion_token_limit,
            "capabilities": sorted(str(capability) for capability in self.capabilities),
        }


DEFAULT_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_STRUCTURED_MODEL = "openai/gpt-oss-20b"

CAPABILITY_METADATA_KEYS = frozenset(
    {
        "capabilities",
        "capability",
        "endpoints",
        "endpoint",
        "features",
        "feature",
        "modalities",
        "modality",
        "supportedcapabilities",
        "supported_capabilities",
        "supportedinputs",
        "supported_inputs",
        "tasks",
        "task",
    }
)
INPUT_CAPABILITY_METADATA_KEYS = frozenset(
    {
        "inputmodalities",
        "input_modality",
        "input_modalities",
        "inputs",
        "inputtypes",
        "input_type",
        "input_types",
        "supportedinputs",
        "supported_inputs",
    }
)
OUTPUT_CAPABILITY_METADATA_KEYS = frozenset(
    {
        "outputmodalities",
        "output_modality",
        "output_modalities",
        "outputs",
        "outputtypes",
        "output_type",
        "output_types",
        "supportedoutputs",
        "supported_outputs",
    }
)
CAPABILITY_ALIASES: dict[str, GroqCapability] = {
    "audio_transcription": GroqCapability.SPEECH_TO_TEXT,
    "chat": GroqCapability.TEXT_GENERATION,
    "chat_completion": GroqCapability.TEXT_GENERATION,
    "chat_completions": GroqCapability.TEXT_GENERATION,
    "completion": GroqCapability.TEXT_GENERATION,
    "completions": GroqCapability.TEXT_GENERATION,
    "compound": GroqCapability.COMPOUND,
    "function_calling": GroqCapability.TOOL_CALLING,
    "image": GroqCapability.VISION,
    "image_input": GroqCapability.VISION,
    "image_understanding": GroqCapability.VISION,
    "json_schema": GroqCapability.STRUCTURED_OUTPUTS,
    "multimodal": GroqCapability.VISION,
    "ocr": GroqCapability.VISION,
    "prompt_caching": GroqCapability.PROMPT_CACHING,
    "reasoning": GroqCapability.REASONING,
    "speech": GroqCapability.TEXT_TO_SPEECH,
    "speech_to_text": GroqCapability.SPEECH_TO_TEXT,
    "stt": GroqCapability.SPEECH_TO_TEXT,
    "structured_output": GroqCapability.STRUCTURED_OUTPUTS,
    "structured_outputs": GroqCapability.STRUCTURED_OUTPUTS,
    "text": GroqCapability.TEXT_GENERATION,
    "text_generation": GroqCapability.TEXT_GENERATION,
    "text_to_speech": GroqCapability.TEXT_TO_SPEECH,
    "tool_calling": GroqCapability.TOOL_CALLING,
    "tools": GroqCapability.TOOL_CALLING,
    "transcription": GroqCapability.SPEECH_TO_TEXT,
    "tts": GroqCapability.TEXT_TO_SPEECH,
    "vision": GroqCapability.VISION,
}
VISION_MODEL_MARKERS = frozenset(
    {
        "image",
        "multimodal",
        "omni",
        "vision",
        "vl",
    }
)
VISION_MODEL_PREFIXES = ("qwen/qwen3.6-",)

BUILT_IN_MODELS: dict[str, GroqModel] = {
    "llama-3.1-8b-instant": GroqModel(
        model_id="llama-3.1-8b-instant",
        context_window=131072,
        max_completion_tokens=131072,
        capabilities=frozenset(
            {
                GroqCapability.TEXT_GENERATION,
                GroqCapability.TOOL_CALLING,
            }
        ),
    ),
    "llama-3.3-70b-versatile": GroqModel(
        model_id="llama-3.3-70b-versatile",
        context_window=131072,
        max_completion_tokens=32768,
        capabilities=frozenset(
            {
                GroqCapability.TEXT_GENERATION,
                GroqCapability.TOOL_CALLING,
            }
        ),
    ),
    "openai/gpt-oss-20b": GroqModel(
        model_id="openai/gpt-oss-20b",
        context_window=131072,
        max_completion_tokens=65536,
        capabilities=frozenset(
            {
                GroqCapability.TEXT_GENERATION,
                GroqCapability.REASONING,
                GroqCapability.STRUCTURED_OUTPUTS,
                GroqCapability.PROMPT_CACHING,
                GroqCapability.TOOL_CALLING,
            }
        ),
    ),
    "openai/gpt-oss-120b": GroqModel(
        model_id="openai/gpt-oss-120b",
        context_window=131072,
        max_completion_tokens=65536,
        capabilities=frozenset(
            {
                GroqCapability.TEXT_GENERATION,
                GroqCapability.REASONING,
                GroqCapability.STRUCTURED_OUTPUTS,
                GroqCapability.PROMPT_CACHING,
                GroqCapability.TOOL_CALLING,
            }
        ),
    ),
    "openai/gpt-oss-safeguard-20b": GroqModel(
        model_id="openai/gpt-oss-safeguard-20b",
        context_window=131072,
        max_completion_tokens=65536,
        capabilities=frozenset(
            {
                GroqCapability.TEXT_GENERATION,
                GroqCapability.REASONING,
                GroqCapability.PROMPT_CACHING,
            }
        ),
    ),
    "qwen/qwen3-32b": GroqModel(
        model_id="qwen/qwen3-32b",
        context_window=131072,
        max_completion_tokens=40960,
        capabilities=frozenset(
            {
                GroqCapability.TEXT_GENERATION,
                GroqCapability.REASONING,
                GroqCapability.TOOL_CALLING,
            }
        ),
    ),
    "groq/compound": GroqModel(
        model_id="groq/compound",
        context_window=131072,
        max_completion_tokens=8192,
        capabilities=frozenset(
            {
                GroqCapability.TEXT_GENERATION,
                GroqCapability.COMPOUND,
            }
        ),
    ),
    "groq/compound-mini": GroqModel(
        model_id="groq/compound-mini",
        context_window=131072,
        max_completion_tokens=8192,
        capabilities=frozenset(
            {
                GroqCapability.TEXT_GENERATION,
                GroqCapability.COMPOUND,
            }
        ),
    ),
    "meta-llama/llama-4-scout-17b-16e-instruct": GroqModel(
        model_id="meta-llama/llama-4-scout-17b-16e-instruct",
        context_window=131072,
        max_completion_tokens=8192,
        capabilities=frozenset(
            {
                GroqCapability.TEXT_GENERATION,
                GroqCapability.VISION,
                GroqCapability.STRUCTURED_OUTPUTS,
                GroqCapability.TOOL_CALLING,
            }
        ),
    ),
    "qwen/qwen3.6-27b": GroqModel(
        model_id="qwen/qwen3.6-27b",
        context_window=131072,
        max_completion_tokens=8192,
        capabilities=frozenset(
            {
                GroqCapability.TEXT_GENERATION,
                GroqCapability.VISION,
                GroqCapability.STRUCTURED_OUTPUTS,
                GroqCapability.TOOL_CALLING,
            }
        ),
    ),
    "meta-llama/llama-4-maverick-17b-128e-instruct": GroqModel(
        model_id="meta-llama/llama-4-maverick-17b-128e-instruct",
        context_window=131072,
        max_completion_tokens=8192,
        capabilities=frozenset(
            {
                GroqCapability.TEXT_GENERATION,
                GroqCapability.VISION,
                GroqCapability.TOOL_CALLING,
            }
        ),
    ),
    "whisper-large-v3": GroqModel(
        model_id="whisper-large-v3",
        capabilities=frozenset({GroqCapability.SPEECH_TO_TEXT}),
    ),
    "whisper-large-v3-turbo": GroqModel(
        model_id="whisper-large-v3-turbo",
        capabilities=frozenset({GroqCapability.SPEECH_TO_TEXT}),
    ),
    "canopylabs/orpheus-v1-english": GroqModel(
        model_id="canopylabs/orpheus-v1-english",
        context_window=4000,
        max_completion_tokens=50000,
        capabilities=frozenset({GroqCapability.TEXT_TO_SPEECH}),
    ),
    "canopylabs/orpheus-arabic-saudi": GroqModel(
        model_id="canopylabs/orpheus-arabic-saudi",
        context_window=4000,
        max_completion_tokens=50000,
        capabilities=frozenset({GroqCapability.TEXT_TO_SPEECH}),
    ),
}

FEATURE_CAPABILITIES: dict[GroqFeature, frozenset[GroqCapability]] = {
    GroqFeature.TEXT_GENERATION: frozenset({GroqCapability.TEXT_GENERATION}),
    GroqFeature.SPEECH_TO_TEXT: frozenset({GroqCapability.SPEECH_TO_TEXT}),
    GroqFeature.TEXT_TO_SPEECH: frozenset({GroqCapability.TEXT_TO_SPEECH}),
    GroqFeature.VISION: frozenset({GroqCapability.VISION}),
    GroqFeature.OCR: frozenset({GroqCapability.VISION}),
    GroqFeature.REASONING: frozenset({GroqCapability.REASONING}),
    GroqFeature.STRUCTURED_OUTPUTS: frozenset({GroqCapability.STRUCTURED_OUTPUTS}),
    GroqFeature.PROMPT_CACHING: frozenset({GroqCapability.PROMPT_CACHING}),
}


def _normalized_capability_token(value: str) -> str:
    """Return a normalized token for capability matching."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _capability_from_token(value: str) -> GroqCapability | None:
    """Return a capability represented by an API metadata token."""
    token = _normalized_capability_token(value)
    if capability := CAPABILITY_ALIASES.get(token):
        return capability
    if "image" in token or "vision" in token:
        return GroqCapability.VISION
    return None


def _capability_from_metadata_key(value: str) -> GroqCapability | None:
    """Return a capability represented by an API metadata object key."""
    token = _normalized_capability_token(value)
    if capability := CAPABILITY_ALIASES.get(token):
        return capability
    return None


def _metadata_value_is_enabled(value: Any) -> bool:
    """Return whether a metadata value indicates support."""
    if value is False or value is None:
        return False
    if isinstance(value, str):
        return _normalized_capability_token(value) not in {
            "false",
            "no",
            "none",
            "unsupported",
        }
    if isinstance(value, dict):
        for key in ("supported", "available", "enabled", "active"):
            if key in value:
                return _metadata_value_is_enabled(value[key])
        return any(_metadata_value_is_enabled(item) for item in value.values())
    if isinstance(value, list | tuple | set):
        return any(_metadata_value_is_enabled(item) for item in value)
    return bool(value)


def _capabilities_from_metadata_value(
    value: Any,
    *,
    allow_vision: bool = True,
) -> set[GroqCapability]:
    """Return capabilities parsed from a model metadata value."""
    capabilities: set[GroqCapability] = set()
    if isinstance(value, str):
        capability = _capability_from_token(value)
        if capability and (allow_vision or capability != GroqCapability.VISION):
            capabilities.add(capability)
        for token in re.split(r"[^a-zA-Z0-9]+", value):
            capability = _capability_from_token(token)
            if capability and (allow_vision or capability != GroqCapability.VISION):
                capabilities.add(capability)
        return capabilities
    if isinstance(value, list | tuple | set):
        for item in value:
            capabilities.update(
                _capabilities_from_metadata_value(item, allow_vision=allow_vision)
            )
        return capabilities
    if isinstance(value, dict):
        for key, item in value.items():
            if not _metadata_value_is_enabled(item):
                continue
            token = _normalized_capability_token(str(key))
            if token in OUTPUT_CAPABILITY_METADATA_KEYS:
                capabilities.update(
                    _capabilities_from_metadata_value(item, allow_vision=False)
                )
                continue
            if token in INPUT_CAPABILITY_METADATA_KEYS:
                capabilities.update(_capabilities_from_metadata_value(item))
                continue
            key_capability = _capability_from_metadata_key(str(key))
            if key_capability and (
                allow_vision or key_capability != GroqCapability.VISION
            ):
                capabilities.add(key_capability)
            if item is True:
                capabilities.update(
                    _capabilities_from_metadata_value(
                        str(key),
                        allow_vision=allow_vision,
                    )
                )
                continue
            capabilities.update(
                _capabilities_from_metadata_value(item, allow_vision=allow_vision)
            )
        return capabilities
    return capabilities


def capabilities_from_api_metadata(data: dict[str, Any]) -> frozenset[GroqCapability]:
    """Return capabilities explicitly advertised by a Groq model payload."""
    capabilities: set[GroqCapability] = set()
    for key, value in data.items():
        token = _normalized_capability_token(key)
        if token in OUTPUT_CAPABILITY_METADATA_KEYS:
            capabilities.update(
                _capabilities_from_metadata_value(value, allow_vision=False)
            )
        elif token in CAPABILITY_METADATA_KEYS | INPUT_CAPABILITY_METADATA_KEYS:
            capabilities.update(_capabilities_from_metadata_value(value))
    return frozenset(capabilities)


def _model_id_tokens(model_id: str) -> set[str]:
    """Return normalized model id tokens."""
    return {
        token
        for token in (
            _normalized_capability_token(part)
            for part in re.split(r"[/_.:-]+", model_id)
        )
        if token
    }


def _is_known_vision_model_id(model_id: str) -> bool:
    """Return whether a sparse model id is known to represent vision input."""
    model = model_id.lower()
    return (
        "llama-4" in model
        or model.startswith(VISION_MODEL_PREFIXES)
        or bool(VISION_MODEL_MARKERS & _model_id_tokens(model))
    )


@lru_cache(maxsize=512)
def infer_capabilities(model_id: str) -> frozenset[GroqCapability]:
    """Infer capabilities for discovered models when Groq metadata is sparse."""
    if model_id in BUILT_IN_MODELS:
        return BUILT_IN_MODELS[model_id].capabilities

    model = model_id.lower()
    capabilities: set[GroqCapability] = set()
    # Groq's /models endpoint is OpenAI-compatible and may not include feature
    # metadata, so discovered models get conservative capability guesses by id.
    if "whisper" in model:
        capabilities.add(GroqCapability.SPEECH_TO_TEXT)
    elif "orpheus" in model:
        capabilities.add(GroqCapability.TEXT_TO_SPEECH)
    elif model.endswith("-tts") or "tts" in model:
        return frozenset()
    else:
        capabilities.add(GroqCapability.TEXT_GENERATION)

    if _is_known_vision_model_id(model_id):
        capabilities.add(GroqCapability.VISION)
    if model_id in REASONING_MODELS:
        capabilities.add(GroqCapability.REASONING)
    if model.startswith("groq/compound"):
        capabilities.add(GroqCapability.COMPOUND)
    if model_id in PROMPT_CACHING_MODELS:
        capabilities.add(GroqCapability.PROMPT_CACHING)
    if model_id in STRUCTURED_OUTPUTS_MODELS:
        capabilities.add(GroqCapability.STRUCTURED_OUTPUTS)
    return frozenset(capabilities)


def model_from_api(data: dict[str, Any]) -> GroqModel:
    """Build GroqModel metadata from the OpenAI-compatible /models response."""
    model_id = str(data.get("id") or data.get("name") or "")
    built_in = BUILT_IN_MODELS.get(model_id)
    capabilities = infer_capabilities(model_id) | capabilities_from_api_metadata(data)
    return GroqModel(
        model_id=model_id,
        active=bool(data.get("active", True)),
        owned_by=data.get("owned_by"),
        context_window=data.get(
            "context_window",
            built_in.context_window if built_in else None,
        ),
        max_completion_tokens=data.get(
            "max_completion_tokens",
            built_in.max_completion_tokens if built_in else None,
        ),
        capabilities=capabilities,
    )


class GroqModelRegistry:
    """Capability-aware registry for built-in and discovered Groq models."""

    def __init__(
        self,
        models: Iterable[GroqModel] | None = None,
        *,
        include_built_ins: bool = True,
    ) -> None:
        self._allow_missing_inference = include_built_ins
        self._models = dict(BUILT_IN_MODELS) if include_built_ins else {}
        if models:
            self.update(models)

    @property
    def models(self) -> dict[str, GroqModel]:
        """Return registered models by id."""
        return self._models

    def update(self, models: Iterable[GroqModel]) -> None:
        """Merge discovered models into the registry."""
        for model in models:
            if model.model_id:
                self._models[model.model_id] = model

    def get(self, model_id: str) -> GroqModel | None:
        """Return model metadata by id."""
        return self._models.get(model_id)

    def completion_token_limit(self, model_id: str) -> int | None:
        """Return the safest known generated-token upper bound for a model."""
        model = self._models.get(model_id) or BUILT_IN_MODELS.get(model_id)
        return model.completion_token_limit if model is not None else None

    def context_window(self, model_id: str) -> int | None:
        """Return the known context-window upper bound for a model."""
        model = self._models.get(model_id) or BUILT_IN_MODELS.get(model_id)
        return model.context_window if model is not None else None

    def models_for_feature(self, feature: GroqFeature) -> list[GroqModel]:
        """Return active models known to support a feature."""
        required = FEATURE_CAPABILITIES[feature]
        if not required:
            return list(self._models.values())
        return sorted(
            (
                model
                for model in self._models.values()
                if model.active and required.issubset(model.capabilities)
            ),
            key=lambda item: item.model_id,
        )

    def all_models(self) -> list[GroqModel]:
        """Return all registered models sorted by id."""
        return sorted(self._models.values(), key=lambda item: item.model_id)

    def supports(
        self,
        model_id: str,
        feature_or_capability: GroqFeature | GroqCapability,
    ) -> bool:
        """Return whether a model supports a feature or direct capability."""
        if isinstance(feature_or_capability, GroqCapability):
            required = frozenset({feature_or_capability})
        else:
            required = FEATURE_CAPABILITIES[feature_or_capability]
        if not required:
            return True
        model = self._models.get(model_id)
        if model is None:
            if not self._allow_missing_inference:
                return False
            return required.issubset(infer_capabilities(model_id))
        return required.issubset(model.capabilities)
