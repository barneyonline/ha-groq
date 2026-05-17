"""Model and capability registry for Groq features."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
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

    if "llama-4" in model or "vision" in model or "vl" in model:
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
        capabilities=infer_capabilities(model_id),
    )


class GroqModelRegistry:
    """Capability-aware registry for built-in and discovered Groq models."""

    def __init__(
        self,
        models: Iterable[GroqModel] | None = None,
        *,
        include_built_ins: bool = True,
    ) -> None:
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
            return required.issubset(infer_capabilities(model_id))
        return required.issubset(model.capabilities)
