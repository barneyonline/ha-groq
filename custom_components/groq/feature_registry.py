"""Feature registry for optional Groq integration capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from homeassistant.const import Platform

from .const import CONF_ENABLED_FEATURES
from .errors import GroqFeatureNotEnabled


class GroqFeature(StrEnum):
    """Feature ids supported by the Groq integration architecture."""

    TEXT_GENERATION = "text_generation"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    VISION = "image_recognition"
    OCR = "ocr"
    REASONING = "reasoning"
    STRUCTURED_OUTPUTS = "structured_outputs"
    PROMPT_CACHING = "prompt_caching"


@dataclass(frozen=True, slots=True)
class GroqFeatureDescriptor:
    """Static metadata for a Groq feature."""

    feature: GroqFeature
    name: str
    required_capabilities: frozenset[str]
    platforms: frozenset[Platform]
    services: frozenset[str]


FEATURE_DESCRIPTORS: dict[GroqFeature, GroqFeatureDescriptor] = {
    GroqFeature.TEXT_GENERATION: GroqFeatureDescriptor(
        feature=GroqFeature.TEXT_GENERATION,
        name="Text generation",
        required_capabilities=frozenset({"text_generation"}),
        platforms=frozenset({Platform.CONVERSATION, Platform.AI_TASK}),
        services=frozenset({"generate_text", "generate_structured"}),
    ),
    GroqFeature.SPEECH_TO_TEXT: GroqFeatureDescriptor(
        feature=GroqFeature.SPEECH_TO_TEXT,
        name="Speech to text",
        required_capabilities=frozenset({"speech_to_text"}),
        platforms=frozenset({Platform.STT}),
        services=frozenset({"transcribe_audio"}),
    ),
    GroqFeature.TEXT_TO_SPEECH: GroqFeatureDescriptor(
        feature=GroqFeature.TEXT_TO_SPEECH,
        name="Text to speech",
        required_capabilities=frozenset({"text_to_speech"}),
        platforms=frozenset({Platform.TTS}),
        services=frozenset(),
    ),
    GroqFeature.VISION: GroqFeatureDescriptor(
        feature=GroqFeature.VISION,
        name="Image recognition",
        required_capabilities=frozenset({"vision"}),
        platforms=frozenset(),
        services=frozenset({"analyze_image", "extract_text_from_image"}),
    ),
    GroqFeature.OCR: GroqFeatureDescriptor(
        feature=GroqFeature.OCR,
        name="OCR",
        required_capabilities=frozenset({"vision"}),
        platforms=frozenset(),
        services=frozenset({"extract_text_from_image"}),
    ),
    GroqFeature.REASONING: GroqFeatureDescriptor(
        feature=GroqFeature.REASONING,
        name="Reasoning",
        required_capabilities=frozenset({"reasoning"}),
        platforms=frozenset(),
        services=frozenset({"generate_text"}),
    ),
    GroqFeature.STRUCTURED_OUTPUTS: GroqFeatureDescriptor(
        feature=GroqFeature.STRUCTURED_OUTPUTS,
        name="Structured outputs",
        required_capabilities=frozenset({"structured_outputs"}),
        platforms=frozenset(),
        services=frozenset({"generate_structured"}),
    ),
    GroqFeature.PROMPT_CACHING: GroqFeatureDescriptor(
        feature=GroqFeature.PROMPT_CACHING,
        name="Local response cache",
        required_capabilities=frozenset(),
        platforms=frozenset(),
        services=frozenset({"clear_cache"}),
    ),
}

DEFAULT_ENABLED_FEATURES: frozenset[GroqFeature] = frozenset(
    {GroqFeature.TEXT_TO_SPEECH}
)


def coerce_feature(value: str | GroqFeature) -> GroqFeature:
    """Convert a stored feature id to a GroqFeature."""
    if isinstance(value, GroqFeature):
        return value
    return GroqFeature(value)


def enabled_features_from_options(options: dict) -> frozenset[GroqFeature]:
    """Return enabled features from config entry options.

    Existing entries do not have feature options yet, so the foundation defaults
    to the existing TTS feature only. Users can opt into additional response
    services from the options flow.
    """
    configured = options.get(CONF_ENABLED_FEATURES)
    if configured is None:
        return DEFAULT_ENABLED_FEATURES

    enabled: set[GroqFeature] = set()
    for feature_id in configured:
        try:
            enabled.add(coerce_feature(feature_id))
        except ValueError:
            continue
    return frozenset(enabled)


class GroqFeatureRegistry:
    """Feature registry for a config entry."""

    def __init__(self, enabled_features: Iterable[GroqFeature]) -> None:
        self._enabled_features = frozenset(enabled_features)

    @property
    def enabled_features(self) -> frozenset[GroqFeature]:
        """Return enabled feature ids."""
        return self._enabled_features

    def is_enabled(self, feature: GroqFeature) -> bool:
        """Return whether a feature is enabled."""
        return feature in self._enabled_features

    def ensure_enabled(self, feature: GroqFeature) -> None:
        """Raise if a feature is not enabled for the entry."""
        if not self.is_enabled(feature):
            descriptor = FEATURE_DESCRIPTORS[feature]
            raise GroqFeatureNotEnabled(f"{descriptor.name} is not enabled")

    def enabled_platforms(self) -> list[Platform]:
        """Return Home Assistant platforms required by enabled features."""
        platforms: set[Platform] = set()
        for feature in self._enabled_features:
            platforms.update(FEATURE_DESCRIPTORS[feature].platforms)
        return sorted(platforms, key=str)

    def enabled_services(self) -> set[str]:
        """Return service names used by enabled features."""
        services: set[str] = set()
        for feature in self._enabled_features:
            services.update(FEATURE_DESCRIPTORS[feature].services)
        return services
