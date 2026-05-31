"""Helpers for validating Groq TTS vocal directions."""

from __future__ import annotations

from typing import Any

from .const import VOCAL_DIRECTION_NONE

VOCAL_DIRECTIONS_ERROR = "invalid_vocal_directions"


def _is_single_word(value: str) -> bool:
    """Return whether value is one alphabetic word with optional hyphenated parts."""
    return all(part.isalpha() for part in value.split("-"))


def normalize_vocal_directions(value: Any) -> str:
    """Return a valid TTS vocal directions value, or an empty string."""
    if value in (None, "", VOCAL_DIRECTION_NONE):
        return ""
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if value in ("", VOCAL_DIRECTION_NONE):
        return ""
    if _is_single_word(value):
        return value
    return ""


def vocal_directions_validation_error(value: Any) -> str | None:
    """Return a validation error for invalid TTS vocal directions."""
    if value in (None, "", VOCAL_DIRECTION_NONE):
        return None
    if not isinstance(value, str):
        return VOCAL_DIRECTIONS_ERROR
    value = value.strip()
    if value in ("", VOCAL_DIRECTION_NONE):
        return None
    if normalize_vocal_directions(value):
        return None
    return VOCAL_DIRECTIONS_ERROR
