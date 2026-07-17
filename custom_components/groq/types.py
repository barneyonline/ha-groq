"""Shared type aliases for the Groq integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntry

if TYPE_CHECKING:
    from .runtime import GroqRuntimeData

type GroqConfigEntry = ConfigEntry[GroqRuntimeData]
