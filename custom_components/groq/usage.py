"""Content-free usage measurements for requests made by this integration."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any


def _number(value: Any) -> float | None:
    """Accept only finite, non-negative provider measurements."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


class GroqUsage:
    """Keep the latest request measurements per service, without response text."""

    def __init__(self) -> None:
        self.values: dict[str, dict[str, float | None]] = {}
        self._listeners: set[Callable[[], None]] = set()

    def subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Subscribe until the entity is removed."""
        self._listeners.add(listener)
        return lambda: self._listeners.discard(listener)

    def record(self, service_id: str | None, usage: dict[str, Any]) -> None:
        """Replace latest metrics; never carry missing values across requests."""
        if not service_id:
            return
        details = usage.get("prompt_tokens_details")
        cached = (
            _number(details.get("cached_tokens")) if isinstance(details, dict) else None
        )
        prompt = _number(usage.get("prompt_tokens"))
        self.values[service_id] = {
            "requests": (self.values.get(service_id, {}).get("requests") or 0) + 1,
            "prompt_tokens": prompt,
            "completion_tokens": _number(usage.get("completion_tokens")),
            "total_tokens": _number(usage.get("total_tokens")),
            "response_time": _number(usage.get("total_time")),
            "cached_tokens": cached,
            "cache_hit_rate": (
                cached / prompt * 100
                if prompt and cached is not None and cached <= prompt
                else None
            ),
        }
        for listener in tuple(self._listeners):
            listener()

    def clear(self) -> None:
        """Release measurements and callbacks when the client unloads."""
        self.values.clear()
        self._listeners.clear()
