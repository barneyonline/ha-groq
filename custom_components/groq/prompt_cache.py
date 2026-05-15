"""Small in-memory response cache used by response services."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic
from typing import Any


@dataclass(slots=True)
class PromptCacheEntry:
    """A cached service response."""

    value: dict[str, Any]
    created_at: float
    expires_at: float | None


class GroqPromptCache:
    """LRU cache for reusable service responses.

    This intentionally local cache avoids adding undocumented payload fields while
    providing reusable response caching for Home Assistant response services.
    """

    def __init__(self, max_size: int = 128, default_ttl: int | None = 300) -> None:
        self._max_size = max(0, max_size)
        self._default_ttl = default_ttl
        self._entries: OrderedDict[str, PromptCacheEntry] = OrderedDict()

    @property
    def size(self) -> int:
        """Return number of cached entries."""
        self._purge_expired()
        return len(self._entries)

    def get(self, key: str) -> dict[str, Any] | None:
        """Return a cached response by key."""
        self._purge_expired()
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return dict(entry.value)

    def set(
        self,
        key: str,
        value: dict[str, Any],
        *,
        ttl: int | None = None,
    ) -> None:
        """Store a response in the cache."""
        if self._max_size == 0:
            return
        ttl_value = self._default_ttl if ttl is None else ttl
        now = monotonic()
        expires_at = None if ttl_value is None else now + int(ttl_value)
        self._entries[key] = PromptCacheEntry(
            value=dict(value),
            created_at=now,
            expires_at=expires_at,
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_size:
            self._entries.popitem(last=False)

    def clear(self) -> int:
        """Clear the cache and return the number of removed entries."""
        count = len(self._entries)
        self._entries.clear()
        return count

    def _purge_expired(self) -> None:
        """Remove expired entries."""
        now = monotonic()
        expired = [
            key
            for key, entry in self._entries.items()
            if entry.expires_at is not None and entry.expires_at <= now
        ]
        for key in expired:
            self._entries.pop(key, None)
