"""Small in-memory response cache used by response services."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from heapq import heappop, heappush
from time import monotonic
from typing import Any, cast

MAX_PROMPT_CACHE_BYTES = 16 * 1024 * 1024


@dataclass(slots=True)
class PromptCacheEntry:
    """A cached service response."""

    value: bytes
    size_bytes: int
    expires_at: float | None


class GroqPromptCache:
    """LRU cache for reusable service responses.

    This intentionally local cache avoids adding undocumented payload fields while
    providing reusable response caching for Home Assistant response services.
    """

    def __init__(
        self,
        max_size: int = 128,
        default_ttl: int | None = 300,
        *,
        max_bytes: int = MAX_PROMPT_CACHE_BYTES,
    ) -> None:
        self._max_size = max(0, max_size)
        self._default_ttl = default_ttl
        self._entries: OrderedDict[str, PromptCacheEntry] = OrderedDict()
        self._expiry_heap: list[tuple[float, str]] = []
        self._max_bytes = max(0, max_bytes)
        self._size_bytes = 0

    @property
    def size_bytes(self) -> int:
        """Return retained serialized content bytes, including keys."""
        self._purge_expired()
        return self._size_bytes

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
        return cast(dict[str, Any], json.loads(entry.value))

    def set(
        self,
        key: str,
        value: dict[str, Any],
        *,
        ttl: int | None = None,
    ) -> None:
        """Store a response in the cache."""
        if self._max_size == 0 or self._max_bytes == 0:
            return
        self._purge_expired()
        encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
        size_bytes = len(encoded) + len(key.encode())
        self._remove(key)
        if size_bytes > self._max_bytes:
            return
        ttl_value = self._default_ttl if ttl is None else ttl
        now = monotonic()
        expires_at = None if ttl_value is None else now + int(ttl_value)
        self._entries[key] = PromptCacheEntry(
            value=encoded,
            size_bytes=size_bytes,
            expires_at=expires_at,
        )
        self._size_bytes += size_bytes
        if expires_at is not None:
            heappush(self._expiry_heap, (expires_at, key))
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_size or self._size_bytes > self._max_bytes:
            self._remove(next(iter(self._entries)))
        self._compact_expiry_heap_if_needed()

    def clear(self) -> int:
        """Clear the cache and return the number of removed entries."""
        count = len(self._entries)
        self._entries.clear()
        self._expiry_heap.clear()
        self._size_bytes = 0
        return count

    def _remove(self, key: str) -> None:
        """Remove an entry and its retained content accounting."""
        if (entry := self._entries.pop(key, None)) is not None:
            self._size_bytes -= entry.size_bytes

    def _purge_expired(self) -> None:
        """Remove expired entries."""
        now = monotonic()
        while self._expiry_heap and self._expiry_heap[0][0] <= now:
            expires_at, key = heappop(self._expiry_heap)
            entry = self._entries.get(key)
            if entry is not None and entry.expires_at == expires_at:
                self._remove(key)

    def _compact_expiry_heap_if_needed(self) -> None:
        """Rebuild expiry bookkeeping when stale heap entries accumulate."""
        if len(self._expiry_heap) <= max(64, len(self._entries) * 2):
            return
        self._expiry_heap = [
            (entry.expires_at, key)
            for key, entry in self._entries.items()
            if entry.expires_at is not None
        ]
        self._expiry_heap.sort()
