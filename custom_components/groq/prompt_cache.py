"""Small in-memory response cache used by response services."""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from heapq import heappop, heappush
from time import monotonic
from typing import Any, cast

from homeassistant.core import HomeAssistant

from .json_utils import (
    JSON_EXECUTOR_THRESHOLD,
    estimated_json_size_exceeds,
    minimum_json_size_exceeds,
)

MAX_PROMPT_CACHE_BYTES = 16 * 1024 * 1024
MAX_PROMPT_CACHE_ENTRY_BYTES = 1024 * 1024
MAX_PROMPT_CACHE_JOBS = 8


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
        self._pending_writes: dict[str, asyncio.Future[bytes | None]] = {}
        self._active_jobs = 0
        self._closed = False

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
        entry = self._get_entry(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return cast(dict[str, Any], json.loads(entry.value))

    async def async_get(self, hass: HomeAssistant, key: str) -> dict[str, Any] | None:
        """Return an independent response, decoding large entries in the executor."""
        entry = self._get_entry(key)
        if entry is None:
            return None
        if len(entry.value) >= JSON_EXECUTOR_THRESHOLD:
            job = self._submit_job(hass, json.loads, entry.value)
            if job is None:
                return None
            self._entries.move_to_end(key)
            return cast(dict[str, Any], await asyncio.shield(job))
        self._entries.move_to_end(key)
        return cast(dict[str, Any], json.loads(entry.value))

    def _get_entry(self, key: str) -> PromptCacheEntry | None:
        """Look up the cached snapshot on the event loop."""
        self._purge_expired()
        return self._entries.get(key)

    def set(
        self,
        key: str,
        value: dict[str, Any],
        *,
        ttl: int | None = None,
    ) -> None:
        """Store a response in the cache."""
        if self._closed or self._max_size == 0 or self._max_bytes == 0:
            return
        encoded = self._encode_bounded(value)
        self._pending_writes.pop(key, None)
        self._store(key, encoded, ttl)

    async def async_set(
        self,
        hass: HomeAssistant,
        key: str,
        value: dict[str, Any],
        *,
        ttl: int | None = None,
    ) -> None:
        """Encode large responses off-loop and commit to the current cache."""
        if self._closed or self._max_size == 0 or self._max_bytes == 0:
            return
        self._pending_writes.pop(key, None)
        if any(
            isinstance(item, str) and len(item) > MAX_PROMPT_CACHE_ENTRY_BYTES
            for item in value.values()
        ):
            self._store(key, None, ttl)
            return
        if not estimated_json_size_exceeds(
            value, JSON_EXECUTOR_THRESHOLD, use_encoded_upper_bound=True
        ):
            self._store(key, self._encode(value), ttl)
            return
        job = self._submit_job(hass, self._encode_bounded, value)
        if job is None:
            return
        self._pending_writes[key] = job
        try:
            encoded = await asyncio.shield(job)
        finally:
            newest = self._pending_writes.get(key) is job
            if newest:
                del self._pending_writes[key]
        if newest:
            self._store(key, encoded, ttl)

    def _submit_job[InputT, ResultT](
        self, hass: HomeAssistant, work: Callable[[InputT], ResultT], value: InputT
    ) -> asyncio.Future[ResultT] | None:
        """Bypass a saturated cache without queuing additional codec jobs."""
        if self._active_jobs >= MAX_PROMPT_CACHE_JOBS:
            return None
        job = asyncio.ensure_future(hass.async_add_executor_job(work, value))
        self._active_jobs += 1
        job.add_done_callback(self._job_finished)
        return job

    def _job_finished(self, job: asyncio.Future[Any]) -> None:
        """Release admission only when the worker settles."""
        self._active_jobs -= 1
        if not job.cancelled():
            job.exception()

    @staticmethod
    def _encode(value: dict[str, Any]) -> bytes:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()

    @classmethod
    def _encode_bounded(cls, value: dict[str, Any]) -> bytes | None:
        if minimum_json_size_exceeds(value, MAX_PROMPT_CACHE_ENTRY_BYTES):
            return None
        return cls._encode(value)

    def _store(self, key: str, encoded: bytes | None, ttl: int | None) -> None:
        self._purge_expired()
        self._remove(key)
        if encoded is None:
            return
        size_bytes = len(encoded) + len(key.encode())
        if size_bytes > min(self._max_bytes, MAX_PROMPT_CACHE_ENTRY_BYTES):
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
        self._pending_writes.clear()
        return count

    def close(self) -> None:
        """Release this entry's cache and discard pending writes."""
        self._closed = True
        self.clear()

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
