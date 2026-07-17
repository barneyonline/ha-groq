"""Rate-limit helpers shared across Groq features."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

from .errors import GroqRateLimitExceeded


@dataclass(frozen=True, slots=True)
class GroqRateLimitInfo:
    """Rate-limit metadata returned by Groq headers."""

    retry_after: str | None = None
    limit_requests: str | None = None
    limit_tokens: str | None = None
    remaining_requests: str | None = None
    remaining_tokens: str | None = None
    reset_requests: str | None = None
    reset_tokens: str | None = None

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> "GroqRateLimitInfo":
        """Build rate-limit info from response headers."""
        lowered = {key.lower(): value for key, value in headers.items()}
        return cls(
            retry_after=lowered.get("retry-after"),
            limit_requests=lowered.get("x-ratelimit-limit-requests"),
            limit_tokens=lowered.get("x-ratelimit-limit-tokens"),
            remaining_requests=lowered.get("x-ratelimit-remaining-requests"),
            remaining_tokens=lowered.get("x-ratelimit-remaining-tokens"),
            reset_requests=lowered.get("x-ratelimit-reset-requests"),
            reset_tokens=lowered.get("x-ratelimit-reset-tokens"),
        )

    def as_dict(self) -> dict[str, str | None]:
        """Return a JSON-serializable representation."""
        return {
            "retry_after": self.retry_after,
            "limit_requests": self.limit_requests,
            "limit_tokens": self.limit_tokens,
            "remaining_requests": self.remaining_requests,
            "remaining_tokens": self.remaining_tokens,
            "reset_requests": self.reset_requests,
            "reset_tokens": self.reset_tokens,
        }

    def error_message(self) -> str:
        """Return a user-facing rate-limit message."""
        details = []
        if self.retry_after:
            details.append(f"retry after {self.retry_after} seconds")
        if self.reset_requests:
            details.append(f"request window resets in {self.reset_requests}")
        if self.reset_tokens:
            details.append(f"token window resets in {self.reset_tokens}")
        suffix = f" ({'; '.join(details)})" if details else ""
        return f"Groq API rate limit exceeded{suffix}."


class GroqRateLimiter:
    """Shared rate-limit utility with optional per-service local guards."""

    def __init__(self) -> None:
        """Initialize tracked local guard state."""
        self._blocked_until: dict[str, float] = {}

    @staticmethod
    def from_headers(headers: Mapping[str, str]) -> GroqRateLimitInfo:
        """Return rate-limit metadata from headers."""
        return GroqRateLimitInfo.from_headers(headers)

    @staticmethod
    def raise_for_headers(
        headers: Mapping[str, str], payload: dict[str, Any] | None = None
    ) -> None:
        """Raise a Groq rate-limit exception using response headers."""
        info = GroqRateLimitInfo.from_headers(headers)
        raise GroqRateLimitExceeded(
            info.error_message(),
            retry_after=info.retry_after,
            reset_requests=info.reset_requests,
            reset_tokens=info.reset_tokens,
            payload=payload,
        )

    def raise_if_blocked(self, guard_key: str | None) -> None:
        """Raise if the local guard has paused a service after rate-limit headers."""
        if not guard_key:
            return
        blocked_until = self._blocked_until.get(guard_key)
        if blocked_until is None:
            return
        now = time.monotonic()
        if blocked_until <= now:
            self._blocked_until.pop(guard_key, None)
            return
        retry_after = max(1, int(blocked_until - now))
        raise GroqRateLimitExceeded(
            "Groq free-tier guard blocked this service request before sending it: "
            f"retry after {retry_after} seconds.",
            retry_after=str(retry_after),
        )

    def update_from_headers(
        self,
        guard_key: str | None,
        headers: Mapping[str, str],
    ) -> None:
        """Update local guard state from Groq rate-limit headers for one service."""
        if not guard_key:
            return
        info = GroqRateLimitInfo.from_headers(headers)
        delay = _guard_delay_seconds(info)
        if delay is None:
            return
        self._blocked_until[guard_key] = time.monotonic() + delay


def _guard_delay_seconds(info: GroqRateLimitInfo) -> int | None:
    """Return a conservative pause duration from rate-limit metadata."""
    if info.retry_after:
        return _duration_seconds(info.retry_after)
    remaining = (info.remaining_requests, info.remaining_tokens)
    if any(value == "0" for value in remaining):
        resets = (
            _duration_seconds(info.reset_requests),
            _duration_seconds(info.reset_tokens),
        )
        return max((value for value in resets if value is not None), default=60)
    return None


def _duration_seconds(value: str | None) -> int | None:
    """Parse simple Groq duration header values into whole seconds."""
    if not value:
        return None
    duration = value.strip().lower()
    try:
        return max(1, math.ceil(float(duration)))
    except ValueError:
        pass
    suffix_multipliers = {
        "ms": 0.001,
        "s": 1,
        "m": 60,
        "h": 3600,
    }
    for suffix, multiplier in suffix_multipliers.items():
        if duration.endswith(suffix):
            try:
                amount = float(duration[: -len(suffix)])
            except ValueError:
                return None
            return max(1, math.ceil(amount * multiplier))
    return None
