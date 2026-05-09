"""Rate-limit helpers shared across Groq features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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
    """Shared rate-limit utility.

    This currently parses server-provided rate-limit metadata. Local request
    guards remain feature-specific until exact non-TTS free-tier limits are
    documented in api_spec.md.
    """

    @staticmethod
    def from_headers(headers: Mapping[str, str]) -> GroqRateLimitInfo:
        """Return rate-limit metadata from headers."""
        return GroqRateLimitInfo.from_headers(headers)

    @staticmethod
    def raise_for_headers(
        headers: Mapping[str, str], payload: dict | None = None
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
