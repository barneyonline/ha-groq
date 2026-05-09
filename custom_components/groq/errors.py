"""Shared exceptions for the Groq integration."""

from __future__ import annotations

from typing import Any

from homeassistant.exceptions import HomeAssistantError


class GroqError(HomeAssistantError):
    """Base class for Groq integration errors."""


class GroqApiError(GroqError):
    """Raised when the Groq API returns an error response."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        error_type: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.payload = payload or {}


class GroqRateLimitExceeded(GroqApiError):
    """Raised when Groq rejects a request because of rate limits."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: str | None = None,
        reset_requests: str | None = None,
        reset_tokens: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status=429,
            error_type="rate_limit_exceeded",
            payload=payload,
        )
        self.retry_after = retry_after
        self.reset_requests = reset_requests
        self.reset_tokens = reset_tokens


class GroqFeatureNotEnabled(GroqError):
    """Raised when a service is called for a disabled feature."""


class GroqModelNotSupported(GroqError):
    """Raised when a model is not known to support the requested feature."""


class GroqUnsupportedCapability(GroqModelNotSupported):
    """Raised when a model does not support a requested capability."""


class GroqResponseError(GroqApiError):
    """Raised when Groq returns a response that cannot be parsed safely."""
