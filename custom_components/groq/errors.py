"""Shared exceptions for the Groq integration."""

from __future__ import annotations

from typing import Any

from homeassistant.exceptions import HomeAssistantError, ServiceValidationError

from .const import DOMAIN


class GroqError(HomeAssistantError):
    """Base class for Groq integration errors."""

    def __init__(
        self,
        message: str,
        *,
        translation_key: str = "integration_error",
        translation_placeholders: dict[str, str] | None = None,
    ) -> None:
        """Initialize an error with Home Assistant translation metadata."""
        super().__init__(
            message,
            translation_domain=DOMAIN,
            translation_key=translation_key,
            translation_placeholders=translation_placeholders,
        )


class GroqApiError(GroqError):
    """Raised when the Groq API returns an error response."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        error_type: str | None = None,
        payload: dict[str, Any] | None = None,
        translation_key: str = "api_error",
    ) -> None:
        super().__init__(message, translation_key=translation_key)
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
            translation_key="rate_limit_exceeded",
        )
        self.retry_after = retry_after
        self.reset_requests = reset_requests
        self.reset_tokens = reset_tokens


class GroqFeatureNotEnabled(GroqError):
    """Raised when a service is called for a disabled feature."""

    def __init__(self, message: str, *, feature: str) -> None:
        """Initialize a translated feature error."""
        super().__init__(
            message,
            translation_key="feature_not_enabled",
            translation_placeholders={"feature": feature},
        )


class GroqModelNotSupported(GroqError):
    """Raised when a model is not known to support the requested feature."""


class GroqUnsupportedCapability(GroqModelNotSupported):
    """Raised when a model does not support a requested capability."""


class GroqResponseError(GroqApiError):
    """Raised when Groq returns a response that cannot be parsed safely."""

    def __init__(self, message: str) -> None:
        """Initialize a translated invalid-response error."""
        super().__init__(message, translation_key="invalid_api_response")


def translated_service_error(
    fallback_message: str,
    translation_key: str,
    **placeholders: object,
) -> ServiceValidationError:
    """Return a translated Home Assistant service validation error."""
    return ServiceValidationError(
        fallback_message,
        translation_domain=DOMAIN,
        translation_key=translation_key,
        translation_placeholders={
            key: str(value) for key, value in placeholders.items()
        },
    )


def translated_error(
    fallback_message: str,
    translation_key: str,
    **placeholders: object,
) -> HomeAssistantError:
    """Return a translated Home Assistant runtime error."""
    translated_placeholders = {key: str(value) for key, value in placeholders.items()}
    return HomeAssistantError(
        fallback_message,
        translation_domain=DOMAIN,
        translation_key=translation_key,
        translation_placeholders=translated_placeholders,
    )
