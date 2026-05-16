"""
TTS Engine for Groq.
"""

from __future__ import annotations
from collections.abc import Callable, Mapping
from hashlib import sha256
import json
import logging
import asyncio
from urllib.error import HTTPError, URLError
from collections import OrderedDict, deque

import aiohttp
from homeassistant.core import HomeAssistant
from asyncio import CancelledError

from homeassistant.exceptions import HomeAssistantError, ConfigEntryAuthFailed
from .const import GROQ_FREE_TIER_LIMITS, VERSION
from .repairs import async_create_model_access_issue

_LOGGER = logging.getLogger(__name__)

RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_DAY_SECONDS = 24 * 60 * 60
TTS_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)

_CLIENTSESSION_FACTORY: Callable[[HomeAssistant], aiohttp.ClientSession] | None = None


def _load_clientsession_factory() -> Callable[[HomeAssistant], aiohttp.ClientSession]:
    """Import and return Home Assistant's shared session helper."""
    from homeassistant.helpers.aiohttp_client import async_get_clientsession as _get

    return _get


async def async_preload_clientsession_helper(hass: HomeAssistant) -> None:
    """Load the session helper before first TTS request handling."""
    global _CLIENTSESSION_FACTORY
    if _CLIENTSESSION_FACTORY is not None:
        return
    if hasattr(hass, "async_add_executor_job"):
        _CLIENTSESSION_FACTORY = await hass.async_add_executor_job(
            _load_clientsession_factory
        )
        return
    _CLIENTSESSION_FACTORY = _load_clientsession_factory()


def async_get_clientsession(hass: HomeAssistant) -> aiohttp.ClientSession:
    """Return Home Assistant's shared aiohttp session."""
    global _CLIENTSESSION_FACTORY
    if _CLIENTSESSION_FACTORY is None:
        _CLIENTSESSION_FACTORY = _load_clientsession_factory()
    return _CLIENTSESSION_FACTORY(hass)


def _payload_mentions_model_access(payload: object) -> bool:
    """Return whether an API error looks like an unavailable model problem."""
    if not isinstance(payload, dict):
        return False
    error = payload.get("error")
    if isinstance(error, dict):
        message = str(error.get("message", ""))
        error_type = str(error.get("type", ""))
    else:
        message = str(error or payload)
        error_type = ""
    text = f"{message} {error_type}".lower()
    return "model" in text and any(
        phrase in text
        for phrase in (
            "not found",
            "does not exist",
            "not available",
            "not accessible",
            "not enabled",
            "access",
        )
    )


class AudioResponse:
    """A simple response wrapper with a 'content' attribute to hold audio bytes."""

    def __init__(self, content: bytes):
        self.content = content


class GroqRateLimitError(HomeAssistantError):
    """Raised when Groq or the local free-tier guard rejects a TTS request."""


class GroqTTSEngine:
    def __init__(
        self,
        api_key: str,
        voice: str,
        model: str,
        url: str,
        cache_max: int | None = None,
        protect_free_tier: bool = True,
        response_format: str = "wav",
    ) -> None:
        self._api_key = api_key
        self._voice = voice
        self._model = model
        self._response_format = response_format
        self._url = url
        self._session: aiohttp.ClientSession | None = None
        self._cache: OrderedDict[tuple[str, str, str, str], bytes] = OrderedDict()
        self._cache_max = cache_max if cache_max is not None else 256
        self._protect_free_tier = protect_free_tier
        self._request_timestamps: deque[float] = deque()
        self._token_timestamps: deque[tuple[float, int]] = deque()
        self._minute_request_timestamps: deque[float] = deque()
        self._minute_token_timestamps: deque[tuple[float, int]] = deque()
        self._daily_token_total = 0
        self._minute_token_total = 0
        self._available = True
        self._unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        """Return whether the last TTS API interaction succeeded."""
        return self._available

    def _mark_unavailable(self, reason: str) -> None:
        """Log the transition to unavailable once per outage."""
        if not self._available and self._unavailable_reason == reason:
            return
        self._available = False
        self._unavailable_reason = reason
        _LOGGER.warning("%s; Groq TTS calls will be retried by Home Assistant", reason)

    def _mark_available(self) -> None:
        """Log recovery once after an outage."""
        if self._available:
            return
        self._available = True
        self._unavailable_reason = None
        _LOGGER.info("Groq TTS API is reachable again")

    @staticmethod
    def _estimate_token_usage(text: str) -> int:
        """Return a conservative local estimate for Groq text token usage."""
        return max(1, len(text))

    def _free_tier_limits(self, model: str | None = None) -> dict[str, int] | None:
        """Return free-tier limits for the configured model, if known."""
        return GROQ_FREE_TIER_LIMITS.get(model or self._model)

    def _prune_local_usage(self, now: float) -> None:
        """Drop local rate-limit records outside active rolling windows."""
        oldest_daily = now - RATE_LIMIT_DAY_SECONDS
        while self._request_timestamps and self._request_timestamps[0] <= oldest_daily:
            self._request_timestamps.popleft()
        while self._token_timestamps and self._token_timestamps[0][0] <= oldest_daily:
            _timestamp, tokens = self._token_timestamps.popleft()
            self._daily_token_total = max(0, self._daily_token_total - tokens)

        oldest_minute = now - RATE_LIMIT_WINDOW_SECONDS
        while (
            self._minute_request_timestamps
            and self._minute_request_timestamps[0] <= oldest_minute
        ):
            self._minute_request_timestamps.popleft()
        while (
            self._minute_token_timestamps
            and self._minute_token_timestamps[0][0] <= oldest_minute
        ):
            _timestamp, tokens = self._minute_token_timestamps.popleft()
            self._minute_token_total = max(0, self._minute_token_total - tokens)

    def _local_usage(self, now: float) -> tuple[int, int, int, int]:
        """Return local request/token usage for minute and day windows."""
        self._prune_local_usage(now)
        minute_requests = len(self._minute_request_timestamps)
        daily_requests = len(self._request_timestamps)
        return (
            minute_requests,
            daily_requests,
            self._minute_token_total,
            self._daily_token_total,
        )

    def _check_local_free_tier_limit(
        self, text: str, model: str | None = None, now: float | None = None
    ) -> int:
        """Raise before sending a request that would exceed local free-tier usage."""
        token_estimate = self._estimate_token_usage(text)
        if not self._protect_free_tier:
            return token_estimate
        limits = self._free_tier_limits(model)
        if limits is None:
            return token_estimate

        now = now if now is not None else asyncio.get_running_loop().time()
        self._prune_local_usage(now)
        # This is an optimistic local guard. Groq remains the source of truth,
        # but blocking obvious free-tier overages avoids unnecessary API calls.
        minute_requests, daily_requests, minute_tokens, daily_tokens = (
            self._local_usage(now)
        )
        if minute_requests >= limits["requests_per_minute"]:
            raise GroqRateLimitError(
                "Groq free tier guard blocked this TTS request before sending it: "
                f"local usage reached {limits['requests_per_minute']} requests per minute."
            )
        if daily_requests >= limits["requests_per_day"]:
            raise GroqRateLimitError(
                "Groq free tier guard blocked this TTS request before sending it: "
                f"local usage reached {limits['requests_per_day']} requests per day."
            )
        if minute_tokens + token_estimate > limits["tokens_per_minute"]:
            raise GroqRateLimitError(
                "Groq free tier guard blocked this TTS request before sending it: "
                f"estimated text usage would exceed {limits['tokens_per_minute']} tokens per minute."
            )
        if daily_tokens + token_estimate > limits["tokens_per_day"]:
            raise GroqRateLimitError(
                "Groq free tier guard blocked this TTS request before sending it: "
                f"estimated text usage would exceed {limits['tokens_per_day']} tokens per day."
            )
        return token_estimate

    def _record_local_usage(
        self,
        token_estimate: int,
        model: str | None = None,
        now: float | None = None,
    ) -> None:
        """Record an uncached API attempt for local free-tier accounting."""
        if not self._protect_free_tier or self._free_tier_limits(model) is None:
            return
        now = now if now is not None else asyncio.get_running_loop().time()
        self._request_timestamps.append(now)
        self._token_timestamps.append((now, token_estimate))
        self._minute_request_timestamps.append(now)
        self._minute_token_timestamps.append((now, token_estimate))
        self._daily_token_total += token_estimate
        self._minute_token_total += token_estimate
        self._prune_local_usage(now)

    @staticmethod
    def _rate_limit_message(headers: Mapping[str, str]) -> str:
        """Build a user-facing message from Groq rate-limit response headers."""
        retry_after = headers.get("retry-after")
        reset_requests = headers.get("x-ratelimit-reset-requests")
        remaining_requests = headers.get("x-ratelimit-remaining-requests")
        remaining_tokens = headers.get("x-ratelimit-remaining-tokens")
        details: list[str] = []
        if retry_after:
            details.append(f"retry after {retry_after} seconds")
        if reset_requests:
            details.append(f"request window resets in {reset_requests}")
        if remaining_requests is not None:
            details.append(f"{remaining_requests} daily requests remaining")
        if remaining_tokens is not None:
            details.append(f"{remaining_tokens} tokens remaining this minute")
        suffix = f" ({'; '.join(details)})" if details else ""
        return f"Groq API rate limit exceeded{suffix}."

    async def async_get_tts(
        self,
        hass: HomeAssistant,
        text: str,
        voice: str | None = None,
        model: str | None = None,
        response_format: str | None = None,
    ) -> AudioResponse:
        """Asynchronous TTS request using aiohttp for Groq API."""
        if voice is None:
            voice = self._voice
        if model is None:
            model = self._model
        if response_format is None:
            response_format = self._response_format

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        # Provide a clear, integration-specific user agent with version
        headers["User-Agent"] = f"homeassistant-groq/{VERSION}"

        data = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
        }

        cache_key = (model, voice, response_format, text)
        if cache_key in self._cache:
            text_hash = sha256(text.encode("utf-8")).hexdigest()[:12]
            _LOGGER.debug(
                "Returning cached audio for model=%s voice=%s format=%s text_hash=%s",
                model,
                voice,
                response_format,
                text_hash,
            )
            self._cache.move_to_end(cache_key)
            return AudioResponse(self._cache[cache_key])

        max_retries = 1
        attempt = 0

        if self._session is None:
            self._session = async_get_clientsession(hass)
        session = self._session

        while True:
            try:
                token_estimate = self._check_local_free_tier_limit(text, model)
                self._record_local_usage(token_estimate, model)
                async with session.post(
                    self._url, json=data, headers=headers, timeout=TTS_REQUEST_TIMEOUT
                ) as resp:
                    content = await resp.read()
                    ctype = resp.headers.get("content-type", "").lower()
                    # Treat non-2xx as errors; try to parse JSON body for details
                    if resp.status < 200 or resp.status >= 300:
                        if resp.status in (401, 403):
                            raise ConfigEntryAuthFailed(
                                "Authentication failed for Groq API"
                            )
                        if resp.status == 429:
                            raise GroqRateLimitError(
                                self._rate_limit_message(resp.headers)
                            )
                        if resp.status >= 500:
                            self._mark_unavailable(
                                f"Groq TTS API returned HTTP {resp.status}"
                            )
                        try:
                            if ctype.startswith("application/json"):
                                payload = json.loads(content)
                                detail = payload.get("error") or payload
                                if _payload_mentions_model_access(payload):
                                    async_create_model_access_issue(hass, model)
                                raise HomeAssistantError(
                                    f"Groq API error (HTTP {resp.status}): {detail}"
                                )
                            raise HomeAssistantError(
                                f"Groq API error (HTTP {resp.status})"
                            )
                        except HomeAssistantError:
                            raise
                        except Exception:
                            raise HomeAssistantError(
                                f"Groq API error (HTTP {resp.status})"
                            )
                    # If JSON arrives on 2xx, check for embedded error structure
                    if ctype.startswith("application/json"):
                        try:
                            error_json = json.loads(content)
                        except Exception:
                            error_json = {}
                        if isinstance(error_json, dict) and "error" in error_json:
                            msg = error_json["error"].get(
                                "message", str(error_json["error"])
                            )
                            _LOGGER.error("Groq API error: %s", msg)
                            raise HomeAssistantError(f"Groq API error: {msg}")
                        # Unexpected JSON with 2xx: treat as error if not explicitly successful
                        raise HomeAssistantError(
                            "Groq API returned JSON but no audio content"
                        )
                    # Guard against unexpected content types on 2xx
                    if not (
                        ctype.startswith("audio/")
                        or ctype.startswith("application/octet-stream")
                    ):
                        raise HomeAssistantError(
                            f"Unexpected content-type from Groq API: {ctype}"
                        )
                    self._mark_available()
                    # Cache successful audio payloads with LRU eviction. Failed
                    # responses are intentionally excluded so retries can recover.
                    self._cache[cache_key] = content
                    if len(self._cache) > self._cache_max:
                        self._cache.popitem(last=False)
                    return AudioResponse(content)
            except CancelledError:
                _LOGGER.debug("TTS request cancelled")
                raise
            except ConfigEntryAuthFailed:
                # Bubble up to trigger reauth flow
                raise
            except HomeAssistantError:
                raise
            except (aiohttp.ClientError, HTTPError, URLError) as net_err:
                status_code = getattr(net_err, "status", None) or getattr(
                    net_err, "code", None
                )
                error_body = getattr(net_err, "message", None)
                self._mark_unavailable("Network error calling Groq TTS API")
                error_hint = ""
                if error_body and "1010" in str(error_body):
                    error_hint = " (Check model access in the Groq Console and confirm the selected TTS model is enabled for your account.)"
                    async_create_model_access_issue(hass, model)
                if attempt < max_retries:
                    attempt += 1
                    await asyncio.sleep(1)
                    _LOGGER.debug("Retrying HTTP call (attempt %d)", attempt + 1)
                    continue
                raise HomeAssistantError(
                    f"Network error occurred while fetching TTS audio (HTTP {status_code}): {error_body}{error_hint}"
                ) from net_err
            except Exception as exc:
                _LOGGER.exception(
                    "Unknown error in async_get_tts on attempt %d", attempt + 1
                )
                if attempt < max_retries:
                    attempt += 1
                    await asyncio.sleep(1)
                    _LOGGER.debug("Retrying HTTP call (attempt %d)", attempt + 1)
                    continue
                raise HomeAssistantError(
                    "An unknown error occurred while fetching TTS audio"
                ) from exc

    def close(self) -> None:
        # Use HA-managed session; do not close here to avoid impacting other integrations
        return None

    @staticmethod
    def get_supported_langs() -> list[str]:
        """Return supported language codes for Groq."""
        return [
            "ar",
            "en",
        ]
