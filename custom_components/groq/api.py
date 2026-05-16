"""Shared OpenAI-compatible Groq API client."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import logging
from asyncio import CancelledError
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable
from urllib.parse import quote, urljoin

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed

from .const import VERSION
from .errors import GroqApiError, GroqResponseError
from .model_registry import GroqModel, model_from_api
from .rate_limit import GroqRateLimiter
from .repairs import async_create_model_access_issue

_LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_COMPLETIONS_PATH = "/chat/completions"
MODELS_PATH = "/models"
AUDIO_TRANSCRIPTIONS_PATH = "/audio/transcriptions"
JSON_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=60)
STREAM_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=60)
MODEL_DETAIL_CONCURRENCY = 5
RESERVED_CHAT_BODY_OPTIONS = frozenset({"messages", "model", "stream"})

_CLIENTSESSION_FACTORY: Callable[[HomeAssistant], aiohttp.ClientSession] | None = None


def _load_clientsession_factory() -> Callable[[HomeAssistant], aiohttp.ClientSession]:
    """Import and return Home Assistant's shared session helper."""
    from homeassistant.helpers.aiohttp_client import async_get_clientsession as _get

    return _get


async def async_preload_clientsession_helper(hass: HomeAssistant) -> None:
    """Load the session helper before request handling reaches latency-sensitive code."""
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


@dataclass(frozen=True, slots=True)
class TextGenerationRequest:
    """Request data for text generation."""

    prompt: str
    model: str
    messages: list[dict[str, str]] | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    service_tier: str | None = None
    reasoning_effort: str | None = None
    reasoning_format: str | None = None
    include_reasoning: bool | None = None
    reasoning: bool = False
    stream: bool = False
    extra_body: dict[str, Any] | None = None
    api_key: str | None = None
    service_id: str | None = None
    protect_free_tier: bool = True


@dataclass(frozen=True, slots=True)
class StructuredGenerationRequest(TextGenerationRequest):
    """Request data for structured JSON generation."""

    schema: dict[str, Any] | None = None
    schema_name: str = "response"
    strict: bool = False


@dataclass(frozen=True, slots=True)
class VisionRequest(TextGenerationRequest):
    """Request data for vision analysis."""

    image_url: str = ""


@dataclass(frozen=True, slots=True)
class ChatCompletionResult:
    """Normalized chat completion response."""

    text: str
    model: str | None
    usage: dict[str, Any]
    raw: dict[str, Any]
    reasoning: str | None = None
    executed_tools: list[dict[str, Any]] | None = None
    usage_breakdown: dict[str, Any] | None = None

    @property
    def content(self) -> str:
        """Return generated text content."""
        return self.text


StructuredOutputRequest = StructuredGenerationRequest


def normalize_base_url(url: str | None) -> str:
    """Normalize configured Groq URL to the OpenAI-compatible base URL."""
    if not url:
        return DEFAULT_BASE_URL
    cleaned = url.rstrip("/")
    if cleaned.endswith("/audio/speech"):
        cleaned = cleaned.removesuffix("/audio/speech")
    return cleaned


def build_text_generation_payload(request: TextGenerationRequest) -> dict[str, Any]:
    """Build an OpenAI-compatible chat completion payload."""
    messages: list[dict[str, str]] = (
        list(request.messages)
        if request.messages is not None
        else [{"role": "user", "content": request.prompt}]
    )
    if request.system_prompt:
        messages.insert(0, {"role": "system", "content": request.system_prompt})

    payload: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
    }
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.max_tokens is not None:
        payload["max_completion_tokens"] = request.max_tokens
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.stop:
        payload["stop"] = request.stop
    if request.seed is not None:
        payload["seed"] = request.seed
    if request.service_tier:
        payload["service_tier"] = request.service_tier
    if request.reasoning_effort:
        payload["reasoning_effort"] = request.reasoning_effort
    if request.reasoning_format:
        payload["reasoning_format"] = request.reasoning_format
    elif request.include_reasoning is not None:
        payload["include_reasoning"] = request.include_reasoning
    if request.stream:
        payload["stream"] = True
    if request.extra_body:
        # Merge last so the advanced passthrough can cover newly added Groq
        # chat-create options without changing the integration schema first.
        # Keep integration-managed fields out of the passthrough so a raw body
        # option cannot bypass model, prompt, or streaming validation.
        payload.update(
            {
                key: value
                for key, value in request.extra_body.items()
                if value is not None and key not in RESERVED_CHAT_BODY_OPTIONS
            }
        )
    return payload


def build_structured_generation_payload(
    request: StructuredGenerationRequest,
) -> dict[str, Any]:
    """Build a chat completion payload for structured JSON output."""
    payload = build_text_generation_payload(request)
    if request.schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": request.schema_name,
                "schema": request.schema,
                "strict": request.strict,
            },
        }
    else:
        payload["response_format"] = {"type": "json_object"}
    return payload


def build_vision_payload(request: VisionRequest) -> dict[str, Any]:
    """Build an OpenAI-compatible multimodal chat completion payload."""
    payload = build_text_generation_payload(request)
    payload["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": request.prompt},
                {"type": "image_url", "image_url": {"url": request.image_url}},
            ],
        }
    ]
    if request.system_prompt:
        payload["messages"].insert(
            0,
            {"role": "system", "content": request.system_prompt},
        )
    return payload


def extract_chat_text(payload: dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI-compatible chat response."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise GroqResponseError("Groq response did not include choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise GroqResponseError("Groq response did not include a message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        if parts:
            return "\n".join(parts)
    raise GroqResponseError("Groq response did not include text content")


def extract_chat_reasoning(payload: dict[str, Any]) -> str | None:
    """Extract reasoning text from a Groq chat response when present."""
    message = _extract_chat_message(payload)
    if message is None:
        return None
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    return reasoning if isinstance(reasoning, str) else None


def extract_executed_tools(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Extract Compound executed tool metadata when present."""
    message = _extract_chat_message(payload)
    if message is None:
        return None
    executed_tools = message.get("executed_tools")
    if not isinstance(executed_tools, list):
        return None
    return [tool for tool in executed_tools if isinstance(tool, dict)]


def _extract_chat_message(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first assistant message from a chat response."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None
    return message


class GroqApiClient:
    """Async Groq API client for OpenAI-compatible endpoints."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        api_key: str | None,
        base_url: str | None = None,
        session: aiohttp.ClientSession | None = None,
        rate_limiter: GroqRateLimiter | None = None,
        request_timeout: aiohttp.ClientTimeout | None = None,
        stream_timeout: aiohttp.ClientTimeout | None = None,
    ) -> None:
        self._hass = hass
        self._api_key = api_key
        self._base_url = normalize_base_url(base_url)
        self._session = session
        self._rate_limiter = rate_limiter or GroqRateLimiter()
        self._request_timeout = request_timeout or JSON_REQUEST_TIMEOUT
        self._stream_timeout = stream_timeout or STREAM_REQUEST_TIMEOUT
        self._available = True
        self._unavailable_reason: str | None = None

    @property
    def base_url(self) -> str:
        """Return the normalized API base URL."""
        return self._base_url

    @property
    def available(self) -> bool:
        """Return whether the last Groq API interaction succeeded."""
        return self._available

    async def async_list_models(self, *, hydrate: bool = True) -> list[GroqModel]:
        """Return models visible to the configured Groq API key."""
        payload = await self._request_json("GET", MODELS_PATH)
        data = payload.get("data")
        if not isinstance(data, list):
            raise GroqResponseError("Groq models response did not include a data list")
        model_items = [item for item in data if isinstance(item, dict)]
        models = [model_from_api(item) for item in model_items]
        if not hydrate:
            return models
        return await self._async_hydrate_models(models, model_items)

    async def async_retrieve_model(self, model_id: str) -> GroqModel:
        """Return detailed metadata for one Groq model."""
        payload = await self._request_json(
            "GET",
            f"{MODELS_PATH}/{quote(model_id, safe='')}",
        )
        return model_from_api(payload)

    async def _async_hydrate_model(
        self,
        model: GroqModel,
        raw_model: dict[str, Any] | None = None,
    ) -> GroqModel:
        """Fetch model detail when the list response lacks token limits."""
        force_detail = raw_model is not None and (
            "context_window" not in raw_model
            or "max_completion_tokens" not in raw_model
        )
        if not force_detail and (
            model.context_window is not None and model.max_completion_tokens is not None
        ):
            return model
        try:
            detail = await self.async_retrieve_model(model.model_id)
            return detail if detail.model_id == model.model_id else model
        except (GroqApiError, GroqResponseError, ConfigEntryAuthFailed) as err:
            _LOGGER.debug(
                "Could not fetch Groq model detail for %s: %s", model.model_id, err
            )
            return model

    async def _async_hydrate_models(
        self,
        models: list[GroqModel],
        raw_models: list[dict[str, Any]] | None = None,
    ) -> list[GroqModel]:
        """Fetch model details with bounded concurrency."""
        if not models:
            return []
        semaphore = asyncio.Semaphore(MODEL_DETAIL_CONCURRENCY)
        if raw_models is None:
            raw_model_items: list[dict[str, Any] | None] = [None for _model in models]
        else:
            raw_model_items = list(raw_models)

        async def hydrate(
            model: GroqModel,
            raw_model: dict[str, Any] | None,
        ) -> GroqModel:
            async with semaphore:
                return await self._async_hydrate_model(model, raw_model)

        return list(
            await asyncio.gather(
                *(
                    hydrate(model, raw_model)
                    for model, raw_model in zip(models, raw_model_items)
                )
            )
        )

    async def async_generate_text(
        self,
        request: TextGenerationRequest,
    ) -> ChatCompletionResult:
        """Generate text with the chat completions API."""
        payload = await self._request_json(
            "POST",
            CHAT_COMPLETIONS_PATH,
            json_payload=build_text_generation_payload(request),
            api_key=request.api_key,
            guard_key=self._guard_key(request),
        )
        return self._chat_result(payload)

    async def async_stream_text(
        self,
        request: TextGenerationRequest,
    ) -> AsyncIterator[str]:
        """Stream generated text chunks from the chat completions API."""
        payload = build_text_generation_payload(request)
        payload["stream"] = True
        async for event in self._request_stream(
            "POST",
            CHAT_COMPLETIONS_PATH,
            json_payload=payload,
            api_key=request.api_key,
            guard_key=self._guard_key(request),
        ):
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            # Groq uses OpenAI-compatible SSE chunks, where incremental text is
            # emitted under choices[0].delta.content.
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str) and content:
                yield content

    async def async_generate_structured(
        self,
        request: StructuredGenerationRequest,
    ) -> dict[str, Any]:
        """Generate structured JSON with the chat completions API."""
        payload = await self._request_json(
            "POST",
            CHAT_COMPLETIONS_PATH,
            json_payload=build_structured_generation_payload(request),
            api_key=request.api_key,
            guard_key=self._guard_key(request),
        )
        result = self._chat_result(payload)
        try:
            parsed = json.loads(result.text)
        except json.JSONDecodeError as err:
            raise GroqResponseError(
                "Groq structured response was not valid JSON"
            ) from err
        return {
            "text": result.text,
            "data": parsed,
            "model": result.model,
            "usage": result.usage,
            "cached": False,
        }

    async def async_analyze_image(
        self,
        request: VisionRequest,
    ) -> ChatCompletionResult:
        """Analyze an image with an OpenAI-compatible vision payload."""
        payload = await self._request_json(
            "POST",
            CHAT_COMPLETIONS_PATH,
            json_payload=build_vision_payload(request),
            api_key=request.api_key,
            guard_key=self._guard_key(request),
        )
        return self._chat_result(payload)

    async def async_transcribe_audio(
        self,
        *,
        audio: bytes,
        filename: str,
        model: str,
        language: str | None = None,
        prompt: str | None = None,
        api_key: str | None = None,
        service_id: str | None = None,
        protect_free_tier: bool = True,
    ) -> str:
        """Transcribe audio with Groq's OpenAI-compatible audio endpoint."""
        form = aiohttp.FormData()
        form.add_field("model", model)
        form.add_field("file", audio, filename=filename)
        form.add_field("response_format", "json")
        if language:
            form.add_field("language", language.split("-")[0])
        if prompt:
            form.add_field("prompt", prompt)
        payload = await self._request_json(
            "POST",
            AUDIO_TRANSCRIPTIONS_PATH,
            data=form,
            api_key=api_key,
            content_type=None,
            guard_key=service_id if protect_free_tier else None,
        )
        text = payload.get("text")
        if not isinstance(text, str):
            raise GroqResponseError("Groq transcription response did not include text")
        return text

    def _chat_result(self, payload: dict[str, Any]) -> ChatCompletionResult:
        """Normalize a chat completion response."""
        usage = payload.get("usage")
        return ChatCompletionResult(
            text=extract_chat_text(payload),
            model=payload.get("model"),
            usage=usage if isinstance(usage, dict) else {},
            raw=payload,
            reasoning=extract_chat_reasoning(payload),
            executed_tools=extract_executed_tools(payload),
            usage_breakdown=(
                payload.get("usage_breakdown")
                if isinstance(payload.get("usage_breakdown"), dict)
                else None
            ),
        )

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        data: Any = None,
        api_key: str | None = None,
        content_type: str | None = "application/json",
        guard_key: str | None = None,
    ) -> dict[str, Any]:
        """Perform a JSON request and return a JSON object."""
        session = self._session or async_get_clientsession(self._hass)
        headers = self._headers(api_key, content_type=content_type)
        self._rate_limiter.raise_if_blocked(guard_key)
        request_kwargs: dict[str, Any] = {
            "headers": headers,
            "timeout": self._request_timeout,
        }
        if json_payload is not None:
            request_kwargs["json"] = json_payload
        if data is not None:
            request_kwargs["data"] = data

        try:
            async with session.request(
                method,
                self._url(path),
                **request_kwargs,
            ) as response:
                body = await response.read()
                self._rate_limiter.update_from_headers(guard_key, response.headers)
                if response.status in (401, 403):
                    raise ConfigEntryAuthFailed("Authentication failed for Groq API")
                if response.status == 429:
                    payload = self._try_decode_json(body)
                    GroqRateLimiter.raise_for_headers(
                        response.headers,
                        payload if isinstance(payload, dict) else None,
                    )
                if response.status < 200 or response.status >= 300:
                    payload = self._try_decode_json(body) or {}
                    self._handle_http_unavailable(response.status, payload)
                    self._create_model_access_issue(
                        response.status, payload, json_payload
                    )
                    raise self._api_error(response.status, payload)
                payload = self._decode_json(body)
                if not isinstance(payload, dict):
                    self._mark_unavailable("Groq API returned non-object JSON")
                    raise GroqResponseError("Groq API returned non-object JSON")
                self._mark_available()
                return payload
        except CancelledError:
            raise
        except (GroqApiError, ConfigEntryAuthFailed):
            raise
        except (aiohttp.ClientError, TimeoutError) as err:
            self._mark_unavailable("Network error calling Groq API")
            raise GroqApiError(f"Network error calling Groq API: {err}") from err

    async def _request_stream(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any],
        api_key: str | None = None,
        guard_key: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Perform an SSE request and yield decoded JSON events."""
        session = self._session or async_get_clientsession(self._hass)
        self._rate_limiter.raise_if_blocked(guard_key)
        try:
            async with session.request(
                method,
                self._url(path),
                json=json_payload,
                headers=self._headers(api_key),
                timeout=self._stream_timeout,
            ) as response:
                self._rate_limiter.update_from_headers(guard_key, response.headers)
                if response.status in (401, 403):
                    raise ConfigEntryAuthFailed("Authentication failed for Groq API")
                if response.status < 200 or response.status >= 300:
                    body = await response.read()
                    payload = self._decode_json(body)
                    if response.status == 429 and isinstance(payload, dict):
                        GroqRateLimiter.raise_for_headers(response.headers, payload)
                    self._handle_http_unavailable(response.status, payload)
                    self._create_model_access_issue(
                        response.status, payload, json_payload
                    )
                    raise self._api_error(response.status, payload)

                self._mark_available()
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    # Each data line is a standalone JSON event; do not buffer
                    # across lines because SSE framing has already done that.
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError as err:
                        raise GroqResponseError(
                            "Groq stream returned invalid JSON"
                        ) from err
                    if isinstance(event, dict):
                        yield event
        except CancelledError:
            raise
        except (GroqApiError, ConfigEntryAuthFailed):
            raise
        except (aiohttp.ClientError, TimeoutError) as err:
            self._mark_unavailable("Network error calling Groq API")
            raise GroqApiError(f"Network error calling Groq API: {err}") from err

    def _handle_http_unavailable(
        self,
        status: int,
        payload: dict[str, Any] | list[Any],
    ) -> None:
        """Track Groq API availability for transient service-side failures."""
        if status >= 500:
            self._mark_unavailable(f"Groq API returned HTTP {status}")
        elif isinstance(payload, dict) and status == 408:
            self._mark_unavailable("Groq API request timed out")

    def _mark_unavailable(self, reason: str) -> None:
        """Log the transition to unavailable once per outage."""
        if not self._available and self._unavailable_reason == reason:
            return
        self._available = False
        self._unavailable_reason = reason
        _LOGGER.warning("%s; Groq API calls will be retried by Home Assistant", reason)

    def _mark_available(self) -> None:
        """Log recovery once after an outage."""
        if self._available:
            return
        self._available = True
        self._unavailable_reason = None
        _LOGGER.info("Groq API is reachable again")

    def _create_model_access_issue(
        self,
        status: int,
        payload: dict[str, Any] | list[Any],
        json_payload: dict[str, Any] | None,
    ) -> None:
        """Create a repair for model errors that require user action."""
        if status not in (400, 404):
            return
        model = json_payload.get("model") if isinstance(json_payload, dict) else None
        if not isinstance(model, str) or not _payload_mentions_model_access(payload):
            return
        with suppress(Exception):
            async_create_model_access_issue(self._hass, model)

    def _headers(
        self,
        api_key: str | None = None,
        *,
        content_type: str | None = "application/json",
    ) -> dict[str, str]:
        """Return request headers, optionally overriding the entry API key."""
        headers = {
            "User-Agent": f"homeassistant-groq/{VERSION}",
        }
        if content_type:
            headers["Content-Type"] = content_type
        effective_api_key = api_key or self._api_key
        if effective_api_key:
            headers["Authorization"] = f"Bearer {effective_api_key}"
        return headers

    def _url(self, path: str) -> str:
        """Return an absolute endpoint URL for a path."""
        return urljoin(f"{self._base_url.rstrip('/')}/", path.lstrip("/"))

    @staticmethod
    def _guard_key(request: TextGenerationRequest) -> str | None:
        """Return the per-service guard key for a request when protection is enabled."""
        return request.service_id if request.protect_free_tier else None

    @staticmethod
    def _decode_json(body: bytes) -> dict[str, Any] | list[Any]:
        """Decode a JSON response body."""
        try:
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            raise GroqResponseError("Groq API returned invalid JSON") from err

    @staticmethod
    def _try_decode_json(body: bytes) -> dict[str, Any] | list[Any] | None:
        """Decode JSON when available without hiding HTTP error classification."""
        try:
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _api_error(status: int, payload: dict[str, Any] | list[Any]) -> GroqApiError:
        """Build a sanitized API error from a Groq error response."""
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message") or f"Groq API error HTTP {status}"
                error_type = error.get("type")
            else:
                message = str(error or payload)
                error_type = None
            return GroqApiError(
                f"Groq API error (HTTP {status}): {message}",
                status=status,
                error_type=error_type,
                payload=payload,
            )
        _LOGGER.debug("Unexpected Groq error payload type: %s", type(payload))
        return GroqApiError(f"Groq API error (HTTP {status})", status=status)


def _payload_mentions_model_access(payload: dict[str, Any] | list[Any]) -> bool:
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
