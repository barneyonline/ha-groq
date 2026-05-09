"""Shared OpenAI-compatible Groq API client."""

from __future__ import annotations

import json
import logging
from asyncio import CancelledError
from dataclasses import dataclass
from typing import Any, AsyncIterator
from urllib.parse import urljoin

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import VERSION
from .errors import GroqApiError, GroqResponseError
from .model_registry import GroqModel, model_from_api
from .rate_limit import GroqRateLimiter

_LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_COMPLETIONS_PATH = "/chat/completions"
MODELS_PATH = "/models"
AUDIO_TRANSCRIPTIONS_PATH = "/audio/transcriptions"
JSON_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=60)
STREAM_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=60)


@dataclass(frozen=True, slots=True)
class TextGenerationRequest:
    """Request data for text generation."""

    prompt: str
    model: str
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
    messages: list[dict[str, str]] = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.append({"role": "user", "content": request.prompt})

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
        payload.update(
            {
                key: value
                for key, value in request.extra_body.items()
                if value is not None
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
    ) -> None:
        self._hass = hass
        self._api_key = api_key
        self._base_url = normalize_base_url(base_url)
        self._session = session

    @property
    def base_url(self) -> str:
        """Return the normalized API base URL."""
        return self._base_url

    async def async_list_models(self) -> list[GroqModel]:
        """Return models visible to the configured Groq API key."""
        payload = await self._request_json("GET", MODELS_PATH)
        data = payload.get("data", [])
        if not isinstance(data, list):
            raise GroqResponseError("Groq models response did not include a data list")
        return [model_from_api(item) for item in data if isinstance(item, dict)]

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
        )
        text = payload.get("text")
        if not isinstance(text, str):
            raise GroqResponseError("Groq transcription response did not include text")
        return text

    def _chat_result(self, payload: dict[str, Any]) -> ChatCompletionResult:
        """Normalize a chat completion response."""
        return ChatCompletionResult(
            text=extract_chat_text(payload),
            model=payload.get("model"),
            usage=(
                payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
            ),
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
    ) -> dict[str, Any]:
        """Perform a JSON request and return a JSON object."""
        session = self._session or async_get_clientsession(self._hass)
        headers = self._headers(api_key, content_type=content_type)
        request_kwargs: dict[str, Any] = {
            "headers": headers,
            "timeout": JSON_REQUEST_TIMEOUT,
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
                payload = self._decode_json(body)
                if response.status == 429:
                    GroqRateLimiter.raise_for_headers(response.headers, payload)
                if response.status in (401, 403):
                    raise ConfigEntryAuthFailed("Authentication failed for Groq API")
                if response.status < 200 or response.status >= 300:
                    raise self._api_error(response.status, payload)
                if not isinstance(payload, dict):
                    raise GroqResponseError("Groq API returned non-object JSON")
                return payload
        except CancelledError:
            raise
        except (GroqApiError, ConfigEntryAuthFailed):
            raise
        except aiohttp.ClientError as err:
            raise GroqApiError(f"Network error calling Groq API: {err}") from err

    async def _request_stream(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any],
        api_key: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Perform an SSE request and yield decoded JSON events."""
        session = self._session or async_get_clientsession(self._hass)
        try:
            async with session.request(
                method,
                self._url(path),
                json=json_payload,
                headers=self._headers(api_key),
                timeout=STREAM_REQUEST_TIMEOUT,
            ) as response:
                if response.status in (401, 403):
                    raise ConfigEntryAuthFailed("Authentication failed for Groq API")
                if response.status < 200 or response.status >= 300:
                    body = await response.read()
                    payload = self._decode_json(body)
                    if response.status == 429 and isinstance(payload, dict):
                        GroqRateLimiter.raise_for_headers(response.headers, payload)
                    raise self._api_error(response.status, payload)

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
        except aiohttp.ClientError as err:
            raise GroqApiError(f"Network error calling Groq API: {err}") from err

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
    def _decode_json(body: bytes) -> dict[str, Any] | list[Any]:
        """Decode a JSON response body."""
        try:
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            raise GroqResponseError("Groq API returned invalid JSON") from err

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
