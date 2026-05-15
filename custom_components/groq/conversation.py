"""Conversation support for Groq text generation services."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    AssistantContent,
    ConversationEntity,
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers import intent

from .api import TextGenerationRequest
from .const import CONF_SUBENTRY_ID, DOMAIN
from .model_registry import GroqModelRegistry
from .runtime import async_get_runtime
from .text_generation import (
    request_body_options_error_message,
    service_include_reasoning,
    service_max_tokens,
    service_model,
    service_name,
    service_protect_free_tier,
    service_reasoning_effort,
    service_reasoning_format,
    service_request_body_options,
    request_context_window_error,
    service_seed,
    service_service_tier,
    service_stop,
    service_stream,
    service_system_prompt,
    service_temperature,
    service_top_p,
    service_unique_id,
    text_generation_service_data,
)

PARALLEL_UPDATES = 1
MAX_HISTORY_MESSAGES = 12


def _content_role(content: Any) -> str | None:
    """Return a chat role for a Home Assistant chat-log content item."""
    if isinstance(content, dict):
        role = content.get("role")
        return role if role in {"user", "assistant"} else None
    role = getattr(content, "role", None)
    if role in {"user", "assistant"}:
        return role
    class_name = content.__class__.__name__.lower()
    if "assistant" in class_name:
        return "assistant"
    if "user" in class_name:
        return "user"
    return None


def _content_text(content: Any) -> str | None:
    """Return text for a Home Assistant chat-log content item."""
    if isinstance(content, dict):
        text = content.get("content") or content.get("text")
    else:
        text = getattr(content, "content", None) or getattr(content, "text", None)
    return text if isinstance(text, str) and text else None


def _chat_log_messages(
    chat_log: conversation.ChatLog,
    current_text: str,
) -> list[dict[str, str]]:
    """Return OpenAI-compatible messages from the chat log plus current input."""
    history: Sequence[Any] = ()
    for attr in ("content", "messages"):
        value = getattr(chat_log, attr, None)
        if isinstance(value, (list, tuple)):
            history = value
            break
    messages: list[dict[str, str]] = []
    for item in history:
        role = _content_role(item)
        text = _content_text(item)
        if role and text:
            messages.append({"role": role, "content": text})
    if len(messages) > MAX_HISTORY_MESSAGES:
        messages = messages[-MAX_HISTORY_MESSAGES:]
    if not messages or messages[-1] != {"role": "user", "content": current_text}:
        messages.append({"role": "user", "content": current_text})
    return messages


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Groq Assist conversation entities from text generation services."""
    runtime = await async_get_runtime(hass, config_entry)
    for service_data in text_generation_service_data(config_entry):
        async_add_entities(
            [
                GroqConversationEntity(
                    hass,
                    config_entry,
                    service_data,
                    runtime.client,
                    runtime.model_registry,
                )
            ],
            config_subentry_id=service_data.get(CONF_SUBENTRY_ID),
        )


class GroqConversationEntity(ConversationEntity):
    """Groq conversation agent backed by a text generation service."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_supports_streaming = True
    _attr_translation_key = "assist"

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        service_data: dict[str, Any],
        client: Any,
        model_registry: GroqModelRegistry | None = None,
    ) -> None:
        """Initialize the conversation entity."""
        self.hass = hass
        self._config_entry = config_entry
        self._service_data = service_data
        self._client = client
        self._model_registry = model_registry or GroqModelRegistry()
        self._service_name = service_name(config_entry, service_data)
        self._attr_unique_id = f"{service_unique_id(config_entry, service_data)}_assist"

    @property
    def supported_languages(self) -> list[str] | str:
        """Return supported languages."""
        return "*"

    @property
    def device_info(self) -> dict:
        """Return device information."""
        return {
            "identifiers": {
                (DOMAIN, service_unique_id(self._config_entry, self._service_data))
            },
            "manufacturer": "Groq",
            "model": service_model(self._config_entry, self._service_data),
            "name": self._service_name,
        }

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> ConversationResult:
        """Generate an Assist response."""
        system_prompt = service_system_prompt(self._config_entry, self._service_data)
        if user_input.extra_system_prompt:
            system_prompt = f"{system_prompt}\n\n{user_input.extra_system_prompt}"

        request = TextGenerationRequest(
            prompt=user_input.text,
            model=service_model(self._config_entry, self._service_data),
            messages=_chat_log_messages(chat_log, user_input.text),
            system_prompt=system_prompt,
            temperature=service_temperature(self._config_entry, self._service_data),
            max_tokens=service_max_tokens(
                self._config_entry,
                self._service_data,
                self._model_registry,
            ),
            top_p=service_top_p(self._config_entry, self._service_data),
            stop=service_stop(self._config_entry, self._service_data),
            seed=service_seed(self._config_entry, self._service_data),
            service_tier=service_service_tier(self._config_entry, self._service_data),
            reasoning_effort=service_reasoning_effort(
                self._config_entry, self._service_data
            ),
            reasoning_format=service_reasoning_format(
                self._config_entry, self._service_data
            ),
            include_reasoning=service_include_reasoning(
                self._config_entry, self._service_data
            ),
            extra_body=service_request_body_options(
                self._config_entry,
                self._service_data,
                self._model_registry,
            ),
            service_id=service_unique_id(self._config_entry, self._service_data),
            protect_free_tier=service_protect_free_tier(
                self._config_entry, self._service_data
            ),
        )
        if error := request_body_options_error_message(
            self._model_registry,
            request.model,
            request.extra_body,
        ):
            raise HomeAssistantError(error)
        if error := request_context_window_error(self._model_registry, request):
            raise HomeAssistantError(error)
        if service_stream(self._config_entry, self._service_data) and hasattr(
            chat_log, "async_add_delta_content_stream"
        ):
            text = await self._async_stream_message(user_input, chat_log, request)
        else:
            result = await self._client.async_generate_text(request)
            text = result.text
            chat_log.async_add_assistant_content_without_tools(
                AssistantContent(user_input.agent_id, text)
            )

        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(text)
        return ConversationResult(
            response=response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=True,
        )

    async def _async_stream_message(
        self,
        user_input: ConversationInput,
        chat_log: conversation.ChatLog,
        request: TextGenerationRequest,
    ) -> str:
        """Stream an Assist response into Home Assistant's chat log."""
        chunks: list[str] = []

        async def content_stream() -> AsyncIterator[dict[str, str]]:
            yield {"role": "assistant"}
            async for chunk in self._client.async_stream_text(request):
                chunks.append(chunk)
                yield {"content": chunk}

        completed: list[str] = []
        # Home Assistant yields the completed assistant content back from the
        # stream helper on recent versions. Keep the raw chunk buffer as a
        # compatibility fallback for versions that only consume the stream.
        async for content in chat_log.async_add_delta_content_stream(
            user_input.agent_id,
            content_stream(),
        ):
            if content.content:
                completed.append(content.content)
        return "".join(completed) or "".join(chunks)
