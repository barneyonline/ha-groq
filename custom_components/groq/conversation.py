"""Conversation support for Groq text generation services."""

from __future__ import annotations

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    AssistantContent,
    ConversationEntity,
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers import intent

from .api import TextGenerationRequest
from .const import DOMAIN
from .runtime import async_get_runtime
from .text_generation import (
    service_api_key,
    service_include_reasoning,
    service_max_tokens,
    service_model,
    service_name,
    service_reasoning_effort,
    service_reasoning_format,
    service_request_body_options,
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


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Groq Assist conversation entities from text generation services."""
    runtime = await async_get_runtime(hass, config_entry)
    entities = [
        GroqConversationEntity(hass, config_entry, service_data, runtime.client)
        for service_data in text_generation_service_data(config_entry)
    ]
    if entities:
        async_add_entities(entities)


class GroqConversationEntity(ConversationEntity):
    """Groq conversation agent backed by a text generation service."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_supports_streaming = True

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        service_data: dict,
        client,
    ) -> None:
        """Initialize the conversation entity."""
        self.hass = hass
        self._config_entry = config_entry
        self._service_data = service_data
        self._client = client
        self._attr_name = service_name(config_entry, service_data)
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
            "name": self._attr_name,
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
            system_prompt=system_prompt,
            temperature=service_temperature(self._config_entry, self._service_data),
            max_tokens=service_max_tokens(self._config_entry, self._service_data),
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
                self._config_entry, self._service_data
            ),
            api_key=service_api_key(self._config_entry, self._service_data),
        )
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

        async def content_stream():
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
