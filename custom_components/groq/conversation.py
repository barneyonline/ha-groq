"""Conversation support for Groq text generation services."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    AssistantContent,
    ConversationEntity,
    ConversationInput,
    ConversationResult,
)
from homeassistant.components.conversation.chat_log import AssistantContentDeltaDict
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .api import GroqApiClient, TextGenerationRequest
from .chat import (
    MAX_TOOL_ITERATIONS,
    _assistant_native,
    _async_chat_log_messages,
    _chat_log_tools,
    _result_tool_calls,
)
from .const import CONF_SUBENTRY_ID, DOMAIN
from .entity import service_device_info
from .errors import translated_error
from .model_registry import GroqCapability, GroqModelRegistry
from .runtime import async_get_runtime
from .text_generation import (
    compound_builtin_tools_error_message,
    request_body_options_error_message,
    request_context_window_error,
    service_generation_options,
    service_model,
    service_name,
    service_protect_free_tier,
    service_stream,
    service_system_prompt,
    service_unique_id,
    text_generation_service_data,
)
from .types import GroqConfigEntry

PARALLEL_UPDATES = 1


def _selected_llm_api(
    config_entry: GroqConfigEntry,
    service_data: dict[str, Any],
) -> str | list[str] | None:
    """Return the selected Home Assistant LLM API for a Groq service."""
    value = service_data.get(CONF_LLM_HASS_API)
    if value is None:
        value = config_entry.options.get(CONF_LLM_HASS_API)
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return None


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: GroqConfigEntry,
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
        config_entry: GroqConfigEntry,
        service_data: dict[str, Any],
        client: GroqApiClient,
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
        if _selected_llm_api(config_entry, service_data):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return "*"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device information."""
        return service_device_info(
            service_unique_id(self._config_entry, self._service_data),
            service_model(self._config_entry, self._service_data),
            self._service_name,
        )

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> ConversationResult:
        """Generate an Assist response."""
        system_prompt = service_system_prompt(self._config_entry, self._service_data)
        request_system_prompt: str | None = system_prompt
        if hasattr(chat_log, "async_provide_llm_data") and hasattr(
            user_input, "as_llm_context"
        ):
            try:
                await chat_log.async_provide_llm_data(
                    user_input.as_llm_context(DOMAIN),
                    _selected_llm_api(self._config_entry, self._service_data),
                    system_prompt,
                    user_input.extra_system_prompt,
                )
            except conversation.ConverseError as err:
                return err.as_conversation_result()
            request_system_prompt = None
        elif user_input.extra_system_prompt:
            system_prompt = f"{system_prompt}\n\n{user_input.extra_system_prompt}"
            request_system_prompt = system_prompt

        tools = _chat_log_tools(chat_log)
        model = service_model(self._config_entry, self._service_data)
        if tools and not self._model_registry.supports(
            model, GroqCapability.TOOL_CALLING
        ):
            raise translated_error(
                f"Groq model {model} is not known to support Home Assistant tool calls",
                "unsupported_model",
                model=model,
                feature="Home Assistant tool calls",
            )
        text = ""
        use_streaming = (
            not tools
            and service_stream(self._config_entry, self._service_data)
            and hasattr(chat_log, "async_add_delta_content_stream")
        )
        attachment_cache: dict[tuple[str, str], Any] = {}
        for _iteration in range(MAX_TOOL_ITERATIONS):
            request = await self._async_text_generation_request(
                user_input,
                chat_log,
                request_system_prompt,
                tools,
                attachment_cache,
            )
            if error := request_body_options_error_message(
                self._model_registry,
                request.model,
                request.extra_body,
            ):
                raise translated_error(error, "invalid_request_options")
            if error := compound_builtin_tools_error_message(
                self._model_registry,
                request.model,
                request.compound_builtin_tools,
            ):
                raise translated_error(error, "invalid_request_options")
            if error := request_context_window_error(self._model_registry, request):
                raise translated_error(error, "invalid_request_options")
            if use_streaming:
                text = await self._async_stream_message(user_input, chat_log, request)
                break

            result = await self._client.async_generate_text(request)
            text = result.text
            tool_calls = _result_tool_calls(result)
            assistant_content = AssistantContent(
                agent_id=user_input.agent_id,
                content=text or None,
                thinking_content=getattr(result, "reasoning", None),
                tool_calls=tool_calls or None,
                native=_assistant_native(result) or None,
            )
            if tool_calls and hasattr(chat_log, "async_add_assistant_content"):
                async for _content in chat_log.async_add_assistant_content(
                    assistant_content
                ):
                    pass
            else:
                chat_log.async_add_assistant_content_without_tools(assistant_content)
            if not getattr(chat_log, "unresponded_tool_results", False):
                break
        else:
            raise translated_error(
                "Groq Assist exceeded the tool-call limit", "tool_call_limit"
            )

        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(text)
        return ConversationResult(
            response=response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=getattr(chat_log, "continue_conversation", True),
        )

    async def _async_text_generation_request(
        self,
        user_input: ConversationInput,
        chat_log: conversation.ChatLog,
        system_prompt: str | None,
        tools: list[dict[str, Any]] | None,
        attachment_cache: dict[tuple[str, str], Any] | None = None,
    ) -> TextGenerationRequest:
        """Build a Groq text generation request for an Assist turn."""
        return TextGenerationRequest(
            prompt=user_input.text,
            model=(model := service_model(self._config_entry, self._service_data)),
            messages=await _async_chat_log_messages(
                self.hass,
                self._model_registry,
                model,
                chat_log,
                user_input.text,
                getattr(user_input, "attachments", None),
                attachment_cache,
            ),
            system_prompt=system_prompt,
            **service_generation_options(
                self._config_entry, self._service_data, self._model_registry
            ),
            service_id=service_unique_id(self._config_entry, self._service_data),
            protect_free_tier=service_protect_free_tier(
                self._config_entry, self._service_data
            ),
            tools=tools,
            tool_choice="auto" if tools else None,
        )

    async def _async_stream_message(
        self,
        user_input: ConversationInput,
        chat_log: conversation.ChatLog,
        request: TextGenerationRequest,
    ) -> str:
        """Stream an Assist response into Home Assistant's chat log."""
        chunks: list[str] = []

        async def content_stream() -> AsyncIterator[AssistantContentDeltaDict]:
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
            if isinstance(content, AssistantContent) and content.content:
                completed.append(content.content)
        return "".join(completed) or "".join(chunks)
