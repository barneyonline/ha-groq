"""Conversation support for Groq text generation services."""

from __future__ import annotations

from contextlib import aclosing
from typing import Any, Literal

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationInput,
    ConversationResult,
)
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .api import GroqApiClient, TextGenerationRequest
from .chat import (
    MAX_TOOL_ITERATIONS,
    ChatHistoryCache,
    _async_chat_log_messages,
    _chat_log_tools,
    assistant_content_from_result,
    prepare_tool_round,
    tool_calls_for_completion,
)
from .const import CONF_SUBENTRY_ID, DOMAIN
from .entity import service_device_info
from .errors import GroqResponseError, translated_error
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
from .tool_tasks import async_add_tool_batch, async_record_chat_completion
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
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                _selected_llm_api(self._config_entry, self._service_data),
                system_prompt,
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

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
        use_streaming = service_stream(self._config_entry, self._service_data)
        history_cache = ChatHistoryCache()
        for iteration in range(MAX_TOOL_ITERATIONS):
            request = await self._async_text_generation_request(
                user_input,
                chat_log,
                tools,
                history_cache,
            )
            request, final_tool_round = prepare_tool_round(request, iteration)
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
            if error := request_context_window_error(
                self._model_registry,
                request,
                string_estimates=history_cache.string_estimates,
            ):
                raise translated_error(error, "invalid_request_options")
            if use_streaming:
                text = await self._async_stream_message(
                    user_input, chat_log, request, final_tool_round=final_tool_round
                )
                if not chat_log.unresponded_tool_results:
                    break
                continue

            result = await self._client.async_generate_text(request)
            text = result.text
            await async_record_chat_completion(
                chat_log,
                request,
                result,
                user_input.agent_id,
                final_tool_round=final_tool_round,
            )
            if not chat_log.unresponded_tool_results:
                break

        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(text)
        return ConversationResult(
            response=response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

    async def _async_text_generation_request(
        self,
        user_input: ConversationInput,
        chat_log: conversation.ChatLog,
        tools: list[dict[str, Any]] | None,
        history_cache: ChatHistoryCache | None = None,
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
                history_cache,
            ),
            system_prompt=None,
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
        *,
        final_tool_round: bool = False,
    ) -> str:
        """Stream an Assist response into Home Assistant's chat log."""
        # Forward text as it arrives, then dispatch completed tools through the
        # ChatLog API that accepts a caller-owned task map.
        if listener := chat_log.delta_listener:
            listener(chat_log, {"role": "assistant"})
        async with aclosing(self._client.async_stream_chat(request)) as chunks:
            async for chunk in chunks:
                if isinstance(chunk, str):
                    if listener := chat_log.delta_listener:
                        listener(chat_log, {"content": chunk})
                else:
                    result = chunk
                    break
            else:
                raise GroqResponseError("Groq stream returned no final result")

        tool_calls = tool_calls_for_completion(
            request, result, final_tool_round=final_tool_round
        )
        assistant_content = assistant_content_from_result(
            result, user_input.agent_id, tool_calls
        )
        if listener := chat_log.delta_listener:
            listener(
                chat_log,
                {
                    "tool_calls": assistant_content.tool_calls,
                    "thinking_content": assistant_content.thinking_content,
                },
            )
        if tool_calls:
            await async_add_tool_batch(chat_log, assistant_content, notify_deltas=True)
        elif (
            assistant_content.content
            or assistant_content.thinking_content
            or assistant_content.native
        ):
            async with aclosing(
                chat_log.async_add_assistant_content(assistant_content)
            ) as contents:
                await anext(contents, None)
        return result.text
