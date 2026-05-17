"""Conversation support for Groq text generation services."""

from __future__ import annotations

import json
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
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers import intent, llm
from voluptuous_openapi import convert

from .api import TextGenerationRequest
from .attachments import async_attachment_content_parts
from .const import CONF_SUBENTRY_ID, DOMAIN
from .feature_registry import GroqFeature
from .model_registry import GroqCapability, GroqModelRegistry
from .runtime import async_get_runtime
from .text_generation import (
    compound_builtin_tools_error_message,
    request_body_options_error_message,
    service_compound_builtin_tools,
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
MAX_TOOL_ITERATIONS = 10


def _selected_llm_api(
    config_entry: ConfigEntry,
    service_data: dict[str, Any],
) -> str | list[str] | None:
    """Return the selected Home Assistant LLM API for a Groq service."""
    value = service_data.get(CONF_LLM_HASS_API)
    if value is None:
        value = config_entry.options.get(CONF_LLM_HASS_API)
    if value in (None, "", []):
        return None
    return value


def _content_role(content: Any) -> str | None:
    """Return a chat role for a Home Assistant chat-log content item."""
    if isinstance(content, dict):
        role = content.get("role")
        return role if role in {"system", "user", "assistant"} else None
    role = getattr(content, "role", None)
    if role in {"system", "user", "assistant"}:
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


def _content_attachments(content: Any) -> Any:
    """Return attachments for a Home Assistant chat-log content item."""
    if isinstance(content, dict):
        return content.get("attachments")
    return getattr(content, "attachments", None)


def _message_contains_input(
    message: dict[str, Any],
    text: str,
    *,
    current_content: str | list[dict[str, Any]] | None = None,
) -> bool:
    """Return whether an OpenAI-compatible message includes the current input."""
    if message["role"] != "user":
        return False
    content = message.get("content")
    if current_content is not None:
        return content == current_content
    if content == text:
        return True
    if isinstance(content, list):
        return any(
            isinstance(part, dict)
            and part.get("type") == "text"
            and part.get("text") == text
            for part in content
        )
    return False


def _trim_turn_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return recent messages without orphaning tool responses."""
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    trimmed = messages[-MAX_HISTORY_MESSAGES:]
    valid_tool_call_ids = {
        tool_call["id"]
        for message in trimmed
        for tool_call in message.get("tool_calls", [])
        if isinstance(tool_call, dict) and isinstance(tool_call.get("id"), str)
    }
    return [
        message
        for message in trimmed
        if message.get("role") != "tool"
        or message.get("tool_call_id") in valid_tool_call_ids
    ]


async def _message_content(
    hass: HomeAssistant,
    model_registry: GroqModelRegistry,
    model: str,
    text: str,
    attachments: Any,
) -> str | list[dict[str, Any]]:
    """Return text or multimodal content for an Assist chat message."""
    if not attachments:
        return text
    if not model_registry.supports(model, GroqFeature.VISION):
        raise HomeAssistantError(
            "Groq Assist attachments require a vision-capable model"
        )
    parts = await async_attachment_content_parts(
        hass,
        attachments,
        text=text,
    )
    return parts or text


def _tool_call_id(tool_call: Any) -> str:
    """Return a stable OpenAI-compatible tool-call id."""
    if isinstance(tool_call, dict):
        tool_id = tool_call.get("id")
        return str(tool_id) if tool_id else "tool_call"
    tool_id = getattr(tool_call, "id", None)
    return str(tool_id) if tool_id else "tool_call"


def _tool_call_message(tool_call: Any) -> dict[str, Any] | None:
    """Return an OpenAI-compatible tool call from a Home Assistant ToolInput."""
    if isinstance(tool_call, dict):
        tool_name = tool_call.get("tool_name")
        tool_args = tool_call.get("tool_args")
    else:
        tool_name = getattr(tool_call, "tool_name", None)
        tool_args = getattr(tool_call, "tool_args", None)
    if not isinstance(tool_name, str) or not isinstance(tool_args, dict):
        return None
    return {
        "id": _tool_call_id(tool_call),
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(tool_args, separators=(",", ":")),
        },
    }


def _tool_result_message(content: Any) -> dict[str, str] | None:
    """Return an OpenAI-compatible tool result message from chat-log content."""
    tool_call_id = getattr(content, "tool_call_id", None)
    tool_name = getattr(content, "tool_name", None)
    tool_result = getattr(content, "tool_result", None)
    if isinstance(content, dict):
        tool_call_id = content.get("tool_call_id", tool_call_id)
        tool_name = content.get("tool_name", tool_name)
        tool_result = content.get("tool_result", tool_result)
    if not isinstance(tool_call_id, str) or not isinstance(tool_name, str):
        return None
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": json.dumps(tool_result, separators=(",", ":"), default=str),
    }


def _chat_log_messages(
    chat_log: conversation.ChatLog,
    current_text: str,
) -> list[dict[str, Any]]:
    """Return OpenAI-compatible messages from the chat log plus current input."""
    history: Sequence[Any] = ()
    for attr in ("content", "messages"):
        value = getattr(chat_log, attr, None)
        if isinstance(value, (list, tuple)):
            history = value
            break
    messages: list[dict[str, Any]] = []
    for item in history:
        if _content_role(item) == "system":
            if text := _content_text(item):
                messages.append({"role": "system", "content": text})
            continue
        if getattr(item, "role", None) == "tool_result" or (
            isinstance(item, dict) and item.get("role") == "tool_result"
        ):
            if tool_result := _tool_result_message(item):
                messages.append(tool_result)
            continue
        role = _content_role(item)
        text = _content_text(item)
        if not role:
            continue
        message: dict[str, Any] = {"role": role, "content": text or ""}
        tool_calls = getattr(item, "tool_calls", None)
        if isinstance(item, dict):
            tool_calls = item.get("tool_calls", tool_calls)
        if role == "assistant" and tool_calls:
            converted_tool_calls = [
                tool_call_message
                for tool_call in tool_calls
                if (tool_call_message := _tool_call_message(tool_call))
            ]
            if converted_tool_calls:
                message["tool_calls"] = converted_tool_calls
        if text or message.get("tool_calls"):
            messages.append(message)
    system_messages = [message for message in messages if message["role"] == "system"]
    turn_messages = [message for message in messages if message["role"] != "system"]
    turn_messages = _trim_turn_messages(turn_messages)
    messages = [*system_messages, *turn_messages]
    has_current_input = any(
        message["role"] == "user" and message.get("content") == current_text
        for message in messages
    )
    if not has_current_input:
        messages.append({"role": "user", "content": current_text})
    return messages


async def _async_chat_log_messages(
    hass: HomeAssistant,
    model_registry: GroqModelRegistry,
    model: str,
    chat_log: conversation.ChatLog,
    current_text: str,
    current_attachments: Any = None,
) -> list[dict[str, Any]]:
    """Return OpenAI-compatible messages, including supported image attachments."""
    history: Sequence[Any] = ()
    for attr in ("content", "messages"):
        value = getattr(chat_log, attr, None)
        if isinstance(value, (list, tuple)):
            history = value
            break
    messages: list[dict[str, Any]] = []
    for item in history:
        if _content_role(item) == "system":
            if text := _content_text(item):
                messages.append({"role": "system", "content": text})
            continue
        if getattr(item, "role", None) == "tool_result" or (
            isinstance(item, dict) and item.get("role") == "tool_result"
        ):
            if tool_result := _tool_result_message(item):
                messages.append(tool_result)
            continue
        role = _content_role(item)
        text = _content_text(item)
        if not role:
            continue
        message: dict[str, Any] = {
            "role": role,
            "content": await _message_content(
                hass,
                model_registry,
                model,
                text or "",
                _content_attachments(item) if role == "user" else None,
            ),
        }
        tool_calls = getattr(item, "tool_calls", None)
        if isinstance(item, dict):
            tool_calls = item.get("tool_calls", tool_calls)
        if role == "assistant" and tool_calls:
            converted_tool_calls = [
                tool_call_message
                for tool_call in tool_calls
                if (tool_call_message := _tool_call_message(tool_call))
            ]
            if converted_tool_calls:
                message["tool_calls"] = converted_tool_calls
        if text or message.get("tool_calls") or _content_attachments(item):
            messages.append(message)
    system_messages = [message for message in messages if message["role"] == "system"]
    turn_messages = [message for message in messages if message["role"] != "system"]
    turn_messages = _trim_turn_messages(turn_messages)
    messages = [*system_messages, *turn_messages]
    current_content = (
        await _message_content(
            hass,
            model_registry,
            model,
            current_text,
            current_attachments,
        )
        if current_attachments
        else None
    )
    has_current_input = any(
        _message_contains_input(
            message,
            current_text,
            current_content=current_content,
        )
        for message in messages
    )
    if not has_current_input:
        messages.append(
            {
                "role": "user",
                "content": (
                    current_content if current_content is not None else current_text
                ),
            }
        )
    return messages


def _format_tool(tool: llm.Tool, custom_serializer: Any = None) -> dict[str, Any]:
    """Return an OpenAI-compatible function tool definition."""
    function: dict[str, Any] = {
        "name": tool.name,
        "parameters": convert(
            tool.parameters,
            custom_serializer=custom_serializer,
        ),
    }
    if tool.description:
        function["description"] = tool.description
    return {
        "type": "function",
        "function": function,
    }


def _chat_log_tools(chat_log: conversation.ChatLog) -> list[dict[str, Any]] | None:
    """Return OpenAI-compatible tools exposed by Home Assistant."""
    llm_api = getattr(chat_log, "llm_api", None)
    if llm_api is None:
        return None
    tools = getattr(llm_api, "tools", None)
    if not tools:
        return None
    custom_serializer = getattr(llm_api, "custom_serializer", None)
    return [_format_tool(tool, custom_serializer) for tool in tools]


def _result_tool_calls(result: Any) -> list[llm.ToolInput]:
    """Return Home Assistant tool inputs from a Groq chat completion result."""
    raw_tool_calls = getattr(result, "tool_calls", None)
    if raw_tool_calls is None:
        raw = getattr(result, "raw", None)
        message = None
        if isinstance(raw, dict):
            choices = raw.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message")
        if isinstance(message, dict):
            raw_tool_calls = message.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return []

    tool_inputs: list[llm.ToolInput] = []
    for tool_call in raw_tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        tool_name = function.get("name")
        arguments = function.get("arguments", {})
        if not isinstance(tool_name, str):
            continue
        if isinstance(arguments, str):
            try:
                tool_args = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                tool_args = {}
        elif isinstance(arguments, dict):
            tool_args = arguments
        else:
            tool_args = {}
        tool_inputs.append(
            llm.ToolInput(
                tool_name=tool_name,
                tool_args=tool_args,
                id=str(tool_call.get("id") or tool_name),
            )
        )
    return tool_inputs


def _assistant_native(result: Any) -> dict[str, Any]:
    """Return Groq metadata for Home Assistant conversation traces."""
    native: dict[str, Any] = {}
    for attr in ("model", "usage", "usage_breakdown", "executed_tools", "tool_calls"):
        value = getattr(result, attr, None)
        if value:
            native[attr] = value
    return native


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
            raise HomeAssistantError(
                f"Groq model {model} is not known to support Home Assistant tool calls"
            )
        text = ""
        use_streaming = (
            not tools
            and service_stream(self._config_entry, self._service_data)
            and hasattr(chat_log, "async_add_delta_content_stream")
        )
        for _iteration in range(MAX_TOOL_ITERATIONS):
            request = await self._async_text_generation_request(
                user_input,
                chat_log,
                request_system_prompt,
                tools,
            )
            if error := request_body_options_error_message(
                self._model_registry,
                request.model,
                request.extra_body,
            ):
                raise HomeAssistantError(error)
            if error := compound_builtin_tools_error_message(
                self._model_registry,
                request.model,
                request.compound_builtin_tools,
            ):
                raise HomeAssistantError(error)
            if error := request_context_window_error(self._model_registry, request):
                raise HomeAssistantError(error)
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
                async for content in chat_log.async_add_assistant_content(
                    assistant_content
                ):
                    if getattr(content, "content", None):
                        text = content.content
            else:
                chat_log.async_add_assistant_content_without_tools(assistant_content)
            if not getattr(chat_log, "unresponded_tool_results", False):
                break
        else:
            raise HomeAssistantError("Groq Assist exceeded the tool-call limit")

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
            ),
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
            compound_builtin_tools=service_compound_builtin_tools(
                self._config_entry,
                self._service_data,
                self._model_registry,
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
