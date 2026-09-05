"""Shared Home Assistant chat and tool adapters for Groq generation."""

from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from typing import Any, cast

from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from .attachments import async_attachment_content_parts
from .errors import translated_error
from .feature_registry import GroqFeature
from .model_registry import GroqModelRegistry


def _optional_import(module_name: str) -> Any:
    """Import an optional module without hiding its broken dependencies."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as err:
        if err.name != module_name:
            raise
        return None


_probatio = _optional_import("probatio")
_voluptuous_openapi = _optional_import("voluptuous_openapi")

MAX_HISTORY_MESSAGES = 12
MAX_TOOL_ITERATIONS = 10


def _to_openapi(
    schema: Any,
    custom_serializer: Any = None,
) -> dict[str, Any]:
    """Convert a tool schema with its matching Home Assistant serializer."""
    if _probatio is not None and (
        _voluptuous_openapi is None or isinstance(schema, _probatio.Schema)
    ):
        return cast(
            dict[str, Any],
            _probatio.to_openapi(
                schema,
                custom_serializer=custom_serializer,
            ),
        )
    if _voluptuous_openapi is not None:
        return cast(
            dict[str, Any],
            _voluptuous_openapi.convert(
                schema,
                custom_serializer=custom_serializer,
            ),
        )
    raise ImportError("No supported OpenAPI schema serializer is available")


def _content_role(content: Any) -> str | None:
    """Return a chat role for a Home Assistant chat-log content item."""
    if isinstance(content, dict):
        role = content.get("role")
        return (
            role
            if isinstance(role, str) and role in {"system", "user", "assistant"}
            else None
        )
    role = getattr(content, "role", None)
    if role in {"system", "user", "assistant"}:
        return str(role)
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


def _trim_turn_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return recent messages without orphaning tool responses."""
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    start = len(messages) - MAX_HISTORY_MESSAGES
    user_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "user"
    ]
    if user_indices:
        start = min(start, user_indices[-1])
    trimmed = messages[start:]
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
    if not model_registry.supports(model, GroqFeature.VISION):
        raise translated_error(
            "Groq Assist attachments require a vision-capable model",
            "vision_model_required",
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


async def _async_chat_log_messages(
    hass: HomeAssistant,
    model_registry: GroqModelRegistry,
    model: str,
    chat_log: conversation.ChatLog,
    current_text: str,
    current_attachments: Any = None,
    attachment_cache: dict[tuple[str, str], Any] | None = None,
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
        attachments = _content_attachments(item) if role == "user" else None
        if attachments:
            message["_attachments"] = attachments
        tool_calls = (
            item.get("tool_calls")
            if isinstance(item, dict)
            else getattr(item, "tool_calls", None)
        )
        if role == "assistant" and tool_calls:
            converted = [
                call for tool in tool_calls if (call := _tool_call_message(tool))
            ]
            if converted:
                message["tool_calls"] = converted
        if text or message.get("tool_calls") or attachments:
            messages.append(message)
    system_messages = [message for message in messages if message["role"] == "system"]
    turn_messages = _trim_turn_messages(
        [message for message in messages if message["role"] != "system"]
    )
    # HA normally already added the current input. Match the latest user turn,
    # not an identical prompt from an older conversation turn.
    latest_user = next(
        (message for message in reversed(turn_messages) if message["role"] == "user"),
        None,
    )
    if (
        latest_user is not None
        and latest_user["content"] == current_text
        and (
            not current_attachments
            or not latest_user.get("_attachments")
            or repr(latest_user["_attachments"]) == repr(current_attachments)
        )
    ):
        if current_attachments:
            latest_user["_attachments"] = current_attachments
    else:
        turn_messages.append(
            {
                "role": "user",
                "content": current_text,
                "_attachments": current_attachments,
            }
        )
    messages = [*system_messages, *turn_messages]
    for message in messages:
        attachments = message.pop("_attachments", None)
        if not attachments:
            continue
        cache_key = (message["content"], repr(attachments))
        if attachment_cache is not None and cache_key in attachment_cache:
            message["content"] = attachment_cache[cache_key]
            continue
        content = await _message_content(
            hass, model_registry, model, message["content"], attachments
        )
        message["content"] = content
        if attachment_cache is not None:
            attachment_cache[cache_key] = content
    return messages


def _format_tool(tool: llm.Tool, custom_serializer: Any = None) -> dict[str, Any]:
    """Return an OpenAI-compatible function tool definition."""
    function: dict[str, Any] = {
        "name": tool.name,
        "parameters": _to_openapi(
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
    call_ids: set[str] = set()
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
            except json.JSONDecodeError as err:
                raise translated_error(
                    "Groq returned malformed tool arguments", "invalid_request_options"
                ) from err
        elif isinstance(arguments, dict):
            tool_args = arguments
        else:
            tool_args = arguments
        call_id = tool_call.get("id")
        if (
            not isinstance(tool_args, dict)
            or not isinstance(call_id, str)
            or not call_id
            or call_id in call_ids
        ):
            raise translated_error(
                "Groq returned invalid tool arguments or tool-call ids",
                "invalid_request_options",
            )
        call_ids.add(call_id)
        tool_inputs.append(
            llm.ToolInput(
                tool_name=tool_name,
                tool_args=tool_args,
                id=call_id,
            )
        )
    return tool_inputs


def _assistant_native(result: Any) -> dict[str, Any]:
    """Return Groq metadata for Home Assistant conversation traces."""
    native: dict[str, Any] = {}
    for attr in (
        "model",
        "usage",
        "usage_breakdown",
        "executed_tools",
        "tool_calls",
        "citations",
    ):
        value = getattr(result, attr, None)
        if value:
            native[attr] = value
    return native
