"""Shared Home Assistant chat and tool adapters for Groq generation."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import probatio
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from .api import ChatCompletionResult
from .attachments import async_attachment_content_parts
from .errors import translated_error
from .feature_registry import GroqFeature
from .model_registry import GroqModelRegistry

MAX_HISTORY_MESSAGES = 12
MAX_TOOL_ITERATIONS = 10


def _content_role(content: Any) -> str | None:
    """Return a chat role for a Home Assistant chat-log content item."""
    role = (
        content.get("role")
        if isinstance(content, dict)
        else getattr(content, "role", None)
    )
    if isinstance(role, str) and role in ("system", "user", "assistant", "tool_result"):
        return str(role)
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


def _drop_orphan_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove tool results whose originating calls fell outside the retained history."""
    valid_tool_call_ids = {
        tool_call["id"]
        for message in messages
        for tool_call in message.get("tool_calls", [])
        if isinstance(tool_call, dict) and isinstance(tool_call.get("id"), str)
    }
    return [
        message
        for message in messages
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


def _tool_call_message(
    tool_call: Any, *, serialize: bool = True
) -> dict[str, Any] | None:
    """Return an OpenAI-compatible tool call from a Home Assistant ToolInput."""
    parts = _tool_call_parts(tool_call)
    if parts is None:
        return None
    tool_name, tool_args = parts
    return {
        "id": _tool_call_id(tool_call),
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": (
                json.dumps(tool_args, separators=(",", ":")) if serialize else tool_args
            ),
        },
    }


def _tool_result_message(
    content: Any, *, serialize: bool = True
) -> dict[str, Any] | None:
    """Return an OpenAI-compatible tool result message from chat-log content."""
    identity = _tool_result_identity(content)
    if identity is None:
        return None
    tool_call_id, tool_name = identity
    result = content["result"] if isinstance(content, dict) else content.result
    tool_result = (
        {"data": result["data"], "error": result.get("error", False)}
        if isinstance(result, dict)
        else {"data": result.data, "error": result.error}
    )
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": (
            json.dumps(tool_result, separators=(",", ":"), default=str)
            if serialize
            else tool_result
        ),
    }


def _tool_result_identity(content: Any) -> tuple[str, str] | None:
    """Check that a tool result can be included without reading its data."""
    if isinstance(content, dict):
        call_id = content.get("tool_call_id")
        name = content.get("tool_name")
    else:
        call_id = getattr(content, "tool_call_id", None)
        name = getattr(content, "tool_name", None)
    if isinstance(call_id, str) and isinstance(name, str):
        return call_id, name
    return None


def _assistant_tool_calls(content: Any) -> Any:
    """Return the raw tool calls on an assistant item."""
    return (
        content.get("tool_calls")
        if isinstance(content, dict)
        else getattr(content, "tool_calls", None)
    )


def _tool_call_parts(tool_call: Any) -> tuple[str, dict[str, Any]] | None:
    """Return the name and arguments of a valid Home Assistant tool call."""
    if isinstance(tool_call, dict):
        tool_name = tool_call.get("tool_name")
        tool_args = tool_call.get("tool_args")
    else:
        tool_name = getattr(tool_call, "tool_name", None)
        tool_args = getattr(tool_call, "tool_args", None)
    if isinstance(tool_name, str) and isinstance(tool_args, dict):
        return tool_name, tool_args
    return None


def _chat_log_history(
    chat_log: conversation.ChatLog,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build descriptors for system instructions and the retained history tail."""
    system_messages: list[dict[str, Any]] = []
    selected: list[tuple[Any, str]] = []
    newest_user_offset = next(
        (
            offset
            for offset, item in enumerate(reversed(chat_log.content))
            if _content_role(item) == "user"
            and (_content_text(item) or _content_attachments(item))
        ),
        -1,
    )
    discarded = False
    for offset, item in enumerate(reversed(chat_log.content)):
        role = _content_role(item)
        if role == "system":
            if text := _content_text(item):
                system_messages.append({"role": role, "content": text})
            continue
        if role is None or discarded:
            continue
        if role == "tool_result":
            if _tool_result_identity(item) is None:
                continue
        elif not (
            _content_text(item)
            or (role == "user" and _content_attachments(item))
            or (
                role == "assistant"
                and any(
                    _tool_call_parts(call) for call in _assistant_tool_calls(item) or ()
                )
            )
        ):
            continue
        if offset > newest_user_offset and len(selected) >= MAX_HISTORY_MESSAGES:
            discarded = True
            continue
        selected.append((item, role))

    system_messages.reverse()
    turns: list[dict[str, Any]] = []
    for item, role in reversed(selected):
        if role == "tool_result":
            if result := _tool_result_message(item, serialize=False):
                turns.append(result)
            continue
        message: dict[str, Any] = {"role": role, "content": _content_text(item) or ""}
        if role == "user" and (attachments := _content_attachments(item)):
            message["_attachments"] = attachments
        if role == "assistant":
            calls = [
                call
                for tool in _assistant_tool_calls(item) or ()
                if (call := _tool_call_message(tool, serialize=False))
            ]
            if calls:
                message["tool_calls"] = calls
        turns.append(message)
    if discarded or len(turns) > MAX_HISTORY_MESSAGES:
        turns = _drop_orphan_tool_results(turns)
    return system_messages, turns


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
    system_messages, turn_messages = _chat_log_history(chat_log)
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
        # Only retained history incurs serialization or attachment I/O. These
        # descriptors own their dictionaries, leaving HA's chat log untouched.
        if message["role"] == "tool":
            message["content"] = json.dumps(
                message["content"], separators=(",", ":"), default=str
            )
        for tool_call in message.get("tool_calls", []):
            function = tool_call["function"]
            function["arguments"] = json.dumps(
                function["arguments"], separators=(",", ":")
            )
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
        "parameters": probatio.to_openapi(
            tool.parameters, custom_serializer=custom_serializer
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
    llm_api = chat_log.llm_api
    if llm_api is None:
        return None
    tools = llm_api.tools
    if not tools:
        return None
    custom_serializer = llm_api.custom_serializer
    return [_format_tool(tool, custom_serializer) for tool in tools]


def _result_tool_calls(result: ChatCompletionResult) -> list[llm.ToolInput]:
    """Return Home Assistant tool inputs from a Groq chat completion result."""
    raw_tool_calls = result.tool_calls or []

    tool_inputs: list[llm.ToolInput] = []
    call_ids: set[str] = set()
    for tool_call in raw_tool_calls:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        tool_name = function.get("name")
        if not isinstance(tool_name, str):
            continue
        wire_arguments = function.get("arguments", {})
        if isinstance(wire_arguments, str):
            # Decode an independent copy for tools that mutate their arguments.
            try:
                tool_args = json.loads(wire_arguments) if wire_arguments else {}
            except json.JSONDecodeError as err:
                raise translated_error(
                    "Groq returned malformed tool arguments",
                    "invalid_request_options",
                ) from err
        else:
            tool_args = wire_arguments
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
        if not isinstance(wire_arguments, str):
            try:
                tool_args = deepcopy(tool_args)
            except RecursionError:
                # Python's JSON decoder accepts deeper arguments than deepcopy.
                tool_args = json.loads(json.dumps(tool_args))
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
