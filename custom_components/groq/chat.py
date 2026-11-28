"""Shared Home Assistant chat and tool adapters for Groq generation."""

from __future__ import annotations

import json
from collections.abc import Sized
from contextvars import copy_context
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, cast

import probatio
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from .api import ChatCompletionResult, TextGenerationRequest, large_tool_arguments
from .attachments import (
    ImageAttachmentBudget,
    ImageAttachmentBudgetExceeded,
    ImageAttachmentGroup,
    ImageAttachmentSources,
    async_load_image_attachments,
)
from .errors import GroqResponseError, translated_error
from .feature_registry import GroqFeature
from .json_utils import JSON_EXECUTOR_THRESHOLD, estimated_json_size_exceeds
from .model_registry import GroqModelRegistry
from .text_generation import _serialized_json_string_token_estimate

MAX_HISTORY_MESSAGES = 12
MAX_TOOL_ITERATIONS = 10


def prepare_tool_round(
    request: TextGenerationRequest, iteration: int
) -> tuple[TextGenerationRequest, bool]:
    """Return the request and whether it must conclude the local tool loop."""
    final_tool_round = iteration == MAX_TOOL_ITERATIONS - 1
    if not final_tool_round or not request.tools:
        return request, final_tool_round
    if request.browser_search:
        # Provider-side search completes within this response. Keep it
        # available while removing the Home Assistant function definitions.
        return replace(request, tools=None), True
    return replace(request, tool_choice="none"), True


async def async_tool_calls_for_completion(
    hass: HomeAssistant,
    request: TextGenerationRequest,
    result: ChatCompletionResult,
    *,
    final_tool_round: bool,
) -> list[llm.ToolInput]:
    """Validate the tool round and allow only functions exposed in the request."""
    if final_tool_round and (result.tool_calls or not result.text.strip()):
        raise translated_error("Groq exceeded the tool-call limit", "tool_call_limit")
    tool_calls = (
        await hass.async_add_executor_job(_result_tool_calls, result)
        if large_tool_arguments(result.tool_calls)
        else _result_tool_calls(result)
    )
    if tool_calls:
        allowed_tools = {
            tool["function"]["name"]
            for tool in request.tools or []
            if tool.get("type") == "function"
        }
        if any(call.tool_name not in allowed_tools for call in tool_calls):
            raise GroqResponseError("Groq returned an unexposed Home Assistant tool")
    return tool_calls


@dataclass(slots=True)
class ChatHistoryCache:
    """Reuse completed log content while a single tool loop appends messages."""

    attachments: dict[ImageAttachmentSources, ImageAttachmentGroup] = field(
        default_factory=dict
    )
    tool_payloads: dict[tuple[int, bool], tuple[Any, str]] = field(default_factory=dict)
    string_estimates: dict[int, tuple[str, int]] = field(default_factory=dict)
    system_messages: list[dict[str, Any]] = field(default_factory=list)
    scanned_count: int = 0

    def system_instructions(self, content: list[Any]) -> list[dict[str, Any]]:
        """Read HA's replaceable first item and scan newly appended instructions."""
        if len(content) < self.scanned_count:
            self.system_messages.clear()
            self.scanned_count = 0
        for index in range(max(1, self.scanned_count), len(content)):
            item = content[index]
            if _content_role(item) == "system" and (prompt := _content_text(item)):
                self.system_messages.append({"role": "system", "content": prompt})
        self.scanned_count = len(content)
        messages: list[dict[str, Any]] = []
        if content and _content_role(content[0]) == "system":
            if prompt := _content_text(content[0]):
                messages.append({"role": "system", "content": prompt})
        messages.extend(message.copy() for message in self.system_messages)
        return messages

    async def async_serialize_tool(
        self,
        hass: HomeAssistant,
        value: dict[str, Any] | llm.ToolResult,
        *,
        result: bool,
    ) -> str:
        """Keep a reference to each value so an object ID cannot be reused."""
        identity = (id(value), result)
        if identity not in self.tool_payloads:
            payload = (
                _tool_result_payload(value) if result else cast(dict[str, Any], value)
            )
            if estimated_json_size_exceeds(
                payload, JSON_EXECUTOR_THRESHOLD, use_encoded_upper_bound=True
            ):
                serialized, estimate = await hass.async_add_executor_job(
                    copy_context().run,
                    _encode_and_estimate_tool_payload,
                    payload,
                    result,
                )
                self.string_estimates[id(serialized)] = (serialized, estimate)
            else:
                serialized = _encode_tool_payload(payload, result)
            self.tool_payloads[identity] = (value, serialized)
        return self.tool_payloads[identity][1]


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


def _same_attachments(previous: Any, current: Any, max_count: int) -> bool:
    """Compare attachment groups whose counts can fit the request."""
    return previous is current or (
        isinstance(previous, Sized)
        and isinstance(current, Sized)
        and len(previous) == len(current) <= max_count
        and previous == current
    )


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


async def _async_add_history_attachments(
    hass: HomeAssistant,
    model_registry: GroqModelRegistry,
    model: str,
    messages: list[dict[str, Any]],
    current: dict[str, Any],
    cache: ChatHistoryCache,
    *,
    budget: ImageAttachmentBudget,
) -> None:
    """Give current and recent images priority while leaving messages in order."""
    retained: dict[ImageAttachmentSources, ImageAttachmentGroup] = {}
    for message in reversed(messages):
        attachments = message.pop("_attachments", None)
        if not attachments:
            continue
        try:
            if not model_registry.supports(model, GroqFeature.VISION):
                raise translated_error(
                    "Groq Assist attachments require a vision-capable model",
                    "vision_model_required",
                )
            sources = budget.admit(attachments)
            group = retained.get(sources) or cache.attachments.get(sources)
            if group is None:
                group = await async_load_image_attachments(
                    hass, sources, budget=budget, historical=message is not current
                )
                if group is None:
                    continue
            budget.charge(group)
        except ImageAttachmentBudgetExceeded:
            if message is current:
                raise
            continue
        message["content"] = group.with_text(message["content"])
        retained[sources] = group
    cache.attachments = retained


def _tool_call_id(tool_call: Any) -> str:
    """Return a stable OpenAI-compatible tool-call id."""
    if isinstance(tool_call, dict):
        tool_id = tool_call.get("id")
        return str(tool_id) if tool_id else "tool_call"
    tool_id = getattr(tool_call, "id", None)
    return str(tool_id) if tool_id else "tool_call"


def _tool_call_message(tool_call: Any) -> dict[str, Any] | None:
    """Build a tool call descriptor before serializing retained arguments."""
    parts = _tool_call_parts(tool_call)
    if parts is None:
        return None
    tool_name, tool_args = parts
    return {
        "id": _tool_call_id(tool_call),
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": tool_args,
        },
    }


def _tool_result_message(content: Any) -> dict[str, Any] | None:
    """Build a tool result descriptor before serializing retained content."""
    identity = _tool_result_identity(content)
    if identity is None:
        return None
    tool_call_id, tool_name = identity
    result = content["result"] if isinstance(content, dict) else content.result
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": result,
    }


def _tool_result_payload(result: Any) -> dict[str, Any]:
    """Build the Home Assistant tool result envelope for Groq."""
    return (
        {"data": result["data"], "error": result.get("error", False)}
        if isinstance(result, dict)
        else {"data": result.data, "error": result.error}
    )


def _encode_tool_payload(payload: dict[str, Any], result: bool) -> str:
    """Serialize an OpenAI tool argument object or a tool result envelope."""
    return json.dumps(payload, separators=(",", ":"), default=str if result else None)


def _encode_and_estimate_tool_payload(
    payload: dict[str, Any], result: bool
) -> tuple[str, int]:
    """Return the wire string and its token estimate from one executor job."""
    serialized = _encode_tool_payload(payload, result)
    return serialized, _serialized_json_string_token_estimate(serialized)


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
    cache: ChatHistoryCache,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build descriptors for system instructions and the retained history tail."""
    system_messages = cache.system_instructions(chat_log.content)
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
            continue
        if role is None:
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
            break
        selected.append((item, role))

    turns: list[dict[str, Any]] = []
    for item, role in reversed(selected):
        if role == "tool_result":
            if result := _tool_result_message(item):
                turns.append(result)
            continue
        message: dict[str, Any] = {"role": role, "content": _content_text(item) or ""}
        if role == "user" and (attachments := _content_attachments(item)):
            message["_attachments"] = attachments
        if role == "assistant":
            calls = [
                call
                for tool in _assistant_tool_calls(item) or ()
                if (call := _tool_call_message(tool))
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
    history_cache: ChatHistoryCache | None = None,
) -> list[dict[str, Any]]:
    """Return OpenAI-compatible messages, including supported image attachments."""
    cache = history_cache if history_cache is not None else ChatHistoryCache()
    budget = ImageAttachmentBudget()
    system_messages, turn_messages = _chat_log_history(chat_log, cache)
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
            or _same_attachments(
                latest_user["_attachments"], current_attachments, budget.remaining_count
            )
        )
    ):
        if current_attachments:
            latest_user["_attachments"] = current_attachments
    else:
        latest_user = {
            "role": "user",
            "content": current_text,
            "_attachments": current_attachments,
        }
        turn_messages.append(latest_user)
    messages = [*system_messages, *turn_messages]
    for message in messages:
        # Only retained history incurs serialization or attachment I/O. These
        # descriptors own their dictionaries, leaving HA's chat log untouched.
        if message["role"] == "tool":
            message["content"] = await cache.async_serialize_tool(
                hass, message["content"], result=True
            )
        for tool_call in message.get("tool_calls", []):
            function = tool_call["function"]
            function["arguments"] = await cache.async_serialize_tool(
                hass, function["arguments"], result=False
            )
    await _async_add_history_attachments(
        hass, model_registry, model, turn_messages, latest_user, cache, budget=budget
    )
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


def assistant_content_from_result(
    result: ChatCompletionResult, agent_id: str, tool_calls: list[llm.ToolInput]
) -> conversation.AssistantContent:
    """Record the completion and its trace metadata in Home Assistant format."""
    return conversation.AssistantContent(
        agent_id=agent_id,
        content=result.text or None,
        thinking_content=result.reasoning,
        tool_calls=tool_calls or None,
        native=_assistant_native(result) or None,
    )
