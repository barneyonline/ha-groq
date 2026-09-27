"""Compatibility regressions for Home Assistant 2026.10 tool results."""

import json
from types import SimpleNamespace

import pytest
from homeassistant.helpers import llm

from custom_components.groq.chat import _async_chat_log_messages, _tool_result_message
from custom_components.groq.model_registry import GroqModelRegistry


class ModernToolContent(SimpleNamespace):
    """Fail if conversion touches the deprecated compatibility property."""

    @property
    def tool_result(self):
        raise AssertionError("Deprecated tool_result property accessed")


@pytest.mark.parametrize("mapping", [False, True])
@pytest.mark.parametrize(
    ("data", "error"),
    [
        ({"state": "on"}, False),
        ({}, False),
        ({"message": "Device unavailable"}, True),
        ({"error": "provider detail", "data": [1, 2]}, True),
    ],
)
@pytest.mark.asyncio
async def test_modern_tool_results_preserve_data_and_failure_status(
    mapping, data, error
):
    """Forward success and failure results without flattening or losing fields."""
    metadata = {
        "role": "tool_result",
        "tool_call_id": "call_1",
        "tool_name": "GetState",
    }
    result_type = getattr(llm, "ToolResult", SimpleNamespace)
    result = result_type(data=data, error=error)
    content = (
        {**metadata, "result": {"data": data, "error": error}, "tool_result": "stale"}
        if mapping
        else ModernToolContent(**metadata, result=result)
    )
    log = SimpleNamespace(
        content=[
            {"role": "user", "content": "Read the state"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "tool_name": "GetState", "tool_args": {}}
                ],
            },
            content,
        ]
    )
    messages = await _async_chat_log_messages(
        None, GroqModelRegistry(), "openai/gpt-oss-20b", log, "Read the state"
    )
    tool_message = next(message for message in messages if message["role"] == "tool")
    assert tool_message["tool_call_id"] == "call_1"
    assert tool_message["name"] == "GetState"
    assert json.loads(tool_message["content"]) == {"data": data, "error": error}


@pytest.mark.parametrize("mapping", [False, True])
def test_legacy_tool_results_preserve_payload(mapping):
    """Legacy payload keys named data/error must not be treated as a wrapper."""
    data = {"data": "original", "error": "original error"}
    fields = {"tool_call_id": "call_1", "tool_name": "GetState", "tool_result": data}
    content = fields if mapping else SimpleNamespace(**fields)
    assert json.loads(_tool_result_message(content)["content"]) == data


def test_serialized_result_defaults_to_success():
    """Accept the default ToolResult error flag when omitted from a mapping."""
    content = {
        "tool_call_id": "call_1",
        "tool_name": "GetState",
        "result": {"data": {}},
    }
    assert json.loads(_tool_result_message(content)["content"]) == {
        "data": {},
        "error": False,
    }
