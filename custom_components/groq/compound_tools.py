"""Helpers for Groq Compound built-in tool allow-lists."""

from __future__ import annotations

from typing import Any

from .const import COMPOUND_BUILTIN_TOOLS, COMPOUND_BUILTIN_TOOLS_REQUIRING_LATEST


def compound_builtin_tool_values(value: Any) -> tuple[Any, ...] | None:
    """Return configured Compound built-in tool values, or None for bad shapes."""
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple | set):
        return tuple(value)
    return None


def compound_builtin_tools_are_valid(value: Any) -> bool:
    """Return whether a value contains only supported Compound built-in tools."""
    values = compound_builtin_tool_values(value)
    if values is None:
        return False
    if not all(
        isinstance(tool, str) and tool in COMPOUND_BUILTIN_TOOLS for tool in values
    ):
        return False
    return "browser_automation" not in values or "web_search" in values


def normalize_compound_builtin_tools(value: Any) -> list[str]:
    """Return supported Compound built-in tool ids in stable order."""
    values = compound_builtin_tool_values(value)
    if values is None:
        return []
    return [
        tool
        for tool in COMPOUND_BUILTIN_TOOLS
        if any(configured_tool == tool for configured_tool in values)
    ]


def compound_builtin_tools_request_value(value: Any) -> list[Any]:
    """Return a request value that preserves invalid config for validation."""
    if compound_builtin_tools_are_valid(value):
        return normalize_compound_builtin_tools(value)
    values = compound_builtin_tool_values(value)
    if values is None:
        return [value]
    return list(values)


def compound_builtin_tools_require_latest(value: Any) -> bool:
    """Return whether enabled tools require the latest Compound system version."""
    values = compound_builtin_tool_values(value)
    if values is None:
        return False
    return any(tool in COMPOUND_BUILTIN_TOOLS_REQUIRING_LATEST for tool in values)


def compound_builtin_tools_payload_value(value: Any) -> list[str]:
    """Return supported Compound built-in tools for an API payload."""
    values = compound_builtin_tool_values(value)
    if values is None:
        raise ValueError("compound_builtin_tools must be a string or list of strings")
    if not all(
        isinstance(tool, str) and tool in COMPOUND_BUILTIN_TOOLS for tool in values
    ):
        raise ValueError("compound_builtin_tools contains unsupported tool ids")
    if "browser_automation" in values and "web_search" not in values:
        raise ValueError("browser_automation requires web_search")
    return normalize_compound_builtin_tools(values)
