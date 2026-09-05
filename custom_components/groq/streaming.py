"""Assemble Groq chat deltas before permitting any local tool execution."""

from __future__ import annotations

import json
from typing import Any

from .errors import GroqResponseError


class ChatStream:
    """Collect one bounded response; tool calls are complete only at stream end."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.calls: dict[int, dict[str, Any]] = {}
        self.usage: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.finish_reason: str | None = None

    def add(self, event: dict[str, Any]) -> str | None:
        """Merge one SSE event and return only its incremental spoken text."""
        if isinstance(event.get("model"), str):
            self.model = event["model"]
        groq = event.get("x_groq")
        usage = event.get("usage")
        if not isinstance(usage, dict) and isinstance(groq, dict):
            usage = groq.get("usage")
        if isinstance(usage, dict):
            self.usage.update(usage)
        breakdown = event.get("usage_breakdown")
        if isinstance(groq, dict) and isinstance(groq.get("usage_breakdown"), dict):
            breakdown = groq["usage_breakdown"]
        if isinstance(breakdown, dict):
            self.metadata["usage_breakdown"] = breakdown
        choices = event.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            return None
        choice = choices[0]
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            delta = {}
        if self.finish_reason and any(
            delta.get(k) for k in ("content", "reasoning", "tool_calls")
        ):
            raise GroqResponseError("Groq sent content after completing a response")
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]
        for key in ("annotations", "citations", "executed_tools"):
            if isinstance(delta.get(key), list):
                self.metadata.setdefault(key, []).extend(delta[key])
        if isinstance(delta.get("reasoning"), str):
            self.reasoning.append(delta["reasoning"])
        for call in delta.get("tool_calls") or []:
            self._add_call(call)
        content = delta.get("content")
        if isinstance(content, str) and content:
            self.content.append(content)
            return content
        return None

    def _add_call(self, delta: Any) -> None:
        """Accumulate interleaved calls by provider index, preserving arguments."""
        if not isinstance(delta, dict):
            raise GroqResponseError("Groq returned an invalid tool delta")
        index = delta.get("index")
        if type(index) is not int or not 0 <= index < 64:
            raise GroqResponseError("Groq returned an invalid tool index")
        call = self.calls.setdefault(
            index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if delta.get("type") not in (None, "function"):
            raise GroqResponseError("Groq returned an unsupported local tool type")
        if delta.get("id") is not None:
            if not isinstance(delta["id"], str) or (
                call["id"] and call["id"] != delta["id"]
            ):
                raise GroqResponseError("Groq changed a streamed tool id")
            call["id"] = delta["id"]
        function = delta.get("function", {})
        if function is None:
            return
        if not isinstance(function, dict):
            raise GroqResponseError("Groq returned an invalid tool function")
        for key in ("name", "arguments"):
            if function.get(key) is not None:
                if not isinstance(function[key], str):
                    raise GroqResponseError("Groq returned an invalid tool fragment")
                call["function"][key] += function[key]

    def result(self) -> dict[str, Any]:
        """Reject unfinished/malformed operations before producing tool calls."""
        if self.calls:
            if self.finish_reason != "tool_calls":
                raise GroqResponseError("Groq did not finish the streamed tool calls")
            ids: set[str] = set()
            for call in self.calls.values():
                function = call["function"]
                try:
                    arguments = json.loads(function["arguments"])
                except json.JSONDecodeError as err:
                    raise GroqResponseError(
                        "Groq returned incomplete tool arguments"
                    ) from err
                if (
                    not isinstance(arguments, dict)
                    or not function["name"]
                    or not call["id"]
                    or call["id"] in ids
                ):
                    raise GroqResponseError("Groq returned invalid streamed tool calls")
                ids.add(call["id"])
        message = {
            **self.metadata,
            "content": "".join(self.content),
            "reasoning": "".join(self.reasoning),
            "tool_calls": [self.calls[i] for i in sorted(self.calls)],
        }
        return {
            "model": self.model,
            "usage": self.usage,
            "usage_breakdown": self.metadata.get("usage_breakdown"),
            "choices": [{"message": message}],
        }
