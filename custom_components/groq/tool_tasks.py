"""Own the Home Assistant tool tasks launched for one completion."""

from __future__ import annotations

import asyncio
from contextlib import aclosing

from homeassistant.components.conversation import AssistantContent, ChatLog
from homeassistant.helpers import llm

from .api import ChatCompletionResult, TextGenerationRequest
from .async_utils import async_await_owned
from .chat import assistant_content_from_result, tool_calls_for_completion


async def async_record_chat_completion(
    chat_log: ChatLog,
    request: TextGenerationRequest,
    result: ChatCompletionResult,
    agent_id: str,
    *,
    final_tool_round: bool,
) -> None:
    """Record a non-streaming completion and dispatch only permitted rounds."""
    tool_calls = tool_calls_for_completion(
        request, result, final_tool_round=final_tool_round
    )
    content = assistant_content_from_result(result, agent_id, tool_calls)
    if tool_calls:
        await async_add_tool_batch(chat_log, content)
    else:
        chat_log.async_add_assistant_content_without_tools(content)


async def async_add_tool_batch(
    chat_log: ChatLog, content: AssistantContent, *, notify_deltas: bool = False
) -> None:
    """Record a batch through ChatLog and cancel its tasks together on exit."""
    tasks: dict[str, asyncio.Task[llm.ToolResult]] = {}

    async def record_results() -> None:
        # Cancellation can reach the shield before its owner resumes. Do not
        # give ChatLog pending tool calls if the turn has already been cancelled.
        if completion.cancelled():
            return
        async with aclosing(
            chat_log.async_add_assistant_content(content, tool_call_tasks=tasks)
        ) as results:
            async for result in results:
                if notify_deltas and (listener := chat_log.delta_listener):
                    listener(chat_log, result.as_dict())

    consumer = asyncio.create_task(record_results(), eager_start=False)
    completion = asyncio.shield(consumer)
    try:
        # HA awaits the tool tasks in order. Shield the consumer so cancellation
        # reaches every tool before waiting for any individual tool to clean up.
        await completion
    except BaseException:
        for task in tasks.values():
            if not task.done() and not task.cancelling():
                task.cancel()
        # Once a tool is running, the consumer exits when that tool does.
        # Before any tools have started, cancel the consumer directly.
        if not tasks and not consumer.done():
            consumer.cancel()
        await async_await_owned(
            asyncio.gather(consumer, *tasks.values(), return_exceptions=True)
        )
        raise
