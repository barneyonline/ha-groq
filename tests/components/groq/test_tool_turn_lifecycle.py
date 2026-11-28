"""Regressions for Home Assistant tool task ownership and turn budgets."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import json
from types import SimpleNamespace

import pytest
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.components.ai_task import GenDataTask
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from custom_components.groq.ai_task import GroqAITaskEntity
from custom_components.groq.api import GroqApiClient, TextGenerationRequest
from custom_components.groq.chat import MAX_TOOL_ITERATIONS, prepare_tool_round
from custom_components.groq.conversation import GroqConversationEntity
from custom_components.groq.errors import GroqApiError, GroqResponseError
from custom_components.groq.tool_tasks import async_add_tool_batch

from .test_foundation import DummyEntry
from .test_transport_architecture import Response

PLATFORMS = ("assist", "assist_stream", "ai_task")


class ToolSession:
    """Supply batches of synthetic local tool calls, then optional final text."""

    def __init__(
        self, batch_width, *, final_round=None, final_text="Done.", arguments="{}"
    ):
        self.batch_width = batch_width
        self.final_round = final_round
        self.final_text = final_text
        self.arguments = arguments
        self.requests = []
        self.responses = []

    def request(self, _method, _url, **kwargs):
        request = kwargs["json"]
        self.requests.append(request)
        round_number = len(self.requests)
        batch_width = 0 if round_number == self.final_round else self.batch_width
        calls = [
            {
                "id": f"{round_number}-{index}",
                "type": "function",
                "function": {"name": "Fixture", "arguments": self.arguments},
            }
            for index in range(batch_width)
        ]
        message = {"content": "Checking." if calls else self.final_text}
        if request.get("stream"):
            event = {"choices": [{"delta": message}]}
            final_delta = {
                "reasoning": "fixture trace",
                "tool_calls": [dict(call, index=i) for i, call in enumerate(calls)],
            }
            final_event = {
                "choices": [
                    {
                        "delta": final_delta,
                        "finish_reason": "tool_calls" if calls else "stop",
                    }
                ]
            }
            body = (
                f"data: {json.dumps(event)}\n\n"
                f"data: {json.dumps(final_event)}\n\ndata: [DONE]\n\n"
            ).encode()
        else:
            message["tool_calls"] = calls
            body = json.dumps({"choices": [{"message": message}]}).encode()
        response = Response(body=body)
        self.responses.append(response)
        return response


def fixture_tool_api(
    hass: HomeAssistant,
    fixture_id: str,
    run_tool: Callable[[llm.ToolInput], Awaitable[llm.ToolResult]],
    *,
    expose_tool: bool = True,
) -> llm.API:
    """Expose one deterministic tool through the Home Assistant LLM API."""

    class FixtureTool(llm.Tool):
        name = "Fixture"
        description = "A deterministic test tool"
        parameters = vol.Schema({vol.Optional("unused"): str})

        async def async_call(self, hass, tool_input, llm_context):
            return await run_tool(tool_input)

    class FixtureAPI(llm.API):
        async def async_get_api_instance(self, llm_context):
            return llm.APIInstance(
                api=self,
                api_prompt="Use the fixture tool to respond.",
                llm_context=llm_context,
                tools=[FixtureTool()] if expose_tool else [],
            )

    return FixtureAPI(hass=hass, id=f"{fixture_id}_api", name="Fixture")


def tool_turn(
    hass: HomeAssistant,
    platform: str,
    session: ToolSession,
    run_tool: Callable[[llm.ToolInput], Awaitable[llm.ToolResult]],
    *,
    fixture_id: str = "fixture",
    browser_search: bool = False,
    expose_tool: bool = True,
    client: GroqApiClient | None = None,
    prompt: str = "Fixture",
) -> tuple[Callable[[], Awaitable[object]], conversation.ChatLog]:
    """Construct a generation turn that uses the real Home Assistant ChatLog."""
    if client is None:
        client = GroqApiClient(hass, api_key="fake", session=session)
    api = fixture_tool_api(hass, fixture_id, run_tool, expose_tool=expose_tool)
    model = "openai/gpt-oss-20b"
    config = {
        "model": model,
        "stream": platform == "assist_stream",
        "browser_search": browser_search,
    }
    log = conversation.ChatLog(hass, fixture_id)
    log.async_add_user_content(conversation.UserContent(content=prompt))
    user_input = conversation.ConversationInput(
        text=prompt,
        context=Context(),
        conversation_id=log.conversation_id,
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id="conversation.groq",
    )

    async def turn():
        if platform == "ai_task":
            log.llm_api = await api.async_get_api_instance(
                user_input.as_llm_context("groq")
            )
            entity = GroqAITaskEntity(hass, DummyEntry(), config, client)
            return await entity._async_generate_text_with_tools(
                GenDataTask(name="fixture", instructions=prompt, llm_api=api),
                log,
                prompt,
                [{"type": "function", "function": {"name": "Fixture"}}],
            )

        unregister = llm.async_register_api(hass, api)
        try:
            entity = GroqConversationEntity(
                hass, DummyEntry(), {**config, "llm_hass_api": api.id}, client
            )
            return await entity._async_handle_message(user_input, log)
        finally:
            unregister()

    return turn, log


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", PLATFORMS)
async def test_cancelled_turn_stops_every_tool_before_return(hass, platform):
    started = [asyncio.Event(), asyncio.Event()]
    stopping = [asyncio.Event(), asyncio.Event()]
    finished = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def run_tool(tool_input):
        index = int(tool_input.id.split("-")[1])
        started[index].set()
        try:
            await release.wait()
            return llm.ToolResult(data={"index": index})
        finally:
            stopping[index].set()
            await release_cleanup.wait()
            finished[index].set()

    session = ToolSession(2)
    turn, _ = tool_turn(hass, platform, session, run_tool)
    task = asyncio.create_task(turn())
    try:
        async with asyncio.timeout(3):
            await asyncio.gather(*(event.wait() for event in started))
            task.cancel()
            await asyncio.gather(*(event.wait() for event in stopping))
            task.cancel()
            assert not task.done()
            release_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert all(event.is_set() for event in finished)
        assert session.responses[0].closed
    finally:
        release.set()
        release_cleanup.set()
        await asyncio.gather(task, return_exceptions=True)
        await hass.async_block_till_done()


@pytest.mark.asyncio
async def test_stream_cancelled_before_dispatch_does_not_start_tools(hass):
    invoked = []

    async def run_tool(tool_input):
        invoked.append(tool_input.id)
        return llm.ToolResult(data={"check": True})

    session = ToolSession(2)
    turn, log = tool_turn(hass, "assist_stream", session, run_tool)

    def on_delta(_log, delta):
        if delta.get("tool_calls"):
            task = asyncio.current_task()
            assert task is not None
            asyncio.get_running_loop().call_soon(task.cancel)

    log.delta_listener = on_delta
    task = asyncio.create_task(turn())
    async with asyncio.timeout(3):
        with pytest.raises(asyncio.CancelledError):
            await task

    assert not invoked
    assert all(item.role not in {"assistant", "tool_result"} for item in log.content)
    assert len(session.responses) == 1
    assert session.responses[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", PLATFORMS)
async def test_cancelling_a_tool_turn_preserves_an_independent_turn(hass, platform):
    first_started, second_started = asyncio.Event(), asyncio.Event()
    release = asyncio.Event()

    async def first_tool(_input):
        first_started.set()
        await release.wait()
        return llm.ToolResult(data={"check": True})

    async def second_tool(_input):
        second_started.set()
        await release.wait()
        return llm.ToolResult(data={"check": True})

    first_turn, _ = tool_turn(hass, platform, ToolSession(1), first_tool)
    second_turn, second_log = tool_turn(
        hass,
        platform,
        ToolSession(1, final_round=2),
        second_tool,
        fixture_id="independent",
    )
    first = asyncio.create_task(first_turn())
    second = asyncio.create_task(second_turn())
    try:
        async with asyncio.timeout(3):
            await asyncio.gather(first_started.wait(), second_started.wait())
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert not second.done()
            release.set()
            await second
        assert second_log.content[-2].result.data == {"check": True}
        assert second_log.content[-1].content == "Done."
    finally:
        for task in (first, second):
            if not task.done():
                task.cancel()
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.asyncio
async def test_nonstream_assist_rejects_calls_without_exposed_tools(hass):
    async def run_tool(_tool_input):
        pytest.fail("an unexposed tool was dispatched")

    session = ToolSession(1)
    turn, log = tool_turn(hass, "assist", session, run_tool, expose_tool=False)

    with pytest.raises(HomeAssistantError, match="unexposed") as error:
        await turn()

    assert error.value.translation_key == "invalid_api_response"
    assert len(session.requests) == 1
    assert "tools" not in session.requests[0]
    assert all(item.role != "tool_result" for item in log.content)
    assert session.responses[0].closed


def test_last_round_without_local_tools_still_requires_an_answer():
    request = TextGenerationRequest(
        prompt="Answer", model="openai/gpt-oss-20b", browser_search=True
    )
    prepared, final_tool_round = prepare_tool_round(request, MAX_TOOL_ITERATIONS - 1)
    assert prepared is request
    assert final_tool_round


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("final_text", (None, "", "Done."))
@pytest.mark.parametrize("browser_search", [False, True])
async def test_last_tool_round_is_reserved_for_text(
    hass, platform, final_text, browser_search
):
    invoked = []

    async def run_tool(tool_input):
        invoked.append(tool_input.id)
        return llm.ToolResult(data={"check": True})

    session = ToolSession(
        1,
        final_round=MAX_TOOL_ITERATIONS if final_text is not None else None,
        final_text=final_text,
    )
    turn, log = tool_turn(
        hass, platform, session, run_tool, browser_search=browser_search
    )
    if final_text:
        await turn()
        assert log.content[-1].content == "Done."
    else:
        with pytest.raises(HomeAssistantError, match="tool-call limit"):
            await turn()
    assert len(session.requests) == MAX_TOOL_ITERATIONS
    assert len(invoked) == MAX_TOOL_ITERATIONS - 1
    if browser_search:
        assert {"type": "browser_search"} in session.requests[0]["tools"]
        assert session.requests[-1]["tools"] == [{"type": "browser_search"}]
        assert session.requests[-1]["tool_choice"] == "auto"
    else:
        assert session.requests[-1]["tool_choice"] == "none"
    assert session.requests[-1]["messages"][-1]["role"] == "tool"
    assert all(response.closed for response in session.responses)


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", PLATFORMS)
async def test_successful_tool_turn_preserves_transcript(hass, platform):
    async def run_tool(tool_input):
        return llm.ToolResult(data={"call": tool_input.id})

    session = ToolSession(2, final_round=2)
    turn, log = tool_turn(hass, platform, session, run_tool)
    deltas = []

    def on_delta(current_log, delta):
        assert current_log is log
        deltas.append(delta)

    log.delta_listener = on_delta
    result = await turn()
    if platform == "ai_task":
        assert result.text == "Done."
    else:
        assert result.response.speech["plain"]["speech"] == "Done."
    first, first_result, second_result, last = log.content[-4:]
    assert first.content == "Checking."
    assert first.native["tool_calls"][0]["id"] == "1-0"
    assert [first_result.tool_call_id, second_result.tool_call_id] == ["1-0", "1-1"]
    assert last.content == "Done."
    if platform != "assist_stream":
        assert not deltas
        return
    assert first.thinking_content == "fixture trace"
    assert deltas[:2] == [{"role": "assistant"}, {"content": "Checking."}]
    assert deltas[2]["thinking_content"] == "fixture trace"
    assert [call.id for call in deltas[2]["tool_calls"]] == ["1-0", "1-1"]
    assert "native" not in deltas[2]
    assert deltas[3:5] == [first_result.as_dict(), second_result.as_dict()]
    assert deltas[5:7] == [{"role": "assistant"}, {"content": "Done."}]


@pytest.mark.asyncio
async def test_cancelled_consumer_stops_before_deferred_tool_creation(hass):
    preparing, closed = asyncio.Event(), asyncio.Event()

    class DeferredChatLog(conversation.ChatLog):
        async def async_add_assistant_content(self, *_args, **_kwargs):
            try:
                preparing.set()
                await asyncio.Future()
                yield
            finally:
                closed.set()

    log = DeferredChatLog(hass, "deferred")
    task = asyncio.create_task(
        async_add_tool_batch(
            log, conversation.AssistantContent(agent_id="groq"), lambda: None
        )
    )
    async with asyncio.timeout(3):
        await preparing.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert closed.is_set()


@pytest.mark.asyncio
async def test_assist_rejects_stream_adapter_without_final_result(hass):
    async def incomplete_stream(_request):
        yield "partial"

    log = conversation.ChatLog(hass, "incomplete")
    entity = SimpleNamespace(
        _client=SimpleNamespace(async_stream_chat=incomplete_stream)
    )
    with pytest.raises(GroqResponseError, match="no final result"):
        await GroqConversationEntity._async_stream_message(
            entity,
            SimpleNamespace(agent_id="groq"),
            log,
            TextGenerationRequest(prompt="fixture", model="model"),
        )
    assert len(log.content) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", PLATFORMS)
async def test_wide_tool_batch_is_rejected_before_dispatch(hass, platform):
    invoked = []

    async def run_tool(tool_input):
        invoked.append(tool_input.id)
        return llm.ToolResult(data={"ok": True})

    session = ToolSession(65, final_round=2)
    turn, log = tool_turn(hass, platform, session, run_tool)
    with pytest.raises(GroqResponseError):
        await turn()
    assert not invoked
    assert not any(item.role == "tool_result" for item in log.content)
    assert len(session.responses) == 1 and session.responses[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("cancel", [False, True])
async def test_tool_batch_limits_running_calls_and_preserves_results(
    hass, platform, cancel
):
    release = asyncio.Event()
    release_first = asyncio.Event()
    started = asyncio.Event()
    invoked = []
    stopped = []

    async def run_tool(tool_input):
        invoked.append(tool_input.id)
        if len(invoked) == 8:
            started.set()
        try:
            if cancel and tool_input.id == "1-0":
                await release_first.wait()
                # Cancel after the first slot is freed and before its waiter resumes.
                asyncio.get_running_loop().call_soon(task.cancel)
            else:
                await release.wait()
            return llm.ToolResult(data={"ok": True})
        finally:
            stopped.append(tool_input.id)

    session = ToolSession(12, final_round=2)
    independent_session = ToolSession(1, final_round=2)

    class TurnSession:
        def request(self, method, url, **kwargs):
            message = next(
                message
                for message in kwargs["json"]["messages"]
                if message["role"] == "user"
            )
            target = (
                independent_session if message["content"] == "Independent" else session
            )
            return target.request(method, url, **kwargs)

    async def run_independent_tool(tool_input):
        return llm.ToolResult(data={"independent": tool_input.id})

    client = GroqApiClient(hass, api_key="fake", session=TurnSession())
    turn, log = tool_turn(hass, platform, session, run_tool, client=client)
    independent, independent_log = tool_turn(
        hass,
        platform,
        independent_session,
        run_independent_tool,
        client=client,
        fixture_id="independent",
        prompt="Independent",
    )
    task = asyncio.create_task(turn())
    try:
        async with asyncio.timeout(3):
            await started.wait()
            await asyncio.sleep(0)
            assert len(invoked) == 8
            if cancel:
                release_first.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert set(stopped) == set(invoked)
                assert len(invoked) == 8
                assert len(session.requests) == 1
            else:
                await independent()
                assert not task.done() and len(invoked) == 8 and not stopped
                assert independent_log.content[-2].result.data == {"independent": "1-0"}
                assert len(independent_session.requests) == 2
                release.set()
                await task
                expected = [f"1-{index}" for index in range(12)]
                assert invoked == expected
                assert [
                    item.tool_call_id
                    for item in log.content
                    if item.role == "tool_result"
                ] == expected
                assert [
                    message["tool_call_id"]
                    for message in session.requests[-1]["messages"]
                    if message["role"] == "tool"
                ] == expected
    finally:
        if not task.done():
            task.cancel()
        release_first.set()
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("unload_at", ["response", "queued"])
async def test_unload_prevents_new_tool_execution(hass, platform, unload_at):
    started, release = asyncio.Event(), asyncio.Event()
    invoked = []

    class HeldResponse(Response):
        async def iter_chunked(self, size):
            if unload_at == "response":
                started.set()
                await release.wait()
            for offset in range(0, len(self.body), size):
                yield self.body[offset : offset + size]

    class HeldSession(ToolSession):
        def request(self, method, url, **kwargs):
            response = super().request(method, url, **kwargs)
            self.responses[-1] = HeldResponse(body=response.body)
            return self.responses[-1]

    async def run_tool(tool_input):
        invoked.append(tool_input.id)
        if len(invoked) == 8:
            started.set()
        await release.wait()
        return llm.ToolResult(data={"ok": True})

    session = HeldSession(12, final_round=2)
    client = GroqApiClient(hass, api_key="fake", session=session)
    turn, log = tool_turn(hass, platform, session, run_tool, client=client)
    task = asyncio.create_task(turn())
    try:
        async with asyncio.timeout(3):
            await started.wait()
            await client.async_shutdown()
            release.set()
            with pytest.raises(GroqApiError, match="unloaded"):
                await task
        assert len(invoked) == (0 if unload_at == "response" else 8)
        if unload_at == "response":
            assert all(item.role != "tool_result" for item in log.content)
        assert len(session.responses) == 1 and session.responses[0].closed
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", PLATFORMS)
async def test_aggregate_tool_argument_budget_rejects_before_dispatch(hass, platform):
    arguments = json.dumps({"unused": "x" * (1024 * 1024)})
    session = ToolSession(2, final_round=2, arguments=arguments)

    async def run_tool(tool_input):
        pytest.fail(f"dispatched excess arguments for {tool_input.id}")

    turn, log = tool_turn(hass, platform, session, run_tool)
    with pytest.raises(GroqResponseError, match="tool arguments exceeded"):
        await turn()
    assert all(item.role != "tool_result" for item in log.content)
    assert len(session.responses) == 1 and session.responses[0].closed
