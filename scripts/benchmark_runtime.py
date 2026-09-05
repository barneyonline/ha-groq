"""Opt-in HA setup/action benchmark: pytest scripts/benchmark_runtime.py -q -s.

Uses real HA lifecycle and service dispatch with deterministic mocked Groq I/O.
Numbers describe local overhead, not provider latency or production throughput.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from importlib.metadata import version
import json
import platform
import statistics
import time
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.groq.api import ChatCompletionResult, GroqApiClient
from custom_components.groq.model_registry import BUILT_IN_MODELS

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.usefixtures("enable_custom_integrations"),
]


async def test_runtime_benchmark(hass):
    """Measure warmed setup/unload and uncached action dispatch with fake I/O."""
    assert await async_setup_component(hass, "homeassistant", {})
    model = "llama-3.3-70b-versatile"
    entry = MockConfigEntry(
        domain="groq",
        unique_id="runtime-benchmark",
        minor_version=2,
        data={"api_key": "benchmark-placeholder"},
        subentries_data=[
            {
                "subentry_id": "benchmark-text",
                "subentry_type": "text_generation",
                "title": "Benchmark text",
                "unique_id": None,
                "data": {
                    "service_type": "text_generation",
                    "name": "Benchmark text",
                    "model": model,
                    "prompt_caching": False,
                },
            }
        ],
    )
    entry.add_to_hass(hass)
    generation = AsyncMock(
        return_value=ChatCompletionResult(text="ok", model=model, usage={}, raw={})
    )
    peak_tick_ms = 0.0

    async def heartbeat():
        nonlocal peak_tick_ms
        previous = time.perf_counter()
        while True:
            await asyncio.sleep(0)
            now = time.perf_counter()
            peak_tick_ms = max(peak_tick_ms, (now - previous) * 1000)
            previous = now

    setup_ms = []
    action_ms = []
    ticker = asyncio.create_task(heartbeat())
    try:
        with (
            patch.object(
                GroqApiClient,
                "async_list_models",
                return_value=list(BUILT_IN_MODELS.values()),
            ),
            patch.object(GroqApiClient, "async_generate_text", generation),
        ):
            # The first setup primes dependency loading; time subsequent reloads.
            assert await hass.config_entries.async_setup(entry.entry_id)
            await hass.async_block_till_done()
            assert await hass.config_entries.async_unload(entry.entry_id)
            await hass.async_block_till_done()
            peak_tick_ms = 0.0
            for _ in range(5):
                started = time.perf_counter()
                assert await hass.config_entries.async_setup(entry.entry_id)
                await hass.async_block_till_done()
                setup_ms.append((time.perf_counter() - started) * 1000)
                for _ in range(20):
                    started = time.perf_counter()
                    response = await hass.services.async_call(
                        "groq",
                        "generate_text",
                        {"service_id": "benchmark-text", "prompt": "Return ok"},
                        blocking=True,
                        return_response=True,
                    )
                    assert response["text"] == "ok"
                    action_ms.append((time.perf_counter() - started) * 1000)
                    await asyncio.sleep(0)
                assert await hass.config_entries.async_unload(entry.entry_id)
                await hass.async_block_till_done()
            assert generation.await_count == 100
    finally:
        ticker.cancel()
        with suppress(asyncio.CancelledError):
            await ticker
    print(
        json.dumps(
            {
                "homeassistant": version("homeassistant"),
                "python": platform.python_version(),
                "setup_samples": len(setup_ms),
                "action_samples": len(action_ms),
                "warmed_setup_median_ms": statistics.median(setup_ms),
                "uncached_action_median_ms": statistics.median(action_ms),
                "uncached_action_p95_ms": sorted(action_ms)[94],
                "peak_event_loop_tick_ms": peak_tick_ms,
                "io": "mocked; imports excluded; not a live latency claim",
            },
            sort_keys=True,
        )
    )
