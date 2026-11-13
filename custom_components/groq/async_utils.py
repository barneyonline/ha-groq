"""Helpers for draining owned async work before propagating cancellation."""

from __future__ import annotations

import asyncio


async def async_finish[T](task: asyncio.Future[T]) -> T:
    """Drain owned work even if cleanup receives repeated cancellation."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def async_await_owned[T](task: asyncio.Future[T]) -> T:
    """Wait for owned work to finish before propagating cancellation."""
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await async_finish(task)
        raise
