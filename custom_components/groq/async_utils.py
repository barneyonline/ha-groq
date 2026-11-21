"""Helpers for draining owned async work before propagating cancellation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(slots=True)
class RequestFlight[T]:
    """A worker shared by callers with independent cancellation."""

    task: asyncio.Task[T]
    waiters: int = 0

    async def async_wait(self) -> T:
        """Cancel and drain the worker if the final caller leaves."""
        self.waiters += 1
        try:
            return await asyncio.shield(self.task)
        finally:
            self.waiters -= 1
            if (
                self.waiters == 0
                and not self.task.done()
                and not self.task.cancelling()
            ):
                self.task.cancel()
                await async_await_owned(
                    asyncio.gather(self.task, return_exceptions=True)
                )


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
