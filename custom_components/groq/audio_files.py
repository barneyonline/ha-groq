"""Cancellation-safe temporary files for ffmpeg audio stitching."""

from __future__ import annotations

import asyncio
import io
import logging
import shutil
import tempfile
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path

from homeassistant.core import HomeAssistant

from .async_utils import async_await_owned, async_finish
from .errors import GroqApiError, translated_error

_LOGGER = logging.getLogger(__name__)

MAX_FFMPEG_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_FFMPEG_STDERR_BYTES = 64 * 1024
FFMPEG_PIPE_CLEANUP_TIMEOUT_SECONDS = 0.5
MAX_AUDIO_SPOOL_BYTES = 250 * 1024 * 1024


@dataclass(slots=True)
class AudioSpoolBudget:
    """Bound concurrent long-announcement disk usage for one config entry."""

    used: int = 0
    failed_removals: dict[str, int] = field(default_factory=dict)

    def reserve(self, size: int) -> None:
        if self.used + size > MAX_AUDIO_SPOOL_BYTES:
            raise GroqApiError("Groq temporary audio storage limit reached")
        self.used += size


async def _async_remove_audio_directory(
    hass: HomeAssistant,
    directory: str,
    budget: AudioSpoolBudget | None,
    reserved: int,
) -> None:
    """Release charged bytes after the directory has been removed."""
    try:
        await hass.async_add_executor_job(shutil.rmtree, directory, False)
    except FileNotFoundError:
        pass
    except OSError as err:
        _LOGGER.warning("Could not remove temporary Groq audio files: %s", err)
        if budget is not None:
            budget.failed_removals[directory] = reserved
        return
    if budget is not None:
        budget.used -= reserved


async def _read_output(
    stream: asyncio.StreamReader, limit: int, truncate: bool
) -> bytes:
    """Bound decoded audio, and retain only a prefix while draining stderr."""
    with io.BytesIO() as content:
        while chunk := await stream.read(64 * 1024):
            remaining = limit - content.tell()
            if len(chunk) > remaining and not truncate:
                raise translated_error(
                    "ffmpeg output exceeded its byte limit", "ffmpeg_failed"
                )
            content.write(chunk[:remaining])
        return content.getvalue()


async def _feed_input(writer: asyncio.StreamWriter, content: bytes | None) -> None:
    """Feed input concurrently with output reads, closing stdin on every path."""
    try:
        if content:
            with suppress(BrokenPipeError, ConnectionResetError):
                writer.write(content)
                await writer.drain()
    finally:
        writer.close()


async def async_communicate_audio(
    process: asyncio.subprocess.Process, content: bytes | None
) -> tuple[bytes, bytes]:
    """Communicate without unbounded decoded output or abandoned pipe readers."""
    assert (
        process.stdin is not None
        and process.stdout is not None
        and process.stderr is not None
    )
    output = asyncio.create_task(
        _read_output(process.stdout, MAX_FFMPEG_OUTPUT_BYTES, False)
    )
    errors = asyncio.create_task(
        _read_output(process.stderr, MAX_FFMPEG_STDERR_BYTES, True)
    )
    feed = asyncio.create_task(_feed_input(process.stdin, content))
    tasks = (output, errors, feed)
    try:
        await asyncio.gather(*tasks)
        await process.wait()
        return output.result(), errors.result()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await async_await_owned(asyncio.gather(*tasks, return_exceptions=True))


async def _discard_output(stream: asyncio.StreamReader) -> None:
    """Resume paused pipes and discard their remaining output in bounded reads."""
    while await stream.read(64 * 1024):
        pass


async def async_stop_audio_process(process: asyncio.subprocess.Process) -> None:
    """Stop the child and release both output pipes, including after child exit."""
    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.kill()
    assert (
        process.stdin is not None
        and process.stdout is not None
        and process.stderr is not None
    )
    process.stdin.close()
    await async_await_owned(
        asyncio.create_task(
            _drain_audio_process(process, process.stdout, process.stderr)
        )
    )


async def _drain_audio_process(
    process: asyncio.subprocess.Process,
    stdout: asyncio.StreamReader,
    stderr: asyncio.StreamReader,
) -> None:
    """Allow paused pipes to drain without waiting indefinitely for descendants."""
    try:
        async with asyncio.timeout(FFMPEG_PIPE_CLEANUP_TIMEOUT_SECONDS):
            await asyncio.gather(
                _discard_output(stdout),
                _discard_output(stderr),
                process.wait(),
                return_exceptions=True,
            )
    except TimeoutError:
        # The high-level Process exposes no method to close its output pipes.
        # Closing its transport releases our ends even if a descendant writes.
        process._transport.close()  # type: ignore[attr-defined]


@asynccontextmanager
async def async_audio_chunk_paths(
    hass: HomeAssistant,
    chunks: AsyncGenerator[bytes],
    budget: AudioSpoolBudget | None = None,
) -> AsyncIterator[list[str]]:
    """Spool chunks in order and own every write until the directory is removed."""
    if budget is not None and budget.failed_removals:
        # Move the retry batch out before awaiting so concurrent calls cannot
        # remove the same directory or release its charge twice.
        pending = budget.failed_removals
        budget.failed_removals = {}

        async def retry_removals() -> None:
            for path, size in pending.items():
                await _async_remove_audio_directory(hass, path, budget, size)

        await async_await_owned(asyncio.create_task(retry_removals()))

    creation = asyncio.ensure_future(
        hass.async_add_executor_job(tempfile.mkdtemp, "", "groq-audio-")
    )
    directory: str | None = None
    reserved = 0
    try:
        try:
            directory = await asyncio.shield(creation)
        except asyncio.CancelledError:
            directory = await async_finish(creation)
            raise

        paths: list[str] = []
        async with aclosing(chunks):
            async for chunk in chunks:
                path = Path(directory) / f"chunk-{len(paths)}.wav"
                if budget is not None:
                    budget.reserve(len(chunk))
                    reserved += len(chunk)
                await async_await_owned(
                    asyncio.ensure_future(
                        hass.async_add_executor_job(path.write_bytes, chunk)
                    )
                )
                del chunk
                paths.append(str(path))
        yield paths
    finally:
        if directory is not None:
            await async_await_owned(
                asyncio.create_task(
                    _async_remove_audio_directory(hass, directory, budget, reserved)
                )
            )
