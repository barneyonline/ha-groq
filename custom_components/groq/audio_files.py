"""Cancellation-safe temporary files for ffmpeg audio stitching."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from homeassistant.core import HomeAssistant

from .errors import translated_error

MAX_FFMPEG_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_FFMPEG_STDERR_BYTES = 64 * 1024


async def _read_output(
    stream: asyncio.StreamReader, limit: int, truncate: bool
) -> bytes:
    """Bound decoded audio, and retain only a prefix while draining stderr."""
    content = bytearray()
    while chunk := await stream.read(64 * 1024):
        remaining = limit - len(content)
        if len(chunk) > remaining and not truncate:
            raise translated_error(
                "ffmpeg output exceeded its byte limit", "ffmpeg_failed"
            )
        content.extend(chunk[:remaining])
    return bytes(content)


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
        await asyncio.gather(*tasks, return_exceptions=True)


def _prepare_audio_chunks(chunks: list[bytes]) -> tuple[str, list[str]]:
    """Create and populate a directory in one executor operation."""
    directory = tempfile.mkdtemp(prefix="groq-audio-")
    try:
        paths = []
        for index, chunk in enumerate(chunks):
            path = Path(directory) / f"chunk-{index}.wav"
            path.write_bytes(chunk)
            paths.append(str(path))
        return directory, paths
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


async def _finish[T](task: asyncio.Future[T]) -> T:
    """Drain owned executor work even if cleanup receives repeated cancellation."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


@asynccontextmanager
async def async_audio_chunk_paths(
    hass: HomeAssistant, chunks: list[bytes]
) -> AsyncIterator[list[str]]:
    """Own creation, writes and cleanup until all executor operations finish."""
    creation = asyncio.ensure_future(
        hass.async_add_executor_job(_prepare_audio_chunks, chunks)
    )
    directory: str | None = None
    try:
        try:
            directory, paths = await asyncio.shield(creation)
        except asyncio.CancelledError:
            prepared = await _finish(creation)
            directory = prepared[0]
            raise
        yield paths
    finally:
        if directory is not None:
            cleanup = asyncio.ensure_future(
                hass.async_add_executor_job(shutil.rmtree, directory, True)
            )
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await _finish(cleanup)
                raise
