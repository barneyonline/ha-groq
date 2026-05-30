"""
Setting up TTS entity.
"""

from __future__ import annotations
from contextlib import suppress
from typing import Any
import logging
from pathlib import Path
import re
import shutil
import struct
import tempfile
import time
import asyncio
from asyncio import CancelledError

from homeassistant.components.tts import TextToSpeechEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.exceptions import HomeAssistantError
from .const import (
    CONF_SERVICE_TYPE,
    CONF_SUBENTRY_ID,
    CONF_INPUT,
    CONF_MODEL,
    CONF_NAME,
    CONF_RESPONSE_FORMAT,
    CONF_VOICE,
    CONF_VOCAL_DIRECTIONS,
    CONF_URL,
    DOMAIN,
    UNIQUE_ID,
    CONF_NORMALIZE_AUDIO,
    CONF_CACHE_SIZE,
    CONF_ENABLE_LONG_TTS,
    CONF_PROTECT_FREE_TIER,
    DEFAULT_CACHE_SIZE,
    DEFAULT_PROTECT_FREE_TIER,
    DEFAULT_RESPONSE_FORMAT,
    FEATURE_TEXT_TO_SPEECH,
)
from .api import GroqApiClient, SpeechRequest, async_preload_clientsession_helper
from .repairs import (
    async_create_ffmpeg_missing_issue,
    async_delete_ffmpeg_missing_issue,
)
from .runtime import async_get_runtime

_LOGGER = logging.getLogger(__name__)

MAX_TTS_INPUT_CHARS = 200
MAX_LONG_TTS_CHUNKS = 10
PARALLEL_UPDATES = 1
ORPHEUS_RESPONSE_FORMAT = "wav"
FFMPEG_OUTPUT_ARGS = {
    "wav": ["-ac", "1", "-ar", "24000", "-f", "wav"],
    "mp3": ["-ac", "1", "-ar", "44100", "-b:a", "128k", "-f", "mp3"],
    "flac": ["-ac", "1", "-ar", "24000", "-compression_level", "5", "-f", "flac"],
}
FFMPEG_LOUDNORM_FILTER = "loudnorm=I=-16:TP=-1:LRA=5"
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_WAV_FORMAT_PCM = 1
_WAV_BITS_PER_SAMPLE = 16


def _audio_needs_compatibility_transcode(audio: bytes) -> bool:
    """Return whether audio bytes should be rewritten for WAV playback."""
    if not audio.startswith(b"RIFF") or audio[8:12] != b"WAVE":
        return True

    offset = 12
    has_compatible_format = False
    while offset + 8 <= len(audio):
        chunk_id = audio[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", audio, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(audio):
            return True
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                return True
            audio_format, _channels, _sample_rate, _byte_rate, _block_align, bits = (
                struct.unpack_from("<HHIIHH", audio, chunk_start)
            )
            if audio_format != _WAV_FORMAT_PCM or bits != _WAV_BITS_PER_SAMPLE:
                return True
            has_compatible_format = True
        elif chunk_id == b"data" and has_compatible_format:
            return chunk_size == 0
        offset = chunk_end + (chunk_size % 2)
    return True


def _split_overlong_tts_segment(segment: str, max_chars: int) -> list[str]:
    """Split a segment that cannot fit in one Groq Orpheus TTS request."""
    chunks: list[str] = []
    current = ""
    for word in segment.split():
        if len(word) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                word[start : start + max_chars]
                for start in range(0, len(word), max_chars)
            )
            continue
        if current and len(f"{current} {word}") > max_chars:
            chunks.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        chunks.append(current)
    if chunks:
        return chunks
    return [
        segment[start : start + max_chars]
        for start in range(0, len(segment), max_chars)
    ]


def _split_tts_text(text: str, max_chars: int) -> list[str]:
    """Split TTS text on sentence boundaries, then words if needed."""
    stripped_text = text.strip()
    if not stripped_text:
        return [""]
    chunks: list[str] = []
    current = ""
    for segment in _SENTENCE_BOUNDARY.split(stripped_text):
        segment = segment.strip()
        if not segment:
            continue
        segment_chunks = (
            [segment]
            if len(segment) <= max_chars
            else _split_overlong_tts_segment(segment, max_chars)
        )
        for segment_chunk in segment_chunks:
            if current and len(f"{current} {segment_chunk}") <= max_chars:
                current = f"{current} {segment_chunk}"
                continue
            if current:
                chunks.append(current)
            current = segment_chunk
    if current:
        chunks.append(current)
    return chunks


def _tts_input_chunks(
    text: str,
    vocal_directions: str,
    max_chars: int = MAX_TTS_INPUT_CHARS,
) -> list[str]:
    """Return Groq TTS request inputs, each within the Orpheus limit."""
    prefix = vocal_directions.strip()
    if prefix:
        available_chars = max_chars - len(prefix) - 1
        if available_chars <= 0:
            raise ValueError(
                "Vocal directions leave no room for TTS input within Groq Orpheus "
                f"maximum length of {max_chars} characters"
            )
        return [f"{prefix} {chunk}" for chunk in _split_tts_text(text, available_chars)]
    return _split_tts_text(text, max_chars)


def _write_audio_chunks(temp_dir: str, input_chunks: list[bytes]) -> list[str]:
    """Write audio chunks to temporary files and return their paths."""
    temp_path = Path(temp_dir)
    chunk_paths = []
    for index, chunk in enumerate(input_chunks):
        chunk_path = temp_path / f"chunk-{index}.wav"
        chunk_path.write_bytes(chunk)
        chunk_paths.append(str(chunk_path))
    return chunk_paths


def _entry_value(
    config_entry: ConfigEntry,
    key: str,
    default: Any = None,
    service_data: dict[str, Any] | None = None,
) -> Any:
    """Return the effective value, allowing options to override setup data."""
    if service_data and key in service_data:
        return service_data[key]
    return config_entry.options.get(key, config_entry.data.get(key, default))


def _normalize_bool_option(value: Any, option: str) -> bool:
    """Return a boolean option value, accepting common service-call strings."""
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"TTS {option} option must be a boolean")


def _normalize_response_format(value: Any) -> str:
    """Return a normalized TTS output format."""
    output_format = DEFAULT_RESPONSE_FORMAT if value in (None, "") else value
    if not isinstance(output_format, str):
        raise ValueError("TTS output format must be a string")
    output_format = output_format.strip().lower()
    if output_format not in FFMPEG_OUTPUT_ARGS:
        raise ValueError(f"Unsupported TTS output format: {output_format}")
    return output_format


def _tts_service_data(config_entry: ConfigEntry) -> list[dict[str, Any] | None]:
    """Return TTS service subentry data for an entry."""
    subentries = getattr(config_entry, "subentries", None) or {}
    services: list[dict[str, Any] | None] = []
    for subentry in subentries.values():
        data = dict(getattr(subentry, "data", {}))
        if data.get(CONF_SERVICE_TYPE) == FEATURE_TEXT_TO_SPEECH:
            subentry_id = getattr(subentry, "subentry_id", data.get(UNIQUE_ID))
            data[CONF_SUBENTRY_ID] = subentry_id
            data[UNIQUE_ID] = subentry_id
            services.append(data)
    if services:
        return services
    if all(config_entry.data.get(key) for key in (CONF_URL, CONF_MODEL, CONF_VOICE)):
        return [None]
    return []


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    runtime = await async_get_runtime(hass, config_entry)
    service_data_list = _tts_service_data(config_entry)
    if service_data_list:
        await async_preload_clientsession_helper(hass)
    entities = []
    for service_data in service_data_list:
        entity = GroqTTSEntity(hass, config_entry, runtime.client, service_data)
        if service_data:
            async_add_entities(
                [entity],
                config_subentry_id=service_data.get(CONF_SUBENTRY_ID),
            )
        else:
            entities.append(entity)
    if entities:
        async_add_entities(entities)


class GroqTTSEntity(TextToSpeechEntity):
    # Home Assistant's TTS manager requires TextToSpeechEntity.name to resolve
    # to a value before it will generate or stream audio, so this entity uses a
    # translated data-point name instead of a device-only name.
    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_translation_key = "text_to_speech"

    def __init__(
        self,
        hass: HomeAssistant,
        config: ConfigEntry,
        client: GroqApiClient,
        service_data: dict[str, Any] | None = None,
    ) -> None:
        self.hass = hass
        self._client = client
        self._config = config
        self._service_data = service_data or {}
        # Prefer the config entry unique_id; fall back to stored value for backward compatibility
        service_unique_id = self._service_data.get(UNIQUE_ID)
        self._attr_unique_id = (
            service_unique_id
            or getattr(config, "unique_id", None)
            or config.data.get(UNIQUE_ID)
        )
        if not self._attr_unique_id:
            self._attr_unique_id = (
                f"{config.data.get(CONF_URL)}_{config.data.get(CONF_MODEL)}"
            )
        self._service_name = _entry_value(
            config,
            CONF_NAME,
            _entry_value(config, CONF_MODEL, "", service_data=self._service_data),
            service_data=self._service_data,
        )
        self._ffmpeg_capabilities: set[tuple[str, bool]] = set()

    async def _async_run_ffmpeg(
        self,
        cmd: list[str],
        input_bytes: bytes | None = None,
        *,
        create_repair: bool = True,
    ) -> bytes:
        """Run ffmpeg without blocking Home Assistant's event loop."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            _LOGGER.error(
                "ffmpeg executable not found. Please install ffmpeg or adjust PATH."
            )
            if create_repair:
                async_create_ffmpeg_missing_issue(
                    self.hass, self._config, self._service_data
                )
            raise HomeAssistantError("ffmpeg not found")
        except OSError as err:
            _LOGGER.error("Unable to start ffmpeg: %s", err)
            if create_repair:
                async_create_ffmpeg_missing_issue(
                    self.hass, self._config, self._service_data
                )
            raise HomeAssistantError("ffmpeg could not start") from err
        try:
            stdout, stderr = await process.communicate(input=input_bytes)
        except CancelledError:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
                with suppress(Exception, CancelledError):
                    await process.wait()
            raise
        if process.returncode != 0:
            stderr_text = stderr.decode(errors="replace").strip()
            _LOGGER.error("ffmpeg error: %s", stderr_text or process.returncode)
            if create_repair:
                async_create_ffmpeg_missing_issue(
                    self.hass, self._config, self._service_data
                )
            raise HomeAssistantError("ffmpeg failed")
        return stdout

    async def _async_check_ffmpeg(
        self,
        output_format: str,
        normalize_audio: bool,
    ) -> None:
        """Ensure ffmpeg can write the requested format before spending Groq quota."""
        capability = (output_format, normalize_audio)
        if capability in self._ffmpeg_capabilities:
            return
        await self._async_run_ffmpeg(["ffmpeg", "-version"])
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=24000:cl=mono",
            "-t",
            "0.01",
        ]
        if normalize_audio:
            cmd.extend(["-af", FFMPEG_LOUDNORM_FILTER])
        cmd.extend(FFMPEG_OUTPUT_ARGS[output_format])
        cmd.append("pipe:1")
        await self._async_run_ffmpeg(cmd)
        self._ffmpeg_capabilities.add(capability)

    def _configured_audio_needs_ffmpeg(self) -> bool:
        """Return whether the stored TTS defaults require ffmpeg."""
        try:
            output_format = _normalize_response_format(
                _entry_value(
                    self._config,
                    CONF_RESPONSE_FORMAT,
                    DEFAULT_RESPONSE_FORMAT,
                    service_data=self._service_data,
                )
            )
        except ValueError:
            return True
        try:
            normalize_audio = _normalize_bool_option(
                _entry_value(
                    self._config,
                    CONF_NORMALIZE_AUDIO,
                    False,
                    service_data=self._service_data,
                ),
                CONF_NORMALIZE_AUDIO,
            )
        except ValueError:
            return True
        try:
            enable_long_tts = _normalize_bool_option(
                _entry_value(
                    self._config,
                    CONF_ENABLE_LONG_TTS,
                    False,
                    service_data=self._service_data,
                ),
                CONF_ENABLE_LONG_TTS,
            )
        except ValueError:
            return True
        return (
            normalize_audio
            or enable_long_tts
            or output_format != ORPHEUS_RESPONSE_FORMAT
        )

    @property
    def default_language(self) -> str:
        return "en"

    @property
    def supported_options(self) -> list:
        # Must match option keys actually read from service/data
        return [
            CONF_INPUT,
            CONF_MODEL,
            CONF_NORMALIZE_AUDIO,
            CONF_RESPONSE_FORMAT,
            CONF_VOICE,
            CONF_VOCAL_DIRECTIONS,
        ]

    @property
    def default_options(self) -> dict:
        """Advertise default options for the TTS service."""
        normalize_audio = _entry_value(
            self._config,
            CONF_NORMALIZE_AUDIO,
            False,
            service_data=self._service_data,
        )
        response_format = _entry_value(
            self._config,
            CONF_RESPONSE_FORMAT,
            DEFAULT_RESPONSE_FORMAT,
            service_data=self._service_data,
        )
        try:
            normalize_audio = _normalize_bool_option(
                normalize_audio, CONF_NORMALIZE_AUDIO
            )
        except ValueError:
            normalize_audio = False
        try:
            response_format = _normalize_response_format(response_format)
        except ValueError:
            response_format = DEFAULT_RESPONSE_FORMAT
        return {
            CONF_NORMALIZE_AUDIO: normalize_audio,
            CONF_MODEL: _entry_value(
                self._config, CONF_MODEL, service_data=self._service_data
            ),
            CONF_VOICE: _entry_value(
                self._config, CONF_VOICE, service_data=self._service_data
            ),
            CONF_RESPONSE_FORMAT: response_format,
            CONF_VOCAL_DIRECTIONS: _entry_value(
                self._config,
                CONF_VOCAL_DIRECTIONS,
                "",
                service_data=self._service_data,
            ),
        }

    @property
    def supported_languages(self) -> list:
        return ["ar", "en"]

    @property
    def device_info(self) -> dict:
        return {
            "identifiers": {(DOMAIN, self._attr_unique_id)},
            "model": _entry_value(
                self._config, CONF_MODEL, service_data=self._service_data
            ),
            "manufacturer": "Groq",
            "name": self._service_name,
        }

    async def async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict | None = None,
    ) -> tuple[str, bytes] | tuple[None, None]:
        """Generate TTS audio asynchronously and optionally normalize it."""
        overall_start = time.monotonic()

        options = options or {}

        try:
            effective_input = str(options.get(CONF_INPUT, message))
            if not effective_input.strip():
                raise ValueError("TTS input cannot be empty")
            vocal_directions = options.get(
                CONF_VOCAL_DIRECTIONS,
                _entry_value(
                    self._config,
                    CONF_VOCAL_DIRECTIONS,
                    "",
                    service_data=self._service_data,
                ),
            )
            if vocal_directions:
                direction_text = str(vocal_directions).strip()
                if direction_text:
                    # Orpheus-style vocal directions are bracketed in the input
                    # text. Let users enter either "warm" or "[warm]".
                    if not (
                        direction_text.startswith("[") and direction_text.endswith("]")
                    ):
                        direction_text = f"[{direction_text}]"
            else:
                direction_text = ""

            effective_model = options.get(
                CONF_MODEL,
                _entry_value(self._config, CONF_MODEL, service_data=self._service_data),
            )
            effective_voice = options.get(
                CONF_VOICE,
                _entry_value(self._config, CONF_VOICE, service_data=self._service_data),
            )
            output_format = _normalize_response_format(
                options.get(
                    CONF_RESPONSE_FORMAT,
                    _entry_value(
                        self._config,
                        CONF_RESPONSE_FORMAT,
                        DEFAULT_RESPONSE_FORMAT,
                        service_data=self._service_data,
                    ),
                )
            )
            normalize_audio = _normalize_bool_option(
                options.get(
                    CONF_NORMALIZE_AUDIO,
                    _entry_value(
                        self._config,
                        CONF_NORMALIZE_AUDIO,
                        False,
                        service_data=self._service_data,
                    ),
                ),
                CONF_NORMALIZE_AUDIO,
            )
            enable_long_tts = _normalize_bool_option(
                _entry_value(
                    self._config,
                    CONF_ENABLE_LONG_TTS,
                    False,
                    service_data=self._service_data,
                ),
                CONF_ENABLE_LONG_TTS,
            )
            _LOGGER.debug("Normalization option: %s", normalize_audio)
            input_chunks = _tts_input_chunks(effective_input, direction_text)
            try:
                protect_free_tier = _normalize_bool_option(
                    _entry_value(
                        self._config,
                        CONF_PROTECT_FREE_TIER,
                        DEFAULT_PROTECT_FREE_TIER,
                        service_data=self._service_data,
                    ),
                    CONF_PROTECT_FREE_TIER,
                )
            except ValueError:
                protect_free_tier = DEFAULT_PROTECT_FREE_TIER
            cache_max = int(
                _entry_value(
                    self._config,
                    CONF_CACHE_SIZE,
                    DEFAULT_CACHE_SIZE,
                    service_data=self._service_data,
                )
            )
            speech_requests = [
                SpeechRequest(
                    text=input_chunk,
                    model=effective_model,
                    voice=effective_voice,
                    response_format=ORPHEUS_RESPONSE_FORMAT,
                    service_id=self._service_data.get(UNIQUE_ID),
                    protect_free_tier=protect_free_tier,
                    cache_max=cache_max,
                )
                for input_chunk in input_chunks
            ]
            if len(input_chunks) > 1 and not enable_long_tts:
                raise ValueError(
                    "Message exceeds Groq Orpheus TTS maximum length of "
                    f"{MAX_TTS_INPUT_CHARS} characters. Enable Long TTS "
                    "to synthesize and stitch longer announcements."
                )
            if len(input_chunks) > MAX_LONG_TTS_CHUNKS:
                raise ValueError(
                    "Message requires too many Groq Orpheus TTS chunks "
                    f"({len(input_chunks)}). Shorten the announcement to "
                    f"{MAX_LONG_TTS_CHUNKS} chunks or fewer."
                )
            needs_ffmpeg = (
                normalize_audio
                or output_format != ORPHEUS_RESPONSE_FORMAT
                or len(input_chunks) > 1
            )
            if needs_ffmpeg:
                await self._async_check_ffmpeg(output_format, normalize_audio)
            if len(input_chunks) > 1 and callable(
                batch_guard := getattr(
                    self._client, "_check_local_tts_free_tier_batch", None
                )
            ):
                batch_guard(speech_requests)

            audio_chunks = []
            api_start = time.monotonic()
            for speech_request in speech_requests:
                _LOGGER.debug("Creating TTS API request")
                audio_chunks.append(
                    await self._client.async_synthesize_speech(speech_request)
                )
            api_duration = (time.monotonic() - api_start) * 1000
            _LOGGER.debug(
                "TTS API call%s completed in %.2f ms",
                "s" if len(audio_chunks) != 1 else "",
                api_duration,
            )
            audio_content = audio_chunks[0]
            if (
                not needs_ffmpeg
                and output_format == ORPHEUS_RESPONSE_FORMAT
                and _audio_needs_compatibility_transcode(audio_content)
            ):
                _LOGGER.debug("Rewriting Groq WAV audio for media player compatibility")
                needs_ffmpeg = True
                await self._async_check_ffmpeg(output_format, normalize_audio)

            async def convert_audio_chunk(input_bytes: bytes) -> bytes:
                """Convert one Orpheus WAV chunk to the requested playback profile."""
                cmd = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    "pipe:0",
                ]
                if normalize_audio:
                    cmd.extend(["-af", FFMPEG_LOUDNORM_FILTER])
                cmd.extend(FFMPEG_OUTPUT_ARGS[output_format])
                cmd.append("pipe:1")
                return await self._async_run_ffmpeg(
                    cmd,
                    input_bytes,
                    create_repair=False,
                )

            async def stitch_audio_chunks(
                input_chunks: list[bytes],
            ) -> bytes:
                """Stitch multiple Orpheus WAV chunks into the playback format."""
                if hasattr(self.hass, "async_add_executor_job"):
                    temp_dir = await self.hass.async_add_executor_job(tempfile.mkdtemp)
                else:
                    temp_dir = tempfile.mkdtemp()
                try:
                    if hasattr(self.hass, "async_add_executor_job"):
                        chunk_paths = await self.hass.async_add_executor_job(
                            _write_audio_chunks, temp_dir, input_chunks
                        )
                    else:
                        chunk_paths = _write_audio_chunks(temp_dir, input_chunks)
                    input_args = [
                        arg for chunk_path in chunk_paths for arg in ("-i", chunk_path)
                    ]
                    filter_inputs = "".join(
                        f"[{index}:a]" for index in range(len(chunk_paths))
                    )
                    filter_complex = (
                        f"{filter_inputs}concat=n={len(chunk_paths)}:v=0:a=1"
                    )
                    if normalize_audio:
                        filter_complex = f"{filter_complex},{FFMPEG_LOUDNORM_FILTER}"
                    cmd = [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        *input_args,
                        "-filter_complex",
                        filter_complex,
                        *FFMPEG_OUTPUT_ARGS[output_format],
                        "pipe:1",
                    ]
                    return await self._async_run_ffmpeg(
                        cmd,
                        create_repair=False,
                    )
                finally:
                    if hasattr(self.hass, "async_add_executor_job"):
                        await self.hass.async_add_executor_job(
                            shutil.rmtree, temp_dir, True
                        )
                    else:
                        shutil.rmtree(temp_dir, ignore_errors=True)

            if needs_ffmpeg:
                try:
                    audio_content = (
                        await convert_audio_chunk(audio_content)
                        if len(audio_chunks) == 1
                        else await stitch_audio_chunks(audio_chunks)
                    )
                except HomeAssistantError:
                    self._ffmpeg_capabilities.discard((output_format, normalize_audio))
                    async_create_ffmpeg_missing_issue(
                        self.hass, self._config, self._service_data
                    )
                    raise
                async_delete_ffmpeg_missing_issue(
                    self.hass, self._config, self._service_data
                )
            else:
                if not self._configured_audio_needs_ffmpeg():
                    async_delete_ffmpeg_missing_issue(
                        self.hass, self._config, self._service_data
                    )

            overall_duration = (time.monotonic() - overall_start) * 1000
            _LOGGER.debug("Overall TTS processing time: %.2f ms", overall_duration)
            return output_format, audio_content

        except CancelledError:
            _LOGGER.debug("TTS task cancelled")
            return None, None
        except ValueError as err:
            _LOGGER.error("Invalid TTS request: %s", err)
            return None, None
        except HomeAssistantError as err:
            _LOGGER.error("TTS request failed: %s", err)
            return None, None
        except Exception:
            _LOGGER.exception("Unknown error in async_get_tts_audio")
        return None, None
