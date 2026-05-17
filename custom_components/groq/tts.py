"""
Setting up TTS entity.
"""

from __future__ import annotations
from contextlib import suppress
from typing import Any
import logging
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
PARALLEL_UPDATES = 1
ORPHEUS_RESPONSE_FORMAT = "wav"
FFMPEG_OUTPUT_ARGS = {
    "wav": ["-ac", "1", "-ar", "24000", "-f", "wav"],
    "mp3": ["-ac", "1", "-ar", "24000", "-b:a", "128k", "-f", "mp3"],
    "flac": ["-ac", "1", "-ar", "24000", "-compression_level", "5", "-f", "flac"],
}
FFMPEG_LOUDNORM_FILTER = "loudnorm=I=-16:TP=-1:LRA=5"


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
        return normalize_audio or output_format != ORPHEUS_RESPONSE_FORMAT

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
            effective_input = options.get(CONF_INPUT, message)
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
                    effective_input = f"{direction_text} {effective_input}"

            if len(effective_input) > MAX_TTS_INPUT_CHARS:
                raise ValueError(
                    f"Message exceeds Groq Orpheus TTS maximum length of {MAX_TTS_INPUT_CHARS} characters"
                )

            effective_model = options.get(
                CONF_MODEL,
                _entry_value(self._config, CONF_MODEL, service_data=self._service_data),
            )
            effective_voice = options.get(
                CONF_VOICE,
                _entry_value(self._config, CONF_VOICE, service_data=self._service_data),
            )
            output_format = options.get(
                CONF_RESPONSE_FORMAT,
                _entry_value(
                    self._config,
                    CONF_RESPONSE_FORMAT,
                    DEFAULT_RESPONSE_FORMAT,
                    service_data=self._service_data,
                ),
            )
            output_format = _normalize_response_format(output_format)

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
            _LOGGER.debug("Normalization option: %s", normalize_audio)
            needs_ffmpeg = normalize_audio or output_format != ORPHEUS_RESPONSE_FORMAT
            if needs_ffmpeg:
                await self._async_check_ffmpeg(output_format, normalize_audio)

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

            _LOGGER.debug("Creating TTS API request")
            api_start = time.monotonic()
            audio_content = await self._client.async_synthesize_speech(
                SpeechRequest(
                    text=effective_input,
                    model=effective_model,
                    voice=effective_voice,
                    response_format=ORPHEUS_RESPONSE_FORMAT,
                    service_id=self._service_data.get(UNIQUE_ID),
                    protect_free_tier=protect_free_tier,
                    cache_max=int(
                        _entry_value(
                            self._config,
                            CONF_CACHE_SIZE,
                            DEFAULT_CACHE_SIZE,
                            service_data=self._service_data,
                        )
                    ),
                )
            )
            api_duration = (time.monotonic() - api_start) * 1000
            _LOGGER.debug("TTS API call completed in %.2f ms", api_duration)

            if needs_ffmpeg:
                # Orpheus returns WAV; ffmpeg handles optional loudness and
                # conversion into common Home Assistant speaker formats.
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
                try:
                    audio_content = await self._async_run_ffmpeg(
                        cmd,
                        audio_content,
                        create_repair=False,
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
            _LOGGER.error("TTS audio generation failed: %s", err)
            return None, None
        except Exception:
            _LOGGER.exception("Unknown error in async_get_tts_audio")
        return None, None
