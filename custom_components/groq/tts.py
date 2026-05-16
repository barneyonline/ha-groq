"""
Setting up TTS entity.
"""

from __future__ import annotations
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
    CONF_API_KEY,
    CONF_SERVICE_TYPE,
    CONF_SUBENTRY_ID,
    CONF_INPUT,
    CONF_MODEL,
    CONF_NAME,
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
    DEFAULT_TTS_URL,
    FEATURE_TEXT_TO_SPEECH,
)
from .tts_engine import GroqTTSEngine, async_preload_clientsession_helper
from .repairs import (
    async_create_ffmpeg_missing_issue,
    async_delete_ffmpeg_missing_issue,
)

_LOGGER = logging.getLogger(__name__)

MAX_TTS_INPUT_CHARS = 200
PARALLEL_UPDATES = 1


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
    api_key = config_entry.data.get(CONF_API_KEY)
    service_data_list = _tts_service_data(config_entry)
    if service_data_list:
        await async_preload_clientsession_helper(hass)
    entities = []
    for service_data in service_data_list:
        engine = GroqTTSEngine(
            _entry_value(config_entry, CONF_API_KEY, api_key),
            _entry_value(config_entry, CONF_VOICE, service_data=service_data),
            _entry_value(config_entry, CONF_MODEL, service_data=service_data),
            _entry_value(
                config_entry,
                CONF_URL,
                DEFAULT_TTS_URL,
                service_data=service_data,
            ),
            cache_max=_entry_value(
                config_entry,
                CONF_CACHE_SIZE,
                DEFAULT_CACHE_SIZE,
                service_data=service_data,
            ),
            protect_free_tier=(service_data or {}).get(
                CONF_PROTECT_FREE_TIER,
                DEFAULT_PROTECT_FREE_TIER,
            ),
            response_format=DEFAULT_RESPONSE_FORMAT,
        )
        entity = GroqTTSEntity(hass, config_entry, engine, service_data)
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
        engine: GroqTTSEngine,
        service_data: dict[str, Any] | None = None,
    ) -> None:
        self.hass = hass
        self._engine = engine
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
            CONF_VOICE,
            CONF_VOCAL_DIRECTIONS,
        ]

    @property
    def default_options(self) -> dict:
        """Advertise default options for the TTS service."""
        return {
            CONF_NORMALIZE_AUDIO: False,
            CONF_MODEL: _entry_value(
                self._config, CONF_MODEL, service_data=self._service_data
            ),
            CONF_VOICE: _entry_value(
                self._config, CONF_VOICE, service_data=self._service_data
            ),
            CONF_VOCAL_DIRECTIONS: _entry_value(
                self._config,
                CONF_VOCAL_DIRECTIONS,
                "",
                service_data=self._service_data,
            ),
        }

    @property
    def supported_languages(self) -> list:
        return self._engine.get_supported_langs()

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
            effective_response_format = DEFAULT_RESPONSE_FORMAT

            _LOGGER.debug("Creating TTS API request")
            api_start = time.monotonic()
            speech = await self._engine.async_get_tts(
                self.hass,
                effective_input,
                voice=effective_voice,
                model=effective_model,
                response_format=effective_response_format,
            )
            api_duration = (time.monotonic() - api_start) * 1000
            _LOGGER.debug("TTS API call completed in %.2f ms", api_duration)
            audio_content = speech.content

            normalize_audio = options.get(
                CONF_NORMALIZE_AUDIO,
                _entry_value(
                    self._config,
                    CONF_NORMALIZE_AUDIO,
                    False,
                    service_data=self._service_data,
                ),
            )
            _LOGGER.debug("Normalization option: %s", normalize_audio)

            async def run_ffmpeg(cmd, input_bytes):
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
                    async_create_ffmpeg_missing_issue(
                        self.hass, self._config, self._service_data
                    )
                    raise HomeAssistantError("ffmpeg not found")
                stdout, stderr = await process.communicate(input=input_bytes)
                if process.returncode != 0:
                    _LOGGER.error("ffmpeg error: %s", stderr.decode())
                    raise HomeAssistantError("ffmpeg failed")
                return stdout

            if normalize_audio:
                # Normalization converts Groq's returned audio into a single
                # predictable MP3 profile for media players that handle raw WAV
                # volume or channel layout inconsistently.
                cmd = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    "pipe:0",
                    "-ac",
                    "1",
                    "-ar",
                    "24000",
                    "-b:a",
                    "128k",
                    "-af",
                    "loudnorm=I=-16:TP=-1:LRA=5",
                    "-f",
                    "mp3",
                    "pipe:1",
                ]
                audio_content = await run_ffmpeg(cmd, audio_content)
                async_delete_ffmpeg_missing_issue(
                    self.hass, self._config, self._service_data
                )

            overall_duration = (time.monotonic() - overall_start) * 1000
            _LOGGER.debug("Overall TTS processing time: %.2f ms", overall_duration)
            return (
                "mp3" if normalize_audio else effective_response_format
            ), audio_content

        except CancelledError:
            _LOGGER.debug("TTS task cancelled")
            return None, None
        except ValueError as err:
            _LOGGER.error("Invalid TTS request: %s", err)
            return None, None
        except Exception:
            _LOGGER.exception("Unknown error in async_get_tts_audio")
        return None, None
