"""Speech-to-text platform for Groq."""

from __future__ import annotations

from collections.abc import AsyncIterable
import io
import logging
import wave

from homeassistant.components import stt
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_API_KEY,
    CONF_LANGUAGE,
    CONF_MODEL,
    CONF_NAME,
    CONF_SERVICE_TYPE,
    DOMAIN,
    FEATURE_SPEECH_TO_TEXT,
    STT_MODELS,
    UNIQUE_ID,
)
from .errors import GroqApiError
from .runtime import async_get_runtime

_LOGGER = logging.getLogger(__name__)


def _stt_service_data(config_entry: ConfigEntry) -> list[dict]:
    """Return configured Speech-to-Text service subentry data."""
    services = []
    for subentry in (getattr(config_entry, "subentries", None) or {}).values():
        data = dict(getattr(subentry, "data", {}))
        if data.get(CONF_SERVICE_TYPE) != FEATURE_SPEECH_TO_TEXT:
            continue
        data[UNIQUE_ID] = getattr(subentry, "subentry_id", data.get(UNIQUE_ID))
        services.append(data)
    return services


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Groq Speech-to-Text entities."""
    runtime = await async_get_runtime(hass, config_entry)
    entities = [
        GroqSTTEntity(config_entry, service_data, runtime.client)
        for service_data in _stt_service_data(config_entry)
    ]
    if entities:
        async_add_entities(entities)


class GroqSTTEntity(stt.SpeechToTextEntity):
    """Groq Speech-to-Text entity backed by a configured service."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(
        self,
        config_entry: ConfigEntry,
        service_data: dict,
        client,
    ) -> None:
        """Initialize the STT entity."""
        self._config_entry = config_entry
        self._service_data = service_data
        self._client = client
        self._attr_name = service_data.get(CONF_NAME, "Groq Speech-to-Text")
        self._service_unique_id = str(
            service_data.get(UNIQUE_ID)
            or getattr(config_entry, "unique_id", None)
            or config_entry.entry_id
        )
        self._attr_unique_id = f"{self._service_unique_id}_stt"

    @property
    def supported_languages(self) -> list[str]:
        """Return supported languages."""
        return [
            "en-US",
            "en-GB",
            "en",
            "de-DE",
            "es-ES",
            "fr-FR",
            "it-IT",
            "pt-PT",
            "nl-NL",
            "ja-JP",
            "ko-KR",
            "zh-CN",
        ]

    @property
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return supported audio formats."""
        return [stt.AudioFormats.WAV, stt.AudioFormats.OGG]

    @property
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return supported audio codecs."""
        return [stt.AudioCodecs.PCM, stt.AudioCodecs.OPUS]

    @property
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return supported audio bit rates."""
        return [
            stt.AudioBitRates.BITRATE_8,
            stt.AudioBitRates.BITRATE_16,
            stt.AudioBitRates.BITRATE_24,
            stt.AudioBitRates.BITRATE_32,
        ]

    @property
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return supported sample rates."""
        return [
            stt.AudioSampleRates.SAMPLERATE_8000,
            stt.AudioSampleRates.SAMPLERATE_11000,
            stt.AudioSampleRates.SAMPLERATE_16000,
            stt.AudioSampleRates.SAMPLERATE_18900,
            stt.AudioSampleRates.SAMPLERATE_22000,
            stt.AudioSampleRates.SAMPLERATE_32000,
            stt.AudioSampleRates.SAMPLERATE_37800,
            stt.AudioSampleRates.SAMPLERATE_44100,
            stt.AudioSampleRates.SAMPLERATE_48000,
        ]

    @property
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return supported audio channels."""
        return [stt.AudioChannels.CHANNEL_MONO, stt.AudioChannels.CHANNEL_STEREO]

    @property
    def device_info(self) -> dict:
        """Return device information."""
        return {
            "identifiers": {(DOMAIN, self._service_unique_id)},
            "manufacturer": "Groq",
            "model": self._service_data.get(CONF_MODEL, STT_MODELS[0]),
            "name": self._attr_name,
        }

    async def async_process_audio_stream(
        self,
        metadata: stt.SpeechMetadata,
        stream: AsyncIterable[bytes],
    ) -> stt.SpeechResult:
        """Transcribe an audio stream."""
        audio = bytearray()
        async for chunk in stream:
            audio.extend(chunk)
        audio_data = bytes(audio)
        filename = f"audio.{metadata.format.value}"

        if metadata.format == stt.AudioFormats.WAV:
            # HA streams WAV frames without a container header here, so wrap the
            # raw PCM bytes before sending them to Groq's transcription endpoint.
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(metadata.channel.value)
                wav_file.setsampwidth(metadata.bit_rate.value // 8)
                wav_file.setframerate(metadata.sample_rate.value)
                wav_file.writeframes(audio_data)
            audio_data = wav_buffer.getvalue()

        language = self._service_data.get(CONF_LANGUAGE) or metadata.language
        try:
            text = await self._client.async_transcribe_audio(
                audio=audio_data,
                filename=filename,
                model=self._service_data.get(CONF_MODEL, STT_MODELS[0]),
                language=language,
                api_key=self._service_data.get(CONF_API_KEY),
            )
        except GroqApiError:
            _LOGGER.exception("Error during Groq speech transcription")
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        if text:
            return stt.SpeechResult(text, stt.SpeechResultState.SUCCESS)
        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)
