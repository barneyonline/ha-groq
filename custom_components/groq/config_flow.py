"""
Config flow for Groq.
"""

from __future__ import annotations
from typing import Any
import voluptuous as vol
import logging
import uuid
import hashlib

from homeassistant import data_entry_flow
from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    ConfigFlow,
    ConfigSubentryFlow,
    OptionsFlow,
)
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.core import callback

from .const import (
    CONF_ADVANCED_OPTIONS,
    CONF_API_KEY,
    CONF_ENABLED_FEATURES,
    CONF_MODEL,
    CONF_NAME,
    CONF_SERVICE_TYPE,
    DOMAIN,
    FEATURE_IMAGE_RECOGNITION,
    FEATURE_LABELS,
    FEATURE_SPEECH_TO_TEXT,
    FEATURE_TEXT_GENERATION,
    FEATURE_TEXT_TO_SPEECH,
    MODELS,
    SETUP_FEATURES,
    UNIQUE_ID,
    VOICES,
)
from .flow_schemas import (
    api_key_selector,
    clean_service_input,
    entry_defaults,
    image_recognition_schema,
    service_type_schema,
    setup_schema,
    speech_to_text_schema,
    text_generation_advanced_schema,
    text_generation_basic_schema,
    text_to_speech_schema,
    validate_text_generation_input,
    validate_user_input,
)

_LOGGER = logging.getLogger(__name__)


def generate_entry_id() -> str:
    return str(uuid.uuid4())


async def fetch_available(hass, endpoint: str, api_key: str | None = None) -> list[str]:
    """Fetch list of items from Groq API endpoint returning JSON data."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        session = async_get_clientsession(hass)
        async with session.get(endpoint, headers=headers, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.json()
                items = data.get("data") or data
                if isinstance(items, list):
                    names = []
                    for item in items:
                        if isinstance(item, dict):
                            name = item.get("id") or item.get("name")
                            if name:
                                names.append(name)
                        elif isinstance(item, str):
                            names.append(item)
                    return sorted(names)
    except Exception as err:  # pylint: disable=broad-except
        _LOGGER.debug("Error fetching %s: %s", endpoint, err)
    return []


def is_tts_model(model: str) -> bool:
    """Return True for Groq model ids that look usable for speech output."""
    model_id = model.lower()
    return model in MODELS or model_id.startswith("canopylabs/orpheus")


def get_model_options(discovered_models: list[str]) -> list[str]:
    """Return filtered dynamic model options while preserving built-in TTS models."""
    models = set(MODELS)
    models.update(model for model in discovered_models if is_tts_model(model))
    return sorted(models)


async def get_dynamic_options(hass, api_key: str | None) -> tuple[list[str], list[str]]:
    """Return a dynamic list of models and the built-in voices."""
    models_endpoint = "https://api.groq.com/openai/v1/models"
    models = get_model_options(await fetch_available(hass, models_endpoint, api_key))
    voices = VOICES
    return models, voices


class GroqConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Groq."""

    VERSION = 1
    data_schema = setup_schema()

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        errors = {}
        schema = setup_schema()
        if user_input is not None:
            try:
                await validate_user_input(user_input)
                entry_data = entry_defaults(user_input)
                # Create a deterministic unique_id from the API key to avoid duplicate account entries.
                uid_hash = hashlib.sha1(
                    entry_data[CONF_API_KEY].encode("utf-8")
                ).hexdigest()
                unique_id = f"groq_{uid_hash}"
                await self.async_set_unique_id(unique_id)
                # Allow multiple named Groq accounts, but do not create a
                # duplicate account for the same API key.
                self._abort_if_unique_id_configured()
                # Store unique id in data for backward-compat device identifiers
                entry_data[UNIQUE_ID] = unique_id
                return self.async_create_entry(
                    title=entry_data[CONF_NAME],
                    data=entry_data,
                )
            except data_entry_flow.AbortFlow:
                return self.async_abort(reason="already_configured")
            except ValueError as e:
                msg = str(e)
                if "API key is required" in msg:
                    errors[CONF_API_KEY] = "required"
                elif "Enabled features are invalid" in msg:
                    errors[CONF_ENABLED_FEATURES] = "invalid_enabled_features"
                else:
                    errors["base"] = "unknown_error"
            except Exception as e:
                _LOGGER.exception("Unexpected error in config flow: %s", e)
                errors["base"] = "unknown_error"
        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors,
        )

    @staticmethod
    def async_get_options_flow(config_entry):
        return GroqOptionsFlow()

    @classmethod
    @callback
    def async_get_supported_subentry_types(cls, config_entry):
        """Return subentry types supported by this integration."""
        return {
            FEATURE_TEXT_GENERATION: GroqServiceSubentryFlow,
            FEATURE_SPEECH_TO_TEXT: GroqServiceSubentryFlow,
            FEATURE_TEXT_TO_SPEECH: GroqServiceSubentryFlow,
            FEATURE_IMAGE_RECOGNITION: GroqServiceSubentryFlow,
        }

    async def async_step_reauth(
        self, entry_data: dict[str, Any]
    ) -> data_entry_flow.FlowResult:
        """Handle reauthentication when credentials are invalid."""
        # Store the entry we're reauthenticating for use in confirm step
        self._reauth_entry = self.hass.config_entries.async_get_entry(
            self.context.get("entry_id")
        )
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> data_entry_flow.FlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            api_key = user_input.get(CONF_API_KEY)
            if not api_key:
                errors[CONF_API_KEY] = "required"
            else:
                # Update only the API key, preserve other data
                reauth_entry = getattr(self, "_reauth_entry", None)
                if reauth_entry is None:
                    return self.async_abort(reason="unknown")
                new_data = dict(reauth_entry.data)
                new_data[CONF_API_KEY] = api_key
                # Abort current flow, update & reload the entry with new credentials
                return self.async_update_reload_and_abort(
                    reauth_entry,
                    data_updates=new_data,
                    reason="reauth_successful",
                )

        schema = vol.Schema({vol.Required(CONF_API_KEY): api_key_selector()})
        return self.async_show_form(
            step_id="reauth_confirm", data_schema=schema, errors=errors
        )


class GroqOptionsFlow(OptionsFlow):
    """Handle options flow for Groq."""

    async def async_step_init(self, user_input: dict | None = None):
        if user_input is not None:
            user_input = dict(user_input)
            if not user_input.get(CONF_API_KEY):
                user_input.pop(CONF_API_KEY, None)
            return self.async_create_entry(title="", data=user_input)
        options_schema = vol.Schema(
            {
                vol.Optional(
                    CONF_API_KEY,
                ): api_key_selector(),
            }
        )
        return self.async_show_form(step_id="init", data_schema=options_schema)


class GroqServiceSubentryFlow(ConfigSubentryFlow):
    """Handle Groq service instance subentries."""

    def __init__(self) -> None:
        """Initialize the service subentry flow."""
        self._service_type: str | None = None
        self._pending_service_data: dict[str, Any] = {}

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        """Add a Groq service subentry."""
        service_type = self._configured_service_type
        if service_type is not None:
            return await getattr(self, f"async_step_{service_type}")(user_input)
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None):
        """Reconfigure a Groq service subentry."""
        service_type = self._configured_service_type
        if service_type is None:
            service_type = self._existing_service_type()
            self._service_type = service_type
        return await getattr(self, f"async_step_{service_type}")(user_input)

    @property
    def _is_reconfigure(self) -> bool:
        """Return whether the flow is updating an existing subentry."""
        return self.source == SOURCE_RECONFIGURE

    @property
    def _configured_service_type(self) -> str | None:
        """Return the service type selected by the subentry flow handler."""
        if self._service_type is not None:
            return self._service_type
        try:
            subentry_type = self._subentry_type
        except (AttributeError, TypeError):
            return None
        if subentry_type in SETUP_FEATURES:
            # Dedicated integration-page buttons start the flow with a subentry
            # type, so that value is the service type unless the generic
            # service selector was used instead.
            self._service_type = subentry_type
            return subentry_type
        return None

    def _existing_service_data(self) -> dict[str, Any]:
        """Return existing subentry data during reconfiguration."""
        if not self._is_reconfigure:
            return {}
        return dict(self._get_reconfigure_subentry().data)

    def _existing_service_type(self) -> str:
        """Return the existing service type for a reconfigure flow."""
        data = self._existing_service_data()
        service_type = data.get(CONF_SERVICE_TYPE)
        if service_type in SETUP_FEATURES:
            return service_type
        if self._configured_service_type in SETUP_FEATURES:
            return self._configured_service_type
        raise ValueError("Unsupported Groq service subentry type")

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        """Show the initial service type selector."""
        if user_input is not None and CONF_SERVICE_TYPE in user_input:
            self._service_type = user_input[CONF_SERVICE_TYPE]
            return await getattr(self, f"async_step_{self._service_type}")()

        return self.async_show_form(
            step_id="init",
            data_schema=service_type_schema(),
        )

    async def async_step_text_generation(
        self, user_input: dict[str, Any] | None = None
    ):
        """Configure a text generation service."""
        if user_input is not None:
            configure_advanced = bool(user_input.pop(CONF_ADVANCED_OPTIONS, False))
            user_input = clean_service_input(user_input)
            errors = validate_text_generation_input(user_input)
            if errors:
                return self.async_show_form(
                    step_id=FEATURE_TEXT_GENERATION,
                    data_schema=text_generation_basic_schema(user_input),
                    errors=errors,
                )
            if configure_advanced:
                # The advanced step edits the same service object; merge with
                # existing data so reconfigure flows keep hidden secrets and
                # advanced defaults until the user explicitly changes them.
                self._pending_service_data = {
                    **self._existing_service_data(),
                    **user_input,
                }
                return self.async_show_form(
                    step_id="text_generation_advanced",
                    data_schema=text_generation_advanced_schema(
                        self._pending_service_data
                    ),
                )
            if self._is_reconfigure:
                user_input = {
                    **self._existing_service_data(),
                    **user_input,
                }
            return self._create_service_entry(FEATURE_TEXT_GENERATION, user_input)

        return self.async_show_form(
            step_id=FEATURE_TEXT_GENERATION,
            data_schema=text_generation_basic_schema(self._existing_service_data()),
        )

    async def async_step_text_generation_advanced(
        self, user_input: dict[str, Any] | None = None
    ):
        """Configure advanced text generation request options."""
        if user_input is not None:
            service_data = {
                **self._pending_service_data,
                **user_input,
            }
            service_data = clean_service_input(service_data)
            errors = validate_text_generation_input(service_data)
            if errors:
                advanced_errors = {
                    "base" if field == CONF_MODEL else field: reason
                    for field, reason in errors.items()
                }
                return self.async_show_form(
                    step_id="text_generation_advanced",
                    data_schema=text_generation_advanced_schema(service_data),
                    errors=advanced_errors,
                )
            self._pending_service_data = {}
            return self._create_service_entry(FEATURE_TEXT_GENERATION, service_data)

        return self.async_show_form(
            step_id="text_generation_advanced",
            data_schema=text_generation_advanced_schema(self._pending_service_data),
        )

    async def async_step_speech_to_text(self, user_input: dict[str, Any] | None = None):
        """Configure a speech-to-text service."""
        if user_input is not None:
            user_input = clean_service_input(user_input)
            return self._create_service_entry(FEATURE_SPEECH_TO_TEXT, user_input)

        return self.async_show_form(
            step_id=FEATURE_SPEECH_TO_TEXT,
            data_schema=speech_to_text_schema(self._existing_service_data()),
        )

    async def async_step_text_to_speech(self, user_input: dict[str, Any] | None = None):
        """Configure a text-to-speech service."""
        if user_input is not None:
            user_input = clean_service_input(user_input)
            return self._create_service_entry(FEATURE_TEXT_TO_SPEECH, user_input)

        return self.async_show_form(
            step_id=FEATURE_TEXT_TO_SPEECH,
            data_schema=text_to_speech_schema(self._existing_service_data()),
        )

    async def async_step_image_recognition(
        self, user_input: dict[str, Any] | None = None
    ):
        """Configure an image recognition service."""
        if user_input is not None:
            user_input = clean_service_input(user_input)
            return self._create_service_entry(FEATURE_IMAGE_RECOGNITION, user_input)

        return self.async_show_form(
            step_id=FEATURE_IMAGE_RECOGNITION,
            data_schema=image_recognition_schema(self._existing_service_data()),
        )

    @staticmethod
    def _service_data_for_schema(
        existing_data: dict[str, Any],
        new_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Return replacement subentry data while preserving hidden secrets."""
        data = dict(new_data)
        if not new_data.get(CONF_API_KEY) and existing_data.get(CONF_API_KEY):
            data[CONF_API_KEY] = existing_data[CONF_API_KEY]
        return data

    def _create_service_entry(self, service_type: str, user_input: dict[str, Any]):
        """Create a service subentry."""
        data = self._service_data_for_schema(self._existing_service_data(), user_input)
        data[CONF_SERVICE_TYPE] = service_type
        title = data.get(CONF_NAME) or FEATURE_LABELS[service_type]
        if self._is_reconfigure:
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                title=title,
                data=data,
            )
        return self.async_create_entry(title=title, data=data)
