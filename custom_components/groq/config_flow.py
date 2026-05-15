"""
Config flow for Groq.
"""

from __future__ import annotations
from typing import Any
import aiohttp
import voluptuous as vol
import logging
import uuid

from homeassistant import data_entry_flow
from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    ConfigFlow,
    ConfigSubentryFlow,
    OptionsFlow,
)
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.core import callback

from .api import GroqApiClient
from .const import (
    CONF_ADVANCED_OPTIONS,
    CONF_API_KEY,
    CONF_ENABLED_FEATURES,
    CONF_MODEL,
    CONF_NAME,
    CONF_SERVICE_TYPE,
    CONF_VOICE,
    DEFAULT_MODEL,
    DOMAIN,
    FEATURE_IMAGE_RECOGNITION,
    FEATURE_LABELS,
    FEATURE_SPEECH_TO_TEXT,
    FEATURE_TEXT_GENERATION,
    FEATURE_TEXT_TO_SPEECH,
    MODELS,
    SETUP_FEATURES,
    STT_MODELS,
    TEXT_MODELS,
    UNIQUE_ID,
    VISION_MODELS,
    VOICES,
    stt_language_default,
    voice_options_for_model,
)
from .feature_registry import GroqFeature
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
    text_generation_model_capability_summary,
    text_to_speech_schema,
    sanitize_text_generation_service_data,
    validate_text_generation_input,
    validate_user_input,
)
from .model_registry import (
    GroqModel,
    GroqModelRegistry,
)
from .errors import GroqApiError, GroqResponseError

_LOGGER = logging.getLogger(__name__)
API_KEY_VALIDATION_TIMEOUT = aiohttp.ClientTimeout(total=10)


def generate_entry_id() -> str:
    return str(uuid.uuid4())


def _new_account_unique_id() -> str:
    """Return a new stable Groq account unique id."""
    return f"groq_{generate_entry_id()}"


def _entry_unique_id(entry) -> str:
    """Return an existing entry unique id, falling back to a new account id."""
    data = getattr(entry, "data", {}) or {}
    return (
        getattr(entry, "unique_id", None)
        or data.get(UNIQUE_ID)
        or _new_account_unique_id()
    )


def _api_key_validation_errors(validation_error: str | None) -> dict[str, str]:
    """Return config-flow errors for a Groq API key validation result."""
    if validation_error == "invalid_auth":
        return {CONF_API_KEY: validation_error}
    if validation_error is not None:
        return {"base": validation_error}
    return {}


def _api_key_duplicate_error(
    hass,
    api_key: str,
    *,
    current_entry_id: str | None = None,
) -> str | None:
    """Return an error when another Groq entry already uses an API key."""
    config_entries = getattr(hass, "config_entries", None)
    async_entries = getattr(config_entries, "async_entries", None)
    if async_entries is None:
        return None
    for entry in async_entries(DOMAIN):
        if getattr(entry, "entry_id", None) == current_entry_id:
            continue
        data = getattr(entry, "data", {}) or {}
        options = getattr(entry, "options", {}) or {}
        if api_key in (data.get(CONF_API_KEY), options.get(CONF_API_KEY)):
            return "duplicate_api_key"
    return None


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


async def async_fetch_available_models(hass, api_key: str) -> list[GroqModel]:
    """Fetch active models visible to a Groq API key."""
    client = GroqApiClient(
        hass,
        api_key=api_key,
        session=async_get_clientsession(hass),
        request_timeout=API_KEY_VALIDATION_TIMEOUT,
    )
    try:
        return await client.async_list_models(hydrate=False)
    except ConfigEntryAuthFailed as err:
        raise ValueError("invalid_auth") from err
    except GroqResponseError as err:
        raise RuntimeError(str(err)) from err


async def async_get_model_registry(
    hass,
    api_key: str | None,
) -> GroqModelRegistry:
    """Return a model registry discovered from Groq, with built-ins as fallback."""
    if not api_key:
        return GroqModelRegistry()
    try:
        models = await async_fetch_available_models(hass, api_key)
    except ValueError:
        raise
    except Exception as err:  # pylint: disable=broad-except
        _LOGGER.debug("Could not fetch Groq model list: %s", err)
        return GroqModelRegistry()
    if not models:
        return GroqModelRegistry()
    return GroqModelRegistry(models, include_built_ins=False)


def _model_ids_for_feature(
    registry: GroqModelRegistry,
    feature: GroqFeature,
    fallback: list[str],
) -> list[str]:
    """Return model ids for a feature, falling back only when discovery is empty."""
    model_ids = [model.model_id for model in registry.models_for_feature(feature)]
    return model_ids or fallback


async def async_validate_api_key(hass, api_key: str) -> str | None:
    """Validate a Groq API key against a lightweight authenticated endpoint."""
    try:
        await async_fetch_available_models(hass, api_key)
    except ValueError:
        return "invalid_auth"
    except (aiohttp.ContentTypeError, TypeError) as err:
        _LOGGER.debug("Groq API key validation returned invalid payload: %s", err)
        return "unknown"
    except GroqApiError as err:
        if str(err).startswith("Network error"):
            _LOGGER.debug("Could not connect to Groq while validating API key: %s", err)
            return "cannot_connect"
        _LOGGER.debug("Groq API key validation failed: %s", err)
        return "unknown"
    except (aiohttp.ClientError, TimeoutError) as err:
        _LOGGER.debug("Could not connect to Groq while validating API key: %s", err)
        return "cannot_connect"
    except Exception as err:  # pylint: disable=broad-except
        _LOGGER.exception("Unexpected error validating Groq API key: %s", err)
        return "unknown"
    return None


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
    registry = await async_get_model_registry(hass, api_key)
    models = _model_ids_for_feature(registry, GroqFeature.TEXT_TO_SPEECH, MODELS)
    return models, VOICES


class GroqConfigFlow(ConfigFlow, domain=DOMAIN):  # type: ignore[call-arg]
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
                unique_id = _new_account_unique_id()
                await self.async_set_unique_id(unique_id)
                # Allow multiple named Groq accounts, but do not create a
                # duplicate account for the same API key.
                self._abort_if_unique_id_configured()
                validation_error = await async_validate_api_key(
                    self.hass,
                    entry_data[CONF_API_KEY],
                )
                errors.update(_api_key_validation_errors(validation_error))
                if not errors and (
                    duplicate_error := _api_key_duplicate_error(
                        self.hass,
                        entry_data[CONF_API_KEY],
                    )
                ):
                    errors["base"] = duplicate_error
                if not errors:
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

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> data_entry_flow.FlowResult:
        """Reconfigure a Groq account entry."""
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}
        if user_input is not None:
            new_name = str(
                user_input.get(CONF_NAME) or getattr(entry, "title", None) or DOMAIN
            )
            api_key = user_input.get(CONF_API_KEY)
            new_data = dict(entry.data)
            new_options = dict(entry.options)
            unique_id = _entry_unique_id(entry)

            if api_key:
                validation_error = await async_validate_api_key(self.hass, api_key)
                errors.update(_api_key_validation_errors(validation_error))
                if not errors and (
                    duplicate_error := _api_key_duplicate_error(
                        self.hass,
                        api_key,
                        current_entry_id=entry.entry_id,
                    )
                ):
                    errors["base"] = duplicate_error
                if not errors:
                    new_data[CONF_API_KEY] = api_key
                    new_options.pop(CONF_API_KEY, None)

            if not errors:
                new_data[CONF_NAME] = new_name
                return self.async_update_reload_and_abort(
                    entry,
                    unique_id=unique_id,
                    title=new_name,
                    data=new_data,
                    options=new_options,
                    reason="reconfigure_successful",
                )

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_NAME,
                    default=entry.data.get(
                        CONF_NAME,
                        getattr(entry, "title", None) or "Groq",
                    ),
                ): str,
                vol.Optional(CONF_API_KEY): api_key_selector(),
            }
        )
        return self.async_show_form(
            step_id="reconfigure",
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
                validation_error = await async_validate_api_key(self.hass, api_key)
                errors.update(_api_key_validation_errors(validation_error))
                if not errors and (
                    duplicate_error := _api_key_duplicate_error(
                        self.hass,
                        api_key,
                        current_entry_id=reauth_entry.entry_id,
                    )
                ):
                    errors["base"] = duplicate_error
                if not errors:
                    new_data = dict(reauth_entry.data)
                    new_data[CONF_API_KEY] = api_key
                    new_options = dict(reauth_entry.options)
                    new_options.pop(CONF_API_KEY, None)
                    unique_id = _entry_unique_id(reauth_entry)
                    # Abort current flow, update & reload the entry with new credentials
                    return self.async_update_reload_and_abort(
                        reauth_entry,
                        unique_id=unique_id,
                        data=new_data,
                        options=new_options,
                        reason="reauth_successful",
                    )

        schema = vol.Schema({vol.Required(CONF_API_KEY): api_key_selector()})
        return self.async_show_form(
            step_id="reauth_confirm", data_schema=schema, errors=errors
        )


class GroqOptionsFlow(OptionsFlow):
    """Handle options flow for Groq."""

    def _current_entry(self):
        """Return the config entry being edited by this options flow."""
        try:
            entry = getattr(self, "config_entry", None)
        except ValueError:
            entry = None
        if entry is not None:
            return entry

        hass = getattr(self, "hass", None)
        config_entries = getattr(hass, "config_entries", None)
        entry_id = getattr(self, "handler", None)
        if isinstance(entry_id, tuple):
            entry_id = entry_id[0] if entry_id else None
        for getter_name in ("async_get_entry", "async_get_known_entry"):
            getter = getattr(config_entries, getter_name, None)
            if getter is None or entry_id is None:
                continue
            entry = getter(entry_id)
            if entry is not None:
                return entry
        return None

    async def async_step_init(self, user_input: dict | None = None):
        errors: dict[str, str] = {}
        if user_input is not None:
            user_input = dict(user_input)
            if not user_input.get(CONF_API_KEY):
                user_input.pop(CONF_API_KEY, None)
            else:
                validation_error = await async_validate_api_key(
                    self.hass,
                    user_input[CONF_API_KEY],
                )
                errors.update(_api_key_validation_errors(validation_error))
                current_entry = self._current_entry()
                current_entry_id = getattr(current_entry, "entry_id", None)
                if not errors and (
                    duplicate_error := _api_key_duplicate_error(
                        self.hass,
                        user_input[CONF_API_KEY],
                        current_entry_id=current_entry_id,
                    )
                ):
                    errors["base"] = duplicate_error
            if not errors:
                current_entry = self._current_entry()
                if current_entry is not None and user_input.get(CONF_API_KEY):
                    new_data = dict(current_entry.data)
                    new_data[CONF_API_KEY] = user_input[CONF_API_KEY]
                    new_options = dict(current_entry.options)
                    new_options.pop(CONF_API_KEY, None)
                    self.hass.config_entries.async_update_entry(
                        current_entry,
                        data=new_data,
                        options=new_options,
                        unique_id=_entry_unique_id(current_entry),
                    )
                    user_input = new_options
                return self.async_create_entry(title="", data=user_input)
        options_schema = vol.Schema(
            {
                vol.Optional(
                    CONF_API_KEY,
                ): api_key_selector(),
            }
        )
        return self.async_show_form(
            step_id="init",
            data_schema=options_schema,
            errors=errors,
        )


class GroqServiceSubentryFlow(ConfigSubentryFlow):
    """Handle Groq service instance subentries."""

    def __init__(self) -> None:
        """Initialize the service subentry flow."""
        self._service_type: str | None = None
        self._pending_service_data: dict[str, Any] = {}
        self._tts_model_context: str | None = None
        self._model_registry_cache: dict[str, GroqModelRegistry] = {}

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

    def _account_api_key(self) -> str | None:
        """Return the account API key for model discovery."""
        try:
            entry = self._get_entry()
        except (AttributeError, TypeError):
            return None
        data = getattr(entry, "data", {}) or {}
        options = getattr(entry, "options", {}) or {}
        return options.get(CONF_API_KEY) or data.get(CONF_API_KEY)

    async def _model_registry(
        self,
        service_data: dict[str, Any] | None = None,
    ) -> GroqModelRegistry:
        """Return discovered model metadata for the active account key."""
        api_key = self._account_api_key()
        cache_key = api_key or ""
        if cache_key in self._model_registry_cache:
            return self._model_registry_cache[cache_key]
        try:
            registry = await async_get_model_registry(self.hass, api_key)
        except ValueError:
            registry = GroqModelRegistry()
        self._model_registry_cache[cache_key] = registry
        return registry

    async def _model_options(
        self,
        feature: GroqFeature,
        fallback: list[str],
        service_data: dict[str, Any] | None = None,
    ) -> tuple[list[str], GroqModelRegistry]:
        """Return valid model ids and the registry used to derive them."""
        registry = await self._model_registry(service_data)
        return _model_ids_for_feature(registry, feature, fallback), registry

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
        model_options, model_registry = await self._model_options(
            GroqFeature.TEXT_GENERATION,
            TEXT_MODELS,
            user_input,
        )
        if user_input is not None:
            configure_advanced = bool(user_input.pop(CONF_ADVANCED_OPTIONS, False))
            user_input = clean_service_input(user_input)
            errors = validate_text_generation_input(user_input, model_registry)
            if errors:
                return self.async_show_form(
                    step_id=FEATURE_TEXT_GENERATION,
                    data_schema=text_generation_basic_schema(
                        user_input,
                        model_options,
                        model_registry,
                    ),
                    errors=errors,
                    description_placeholders={
                        "model_capabilities": text_generation_model_capability_summary(
                            user_input.get(CONF_MODEL, ""),
                            model_registry,
                        )
                    },
                )
            if configure_advanced:
                # The advanced step edits the same service object; merge with
                # existing data so reconfigure flows keep hidden secrets and
                # advanced defaults until the user explicitly changes them.
                self._pending_service_data = {
                    **self._existing_service_data(),
                    **user_input,
                }
                self._pending_service_data = sanitize_text_generation_service_data(
                    self._pending_service_data,
                    model_registry,
                )
                return self.async_show_form(
                    step_id="text_generation_advanced",
                    data_schema=text_generation_advanced_schema(
                        self._pending_service_data,
                        model_registry,
                    ),
                )
            if self._is_reconfigure:
                user_input = {
                    **self._existing_service_data(),
                    **user_input,
                }
                user_input = sanitize_text_generation_service_data(
                    user_input,
                    model_registry,
                )
                errors = validate_text_generation_input(user_input, model_registry)
                if errors:
                    return self.async_show_form(
                        step_id=FEATURE_TEXT_GENERATION,
                        data_schema=text_generation_basic_schema(
                            user_input,
                            model_options,
                            model_registry,
                        ),
                        errors={
                            "base" if field != CONF_NAME else field: reason
                            for field, reason in errors.items()
                        },
                    )
            return self._create_service_entry(FEATURE_TEXT_GENERATION, user_input)

        return self.async_show_form(
            step_id=FEATURE_TEXT_GENERATION,
            data_schema=text_generation_basic_schema(
                self._existing_service_data(),
                model_options,
                model_registry,
            ),
            description_placeholders={
                "model_capabilities": text_generation_model_capability_summary(
                    (
                        self._existing_service_data().get(CONF_MODEL)
                        or model_options[0]
                        if model_options
                        else ""
                    ),
                    model_registry,
                )
            },
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
            model_registry = await self._model_registry(service_data)
            errors = validate_text_generation_input(service_data, model_registry)
            if errors:
                advanced_errors = {
                    "base" if field == CONF_MODEL else field: reason
                    for field, reason in errors.items()
                }
                return self.async_show_form(
                    step_id="text_generation_advanced",
                    data_schema=text_generation_advanced_schema(
                        service_data,
                        model_registry,
                    ),
                    errors=advanced_errors,
                )
            self._pending_service_data = {}
            return self._create_service_entry(FEATURE_TEXT_GENERATION, service_data)

        model_registry = await self._model_registry(self._pending_service_data)
        return self.async_show_form(
            step_id="text_generation_advanced",
            data_schema=text_generation_advanced_schema(
                self._pending_service_data,
                model_registry,
            ),
        )

    async def async_step_speech_to_text(self, user_input: dict[str, Any] | None = None):
        """Configure a speech-to-text service."""
        model_options, _model_registry = await self._model_options(
            GroqFeature.SPEECH_TO_TEXT,
            STT_MODELS,
            user_input,
        )
        if user_input is not None:
            user_input = clean_service_input(user_input)
            return self._create_service_entry(FEATURE_SPEECH_TO_TEXT, user_input)

        hass_language = getattr(getattr(self.hass, "config", None), "language", None)
        return self.async_show_form(
            step_id=FEATURE_SPEECH_TO_TEXT,
            data_schema=speech_to_text_schema(
                self._existing_service_data(),
                model_options,
                stt_language_default(hass_language),
            ),
        )

    async def async_step_text_to_speech(self, user_input: dict[str, Any] | None = None):
        """Configure a text-to-speech service."""
        model_options, _model_registry = await self._model_options(
            GroqFeature.TEXT_TO_SPEECH,
            MODELS,
            user_input,
        )
        existing_data = self._existing_service_data()
        baseline_model = (
            self._tts_model_context or existing_data.get(CONF_MODEL) or DEFAULT_MODEL
        )
        schema_data = user_input or existing_data
        selected_model = schema_data.get(CONF_MODEL) or baseline_model
        voice_options = voice_options_for_model(selected_model)
        if user_input is not None:
            user_input = clean_service_input(user_input)
            selected_model = user_input.get(CONF_MODEL) or baseline_model
            voice_options = voice_options_for_model(selected_model)
            if selected_model != baseline_model and user_input.get(CONF_VOICE):
                self._tts_model_context = selected_model
                user_input.pop(CONF_VOICE, None)
                return self.async_show_form(
                    step_id=FEATURE_TEXT_TO_SPEECH,
                    data_schema=text_to_speech_schema(
                        user_input,
                        model_options,
                        voice_options,
                        clear_voice=True,
                    ),
                    errors={CONF_VOICE: "select_voice_for_model"},
                )
            if user_input.get(CONF_VOICE) not in voice_options:
                self._tts_model_context = selected_model
                user_input.pop(CONF_VOICE, None)
                return self.async_show_form(
                    step_id=FEATURE_TEXT_TO_SPEECH,
                    data_schema=text_to_speech_schema(
                        user_input,
                        model_options,
                        voice_options,
                        clear_voice=True,
                    ),
                    errors={CONF_VOICE: "invalid_voice"},
                )
            self._tts_model_context = None
            return self._create_service_entry(FEATURE_TEXT_TO_SPEECH, user_input)

        return self.async_show_form(
            step_id=FEATURE_TEXT_TO_SPEECH,
            data_schema=text_to_speech_schema(
                existing_data,
                model_options,
                voice_options,
            ),
        )

    async def async_step_image_recognition(
        self, user_input: dict[str, Any] | None = None
    ):
        """Configure an image recognition service."""
        model_options, _model_registry = await self._model_options(
            GroqFeature.VISION,
            VISION_MODELS,
            user_input,
        )
        if user_input is not None:
            user_input = clean_service_input(user_input)
            return self._create_service_entry(FEATURE_IMAGE_RECOGNITION, user_input)

        return self.async_show_form(
            step_id=FEATURE_IMAGE_RECOGNITION,
            data_schema=image_recognition_schema(
                self._existing_service_data(),
                model_options,
            ),
        )

    @staticmethod
    def _service_data_for_schema(
        existing_data: dict[str, Any],
        new_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Return replacement subentry data."""
        return dict(new_data)

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
