"""AI task support for Groq text generation services."""

from __future__ import annotations

import json
from typing import Any

import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.components.ai_task import (
    AITaskEntity,
    AITaskEntityFeature,
    GenDataTask,
    GenDataTaskResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .api import StructuredGenerationRequest, TextGenerationRequest
from .const import CONF_SUBENTRY_ID, DOMAIN
from .errors import GroqApiError
from .feature_registry import GroqFeature
from .model_registry import GroqModelRegistry
from .runtime import async_get_runtime
from .text_generation import (
    request_body_options_error_message,
    request_context_window_error,
    service_include_reasoning,
    service_max_tokens,
    service_model,
    service_name,
    service_protect_free_tier,
    service_reasoning_effort,
    service_reasoning_format,
    service_request_body_options,
    service_schema,
    service_schema_name,
    service_seed,
    service_service_tier,
    service_stop,
    service_strict,
    service_structured_outputs,
    service_system_prompt,
    service_temperature,
    service_top_p,
    service_unique_id,
    text_generation_service_data,
    voluptuous_schema_to_json_schema,
)

PARALLEL_UPDATES = 1


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Groq AI task entities from text generation services."""
    runtime = await async_get_runtime(hass, config_entry)
    for service_data in text_generation_service_data(config_entry):
        if not runtime.model_registry.supports(
            service_model(config_entry, service_data),
            GroqFeature.STRUCTURED_OUTPUTS,
        ):
            continue
        async_add_entities(
            [
                GroqAITaskEntity(
                    hass,
                    config_entry,
                    service_data,
                    runtime.client,
                    runtime.model_registry,
                )
            ],
            config_subentry_id=service_data.get(CONF_SUBENTRY_ID),
        )


def _strip_json_fence(text: str) -> str:
    """Return text without a Markdown JSON code fence."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _structure_description(schema: vol.Schema) -> str:
    """Return a compact structure description for an AI task prompt."""
    fields: list[str] = []
    schema_data = getattr(schema, "schema", {})
    if isinstance(schema_data, dict):
        for marker, validator in schema_data.items():
            name = getattr(marker, "schema", marker)
            description = getattr(marker, "description", None)
            required = marker.__class__.__name__ == "Required"
            details = f"- {name}"
            if required:
                details += " (required)"
            if description:
                details += f": {description}"
            details += f" [{validator!r}]"
            fields.append(details)
    return "\n".join(fields) if fields else repr(schema_data)


def _can_retry_structured_error(err: GroqApiError) -> bool:
    """Return whether Groq's structured-output failure should fall back to JSON."""
    if err.status != 400:
        return False
    details = str(err).lower()
    if err.payload:
        details = f"{details} {json.dumps(err.payload, sort_keys=True).lower()}"
    return "failed to validate json" in details or "failed_generation" in details


class GroqAITaskEntity(AITaskEntity):
    """Groq AI task entity backed by a text generation service."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_supported_features = AITaskEntityFeature.GENERATE_DATA
    _attr_translation_key = "data_generation_tasks"

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        service_data: dict[str, Any],
        client: Any,
        model_registry: GroqModelRegistry | None = None,
    ) -> None:
        """Initialize the AI task entity."""
        self.hass = hass
        self._config_entry = config_entry
        self._service_data = service_data
        self._client = client
        self._model_registry = model_registry or GroqModelRegistry()
        self._service_name = service_name(config_entry, service_data)
        self._attr_unique_id = (
            f"{service_unique_id(config_entry, service_data)}_ai_task"
        )

    @property
    def device_info(self) -> dict:
        """Return device information."""
        unique_id = service_unique_id(self._config_entry, self._service_data)
        return {
            "identifiers": {(DOMAIN, unique_id)},
            "manufacturer": "Groq",
            "model": service_model(self._config_entry, self._service_data),
            "name": self._service_name,
        }

    def _text_generation_request(self, instructions: str) -> TextGenerationRequest:
        """Build a text generation request from the configured service."""
        return TextGenerationRequest(
            prompt=instructions,
            model=service_model(self._config_entry, self._service_data),
            system_prompt=service_system_prompt(self._config_entry, self._service_data),
            temperature=service_temperature(self._config_entry, self._service_data),
            max_tokens=service_max_tokens(
                self._config_entry,
                self._service_data,
                self._model_registry,
            ),
            top_p=service_top_p(self._config_entry, self._service_data),
            stop=service_stop(self._config_entry, self._service_data),
            seed=service_seed(self._config_entry, self._service_data),
            service_tier=service_service_tier(self._config_entry, self._service_data),
            reasoning_effort=service_reasoning_effort(
                self._config_entry, self._service_data
            ),
            reasoning_format=service_reasoning_format(
                self._config_entry, self._service_data
            ),
            include_reasoning=service_include_reasoning(
                self._config_entry, self._service_data
            ),
            extra_body=service_request_body_options(
                self._config_entry,
                self._service_data,
                self._model_registry,
            ),
            service_id=service_unique_id(self._config_entry, self._service_data),
            protect_free_tier=service_protect_free_tier(
                self._config_entry, self._service_data
            ),
        )

    def _structured_generation_request(
        self,
        instructions: str,
        schema: dict[str, Any],
        schema_name: str,
        *,
        strict: bool,
    ) -> StructuredGenerationRequest:
        """Build a structured generation request from the configured service."""
        text_request = self._text_generation_request(instructions)
        return StructuredGenerationRequest(
            prompt=text_request.prompt,
            model=text_request.model,
            system_prompt=text_request.system_prompt,
            temperature=text_request.temperature,
            max_tokens=text_request.max_tokens,
            top_p=text_request.top_p,
            stop=text_request.stop,
            seed=text_request.seed,
            service_tier=text_request.service_tier,
            reasoning_effort=text_request.reasoning_effort,
            reasoning_format=text_request.reasoning_format,
            include_reasoning=text_request.include_reasoning,
            extra_body=text_request.extra_body,
            service_id=text_request.service_id,
            protect_free_tier=text_request.protect_free_tier,
            schema=schema,
            schema_name=schema_name,
            strict=strict,
        )

    async def _async_generate_json_fallback(
        self,
        task: GenDataTask,
        chat_log: conversation.ChatLog,
        instructions: str,
    ) -> GenDataTaskResult:
        """Generate and validate JSON without Groq json_schema mode."""
        if task.structure is not None:
            instructions = (
                f"{instructions}\n\n"
                "Return only a valid JSON object matching this output structure. "
                "Do not include Markdown, explanations, or extra keys.\n"
                f"{_structure_description(task.structure)}"
            )
        request = self._text_generation_request(instructions)
        if error := request_body_options_error_message(
            self._model_registry,
            request.model,
            request.extra_body,
        ):
            raise HomeAssistantError(error)
        if error := request_context_window_error(self._model_registry, request):
            raise HomeAssistantError(error)
        result = await self._client.async_generate_text(request)
        data: Any = result.text
        if task.structure is not None:
            try:
                data = json.loads(_strip_json_fence(result.text))
                data = task.structure(data)
            except (json.JSONDecodeError, vol.Invalid) as err:
                raise HomeAssistantError(
                    "Groq returned data that did not match the requested structure"
                ) from err
        return GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )

    async def _async_generate_data(
        self,
        task: GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> GenDataTaskResult:
        """Generate data for a Home Assistant AI task."""
        instructions = task.instructions
        schema = None
        schema_name = service_schema_name(
            self._config_entry,
            self._service_data,
            task.name,
        )
        if task.structure is not None:
            schema = voluptuous_schema_to_json_schema(task.structure)
        elif service_structured_outputs(self._config_entry, self._service_data):
            schema = service_schema(self._config_entry, self._service_data)

        model = service_model(self._config_entry, self._service_data)
        supports_structured_outputs = self._model_registry.supports(
            model, GroqFeature.STRUCTURED_OUTPUTS
        )
        if schema and supports_structured_outputs:
            # Prefer Groq structured outputs whenever Home Assistant supplies a
            # task structure, otherwise use the service-level schema if enabled.
            request = self._structured_generation_request(
                instructions,
                schema,
                schema_name,
                strict=(
                    True
                    if task.structure is not None
                    else service_strict(self._config_entry, self._service_data)
                ),
            )
            if error := request_body_options_error_message(
                self._model_registry,
                request.model,
                request.extra_body,
            ):
                raise HomeAssistantError(error)
            if error := request_context_window_error(self._model_registry, request):
                raise HomeAssistantError(error)
            try:
                response = await self._client.async_generate_structured(request)
            except GroqApiError as err:
                if task.structure is None or not _can_retry_structured_error(err):
                    raise
                return await self._async_generate_json_fallback(
                    task, chat_log, instructions
                )
            data: Any = response["data"]
            if task.structure is not None:
                try:
                    data = task.structure(data)
                except vol.Invalid as err:
                    raise HomeAssistantError(
                        "Groq returned data that did not match the requested structure"
                    ) from err
        else:
            return await self._async_generate_json_fallback(
                task, chat_log, instructions
            )

        return GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )
