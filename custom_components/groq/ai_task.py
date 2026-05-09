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
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .api import StructuredGenerationRequest, TextGenerationRequest
from .const import DOMAIN
from .runtime import async_get_runtime
from .text_generation import (
    service_api_key,
    service_include_reasoning,
    service_max_tokens,
    service_model,
    service_name,
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


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Groq AI task entities from text generation services."""
    runtime = await async_get_runtime(hass, config_entry)
    entities = [
        GroqAITaskEntity(hass, config_entry, service_data, runtime.client)
        for service_data in text_generation_service_data(config_entry)
    ]
    if entities:
        async_add_entities(entities)


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


class GroqAITaskEntity(AITaskEntity):
    """Groq AI task entity backed by a text generation service."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_supported_features = AITaskEntityFeature.GENERATE_DATA

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        service_data: dict,
        client,
    ) -> None:
        """Initialize the AI task entity."""
        self.hass = hass
        self._config_entry = config_entry
        self._service_data = service_data
        self._client = client
        self._attr_name = service_name(config_entry, service_data)
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
            "name": self._attr_name,
        }

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

        if schema:
            # Prefer Groq structured outputs whenever Home Assistant supplies a
            # task structure, otherwise use the service-level schema if enabled.
            request = StructuredGenerationRequest(
                prompt=instructions,
                model=service_model(self._config_entry, self._service_data),
                system_prompt=service_system_prompt(
                    self._config_entry, self._service_data
                ),
                temperature=service_temperature(self._config_entry, self._service_data),
                max_tokens=service_max_tokens(self._config_entry, self._service_data),
                top_p=service_top_p(self._config_entry, self._service_data),
                stop=service_stop(self._config_entry, self._service_data),
                seed=service_seed(self._config_entry, self._service_data),
                service_tier=service_service_tier(
                    self._config_entry, self._service_data
                ),
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
                    self._config_entry, self._service_data
                ),
                api_key=service_api_key(self._config_entry, self._service_data),
                schema=schema,
                schema_name=schema_name,
                strict=service_strict(self._config_entry, self._service_data),
            )
            response = await self._client.async_generate_structured(request)
            data: Any = response["data"]
            if task.structure is not None:
                try:
                    data = task.structure(data)
                except vol.Invalid as err:
                    raise HomeAssistantError(
                        "Groq returned data that did not match the requested structure"
                    ) from err
        else:
            if task.structure is not None:
                instructions = (
                    f"{instructions}\n\n"
                    "Return only a valid JSON object matching this output structure. "
                    "Do not include Markdown, explanations, or extra keys.\n"
                    f"{_structure_description(task.structure)}"
                )
            request = TextGenerationRequest(
                prompt=instructions,
                model=service_model(self._config_entry, self._service_data),
                system_prompt=service_system_prompt(
                    self._config_entry, self._service_data
                ),
                temperature=service_temperature(self._config_entry, self._service_data),
                max_tokens=service_max_tokens(self._config_entry, self._service_data),
                top_p=service_top_p(self._config_entry, self._service_data),
                stop=service_stop(self._config_entry, self._service_data),
                seed=service_seed(self._config_entry, self._service_data),
                service_tier=service_service_tier(
                    self._config_entry, self._service_data
                ),
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
                    self._config_entry, self._service_data
                ),
                api_key=service_api_key(self._config_entry, self._service_data),
            )
            result = await self._client.async_generate_text(request)
            data = result.text

            if task.structure is not None:
                # Fallback path for models/services not using structured outputs:
                # prompt for JSON, strip common Markdown fences, then validate
                # against Home Assistant's requested voluptuous structure.
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
