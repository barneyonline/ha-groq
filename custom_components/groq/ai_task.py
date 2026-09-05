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
from homeassistant.components.conversation import AssistantContent
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .api import GroqApiClient, StructuredGenerationRequest, TextGenerationRequest
from .chat import (
    MAX_TOOL_ITERATIONS,
    _assistant_native,
    _async_chat_log_messages,
    _chat_log_tools,
    _result_tool_calls,
)
from .const import CONF_SUBENTRY_ID
from .entity import service_device_info
from .errors import GroqApiError, translated_error
from .feature_registry import GroqFeature
from .model_registry import GroqCapability, GroqModelRegistry
from .runtime import async_get_runtime
from .structured import validate_json_schema_data
from .text_generation import (
    compound_builtin_tools_error_message,
    request_body_options_error_message,
    request_context_window_error,
    service_generation_options,
    service_model,
    service_name,
    service_protect_free_tier,
    service_schema,
    service_schema_name,
    service_strict,
    service_structured_outputs,
    service_system_prompt,
    service_unique_id,
    structured_generation_request,
    text_generation_service_data,
    voluptuous_schema_to_json_schema,
)
from .types import GroqConfigEntry

PARALLEL_UPDATES = 1
SUPPORT_ATTACHMENTS = getattr(
    AITaskEntityFeature, "SUPPORT_ATTACHMENTS", AITaskEntityFeature(0)
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: GroqConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Groq AI task entities from text generation services."""
    runtime = await async_get_runtime(hass, config_entry)
    for service_data in text_generation_service_data(config_entry):
        if not runtime.model_registry.supports(
            service_model(config_entry, service_data),
            GroqFeature.TEXT_GENERATION,
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
    return any(
        code in details
        for code in (
            "failed to validate json",
            "failed_generation",
            "json_validate_failed",
        )
    )


def _json_fallback_instructions(
    instructions: str,
    task: GenDataTask,
    schema: dict[str, Any] | None = None,
) -> str:
    """Return instructions that ask for JSON when structured mode is unavailable."""
    if output_instruction := _json_output_instruction(task, schema):
        return f"{instructions}\n\n{output_instruction}"
    return instructions


def _json_output_instruction(
    task: GenDataTask,
    schema: dict[str, Any] | None = None,
) -> str | None:
    """Return JSON-only output instructions for structured AI tasks."""
    if task.structure is not None:
        schema_description = _structure_description(task.structure)
    elif schema is not None:
        schema_description = json.dumps(schema, separators=(",", ":"), sort_keys=True)
    else:
        return None
    return (
        "Return only a valid JSON object matching this output structure. "
        "Do not include Markdown, explanations, or extra keys.\n"
        f"{schema_description}"
    )


class GroqAITaskEntity(AITaskEntity):
    """Groq AI task entity backed by a text generation service."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_supported_features = AITaskEntityFeature.GENERATE_DATA
    _attr_translation_key = "data_generation_tasks"

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: GroqConfigEntry,
        service_data: dict[str, Any],
        client: GroqApiClient,
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
        self._attr_supported_features = AITaskEntityFeature.GENERATE_DATA
        if self._model_registry.supports(
            service_model(config_entry, service_data),
            GroqFeature.VISION,
        ):
            self._attr_supported_features |= SUPPORT_ATTACHMENTS

    @property
    def device_info(self) -> DeviceInfo:
        """Return device information."""
        unique_id = service_unique_id(self._config_entry, self._service_data)
        return service_device_info(
            unique_id,
            service_model(self._config_entry, self._service_data),
            self._service_name,
        )

    def _text_generation_request(
        self,
        instructions: str,
        messages: list[dict[str, Any]] | None = None,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> TextGenerationRequest:
        """Build a text generation request from the configured service."""
        return TextGenerationRequest(
            prompt=instructions,
            model=service_model(self._config_entry, self._service_data),
            messages=messages,
            system_prompt=service_system_prompt(self._config_entry, self._service_data),
            tools=tools,
            tool_choice=tool_choice,
            **service_generation_options(
                self._config_entry, self._service_data, self._model_registry
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
        messages: list[dict[str, Any]] | None = None,
        strict: bool,
    ) -> StructuredGenerationRequest:
        """Build a structured generation request from the configured service."""
        text_request = self._text_generation_request(instructions, messages)
        return structured_generation_request(
            text_request, schema, schema_name, strict=strict
        )

    async def _async_task_messages(
        self,
        task: GenDataTask,
        instructions: str,
        chat_log: conversation.ChatLog,
    ) -> list[dict[str, Any]]:
        """Use HA's prepared prompt and history for every AI task request."""
        messages = await _async_chat_log_messages(
            self.hass,
            self._model_registry,
            service_model(self._config_entry, self._service_data),
            chat_log,
            task.instructions,
            getattr(task, "attachments", None),
        )
        message = next(
            message for message in reversed(messages) if message["role"] == "user"
        )
        if isinstance(message["content"], list):
            message["content"] = [dict(part) for part in message["content"]]
            message["content"][0] = {"type": "text", "text": instructions}
        else:
            message["content"] = instructions
        return messages

    def _record_result(self, chat_log: conversation.ChatLog, text: str) -> None:
        """Record one completed no-tool result for HA traces and context."""
        chat_log.async_add_assistant_content_without_tools(
            AssistantContent(
                agent_id=service_unique_id(self._config_entry, self._service_data),
                content=text,
            )
        )

    async def _validate_result(
        self, data: Any, task: GenDataTask, schema: dict[str, Any] | None
    ) -> Any:
        """Apply task or service structure consistently across generation paths."""
        if task.structure is not None:
            try:
                return task.structure(data)
            except vol.Invalid as err:
                raise translated_error(
                    "Groq returned data that did not match the requested structure",
                    "structured_response_invalid",
                ) from err
        if schema is not None:
            return await self.hass.async_add_executor_job(
                validate_json_schema_data, data, schema
            )
        return data

    def _raise_request_errors(self, request: TextGenerationRequest) -> None:
        """Raise Home Assistant errors for invalid Groq request options."""
        if error := compound_builtin_tools_error_message(
            self._model_registry,
            request.model,
            request.compound_builtin_tools,
        ):
            raise translated_error(error, "invalid_request_options")
        if error := request_body_options_error_message(
            self._model_registry,
            request.model,
            request.extra_body,
        ):
            raise translated_error(error, "invalid_request_options")
        if error := request_context_window_error(self._model_registry, request):
            raise translated_error(error, "invalid_request_options")

    async def _async_tool_generation_request(
        self,
        task: GenDataTask,
        chat_log: conversation.ChatLog,
        instructions: str,
        tools: list[dict[str, Any]],
        output_instruction: str | None,
        attachment_cache: dict[tuple[str, str], Any] | None = None,
    ) -> TextGenerationRequest:
        """Build an AI task request that includes Home Assistant tool state."""
        model = service_model(self._config_entry, self._service_data)
        messages = await _async_chat_log_messages(
            self.hass,
            self._model_registry,
            model,
            chat_log,
            instructions,
            getattr(task, "attachments", None),
            attachment_cache,
        )
        if output_instruction:
            messages = [dict(message) for message in messages]
            for message in messages:
                if message["role"] == "system":
                    message["content"] = (
                        f"{message.get('content', '')}\n\n{output_instruction}"
                    )
                    break
            else:
                messages.insert(0, {"role": "system", "content": output_instruction})
        return self._text_generation_request(
            instructions,
            messages,
            tools=tools,
            tool_choice="auto",
        )

    async def _async_generate_text_with_tools(
        self,
        task: GenDataTask,
        chat_log: conversation.ChatLog,
        instructions: str,
        tools: list[dict[str, Any]],
        output_instruction: str | None = None,
    ) -> Any:
        """Generate AI task text while executing Home Assistant LLM tools."""
        model = service_model(self._config_entry, self._service_data)
        if not self._model_registry.supports(model, GroqCapability.TOOL_CALLING):
            raise translated_error(
                f"Groq model {model} is not known to support Home Assistant tool calls",
                "unsupported_model",
                model=model,
                feature="Home Assistant tool calls",
            )

        attachment_cache: dict[tuple[str, str], Any] = {}
        for _iteration in range(MAX_TOOL_ITERATIONS):
            request = await self._async_tool_generation_request(
                task,
                chat_log,
                instructions,
                tools,
                output_instruction,
                attachment_cache,
            )
            self._raise_request_errors(request)
            result = await self._client.async_generate_text(request)
            tool_calls = _result_tool_calls(result)
            assistant_content = AssistantContent(
                agent_id=service_unique_id(self._config_entry, self._service_data),
                content=result.text or None,
                thinking_content=getattr(result, "reasoning", None),
                tool_calls=tool_calls or None,
                native=_assistant_native(result) or None,
            )
            if tool_calls:
                async for _content in chat_log.async_add_assistant_content(
                    assistant_content
                ):
                    pass
            else:
                chat_log.async_add_assistant_content_without_tools(assistant_content)
            if not getattr(chat_log, "unresponded_tool_results", False):
                return result
        raise translated_error(
            "Groq AI task exceeded the tool-call limit", "tool_call_limit"
        )

    async def _async_generate_json_fallback(
        self,
        task: GenDataTask,
        chat_log: conversation.ChatLog,
        instructions: str,
        schema: dict[str, Any] | None = None,
    ) -> GenDataTaskResult:
        """Generate and validate JSON without Groq json_schema mode."""
        instructions = _json_fallback_instructions(instructions, task, schema)
        request = self._text_generation_request(
            instructions,
            await self._async_task_messages(task, instructions, chat_log),
        )
        self._raise_request_errors(request)
        result = await self._client.async_generate_text(request)
        data: Any = result.text
        if task.structure is not None or schema is not None:
            try:
                data = json.loads(_strip_json_fence(result.text))
            except json.JSONDecodeError as err:
                raise translated_error(
                    "Groq returned data that did not match the requested structure",
                    "structured_response_invalid",
                ) from err
        data = await self._validate_result(data, task, schema)
        self._record_result(chat_log, result.text)
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

        if tools := _chat_log_tools(chat_log):
            result = await self._async_generate_text_with_tools(
                task,
                chat_log,
                instructions,
                tools,
                _json_output_instruction(task, schema),
            )
            data: Any = result.text
            if schema is not None:
                try:
                    data = json.loads(_strip_json_fence(result.text))
                    data = await self._validate_result(data, task, schema)
                except (json.JSONDecodeError, vol.Invalid) as err:
                    raise translated_error(
                        "Groq returned data that did not match the requested structure",
                        "structured_response_invalid",
                    ) from err
            return GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=data,
            )

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
                messages=await self._async_task_messages(task, instructions, chat_log),
                strict=(
                    True
                    if task.structure is not None
                    else service_strict(self._config_entry, self._service_data)
                ),
            )
            self._raise_request_errors(request)
            try:
                response = await self._client.async_generate_structured(request)
            except GroqApiError as err:
                if task.structure is None or not _can_retry_structured_error(err):
                    raise
                return await self._async_generate_json_fallback(
                    task, chat_log, instructions, schema
                )
            structured_data: Any = response["data"]
            # The API already validated its JSON Schema; only HA's richer
            # task selector validation remains here.
            structured_data = await self._validate_result(structured_data, task, None)
            self._record_result(chat_log, response["text"])
        else:
            return await self._async_generate_json_fallback(
                task, chat_log, instructions, schema
            )

        return GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=structured_data,
        )
