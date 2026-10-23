"""Conditional marker contracts and persistence when hidden fields are omitted.

The visibility marker is a test double, not a newer HA runtime or frontend.
Real HA flow-manager coverage for the older fallback lives in lifecycle tests.
"""

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
import voluptuous as vol
from homeassistant import data_entry_flow

from custom_components.groq import config_flow, flow_schemas
from custom_components.groq.model_registry import GroqModelRegistry

from .test_integration_harness import _patch_flow_common

MODEL = "openai/gpt-oss-20b"
SCHEMA_FIELDS = {"schema_name", "schema", "strict"}
SAVED = {
    "schema_name": "saved_response",
    "schema": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    },
    "strict": True,
}


class ConditionalOptional(vol.Optional):
    """Capture the documented HA marker arguments without emulating rendering."""

    def __init__(self, *args, visible=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.visible = visible


@pytest.fixture(params=[False, True], ids=["older-fallback", "conditional-contract"])
def conditional_api(monkeypatch, request):
    """Exercise API presence and absence independently of the installed HA."""
    if request.param:
        monkeypatch.setattr(
            data_entry_flow, "Optional", ConditionalOptional, raising=False
        )
    else:
        monkeypatch.delattr(data_entry_flow, "Optional", raising=False)
    return request.param


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("saved", [{}, SAVED])
def test_only_structured_schema_fields_have_visibility(conditional_api, enabled, saved):
    schema = flow_schemas.text_generation_advanced_schema(
        {"model": MODEL, "structured_outputs": enabled, **saved}
    )
    markers = {key.schema: key for key in schema.schema}
    visible = {
        key: marker.visible
        for key, marker in markers.items()
        if getattr(marker, "visible", None) is not None
    }
    assert visible == (
        {key: {"field": "structured_outputs", "value": True} for key in SCHEMA_FIELDS}
        if conditional_api
        else {}
    )
    assert markers["structured_outputs"].default() is enabled
    for key, default in {
        "schema_name": "response",
        "schema": {},
        "strict": False,
    }.items():
        assert isinstance(markers[key], vol.Optional)
        assert markers[key].default() == saved.get(key, default)
    # Reasoning controls retain their ordinary markers in both API branches.
    for key in ("reasoning_effort", "reasoning_format", "include_reasoning"):
        assert type(markers[key]) is vol.Optional


def test_unsupported_model_has_no_structured_fields(conditional_api):
    schema = flow_schemas.text_generation_advanced_schema(
        {"model": "llama-3.1-8b-instant", "structured_outputs": True, **SAVED}
    )
    assert not (SCHEMA_FIELDS | {"structured_outputs"}) & {
        key.schema for key in schema.schema
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [False, True])
async def test_omitted_fields_survive_repeated_saves_and_reopening(
    monkeypatch, conditional_api, strict
):
    saved = {**deepcopy(SAVED), "strict": strict}
    stored = {
        "name": "Structured text",
        "model": MODEL,
        "service_type": "text_generation",
        "structured_outputs": True,
        **saved,
    }
    # Call the handler directly so no schema defaults fill in omitted fields.
    for submitted in (
        {"structured_outputs": False},
        {},
        {"structured_outputs": True},
        {"structured_outputs": True, "schema_name": "edited_response"},
        {"structured_outputs": False},
        {"structured_outputs": True},
    ):
        flow = config_flow.GroqServiceSubentryFlow()
        _patch_flow_common(monkeypatch, flow)
        monkeypatch.setattr(flow, "_existing_service_data", lambda: deepcopy(stored))
        monkeypatch.setattr(
            flow, "_model_registry", AsyncMock(return_value=GroqModelRegistry())
        )
        form = await flow.async_step_text_generation(
            {"name": stored["name"], "model": MODEL, "advanced_options": True}
        )
        assert form["step_id"] == "text_generation_advanced"
        for key in SCHEMA_FIELDS:
            marker = next(key_ for key_ in form["data_schema"].schema if key_ == key)
            assert marker.default() == stored[key]
        reopened = await flow.async_step_text_generation_advanced()
        assert reopened["step_id"] == "text_generation_advanced"
        result = await flow.async_step_text_generation_advanced(submitted)
        assert result["type"] == "create_entry"
        assert result["data"]["structured_outputs"] is submitted.get(
            "structured_outputs", stored["structured_outputs"]
        )
        saved.update(
            {key: value for key, value in submitted.items() if key in SCHEMA_FIELDS}
        )
        assert {key: result["data"][key] for key in SCHEMA_FIELDS} == saved
        assert flow._pending_service_data == {}
        stored = result["data"]


@pytest.mark.asyncio
async def test_validation_retry_keeps_saved_schema(monkeypatch, conditional_api):
    flow = config_flow.GroqServiceSubentryFlow()
    _patch_flow_common(monkeypatch, flow)
    flow._pending_service_data = {"model": MODEL, "structured_outputs": True, **SAVED}
    result = await flow.async_step_text_generation_advanced(
        {"structured_outputs": False, "request_body_options": {"model": "invalid"}}
    )
    assert result["errors"] == {"request_body_options": "reserved_request_body_option"}
    defaults = result["data_schema"]({})
    assert defaults["structured_outputs"] is False
    assert {key: defaults[key] for key in SCHEMA_FIELDS} == SAVED
    result = await flow.async_step_text_generation_advanced(
        {"structured_outputs": False, "request_body_options": {}}
    )
    assert result["type"] == "create_entry"
    assert {key: result["data"][key] for key in SCHEMA_FIELDS} == SAVED


@pytest.mark.asyncio
async def test_model_change_clears_hidden_schema(monkeypatch, conditional_api):
    flow = config_flow.GroqServiceSubentryFlow()
    _patch_flow_common(monkeypatch, flow)
    monkeypatch.setattr(
        flow,
        "_existing_service_data",
        lambda: {"model": MODEL, "structured_outputs": False, **deepcopy(SAVED)},
    )
    result = await flow.async_step_text_generation(
        {"model": "llama-3.1-8b-instant", "advanced_options": True}
    )
    assert result["step_id"] == "text_generation_advanced"
    assert (
        not (SCHEMA_FIELDS | {"structured_outputs"}) & flow._pending_service_data.keys()
    )
    result = await flow.async_step_text_generation_advanced({})
    assert result["type"] == "create_entry"
    assert not (SCHEMA_FIELDS | {"structured_outputs"}) & result["data"].keys()
