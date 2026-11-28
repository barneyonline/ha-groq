"""Regression tests across real Home Assistant configuration and registry boundaries."""

import asyncio
from copy import deepcopy
from datetime import timedelta
from types import MappingProxyType
from unittest.mock import MagicMock, Mock, patch

import pytest
from homeassistant.components.conversation import ChatLog, ConversationInput
from homeassistant.config_entries import ConfigEntryDisabler, ConfigEntryState
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import Context
from homeassistant.data_entry_flow import InvalidData
from homeassistant.exceptions import (
    ConfigEntryAuthFailed,
    ConfigEntryError,
    HomeAssistantError,
)
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.intent import IntentResponseType
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import (
    MockConfigEntry,
    async_fire_time_changed,
)

from custom_components.groq import async_migrate_entry, async_unload_entry
from custom_components.groq.api import (
    GroqApiClient,
    SpeechRequest,
    TextGenerationRequest,
)
from custom_components.groq.config_flow import (
    GroqServiceSubentryFlow,
    async_get_model_registry,
)
from custom_components.groq.const import enabled_features_from_entry
from custom_components.groq.conversation import GroqConversationEntity
from custom_components.groq.diagnostics import async_get_config_entry_diagnostics
from custom_components.groq.errors import GroqApiError
from custom_components.groq.feature_registry import GroqFeature
from custom_components.groq.model_registry import (
    BUILT_IN_MODELS,
    GroqCapability,
    GroqModel,
    GroqModelRegistry,
)
from custom_components.groq.repairs import (
    _hashed_issue_id,
    async_create_ffmpeg_missing_issue,
    async_create_model_access_issue,
    async_create_model_configuration_issue,
    async_delete_ffmpeg_missing_issue,
    async_delete_model_access_issue,
    async_delete_model_configuration_issue,
    async_reconcile_entry_issues,
    async_reconcile_startup_issues,
)
from custom_components.groq.runtime import build_runtime
from custom_components.groq.services import _ensure_model
from custom_components.groq.stt import _stt_service_data
from custom_components.groq.subentries import service_data_for_type
from custom_components.groq.text_generation import service_unique_id
from custom_components.groq.tts import GroqTTSEntity

from .test_foundation import DummyResponse, DummySession

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.usefixtures("enable_custom_integrations"),
]


@pytest.fixture(autouse=True)
async def initialized_homeassistant(hass):
    """Initialize HA's own shared exposure registry before conversation setup."""
    assert await async_setup_component(hass, "homeassistant", {})


TEXT_MODEL = "llama-3.3-70b-versatile"
TTS_MODEL = "canopylabs/orpheus-v1-english"


def account(
    hass,
    *,
    entry_id="account-a",
    services=(),
    options=None,
    data=None,
    disabled_by=None,
):
    """Register a real HA entry with immutable service data."""
    entry = MockConfigEntry(
        domain="groq",
        entry_id=entry_id,
        unique_id=entry_id,
        minor_version=2,
        disabled_by=disabled_by,
        data=data or {"api_key": "test-key", "name": "Test Groq"},
        options=options or {},
        subentries_data=list(services),
    )
    entry.add_to_hass(hass)
    return entry


def text_service(*, model=TEXT_MODEL):
    return {
        "subentry_id": "text-service",
        "subentry_type": "text_generation",
        "title": "Text service",
        "unique_id": None,
        "data": {
            "service_type": "text_generation",
            "name": "Text service",
            "model": model,
            "llm_hass_api": ["assist"],
            "max_tokens": 321,
        },
    }


def voice_service():
    return {
        "subentry_id": "voice",
        "subentry_type": "text_to_speech",
        "title": "Voice",
        "unique_id": None,
        "data": {"service_type": "text_to_speech", "model": TTS_MODEL, "voice": "troy"},
    }


def own_issues(hass):
    return {
        key: value
        for key, value in ir.async_get(hass).issues.items()
        if key[0] == "groq"
    }


@pytest.mark.parametrize("signed_in", [False, True])
async def test_mixed_assist_tool_sets_require_an_admin(
    hass, hass_read_only_user, signed_in
):
    """Core rejects the entire set before asking Groq for a response."""
    assert await async_setup_component(hass, "llm", {})
    stored_service = text_service()
    stored_service["data"][CONF_LLM_HASS_API] = ["assist", "homeassistant"]
    entry = account(hass, services=[stored_service])
    service = service_data_for_type(entry, "text_generation")[0]
    client = Mock(spec=GroqApiClient)
    agent = GroqConversationEntity(hass, entry, service, client)
    log = ChatLog(hass, "shared-assist")
    user_input = ConversationInput(
        text="Turn on the light",
        context=Context(user_id=hass_read_only_user.id if signed_in else None),
        conversation_id=log.conversation_id,
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id="conversation.groq",
    )

    result = await agent._async_handle_message(user_input, log)

    assert result.response.response_type is IntentResponseType.ERROR
    assert not client.mock_calls


class LifecycleSession:
    """Keep HA lifecycle machinery real while controlling provider responses."""

    def __init__(self):
        self.offline = False
        self.revoked = set()
        self.requests = []

    def request(self, method, url, **kwargs):
        self.requests.append((url, kwargs))
        credential = kwargs["headers"].get("Authorization")
        if credential in self.revoked:
            return DummyResponse(401, {}, {})
        if self.offline:
            return DummyResponse(503, {}, {})
        if url.endswith("/models"):
            return DummyResponse(
                200, {}, {"data": [{"id": model} for model in BUILT_IN_MODELS]}
            )
        return DummyResponse(
            200, {}, {"choices": [{"message": {"content": "Available"}}]}
        )


class PausedProviderResponse(DummyResponse):
    """Hold a provider response until the test releases it."""

    def __init__(self, status: int, payload: dict) -> None:
        super().__init__(status, {}, payload)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __aenter__(self):
        self.started.set()
        await self.release.wait()
        return self


async def test_setup_retry_recovers_through_home_assistant(hass):
    entry = account(hass, services=[text_service()])
    session = LifecycleSession()
    session.offline = True
    with patch(
        "custom_components.groq.api.async_get_clientsession", return_value=session
    ):
        assert not await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        assert entry.state is ConfigEntryState.SETUP_RETRY
        assert not entry.update_listeners
        assert not er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
        assert not hass.config_entries.flow.async_progress()
        session.offline = False
        async_fire_time_changed(hass, dt_util.utcnow() + timedelta(minutes=1))
        await hass.async_block_till_done(wait_background_tasks=True)
        assert entry.state is ConfigEntryState.LOADED
        assert len(entry.update_listeners) == 1
        entities = er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
        assert len(entities) == 9
        sensors = [entity for entity in entities if entity.domain == "sensor"]
        assert len(sensors) == 7
        assert all(
            entity.disabled_by is er.RegistryEntryDisabler.INTEGRATION
            for entity in sensors
        )
        assert {entity.domain for entity in entities if not entity.disabled} == {
            "ai_task",
            "conversation",
        }
        response = await entry.runtime_data.client.async_generate_text(
            TextGenerationRequest(prompt="Hello", model=TEXT_MODEL)
        )
        assert response.text == "Available"
        assert await hass.config_entries.async_unload(entry.entry_id)


@pytest.mark.parametrize("old_credentials_rejected", [False, True])
async def test_edit_during_setup_loads_the_current_runtime(
    hass, old_credentials_rejected
):
    """A setup-time credential update supersedes the in-flight provider result."""
    entry = account(hass, services=[text_service()])
    entered, released = asyncio.Event(), asyncio.Event()
    clients = []

    async def list_models(client, **_kwargs):
        clients.append(client)
        if len(clients) == 1:
            entered.set()
            await released.wait()
            if old_credentials_rejected:
                raise ConfigEntryAuthFailed("old credential")
        return list(BUILT_IN_MODELS.values())

    with patch.object(
        GroqApiClient, "async_list_models", autospec=True, side_effect=list_models
    ):
        setup = asyncio.create_task(hass.config_entries.async_setup(entry.entry_id))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            hass.config_entries.async_update_entry(
                entry, data={**entry.data, "api_key": "updated-key"}
            )
        finally:
            released.set()
        await setup
        await hass.async_block_till_done()

    assert len(clients) == 2
    assert entry.state is ConfigEntryState.LOADED
    assert entry.runtime_data.client is clients[1]
    assert entry.runtime_data.client.can_update_repairs
    assert not hass.config_entries.flow.async_progress()


async def test_setup_authentication_error_for_current_credentials_starts_reauth(hass):
    """Preserve Home Assistant's normal reauth path for the current snapshot."""
    entry = account(hass)
    with patch.object(
        GroqApiClient,
        "async_list_models",
        side_effect=ConfigEntryAuthFailed("current key"),
    ):
        assert not await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    assert any(
        flow["context"]["source"] == "reauth"
        for flow in hass.config_entries.flow.async_progress()
    )


async def test_failed_platform_unload_preserves_runtime_and_descriptions(hass):
    from homeassistant.helpers.service import SERVICE_DESCRIPTION_CACHE

    entry = account(hass, services=[text_service()])
    session = LifecycleSession()
    with patch(
        "custom_components.groq.api.async_get_clientsession", return_value=session
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        runtime = entry.runtime_data
        descriptions = deepcopy(hass.data[SERVICE_DESCRIPTION_CACHE])
        with (
            patch.object(
                hass.config_entries, "async_unload_platforms", return_value=False
            ),
            patch.object(
                runtime.client, "async_shutdown", wraps=runtime.client.async_shutdown
            ) as shutdown,
        ):
            with pytest.raises(ConfigEntryError) as raised:
                await async_unload_entry(hass, entry)
            assert raised.value.translation_domain == "homeassistant"
            assert raised.value.translation_key == "unload_platforms_failed"
            shutdown.assert_not_awaited()
        assert entry.runtime_data is runtime
        assert hass.data[SERVICE_DESCRIPTION_CACHE] == descriptions
        assert (
            await runtime.client.async_generate_text(
                TextGenerationRequest(prompt="Hello", model=TEXT_MODEL)
            )
        ).text == "Available"
        assert await hass.config_entries.async_unload(entry.entry_id)
        assert not entry.update_listeners


async def test_runtime_reauth_reloads_only_affected_account(hass):
    first_service = text_service()
    second_service = text_service()
    second_service["subentry_id"] = "other-text-service"
    first = account(hass, services=[first_service])
    second = account(
        hass,
        entry_id="account-b",
        services=[second_service],
        data={"api_key": "other-key", "name": "Other Groq"},
    )
    session = LifecycleSession()
    with (
        patch(
            "custom_components.groq.api.async_get_clientsession", return_value=session
        ),
        patch(
            "custom_components.groq.config_flow.async_get_clientsession",
            return_value=session,
        ),
    ):
        assert await async_setup_component(hass, "groq", {})
        await hass.async_block_till_done()
        first_runtime, second_runtime = first.runtime_data, second.runtime_data
        registry = er.async_get(hass)
        original_entities = {
            item.entity_id: item.config_subentry_id
            for entry in (first, second)
            for item in er.async_entries_for_config_entry(registry, entry.entry_id)
        }
        request = TextGenerationRequest(prompt="Hello", model=TEXT_MODEL)
        session.revoked.add("Bearer test-key")
        for _ in range(2):
            with pytest.raises(ConfigEntryAuthFailed):
                await first_runtime.client.async_generate_text(request)
        await hass.async_block_till_done()
        flows = hass.config_entries.flow.async_progress()
        assert len(flows) == 1
        assert flows[0]["context"]["entry_id"] == first.entry_id
        assert flows[0]["context"]["source"] == "reauth"
        assert (
            await second_runtime.client.async_generate_text(request)
        ).text == "Available"
        with patch.object(
            hass.config_entries, "async_reload", wraps=hass.config_entries.async_reload
        ) as reload:
            result = await hass.config_entries.flow.async_configure(
                flows[0]["flow_id"], {"api_key": "replacement-key"}
            )
            await hass.async_block_till_done()
        assert result["reason"] == "reauth_successful"
        reload.assert_awaited_once_with(first.entry_id)
        assert first.state is second.state is ConfigEntryState.LOADED
        assert first.data["api_key"] == "replacement-key"
        assert second.data["api_key"] == "other-key"
        assert first.runtime_data is not first_runtime
        assert second.runtime_data is second_runtime
        assert not hass.config_entries.flow.async_progress()
        assert (
            await first.runtime_data.client.async_generate_text(request)
        ).text == "Available"
        assert (
            session.requests[-1][1]["headers"]["Authorization"]
            == "Bearer replacement-key"
        )
        with pytest.raises(GroqApiError, match="unloaded"):
            await first_runtime.client.async_synthesize_speech(
                SpeechRequest(text="Hello", model=TTS_MODEL, voice="troy")
            )
        assert {
            item.entity_id: item.config_subentry_id
            for entry in (first, second)
            for item in er.async_entries_for_config_entry(registry, entry.entry_id)
        } == original_entities
        assert len(first.update_listeners) == len(second.update_listeners) == 1
        for entry in (first, second):
            assert await hass.config_entries.async_unload(entry.entry_id)


async def test_real_entry_mapping_diagnostics(hass):
    entry = account(hass, services=[text_service()])
    assert isinstance(entry.data, MappingProxyType)
    assert isinstance(entry.subentries["text-service"].data, MappingProxyType)
    assert enabled_features_from_entry(entry) == ["text_generation"]
    diagnostics = await async_get_config_entry_diagnostics(hass, entry)
    assert diagnostics["summary"]["enabled_features"] == ["text_generation"]
    assert diagnostics["summary"]["service_counts"] == {"text_generation": 1}


async def test_legacy_mapping_migration_is_idempotent(hass):
    entry = MockConfigEntry(
        domain="groq",
        unique_id=None,
        data={"unique_id": "stable-id", "api_key": "test-key"},
    )
    entry.add_to_hass(hass)
    assert await async_migrate_entry(hass, entry) is None
    assert entry.unique_id == "stable-id"
    assert entry.minor_version == 2
    assert dict(entry.data) == {"api_key": "test-key"}
    assert await async_migrate_entry(hass, entry) is None


@pytest.mark.parametrize("version", [2, 99])
async def test_future_major_migration_explains_rejection_without_changes(hass, version):
    entry = MockConfigEntry(
        domain="groq",
        version=version,
        minor_version=7,
        unique_id="account-id",
        data={"api_key": "test-key", "unique_id": "legacy-id"},
        options={"name": "Groq account"},
    )
    entry.add_to_hass(hass)
    before = deepcopy(entry.as_dict())
    with (
        patch.object(hass.config_entries, "async_update_entry") as update,
        pytest.raises(ConfigEntryError) as raised,
    ):
        await async_migrate_entry(hass, entry)
    update.assert_not_called()
    assert entry.as_dict() == before
    error = raised.value
    assert error.translation_domain == "groq"
    assert error.translation_key == "unsupported_config_version"
    assert error.translation_placeholders == {
        "version": str(version),
        "supported_version": "1",
    }
    assert str(error) == (
        f"This Groq configuration uses version {version}, but the installed "
        "integration supports up to version 1. Update the Groq integration "
        "to a version that supports this configuration."
    )


async def test_newer_minor_migration_is_accepted_without_changes(hass):
    entry = MockConfigEntry(domain="groq", version=1, minor_version=99)
    entry.add_to_hass(hass)
    before = deepcopy(entry.as_dict())
    with patch.object(hass.config_entries, "async_update_entry") as update:
        assert await async_migrate_entry(hass, entry) is None
    update.assert_not_called()
    assert entry.as_dict() == before


async def test_future_major_blocks_setup_without_retry_or_platforms(hass):
    entry = MockConfigEntry(domain="groq", version=2, data={"api_key": "test-key"})
    entry.add_to_hass(hass)
    with (
        patch("custom_components.groq.async_setup_entry") as setup,
        patch.object(hass.config_entries, "async_forward_entry_setups") as forward,
        patch(
            "custom_components.groq.async_migrate_entry", wraps=async_migrate_entry
        ) as migrate,
    ):
        assert not await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        assert entry.state is ConfigEntryState.MIGRATION_ERROR
        # Newer HA rejects future major versions before invoking the integration.
        migration_calls = migrate.await_count
        assert migration_calls <= 1
        async_fire_time_changed(hass, dt_util.utcnow() + timedelta(hours=1))
        await hass.async_block_till_done()
        assert migrate.await_count == migration_calls
        setup.assert_not_called()
        forward.assert_not_called()
        assert entry.state is ConfigEntryState.MIGRATION_ERROR


async def test_legacy_tts_options_have_consistent_diagnostics(hass):
    entry = account(
        hass,
        options={
            "url": "https://api.groq.com/openai/v1/audio/speech",
            "model": TTS_MODEL,
            "voice": "troy",
        },
    )
    diagnostics = await async_get_config_entry_diagnostics(hass, entry)
    assert diagnostics["summary"]["enabled_features"] == ["text_to_speech"]


@pytest.mark.parametrize("replacement", ["", "replacement-key"])
async def test_real_options_flow_preserves_legacy_options(hass, replacement):
    original = {
        "api_key": "legacy-working-key",
        "voice": "troy",
        "enabled_features": ["text_to_speech"],
        "cache_size": 4,
    }
    entry = account(hass, options=original)
    with (
        patch(
            "custom_components.groq.config_flow.async_validate_api_key",
            return_value=None,
        ),
        patch.object(hass.config_entries, "async_reload", return_value=True) as reload,
    ):
        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], {"api_key": replacement}
        )
    assert result["type"] == "create_entry"
    expected = dict(original)
    if replacement:
        expected.pop("api_key")
        assert entry.data["api_key"] == replacement
    assert dict(entry.options) == expected
    reload.assert_awaited_once_with(entry.entry_id)


@pytest.mark.parametrize("advanced", [False, True])
async def test_real_subentry_flow_clears_assist_control(hass, advanced):
    entry = account(hass, services=[text_service()])
    with patch(
        "custom_components.groq.config_flow.async_get_model_registry",
        return_value=GroqModelRegistry(),
    ):
        result = await hass.config_entries.subentries.async_init(
            (entry.entry_id, "text_generation"),
            context={"source": "reconfigure", "subentry_id": "text-service"},
        )
        assert result["type"] == "form"
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"],
            {
                "name": "Text service",
                "model": TEXT_MODEL,
                "llm_hass_api": [],
                "advanced_options": advanced,
            },
        )
        if advanced:
            assert result["step_id"] == "text_generation_advanced"
            result = await hass.config_entries.subentries.async_configure(
                result["flow_id"], {}
            )
    assert result["type"] == "abort"
    assert result["reason"] == "reconfigure_successful"
    stored = entry.subentries["text-service"].data
    assert "llm_hass_api" not in stored
    assert stored["max_tokens"] == 321


async def test_real_structured_output_visibility_repeated_saves(hass):
    """Validate visible fields and persistence through the HA flow manager."""
    service = text_service(model="openai/gpt-oss-20b")
    service["data"].pop("llm_hass_api")
    entry = account(hass, services=[service])
    schema_fields = {"schema_name", "schema", "strict"}
    saved = {
        "schema_name": "saved_response",
        "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
        "strict": True,
    }
    enabled = False
    with patch(
        "custom_components.groq.config_flow.async_get_model_registry",
        return_value=GroqModelRegistry(),
    ):
        for submitted in (
            {"structured_outputs": True, **saved},
            {"structured_outputs": False, "schema_name": "ignored_hidden_value"},
            {},
            {"structured_outputs": True},
            {"strict": False},
            {},
        ):
            result = await hass.config_entries.subentries.async_init(
                (entry.entry_id, "text_generation"),
                context={"source": "reconfigure", "subentry_id": "text-service"},
            )
            result = await hass.config_entries.subentries.async_configure(
                result["flow_id"],
                {
                    "name": "Text service",
                    "model": "openai/gpt-oss-20b",
                    "advanced_options": True,
                },
            )
            assert result["step_id"] == "text_generation_advanced"
            markers = {key.schema: key for key in result["data_schema"].schema}
            for key in schema_fields:
                assert markers[key].visible == {
                    "field": "structured_outputs",
                    "value": True,
                }
            assert markers["structured_outputs"].default() is enabled
            defaults = result["data_schema"]({})
            previous = entry.subentries["text-service"].data
            for key, default in {
                "schema_name": "response",
                "schema": {},
                "strict": False,
            }.items():
                assert defaults[key] == previous.get(key, default)
            result = await hass.config_entries.subentries.async_configure(
                result["flow_id"], submitted
            )
            assert result["type"] == "abort"
            assert result["reason"] == "reconfigure_successful"
            enabled = submitted.get("structured_outputs", enabled)
            saved.update(
                {
                    key: value
                    for key, value in submitted.items()
                    if enabled and key in schema_fields
                }
            )
            stored = entry.subentries["text-service"].data
            assert stored["structured_outputs"] is enabled
            assert {key: stored[key] for key in schema_fields} == saved


async def test_real_reasoning_visibility_preserves_edits_after_retry(hass):
    """Keep the toggle when a retry hides it after a visible edit."""
    service = text_service(model="openai/gpt-oss-20b")
    entry = account(hass, services=[service])
    with patch(
        "custom_components.groq.config_flow.async_get_model_registry",
        return_value=GroqModelRegistry(),
    ):
        form = await hass.config_entries.subentries.async_init(
            (entry.entry_id, "text_generation"),
            context={"source": "reconfigure", "subentry_id": "text-service"},
        )
        flow_id = form["flow_id"]
        form = await hass.config_entries.subentries.async_configure(
            flow_id,
            {
                "name": "Text service",
                "model": "openai/gpt-oss-20b",
                "advanced_options": True,
            },
        )
        assert form["step_id"] == "text_generation_advanced"
        form = await hass.config_entries.subentries.async_configure(
            flow_id,
            {
                "reasoning_format": "",
                "include_reasoning": True,
                "request_body_options": {"model": "invalid"},
            },
        )
        assert form["errors"] == {
            "request_body_options": "reserved_request_body_option"
        }
        result = await hass.config_entries.subentries.async_configure(
            flow_id,
            {
                "reasoning_format": "parsed",
                "include_reasoning": False,
                "request_body_options": {},
            },
        )
    assert result["type"] == "abort"
    assert result["reason"] == "reconfigure_successful"
    saved = entry.subentries["text-service"].data
    assert saved["reasoning_format"] == "parsed"
    assert saved["include_reasoning"] is True


@pytest.mark.parametrize("advanced", [False, True])
async def test_real_model_change_clears_disabled_structured_outputs(hass, advanced):
    """Capability cleanup still removes saved settings when their toggle is off."""
    service = text_service(model="openai/gpt-oss-20b")
    service["data"].update(
        structured_outputs=False,
        schema_name="saved_response",
        schema={"type": "object"},
        strict=True,
    )
    entry = account(hass, services=[service])
    with patch(
        "custom_components.groq.config_flow.async_get_model_registry",
        return_value=GroqModelRegistry(),
    ):
        result = await hass.config_entries.subentries.async_init(
            (entry.entry_id, "text_generation"),
            context={"source": "reconfigure", "subentry_id": "text-service"},
        )
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"],
            {
                "name": "Text service",
                "model": "llama-3.1-8b-instant",
                "advanced_options": advanced,
            },
        )
        if advanced:
            assert result["step_id"] == "text_generation_advanced"
            assert not {"structured_outputs", "schema_name", "schema", "strict"} & {
                key.schema for key in result["data_schema"].schema
            }
            result = await hass.config_entries.subentries.async_configure(
                result["flow_id"], {}
            )
    assert result["reason"] == "reconfigure_successful"
    assert (
        not {"structured_outputs", "schema_name", "schema", "strict"}
        & entry.subentries["text-service"].data.keys()
    )


@pytest.mark.parametrize(
    "feature",
    ["text_generation", "speech_to_text", "text_to_speech", "image_recognition"],
)
async def test_real_subentry_flow_has_no_unavailable_fallback(hass, feature):
    entry = account(hass)
    # Successful discovery is authoritative even when no model serves this feature.
    registry = GroqModelRegistry(
        [GroqModel("inactive", active=False)], include_built_ins=False
    )
    with patch(
        "custom_components.groq.config_flow.async_get_model_registry",
        return_value=registry,
    ):
        result = await hass.config_entries.subentries.async_init(
            (entry.entry_id, feature), context={"source": "user"}
        )
    assert result["type"] == "abort"
    assert result["reason"] == "no_compatible_models"
    assert not entry.subentries


@pytest.mark.parametrize(
    "feature,model",
    [
        ("text_generation", TEXT_MODEL),
        ("speech_to_text", "whisper-large-v3"),
        ("text_to_speech", TTS_MODEL),
        ("image_recognition", "meta-llama/llama-4-scout-17b-16e-instruct"),
    ],
)
async def test_real_subentry_flow_rejects_unavailable_model(hass, feature, model):
    entry = account(hass)
    with patch(
        "custom_components.groq.config_flow.async_get_model_registry",
        return_value=GroqModelRegistry(),
    ):
        result = await hass.config_entries.subentries.async_init(
            (entry.entry_id, feature), context={"source": "user"}
        )
        with pytest.raises(InvalidData):
            await hass.config_entries.subentries.async_configure(
                result["flow_id"], {"name": "Bad model", "model": "retired-model"}
            )
        flow = GroqServiceSubentryFlow()
        flow.hass = hass
        flow.handler = (entry.entry_id, feature)
        flow.context = {"source": "user"}
        result = await flow._async_service_step(
            feature, {"name": "Bad model", "model": "retired-model"}
        )
    assert result["type"] == "abort"
    assert result["reason"] == "model_unavailable"


async def test_discovery_failure_preserves_builtin_fallback(hass):
    with patch(
        "custom_components.groq.config_flow.async_fetch_available_models",
        side_effect=TimeoutError,
    ):
        registry = await async_get_model_registry(hass, "test-key")
    assert registry.models_for_feature(GroqFeature.TEXT_TO_SPEECH)


async def test_empty_successful_catalog_does_not_offer_builtin_models(hass):
    entry = account(hass)
    with patch(
        "custom_components.groq.config_flow.async_fetch_available_models",
        return_value=[],
    ):
        result = await hass.config_entries.subentries.async_init(
            (entry.entry_id, "text_generation"), context={"source": "user"}
        )
    assert result["type"] == "abort"
    assert result["reason"] == "no_compatible_models"


async def test_inactive_model_capability_guard():
    model = GroqModel(
        "retired",
        active=False,
        capabilities=frozenset({GroqCapability.TEXT_GENERATION}),
    )
    registry = GroqModelRegistry([model], include_built_ins=False)
    assert not registry.supports(model.model_id, GroqFeature.TEXT_GENERATION)
    assert not registry.supports(model.model_id, GroqCapability.TEXT_GENERATION)
    assert not registry.models_for_feature(GroqFeature.TEXT_GENERATION)


async def test_model_access_repairs_are_owned_and_clear_after_matching_success(hass):
    for entry_id in ("account-a", "account-b"):
        account(hass, entry_id=entry_id)
        async_create_model_access_issue(
            hass, TEXT_MODEL, entry_id=entry_id, reason="model_permission_blocked_org"
        )
    assert len(own_issues(hass)) == 2
    async_delete_model_access_issue(hass, TEXT_MODEL, entry_id="account-a")
    remaining = list(own_issues(hass).values())
    assert len(remaining) == 1
    assert remaining[0].data["entry_id"] == "account-b"
    async_delete_model_access_issue(hass, TEXT_MODEL, entry_id="account-b")
    assert not own_issues(hass)


async def test_inflight_text_success_preserves_dismissal_during_disable(hass):
    """Core hides the dismissed repair while the account unloads."""
    unload_started, release_unload = asyncio.Event(), asyncio.Event()

    config_entries = hass.config_entries
    unload_platforms = config_entries.async_unload_platforms

    async def pause_platform_unload(entry, platforms):
        unload_started.set()
        await release_unload.wait()
        return await unload_platforms(entry, platforms)

    service = text_service()
    subentry_id = service["subentry_id"]
    entry = account(hass, services=[service])
    session = LifecycleSession()
    with patch(
        "custom_components.groq.api.async_get_clientsession", return_value=session
    ):
        assert await config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        client = entry.runtime_data.client
        async_create_model_access_issue(
            hass, TEXT_MODEL, entry_id=entry.entry_id, subentry_id=subentry_id
        )
        (issue_key,) = own_issues(hass)
        ir.async_ignore_issue(hass, *issue_key, True)
        dismissed_issues = own_issues(hass)
        issue = dismissed_issues[issue_key]
        assert issue.dismissed_version is not None

        response = PausedProviderResponse(
            200, {"choices": [{"message": {"content": "Available"}}]}
        )
        with (
            patch.object(session, "request", return_value=response),
            patch.object(
                config_entries,
                "async_unload_platforms",
                side_effect=pause_platform_unload,
            ),
        ):
            request_task = asyncio.create_task(
                client.async_generate_text(
                    TextGenerationRequest(
                        prompt="Hello",
                        model=TEXT_MODEL,
                        service_id=subentry_id,
                        config_subentry_id=subentry_id,
                    )
                )
            )
            disable_task = None
            try:
                await asyncio.wait_for(response.started.wait(), timeout=5)
                disable_task = asyncio.create_task(
                    config_entries.async_set_disabled_by(
                        entry.entry_id, ConfigEntryDisabler.USER
                    )
                )
                await asyncio.wait_for(unload_started.wait(), timeout=5)
                assert entry.disabled_by is ConfigEntryDisabler.USER
                assert not disable_task.done()
                response.release.set()
                result = await asyncio.wait_for(request_task, timeout=5)
                assert result.text == "Available"
                assert own_issues(hass) == dismissed_issues
                assert ir.async_get(hass).async_is_hidden(issue)
            finally:
                response.release.set()
                release_unload.set()
                async with asyncio.timeout(5):
                    await asyncio.gather(
                        *(
                            task
                            for task in (request_task, disable_task)
                            if task is not None
                        ),
                        return_exceptions=True,
                    )

        assert await disable_task
        assert entry.state is ConfigEntryState.NOT_LOADED
        assert await config_entries.async_set_disabled_by(entry.entry_id, None)
        assert entry.state is ConfigEntryState.LOADED
        assert own_issues(hass) == dismissed_issues
        assert not ir.async_get(hass).async_is_hidden(own_issues(hass)[issue_key])
        assert await config_entries.async_unload(entry.entry_id)


@pytest.mark.parametrize("status", [200, 403])
@pytest.mark.parametrize("changed", ["shutdown", "account", "options", "service"])
async def test_old_client_cannot_change_repairs_after_configuration_changes(
    hass, status, changed
):
    """A late response cannot mutate repairs once the request context is obsolete."""
    payload = (
        {"choices": [{"message": {"content": "old result"}}]}
        if status == 200
        else {"error": {"code": "model_permission_blocked_org"}}
    )

    entry = account(hass, services=[text_service()])
    client = build_runtime(hass, entry).client
    response = PausedProviderResponse(status, payload)
    client._session = DummySession(response)
    task = asyncio.create_task(
        client.async_generate_text(
            TextGenerationRequest(
                prompt="Hello", model=TEXT_MODEL, service_id="text-service"
            )
        )
    )
    await asyncio.wait_for(response.started.wait(), timeout=5)
    try:
        if changed == "shutdown":
            await client.async_shutdown()
            if status == 403:
                hass.config_entries.async_remove_subentry(entry, "text-service")
        elif changed == "account":
            hass.config_entries.async_update_entry(
                entry, data={**entry.data, "api_key": "new-key"}
            )
        elif changed == "options":
            hass.config_entries.async_update_entry(entry, options={"cache_size": 2})
        else:
            service = entry.subentries["text-service"]
            hass.config_entries.async_update_subentry(
                entry, service, data={**service.data, "temperature": 0.1}
            )
        if status == 200:
            async_create_model_access_issue(
                hass, TEXT_MODEL, "text-service", entry_id=entry.entry_id
            )
        expected = own_issues(hass)
        assert len(expected) == (1 if status == 200 else 0)
    finally:
        response.release.set()

    if status == 403:
        with pytest.raises(GroqApiError):
            await task
    else:
        assert (await task).text == "old result"
    assert own_issues(hass) == expected


async def test_ffmpeg_failure_for_changed_profile_does_not_create_repair(hass):
    """An ffmpeg subprocess can finish after its entity's configuration changes."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def fail_to_spawn(*_args, **_kwargs):
        entered.set()
        await release.wait()
        raise FileNotFoundError("ffmpeg")

    stored_service = {
        "subentry_id": "voice",
        "subentry_type": "text_to_speech",
        "title": "Voice",
        "unique_id": None,
        "data": {
            "service_type": "text_to_speech",
            "model": TTS_MODEL,
            "normalize_audio": True,
        },
    }
    entry = account(hass, services=[stored_service])
    service_data = service_data_for_type(entry, "text_to_speech")[0]
    voice = GroqTTSEntity(hass, entry, build_runtime(hass, entry).client, service_data)
    with patch(
        "custom_components.groq.tts.asyncio.create_subprocess_exec",
        side_effect=fail_to_spawn,
    ):
        task = asyncio.create_task(voice._async_run_ffmpeg(["ffmpeg"]))
        await asyncio.wait_for(entered.wait(), timeout=5)
        try:
            service = entry.subentries["voice"]
            hass.config_entries.async_update_subentry(
                entry, service, data={**service.data, "normalize_audio": False}
            )
        finally:
            release.set()
        with pytest.raises(HomeAssistantError, match="ffmpeg not found"):
            await task

    assert not own_issues(hass)


@pytest.mark.parametrize("has_metadata", [False, True])
async def test_ffmpeg_repair_uses_actual_subentry_despite_other_service_alias(
    hass, has_metadata
):
    """An unrelated stored alias cannot suppress an identified voice repair."""
    text = text_service()
    text["data"]["unique_id"] = "voice"
    entry = account(hass, services=[voice_service(), text])
    (service_data,) = service_data_for_type(entry, "text_to_speech")
    issue_id = _hashed_issue_id("ffmpeg_missing", entry.entry_id, "voice")
    ir.async_create_issue(
        hass,
        "groq",
        issue_id,
        config_entry_id=entry.entry_id if has_metadata else None,
        data=(
            {"entry_id": entry.entry_id, "service_id": "voice"}
            if has_metadata
            else None
        ),
        is_fixable=False,
        is_persistent=True,
        severity=ir.IssueSeverity.ERROR,
        translation_key="ffmpeg_missing",
    )
    ir.async_ignore_issue(hass, "groq", issue_id, True)
    ignored = own_issues(hass)[("groq", issue_id)].dismissed_version
    async_reconcile_startup_issues(hass)
    async_reconcile_entry_issues(hass, entry)
    entity = GroqTTSEntity(hass, entry, build_runtime(hass, entry).client, service_data)
    with (
        patch(
            "custom_components.groq.tts.asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError,
        ),
        pytest.raises(HomeAssistantError, match="ffmpeg not found"),
    ):
        await entity._async_run_ffmpeg(["ffmpeg"])
    (issue,) = own_issues(hass).values()
    assert issue.issue_id == issue_id
    assert issue.dismissed_version == ignored
    assert issue.data["subentry_id"] == "voice"
    assert getattr(issue, "config_subentry_id", "voice") == "voice"

    hass.config_entries.async_remove_subentry(entry, "text-service")
    async_reconcile_entry_issues(hass, entry)
    async_create_ffmpeg_missing_issue(hass, entry, service_data)
    assert {candidate.issue_id for candidate in own_issues(hass).values()} == {
        issue.issue_id
    }
    async_delete_ffmpeg_missing_issue(hass, entry, service_data)
    assert not own_issues(hass)


async def test_http_repairs_keep_colliding_named_services_separate(hass):
    """Two services with the same historical key can fail and recover independently."""
    first = text_service()
    first["data"]["unique_id"] = "other-service"
    second = {**text_service(), "subentry_id": "other-service"}
    entry = account(hass, services=[first, second])
    runtime = build_runtime(hass, entry)
    response = DummyResponse(
        403, {}, {"error": {"code": "model_permission_blocked_org"}}
    )
    runtime.client._session = DummySession(response)
    requests = [
        TextGenerationRequest(
            prompt="Hello",
            model=TEXT_MODEL,
            service_id="other-service",
            config_subentry_id=subentry_id,
        )
        for subentry_id in ("text-service", "other-service")
    ]
    for request in requests:
        with pytest.raises(GroqApiError):
            await runtime.client.async_generate_text(request)
    assert {issue.data["subentry_id"] for issue in own_issues(hass).values()} == {
        "text-service",
        "other-service",
    }

    runtime.client._session = DummySession(
        DummyResponse(200, {}, {"choices": [{"message": {"content": "ok"}}]})
    )
    await runtime.client.async_generate_text(requests[0])
    assert [issue.data["subentry_id"] for issue in own_issues(hass).values()] == [
        "other-service"
    ]


def store_legacy_model_repair(hass, entry, service_id, model=TEXT_MODEL, owner=None):
    """Seed a persistent repair with the previous integration's key shape."""
    issue_id = _hashed_issue_id("model_access", entry.entry_id, model, service_id)
    data = {"entry_id": entry.entry_id, "service_id": service_id, "model": model}
    if owner is not None:
        data["subentry_id"] = owner
    ir.async_create_issue(
        hass,
        "groq",
        issue_id,
        config_entry_id=entry.entry_id,
        data=data,
        is_fixable=False,
        is_persistent=True,
        severity=ir.IssueSeverity.ERROR,
        translation_key="model_access",
    )
    return issue_id


@pytest.mark.parametrize("has_named_tts", [False, True])
@pytest.mark.parametrize("has_metadata", [False, True])
async def test_startup_adopts_legacy_ffmpeg_only_for_active_flat_tts(
    hass, has_named_tts, has_metadata
):
    entry = account(
        hass,
        services=[voice_service()] if has_named_tts else [],
        data={
            "name": "Legacy Groq",
            "url": "https://api.groq.com",
            "model": TTS_MODEL,
            "voice": "troy",
            "api_key": "test-key",
        },
    )
    issue_id = _hashed_issue_id("ffmpeg_missing", entry.entry_id, None)
    ir.async_create_issue(
        hass,
        "groq",
        issue_id,
        config_entry_id=entry.entry_id if has_metadata else None,
        data={"entry_id": entry.entry_id} if has_metadata else None,
        is_fixable=False,
        is_persistent=True,
        severity=ir.IssueSeverity.ERROR,
        translation_key="ffmpeg_missing",
    )
    ir.async_ignore_issue(hass, "groq", issue_id, True)
    original = own_issues(hass)[("groq", issue_id)]
    async_reconcile_startup_issues(hass)

    if has_named_tts:
        assert not any(issue.active for issue in own_issues(hass).values())
    else:
        (issue,) = own_issues(hass).values()
        assert issue.config_entry_id == entry.entry_id
        assert issue.issue_id == issue_id
        assert issue.dismissed_version == original.dismissed_version


@pytest.mark.parametrize("kind", ["model_access", "model_configuration"])
async def test_startup_repair_adoption_requires_a_verifiable_historical_key(hass, kind):
    entry = account(hass, services=[text_service()])
    issue_id = _hashed_issue_id(
        kind, "removed-account", "text-service", TEXT_MODEL, "reasoning"
    )
    ir.async_create_issue(
        hass,
        "groq",
        issue_id,
        is_fixable=False,
        is_persistent=True,
        severity=ir.IssueSeverity.ERROR,
        translation_key=kind,
        translation_placeholders={"model": TEXT_MODEL, "feature": "reasoning"},
    )

    async_reconcile_startup_issues(hass)
    assert own_issues(hass)[("groq", issue_id)].config_entry_id is None
    async_reconcile_entry_issues(hass, entry)
    assert not own_issues(hass)


async def test_attributed_recovery_clears_unowned_historical_collision(hass):
    first = text_service()
    first["data"]["unique_id"] = "other-service"
    entry = account(
        hass, services=[first, {**text_service(), "subentry_id": "other-service"}]
    )
    store_legacy_model_repair(hass, entry, "other-service")
    async_reconcile_entry_issues(hass, entry)
    for subentry_id in ("text-service", "other-service"):
        async_delete_model_access_issue(
            hass,
            TEXT_MODEL,
            "other-service",
            entry_id=entry.entry_id,
            subentry_id=subentry_id,
        )
    assert not own_issues(hass)


@pytest.mark.parametrize("action", [False, True])
async def test_text_service_with_null_legacy_id_preserves_historical_repairs(
    hass, action
):
    stored = text_service()
    stored["data"]["unique_id"] = None
    entry = account(hass, services=[stored])
    service = service_data_for_type(entry, "text_generation")[0]
    historical_id = service_unique_id(entry, service)
    issue_id = store_legacy_model_repair(hass, entry, None if action else historical_id)
    ir.async_ignore_issue(hass, "groq", issue_id, True)
    original = own_issues(hass)[("groq", issue_id)]
    async_reconcile_entry_issues(hass, entry)
    async_create_model_access_issue(
        hass,
        TEXT_MODEL,
        historical_id,
        entry_id=entry.entry_id,
        subentry_id="text-service",
    )
    (issue,) = own_issues(hass).values()
    assert issue.issue_id == issue_id
    assert issue.dismissed_version == original.dismissed_version
    assert issue.data["subentry_id"] == "text-service"
    async_delete_model_access_issue(
        hass,
        TEXT_MODEL,
        historical_id,
        entry_id=entry.entry_id,
        subentry_id="text-service",
    )
    assert not own_issues(hass)


@pytest.mark.parametrize("has_named_tts", [False, True])
async def test_blank_account_repair_is_not_claimed_by_an_idless_text_action(
    hass, has_named_tts
):
    repair_model = "canopylabs/orpheus-arabic-saudi" if has_named_tts else TTS_MODEL
    stored = text_service(model=repair_model)
    stored["data"]["unique_id"] = None
    entry = account(
        hass,
        services=[stored, voice_service()] if has_named_tts else [stored],
        data={
            "name": "Legacy Groq",
            "url": "https://api.groq.com",
            "model": TTS_MODEL,
            "voice": "troy",
            "api_key": "test-key",
        },
    )
    issue_id = store_legacy_model_repair(hass, entry, None, repair_model)
    original = own_issues(hass)[("groq", issue_id)]
    async_reconcile_entry_issues(hass, entry)
    async_delete_model_access_issue(
        hass, repair_model, entry_id=entry.entry_id, subentry_id="text-service"
    )
    assert own_issues(hass)[("groq", issue_id)] is original


async def test_flat_tts_model_override_repair_survives_unrelated_subentry_removal(hass):
    override_model = "canopylabs/orpheus-arabic-saudi"
    stored = text_service()
    stored["data"]["unique_id"] = None
    entry = account(
        hass,
        services=[stored],
        data={
            "name": "Legacy Groq",
            "url": "https://api.groq.com",
            "model": TTS_MODEL,
            "voice": "troy",
            "api_key": "test-key",
        },
    )
    runtime = build_runtime(hass, entry)
    runtime.client._session = DummySession(
        DummyResponse(403, {}, {"error": {"code": "model_permission_blocked_org"}})
    )
    entity = GroqTTSEntity(hass, entry, runtime.client)
    assert await entity.async_get_tts_audio(
        "Hello", "en", {"model": override_model, "voice": "fahad"}
    ) == (None, None)
    ((key, issue),) = own_issues(hass).items()
    assert issue.data["model"] == override_model
    ir.async_ignore_issue(hass, *key, True)
    original = own_issues(hass)[key]

    hass.config_entries.async_update_entry(
        entry,
        data={field: value for field, value in entry.data.items() if field != "voice"},
    )
    async_delete_model_access_issue(
        hass, override_model, entry_id=entry.entry_id, subentry_id="text-service"
    )
    async_reconcile_startup_issues(hass)
    hass.config_entries.async_remove_subentry(entry, "text-service")
    assert own_issues(hass)[key] is original


async def test_model_access_repair_ignores_ids_unused_by_other_historical_producers(
    hass,
):
    first = text_service()
    first["data"]["unique_id"] = "first-historical-alias"
    second = {**text_service(), "subentry_id": "second-physical-id"}
    second["data"]["unique_id"] = "text-service"
    tts = voice_service()
    tts["data"]["unique_id"] = "text-service"
    entry = account(hass, services=[first, second, tts])
    issue_id = store_legacy_model_repair(hass, entry, "text-service")
    ir.async_ignore_issue(hass, "groq", issue_id, True)
    original = own_issues(hass)[("groq", issue_id)]
    async_create_model_access_issue(
        hass,
        TEXT_MODEL,
        "first-historical-alias",
        entry_id=entry.entry_id,
        subentry_id="text-service",
    )
    assert own_issues(hass)[("groq", issue_id)] is original
    async_delete_model_access_issue(
        hass,
        TEXT_MODEL,
        "first-historical-alias",
        entry_id=entry.entry_id,
        subentry_id="text-service",
    )
    async_reconcile_entry_issues(hass, entry)
    async_create_model_access_issue(
        hass,
        TEXT_MODEL,
        "text-service",
        entry_id=entry.entry_id,
        subentry_id="second-physical-id",
    )
    (issue,) = own_issues(hass).values()
    assert issue.issue_id == issue_id
    assert issue.dismissed_version == original.dismissed_version
    assert issue.data["subentry_id"] == "second-physical-id"


async def test_stt_entity_and_action_share_historic_model_repair(hass):
    model = "whisper-large-v3"
    entry = account(
        hass,
        services=[
            {
                "subentry_id": "speech",
                "subentry_type": "speech_to_text",
                "title": "Speech",
                "unique_id": None,
                "data": {
                    "service_type": "speech_to_text",
                    "model": model,
                    "unique_id": "legacy",
                },
            }
        ],
    )
    (entity_service,) = _stt_service_data(entry)
    (action_service,) = service_data_for_type(entry, "speech_to_text")
    issue_id = store_legacy_model_repair(
        hass, entry, action_service["unique_id"], model
    )
    ir.async_ignore_issue(hass, "groq", issue_id, True)
    dismissed = own_issues(hass)[("groq", issue_id)].dismissed_version
    for service in (entity_service, action_service):
        async_create_model_access_issue(
            hass,
            model,
            service["unique_id"],
            entry_id=entry.entry_id,
            subentry_id=service["subentry_id"],
        )
    (retained,) = own_issues(hass).values()
    assert retained.issue_id == issue_id
    assert retained.dismissed_version == dismissed
    async_delete_model_access_issue(
        hass,
        model,
        entity_service["unique_id"],
        entry_id=entry.entry_id,
        subentry_id=entity_service["subentry_id"],
    )
    assert not own_issues(hass)


async def test_historical_key_owned_by_another_service_is_not_reused(hass):
    first = text_service()
    first["data"]["unique_id"] = "other-service"
    entry = account(
        hass, services=[first, {**text_service(), "subentry_id": "other-service"}]
    )
    issue_id = store_legacy_model_repair(
        hass, entry, "other-service", owner="text-service"
    )
    original = own_issues(hass)[("groq", issue_id)]
    async_create_model_access_issue(
        hass,
        TEXT_MODEL,
        "other-service",
        entry_id=entry.entry_id,
        subentry_id="other-service",
    )
    assert own_issues(hass)[("groq", issue_id)] is original
    assert {issue.data["subentry_id"] for issue in own_issues(hass).values()} == {
        "text-service",
        "other-service",
    }


@pytest.mark.parametrize("has_metadata", [False, True])
async def test_configuration_repair_uses_recorded_feature_to_resolve_alias(
    hass, has_metadata
):
    first = text_service()
    first["data"]["unique_id"] = "vision"
    vision = {
        "subentry_id": "vision",
        "subentry_type": "image_recognition",
        "title": "Vision",
        "unique_id": None,
        "data": {"service_type": "image_recognition", "model": "vision-model"},
    }
    entry = account(hass, services=[first, vision])
    issue_id = _hashed_issue_id(
        "model_configuration", entry.entry_id, "vision", "vision-model", "ocr"
    )
    ir.async_create_issue(
        hass,
        "groq",
        issue_id,
        config_entry_id=entry.entry_id if has_metadata else None,
        data=(
            {
                "entry_id": entry.entry_id,
                "service_id": "vision",
                "model": "vision-model",
                "feature": "ocr",
            }
            if has_metadata
            else None
        ),
        is_fixable=False,
        is_persistent=True,
        severity=ir.IssueSeverity.ERROR,
        translation_key="model_configuration",
        translation_placeholders={"model": "vision-model", "feature": "ocr"},
    )
    ir.async_ignore_issue(hass, "groq", issue_id, True)
    original = own_issues(hass)[("groq", issue_id)]
    async_reconcile_startup_issues(hass)
    async_reconcile_entry_issues(hass, entry)
    (service,) = service_data_for_type(entry, "image_recognition")
    async_create_model_configuration_issue(hass, entry, service, "vision-model", "ocr")
    (retained,) = own_issues(hass).values()
    assert retained.issue_id == issue_id
    assert retained.dismissed_version == original.dismissed_version
    assert retained.data["subentry_id"] == "vision"


async def test_configuration_repair_distinguishes_legacy_aliases_by_model(hass):
    first = text_service(model="different-model")
    first["data"]["unique_id"] = "legacy"
    second = {**text_service(), "subentry_id": "second"}
    second["data"]["unique_id"] = "legacy"
    entry = account(hass, services=[first, second])
    issue_id = _hashed_issue_id(
        "model_configuration", entry.entry_id, "legacy", TEXT_MODEL, "reasoning"
    )
    ir.async_create_issue(
        hass,
        "groq",
        issue_id,
        config_entry_id=entry.entry_id,
        data={
            "entry_id": entry.entry_id,
            "service_id": "legacy",
            "model": TEXT_MODEL,
            "feature": "reasoning",
        },
        is_fixable=False,
        is_persistent=True,
        severity=ir.IssueSeverity.ERROR,
        translation_key="model_configuration",
    )
    ir.async_ignore_issue(hass, "groq", issue_id, True)
    original = own_issues(hass)[("groq", issue_id)]
    async_reconcile_entry_issues(hass, entry)
    service = next(
        data
        for data in service_data_for_type(entry, "text_generation")
        if data["subentry_id"] == "second"
    )
    async_create_model_configuration_issue(
        hass, entry, service, TEXT_MODEL, "reasoning"
    )
    (retained,) = own_issues(hass).values()
    assert retained.issue_id == issue_id
    assert retained.dismissed_version == original.dismissed_version
    assert retained.data["subentry_id"] == "second"


async def test_old_account_credentials_do_not_start_reauthentication(hass):
    """An obsolete request still fails without starting a new reauth flow."""
    entry = account(hass)
    client = build_runtime(hass, entry).client
    with patch.object(client, "_auth_failure_callback") as reauth:
        assert isinstance(client._authentication_failed(), ConfigEntryAuthFailed)
        reauth.assert_called_once_with()
        reauth.reset_mock()
        hass.config_entries.async_update_entry(
            entry, data={**entry.data, "api_key": "new-key"}
        )
        assert isinstance(client._authentication_failed(), ConfigEntryAuthFailed)
        reauth.assert_not_called()


async def test_model_configuration_repair_recovers_and_tracks_service_changes(hass):
    entry = account(hass, services=[text_service()])
    service = service_data_for_type(entry, "text_generation")[0]
    async_create_model_configuration_issue(
        hass, entry, service, TEXT_MODEL, "text_generation"
    )
    assert len(own_issues(hass)) == 1
    async_delete_model_configuration_issue(
        hass, entry, service, TEXT_MODEL, "text_generation"
    )
    assert not own_issues(hass)
    async_create_model_configuration_issue(
        hass, entry, service, TEXT_MODEL, "text_generation"
    )
    async_reconcile_entry_issues(hass, entry, model_registry=GroqModelRegistry())
    assert not own_issues(hass)
    async_create_model_configuration_issue(
        hass, entry, service, TEXT_MODEL, "text_generation"
    )
    hass.config_entries.async_update_subentry(
        entry,
        entry.subentries["text-service"],
        data={**service, "model": "replacement"},
    )
    async_reconcile_entry_issues(hass, entry)
    assert not own_issues(hass)


@pytest.mark.parametrize("supports_model", [False, True])
async def test_action_with_stale_runtime_does_not_mutate_model_repair(
    hass, supports_model
):
    """A pending reload can leave service actions using the former runtime."""
    entry = account(hass, services=[text_service()])
    runtime = build_runtime(hass, entry)
    service = entry.subentries["text-service"]
    service_data = service_data_for_type(entry, "text_generation")[0]
    hass.config_entries.async_update_subentry(
        entry, service, data={**service.data, "temperature": 0.1}
    )
    if supports_model:
        async_create_model_configuration_issue(
            hass, entry, service_data, TEXT_MODEL, "text_generation"
        )
    expected = own_issues(hass)
    with patch.object(runtime.model_registry, "supports", return_value=supports_model):
        if supports_model:
            _ensure_model(
                runtime,
                TEXT_MODEL,
                GroqFeature.TEXT_GENERATION,
                hass=hass,
                entry=entry,
                service_data=service_data,
            )
        else:
            with pytest.raises(HomeAssistantError):
                _ensure_model(
                    runtime,
                    TEXT_MODEL,
                    GroqFeature.TEXT_GENERATION,
                    hass=hass,
                    entry=entry,
                    service_data=service_data,
                )

    assert own_issues(hass) == expected


async def test_repair_removal_preserves_unrelated_issues(hass):
    entry = account(hass, services=[text_service()])
    account(hass, entry_id="other-account")
    service = service_data_for_type(entry, "text_generation")[0]
    async_create_model_configuration_issue(
        hass, entry, service, TEXT_MODEL, "text_generation"
    )
    async_create_model_access_issue(hass, TEXT_MODEL, entry_id=entry.entry_id)
    async_create_model_access_issue(hass, TEXT_MODEL, entry_id="other-account")
    ir.async_create_issue(
        hass,
        "other",
        "anything",
        is_fixable=False,
        severity=ir.IssueSeverity.WARNING,
        translation_key="anything",
    )
    hass.config_entries.async_remove_subentry(entry, "text-service")
    async_reconcile_entry_issues(hass, entry)
    assert [issue.data["entry_id"] for issue in own_issues(hass).values()] == [
        "other-account"
    ]
    assert ("other", "anything") in ir.async_get(hass).issues


async def test_real_account_setup_unload_reload_and_removal(hass):
    entry = account(hass)
    with patch(
        "custom_components.groq.api.GroqApiClient.async_list_models",
        return_value=list(BUILT_IN_MODELS.values()),
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        assert entry.state is ConfigEntryState.LOADED
        first_runtime = entry.runtime_data
        assert not er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
        assert hass.services.has_service("groq", "generate_text")
        assert await hass.config_entries.async_reload(entry.entry_id)
        await hass.async_block_till_done()
        assert entry.runtime_data is not first_runtime
        async_create_model_access_issue(hass, TEXT_MODEL, entry_id=entry.entry_id)
        assert await hass.config_entries.async_remove(entry.entry_id)
        await hass.async_block_till_done()
        assert not own_issues(hass)


@pytest.mark.parametrize("legacy", [False, True])
async def test_reconcile_legacy_and_invalid_persistent_issue_metadata(
    hass, monkeypatch, legacy
):
    if legacy:
        monkeypatch.delattr(ir, "async_refresh_issues")
    entry = account(hass, services=[text_service()])
    registry = ir.async_get(hass)
    for issue_id, metadata in (
        ("model_access_legacy", None),
        (
            "model_configuration_invalid",
            {
                "entry_id": entry.entry_id,
                "model": TEXT_MODEL,
                "feature": "retired-feature",
            },
        ),
    ):
        ir.async_create_issue(
            hass,
            "groq",
            issue_id,
            is_fixable=False,
            is_persistent=True,
            severity=ir.IssueSeverity.ERROR,
            translation_key="model_access",
            data=metadata,
        )
    async_reconcile_entry_issues(hass, entry, model_registry=GroqModelRegistry())
    assert not own_issues(hass)
    assert not registry.issues


async def test_untyped_legacy_subentry_does_not_enable_features(hass):
    service = text_service()
    service["data"] = {"name": "Incomplete legacy entry"}
    entry = account(hass, services=[service])
    assert not service_data_for_type(entry, "text_generation")
    assert enabled_features_from_entry(entry) == []


async def test_loaded_subentry_update_reloads_once_and_retains_ownership(hass):
    entry = account(hass, services=[text_service()])
    with patch(
        "custom_components.groq.api.GroqApiClient.async_list_models",
        return_value=list(BUILT_IN_MODELS.values()),
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        registry = er.async_get(hass)
        initial = {
            entity.entity_id
            for entity in er.async_entries_for_config_entry(registry, entry.entry_id)
        }
        assert len(initial) == 9
        assert all(
            entity.config_subentry_id == "text-service"
            for entity in er.async_entries_for_config_entry(registry, entry.entry_id)
        )
        with patch.object(
            hass.config_entries, "async_reload", wraps=hass.config_entries.async_reload
        ) as reload:
            service = entry.subentries["text-service"]
            hass.config_entries.async_update_subentry(
                entry, service, data={**service.data, "temperature": 0.4}
            )
            await hass.async_block_till_done()
        reload.assert_awaited_once_with(entry.entry_id)
        assert {
            entity.entity_id
            for entity in er.async_entries_for_config_entry(registry, entry.entry_id)
        } == initial
        assert len(entry.update_listeners) == 1
        assert await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()
        assert not entry.update_listeners


@pytest.mark.parametrize(
    "feature", ["speech_to_text", "text_to_speech", "image_recognition"]
)
async def test_real_subentry_flow_saves_valid_default_schema(hass, feature):
    entry = account(hass)
    with patch(
        "custom_components.groq.config_flow.async_get_model_registry",
        return_value=GroqModelRegistry(),
    ):
        result = await hass.config_entries.subentries.async_init(
            (entry.entry_id, feature), context={"source": "user"}
        )
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"], {}
        )
    assert result["type"] == "create_entry"
    service = next(iter(entry.subentries.values()))
    assert service.data["service_type"] == feature
    assert service.data["model"]


@pytest.mark.parametrize("status", [200, 403])
async def test_repair_registry_failure_preserves_inference_outcome(
    hass, monkeypatch, caplog, expected_repair_errors, status
):
    """Repair reporting keeps its exception visible without replacing inference."""
    failure = OSError("injected registry storage failure")
    expected_repair_errors.append(failure)

    def unavailable_registry(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(
        ir,
        "async_delete_issue" if status == 200 else "async_create_issue",
        unavailable_registry,
    )
    payload = (
        {"choices": [{"message": {"content": "Available"}}]}
        if status == 200
        else {"error": {"code": "model_not_found", "message": "Model not found"}}
    )
    client = GroqApiClient(
        hass,
        api_key="test-key",
        session=DummySession(DummyResponse(status, {}, payload)),
    )
    request = TextGenerationRequest(prompt="Hello", model=TEXT_MODEL)
    if status == 200:
        assert (await client.async_generate_text(request)).text == "Available"
    else:
        with pytest.raises(GroqApiError) as error:
            await client.async_generate_text(request)
        assert error.value.status == 403
    records = [
        record
        for record in caplog.records
        if record.name == "custom_components.groq.repairs"
    ]
    assert len(records) == 1
    assert records[0].getMessage() == "Could not update the Groq repair registry"
    assert records[0].exc_info[1] is failure


async def test_repair_refresh_adopts_ignored_legacy_issue_in_place(hass, monkeypatch):
    """Check native call arguments even on the currently supported Core minimum."""
    entry = account(hass, services=[text_service()])
    # Seed an issue created by the pre-2026.11 integration.
    with monkeypatch.context() as old_core:
        # Absence, rather than a None value, selects the legacy create API.
        old_core.delattr(ir, "async_refresh_issues", raising=False)
        async_create_model_access_issue(hass, TEXT_MODEL, entry_id=entry.entry_id)
    registry = ir.async_get(hass)
    issue_id = next(iter(own_issues(hass)))[1]
    ir.async_ignore_issue(hass, "groq", issue_id, True)
    original = registry.async_get_issue("groq", issue_id)
    create_issue = ir.async_create_issue
    native_ownership = []

    def create_owned_issue(*args, **kwargs):
        native_ownership.append(kwargs["config_entry_id"])
        # On 2026.10, mock just the new keyword boundary, retaining the real
        # registry's create/update/ignore behavior. Native tests below use it as-is.
        if not hasattr(original, "config_entry_id"):
            kwargs.pop("config_entry_id")
        return create_issue(*args, **kwargs)

    refresh = MagicMock()
    monkeypatch.setattr(ir, "async_refresh_issues", refresh, raising=False)
    monkeypatch.setattr(ir, "async_create_issue", create_owned_issue)
    async_reconcile_entry_issues(hass, entry)
    refresh.assert_called_once_with(hass, "groq", issue_ids={issue_id: original})
    assert native_ownership == [entry.entry_id]
    retained = registry.async_get_issue("groq", issue_id)
    assert retained.created == original.created
    assert retained.dismissed_version == original.dismissed_version
    assert retained.data == original.data
    assert retained.translation_placeholders == original.translation_placeholders


@pytest.mark.parametrize("supports_subentry", [False, True])
async def test_native_repair_creation_links_every_entry_issue(
    hass, monkeypatch, supports_subentry
):
    """Exercise both Core ownership APIs and resolve legacy service identifiers."""
    stored_service = text_service()
    stored_service["data"]["unique_id"] = "legacy-text-service"
    entry = account(hass, services=[stored_service, voice_service()])
    service = service_data_for_type(entry, "text_generation")[0]
    voice = service_data_for_type(entry, "text_to_speech")[0]
    create = Mock()
    if supports_subentry:
        monkeypatch.setattr(ir.IssueEntry, "config_subentry_id", None, raising=False)
    else:
        monkeypatch.delattr(ir.IssueEntry, "config_subentry_id", raising=False)
    monkeypatch.setattr(ir, "async_refresh_issues", MagicMock(), raising=False)
    monkeypatch.setattr(ir, "async_create_issue", create)
    async_create_ffmpeg_missing_issue(hass, entry, voice)
    async_create_model_access_issue(
        hass, TEXT_MODEL, "text-service", entry_id=entry.entry_id
    )
    async_create_model_configuration_issue(
        hass, entry, service, TEXT_MODEL, "text_generation"
    )
    async_create_model_access_issue(hass, TEXT_MODEL, entry_id=entry.entry_id)
    async_create_ffmpeg_missing_issue(hass, entry, {"unique_id": "removed-service"})
    async_create_model_access_issue(
        hass,
        TEXT_MODEL,
        "text-service",
        entry_id="removed-account",
        subentry_id="text-service",
    )
    service_issues, account_issues = (
        create.call_args_list[:3],
        create.call_args_list[3:],
    )
    assert len(service_issues) == 3
    assert len(account_issues) == 1
    assert all(
        call.kwargs["config_entry_id"] == entry.entry_id
        for call in create.call_args_list
    )
    for call, owner in zip(
        service_issues, ("voice", "text-service", "text-service"), strict=True
    ):
        if supports_subentry:
            assert call.kwargs["config_subentry_id"] == owner
        else:
            assert "config_subentry_id" not in call.kwargs
    assert all("config_subentry_id" not in call.kwargs for call in account_issues)


@pytest.mark.parametrize("after_removal", ["reconcile", "failure", "success"])
async def test_ambiguous_legacy_service_id_cannot_pass_its_dismissal_to_another_service(
    hass, after_removal
):
    """The shared issue ID cannot prove which former service was dismissed."""
    first = text_service()
    first["data"]["unique_id"] = "other-service"
    second = {**text_service(), "subentry_id": "other-service"}
    entry = account(hass, services=[first, second])
    key = (
        "groq",
        _hashed_issue_id("model_access", entry.entry_id, TEXT_MODEL, "other-service"),
    )
    ir.async_create_issue(
        hass,
        *key,
        config_entry_id=entry.entry_id,
        data={
            "entry_id": entry.entry_id,
            "service_id": "other-service",
            "model": TEXT_MODEL,
        },
        is_fixable=False,
        is_persistent=True,
        severity=ir.IssueSeverity.ERROR,
        translation_key="model_access",
    )
    ir.async_ignore_issue(hass, *key, True)
    original = own_issues(hass)[key]
    async_create_model_access_issue(
        hass, TEXT_MODEL, "other-service", entry_id=entry.entry_id
    )
    assert own_issues(hass) == {key: original}

    async_reconcile_entry_issues(hass, entry)
    assert own_issues(hass)[key].dismissed_version == original.dismissed_version
    hass.config_entries.async_remove_subentry(entry, "other-service")
    if after_removal == "reconcile":
        async_reconcile_entry_issues(hass, entry)
    elif after_removal == "success":
        async_delete_model_access_issue(
            hass, TEXT_MODEL, "other-service", entry_id=entry.entry_id
        )
    else:
        async_create_model_access_issue(
            hass, TEXT_MODEL, "other-service", entry_id=entry.entry_id
        )
        (current_issue,) = own_issues(hass).values()
        assert current_issue.dismissed_version is None
        assert (
            getattr(current_issue, "config_subentry_id", "text-service")
            == "text-service"
        )
        return
    assert not own_issues(hass)


async def test_reconcile_drops_repair_when_owned_service_loses_legacy_id(hass):
    """The service's new requests use its canonical ID and cannot clear the old key."""
    stored_service = text_service()
    stored_service["data"]["unique_id"] = "old-id"
    entry = account(hass, services=[stored_service])
    ir.async_create_issue(
        hass,
        "groq",
        _hashed_issue_id("model_access", entry.entry_id, TEXT_MODEL, "old-id"),
        config_entry_id=entry.entry_id,
        data={
            "entry_id": entry.entry_id,
            "service_id": "old-id",
            "subentry_id": "text-service",
            "model": TEXT_MODEL,
        },
        is_fixable=False,
        is_persistent=True,
        severity=ir.IssueSeverity.ERROR,
        translation_key="model_access",
    )
    assert len(own_issues(hass)) == 1
    service = entry.subentries["text-service"]
    hass.config_entries.async_update_subentry(
        entry, service, data=text_service()["data"]
    )

    async_reconcile_entry_issues(hass, entry)

    assert not own_issues(hass)


@pytest.mark.skipif(
    not hasattr(ir.IssueEntry, "config_subentry_id"),
    reason="Requires Core 2026.12 issue API",
)
async def test_existing_service_repair_adopts_subentry_and_keeps_dismissal(
    hass, monkeypatch
):
    """Promote a 2026.11 repair in place and let Core clear it on service removal."""
    entry = account(hass, services=[text_service()])
    create_issue = ir.async_create_issue

    def create_account_owned_issue(*args, **kwargs):
        kwargs.pop("config_subentry_id", None)
        create_issue(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(ir, "async_create_issue", create_account_owned_issue)
        async_create_model_access_issue(
            hass, TEXT_MODEL, "text-service", entry_id=entry.entry_id
        )
    async_create_model_access_issue(hass, TEXT_MODEL, entry_id=entry.entry_id)
    key, original = next(
        (key, issue)
        for key, issue in own_issues(hass).items()
        if issue.data["service_id"]
    )
    assert original.config_subentry_id is None
    ir.async_ignore_issue(hass, *key, True)
    dismissed = own_issues(hass)[key]

    async_reconcile_entry_issues(hass, entry)

    adopted = own_issues(hass)[key]
    assert adopted.config_subentry_id == "text-service"
    assert adopted.dismissed_version == dismissed.dismissed_version
    assert adopted.created == original.created
    assert adopted.data == original.data

    hass.config_entries.async_remove_subentry(entry, "text-service")
    assert key not in own_issues(hass)
    (unrelated,) = own_issues(hass).values()
    assert unrelated.config_entry_id == entry.entry_id
    assert unrelated.config_subentry_id is None


async def test_sensor_rename_calls_parent_without_resubscribing(hass):
    """Opt-in delegates to Core while retaining the original service listener."""
    from homeassistant.components.sensor import SensorEntity

    from custom_components.groq.sensor import GroqUsageSensor, SENSORS
    from custom_components.groq.usage import GroqUsage

    usage = GroqUsage()
    entity = GroqUsageSensor(
        {"unique_id": "service", "name": "Service", "model": TEXT_MODEL},
        usage,
        SENSORS[0],
    )
    entity.hass = hass
    entity.entity_id = "sensor.renamed"
    with (
        patch.object(SensorEntity, "async_entity_id_changed", create=True) as parent,
        patch.object(usage, "subscribe", wraps=usage.subscribe) as subscribe,
    ):
        await entity.async_added_to_hass()
        entity.async_entity_id_changed("sensor.original")
        parent.assert_called_once_with("sensor.original")
        subscribe.assert_called_once()
        assert len(usage._listeners) == 1


@pytest.mark.parametrize("legacy", [False, True])
async def test_ignored_repairs_survive_reload_and_clear_only_after_recovery(
    hass, monkeypatch, legacy
):
    """Neither reload nor configuration reconciliation rechecks failed requests."""
    if legacy:
        monkeypatch.delattr(ir, "async_refresh_issues", raising=False)
    entry = account(
        hass,
        services=[text_service()],
        data={
            "name": "Legacy Groq",
            "url": "https://api.groq.com",
            "model": TTS_MODEL,
            "voice": "troy",
            "api_key": "test-key",
        },
    )
    service = service_data_for_type(entry, "text_generation")[0]
    with patch(
        "custom_components.groq.api.GroqApiClient.async_list_models",
        return_value=list(BUILT_IN_MODELS.values()),
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        async_create_model_access_issue(
            hass, TEXT_MODEL, service["unique_id"], entry_id=entry.entry_id
        )
        async_create_ffmpeg_missing_issue(hass, entry)
        registry = ir.async_get(hass)
        for _, issue_id in own_issues(hass):
            ir.async_ignore_issue(hass, "groq", issue_id, True)
        originals = own_issues(hass)
        assert await hass.config_entries.async_reload(entry.entry_id)
        await hass.async_block_till_done()
        assert own_issues(hass) == originals
        for key, original in originals.items():
            assert registry.issues[key].dismissed_version == original.dismissed_version
        async_delete_model_access_issue(
            hass, TEXT_MODEL, service["unique_id"], entry_id=entry.entry_id
        )
        assert len(own_issues(hass)) == 1
        async_delete_ffmpeg_missing_issue(hass, entry)
        assert not own_issues(hass)
        assert await hass.config_entries.async_unload(entry.entry_id)


@pytest.mark.skipif(
    not hasattr(ir, "async_refresh_issues"), reason="Requires Core 2026.11 issue API"
)
async def test_native_repair_refresh_and_config_entry_ownership(hass):
    """Exercise actual Core refresh, ownership, disabled visibility and removal."""
    first = account(hass, services=[text_service(), voice_service()])
    second = account(hass, entry_id="account-b")
    service = service_data_for_type(first, "text_generation")[0]
    async_create_model_access_issue(hass, TEXT_MODEL, entry_id=second.entry_id)
    async_create_model_access_issue(
        hass, TEXT_MODEL, service["unique_id"], entry_id=first.entry_id
    )
    async_create_ffmpeg_missing_issue(
        hass, first, service_data_for_type(first, "text_to_speech")[0]
    )
    registry = ir.async_get(hass)
    for issue in own_issues(hass).values():
        assert issue.config_entry_id == issue.data["entry_id"]
        ir.async_ignore_issue(hass, "groq", issue.issue_id, True)
    originals = own_issues(hass)
    async_reconcile_entry_issues(hass, first)
    assert own_issues(hass) == originals
    await hass.config_entries.async_set_disabled_by(
        first.entry_id, ConfigEntryDisabler.USER
    )
    first_issues = [
        i for i in own_issues(hass).values() if i.config_entry_id == first.entry_id
    ]
    assert all(registry.async_is_hidden(i) for i in first_issues)
    # Visibility belongs to the registry; enabling must not contact the provider.
    with patch.object(hass.config_entries, "async_setup", return_value=True):
        await hass.config_entries.async_set_disabled_by(first.entry_id, None)
        await hass.async_block_till_done()
    assert all(not registry.async_is_hidden(i) for i in first_issues)
    hass.config_entries.async_remove_subentry(first, "text-service")
    hass.config_entries.async_remove_subentry(first, "voice")
    async_reconcile_entry_issues(hass, first)
    remaining = list(own_issues(hass).values())
    assert len(remaining) == 1
    assert remaining[0].config_entry_id == second.entry_id
    # Prove native removal handles issues even without Groq's legacy cleanup hook.
    with patch("custom_components.groq.async_remove_entry"):
        await hass.config_entries.async_remove(second.entry_id)
    await hass.async_block_till_done()
    assert not own_issues(hass)


@pytest.mark.parametrize("emulate_native_ownership", [False, True])
async def test_startup_adopts_legacy_repairs_before_account_setup(
    hass, monkeypatch, emulate_native_ownership
):
    native = hasattr(ir, "async_refresh_issues")
    if emulate_native_ownership == native:
        pytest.skip("Tests the native API or the compatibility test double")
    disabled = account(
        hass, services=[text_service()], disabled_by=ConfigEntryDisabler.USER
    )
    enabled = account(hass, entry_id="account-b", services=[text_service()])
    with monkeypatch.context() as old_core:
        old_core.delattr(ir, "async_refresh_issues", raising=False)
        for entry in (disabled, enabled):
            service = service_data_for_type(entry, "text_generation")[0]
            async_create_model_access_issue(
                hass, TEXT_MODEL, service["unique_id"], entry_id=entry.entry_id
            )
    registry = ir.async_get(hass)
    originals = own_issues(hass)
    assert len(originals) == 2
    if emulate_native_ownership:
        monkeypatch.setattr(ir, "async_refresh_issues", MagicMock(), raising=False)
    with (
        patch.object(
            ir, "async_create_issue", wraps=ir.async_create_issue if native else None
        ) as create,
        patch.object(
            GroqApiClient, "async_list_models", side_effect=GroqApiError("offline")
        ) as list_models,
    ):
        assert await async_setup_component(hass, "groq", {})
        await hass.async_block_till_done()
        list_models.assert_awaited_once()
    assert disabled.state is ConfigEntryState.NOT_LOADED
    assert enabled.state is ConfigEntryState.SETUP_RETRY
    assert {call.kwargs["config_entry_id"] for call in create.call_args_list} == {
        disabled.entry_id,
        enabled.entry_id,
    }
    if native:
        for key, original in originals.items():
            issue = registry.issues[key]
            assert issue.config_entry_id == original.data["entry_id"]
            assert issue.created == original.created
            assert registry.async_is_hidden(issue) == (
                issue.config_entry_id == disabled.entry_id
            )


async def test_startup_clears_stored_repairs_for_removed_services_while_offline(hass):
    """Repair reconciliation uses local configuration before model discovery."""
    entry = account(hass)
    ir.async_create_issue(
        hass,
        "groq",
        "ffmpeg_missing_removed_service",
        config_entry_id=entry.entry_id,
        data={"entry_id": entry.entry_id, "service_id": "removed-service"},
        is_fixable=False,
        is_persistent=True,
        severity=ir.IssueSeverity.ERROR,
        translation_key="ffmpeg_missing",
    )
    async_create_model_access_issue(hass, TEXT_MODEL, entry_id=entry.entry_id)
    for domain, issue_id in (("other", "unrelated"), ("groq", "model_access_unowned")):
        ir.async_create_issue(
            hass,
            domain,
            issue_id,
            is_fixable=False,
            severity=ir.IssueSeverity.WARNING,
            translation_key="unrelated",
        )

    with patch.object(
        GroqApiClient, "async_list_models", side_effect=GroqApiError("offline")
    ):
        assert await async_setup_component(hass, "groq", {})
        await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.SETUP_RETRY
    assert sorted(issue.translation_key for issue in own_issues(hass).values()) == [
        "model_access",
        "unrelated",
    ]
    assert ir.async_get(hass).async_get_issue("other", "unrelated") is not None


async def test_usage_sensor_registry_rename_keeps_subscription_until_unload(hass):
    """Rename through HA's registry, then observe real pushed states and cleanup."""
    entry = account(hass, services=[text_service()])
    with patch(
        "custom_components.groq.api.GroqApiClient.async_list_models",
        return_value=list(BUILT_IN_MODELS.values()),
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        registry = er.async_get(hass)
        sensor_entry = next(
            item
            for item in er.async_entries_for_config_entry(registry, entry.entry_id)
            if item.domain == "sensor" and item.unique_id.endswith("_requests")
        )
        registry.async_update_entity(sensor_entry.entity_id, disabled_by=None)
        await hass.async_block_till_done()
        assert await hass.config_entries.async_reload(entry.entry_id)
        await hass.async_block_till_done()
        usage = entry.runtime_data.client.usage
        component = hass.data["sensor"]
        entity = component.get_entity(sensor_entry.entity_id)
        assert entity is not None
        assert len(usage._listeners) == 1
        service_id = service_data_for_type(entry, "text_generation")[0]["unique_id"]
        usage.record(service_id, {})
        assert hass.states.get(entity.entity_id).state == "1"
        with (
            patch.object(usage, "subscribe", wraps=usage.subscribe) as subscribe,
            patch.object(
                entity, "async_added_to_hass", wraps=entity.async_added_to_hass
            ) as added,
            patch.object(
                entity,
                "async_will_remove_from_hass",
                wraps=entity.async_will_remove_from_hass,
            ) as removed,
        ):
            for new_id in ("sensor.groq_renamed_once", "sensor.groq_renamed_twice"):
                old_id = entity.entity_id
                registry.async_update_entity(old_id, new_entity_id=new_id)
                await hass.async_block_till_done()
                assert component.get_entity(new_id) is entity
                assert hass.states.get(old_id) is None
                assert len(usage._listeners) == 1
                assert hass.states.get(new_id).state == "1"
            # Core 2026.10 uses remove/add; Core 2026.11 must preserve the listener.
            from homeassistant.helpers.entity import Entity

            if hasattr(Entity, "async_entity_id_changed"):
                subscribe.assert_not_called()
                added.assert_not_awaited()
                removed.assert_not_awaited()
        with patch.object(
            entity, "async_write_ha_state", wraps=entity.async_write_ha_state
        ) as write:
            usage.record(service_id, {})
            write.assert_called_once()
        assert hass.states.get(entity.entity_id).state == "2"
        assert await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()
        assert not usage._listeners
        assert not usage.values
        with patch.object(entity, "async_write_ha_state") as write:
            usage.record(service_id, {})
            write.assert_not_called()


@pytest.mark.parametrize("native_boundary", [False, True])
async def test_inactive_unowned_ignored_repair_keeps_dismissal(
    hass, monkeypatch, native_boundary
):
    """Core's stored dismissal can lack all the old issue metadata."""
    from dataclasses import replace

    entry = account(hass, services=[text_service()])
    async_create_model_access_issue(hass, TEXT_MODEL, entry_id=entry.entry_id)
    registry = ir.async_get(hass)
    issue_id = next(iter(own_issues(hass)))[1]
    ir.async_ignore_issue(hass, "groq", issue_id, True)
    original = registry.async_get_issue("groq", issue_id)
    values = {
        "active": False,
        "data": None,
        "is_fixable": None,
        "severity": None,
        "translation_key": None,
    }
    if hasattr(original, "config_entry_id"):
        values["config_entry_id"] = None
    registry.issues[("groq", issue_id)] = inactive = replace(original, **values)
    with monkeypatch.context() as api:
        refresh = MagicMock()
        if native_boundary:
            api.setattr(ir, "async_refresh_issues", refresh, raising=False)
        else:
            api.delattr(ir, "async_refresh_issues", raising=False)
        async_reconcile_entry_issues(hass, entry)
        assert registry.async_get_issue("groq", issue_id) is inactive
        if native_boundary:
            refresh.return_value.__enter__.return_value.async_keep.assert_called_once_with(
                [issue_id]
            )
    async_create_model_access_issue(hass, TEXT_MODEL, entry_id=entry.entry_id)
    recreated = registry.async_get_issue("groq", issue_id)
    assert recreated.active
    assert recreated.dismissed_version == original.dismissed_version


@pytest.mark.parametrize(
    "incomplete",
    [
        {"active": False},
        {"is_fixable": None},
        {"severity": None},
        {"translation_key": None},
    ],
)
async def test_incomplete_legacy_repair_waits_for_real_check(
    hass, monkeypatch, incomplete
):
    """Do not turn an inactive or incomplete record into an asserted failure."""
    from dataclasses import replace

    entry = account(hass, services=[text_service()])
    async_create_model_access_issue(hass, TEXT_MODEL, entry_id=entry.entry_id)
    registry = ir.async_get(hass)
    issue_id = next(iter(own_issues(hass)))[1]
    original = registry.async_get_issue("groq", issue_id)
    if hasattr(original, "config_entry_id"):
        incomplete = {**incomplete, "config_entry_id": None}
    registry.issues[("groq", issue_id)] = original = replace(original, **incomplete)
    refresh = MagicMock()
    monkeypatch.setattr(ir, "async_refresh_issues", refresh, raising=False)
    create = Mock()
    monkeypatch.setattr(ir, "async_create_issue", create)
    async_reconcile_entry_issues(hass, entry)
    create.assert_not_called()
    refresh.return_value.__enter__.return_value.async_keep.assert_called_once_with(
        [issue_id]
    )
    assert registry.async_get_issue("groq", issue_id) is original


@pytest.mark.skipif(
    not hasattr(ir, "async_refresh_issues"), reason="Requires Core 2026.11 issue API"
)
async def test_native_refresh_cancels_stale_deletes_if_reconciliation_fails(
    hass, expected_repair_errors
):
    """A partially checked issue set must survive an unexpected registry failure."""
    entry = account(hass, services=[text_service()])
    ir.async_create_issue(
        hass,
        "groq",
        "ffmpeg_missing_removed_service",
        config_entry_id=entry.entry_id,
        data={"entry_id": entry.entry_id, "service_id": "removed-service"},
        is_fixable=False,
        is_persistent=True,
        severity=ir.IssueSeverity.ERROR,
        translation_key="ffmpeg_missing",
    )
    service = service_data_for_type(entry, "text_generation")[0]
    async_create_model_configuration_issue(
        hass, entry, service, TEXT_MODEL, "text_generation"
    )
    originals = own_issues(hass)
    failure = OSError("injected capability reconciliation failure")
    expected_repair_errors.append(failure)
    model_registry = GroqModelRegistry()
    with patch.object(model_registry, "supports", side_effect=failure):
        async_reconcile_entry_issues(hass, entry, model_registry=model_registry)
    assert own_issues(hass) == originals
    assert not ir.async_get(hass)._refreshes
    async_reconcile_entry_issues(hass, entry, model_registry=model_registry)
    assert not own_issues(hass)
