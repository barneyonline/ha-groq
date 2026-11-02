"""Regression tests across real Home Assistant configuration and registry boundaries."""

from copy import deepcopy
from datetime import timedelta
from types import MappingProxyType
from unittest.mock import MagicMock, Mock, patch

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.data_entry_flow import InvalidData
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import issue_registry as ir
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import (
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
    async_create_ffmpeg_missing_issue,
    async_create_model_access_issue,
    async_create_model_configuration_issue,
    async_delete_model_access_issue,
    async_delete_model_configuration_issue,
    async_reconcile_entry_issues,
)
from custom_components.groq.subentries import service_data_for_type

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


def account(hass, *, entry_id="account-a", services=(), options=None, data=None):
    """Register a real HA entry with immutable service data."""
    entry = MockConfigEntry(
        domain="groq",
        entry_id=entry_id,
        unique_id=entry_id,
        minor_version=2,
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


def own_issues(hass):
    return {
        key: value
        for key, value in ir.async_get(hass).issues.items()
        if key[0] == "groq"
    }


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
            assert not await async_unload_entry(hass, entry)
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
    assert await async_migrate_entry(hass, entry)
    assert entry.unique_id == "stable-id"
    assert entry.minor_version == 2
    assert dict(entry.data) == {"api_key": "test-key"}
    assert await async_migrate_entry(hass, entry)


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
        assert await async_migrate_entry(hass, entry)
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


async def test_real_structured_output_fallback_repeated_saves(hass):
    """Validate defaults and persistence through the older HA flow manager."""
    from homeassistant import data_entry_flow

    if hasattr(data_entry_flow, "Optional"):
        pytest.skip("This regression exercises HA without conditional field support")

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
            {"structured_outputs": False},
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
            assert all(
                getattr(key, "visible", None) is None for key in markers.values()
            )
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
                {key: value for key, value in submitted.items() if key in schema_fields}
            )
            stored = entry.subentries["text-service"].data
            assert stored["structured_outputs"] is enabled
            assert {key: stored[key] for key in schema_fields} == saved


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


async def test_repair_removal_preserves_unrelated_issues(hass):
    entry = account(hass, services=[text_service()])
    account(hass, entry_id="other-account")
    service = service_data_for_type(entry, "text_generation")[0]
    async_create_ffmpeg_missing_issue(hass, entry, service)
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


async def test_reconcile_legacy_and_invalid_persistent_issue_metadata(hass):
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


async def test_native_repair_creation_links_every_entry_issue(hass, monkeypatch):
    """Exercise the new keyword without requiring it in the minimum Core API."""
    entry = account(hass, services=[text_service()])
    service = service_data_for_type(entry, "text_generation")[0]
    create = Mock()
    monkeypatch.setattr(ir, "async_refresh_issues", MagicMock(), raising=False)
    monkeypatch.setattr(ir, "async_create_issue", create)
    async_create_ffmpeg_missing_issue(hass, entry, service)
    async_create_model_access_issue(hass, TEXT_MODEL, entry_id=entry.entry_id)
    async_create_model_configuration_issue(
        hass, entry, service, TEXT_MODEL, "text_generation"
    )
    assert len(create.call_args_list) == 3
    assert all(
        call.kwargs["config_entry_id"] == entry.entry_id
        for call in create.call_args_list
    )


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
    entry = account(hass, services=[text_service()])
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
        async_create_ffmpeg_missing_issue(hass, entry, service)
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
        from custom_components.groq.repairs import async_delete_ffmpeg_missing_issue

        async_delete_ffmpeg_missing_issue(hass, entry, service)
        assert not own_issues(hass)
        assert await hass.config_entries.async_unload(entry.entry_id)


@pytest.mark.skipif(
    not hasattr(ir, "async_refresh_issues"), reason="Requires Core 2026.11 issue API"
)
async def test_native_repair_refresh_and_config_entry_ownership(hass):
    """Exercise actual Core refresh, ownership, disabled visibility and removal."""
    from homeassistant.config_entries import ConfigEntryDisabler

    first = account(hass, services=[text_service()])
    second = account(hass, entry_id="account-b")
    service = service_data_for_type(first, "text_generation")[0]
    async_create_model_access_issue(hass, TEXT_MODEL, entry_id=second.entry_id)
    async_create_model_access_issue(
        hass, TEXT_MODEL, service["unique_id"], entry_id=first.entry_id
    )
    async_create_ffmpeg_missing_issue(hass, first, service)
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
    async_reconcile_entry_issues(hass, first)
    remaining = list(own_issues(hass).values())
    assert len(remaining) == 1
    assert remaining[0].config_entry_id == second.entry_id
    # Prove native removal handles issues even without Groq's legacy cleanup hook.
    with patch("custom_components.groq.async_remove_entry"):
        await hass.config_entries.async_remove(second.entry_id)
    await hass.async_block_till_done()
    assert not own_issues(hass)


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
    async_create_ffmpeg_missing_issue(hass, entry, {"unique_id": "removed-service"})
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
