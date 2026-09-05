"""Regression tests across real Home Assistant configuration and registry boundaries."""

from types import MappingProxyType
from unittest.mock import patch

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.data_entry_flow import InvalidData
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import issue_registry as ir
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.groq import async_migrate_entry
from custom_components.groq.config_flow import (
    GroqServiceSubentryFlow,
    async_get_model_registry,
)
from custom_components.groq.const import enabled_features_from_entry
from custom_components.groq.diagnostics import async_get_config_entry_diagnostics
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


async def test_real_entry_mapping_diagnostics(hass):
    entry = account(hass, services=[text_service()])
    assert isinstance(entry.data, MappingProxyType)
    assert isinstance(entry.subentries["text-service"].data, MappingProxyType)
    assert enabled_features_from_entry(entry) == ["text_generation"]
    diagnostics = await async_get_config_entry_diagnostics(hass, entry)
    assert diagnostics["summary"]["enabled_features"] == ["text_generation"]
    assert diagnostics["summary"]["service_counts"] == {"text_generation": 1}


async def test_legacy_mapping_migration_and_version_guard(hass):
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
    future = MockConfigEntry(domain="groq", version=2)
    assert not await async_migrate_entry(hass, future)


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
        assert len(initial) == 2
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
