"""Owned, recoverable repair issues for Groq configuration and model access."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from hashlib import sha1
from typing import Any

from homeassistant.components.repairs.models import RepairsFlow, RepairsFlowResult
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir

from .const import (
    CONF_MODEL,
    CONF_NAME,
    CONF_SERVICE_TYPE,
    CONF_SUBENTRY_ID,
    DOMAIN,
    FEATURE_TEXT_GENERATION,
    FEATURE_TEXT_TO_SPEECH,
    UNIQUE_ID,
    enabled_features_from_entry,
    entry_value,
)
from .feature_registry import GroqFeature
from .model_registry import GroqModelRegistry
from .subentries import (
    has_legacy_tts_config,
    service_data_by_type,
    service_data_for_type,
)
from .types import GroqConfigEntry

ISSUE_FFMPEG_MISSING = "ffmpeg_missing"
ISSUE_MODEL_ACCESS = "model_access"
ISSUE_MODEL_CONFIGURATION = "model_configuration"
_ISSUE_PREFIXES = tuple(
    f"{kind}_"
    for kind in (ISSUE_FFMPEG_MISSING, ISSUE_MODEL_ACCESS, ISSUE_MODEL_CONFIGURATION)
)
_AMBIGUOUS_SERVICE_OWNER = "ambiguous_service_owner"
_ACCOUNT_SCOPED = "account_scoped"
_CONFIGURATION_SERVICE_TYPES: dict[str, str] = {
    GroqFeature.TEXT_GENERATION: FEATURE_TEXT_GENERATION,
    GroqFeature.REASONING: FEATURE_TEXT_GENERATION,
    GroqFeature.STRUCTURED_OUTPUTS: FEATURE_TEXT_GENERATION,
    GroqFeature.SPEECH_TO_TEXT: GroqFeature.SPEECH_TO_TEXT.value,
    GroqFeature.VISION: GroqFeature.VISION.value,
    GroqFeature.OCR: GroqFeature.VISION.value,
}
_LOGGER = logging.getLogger(__name__)


class GroqRepairsFlow(RepairsFlow):
    """Repairs flow for non-fixable Groq issues."""

    async def async_step_init(
        self, user_input: dict[str, str] | None = None
    ) -> RepairsFlowResult:
        """Direct users to the resolution described by the repair issue."""
        return self.async_abort(reason="not_fixable")


async def async_create_fix_flow(
    _hass: HomeAssistant,
    _issue_id: str,
    _data: dict[str, str | int | float | None] | None,
) -> RepairsFlow:
    """Create a repair flow for a Groq issue."""
    return GroqRepairsFlow()


def _safe(value: Any, fallback: str = "unknown") -> str:
    """Return a bounded placeholder value."""
    return str(value or fallback)[:128]


def _hashed_issue_id(issue_type: str, *parts: Any) -> str:
    """Return a stable id without embedding user data in issue identifiers."""
    encoded = "|".join(str(part or "") for part in parts)
    return f"{issue_type}_{sha1(encoded.encode('utf-8')).hexdigest()[:12]}"


def _service_name(service_data: dict[str, Any] | None) -> str:
    """Return a bounded user-facing service label."""
    if not service_data:
        return "Groq"
    return _safe(service_data.get(CONF_NAME) or service_data.get(UNIQUE_ID), "Groq")


def _update_issue(operation: Callable[[], None]) -> None:
    """Keep optional repair reporting from breaking requests, with visible failures."""
    try:
        operation()
    except Exception:  # Repairs must not replace the original request outcome.
        _LOGGER.exception("Could not update the Groq repair registry")


def _services_by_identifier(
    entry: GroqConfigEntry,
) -> dict[str | None, list[dict[str, Any]]]:
    """Index configured services by subentry ID and any stored legacy ID."""
    services: dict[str | None, list[dict[str, Any]]] = {}
    for group in service_data_by_type(entry).values():
        for service in group:
            for identifier in _service_identifiers(entry, service):
                services.setdefault(identifier, []).append(service)
            if not service.get(UNIQUE_ID):
                services.setdefault(None, []).append(service)
    return services


def _service_identifiers(
    entry: GroqConfigEntry, service: dict[str, Any]
) -> tuple[str, ...]:
    """Return the physical ID and any historical ID used in requests."""
    legacy_id = service.get(UNIQUE_ID)
    if not legacy_id and service.get(CONF_SERVICE_TYPE) == FEATURE_TEXT_GENERATION:
        legacy_id = getattr(entry, "unique_id", None) or entry.entry_id
    return tuple(
        dict.fromkeys(
            identifier
            for identifier in (service.get(CONF_SUBENTRY_ID), legacy_id)
            if isinstance(identifier, str) and identifier
        )
    )


@dataclass(frozen=True, slots=True)
class _RepairKey:
    issue_type: str
    model: str | None = None
    feature: str | None = None

    def issue_id(self, entry_id: str | None, service_id: str | None) -> str:
        """Compute the original issue key for an account and service identifier."""
        if self.issue_type == ISSUE_FFMPEG_MISSING:
            return _hashed_issue_id(self.issue_type, entry_id, service_id)
        if self.issue_type == ISSUE_MODEL_ACCESS:
            return _hashed_issue_id(self.issue_type, entry_id, self.model, service_id)
        return _hashed_issue_id(
            self.issue_type, entry_id, service_id, self.model, self.feature
        )

    def matches(self, service: dict[str, Any], identifier: str | None) -> bool:
        """Check which service types could have produced a stored repair."""
        service_type = service.get(CONF_SERVICE_TYPE)
        if self.issue_type == ISSUE_FFMPEG_MISSING:
            return (
                service_type == FEATURE_TEXT_TO_SPEECH
                and service.get(CONF_SUBENTRY_ID) == identifier
            )
        if (
            self.issue_type == ISSUE_MODEL_ACCESS
            and service_type == FEATURE_TEXT_TO_SPEECH
        ):
            return service.get(CONF_SUBENTRY_ID) == identifier
        if self.issue_type == ISSUE_MODEL_CONFIGURATION and self.feature is not None:
            expected = _CONFIGURATION_SERVICE_TYPES.get(self.feature)
            return (expected is None or service_type == expected) and (
                self.model is None or service.get(CONF_MODEL) == self.model
            )
        return True


@dataclass(frozen=True, slots=True)
class _ServiceRepair:
    issue_id: str
    service_id: str | None
    subentry_id: str | None
    current_ids: set[str]
    obsolete_ids: set[str]


def _repair_candidates(
    entry: GroqConfigEntry,
    services: dict[str | None, list[dict[str, Any]]],
    key: _RepairKey,
    identifier: str | None,
    *,
    historical: bool = False,
) -> list[dict[str, Any]]:
    # A flat TTS entity could use any request model under the same blank key
    # as actions for old subentries missing a service ID.
    if (
        identifier is None
        and key.issue_type == ISSUE_MODEL_ACCESS
        and has_legacy_tts_config(entry)
    ):
        return []
    return [
        service
        for service in services.get(identifier, [])
        if key.matches(service, identifier)
        and (
            not historical
            or _used_historical_identifier(entry, key, service, identifier)
        )
    ]


def _used_historical_identifier(
    entry: GroqConfigEntry,
    key: _RepairKey,
    service: dict[str, Any],
    identifier: str | None,
) -> bool:
    """Identify keys a previous request could have produced without recording its owner."""
    service_type = service.get(CONF_SERVICE_TYPE)
    if key.issue_type == ISSUE_FFMPEG_MISSING or (
        key.issue_type == ISSUE_MODEL_ACCESS and service_type == FEATURE_TEXT_TO_SPEECH
    ):
        return identifier == service.get(CONF_SUBENTRY_ID)
    legacy_id = service.get(UNIQUE_ID) or None
    if identifier == legacy_id:
        return True
    if key.issue_type == ISSUE_MODEL_CONFIGURATION:
        return False
    if service_type == GroqFeature.SPEECH_TO_TEXT:
        return identifier == service.get(CONF_SUBENTRY_ID)
    return (
        service_type == FEATURE_TEXT_GENERATION
        and legacy_id is None
        and identifier == (getattr(entry, "unique_id", None) or entry.entry_id)
    )


def _issue_ownership(entry_id: str | None, subentry_id: str | None) -> dict[str, Any]:
    """Use native subentry ownership where Core supports it."""
    if not hasattr(ir, "async_refresh_issues"):
        return {}
    ownership = {"config_entry_id": entry_id}
    if (
        hasattr(ir.IssueEntry, "config_subentry_id")
        and entry_id is not None
        and subentry_id is not None
    ):
        ownership["config_subentry_id"] = subentry_id
    return ownership


def _current_service_repair(
    hass: HomeAssistant,
    key: _RepairKey,
    entry_id: str,
    service_id: str,
    subentry_id: str | None,
) -> _ServiceRepair | None:
    """Resolve current and historical keys for one configured service."""
    if (entry := hass.config_entries.async_get_entry(entry_id)) is None:
        return None
    services = _services_by_identifier(entry)
    request_id = subentry_id or service_id
    matches = [
        service
        for service in _repair_candidates(entry, services, key, request_id)
        if subentry_id is None or service.get(CONF_SUBENTRY_ID) == subentry_id
    ]
    if len(matches) != 1:
        return None

    service = matches[0]
    owner = service.get(CONF_SUBENTRY_ID)
    identifiers: list[str | None] = [
        identifier
        for identifier in _service_identifiers(entry, service)
        if key.matches(service, identifier)
    ]
    canonical_id = identifiers[0]
    if not service.get(UNIQUE_ID) and key.matches(service, None):
        identifiers.append(None)
    canonical_key = key.issue_id(entry_id, canonical_id)
    registry = ir.async_get(hass)
    current: dict[str, str | None] = {}
    obsolete: set[str] = set()
    canonical_shared = False

    for identifier in identifiers:
        possible = _repair_candidates(entry, services, key, identifier, historical=True)
        base = key.issue_id(entry_id, identifier)
        keys = (base, _hashed_issue_id(base, owner)) if owner is not None else (base,)
        for issue_id in keys:
            if (previous := registry.async_get_issue(DOMAIN, issue_id)) is None:
                continue
            if (previous.data or {}).get(_ACCOUNT_SCOPED):
                continue
            recorded_owner = _recorded_subentry_id(previous)
            if recorded_owner is not None and recorded_owner != owner:
                if issue_id == canonical_key:
                    canonical_shared = True
                continue
            if recorded_owner is None and service not in possible:
                if issue_id == canonical_key:
                    canonical_shared = True
                continue
            if recorded_owner is None and (
                len(possible) > 1 or (previous.data or {}).get(_AMBIGUOUS_SERVICE_OWNER)
            ):
                # An attributed outcome supersedes an old record with no
                # identifiable owner. Never carry its dismissal to this service.
                if subentry_id is not None or len(possible) == 1:
                    obsolete.add(issue_id)
                continue
            current[issue_id] = identifier

    if current:
        issue_id, identifier = next(iter(current.items()))
    else:
        issue_id = (
            _hashed_issue_id(canonical_key, owner)
            if canonical_shared
            else canonical_key
        )
        identifier = canonical_id
    return _ServiceRepair(issue_id, identifier, owner, set(current), obsolete)


def _create_issue(
    hass: HomeAssistant,
    key: _RepairKey,
    placeholders: dict[str, str],
    *,
    entry_id: str | None,
    service_id: str | None = None,
    subentry_id: str | None = None,
    reason: str | None = None,
) -> None:
    """Create an issue with nonsecret ownership metadata for later reconciliation."""

    def create() -> None:
        target_id = key.issue_id(entry_id, service_id)
        identifier = service_id
        owner = None
        if (
            entry_id is not None
            and service_id is not None
            and hasattr(ir, "async_refresh_issues")
        ):
            target = _current_service_repair(
                hass, key, entry_id, service_id, subentry_id
            )
            if target is None:
                return
            target_id, owner, identifier = (
                target.issue_id,
                target.subentry_id,
                target.service_id,
            )
            for issue_id in target.obsolete_ids:
                ir.async_delete_issue(hass, DOMAIN, issue_id)
        ir.async_create_issue(
            hass,
            DOMAIN,
            target_id,
            **_issue_ownership(entry_id, owner),
            is_fixable=False,
            is_persistent=True,
            issue_domain=DOMAIN,
            severity=ir.IssueSeverity.ERROR,
            translation_key=key.issue_type,
            translation_placeholders=placeholders,
            data={
                "entry_id": entry_id,
                "service_id": identifier,
                **({CONF_SUBENTRY_ID: owner} if owner is not None else {}),
                **(
                    {_ACCOUNT_SCOPED: True}
                    if entry_id is not None
                    and service_id is None
                    and key.issue_type == ISSUE_MODEL_ACCESS
                    else {}
                ),
                "model": key.model,
                "feature": key.feature,
                "reason": reason,
            },
        )

    _update_issue(create)


def _recorded_subentry_id(issue: ir.IssueEntry) -> str | None:
    """Read the recorded owner across Core versions."""
    value = getattr(issue, "config_subentry_id", None) or (issue.data or {}).get(
        CONF_SUBENTRY_ID
    )
    return value if isinstance(value, str) else None


def _delete_issue(
    hass: HomeAssistant,
    key: _RepairKey,
    entry_id: str | None,
    service_id: str | None,
    subentry_id: str | None = None,
) -> None:
    """Clear issues from a matching request with current ownership."""

    def delete() -> None:
        if (
            entry_id is not None
            and service_id is not None
            and hasattr(ir, "async_refresh_issues")
        ):
            if not ir.async_get(hass).issues:
                return
            target = _current_service_repair(
                hass, key, entry_id, service_id, subentry_id
            )
            if target is not None:
                for issue_id in target.current_ids | target.obsolete_ids:
                    ir.async_delete_issue(hass, DOMAIN, issue_id)
        else:
            ir.async_delete_issue(hass, DOMAIN, key.issue_id(entry_id, service_id))

    _update_issue(delete)


def async_create_ffmpeg_missing_issue(
    hass: HomeAssistant,
    entry: GroqConfigEntry,
    service_data: dict[str, Any] | None = None,
) -> None:
    """Create a repair issue when ffmpeg is required but missing."""
    subentry_id = (service_data or {}).get(CONF_SUBENTRY_ID)
    service_id = (service_data or {}).get(UNIQUE_ID) or subentry_id
    _create_issue(
        hass,
        _RepairKey(ISSUE_FFMPEG_MISSING),
        {"service_name": _service_name(service_data)},
        entry_id=entry.entry_id,
        service_id=service_id,
        subentry_id=subentry_id,
    )


def async_delete_ffmpeg_missing_issue(
    hass: HomeAssistant,
    entry: GroqConfigEntry,
    service_data: dict[str, Any] | None = None,
) -> None:
    """Delete the ffmpeg repair issue after audio processing succeeds."""
    subentry_id = (service_data or {}).get(CONF_SUBENTRY_ID)
    service_id = (service_data or {}).get(UNIQUE_ID) or subentry_id
    _delete_issue(
        hass,
        _RepairKey(ISSUE_FFMPEG_MISSING),
        entry.entry_id,
        service_id,
        subentry_id,
    )


def async_create_model_access_issue(
    hass: HomeAssistant,
    model: str,
    service_id: str | None = None,
    *,
    entry_id: str | None = None,
    reason: str | None = None,
    subentry_id: str | None = None,
) -> None:
    """Report a model access failure scoped to its account and service."""
    service_id = service_id or subentry_id
    _create_issue(
        hass,
        _RepairKey(ISSUE_MODEL_ACCESS, model),
        {"model": _safe(model)},
        entry_id=entry_id,
        service_id=service_id,
        subentry_id=subentry_id,
        reason=reason,
    )


def async_delete_model_access_issue(
    hass: HomeAssistant,
    model: str,
    service_id: str | None = None,
    *,
    entry_id: str | None = None,
    subentry_id: str | None = None,
) -> None:
    """Clear matching account/service access failure after successful inference."""
    service_id = service_id or subentry_id
    _delete_issue(
        hass,
        _RepairKey(ISSUE_MODEL_ACCESS, model),
        entry_id,
        service_id,
        subentry_id,
    )


def async_create_model_configuration_issue(
    hass: HomeAssistant,
    entry: GroqConfigEntry,
    service_data: dict[str, Any],
    model: str,
    feature: str,
) -> None:
    """Report a configured model that cannot serve a feature."""
    subentry_id = service_data.get(CONF_SUBENTRY_ID)
    service_id = service_data.get(UNIQUE_ID) or subentry_id
    _create_issue(
        hass,
        _RepairKey(ISSUE_MODEL_CONFIGURATION, model, feature),
        {
            "service_name": _service_name(service_data),
            "model": _safe(model),
            "feature": _safe(feature),
        },
        entry_id=entry.entry_id,
        service_id=service_id,
        subentry_id=subentry_id,
    )


def async_delete_model_configuration_issue(
    hass: HomeAssistant,
    entry: GroqConfigEntry,
    service_data: dict[str, Any],
    model: str,
    feature: str,
) -> None:
    """Clear the configured-model issue after capability validation succeeds."""
    subentry_id = service_data.get(CONF_SUBENTRY_ID)
    service_id = service_data.get(UNIQUE_ID) or subentry_id
    _delete_issue(
        hass,
        _RepairKey(ISSUE_MODEL_CONFIGURATION, model, feature),
        entry.entry_id,
        service_id,
        subentry_id,
    )


def _adopt_issue_ownership(
    hass: HomeAssistant,
    issue_id: str,
    issue: ir.IssueEntry,
    entry_id: str,
    matches: list[dict[str, Any]],
) -> bool:
    """Link an active repair in place, preserving its dismissal."""
    current_entry_id = getattr(issue, "config_entry_id", None)
    if (
        current_entry_id not in (None, entry_id)
        or not issue.active
        or issue.is_fixable is None
        or issue.severity is None
        or issue.translation_key is None
        or getattr(issue, "config_subentry_id", None) is not None
    ):
        return False
    data = issue.data
    if len(matches) > 1:
        # Remember that this legacy record could belong to more than one
        # service. Removing one candidate does not establish the original owner.
        data = {**(data or {}), _AMBIGUOUS_SERVICE_OWNER: True}
    service = (
        matches[0]
        if len(matches) == 1 and not (data or {}).get(_AMBIGUOUS_SERVICE_OWNER)
        else None
    )
    owner = service.get(CONF_SUBENTRY_ID) if service else None
    if (
        owner is not None
        and data is not None
        and "service_id" in data
        and not data["service_id"]
    ):
        data = {**data, CONF_SUBENTRY_ID: owner}
    ownership = _issue_ownership(entry_id, owner)
    if current_entry_id == entry_id and (
        "config_subentry_id" not in ownership and data == issue.data
    ):
        return False
    ir.async_create_issue(
        hass,
        DOMAIN,
        issue_id,
        **ownership,
        breaks_in_ha_version=issue.breaks_in_ha_version,
        data=data,
        is_fixable=issue.is_fixable,
        is_persistent=issue.is_persistent,
        issue_domain=issue.issue_domain,
        learn_more_url=issue.learn_more_url,
        severity=issue.severity,
        translation_key=issue.translation_key,
        translation_placeholders=issue.translation_placeholders,
    )
    return True


def async_reconcile_startup_issues(hass: HomeAssistant) -> None:
    """Adopt stored repairs and drop removed-service repairs before discovery."""
    if not hasattr(ir, "async_refresh_issues"):
        return

    def reconcile() -> None:
        entries = {
            entry.entry_id: entry for entry in hass.config_entries.async_entries(DOMAIN)
        }
        service_indexes: dict[str, dict[str | None, list[dict[str, Any]]]] = {}
        for (domain, issue_id), issue in list(ir.async_get(hass).issues.items()):
            if domain != DOMAIN or not issue_id.startswith(_ISSUE_PREFIXES):
                continue
            data = issue.data or {}
            entry_id = getattr(issue, "config_entry_id", None) or data.get("entry_id")
            if entry_id is None and not data:
                if (
                    recovered := _recover_legacy_issue_data(
                        issue, entries, service_indexes
                    )
                ) is not None:
                    entry_id, data = recovered
                    issue = replace(issue, data=data)
            if not isinstance(entry_id, str) or entry_id not in entries:
                continue
            if entry_id not in service_indexes:
                service_indexes[entry_id] = _services_by_identifier(entries[entry_id])
            service_scoped, matches = _issue_service_matches(
                entries[entry_id], issue, service_indexes[entry_id]
            )
            if (
                service_scoped
                and not matches
                or _ambiguous_owner_changed(data, matches)
            ):
                ir.async_delete_issue(hass, DOMAIN, issue_id)
            else:
                _adopt_issue_ownership(hass, issue_id, issue, entry_id, matches)

    _update_issue(reconcile)


def _recover_legacy_issue_data(
    issue: ir.IssueEntry,
    entries: dict[str, GroqConfigEntry],
    service_indexes: dict[str, dict[str | None, list[dict[str, Any]]]],
) -> tuple[str, dict[str, str | int | float | None]] | None:
    """Restore ownership for older persistent notices with a verifiable hash."""
    if not issue.active or not issue.is_persistent:
        return None
    placeholders = issue.translation_placeholders or {}
    if issue.translation_key == ISSUE_FFMPEG_MISSING:
        key = _RepairKey(ISSUE_FFMPEG_MISSING)
    elif issue.translation_key == ISSUE_MODEL_CONFIGURATION and (
        (model := placeholders.get("model"))
        and (feature := placeholders.get("feature"))
    ):
        key = _RepairKey(ISSUE_MODEL_CONFIGURATION, model, feature)
    else:
        return None

    owners: list[tuple[str, str | None]] = []
    for entry_id, entry in entries.items():
        if entry_id not in service_indexes:
            service_indexes[entry_id] = _services_by_identifier(entry)
        services = service_indexes[entry_id]
        for identifier in set(services) | {None}:
            if key.issue_id(entry_id, identifier) != issue.issue_id:
                continue
            if (
                key.issue_type == ISSUE_FFMPEG_MISSING
                and identifier is None
                or _repair_candidates(entry, services, key, identifier, historical=True)
            ):
                owners.append((entry_id, identifier))

    if len(owners) != 1:
        return None
    entry_id, identifier = owners[0]
    return entry_id, {
        "entry_id": entry_id,
        "service_id": identifier,
        "model": key.model,
        "feature": key.feature,
    }


def _issue_service_matches(
    entry: GroqConfigEntry,
    issue: ir.IssueEntry,
    services: dict[str | None, list[dict[str, Any]]],
) -> tuple[bool, list[dict[str, Any]]]:
    """Find candidates using recorded ownership or the preceding release's producer."""
    subentry_id = _recorded_subentry_id(issue)
    data = issue.data or {}
    if data.get(_ACCOUNT_SCOPED):
        return False, []
    service_id = data.get("service_id", subentry_id)
    identifier = service_id if isinstance(service_id, str) and service_id else None
    feature = data.get("feature")
    model = data.get("model")
    key = _RepairKey(
        issue.translation_key or "",
        model=model if isinstance(model, str) else None,
        feature=feature if isinstance(feature, str) else None,
    )
    if key.issue_type == ISSUE_FFMPEG_MISSING and identifier is None:
        flat_tts_active = (
            has_legacy_tts_config(entry)
            and FEATURE_TEXT_TO_SPEECH in enabled_features_from_entry(entry)
            and not service_data_for_type(entry, FEATURE_TEXT_TO_SPEECH)
        )
        return not flat_tts_active, []
    matches = (
        _repair_candidates(entry, services, key, identifier, historical=True)
        if subentry_id is None
        else [
            service
            for service in services.get(identifier, [])
            if service.get(CONF_SUBENTRY_ID) == subentry_id
            and key.matches(service, identifier)
        ]
    )
    return service_id is not None or subentry_id is not None or bool(matches), matches


def _ambiguous_owner_changed(
    data: dict[str, str | int | float | None], matches: list[dict[str, Any]]
) -> bool:
    """Retire a historic ambiguous issue once fewer than two candidates remain."""
    return bool(data.get(_AMBIGUOUS_SERVICE_OWNER) and len(matches) < 2)


def async_reconcile_entry_issues(
    hass: HomeAssistant,
    entry: GroqConfigEntry,
    *,
    model_registry: GroqModelRegistry | None = None,
    remove_entry: bool = False,
) -> None:
    """Remove obsolete owned issues after setup, configuration changes, or removal."""

    def reconcile() -> None:
        registry = ir.async_get(hass)
        services = _services_by_identifier(entry)
        configured_models = {
            service.get(CONF_MODEL)
            for matches in services.values()
            for service in matches
        }
        configured_models.add(entry_value(entry, CONF_MODEL))
        issues = {
            issue_id: issue
            for (domain, issue_id), issue in list(registry.issues.items())
            if domain == DOMAIN
            and issue_id.startswith(_ISSUE_PREFIXES)
            and (
                getattr(issue, "config_entry_id", None)
                or (issue.data or {}).get("entry_id", entry.entry_id)
            )
            == entry.entry_id
        }
        refresh_issues = getattr(ir, "async_refresh_issues", None)
        # Include legacy issues by exact ID: filtering by native ownership would
        # exclude them before adoption. Never refresh around platform forwarding;
        # setup does not recheck request-time access/ffmpeg failures.
        with (
            refresh_issues(hass, DOMAIN, issue_ids=issues)
            if refresh_issues is not None
            else nullcontext()
        ) as refresh:
            for issue_id, issue in issues.items():
                data = issue.data or {}
                # Core can retain an ignored, inactive issue without its former
                # metadata. Keep that dismissal until a real check recreates it;
                # it has no visible notice or safely identifiable owner to clean.
                if (
                    not issue.active
                    and issue.dismissed_version is not None
                    and "entry_id" not in data
                    and getattr(issue, "config_entry_id", None) is None
                ):
                    if refresh is not None:
                        refresh.async_keep([issue_id])
                    continue
                obsolete = remove_entry or (
                    "entry_id" not in data
                    and getattr(issue, "config_entry_id", None) is None
                )
                service_scoped, matches = _issue_service_matches(entry, issue, services)
                obsolete |= (
                    service_scoped
                    and not matches
                    or _ambiguous_owner_changed(data, matches)
                )
                model = data.get("model")
                if model is not None:
                    if matches:
                        obsolete |= all(
                            service.get(CONF_MODEL) != model for service in matches
                        )
                    else:
                        obsolete |= model not in configured_models
                feature = data.get("feature")
                if (
                    model_registry is not None
                    and isinstance(model, str)
                    and isinstance(feature, str)
                ):
                    try:
                        obsolete |= model_registry.supports(model, GroqFeature(feature))
                    except ValueError:
                        obsolete = True
                if obsolete:
                    if refresh is None:
                        ir.async_delete_issue(hass, DOMAIN, issue_id)
                    continue
                if refresh is not None and not _adopt_issue_ownership(
                    hass, issue_id, issue, entry.entry_id, matches
                ):
                    refresh.async_keep([issue_id])

    _update_issue(reconcile)
