# Lifecycle and configuration architectural review

Status: original review brief; implementation is complete. See the
[implementation record](../architectural-improvements-2026-09-05.md).
Owner: `lifecycle_review`. Review date: 2026-09-05.

Scope: `__init__.py`, `runtime.py`, `config_flow.py`, `flow_schemas.py`,
`model_registry.py`, `feature_registry.py`, `repairs.py`, `diagnostics.py`,
`entity.py`, and directly related constants/subentry helpers and tests.
Line references describe the reviewed baseline, before implementation.

## Assessment

The integration already uses typed `ConfigEntry.runtime_data`, an entry-scoped shared
client/rate limiter/cache, awaited platform forwarding and unloading, an unload-bound
update listener, setup retry/authentication exceptions, service subentries, and executor
offloading for ffmpeg discovery. These are sound foundations. There is no reason to add
a polling coordinator to this request-driven integration merely to match a pattern.
The principal weaknesses here are configuration preservation, real HA data contracts,
and repair lifecycle management. Synthetic test entries conceal real integration errors.

## Confirmed fixes, in priority order

### L1 — P1: Clearing Assist control restores the previous selection

Evidence: `config_flow.py:687-689` cleans submitted basic options before either merge
at `712-715` or `727-731`; `flow_schemas.py:778-779` deletes an empty `llm_hass_api`.
An existing `llm_hass_api: [assist]` is therefore merged back when the user deselects
all control APIs. This affects both saving the basic form and proceeding to advanced
configuration. The advertised removal of Home Assistant control does not take effect.

Fix: preserve explicit clearing intent across reconfiguration. Merge the submitted
basic fields into existing data before deleting cleared optional fields, while keeping
hidden advanced options intact. Avoid changing fresh-service defaults.

Acceptance tests: reconfigure an existing control-enabled service with an empty API
selection through both paths; assert stored service data has no enabled control API
and the conversation configuration resolves to no tools. Verify retained advanced
options and explicitly selected APIs survive. Use the actual flow manager where
practical, not only a mocked `async_show_form`/`async_create_entry` return dictionary.

### L2 — P2: Submitting a blank account API key deletes existing options

Evidence: `config_flow.py:528-534` copies the current options, then replaces the copy
with the submitted empty dictionary when no replacement key was supplied. The form
exposes only the API key, so this loses unrelated legacy options, including a working
key stored in options, service enablement, model/voice settings, and cache settings.
`test_integration_harness.py:2837-2904` only exercises an entry whose options are empty.

Fix: preserve existing options on an empty-key submission. When a replacement key is
provided, write it to entry data and remove only the legacy key override from options.

Acceptance tests: start with nonempty options, including a legacy API key different
from entry data; submit blank input and assert all effective configuration is unchanged.
Replace the key and assert unrelated options survive and exactly one reload occurs.

### L3 — P2: Real Home Assistant mappings fail legacy migration and diagnostics

Evidence: `__init__.py:101` accepts only `dict` for legacy unique-ID migration;
`const.py:310-313` accepts only `dict` for subentry feature discovery. Home Assistant
wraps entry data/options and subentry data in `MappingProxyType`. Consequently the
migration branch cannot run for a real entry, and diagnostics report no enabled
services for normal subentry-only configurations. Platform setup uses another helper
and is not itself disabled by the diagnostics defect.

Fix: use the declared mapping contract, either direct membership/access for typed
entries or `collections.abc.Mapping` when defensive handling remains necessary.
Also make the diagnostics legacy fallback honor option overrides consistently with
`runtime.py:69-76`. Check historical entry versions before choosing any migration
version bump: `ConfigFlow.VERSION` is currently 1, and fixing the callback alone does
not guarantee that HA will invoke it for all historical entries.

Acceptance tests: real HA `MockConfigEntry`/`ConfigSubentry` or immutable mapping
fixtures; confirm migration writes the existing stable ID without mutation of input,
and diagnostics lists configured service types/counts. Keep existing-ID entries
unchanged. Test legacy TTS configuration stored partly in options.

Primary evidence: [Home Assistant ConfigEntry implementation](https://raw.githubusercontent.com/home-assistant/core/dev/homeassistant/config_entries.py),
specifically the `ConfigEntry` constructor and `ConfigSubentry.data` declaration.

### L4 — P2: Persistent model repair issues have no recovery lifecycle

Evidence: `repairs.py:105-156` creates persistent model-access and model-configuration
issues; no corresponding delete functions exist or are called anywhere in the
integration. API creation at `api.py:1032` supplies only the model, so different
accounts collide on one issue ID. Configuration issues include the old model in
their ID, leaving obsolete warnings after service reconfiguration/removal. The
ffmpeg issue has a success deletion path but also needs service-removal handling.

Fix: coordinate one owner across repairs, API, services and entry lifecycle. Scope
issues to account/service as appropriate; record sufficient nonsecret ownership
metadata; clear matching access issues after successful access, and reconcile
configuration issues after configuration changes or service/account removal. A
successful request for one account must not clear another account's failure. Avoid
treating successful `/models` discovery alone as proof of model inference access.
Coordinate with the transport owner's HTTP 403 classification fix so organization/
project model-permission failures use this repair path rather than reauthentication.
Pass account context and a bounded permission reason, never raw response content.

Acceptance tests: denied access followed by success clears only the matching issue;
two accounts using the same model remain independent; service model changes and
deletion remove obsolete configuration issues; unrelated active issues survive.
Use the real issue registry and verify its resulting contents.

Guidance: [Home Assistant repairs](https://developers.home-assistant.io/docs/core/platform/repairs/)
and [repair quality rule](https://developers.home-assistant.io/docs/core/integration-quality-scale/rules/repair-issues/).

### L5 — P2: Runtime capability checks accept explicitly inactive models

Evidence: `model_registry.py:559` filters inactive models from selectors, but
`supports()` at `580-585` only checks capabilities. `services.py:449-477` relies on
that method before issuing requests. A stored model advertised with `active: false`
therefore passes the local runtime guard and can fail on every API request.

Fix: require active status for registered models in `supports`, while preserving
the deliberate fallback-inference policy when discovery was unavailable.

Acceptance tests: registered inactive model with matching capabilities returns false
for both feature and direct-capability queries; active equivalents return true;
unknown-model behavior remains consistent between strict and fallback registries.
Prove the service handler rejects an inactive configured model before network I/O.

### L6 — P2: Successful restricted discovery is replaced with unavailable built-ins

Evidence: `config_flow.py:212` correctly creates a strict registry from discovered
models, but `_model_ids_for_feature` at `215-222` falls back whenever that particular
feature has no models. An account that successfully discovers only text models is
therefore offered built-in speech/vision models absent from its own strict registry.
Speech/image flows can save those selections without capability validation.

Fix: distinguish failed/empty discovery fallback from a successful catalogue with
zero matching active models. Present a translated no-compatible-models result rather
than silently advertising unavailable models. Keep fallback selectors usable during
genuine discovery failure, and validate feature compatibility before saving services.
This touches UI/translation behavior and should be a separate bounded change.

Acceptance tests: successful text-only discovery does not offer built-in TTS/vision
models; discovery failure still permits the existing fallback; reconfiguration gives
a clear result for a removed configured model; all new translation keys are complete.

## Optional cleanup assignments

1. Remove production-unreferenced helpers after a repository reference check:
   `__init__._has_other_loaded_entries` (`75-89`), `config_flow.fetch_available`
   (`150-175`), and `is_tts_model`, `get_model_options`, `get_dynamic_options`
   (`258-278`). Current references are definitions and tests/patches. Remove their
   obsolete tests and monkeypatches instead of preserving test-only production code.
2. Inline `config_flow._service_data_for_schema` (`991-996`), which ignores
   `existing_data` and only copies `new_data`. Remove the unused `service_data`
   plumbing on `_model_registry` if no actual account/service behavior requires it.
3. Consolidate account-value precedence and enabled-feature derivation after L2/L3.
   `diagnostics._entry_value` duplicates `runtime.entry_value`; runtime duplicates
   `CONF_BASE_URL` from `const.py`; feature capabilities exist in both feature and
   model registries. Prefer a small dependency-light configuration helper instead
   of importing the API runtime into diagnostics solely to reuse one expression.
4. Reduce repeated TTS error-form construction with a local rendering helper only
   after behavior tests protect model/voice reselection and ffmpeg requirements.
5. Replace blanket `suppress(Exception)` around repair registry operations with
   narrow, observable handling where justified. Unexpected programming errors should
   not silently make repairs disappear; avoid converting optional repair failures
   into failed audio/text requests.

## Compatibility and validation boundaries

Preserve actual supported-version behavior: legacy account TTS configuration and
the schema-type-aware OpenAPI compatibility path are purposeful compatibility code.
Do not remove them because they look redundant. In contrast, fallbacks for missing
`hass.services`, missing executor helpers, absent entry lookups, runtime-data setters
raising `AttributeError`, and the long OptionsFlow entry-lookup ladder are primarily
test-double accommodations unless a supported HA version can be demonstrated to need
them. First improve fixtures and verify minimum/current Core before removing them.

This review used source/call-site analysis and current official HA implementation;
it did not run a runtime test suite or measure throughput. Claims of performance
improvement require measurements. Implementation acceptance should use targeted
behavioral tests, then the repository's required validation set before any push.
The large `test_coverage_gaps.py` should not receive all new tests by default: group
new regression tests by configuration, registry, diagnostics, and repairs behavior.
