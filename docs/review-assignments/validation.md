# Validation, CI, typing, and documentation assignment

Status: original plan, now implemented. Owner: `lifecycle_review` for the review
package; the root agent implemented validation and documentation. See the
[implementation record](../architectural-improvements-2026-09-05.md).
Review date: 2026-09-05.

## Verified baseline and its limits

The parent reviewer reports 322 passing tests, 100% statement coverage, strict typing
passing for 25 files, and a passing import-warning gate in Home Assistant 2026.7.2 /
Python 3.14.6. Its cold import measurement for 25 integration modules was 1,863 ms.
These results were supplied by the parent and were not rerun in this assignment.
The cold aggregate includes dependency imports and is not representative HA startup
latency, a throughput measurement, or evidence that streaming is nonblocking.

The test and configuration source were inspected directly. Current tests are heavily
built around `Dummy*` and `SimpleNamespace` implementations; repository searches found
no `MockConfigEntry`, `enable_custom_integrations`, or `async_setup_component` usage.
`test_coverage_gaps.py` alone has 4,228 lines; `test_foundation.py` and
`test_integration_harness.py` add 6,032. Complete statement coverage does not prove
HA interface compatibility or that a significant branch was tested with realistic
values. The immutable-entry defects in lifecycle assignment L3 demonstrate this gap.

## V1 — P1: Establish genuine Home Assistant boundary tests

Owner: validation agent. Files: new `tests/components/groq/conftest.py` and focused
test modules for lifecycle, config flow, conversation, service execution, and repairs.
Coordinate fixtures and acceptance cases with the lifecycle/platform/transport owners;
each owner retains responsibility for regression tests specific to their changes.

Use the real `hass` fixture, enable custom integration loading, add `MockConfigEntry`
objects to HA, and enter setup through `hass.config_entries.async_setup` or
`async_setup_component`. Mock outbound Groq network operations, clocks, and ffmpeg
processes; preserve actual HA lifecycle, registries, flow managers, service dispatch,
and conversation structures. Keep pure unit tests for algorithms and request parsing.

Required acceptance scenarios:

1. Account-only setup creates runtime data without service entities. Adding each
   supported subentry creates correctly owned entity/device registry entries on
   required platforms. Failed authentication and transient setup errors produce the
   expected real config-entry states and retries without duplicate entities/listeners.
2. Reconfigure, reauthenticate, add/delete subentries, unload and reload through HA;
   assert final entity ownership, service behavior, and exactly the intended reload.
   Test actual immutable entry/subentry data and runtime cache lifecycle.
3. Drive account and subentry flows through real flow managers. Cover blank API-key
   preservation, duplicate credentials, clearing Assist control, hidden advanced-field
   preservation, model/voice changes, and required translated validation outcomes.
4. Use real conversation `ChatLog`/content structures through public HA conversation
   processing to validate streaming text, tool-call rounds, tool results, cancellation,
   and error responses. Do not recreate ChatLog behavior in another dummy class.
5. Call registered services through `hass.services.async_call` with the appropriate
   response mode. Assert schema rejection, actual response objects, account/service
   selection, unloaded-entry rejection, authentication failures, and feature guards.
6. Use the real issue registry to prove repair creation, recovery, account isolation,
   and service/account-removal cleanup; inspect final registry contents.

Definition of done: these cases run in both supported-version environments; at least
the immutable-mapping and control-clearing regressions fail on the reviewed baseline
and pass with their corresponding fixes. No real network or user notifications occur.
All existing 100% integration and changed-module coverage requirements stay in force.

## V2 — P2: Make compatibility environments reproducible and truthful

Owner: validation agent. Files: `devtools/docker/Dockerfile`,
`devtools/docker/docker-compose.yml`, dependency files, `.github/workflows/tests.yml`,
runner scripts and their focused script tests.

Evidence: Dockerfile line 1 and compose pin Core 2026.7.2;
`requirements-dev.txt:4` pins `pytest-homeassistant-custom-component==0.13.346`.
The main checks job uses that environment. The minimum job at
`.github/workflows/tests.yml:119-136` installs helper 0.13.336 with `--no-deps`, then
separately pins Core 2026.6.0. Neither path asserts actual Core after dependency
installation or runs `pip check`. Because the helper pins Core, changing `HA_IMAGE`
alone can let pip replace the image's Core with the helper's different version.

Implement two explicit environments: the advertised minimum (currently 2026.6.0)
and a verified current stable Core release. Resolve compatible helper/dependency
versions from their release metadata at implementation time; do not invent a matching
helper version or assume changing a container tag is enough. Keep immutable image
digests where the environment uses Docker. Align versions in one documented source
of truth or add a small consistency check for necessarily duplicated declarations.

Acceptance criteria:

- After all installs, assert the exact `importlib.metadata.version("homeassistant")`
  and expected Python version, and record the helper version. Fail before tests if
  these disagree with the job label; demonstrate with a deliberately conflicting pin
  in the runner's regression tests. An `HA_IMAGE` override must either be honored
  with compatible dependencies or fail clearly, never silently downgrade Core.
- Run `python -m pip check` in both environments. Resolve incompatible pins rather
  than hiding them through `--no-deps`. If no compatible helper exists, document and
  choose a supported fixture-source strategy instead of calling the environment green.
- Run component and genuine HA boundary tests in both matrix legs. Keep the full
  required local validation set and 100% coverage gates in the primary environment;
  never reduce thresholds to accommodate the migration.
- Include dependency/runner/matrix changes in cache keys. Publish concise actual
  environment metadata with test artifacts so a reviewer can verify what was tested.
- Once established, local runner defaults and contributor instructions match the
  current stable leg; minimum compatibility remains a distinct explicit command/job.

The existing minimum job is valuable and should be improved, not described as absent.
This plan does not assert it currently downgrades Core: the confirmed defect is the
missing environment assertion and the dependency mechanism that permits a mismatch.

## V3 — P2: Expose HA typing boundaries currently treated as Any

Owner: validation agent; source changes returned to their module owner.
Files: `pyproject.toml`, `scripts/strict-typing`, optional small typing-contract probe.

Evidence: `pyproject.toml:22` disables `disallow_subclassing_any`; `24-33` suppresses
missing typing imports for all `homeassistant.*`. `scripts/strict-typing` simply runs
mypy over the integration. This is useful for local annotations but cannot by itself
establish that overrides, config-entry mappings, callbacks or ChatLog interactions
match current Core when those imports resolve to Any.

First perform a bounded investigation in each verified Core environment. Use a
small mypy probe to inspect inferred types for `ConfigEntry.data`, `ConfigSubentry`,
the relevant entity bases, and ChatLog. Compare against that exact Core version's
source/type declarations. Evaluate a scoped check against installed typed source or
a pinned source checkout with narrow follow-import configuration. Keep this work
separate from enabling every dependency's strict checking.

Acceptance: the added check detects a deliberately incorrect mapping mutation or
callback/entity signature in a fixture, while a valid contract passes. Restore
`disallow_subclassing_any` for demonstrably typed boundaries and narrow broad ignores
only where evidence supports it. Document remaining Any boundaries. Do not promise
immediate full upstream typing or replace HA types with permissive handwritten stubs
that merely repeat the integration's assumptions. Real boundary tests remain required.

## V4 — P2: Replace obsolete architecture prose with the implemented design

Owner: documentation agent (may be the validation agent after V1–V3).
Files: `docs/architecture.md`, README, contributor docs, quality-scale evidence entries
as needed. No production behavior changes belong in this package.

Evidence: `docs/architecture.md` opens as a future expansion beyond TTS, proposes
nonexistent modules, omits AI Task from actual platform mapping, describes old
account-level feature options, and says services register in `async_setup_entry`.
Actual registration is in `__init__.async_setup`. README line 75 incorrectly states
local development tests the minimum, despite the 2026.7.2 Docker pin and 2026.6.0
minimum. Quality validation establishes declarations/references, not independently
verified architectural compliance or official Home Assistant certification.

Rewrite the architecture document as a description of current behavior: account
credentials, service subentries, typed entry runtime, platform mappings including
AI Task, shared API/cache/rate limits, stream/tool orchestration, entry reload/unload,
service registration, error/repair ownership, model discovery/fallback, and preserved
legacy configuration. Use a compact static Mermaid dependency diagram if useful.
Describe actual local cache behavior separately from provider caching. Move genuinely
future proposals into an explicitly optional section or remove obsolete ones.

Acceptance: every listed module and named public method exists; flow/lifecycle/service
descriptions match final code; README/CI claims match actual verified environments;
quality evidence points to meaningful tests, including remaining limitations. Keep
the self-assessment wording distinct from an independent HA certification. State that
passing 100% statement coverage, strict typing with known boundaries, and static
checks does not alone establish production reliability or current-Core compatibility.
Use documentation-scoped checks for documentation-only edits rather than running the
whole runtime suite again without a relevant code/dependency change.

## Optional validation maintenance

- Split oversized catch-all test modules gradually by behavioral area while retaining
  meaningful cases; remove tests that exist solely for confirmed dead production
  helpers when those helpers are removed. Avoid a wholesale mechanical rewrite.
- Collect branch coverage as additional evidence for the changed asynchronous/error
  paths. Keep the existing 100% statement gate; do not invent a repository-wide branch
  threshold before measuring its baseline. Prioritize cancellation, recovery, timeout,
  stream-end, partial-result and multi-account branches over coverage-only assertions.
- Establish warmed integration setup and first-response/event-loop responsiveness
  benchmarks with deterministic fake I/O before claiming performance improvements.
  Report dependency-cold imports separately; retain the warning gate's original role.

## Delivery order and handoff

V1 fixtures and V2 environment checks precede claims that implementation is validated.
Regression fixes can proceed alongside them, with module owners supplying their
behavioral cases. V3 is a bounded typing investigation with explicit residual limits;
V4 describes the resulting architecture after fixes stabilize. The root reviewer
integrates these packages and performs the repository-required final verification.

Primary references:

- [Home Assistant testing guidance](https://developers.home-assistant.io/docs/development_testing/)
- [Home Assistant test coverage quality rule](https://developers.home-assistant.io/docs/core/integration-quality-scale/rules/test-coverage/)
- [Home Assistant integration quality scale](https://developers.home-assistant.io/docs/core/integration-quality-scale/)
