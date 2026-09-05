# Architectural improvements and critical review — 5 September 2026

The [baseline review](architectural-review-2026-09-05.md) identified configuration,
transport, generation, and validation defects in `8dca993` (1.4.4). The three assigned
subagents implemented their workstreams. The root agent implemented the validation
harness and documentation, reviewed the combined patch, and fixed the additional
findings below. The changes form one reviewed branch change set; no release or live
configuration change was performed.

## Implementation record

All confirmed findings are addressed. The IDs below map to the original assignment
briefs; the optional investigations and scope decisions are recorded separately.

| Finding | Implemented behavior | Main evidence |
| --- | --- | --- |
| L1 | An explicitly cleared Assist API selection replaces the old setting before blank-field cleanup. | Real subentry reconfiguration and entity control tests |
| L2 | Blank or replacement account keys preserve unrelated legacy options. | Real options flow tests |
| L3 | Migration, feature derivation, and diagnostics accept HA's immutable mappings; minor version 2 schedules migration. | Real `MockConfigEntry` and `MappingProxyType` tests |
| L4 | Repair issues carry account/service/model ownership and clear after recovery, reconfiguration, or removal. | Real issue registry tests with two accounts |
| L5 | Explicitly inactive models fail capability checks. | Model registry regressions |
| L6 | Successful restricted or empty discovery remains authoritative; failed discovery retains the built-in fallback. | Real subentry flow tests |
| T1 | Organization/project model permissions create model access errors without account reauthentication. | JSON, SSE, audio, and multipart STT transport tests |
| T2 | Compound reset durations parse fully; guards use exhausted windows and round waits conservatively. | Duration/window regression tests |
| T3 | SSE HTTP errors retain classification; malformed UTF-8/JSON, error events, and incomplete streams raise integration errors. | Stream framing and error tests |
| T4 | API error payloads retain only bounded classification codes/types, excluding echoed prompts and provider messages. | Sensitive-data regressions and real HTTP-to-AI fallback test |
| T5 | Prompt-cache values are serialized and decoded independently, isolating nested mutations. | Nested mutation regression |
| T6 | One temporary-file owner drains executor creation/writes/cleanup under cancellation. | Creation failure and repeated-cancellation tests |
| T7 | Media readers request at most the size limit plus one byte, then reject growth beyond the limit. | Growing-file and bounded-read tests |
| T8 | Prompt responses retain at most 16 MiB; speech caches share a 64 MiB account budget, including keys. Expired entries are purged before eviction. | Byte accounting, LRU, expiry, and credential namespace tests |
| T9 | Non-finite speed and fractional/non-finite sample rates are rejected; invalid stored defaults remain selector-safe. | Runtime and flow numeric regressions |
| T10 | Outage logs describe recovery on the next request without promising an unscheduled retry. | Availability/logging tests |
| G1 | Inline image transport bytes are excluded from text context estimates. | Large inline-image regression |
| G2 | Each selected service's cache preference controls cache access; disabled requests skip key construction. | Registered service tests with sibling preferences |
| G3 | HA service schemas preserve configured strictness/schema names when call overrides are omitted. | Real registered service schema tests |
| G4 | Native, fallback, and tool-assisted structured results use local validation; remote schema fetching is disabled. | Invalid result shape, local reference, and task schema tests |
| G5 | History is trimmed before attachment reads; retained attachments are reused within a tool turn. | Attachment read-count and tool-history tests |
| G6 | Malformed tool argument objects and missing/duplicate call IDs fail before tool execution. | Tool dispatch regressions |
| G7 | Explicit reasoning overrides control both request options and capability validation. | Registered action override test |
| G8 | Bounded local reads, image encoding, inline image validation, and vision cache-key hashing run through the executor. | Media boundary and real executor tests |
| G9 | AI tasks retain HA's prepared API prompt with zero tools and record one final assistant message for successful native/fallback/plain responses. | Real `internal_async_generate_data` and `ChatLog` tests |
| G10–G14 | Shared chat adapters and JSON validation have neutral modules; typed request-option resolution replaces repeated mappings; vision/OCR handlers and text schemas share implementations; dead helpers are removed. | Cross-entrypoint regression suites and strict typing |
| V1 | New regressions exercise real HA flows, service schemas, registries, callbacks, immutable data, and ChatLog; external I/O stays mocked. | Three focused architecture regression modules |
| V2 | Current/minimum Docker environments pin matching Core/helper versions, install resolved dependencies normally, and verify environment versions before tests. | Environment mismatch tests and both Docker runs |
| V3 | Strict typing resolves actual HA source types. Positive/negative probes reject immutable-data writes and invalid callbacks/chat content. | Source typing checks on both versions |
| V4 | Architecture, development instructions, README, changelog, and quality evidence describe the resulting implementation. | Documentation review and quality validator |

## Additional findings fixed during critical review

- A fresh speech caller could join an identical request whose final previous waiter
  had already cancelled it. New callers now wait for that cleanup, then start fresh
  work. The client retains shutdown ownership throughout; duplicate active requests
  still share one synthesis operation.
- Evicted or unsuccessful speech requests could leave empty cache namespaces behind.
  Namespace allocation now occurs only when retaining audio; eviction removes empty
  namespaces.
- Empty successful model listings still took the discovery-failure fallback. They
  now abort service creation with `no_compatible_models`.
- Optional numeric selectors used `None` defaults that HA's real flow schema could
  reject on an otherwise empty advanced form. Unset defaults now use `vol.UNDEFINED`.
- Stored non-finite TTS defaults could break selectors or retain NaN. Defaults now
  fall back safely, and fractional sample rates are not silently truncated.
- ffmpeg pipe I/O errors could bypass process cleanup. These failures now kill/reap
  the child and raise the translated ffmpeg error; stdout and stderr have separate
  bounded readers.
- Multipart STT needed separate model context to create and clear the same repair
  issue without changing its HTTP form payload.
- Provider error sanitization removed message text used by AI JSON fallback. The
  fallback now recognizes the retained `json_validate_failed` classification code,
  verified through the HTTP-to-runtime path.
- Inline image validation and eligible vision cache-key generation still performed
  potentially large CPU work on the event loop. Both now use executor jobs.
- The minimum helper pins `ast-serialize==0.3.0`, incompatible with current mypy.
  Minimum tests pin compatible mypy 2.1.0; current tests use 2.3.1. Both enforce the
  same strict contracts. Shared workspace caches caused concurrent checker failures;
  cache paths now separate Core version, mypy version, and check purpose.
- Obsolete unit fixtures depended on production fallbacks for missing HA objects.
  Fixtures now supply the required HA context; production test-only guards were not
  restored.

## Cleanup and measured design decisions

Lifecycle cleanup removed unused discovery/schema helpers, duplicate configuration
precedence and capability tables, repeated TTS error-form construction, and redundant
repair exception suppression. Optional repair failures remain observable in one
logging boundary without replacing the original request outcome.

Generation cleanup moved shared chat adaptation to `chat.py` and local JSON validation
to `structured.py`, removed the dead synchronous history adapter and unused model
helpers, shared typed option resolution, and consolidated vision/OCR action handling.
The schema-type-based Probatio/Voluptuous compatibility path remains supported.

Transport cleanup removed the unused structured-request alias, reasoning flag, and
ffmpeg argument table. TTS uses a public batch-check interface. Identical synthesis
requests coalesce with at most eight owned operations and explicit cancellation and
shutdown behavior. Temporary files and bounded subprocess I/O moved to `audio_files.py`.
Further cache/quota class extraction was evaluated but would mainly relocate existing
state after these focused changes; no additional framework or class split is needed
to implement the corrected contracts.

The local TTS guard remains a conservative per-service estimate of request attempts,
not authoritative organization billing or retry accounting. Changing that policy
would alter user-visible quota behavior and was not justified by the review evidence.

WAV packaging was measured with seven samples per size in the development container:
32 KiB median/max 0.013/0.031 ms; 1 MiB 0.265/0.470 ms; 25 MiB 8.064/10.602 ms.
These measurements did not justify executor overhead for every small STT chunk.
ffmpeg output is bounded at 64 MiB, with only a 64 KiB stderr prefix retained while
draining the stream. Measurements on this host do not establish Raspberry Pi latency.

New tests are grouped by lifecycle, transport, and generation behavior. Existing
useful unit tests remain; obsolete helper-only tests were removed. A wholesale
mechanical split of the older test files was deliberately avoided.

## Validation

| Check | Result |
| --- | --- |
| Current environment | HA 2026.9.0, Python 3.14.6, pytest helper 0.13.363; dependency check passes |
| Minimum environment | HA 2026.6.0, Python 3.14.5, pytest helper 0.13.336; dependency check passes |
| Complete current suite | 474 tests pass; 4,470 integration statements, 100% coverage across 28 modules |
| Minimum component suite | 425 tests pass |
| Supplementary branch run | 425 tests pass; 99% combined statement/branch coverage, 49 partial branches; no new branch threshold imposed |
| Changed-module coverage | Independent component-only run passes; all 23 changed modules at 100% (4,369 statements) |
| Strict integration typing | All 28 modules pass against actual HA source |
| HA contract probes | Positive controls pass; invalid mapping writes/callbacks/chat content are rejected on both versions |
| Static checks | Pinned pre-commit hooks, quality evidence, integration import warning gate, and whitespace validation |

The final deterministic benchmark used five warmed setup samples and 100 uncached
registered actions with network I/O mocked. Setup median was 20.79 ms; action median
was 0.040 ms and p95 0.382 ms; the peak event-loop tick was 13.41 ms. These measure
integration overhead, not live Groq latency. A separate cold import sample for all
28 modules was 6,161 ms, including dependency initialization. The original audit's
1,863 ms sample used an older HA version and different load, so these samples do not
establish an import-time regression or improvement. The integration warning gate
passes; pytest still emits a SyntaxWarning from the external `rich` dependency.

Final verification commands included:

```sh
scripts/test pre-commit run --all-files
scripts/strict-typing
scripts/test python scripts/validate_quality_scale.py
scripts/test python scripts/check_ha_typing.py
scripts/test --minimum python scripts/check_ha_typing.py
scripts/test --minimum python -m pytest tests/components/groq -q
scripts/test python scripts/importtime_profile.py --strict-integration-warnings --output /tmp/groq-final-importtime.log
scripts/test python -m pytest --cov=custom_components.groq --cov-report=term-missing --cov-report=xml --cov-fail-under=100 --junitxml=junit.xml -o junit_family=legacy
scripts/test env COVERAGE_FILE=/tmp/groq.coverage python -m coverage report -m '--include=custom_components/groq/*.py' --fail-under=100
scripts/test env COVERAGE_FILE=/tmp/groq-branch.coverage python -m coverage run --branch -m pytest tests/components/groq -q
scripts/test env COVERAGE_FILE=/tmp/groq-branch.coverage python -m coverage report '--include=custom_components/groq/*.py'
scripts/test python -m pytest scripts/benchmark_runtime.py -q -s
git diff --check
```

The `/tmp/groq.coverage` report uses an independent erased-and-rerun component suite;
the explicit list of 23 changed modules was also checked at 100%. Pre-commit was run
with `--files` over changed and untracked files as well as `--all-files`, so newly
added modules, tests, scripts, and documents were included before staging.

These are local Docker checks, not a claim that unpublished GitHub Actions checks or
live Groq requests were run. The integration's Platinum declaration remains explicitly
self-assessed. The review supports the corrected configuration, resource-ownership,
and HA compatibility contracts; live provider performance remains outside this test
evidence.
