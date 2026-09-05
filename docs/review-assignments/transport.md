# Transport, caching and media review assignment

Reviewed 2026-09-05. Owner: `transport_review` subagent. Scope: `api.py`,
`rate_limit.py`, `prompt_cache.py`, `attachments.py`, `errors.py`, `stt.py`,
`tts.py`, their callers, and relevant existing tests. This is an implementation
brief preserved from the original review. References below describe the
checkout before implementation. Implementation is complete; see the
[implementation record](../architectural-improvements-2026-09-05.md).

## Assessment

The integration already has a strong transport foundation: Home Assistant's
shared aiohttp session, asynchronous requests, explicit timeouts, disabled
redirect following, bounded provider response reads, bounded model-detail
concurrency, translated exception classes, and explicit cancellation propagation.
Attachment disk I/O is moved to the executor. ffmpeg communication has a timeout
and kills/reaps a running process when cancelled. Speech input/chunk limits and
cache eviction tests exist.

Those strengths support maintainability and predictable resource use, but the
defects below prevent an unqualified reliability/performance endorsement.

## Confirmed defects and assigned changes

### T1 — P1: Model permissions incorrectly trigger account reauthentication

**Evidence:** `api.py:812`, `api.py:871`, and `api.py:939` treat both 401 and 403 as
authentication failure before examining the response. `_authentication_failed`
invokes the account reauthentication callback. `_create_model_access_issue` at
`api.py:1019-1035` only accepts 400/404. The documented Groq restricted-model
responses are 403 with `permissions_error` and codes
`model_permission_blocked_org` / `model_permission_blocked_project`. Those
restrictions apply independently of API keys. A blocked model therefore starts a
misleading account repair while other models remain usable.
[Groq model permissions](https://console.groq.com/docs/model-permissions).

**Change:** Preserve the fast 401 path. Classify bounded 403 responses and route
documented model restrictions to a model-access error/repair without invoking
reauthentication. Define a deliberate fallback for unknown or malformed 403
responses. Share classification across JSON, SSE, and audio transports.

**Coordination:** The lifecycle assignment owns repair identity and lifecycle.
Pass account context, requested model and a bounded permission reason into its
repair interface; do not introduce a second repair system here.

**Acceptance:** Parameterize the three transports over both documented 403 codes,
401, unknown 403, and malformed/oversized bodies. A known model restriction must
retain status/type, create the correct model repair, and never call reauth. A
subsequent allowed-model request must still work. Preserve existing 401-before-
body-limit behavior.

### T2 — P2: Rate-limit duration parsing and window selection are incorrect

**Evidence:** `rate_limit.py:123-159`. The duration parser supports one suffix,
while Groq documents `2m59.56s`. The guard selects the maximum of both reset
windows even when just one resource is exhausted. Thus token exhaustion can
incorrectly block until the unrelated daily request window resets. An invalid
`retry-after` returns immediately without considering useful exhausted-window
metadata. Numeric infinity can raise `OverflowError`; a suffixed NaN can raise
`ValueError` outside the existing conversion guard.
[Groq rate-limit headers](https://console.groq.com/docs/rate-limits).

**Change:** Parse complete finite nonnegative numeric/composite durations, round
up, and reject malformed inputs without escaping exceptions. Use only exhausted
resources' reset windows, falling back conservatively when their reset is absent.
Allow invalid `retry-after` to fall back to the valid reset metadata. Use ceiling
for the remaining user-facing wait at `rate_limit.py:101`.

**Acceptance:** Cover documented composite values, decimals, milliseconds,
whitespace, NaN/infinity, malformed combinations, and token-only/request-only/both
exhaustion. Verify local guards neither raise arbitrary numeric exceptions nor
block for an unrelated window.

**Direct reproduction:** Executing the checked-in helpers returned `None` for
`2m59.56s`; `inf` and `infs` raised `OverflowError`; `nans` raised `ValueError`.
For remaining requests=14370, remaining tokens=0, request reset=180s and token
reset=7.66s, the current guard selected 180 seconds instead of 8.

### T3 — P2: SSE errors lose status and can escape the integration error boundary

**Evidence:** `_request_stream`, `api.py:873-905`, calls strict `_decode_json`
before classifying HTTP errors. Plain-text/HTML 429 and 503 bodies therefore
become invalid-JSON errors with no HTTP status; 429 loses its specialized error
and 503 does not mark the client unavailable. JSON/audio already use tolerant
decoding for these paths. A list-shaped 429 also bypasses `GroqRateLimitExceeded`.
At line 890, invalid UTF-8 raises an untranslated `UnicodeDecodeError`.

**Change:** Reuse the bounded tolerant HTTP error classifier from T1. Handle 429
independently of body shape, preserve transient status/availability, and translate
invalid stream encoding to `GroqResponseError`. Keep successful SSE framing
separate from HTTP error-body parsing.

**Acceptance:** Test plain-text, empty, list and malformed JSON bodies for 429 and
503, asserting status, retry headers, availability and response-context exit.
Test invalid UTF-8 and retain existing malformed-JSON/cancellation tests.

### T4 — P2: Provider error messages are described as sanitized but are not

**Evidence:** `_api_error`, `api.py:1313-1328`, interpolates the entire provider
`error.message`, arbitrary `error`, or whole payload into an exception. `tts.py:882`
logs that exception, and STT logs a traceback at `stt.py:196`. The JSON response
ceiling still permits multi-megabyte exceptions and providers can echo request
content. `GroqApiError.payload` also retains the raw dictionary. This is a
concrete lack of sanitization; no real credential disclosure was observed.

**Change:** Define an error-data contract: bounded status/type/code plus a safe
user-facing message. Do not stringify arbitrary response dictionaries. Redact
known sensitive field values before any retained diagnostic data, and avoid
passing raw provider messages through routine logs. Preserve metadata needed by
model-access classification before applying the presentation policy.

**Acceptance:** Use synthetic provider bodies containing nested request text,
credential-shaped fields, non-string messages and very long strings. Assert
bounded exception/log output without those sensitive values, while status and
permission/rate-limit classification remain usable.

### T5 — P2: Prompt cache exposes mutable nested responses

**Evidence:** `prompt_cache.py:47` and `prompt_cache.py:63` copy only the outer
dictionary. Response services cache nested `data`, `usage`, and tool metadata
(`services.py:649-672`, `services.py:1087-1100`, `services.py:1150-1157`). Callers
can mutate the cached value via either the original response or a retrieved hit.

**Change:** Give cached responses an ownership boundary, using deep copies of
the supported JSON-shaped response data or serialized immutable storage. Keep
copy/serialization costs bounded by the cache memory policy in T8.

**Acceptance:** Mutate nested lists/dicts in both the original value and a cache
hit, then assert subsequent reads preserve the stored response. Existing LRU,
TTL, disabled-cache and expiry-heap compaction tests must still pass.

**Direct reproduction:** After storing `{'data': {'value': 1}}`, changing the
original nested value to 2 changed the cache. Changing a retrieved nested value
to 3 changed the next cache hit again.

### T6 — P2: Cancelled temporary-directory creation leaks a directory

**Evidence:** `tts.py:793-797` awaits executor `mkdtemp` before entering the
`try/finally` that removes it. Cancellation during that await does not stop the
executor operation and cleanup has neither started nor acquired the path. The
related write at `tts.py:799-801` can also continue after cancellation while
cleanup starts, creating a race between file writing and deletion.

**Change:** Move temporary resource ownership into a cancellation-safe helper.
Track executor completion, obtain/clean its result if the caller is cancelled,
and ensure file writes finish before removal. Remove the synchronous fallback
branches for missing Home Assistant executor APIs when extracting this helper;
real Home Assistant always supplies the API.

**Acceptance:** Use synchronization events to cancel deterministically during
directory creation and file writing, then assert there are no residual files,
cleanup is complete before return, and `CancelledError` propagates. Keep tests
for cancelling/timing out a running ffmpeg process.

**Direct reproduction:** The checked-in nested `stitch_audio_chunks` function
was extracted and executed with an executor-backed fake Home Assistant and a
delayed `mkdtemp`. Cancelling after the worker started left a created directory
after the function returned. The isolated proof cleaned up its own `/tmp` files.

### T7 — P2: Attachment limit is checked after an unbounded read

**Evidence:** `attachments.py:50-63` checks `stat`, performs `Path.read_bytes`,
then checks length. The second check prevents uploading a grown file but cannot
prevent allocating the entire grown file. The existing regression at
`test_security_regressions.py:183` validates rejection after growth, not bounded
allocation. A similar local-file path exists in `services.py:835-860`.

**Change:** Use a shared executor-side bounded regular-file reader that reads
at most limit+1 bytes, translates expected file I/O errors, and preserves caller
authorization/MIME rules. The attachment and service paths should reuse the
reader while keeping their distinct error presentation requirements.

**Acceptance:** A fake growing file must record a bounded read size; never use
unbounded `read()` or `read_bytes()`. Cover disappeared/unreadable files, exact
limit, one-byte overflow, multiple-image total/count limits and unchanged
allowlist/permission behavior.

**Coordination:** The generation/service assignment owns changes to `services.py`;
agree on the helper API before editing it.

### T8 — P2: Cache entry limits do not bound retained audio/response memory

**Evidence:** `api.py:752-755` only limits audio item count; `_speech_caches`
(`api.py:486-493`, `api.py:1073-1077`) maintains a separate cache per service.
Default `SpeechRequest.cache_max` is 256 and each accepted audio response can be
25 MiB. Thus the formal per-service ceiling is 6.25 GiB of audio, before keys and
other services. Typical short announcements are much smaller; this is a
worst-case bound, not a measured incident. The prompt cache also limits item
count rather than retained bytes (`prompt_cache.py:28-32`, `prompt_cache.py:70`).

**Change:** Add conservative byte budgets alongside entry limits, evict LRU
entries until both constraints hold, and skip caching a single oversized item.
Provide an account-wide ceiling across speech namespaces. Decide/document budget
constants without adding unnecessary UI controls. Keep the network response
ceilings, which solve a different part of resource management.

**Acceptance:** Insert controlled varying-size payloads and verify exact budget
accounting after replacement, hits, TTL expiry, eviction and clear. Cover multiple
speech namespaces and oversized uncached results. No large-memory test fixtures
are needed; patch the budget small.

### T9 — P3: TTS numeric validation accepts NaN

**Evidence:** `_normalize_speed`, `tts.py:247-257`, accepts `float('nan')` because
both range comparisons are false. It subsequently reaches the outbound speech
payload, after local quota accounting. `_normalize_sample_rate` also converts
fractional numeric values with `int`, silently truncating before membership
validation.

**Change:** Require finite speed and integral sample-rate values, preserving
documented numeric-string inputs. Reuse the same normalization contract in the
flow if it contains corresponding conversions.

**Acceptance:** Reject NaN/infinity, fractional rates and malformed values before
an API call or quota debit; retain supported numeric values and strings.

### T10 — P3: Outage log promises runtime retries that are not scheduled

**Evidence:** `api.py:1009` says calls will be retried by Home Assistant. JSON and
stream runtime failures propagate to their caller, and TTS returns an error after
its one transport retry. No retry is scheduled by `_mark_unavailable`. Home
Assistant's automatic retry guarantee applies to setup failures, not arbitrary
failed response services.
[Home Assistant setup failure handling](https://developers.home-assistant.io/docs/integration_setup_failures/).

**Change:** Describe the observed failure and recovery-on-next-request accurately.
Do not add automatic replay of generative/tool-calling requests to make the old
log message true.

**Acceptance:** Verify outage/recovery transition logs and no misleading replay
promise. Preserve cancellation and existing audio retry behavior.

## Optional improvements and further validation

These are bounded follow-up opportunities, not equivalent to confirmed production
failures above.

1. **Extract cohesive internal components:** `api.py` is 1,356 lines and combines
   payload models/builders, response normalization, transport and TTS caching/quota
   accounting. `tts.py` is 886 lines, with a roughly 330-line synthesis method.
   Prefer extracting speech cache/quota policy and ffmpeg temporary-resource
   management while fixing T6/T8. Avoid creating a generic transport framework.
2. **Remove genuinely redundant surfaces:** `StructuredOutputRequest` at
   `api.py:179` aliases a request class with no production callers; the
   `TextGenerationRequest.reasoning` flag at `api.py:107` is unused by payload
   construction; `FFMPEG_OUTPUT_ARGS` at `tts.py:276` is only consumed by tests.
   Confirm no documented compatibility promise, remove dead surfaces, and have
   tests validate `_ffmpeg_output_args` directly. Redundant subclass entries in
   exception tuples can be simplified when touching those paths.
3. **Public batch policy boundary:** `tts.py:727-733` probes the client's private
   `_check_local_tts_free_tier_batch` method with `getattr`/`callable`. Give it a
   stable public interface or an explicit injected policy rather than shaping
   production code around partial test doubles.
4. **Concurrent duplicate synthesis:** Two identical cache misses can issue two
   audio API calls and consume duplicate quota. Consider bounded in-flight
   coalescing only with explicit cancellation/unload ownership and tests; avoid
   serializing unrelated generation requests.
5. **Local quota semantics:** Usage is per service/voice and counted before the
   request, while internal audio retries are not counted separately. It is a
   conservative local guard rather than authoritative organization accounting.
   Clarify that boundary; changing to account/model accounting is a policy change
   requiring separate regression analysis and documentation.
6. **Streaming validation:** A successful HTTP SSE response containing an
   `error` object is currently ignored by `async_stream_text` because it lacks
   choices (`api.py:615-631`). Define expected provider error-event handling and
   completion/truncation semantics before enforcing new framing requirements.
7. **Expiry before cache eviction:** `GroqPromptCache.set` does not purge expired
   entries before enforcing capacity. With differing TTLs/LRU order this can
   evict a live entry while an expired entry remains. Purge before capacity
   enforcement as part of T8 and test with a fake monotonic clock.
8. **Media CPU/output budgets:** STT builds/copies a full in-memory WAV near the
   25 MiB ceiling, and ffmpeg `communicate` buffers unrestricted output. Profile
   these paths with representative Home Assistant hardware before selecting
   further limits or executor offloading; there is no measured event-loop latency
   or memory incident from this review.

## Validation status

Read all assigned production modules and relevant existing tests. Executed
isolated reproductions of the checked-in duration helpers, prompt-cache ownership
behavior and actual nested TTS temporary-directory function. Checked current
primary Groq documentation for rate headers and model permissions, and Home
Assistant documentation for executor work and setup retry/reauth semantics.
No full Home Assistant test suite, live Groq request, performance benchmark or
integration runtime was executed by this subagent. Implementation must use the
repository Docker harness and cover the regression acceptance cases above before
claiming these issues are fixed.
