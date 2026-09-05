# Generation, Assist, AI task and response-service review

Reviewed 2026-09-05. Assigned agent: `generation_review`. This preserves the original analysis and proposal. Line references describe the baseline. Implementation is complete; see the [implementation record](../architectural-improvements-2026-09-05.md).

The integration already has useful boundaries: typed API request objects, shared model capability lookup, Home Assistant-managed ChatLog tool execution, an explicit tool iteration ceiling, executor-based attachment reads, translated user-facing errors, and per-service request identities. The findings below prevent an unqualified reliability or performance endorsement.

## Confirmed changes, in priority order

### G1 — P1: Treat image input separately from text context estimates

Evidence: `custom_components/groq/text_generation.py:452-489` serializes the complete request, including inline image data URLs, and divides its UTF-8 size by four. Both Assist and AI tasks call this gate. The actual helper, extracted with Python AST and executed unchanged, estimated a 400 KiB image payload at **136,573 tokens**, exceeding a 131,072-token context before any completion allowance. This conflicts with the accepted attachment limits in `attachments.py:13-15`. Base64 bytes represent image transport, not text tokens. Serializing the same image also adds avoidable event-loop work.

Bounded fix: estimate textual messages, tool definitions and schemas without treating image data as text. Use a documented model-specific image token estimate where available; otherwise avoid a definitive local rejection based on image bytes and let the provider enforce its context limit. Rename the existing helper so it does not promise an upper bound that the heuristic cannot guarantee. Retain separate byte-size limits.

Acceptance tests: realistic inline image above 400 KiB with a short prompt is not rejected because of base64 length; genuinely excessive text/schema is still rejected; image count/size safeguards still apply; text-only completion accounting is unchanged.

### G2 — P1: Enforce the selected service's response-cache preference

Evidence: `runtime.py:120-128` enables the account-level `PROMPT_CACHING` feature when any text service opts in. `services.py:642-672` checks only that feature and model eligibility. A second eligible service with `prompt_caching=False` therefore reads and writes cached results. The cache key includes service identity, which prevents cross-service response mixing but does not preserve each service's opt-out.

Bounded fix: pass the selected service's effective cache policy into cache eligibility. Preserve any explicitly supported legacy account configuration, while an explicit service opt-out must win. Construct cache keys only when caching is eligible.

Acceptance tests: two services on one account, same eligible model, one enabled and one disabled; duplicate calls to the disabled service invoke the API twice while the enabled service reuses its response. Include explicit legacy-policy precedence and disabled-cache key-generation coverage.

### G3 — P2: Preserve configured structured-output defaults through HA service validation

Evidence: `services.py:172-191` supplies schema defaults `schema_name="response"` and `strict=False`. After Home Assistant validates a call, `_service_value` at lines 427-436 sees those injected keys as explicit call overrides. The configured service schema name and strictness used at lines 1048-1054 are therefore ignored when users omit these fields. Direct handler tests do not reproduce the schema-injection step.

Bounded fix: remove these defaults from both service schemas and resolve defaults in the request builder after explicit call overrides and service settings. Keep both existing service names and behavior.

Acceptance tests: invoke the registered service or apply its real schema before invoking the handler. Omitted fields inherit a configured custom name and `strict=True`; explicit `strict=False` and explicit names override; an unconfigured service still uses existing fallback values.

### G4 — P2: Apply one structured-result validation policy across AI tasks and response services

Evidence: `ai_task.py:499-500` loads a service-level schema. The tool branch validates it at lines 511-517, but the native structured branch validates only `task.structure` at lines 555-563. `api.py:627-654` only parses JSON. A non-strict provider response can consequently return the wrong service-level shape as successful data. The fallback at `ai_task.py:455-482` also ignores the effective service schema entirely, so when the chosen model lacks native structured output the configured schema can silently become free text.

Direct response services have the same validation gap: `services.py:1087-1101` takes `async_generate_structured` data straight into the cache and response without validating the supplied/configured schema. `_handle_generate_structured` at lines 1106-1108 delegates to the same handler with service-schema inference disabled; an explicitly supplied schema follows the same unchecked path.

Bounded fix: compute the effective schema once; centralize parsing and local validation for native, tools, and fallback outcomes, including direct response services before caching. Pass the service schema to JSON fallback instructions and validate its result. Keep task-level structure precedence. JSON Schema validation is synchronous CPU work, not inherently blocking I/O; measure representative schema/result complexity before deciding whether executor offload or validator reuse is justified. Do not add network reference fetching.

Acceptance tests: valid and invalid service schemas/results across native, tool and fallback paths; both direct text/structured actions reject wrong-shaped data and do not cache it; task structure still takes precedence; fenced fallback JSON; malformed JSON; unresolved local references; explicit no-schema text output. Verify the normal non-strict service-schema case, not just strict mode.

### G5 — P2: Select retained history before loading image attachments

Evidence: `conversation.py:338-383` resolves every historic user attachment; history is trimmed only at lines 376-379. Lines 380-390 resolve the current attachments again even when that user content was already converted. The helper runs again on each tool iteration. A missing attachment in a turn older than the retained history can therefore abort an unrelated current request; long image conversations repeatedly reread files that will be discarded.

Bounded fix: select retained history and preserve complete tool-call/result relationships before asynchronous attachment conversion. Reuse current-turn attachment conversion within a generation request/turn. Preserve the current user request if a large tool batch consumes the message budget; do not solve this with a persistent unbounded image cache.

Acceptance tests: discarded historic attachment is never opened; a missing retained attachment still gives the intended translated error; current attachment is read once per turn; text and image turns retain ordering; multiple tool results and a history cutoff do not orphan results or unnecessarily remove the current user request.

### G6 — P2: Reject malformed tool arguments before dispatch

Evidence: `conversation.py:468-481` converts malformed JSON and invalid argument types into `{}`, then returns executable `llm.ToolInput`. Valid JSON that is an array, scalar or null is not checked for dictionary shape. The test at `tests/components/groq/test_foundation.py:2003-2052` currently asserts malformed input becomes an empty argument object. For tools with optional/default arguments, this can invoke a valid default operation even though the model supplied an invalid call.

Bounded fix: validate that decoded arguments are a JSON object before dispatch; return a controlled model/tool error or fail the turn without executing the malformed call. Validate missing or duplicate IDs rather than collapsing unrelated calls onto the tool name. Retain legitimate empty-object calls.

Acceptance tests: malformed JSON, array/null/scalar arguments and duplicate IDs do not execute tools; valid `{}` still works; separate same-name calls with distinct IDs execute independently; Assist and AI tasks share the behavior.

### G7 — P2: Make explicit reasoning overrides control capability validation

Evidence: `services.py:481-494` ORs call values with configured values, while `_request_options` at lines 497-538 uses call-first precedence. An explicit `include_reasoning=False` can therefore be rejected against a non-reasoning model merely because the selected service configured it true, even though the outgoing request omits reasoning.

Bounded fix: derive capability checks from the resolved request/options, using the same precedence as payload generation. Apply the same rule to supported ways of clearing optional reasoning fields.

Acceptance tests: call-level false overrides a true service default without a reasoning capability error; omitted call value inherits the default and is checked; an actually enabled reasoning option still rejects an unsupported model.

### G8 — P2: Bound local media reads and move large transformations off the event loop

Evidence: `services.py:825-853` and `899-929` check file size and then separately call unbounded `Path.read_bytes`, with no post-read bound. A file that grows between stat and read can exceed the intended 20/25 MiB limit. The image path and camera path then base64-encode on the event loop (`services.py:675-678`, 822, 853), and vision cache keys serialize/hash full image URLs even when caching is disabled (1139-1150 and 1191-1202). Attachment reads already run their encoding in an executor, providing a local pattern to reuse.

Bounded fix: shared executor helper performs a bounded read of at most limit+1 bytes and validates actual bytes read; include MIME detection/encoding where relevant. Preserve allowlist and user authorization checks. Skip cache-key construction for ineligible requests. Coordinate attachment-reader changes with the transport/media assignment.

Acceptance tests: simulate a growing file and confirm no over-limit API request; image/audio size errors stay translated; valid media paths and authorization checks behave identically; oversized reads remain bounded; expensive encoding runs through the executor; no image cache key is built when caching is off.

## Best-practice improvement requiring careful scope

### G9 — P2/P3: Use and complete AI task ChatLog consistently

Evidence: no-tools AI branches at `ai_task.py:455-482` and `535-572` use direct instructions/attachments and do not append final `AssistantContent`; the tool branch does both. A supplied LLM API with a prompt but an empty tool list also takes the no-tools path, discarding that prepared prompt.

Home Assistant's [AI task developer guidance](https://developers.home-assistant.io/docs/core/entity/ai-task/) recommends using ChatLog context. Current [AITaskEntity implementation](https://github.com/home-assistant/core/blob/dev/homeassistant/components/ai_task/entity.py) prepares its system/API data and user message before calling the integration. However, the public [generation API](https://github.com/home-assistant/core/blob/dev/homeassistant/components/ai_task/task.py) creates a fresh ChatSession and does not accept a continuation ID. This is a prompt-consistency and observability gap; it is **not evidence that ordinary public AI task multi-turn continuation is broken**, nor an unconditional platform-contract violation.

Bounded fix: share a chat-message adapter across tool and no-tool branches, define how the configured service prompt composes with HA's prepared prompt, and append one completed assistant result for successful native/fallback calls. Preserve JSON validation and avoid duplicating the current user message.

Acceptance tests: real ChatLog through `internal_async_generate_data`, including an LLM API with an API prompt and zero tools; generated request includes the required prompt; final assistant content is recorded once for native JSON, fallback JSON and plain text; task result shape remains unchanged.

## Optional cleanup assignments

- **G10: Move shared chat adaptation out of a platform module.** `ai_task.py:26-32` imports five private helpers/constants from `conversation.py`; move narrowly shared conversion/tool helpers into a neutral module. This makes platform dependencies and ownership clearer without merging the distinct HA entrypoints.
- **G11: Remove production-dead compatibility helpers.** `conversation.py:268-319` contains a synchronous `_chat_log_messages` implementation used only in tests and largely duplicates the live async helper. `text_generation.py:233-238`, `490-497` likewise defines `service_prompt_caching`, `is_reasoning_model`, and `is_prompt_caching_model` with no production callers. Prefer using the cache helper in G2 if appropriate; otherwise remove unused implementations and move useful test assertions onto real call paths. Do not remove the mixed Probatio/Voluptuous serializer support while the declared HA version range needs it.
- **G12: Consolidate duplicated vision action handling.** `services.py:1111-1160` and `1163-1212` duplicate request construction, cache access, client invocation and result shape; parameterize the existing factory with the capability and cache namespace, preserving OCR's prompt/schema and feature checks. The two text service schemas at lines 172-191 are identical and can share a definition.
- **G13: Share resolved text request options.** `conversation.py:661-715`, `ai_task.py:227-334`, and `services.py:497-538` repeatedly map the same generation settings and validation. Extract typed option resolution with explicit call/service/account precedence, then keep platform-specific prompt/history/structured handling separate. This should follow the behavioral regressions above, with parity tests for those regressions, rather than precede them as a sweeping refactor.
- **G14: Test real boundaries and organize by behavior.** The concentrated `test_foundation.py` and `test_coverage_gaps.py` suites heavily call private helpers with Dummy/SimpleNamespace objects. Retain useful unit tests, but put new service-default regressions through the registered schema and AI/Assist regressions through real ChatLog behavior. Split future tests by service/entity/transport behavior; avoid merely preserving tests for otherwise unused production functions to maintain a coverage percentage.

## Validation and assignment boundaries

Read-only source review covered the five assigned modules plus relevant attachment, API, cache, runtime, flow-schema and test call sites. An isolated execution of the actual token estimator confirmed G1. No full test run or live provider/HA execution was performed for this review; acceptance tests above are implementation requirements, not claimed results.

Suggested execution order: G1-G4 and G6 first; G5/G7/G8 next; G9 after prompt semantics are settled; combine only tightly related cleanup (G10/G11 with chat changes, G12 with vision changes, G13 after behavior is covered). Keep media/runtime shared-file ownership coordinated with the other assignments. All changes require focused regression tests and the repository's required verification before a push.
