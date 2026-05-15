# Groq Integration Architecture

This document defines the target architecture for expanding the Groq Home
Assistant integration beyond the current text-to-speech implementation. It uses
`api_spec.md` as the baseline API contract and follows current Home Assistant
developer guidance for config entries, typed runtime data, platform forwarding,
entity properties, response services, diagnostics, and conversation/LLM support.

The requested feature set is:

- Text generation
- Speech-to-text
- Text-to-speech
- OCR and image recognition
- Reasoning
- Structured outputs
- Local response caching

## Design Goals

- One Groq config entry represents a Groq account/project and API base URL, not
  one model or one feature.
- Features are opt-in modules that can be enabled, disabled, added, or removed
  without rewriting the whole integration.
- Home Assistant-native surfaces are used where they exist: `tts`, `stt`, and
  `conversation` entities. Non-entity operations use response services.
- API transport, authentication, error handling, rate limits, diagnostics, and
  model discovery are shared across all features.
- Prompt text, image bytes, audio bytes, and API keys are never logged or exposed
  in diagnostics.
- Existing TTS behavior remains compatible while being moved behind the shared
  client and feature registry.

## Home Assistant Fit

Current Home Assistant guidance affects the architecture in these ways:

- Store shared runtime objects in typed `ConfigEntry.runtime_data`.
- Use `async_forward_entry_setups` and `async_unload_platforms` for platform
  setup and teardown.
- Keep entity properties memory-only; network I/O belongs in async action
  methods such as TTS generation, STT stream processing, and conversation
  handling.
- Use `ConversationEntity` and the Home Assistant LLM API for Assist/text
  generation that may control Home Assistant.
- Use `SpeechToTextEntity` for streaming speech-to-text.
- Use `TextToSpeechEntity` for `tts.speak` and support streaming TTS later when
  the Groq API and Home Assistant API shape make that practical.
- Use response services with `SupportsResponse.ONLY` for direct automation calls
  that return generated text, structured JSON, OCR results, or image analysis.
- Use repairs for user-actionable setup issues such as revoked credentials,
  disabled models, deprecated model IDs, or features enabled without a capable
  model.

## Proposed File Layout

```text
custom_components/groq/
  __init__.py
  api.py
  capabilities.py
  config_flow.py
  const.py
  diagnostics.py
  errors.py
  feature_registry.py
  model_registry.py
  prompt_cache.py
  rate_limit.py
  runtime.py
  services.py
  services.yaml
  tts.py
  tts_engine.py
  stt.py
  conversation.py
  vision.py
  structured.py
  translations/
```

`tts_engine.py` can either be retained as the TTS adapter during migration or
folded into `api.py` plus `tts.py` after compatibility tests are in place.

## Runtime Model

`__init__.py` should create one runtime object per config entry:

```python
@dataclass(slots=True)
class GroqRuntimeData:
    client: GroqApiClient
    model_registry: GroqModelRegistry
    feature_registry: GroqFeatureRegistry
    rate_limiter: GroqRateLimiter
    prompt_cache: GroqPromptCache
```

The config entry data should contain durable setup data:

- API key
- Base URL, defaulting to `https://api.groq.com/openai/v1`
- Entry/account label when no account identity endpoint is available
- Entry schema version

The config entry options should contain changeable behavior:

- Enabled features
- Default models per feature
- Default TTS voice/format/vocal directions/normalization
- Default STT language/model
- Default conversation model, system prompt, and Home Assistant LLM API setting
- Vision/OCR model defaults
- Reasoning settings
- Structured output defaults
- Prompt cache settings
- Local free-tier guard settings

`PLATFORMS` should become dynamic:

```python
FEATURE_PLATFORMS = {
    GroqFeature.TEXT_TO_SPEECH: {Platform.TTS},
    GroqFeature.SPEECH_TO_TEXT: {Platform.STT},
    GroqFeature.TEXT_GENERATION: {Platform.CONVERSATION},
}
```

OCR/image recognition, structured outputs, local response cache administration, and
one-shot text generation should be services because they return data and do not
map cleanly to long-lived Home Assistant state. If an image preview is later
useful, add an optional `image` platform entity for the last generated/analyzed
image metadata only; do not make OCR itself an entity.

## Feature Registry

Define feature IDs once:

```python
class GroqFeature(StrEnum):
    TEXT_GENERATION = "text_generation"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    VISION = "vision"
    OCR = "ocr"
    REASONING = "reasoning"
    STRUCTURED_OUTPUTS = "structured_outputs"
    PROMPT_CACHING = "prompt_caching"
```

Each feature descriptor should include:

- Feature ID and display translation key
- Required model capabilities
- Home Assistant platform, if any
- Services to register, if any
- Options schema fragment
- Validation callback
- Diagnostics summary callback

This lets the options flow render a multi-select "enabled features" field and
only show feature-specific options when that feature is enabled. On options
change, reload the entry. Reloading unloads disabled platforms and services,
then forwards setup only for the currently enabled platforms.

## Shared API Client

`GroqApiClient` should be the only module that performs Groq HTTP calls.

Responsibilities:

- Use Home Assistant's shared aiohttp session.
- Apply bearer-token authentication and the integration user agent.
- Build endpoint URLs from the base URL plus endpoint paths.
- Provide `_request_json` and `_request_bytes` helpers.
- Preserve cancellation.
- Map 401/403 to `ConfigEntryAuthFailed` when credentials are invalid.
- Map 429 to a Groq rate-limit exception with `retry-after` and reset headers.
- Parse JSON error bodies without logging secrets or user content.
- Emit structured response metadata for diagnostics without storing payloads.

Recommended public methods:

```python
async def async_list_models() -> list[GroqModel]
async def async_generate_text(request: TextGenerationRequest) -> TextGenerationResult
async def async_transcribe(request: TranscriptionRequest) -> TranscriptionResult
async def async_speech(request: SpeechRequest) -> bytes
async def async_analyze_image(request: VisionRequest) -> VisionResult
```

Before implementing features beyond TTS, extend `api_spec.md` with the exact
Groq endpoint, request fields, response fields, rate-limit dimensions, supported
models, and error behavior for that feature. The architecture should not encode
undocumented payload fields directly into platform code.

## Model And Capability Registry

Model discovery should become capability-aware:

```python
@dataclass(frozen=True, slots=True)
class GroqModel:
    model_id: str
    active: bool
    owned_by: str | None
    context_window: int | None
    max_completion_tokens: int | None
    capabilities: frozenset[GroqCapability]
```

Capabilities should be inferred from:

- Built-in capability tables from `api_spec.md`
- `/models` discovery response
- Future Groq model metadata, if it exposes modalities or capabilities

Built-in models remain available when discovery fails. Discovered models should
not be offered for a feature unless they are known or inferred to support that
feature.

## Feature Surfaces

### Text Generation

Expose text generation in two ways:

- `conversation.py`: a `ConversationEntity` for Assist and chat-style use.
- `groq.generate_text`: a response service for automations/scripts that need a
  generated text value.

The conversation entity should support Home Assistant's LLM API tools when the
user explicitly enables control. The direct service should not control Home
Assistant; it should only return generated content.

### Speech-To-Text

Add `stt.py` with `SpeechToTextEntity`.

The entity should advertise supported languages, audio formats, codecs, bit
rates, sample rates, and channels from memory. `async_process_audio_stream`
should stream or upload audio according to the Groq API contract once documented
in `api_spec.md`.

### Text-To-Speech

Keep `tts.py` as the Home Assistant TTS platform and migrate request execution
to `GroqApiClient.async_speech`.

Current Orpheus-specific behavior remains a TTS feature option:

- Voice
- Response format
- Vocal directions
- Audio normalization
- Audio cache size
- Free-tier guard

The current 200-character Orpheus limit should move into model capability data
so future TTS models can use their own limits.

### OCR And Image Recognition

Expose OCR/image recognition as response services:

- `groq.analyze_image`
- `groq.extract_text_from_image`

Inputs should use Home Assistant-friendly selectors and references, such as
media source IDs, URLs, or camera/image entity IDs. The service layer resolves
those references into bytes or URLs, then calls `GroqApiClient.async_analyze_image`.

Responses should include:

- Extracted text, if requested
- Summary/description
- Detected objects or labels when the model returns them
- Confidence or raw model metadata only when documented and useful

Do not persist analyzed images by default.

### Reasoning

Reasoning is a model capability and request mode, not a separate platform.

Expose it as options on:

- Conversation entity
- `groq.generate_text`
- `groq.generate_structured`

Reasoning configuration should be explicit and model-gated. If a non-reasoning
model is selected with reasoning enabled, validation should block setup/options
or the service call before sending the API request.

### Structured Outputs

Expose structured outputs as:

- `groq.generate_structured` response service
- Optional schema mode on `groq.generate_text`

The service accepts a JSON schema and returns validated JSON. Validation should
happen both before the request and after the response:

- Reject invalid schemas locally.
- Parse the model response as JSON.
- Validate the parsed result against the requested schema.
- Return a clear Home Assistant error if validation fails.

Use structured parsing libraries instead of ad hoc string extraction.

### Prompt Caching

Prompt caching has two layers and the UI must name them clearly:

- Groq prompt caching: provider-side behavior and usage metadata documented by
  Groq. The integration should surface cached-token metadata when the API returns
  it and only send cache-control fields if `api_spec.md` documents them.
- Local response cache: optional Home Assistant memory cache for deterministic
  calls. This is not the same as Groq prompt caching and should remain disabled
  by default for text generation unless the user opts in.

`prompt_cache.py` should provide shared key normalization and cache namespaces:

- `tts_audio`
- `text_generation`
- `structured_outputs`
- `vision`

Cache keys must include model, feature, relevant options, prompt hash, and input
hashes. Never include raw API keys, raw prompt text, audio bytes, or image bytes
in keys exposed to logs or diagnostics.

## Services

Register services from `services.py` during `async_setup_entry`, using
`SupportsResponse.ONLY` for services that return generated data.

Proposed service set:

- `groq.generate_text`
- `groq.generate_structured`
- `groq.analyze_image`
- `groq.extract_text_from_image`
- `groq.clear_cache`
- `groq.list_models`

Service handlers should locate the target config entry, verify the required
feature is enabled, validate the selected model capability, call the shared
client, and return a typed response dictionary.

`services.yaml` should define selectors for model, feature, schema, prompt,
image source, and target entry. Prefer service responses over writing generated
content into entity state.

## Options Flow

Use a staged options flow instead of one large form:

1. Feature selection
2. Text generation/conversation options
3. Speech-to-text options
4. Text-to-speech options
5. Vision/OCR options
6. Caching and rate-limit options

Only show steps for enabled features. The final options payload should include
all feature defaults so disabling and re-enabling a feature can preserve the
user's previous settings.

## Diagnostics And Repairs

Diagnostics should include:

- Enabled features
- Base URL with API key redacted
- Selected models per feature
- Cache sizes and hit/miss counters
- Rate-limit guard configuration
- Last API error class/status per feature

Diagnostics must not include:

- API keys
- Prompt text
- Conversation history
- Structured output payloads
- Image/audio bytes
- OCR extracted text unless the user explicitly attaches it to an issue

Repairs should be created for:

- Authentication failure
- Enabled feature has no capable model
- Selected model is no longer returned by discovery
- Deprecated model ID when a replacement is known
- Invalid feature combination, such as reasoning enabled on a non-reasoning model

## Error Handling

All feature code should raise integration-specific exceptions from `errors.py`:

- `GroqAuthError`
- `GroqRateLimitError`
- `GroqModelAccessError`
- `GroqFeatureNotEnabled`
- `GroqUnsupportedCapability`
- `GroqResponseValidationError`
- `GroqTransientError`

Platform/service handlers translate these into Home Assistant exceptions or
conversation errors. Error messages should be actionable and should never contain
secrets or full user payloads.

## Migration Plan

1. Add shared runtime scaffolding, feature registry, and config constants while
   keeping TTS behavior unchanged.
2. Migrate `GroqTTSEngine` HTTP logic into `GroqApiClient.async_speech`.
3. Change config entry semantics from TTS-specific to account/project-specific,
   preserving existing entries with a migration that enables only TTS.
4. Add response service registration and service tests.
5. Add text generation and structured output services.
6. Add conversation entity with optional Home Assistant LLM API tools.
7. Add speech-to-text entity.
8. Add vision/OCR services.
9. Add provider prompt-cache metadata and optional local response cache.
10. Expand diagnostics, repairs, and quality-scale coverage.

## Test Strategy

For each feature:

- Unit-test request payload construction.
- Unit-test response parsing and validation.
- Unit-test model capability gating.
- Unit-test config/options flow validation.
- Unit-test service response shape.
- Unit-test diagnostics redaction.
- Unit-test authentication and rate-limit handling.

For the shared client:

- Verify headers and user agent.
- Verify cancellation is preserved.
- Verify JSON error parsing.
- Verify binary response handling for TTS.
- Verify 401/403 reauth behavior.
- Verify 429 retry/reset details.

Use the existing `scripts/test` Docker harness for Home Assistant parity.

## References

- `api_spec.md`
- Home Assistant developer docs: config entries, runtime data, platform
  forwarding/unloading, TTS entity, STT entity, conversation entity, LLM API,
  image entity, response services, diagnostics, and repairs.
