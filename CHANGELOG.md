# Changelog

All notable changes to this project will be documented in this file.

## v1.3.2 - 2026-07-06

### 🚧 Breaking changes
- None

### ✨ New features
- Added expanded Groq TTS controls for playback format, sample rate, and speed, with direct native Groq output used for single-part unprocessed requests. (#31)

### 🐛 Bug fixes
- None

### 🔧 Improvements
- Kept ffmpeg handling for normalization, Long TTS stitching, and WAV compatibility repair while allowing compatible unprocessed TTS requests to use Groq's native output format directly. (#31)
- Added Indonesian (`id-ID`) to STT language hint selectors so it is available when configuring STT services and calling the `groq.transcribe_audio` action. (#33)

### 🔄 Other changes
- Updated README guidance, translations, diagnostics, and regression coverage for expanded TTS controls and STT language option parity. (#31, #33)
- Bumped the integration manifest version to `1.3.2`.

## v1.3.1 - 2026-06-28

### 🚧 Breaking changes
- Groq response actions now require explicit target selection: service-level actions require `service_id`, and account-level actions require `config_entry_id`, preventing automations from silently changing behavior as accounts or configured services are added. (#29)

### ✨ New features
- Added support for the Groq `qwen/qwen3.6-27b` vision model and improved dynamic vision model discovery from Groq model metadata. (#26)

### 🐛 Bug fixes
- Replaced implicit single-account or single-service action fallback behavior with explicit action validation so service calls fail clearly when the required target is missing. (#29)

### 🔧 Improvements
- Refreshed discovered model handling so successful Groq `/models` refreshes replace stale built-ins, allowing newly visible models to appear and removed models to stop passing runtime validation. (#26)
- Moved Groq service action names and descriptions from `services.yaml` into integration translations in line with current Home Assistant guidance. (#27)
- Added Material Design Icons for every Groq service action so Home Assistant can show action-specific icons in the UI. (#28)

### 🔄 Other changes
- Updated README guidance for explicit Groq service and account action selection. (#29)
- Expanded regression coverage for dynamic model registry behavior, service translation metadata, service icons, and explicit action target validation. (#26, #27, #28, #29)
- Bumped the integration manifest version to `1.3.1`.

## v1.3.0 - 2026-06-25

### 🚧 Breaking changes
- Raised the minimum supported Home Assistant version to `2026.6.0` to match the patched floor for CVE-2026-54317 / GHSA-x84v-g949-293w. (#23)

### ✨ New features
- None

### 🐛 Bug fixes
- Fixed repairs platform registration by adding an aborting repair flow for non-fixable Groq issues. (#22)

### 🔧 Improvements
- None

### 🔄 Other changes
- Updated installation documentation now that Groq is available as a default HACS repository. (#24)
- Expanded regression coverage for repairs platform fix-flow registration. (#22)
- Bumped the integration manifest version to `1.3.0`.

## v1.2.3 - 2026-06-08

### 🚧 Breaking changes
- None

### ✨ New features
- None

### 🐛 Bug fixes
- Fixed TTS vocal direction option handling so users can clear defaults, select explicit None, and avoid storing or speaking invalid sentence-length directions. (#19)
- Fixed the Home Assistant 2026.6 config-entry reload deprecation by replacing update-listener reload behavior with explicit reload paths from config and options flows. (#20)

### 🔧 Improvements
- None

### 🔄 Other changes
- Expanded regression coverage for TTS vocal direction validation and Home Assistant config-entry reload behavior. (#19, #20)
- Bumped the integration manifest version to `1.2.3`.

## v1.2.2 - 2026-05-30

### 🚧 Breaking changes
- None

### ✨ New features
- None

### 🐛 Bug fixes
- Fixed MP3 TTS playback on HomePod and Apple TV targets by converting MP3 output with a HomePod-tested 44.1 kHz mono 128 kbps profile. (#17)

### 🔧 Improvements
- None

### 🔄 Other changes
- Added regression coverage for the HomePod-safe MP3 ffmpeg conversion profile. (#17)
- Bumped the integration manifest version to `1.2.2`.

## v1.2.1 - 2026-05-30

### 🚧 Breaking changes
- None

### ✨ New features
- None

### 🐛 Bug fixes
- Fixed HomePod and Apple TV TTS playback by validating Groq WAV output before serving it directly and rewriting malformed, non-WAV, or non-16-bit PCM WAV payloads through the ffmpeg WAV compatibility profile. (#15)

### 🔧 Improvements
- Optimized prompt cache expiry handling with heap-backed expiry bookkeeping and stale-entry compaction, avoiding full-cache scans on cache hits. (#14)
- Deferred Home Assistant camera and media-source helper imports until service paths need them, reducing import-time overhead during service registration. (#14)

### 🔄 Other changes
- Expanded tests for prompt-cache stale expiry compaction and TTS WAV compatibility handling. (#14, #15)
- Bumped the integration manifest version to `1.2.1`.

## v1.2.0 - 2026-05-17

### 🚧 Breaking changes
- None

### ✨ New features
- Added opt-in Compound built-in tool controls with validation for dedicated options and raw `compound_custom.tools.enabled_tools` payloads. (#10)
- Added optional Long TTS announcements that split long Orpheus text into Groq-sized chunks, synthesize them sequentially, and stitch the result with ffmpeg. (#11)
- Added selectable TTS playback output formats, keeping Groq Orpheus requests in WAV while allowing local ffmpeg conversion to MP3 or FLAC for speaker compatibility. (#12)

### 🐛 Bug fixes
- Handled Groq TTS request timeouts as expected network failures with clearer unavailable-state and Home Assistant error reporting. (#8)
- Preflighted ffmpeg before spending Groq quota for converted TTS output and kept missing-ffmpeg repair issues aligned with the configured audio processing state. (#12)

### 🔧 Improvements
- Migrated TTS synthesis into the shared `GroqApiClient`, removing the standalone TTS engine and reusing the common HTTP session, rate-limit, and network-error paths. (#9)
- Tightened dynamic TTS model capability inference so unsupported TTS-looking models are not offered in voice/model pickers. (#9)
- Sent explicit empty Compound tool allow-lists by default and `Groq-Model-Version: latest` only when latest-only Compound tools are enabled. (#10)
- Disabled Long TTS and audio normalization options when ffmpeg is unavailable, and validated Long TTS batches before sending partial requests. (#11)

### 🔄 Other changes
- Updated README, architecture notes, quality-scale metadata, translation strings, and TTS benchmark helpers for the new TTS and Compound tool behavior. (#9, #10, #11, #12)
- Expanded tests for TTS timeout handling, shared API-client synthesis, Compound tools, Long TTS chunking/stitching, TTS output conversion, diagnostics, translations, and coverage paths. (#8, #9, #10, #11, #12)
- Bumped the integration manifest version to `1.2.0`.

## v1.1.0 - 2026-05-16

### 🚧 Breaking changes
- None

### ✨ New features
- Added Home Assistant AI Task support for Groq text and structured data generation. (#5)
- Added Home Assistant LLM tool-calling support for Assist and AI Tasks, including tool request/result conversion and guardrails for unsupported models. (#5)
- Added multimodal attachment handling for supported Assist and AI Task image inputs. (#5)

### 🐛 Bug fixes
- Hardened Groq service input validation, reserved request-body option handling, and Assist context handling. (#3)

### 🔧 Improvements
- Improved Groq API and TTS request performance by preloading Home Assistant's shared aiohttp session helper and reusing the managed session. (#6)
- Added expanded translation coverage for Bulgarian, Danish, English regional variants, Spanish, Estonian, Finnish, French, Hungarian, Italian, Lithuanian, Latvian, Norwegian Bokmal, Dutch, Polish, Brazilian Portuguese, Romanian, and Swedish. (#4)
- Updated README documentation for AI Tasks, image/audio workflows, and current Home Assistant development expectations. (#2)

### 🔄 Other changes
- Added a TTS rate-limit benchmark helper for local performance checks. (#6)
- Declared the `jsonschema` runtime dependency required for AI Task structured output validation. (#5)
- Expanded tests for AI Tasks, tool calls, translation coverage, manifest metadata, rate-limit handling, and preload fallbacks. (#3, #4, #5, #6)
- Bumped the integration manifest version to `1.1.0`.

## v1.0.2 - 2026-05-12

### 🚧 Breaking changes
- None

### ✨ New features
- None

### 🐛 Bug fixes
- None

### 🔧 Improvements
- None

### 🔄 Other changes
- Bumped the integration manifest version to `1.0.2`.

## v1.0.1 - 2026-05-12

### 🚧 Breaking changes
- None

### ✨ New features
- None

### 🐛 Bug fixes
- Improved config flow handling so API keys are no longer hashed during validation.

### 🔧 Improvements
- Added Home Assistant brand assets, including dark-theme and high-resolution icon and logo variants.

### 🔄 Other changes
- None

## v1.0.0 - 2026-05-10

### 🚧 Breaking changes
- None

### ✨ New features
- Added the initial Groq Home Assistant custom integration with config flow support.
- Added Groq-backed Assist conversation, text generation, structured generation, image analysis, speech-to-text, and text-to-speech services.
- Added service subentries, diagnostics, repair flows, runtime helpers, model registry, feature registry, prompt caching, and rate-limit handling.
- Added Home Assistant metadata, HACS metadata, integration icons, translations, and documentation.

### 🐛 Bug fixes
- Fixed HACS display metadata, validation metadata, hassfest checks, pre-commit failures, and CI environment checks.

### 🔧 Improvements
- Tuned setup key validation, default service configuration, and Groq service configuration.
- Uplifted the integration quality-scale metadata.
- Simplified README content and added security, contribution, architecture, and API documentation.

### 🔄 Other changes
- Added repository automation, issue templates, release workflows, quality-scale validation, Docker test harnesses, strict typing checks, and pytest coverage.
- Bumped pytest in the Docker development requirements. (#1)
