# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### 🐛 Bug fixes
- None

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
