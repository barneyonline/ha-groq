# Groq - Home Assistant Custom Integration

<!-- Badges -->
[![Release](https://img.shields.io/github/v/release/barneyonline/ha-groq?display_name=tag&sort=semver)](https://github.com/barneyonline/ha-groq/releases)
[![Stars](https://img.shields.io/github/stars/barneyonline/ha-groq)](https://github.com/barneyonline/ha-groq/stargazers)
[![License](https://img.shields.io/github/license/barneyonline/ha-groq)](LICENSE)

[![Tests](https://img.shields.io/github/actions/workflow/status/barneyonline/ha-groq/tests.yml?branch=main&label=tests)](https://github.com/barneyonline/ha-groq/actions/workflows/tests.yml)
[![Hassfest](https://img.shields.io/github/actions/workflow/status/barneyonline/ha-groq/hassfest.yml?branch=main&label=hassfest)](https://github.com/barneyonline/ha-groq/actions/workflows/hassfest.yml)
[![Quality Scale](https://img.shields.io/github/actions/workflow/status/barneyonline/ha-groq/quality-scale.yml?branch=main&label=quality%20scale)](https://developers.home-assistant.io/docs/integration_quality_scale_index)

[![HACS](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://hacs.xyz)
[![Open Issues](https://img.shields.io/github/issues/barneyonline/ha-groq)](https://github.com/barneyonline/ha-groq/issues)
![Development Status](https://img.shields.io/badge/development-active-success?style=flat-square)

Cloud-based Home Assistant integration for Groq AI services.

> [!IMPORTANT]
> This is an unofficial community project. It is not affiliated with, endorsed by, or supported by Groq.
>
> The integration uses Groq's cloud APIs. Feature availability, model availability, rate limits, and request options can vary by Groq account, project, and model.

## Supported service categories

- Text Generation services for Home Assistant Assist, AI Tasks, direct text generation, structured outputs, reasoning-capable models, prompt caching, and Groq Compound models
- Text-to-Speech services using Groq Orpheus models through Home Assistant `tts.speak`
- Image Recognition services for image analysis from automations and scripts
- Speech-to-Text services for Home Assistant voice pipelines

OCR support is present in the service architecture and model registry, but the setup flow currently exposes Image Recognition as the user-facing vision service.

## Key features

- Guided onboarding that asks only for a friendly integration name and a hidden Groq API key
- Service-specific subentry buttons for creating multiple named services under each Groq account
- Per-service API keys so usage can be tracked separately for Text Generation, Text-to-Speech, Image Recognition, or future service profiles
- Text Generation entities for Home Assistant Assist conversations and AI Task data generation
- Streaming Assist responses when Home Assistant requests streaming chat output
- Default Home Assistant-oriented system prompt for Text Generation services
- First-class Text Generation options for model, system prompt, temperature, maximum tokens, top-p, stop sequences, seed, service tier, streaming, structured outputs, reasoning, and prompt caching
- Advanced Text Generation request body passthrough for Groq chat completion options that do not need a dedicated UI control
- Model gating for reasoning, structured outputs, and prompt caching so unsupported models are not offered those settings
- Support for Groq Compound models: `groq/compound` and `groq/compound-mini`
- Structured output support through Groq `json_schema` response formats and Home Assistant AI Task schemas
- Image analysis response service using Groq vision-capable models
- Text-to-Speech entities with Orpheus voices, vocal direction presets, custom vocal directions, optional audio normalization, local cache sizing, and free-tier protection
- Diagnostics with API keys redacted

## Quick install (HACS)

1. HACS -> Integrations -> Custom repositories
2. Add `https://github.com/barneyonline/ha-groq` as an Integration repository
3. Install Groq and restart Home Assistant
4. Settings -> Devices & Services -> Add Integration -> Groq
5. Enter a friendly name and your Groq API key
6. Open the Groq integration page and choose the service-specific button you need, such as Add Text Generation, Add Speech-to-Text, Add Text-to-Speech, or Add Image Recognition

## Manual install

1. Download this repository
2. Copy `custom_components/groq` into your Home Assistant `custom_components` directory

```text
<homeassistant_config_dir>/
  custom_components/
    groq/
      __init__.py
      manifest.json
      ...
```

3. Restart Home Assistant
4. Settings -> Devices & Services -> Add Integration -> Groq
5. Enter a friendly name and your Groq API key
6. Open the Groq integration page and choose the service-specific button you need, such as Add Text Generation, Add Speech-to-Text, Add Text-to-Speech, or Add Image Recognition

## Compatibility

- A recent Home Assistant version with config subentry support is required. The local development environment is tested with Home Assistant `2026.4.1`.
- A Groq API key is required from [Groq Console](https://console.groq.com/).
- Text-to-Speech audio normalization requires `ffmpeg` on the Home Assistant host.
- The integration domain is `groq`.
- This integration is cloud-based and requires network access to Groq APIs.

## Authentication

Enter a Groq API key during initial setup. The key is stored by Home Assistant and is redacted from diagnostics.

You can add more Groq accounts from the integration page when you want separate account-level names, default API keys, or billing projects.

Each service can also have its own optional API key. If a service API key is set, that service uses its own key instead of the integration-level key. This is useful when you want separate Groq projects or keys for Text Generation, Text-to-Speech, Image Recognition, or future service types.

If Groq returns an authentication error, Home Assistant starts a reauthentication flow so you can enter a new key.

## Usage examples

### Text Generation

Use the generated Conversation entity with Home Assistant Assist, the generated AI Task entity for data generation tasks, or call the response service directly:

```yaml
action: groq.generate_text
data:
  prompt: Summarize the current home status for a dashboard notification.
  model: llama-3.3-70b-versatile
  temperature: 0.2
response_variable: groq_text
```

Advanced Groq chat completion request fields can be passed through with `request_body_options`:

```yaml
action: groq.generate_text
data:
  prompt: Create a short goodnight message.
  model: openai/gpt-oss-20b
  request_body_options:
    user: home-assistant
    metadata:
      workflow: bedtime
response_variable: groq_text
```

### Structured Outputs

```yaml
action: groq.generate_structured
data:
  prompt: Extract the room, device, and requested action from "turn off the kitchen lights".
  model: openai/gpt-oss-20b
  strict: true
  schema_name: home_intent
  schema:
    type: object
    properties:
      room:
        type: string
      device:
        type: string
      action:
        type: string
    required:
      - room
      - device
      - action
    additionalProperties: false
response_variable: groq_structured
```

### Text-to-Speech

Replace the entity ID with the Groq TTS entity created by your named Text-to-Speech service.

```yaml
action: tts.speak
target:
  entity_id: tts.groq_text_to_speech
data:
  cache: true
  media_player_entity_id: media_player.living_room_speaker
  message: The front door has been open for five minutes.
  options:
    voice: hannah
    response_format: wav
    vocal_directions: calm
```

### Image Recognition

```yaml
action: groq.analyze_image
data:
  prompt: Describe anything unusual in this camera image.
  image_url: https://example.com/snapshot.jpg
  model: meta-llama/llama-4-scout-17b-16e-instruct
response_variable: groq_image
```

## Supported Groq models

Built-in model lists are used when Groq model discovery is unavailable.

Text Generation:
- `llama-3.3-70b-versatile`
- `llama-3.1-8b-instant`
- `openai/gpt-oss-20b`
- `openai/gpt-oss-120b`
- `openai/gpt-oss-safeguard-20b`
- `qwen/qwen3-32b`
- `groq/compound`
- `groq/compound-mini`

Reasoning:
- `openai/gpt-oss-20b`
- `openai/gpt-oss-120b`
- `openai/gpt-oss-safeguard-20b`
- `qwen/qwen3-32b`

Prompt caching:
- `openai/gpt-oss-20b`
- `openai/gpt-oss-120b`
- `openai/gpt-oss-safeguard-20b`

Text-to-Speech:
- `canopylabs/orpheus-v1-english`
- `canopylabs/orpheus-arabic-saudi`

Image Recognition:
- `meta-llama/llama-4-scout-17b-16e-instruct`
- `meta-llama/llama-4-maverick-17b-128e-instruct`

Speech-to-Text profiles:
- `whisper-large-v3`
- `whisper-large-v3-turbo`

## Diagnostics

Download diagnostics from the Groq integration page when reporting issues. Diagnostics redact API keys and include setup options, enabled service types, selected models, and runtime configuration needed for support.

Do not include Groq API keys, private prompts, camera images, generated audio, or other sensitive household data in public issues.

## Development and testing

All tests run inside a Home Assistant Docker-based environment so imports, entity base classes, exceptions, and bundled runtime dependencies match Home Assistant instead of local Python stubs. The harness asserts Python `3.14` at build time.

```bash
scripts/test
```

To pass a custom pytest command:

```bash
scripts/test python -m pytest tests/components/groq -q
```

The Docker image is built from `ghcr.io/home-assistant/home-assistant:stable` by default and expects Python `3.14`. To test against another Home Assistant image:

```bash
HA_IMAGE=ghcr.io/home-assistant/home-assistant:2026.5.0b3 scripts/test
```

To override the expected Python version for an experimental image:

```bash
PYTHON_VERSION=3.14 HA_IMAGE=ghcr.io/home-assistant/home-assistant:stable scripts/test
```

Additional harness checks are available for CI parity:

```bash
scripts/test python scripts/validate_quality_scale.py
scripts/test python scripts/importtime_profile.py --strict-integration-warnings
scripts/test python -m pytest --cov=custom_components.groq --cov-report=term-missing --cov-fail-under=80 -q
```

The multi-feature architecture for text generation, speech-to-text, text-to-speech, OCR/image recognition, reasoning, structured outputs, and prompt caching is documented in [docs/architecture.md](docs/architecture.md).

## Documentation

- [Groq Console](https://console.groq.com/)
- [Groq status page](https://groqstatus.com/)
- [Groq API reference](https://console.groq.com/docs/api-reference)
- [Groq structured outputs](https://console.groq.com/docs/structured-outputs)
- [Groq text-to-speech](https://console.groq.com/docs/text-to-speech)
- [Home Assistant custom integration docs](https://developers.home-assistant.io/docs/creating_integration_file_structure)
- [Architecture notes](docs/architecture.md)
