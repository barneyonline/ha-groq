# Groq - Home Assistant Custom Integration

[![Release](https://img.shields.io/github/v/release/barneyonline/ha-groq?display_name=tag&sort=semver)](https://github.com/barneyonline/ha-groq/releases)
[![Stars](https://img.shields.io/github/stars/barneyonline/ha-groq)](https://github.com/barneyonline/ha-groq/stargazers)
[![License](https://img.shields.io/github/license/barneyonline/ha-groq)](LICENSE)

[![Tests](https://img.shields.io/github/actions/workflow/status/barneyonline/ha-groq/ci.yml?branch=main&label=tests)](https://github.com/barneyonline/ha-groq/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/barneyonline/ha-groq/branch/main/graph/badge.svg)](https://codecov.io/gh/barneyonline/ha-groq)
[![Hassfest](https://img.shields.io/github/actions/workflow/status/barneyonline/ha-groq/hassfest.yml?branch=main&label=hassfest)](https://github.com/barneyonline/ha-groq/actions/workflows/hassfest.yml)
[![Quality Scale](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbarneyonline%2Fha-groq%2Fmain%2Fcustom_components%2Fgroq%2Fmanifest.json&query=%24.quality_scale&label=quality%20scale&cacheSeconds=3600)](https://developers.home-assistant.io/docs/integration_quality_scale_index)

[![HACS](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://hacs.xyz)
[![Open Issues](https://img.shields.io/github/issues/barneyonline/ha-groq)](https://github.com/barneyonline/ha-groq/issues)
![Development Status](https://img.shields.io/badge/development-active-success?style=flat-square)

Groq is a cloud API service for fast language, speech, and vision models. This Home Assistant custom integration connects a Groq account to Assist, AI Tasks, speech-to-text, text-to-speech, image analysis, and response actions.

> [!IMPORTANT]
> This is an unofficial community project. It is not affiliated with, endorsed by, or supported by Groq.
>
> Feature availability, model availability, rate limits, token limits, and billing behavior are controlled by Groq and can vary by account, project, and model.

## Supported Functionality

This integration supports Groq cloud accounts. It does not discover or control physical devices.

Supported Home Assistant platforms:

- `conversation`: Assist conversation agents backed by configured Groq text generation services.
- `ai_task`: data generation tasks for services and automations that need structured output.
- `stt`: speech-to-text entities for Home Assistant voice pipelines.
- `tts`: text-to-speech entities for `tts.speak`.

Provided response actions:

- `groq.generate_text`: generate a text response.
- `groq.generate_structured`: generate JSON or schema-shaped output.
- `groq.analyze_image`: ask a question about a camera image, media image, local image, or image URL.
- `groq.extract_text_from_image`: OCR-style text extraction from an image.
- `groq.transcribe_audio`: transcribe a local or media-source audio file.
- `groq.clear_cache`: clear the local response cache for a Groq account.
- `groq.list_models`: list models visible to a Groq account.

Each configured Groq service creates its own Home Assistant device and the relevant entity for that platform. Text generation services can create Assist and AI Task entities. Speech-to-text and text-to-speech services create STT and TTS entities.

## Installation

### HACS

1. Open HACS.
2. Go to Integrations, then Custom repositories.
3. Add `https://github.com/barneyonline/ha-groq` as an Integration repository.
4. Install Groq.
5. Restart Home Assistant.
6. Go to Settings -> Devices & services -> Add integration -> Groq.

### Manual

1. Copy `custom_components/groq` into your Home Assistant `custom_components` directory.
2. Restart Home Assistant.
3. Go to Settings -> Devices & services -> Add integration -> Groq.

## Requirements

- A recent Home Assistant version with config subentry support. Local development is tested with Home Assistant `2026.4.1`.
- A Groq API key from [Groq Console](https://console.groq.com/).
- Network access from Home Assistant to `https://api.groq.com`.
- Optional: `ffmpeg` on the Home Assistant host if you enable TTS audio normalization.

This integration does not use Home Assistant application credentials or OAuth. Groq API keys act as account or project credentials. Use separate Groq keys when you want separate projects, billing pools, environments, or rate-limit isolation.

## Configuration

Initial account setup asks for:

- Name: friendly name for this Groq account in Home Assistant.
- Groq API key: secret key used for Groq API requests. The key is stored by Home Assistant and redacted from diagnostics.

After adding an account, open the Groq integration page and add one or more services:

- Text Generation: name, model, system prompt, temperature, free-tier protection, and optional advanced request options.
- Speech-to-Text: name, model, language hint, and free-tier protection.
- Text-to-Speech: name, model, voice, optional vocal directions, optional audio normalization, and free-tier protection.
- Image Recognition: name, model, system prompt, and free-tier protection.

Advanced Text Generation options include max completion tokens, top P, stop sequences, seed, service tier, streaming, reasoning options, local response caching, structured output schema, strict schema mode, and additional Groq request body options.

You can add more than one Groq account. The integration prevents adding the same API key twice.

## Known Limitations

- This is a cloud integration and will not work without internet access to Groq.
- Groq can change model availability, limits, and request option support outside this integration.
- Some advanced options work only on models that support them. The setup flow validates known model capabilities where possible.
- TTS input is limited by Groq Orpheus model limits; this integration locally blocks overly long TTS requests.
- Audio normalization needs `ffmpeg` and uses extra CPU.
- This integration does not discover devices. It supports Groq cloud accounts and user-created Groq service entries.

## Troubleshooting

- Invalid API key: create or copy a fresh key from Groq Console, then reauthenticate the Groq integration entry.
- Cannot connect: check Home Assistant network/DNS access to `api.groq.com` and the [Groq status page](https://groqstatus.com/).
- Model missing: use `groq.list_models` to see models visible to the selected account, or choose a known compatible model.
- Multiple accounts or services: provide `config_entry_id` or `service_id` in the action data so Home Assistant can select the intended Groq account or service.
- Rate-limit errors: wait for Groq's reset window, lower automation frequency, choose a smaller model, or use separate Groq projects/keys where appropriate.
- TTS normalization fails: install `ffmpeg` on the Home Assistant host or disable audio normalization.
- Local image or audio file fails: make sure the path is allowed by Home Assistant `allowlist_external_dirs`, or use a media-source file.
- Diagnostics: download diagnostics from the integration page. API keys and prompt fields are redacted.

## Removal

1. Go to Settings -> Devices & services -> Groq.
2. Delete any Groq service entries you no longer want.
3. Delete the Groq account entry.
4. Restart Home Assistant if you plan to remove the custom integration files.
5. If installed through HACS, remove Groq from HACS. For a manual install, delete `custom_components/groq`.

Removing the integration stops future Groq API calls from Home Assistant. It does not delete Groq projects, keys, billing data, or logs in Groq Console.

## Useful Links

- [Project Wiki](https://github.com/barneyonline/ha-groq/wiki)
- [Groq Console](https://console.groq.com/)
- [Groq status page](https://groqstatus.com/)
- [Groq API reference](https://console.groq.com/docs/api-reference)
- [Architecture notes](docs/architecture.md)
