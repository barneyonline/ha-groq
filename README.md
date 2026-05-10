# Groq - Home Assistant Custom Integration

<!-- Badges -->
[![Release](https://img.shields.io/github/v/release/barneyonline/ha-groq?display_name=tag&sort=semver)](https://github.com/barneyonline/ha-groq/releases)
[![Stars](https://img.shields.io/github/stars/barneyonline/ha-groq)](https://github.com/barneyonline/ha-groq/stargazers)
[![License](https://img.shields.io/github/license/barneyonline/ha-groq)](LICENSE)

[![Tests](https://img.shields.io/github/actions/workflow/status/barneyonline/ha-groq/ci.yml?branch=main&label=tests)](https://github.com/barneyonline/ha-groq/actions/workflows/ci.yml)
[![Hassfest](https://img.shields.io/github/actions/workflow/status/barneyonline/ha-groq/hassfest.yml?branch=main&label=hassfest)](https://github.com/barneyonline/ha-groq/actions/workflows/hassfest.yml)
[![Quality Scale](https://img.shields.io/github/actions/workflow/status/barneyonline/ha-groq/quality-scale.yml?branch=main&label=quality%20scale)](https://developers.home-assistant.io/docs/integration_quality_scale_index)

[![HACS](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://hacs.xyz)
[![Open Issues](https://img.shields.io/github/issues/barneyonline/ha-groq)](https://github.com/barneyonline/ha-groq/issues)
![Development Status](https://img.shields.io/badge/development-active-success?style=flat-square)

Cloud-based Home Assistant integration for Groq AI services.

> [!IMPORTANT]
> This is an unofficial community project. It is not affiliated with, endorsed by, or supported by Groq.
>
> The integration uses Groq cloud APIs. Feature availability, model availability, rate limits, and request options can vary by Groq account, project, and model.

## Supported service categories

- Text Generation services for Home Assistant Assist, AI Tasks, response services, structured outputs, reasoning-capable models, prompt caching, and Groq Compound models
- Speech-to-Text services for Home Assistant voice pipelines and audio transcription actions
- Text-to-Speech services using Groq Orpheus models through Home Assistant `tts.speak`
- Image Recognition services for camera snapshots, media images, local images, image URLs, and OCR-style prompts

## Key features

- Guided onboarding for Groq account setup with a friendly name and redacted API key storage
- Service-specific subentry buttons for creating multiple named services under each Groq account
- Groq model discovery during service setup, filtered to the selected service type
- Text Generation entities for Home Assistant Assist conversations and AI Task data generation
- Response services for text generation, structured output, image analysis, OCR-style extraction, audio transcription, cache clearing, and model listing
- Text Generation options for system prompts, sampling controls, service tiers, streaming, structured outputs, reasoning, prompt caching, and advanced request-body options
- Text-to-Speech entities with Orpheus voices, vocal direction presets, custom vocal directions, optional audio normalization, and local cache sizing
- Per-service free-tier protection for Text Generation, Speech-to-Text, Text-to-Speech, and Image Recognition services
- Diagnostics with Groq API keys redacted

## Quick install (HACS)

1. HACS -> Integrations -> Custom repositories
2. Add `https://github.com/barneyonline/ha-groq` as an Integration repository
3. Install Groq and restart Home Assistant
4. Add the integration and enter a friendly name and Groq API key
5. Open the Groq integration page and add the service type you want to use

Manual install steps: see the wiki Installation page.

## Compatibility

- A recent Home Assistant version with config subentry support is required. The local development environment is tested with Home Assistant `2026.4.1`.
- A Groq API key is required from [Groq Console](https://console.groq.com/).
- Text-to-Speech audio normalization requires `ffmpeg` on the Home Assistant host.
- The integration domain is `groq`.
- This integration is cloud-based and requires network access to Groq APIs.

## Authentication

Enter a Groq API key during initial setup. The key is stored by Home Assistant and is redacted from diagnostics.

You can add more Groq accounts from the integration page when you want separate API keys, Groq projects, billing pools, or production/testing services.

## Documentation

Refer to the [Wiki](https://github.com/barneyonline/ha-groq/wiki) for installation, service setup, usage examples, troubleshooting, and development notes.

Useful links:

- [Groq Console](https://console.groq.com/)
- [Groq status page](https://groqstatus.com/)
- [Groq API reference](https://console.groq.com/docs/api-reference)
- [Architecture notes](docs/architecture.md)
