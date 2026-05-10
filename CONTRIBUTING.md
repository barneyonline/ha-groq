# Contributing to Groq

Thanks for helping improve this Home Assistant custom integration. We follow Home Assistant and HACS standards to keep the project healthy and dependable. Please read through this guide before opening a pull request.

## Code of Conduct

By participating you agree to uphold the [Home Assistant Code of Conduct](https://www.home-assistant.io/code_of_conduct/). Be respectful and constructive when interacting with the community.

## How to Help

- Report reproducible bugs and attach diagnostics or logs where possible.
- Report model, voice, Assist, STT, TTS, AI Task, image, or audio workflow gaps with the affected Groq model and Home Assistant version.
- Suggest enhancements or improvements to documentation.
- Contribute code, tests, translations, or quality scale compliance work.
- Review open pull requests and share constructive feedback.

Before starting large features, open an issue or discussion so we can agree on scope and fit.

## Reporting Bugs And Support Gaps

Use the GitHub issue forms so reports are routed into the right triage path:

- `Bug report` for regressions, setup failures, broken actions, incorrect responses, repair issues, diagnostics problems, or import/test failures.
- `Feature request` for new capabilities, new service options, additional model support, or documentation improvements.

Before opening a bug issue, capture diagnostics in Home Assistant:

1. Go to `Settings -> Devices & services -> Integrations`.
2. Open `Groq` and choose `Download diagnostics`.
3. For service-specific problems, include the affected service name, platform, model, voice, and entity ID where applicable.
4. Review the redacted file, then attach it to the GitHub issue body or a follow-up comment.

Include the integration version, Home Assistant version, installation method, selected Groq model or voice, affected action or entity ID, and whether the issue started after an update. Do not include Groq API keys or unredacted request payloads containing private data.

## Development Workflow

1. Fork and clone the repository.
2. Create a feature branch (`feature/...`, `bugfix/...`, or `docs/...`) from the latest `main`.
3. Build the pinned Docker environment:

```bash
docker compose -f devtools/docker/docker-compose.yml build ha-dev
```

4. Develop and test your changes inside `ha-dev` or through the `scripts/test` wrapper.
5. Start `ha-runtime` when you need a real Home Assistant UI for manual verification:

```bash
mkdir -p .ha-config
docker compose -f devtools/docker/docker-compose.yml up -d ha-runtime
```

This runs the official Home Assistant container, mounts `.ha-config/` to `/config`, and mounts this checkout's `custom_components/groq` into Home Assistant.

6. Commit with clear messages and push your branch.
7. Open a pull request using the template. Fill in every section and link any related issues.

Use the pinned Docker environment for linting, formatting, coverage, and tests:

```bash
scripts/test ruff check custom_components/groq tests/components/groq tests/scripts scripts
scripts/test black custom_components/groq tests/components/groq tests/scripts
scripts/test python scripts/validate_quality_scale.py
scripts/test python scripts/importtime_profile.py --strict-integration-warnings
scripts/strict-typing
scripts/test python -m pytest tests/components/groq -q
scripts/test python -m pytest -q
```

## Coding Standards And Tooling

Home Assistant integrations must follow the [core development guidelines](https://developers.home-assistant.io/docs/development_guidelines/). Key points:

- Use modern Python syntax, type hints, and async Home Assistant APIs for I/O.
- Keep logging format strings lazy and never log API keys, tokens, or private prompt content.
- Keep network calls resilient and non-blocking. Use Home Assistant aiohttp sessions for Groq HTTP calls.
- Keep action errors and repairs actionable for users, with translated exception keys where Home Assistant supports them.
- Keep YAML and documentation formatting consistent with the [Home Assistant style guide](https://developers.home-assistant.io/docs/documenting/yaml-style-guide/).

This repository relies on the following checks. Please run the relevant subset locally before pushing:

```bash
scripts/test ruff check custom_components/groq tests/components/groq tests/scripts scripts
scripts/test black custom_components/groq tests/components/groq tests/scripts
scripts/test python -m pytest tests/components/groq -q
scripts/test python scripts/validate_quality_scale.py
scripts/test python scripts/importtime_profile.py --strict-integration-warnings
scripts/strict-typing
```

Hassfest validation runs automatically in CI via [`home-assistant/actions/hassfest`](https://github.com/home-assistant/actions/tree/master/hassfest). If you need to run it locally, clone the Home Assistant Core repository and execute `python -m script.hassfest` from your integration checkout.

The `ha-runtime` service is for manual verification only. Keep automated checks on `ha-dev` so test and lint runs stay fast and deterministic. `ha-runtime` inherits the `TZ` environment variable from your shell and defaults to `UTC` when `TZ` is unset.

## Translations

- Place new or updated translations under `custom_components/groq/translations/<language>.json`.
- Keep `custom_components/groq/strings.json` and English translations aligned when adding config flow strings, selector labels, repair issues, service exceptions, or entity names.
- Follow Home Assistant translation conventions. Language files should mirror English keys and use native phrasing.
- Keep JSON valid UTF-8. ASCII is preferred unless the language requires accented characters.

## Documentation

- Update `README.md` when behavior, options, supported models, actions, diagnostics, repairs, or installation requirements change.
- For codebase orientation, see [docs/architecture.md](docs/architecture.md).
- Update `quality_scale.yaml` when a change affects Home Assistant quality scale evidence.
- Update service descriptions in `custom_components/groq/services.yaml` when action inputs or response shapes change.

## Tests

- Add or update tests in `tests/components/groq/` for new functionality or bug fixes.
- Add or update tests in `tests/scripts/` for repository maintenance scripts.
- Prefer mocks for Groq network calls, camera captures, media-source resolution, subprocesses, and Home Assistant service boundaries.
- Keep pytest fast and deterministic. Do not require a real Groq API key for automated tests.
- For touched Python modules, run focused tests first, then the full suite when practical.

## Pull Request Expectations

- Keep pull requests focused. Separate refactors from functional changes.
- Rebase on top of the latest `main` before requesting review to avoid merge conflicts.
- Fill out the pull request template, including exact test commands.
- Ensure GitHub Actions workflows for tests, hassfest, validation, and quality scale pass.
- For UI, translation, or Home Assistant flow changes, include screenshots or highlight impacted strings.
- For user-facing behavior changes, update `README.md` in the same pull request.

## Release Process

Maintainers cut releases by updating the manifest version, release notes, and tagging the release. Contributors generally do not publish releases directly, but you can help by keeping user-facing changes clear in pull requests and documentation.

## Getting Help

Open a discussion or issue if you are blocked. When in doubt, reference:

- [Home Assistant developer documentation](https://developers.home-assistant.io/) for integration patterns and quality scale rules.
- [HACS documentation](https://hacs.xyz/docs/) for repository requirements, manifests, and validation expectations.
- [Groq documentation](https://console.groq.com/docs/) for current model, voice, and API behavior.

Thank you for contributing.
