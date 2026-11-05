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
scripts/test --prepare
```

4. Develop and test your changes inside `ha-dev` or through the `scripts/test` wrapper. The wrapper builds the test image only when its Docker inputs change and reuses a per-worktree container between commands. Run `scripts/test --stop` when you want to remove that container.
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

The primary harness uses Home Assistant 2026.10.0. Verify the advertised minimum
separately with `scripts/test --minimum python -m pytest tests/components/groq -q`.
Both environments verify the actual Core/helper versions and run dependency checks
when built. The test image uses digest-pinned Python 3.14 and installs only Core,
the matching helper, and the component dependencies needed by Groq. `HA_VERSION`
overrides must match the helper dependency; incompatible versions fail before
tests. `PYTHON_IMAGE` overrides the interpreter image. Stop the minimum container
with `scripts/test --minimum --stop`. The old `HA_IMAGE` override is rejected;
use `HA_VERSION` for Core and `PYTHON_IMAGE` for the interpreter image instead.

### Managed cloud environment

Configure the saved cloud environment to run these commands from the checkout:

- Install script: `scripts/cloud-environment install`
- Session startup script: `scripts/cloud-environment start`

Installation builds `ha-dev` using normal Docker layer caching. Session startup
recreates the container to refresh Docker's injected proxy settings, verifies
the exact HA/helper versions and dependencies, and leaves it ready for
`scripts/test <command>`. Neither hook runs tests or starts `ha-runtime`.
Run relevant tests while developing and the full required checks before pushing.

The hooks use the managed local Docker socket and preserve Docker's proxy and
registry configuration. When `CODEX_PROXY_CERT` is present, `scripts/test` supplies
the combined system CA bundle as a BuildKit secret and a read-only runtime mount.
It configures Python, pip, Requests and Git trust without baking session
certificates into the image. Set `HA_GROQ_CA_BUNDLE` to use another combined CA
bundle. Without a custom bundle, normal system trust applies.

Use `scripts/cloud-environment reset` only for an explicit reset: it removes this
worktree's test container and rebuilds with `--pull --no-cache`. Ordinary startup
reuses the image and build cache. Keep only the Docker environment; a separate
native virtualenv and copied HA source are unnecessary. Start the digest-pinned
HA 2026.10.0 `ha-runtime` service only for manual UI checks.

`scripts/strict-typing` discovers the installed HA source for actual interface checks.
Run `scripts/test python scripts/check_ha_typing.py` for positive and negative typing
controls. Keep the 100% integration and changed-module coverage gates. The optional
`python -m pytest scripts/benchmark_runtime.py -q -s` command inside the harness
measures warmed local overhead with mocked provider I/O; it is not live latency.

## Warnings And CI Logs

Pytest treats Python warnings as errors. The only exception in `pytest.ini` is
the exact aiohttp `NotAppKeyWarning` payload, including its documentation link,
from
`homeassistant.components.http.server`: pinned Home Assistant 2026.10.0 retains
the `app["hass"]` string key for backward compatibility. Review this exception
when updating Core. The same warning from Groq or any other module still fails
tests, as do other warning messages and categories. The import-time check also
rejects integration deprecations. Avoid `-W always` on normal verification runs;
it overrides this policy and is useful only when auditing filtered warnings.

Python warnings and logging records are separate. CI retains WARNING-level
runtime logs for both Home Assistant profiles, including failed test runs.
Keep production error logs visible. Tests that exercise API failures, oversized
audio, or missing ffmpeg should assert the expected records with `caplog`.
Unit tests that isolate API or audio behavior must explicitly model the repair
reporting boundary; registry and lifecycle tests should use the real `hass`
fixture. A passing test must not rely on caught repair errors from incomplete
test doubles.

Some upstream tools emit informational warnings even when their checks succeed:

| Output | Treatment |
| --- | --- |
| Codecov GPG key has no trusted certification | Keep signature and checksum validation enabled. A good signature followed by successful integrity verification is expected on a fresh runner without a personal trust database. |
| Git checkout initial-branch hint | Keep checkout's output; the requested ref is selected afterward. |
| Home Assistant custom-integration notice | Keep the loader warning. Groq is a custom integration. |
| API retry, STT limit, or ffmpeg failure logs | Keep actionable production messages and assert them in the tests that deliberately trigger the failure. |
| Asyncio slow-callback warning | Retain the debug threshold. Rerun the affected test without concurrent build or benchmark work, then profile a reproducible delay before changing runtime code. |
| Managed dependency-graph CA bundle rehash warning | GitHub owns this job. A bundle contains multiple certificates, while rehash expects individual certificates. Investigate if the job develops TLS failures. |

The disposable Docker test image explicitly acknowledges pip's intentional root
installs with `PIP_ROOT_USER_ACTION=ignore`; dependency checks and resolver warnings
remain enabled. Codecov uploads use the reports generated inside Docker rather
than optional host-side coverage collectors. Workflow runner versions are pinned
to the verified Ubuntu 26.04 baseline; update them deliberately alongside the
verification environment.

Both PR test profiles check out the PR head SHA so their reports and reporting
attribution refer to the same source. Coverage.py stores relative file paths.
The isolated default-branch reporter fetches Git metadata and loads the verified
commit into Git's index with `git read-tree`, without checking out candidate
files. Codecov reads that native Git inventory using `root_dir`, `GIT_DIR` and
`GIT_WORK_TREE`; its working directory stays separate. Plugins, file fixes and
report discovery remain disabled, and uploads use OIDC. No checkout safety guard
is bypassed. A bounded public API check requires the exact attempt's merged,
nonempty coverage and processed JUnit. Verify the fresh main commit after merging
a reporter change; the pre-merge reporter still comes from current main.

See Codecov's [source inventory guidance](https://docs.codecov.com/docs/error-reference),
[action inputs](https://github.com/codecov/codecov-action#arguments) and
[Git index loading](https://git-scm.com/docs/git-read-tree) and
[coverage.py relative paths](https://coverage.readthedocs.io/en/latest/config.html#run-relative-files).

## Dependency Ownership

Keep the Home Assistant package and matching test plugin aligned. They supply
their own dependencies, including pytest, its asyncio/coverage plugins,
freezegun, and PyYAML. Do not duplicate those dependencies in the repository's
development requirements or export the complete Home Assistant environment to
GitHub's dependency graph.

Declare the integration's own runtime requirements and independently upgradable
development tools. Dependabot remains enabled for those declarations. Upstream
Home Assistant requirements are resolved by the image and test plugin; removing
duplicate declarations does not change the packages installed in Home Assistant.

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
- Mock Groq network calls, camera captures and subprocesses. For integration behavior, exercise real HA config entries, flow managers, registries, service dispatch and ChatLog instead of mocking those boundaries.
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
