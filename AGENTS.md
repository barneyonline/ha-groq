# Repository Guidelines

## Project Structure & Module Organization
- `custom_components/groq/`: Home Assistant integration source.
  - `config_flow.py`: UI setup/options logic.
  - `tts.py`: TTS entity and audio post‑processing (ffmpeg).
  - `tts_engine.py`: Async client for Groq API.
  - `const.py`: Constants and built‑ins (models, voices).
- `tests/`: Pytest unit tests run against the Home Assistant Docker test image.
  - `tests/components/groq/`: integration tests and Home Assistant component checks.
  - `tests/scripts/`: tests for repository maintenance scripts.
- `README.md`, `hacs.json`, `manifest.json`: Integration metadata and docs.

## Build, Test, and Development Commands
- Run tests: `scripts/test`
  - Builds/runs the Home Assistant Docker test image and executes `python -m pytest -q`.
  - Pass extra commands after the script when needed, for example `scripts/test python -m pytest tests/components/groq -q`.
- CI parity checks:
  - `scripts/test python scripts/validate_quality_scale.py`
  - `scripts/test python scripts/importtime_profile.py --strict-integration-warnings`
  - `scripts/test python -m pytest --cov=custom_components.groq --cov-report=term-missing --cov-fail-under=80 -q`
- Lint/format: use the Docker harness or local `ruff`/`black`; keep diffs minimal.
- No build step required. For end‑to‑end checks, install in a HA instance per README.

## Coding Style & Naming Conventions
- Python 3.14, 4‑space indentation, PEP 8 style.
- Use type hints; prefer `async`/await for I/O and HA APIs.
- Filenames and module symbols: `snake_case`; constants in `const.py` are `UPPER_SNAKE_CASE`.
- Log with `_LOGGER` and avoid logging secrets (API keys, tokens).

## Testing Guidelines
- Framework: `pytest` with `@pytest.mark.asyncio` for async tests.
- Add integration tests under `tests/components/groq/` and script tests under `tests/scripts/`; files are named `test_*.py` and functions start with `test_`.
- Cover: config validation (`config_flow.validate_user_input`), TTS audio processing options, and network/error paths in `GroqTTSEngine`.
- Run: `scripts/test`; keep tests isolated with mocks for network/process boundaries.

## Commit & Pull Request Guidelines
- Commits: concise, imperative mood (e.g., "Add dynamic voice selector").
- Before pushing to `origin`, run the full local verification set and fix any failures:
  - `scripts/test pre-commit run --all-files`
  - `scripts/strict-typing`
  - `scripts/test python scripts/validate_quality_scale.py`
  - `scripts/test python scripts/importtime_profile.py --strict-integration-warnings`
  - `scripts/test python -m pytest --cov=custom_components.groq --cov-report=term-missing --cov-report=xml --cov-fail-under=100 --junitxml=junit.xml -o junit_family=legacy`
  - `scripts/test bash -lc "COVERAGE_FILE=/tmp/groq.coverage python -m coverage erase && COVERAGE_FILE=/tmp/groq.coverage python -m coverage run -m pytest tests/components/groq -q && COVERAGE_FILE=/tmp/groq.coverage python -m coverage report -m --include=<changed-custom-components-groq-python-files-comma-separated> --fail-under=100"`
  - `git diff --check`
- PRs must include:
  - Clear description and rationale; link related issues.
  - Tests for new logic or bug fixes.
  - Updates to `README.md` if options/models/usage change.
  - Screenshots of HA config screens if UI flows change.

## Security & Configuration Tips
- Do not commit API keys or real endpoints beyond defaults.
- Network calls must be async and resilient; never block the event loop.
