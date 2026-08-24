ARG HA_IMAGE=ghcr.io/home-assistant/home-assistant:2026.7.2@sha256:1476924357b46e80735c13e94232ba5c853cac052e9df4bb28d50fa56348097b
ARG PYTHON_VERSION=3.14
FROM ${HA_IMAGE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COMPILE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=0 \
    PYTEST_PLUGINS=pytest_asyncio.plugin,pytest_homeassistant_custom_component.plugins,pytest_cov.plugin \
    PYTEST_ASYNCIO_MODE=strict \
    PYTEST_ASYNCIO_DEFAULT_FIXTURE_LOOP_SCOPE=function

WORKDIR /workspace

COPY devtools/docker/requirements-dev.txt requirements-dev.txt

RUN python - <<'PY'
import os
import platform
import sys

expected = tuple(int(part) for part in os.environ.get("PYTHON_VERSION", "3.14").split("."))
actual = sys.version_info[: len(expected)]
if actual != expected:
    raise SystemExit(
        f"Expected Python {'.'.join(map(str, expected))}, got {platform.python_version()}"
    )
PY

RUN python -m pip install --no-cache-dir --no-compile -r requirements-dev.txt

RUN git config --global --add safe.directory /workspace \
    && git config --global --add safe.directory '/root/.cache/pre-commit/*'

ARG TEST_ENVIRONMENT_HASH=manual

LABEL io.ha-groq.test-environment-hash="${TEST_ENVIRONMENT_HASH}"

CMD ["bash"]
