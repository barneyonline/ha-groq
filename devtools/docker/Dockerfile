ARG HA_IMAGE=ghcr.io/home-assistant/home-assistant:2026.10.0@sha256:1b64d38f38d922bf9d59336451fd6453e1d614f934456af4ee3d2a51061be3a4
FROM ${HA_IMAGE}
ARG PYTHON_VERSION=3.14
ARG REQUIREMENTS_FILE=requirements-dev.txt
ENV PYTHON_VERSION=${PYTHON_VERSION}

# Root installs are intentional in this disposable test image. Dependency
# resolver warnings and the explicit pip check remain enabled.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_NO_COMPILE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=0 \
    PYTEST_PLUGINS=pytest_asyncio.plugin,pytest_homeassistant_custom_component.plugins,pytest_cov.plugin \
    PYTEST_ASYNCIO_MODE=strict \
    PYTEST_ASYNCIO_DEFAULT_FIXTURE_LOOP_SCOPE=function

WORKDIR /workspace

COPY devtools/docker/requirements-*.txt /opt/groq-test/

RUN python - <<'PY'
import os
from importlib.metadata import version
from pathlib import Path
import platform
import sys

Path("/opt/groq-test/image-core-version").write_text(version("homeassistant"))
expected = tuple(int(part) for part in os.environ.get("PYTHON_VERSION", "3.14").split("."))
actual = sys.version_info[: len(expected)]
if actual != expected:
    raise SystemExit(
        f"Expected Python {'.'.join(map(str, expected))}, got {platform.python_version()}"
    )
PY

# The all-integrations image bundles surepy for Sure Petcare. Its rich<11 pin
# conflicts with the HA 2026.10 pytest helper's pytest-codspeed (rich>=13.8.1).
# Groq does not use surepy; remove it only from this disposable test image.
RUN python -m pip uninstall --yes surepy \
    && python -m pip install --no-cache-dir --no-compile -r "/opt/groq-test/${REQUIREMENTS_FILE}"
COPY scripts/verify_test_environment.py /opt/groq-test/verify_test_environment.py
RUN python /opt/groq-test/verify_test_environment.py \
    --expected-core-file /opt/groq-test/image-core-version --check-dependencies

RUN git config --global --add safe.directory /workspace \
    && git config --global --add safe.directory '/root/.cache/pre-commit/*'

ARG TEST_ENVIRONMENT_HASH=manual

LABEL io.ha-groq.test-environment-hash="${TEST_ENVIRONMENT_HASH}"

CMD ["bash"]
