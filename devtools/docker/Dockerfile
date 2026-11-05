# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.14.7-slim-bookworm@sha256:82bc3c539b8813ada9d68c63b40158fa002f7f33de9bf3312a3dfdc0620dff56
FROM ${PYTHON_IMAGE}
ARG PYTHON_VERSION=3.14
ARG HA_VERSION=2026.10.0
ARG REQUIREMENTS_FILE=requirements-dev.txt
ENV PYTHON_VERSION=${PYTHON_VERSION}

# Root installs are intentional in this disposable test image. Dependency
# resolver warnings and the explicit pip check remain enabled.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_NO_COMPILE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    PYTEST_PLUGINS=pytest_asyncio.plugin,pytest_homeassistant_custom_component.plugins,pytest_cov.plugin \
    PYTEST_ASYNCIO_MODE=strict \
    PYTEST_ASYNCIO_DEFAULT_FIXTURE_LOOP_SCOPE=function

COPY devtools/docker/requirements-*.txt scripts/verify_test_environment.py /opt/groq-test/

# Keep the large dependency install in one layer for cloud daemons using vfs.
# The optional session CA is mounted only for this step and never baked in.
RUN --mount=type=secret,id=proxy_ca <<'BUILD'
set -eu
if [ -s /run/secrets/proxy_ca ]; then
    export PIP_CERT=/run/secrets/proxy_ca
fi
apt-get update
apt-get install -y --no-install-recommends ca-certificates git ffmpeg libturbojpeg0
rm -rf /var/lib/apt/lists/*
python -m pip install --no-cache-dir --no-compile \
    "homeassistant==${HA_VERSION}" -r "/opt/groq-test/${REQUIREMENTS_FILE}"
python - <<'PY'
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

expected = tuple(int(part) for part in os.environ['PYTHON_VERSION'].split('.'))
if sys.version_info[:len(expected)] != expected:
    raise SystemExit(f'Expected Python {expected}, got {platform.python_version()}')
source = Path(importlib.util.find_spec('homeassistant').origin).parent
# Core component manifests own these pins; do not maintain another pin list.
requirements = sorted({
    requirement
    for component in ('conversation', 'tts', 'ffmpeg', 'camera')
    for requirement in json.loads(
        (source / 'components' / component / 'manifest.json').read_text()
    ).get('requirements', [])
})
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--no-compile',
    *requirements,
], check=True)
# Expose only HA source to mypy, without adding all site-packages to MYPYPATH.
destination = Path('/opt/homeassistant-source')
destination.mkdir()
shutil.move(str(source), destination / 'homeassistant')
PY
printf '%s' "$HA_VERSION" > /opt/groq-test/image-core-version
PYTHONPATH=/opt/homeassistant-source python /opt/groq-test/verify_test_environment.py \
    --expected-core-file /opt/groq-test/image-core-version --check-dependencies
mkdir -p /workspace
git config --global --add safe.directory /workspace
git config --global --add safe.directory '/root/.cache/pre-commit/*'
BUILD

ENV PYTHONPATH=/opt/homeassistant-source
WORKDIR /workspace
ARG TEST_ENVIRONMENT_HASH=manual
LABEL io.ha-groq.test-environment-hash="${TEST_ENVIRONMENT_HASH}"
CMD ["bash"]
