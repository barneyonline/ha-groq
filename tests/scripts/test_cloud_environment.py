"""Exercise cloud hooks against a fake Docker daemon without network access."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).parents[2]


def test_default_ca_is_an_empty_regular_file_inside_the_build_context():
    """Bake must not require external filesystem access on non-cloud runners."""
    directory = ROOT / "devtools/docker"
    compose = yaml.safe_load((directory / "docker-compose.yml").read_text())
    source = compose["secrets"]["proxy_ca"]["file"]
    default = re.fullmatch(r"\$\{HA_GROQ_CA_BUNDLE:-(.+)\}", source)[1]
    path = (directory / default).resolve()

    assert path.is_relative_to(ROOT)
    assert path.is_file()
    assert path.read_bytes() == b""
    volumes = compose["services"]["ha-dev"]["volumes"]
    assert f"{source}:/run/ha-groq-ca.pem:ro" in volumes


@pytest.fixture
def cloud_docker(tmp_path):
    """Record Docker operations and persist image metadata between invocations."""
    docker = tmp_path / "docker"
    docker.write_text(f"#!{sys.executable}\n" + r"""
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
with Path(os.environ['DOCKER_CALLS']).open('a') as log:
    log.write(json.dumps(args) + '\n')
state = Path(os.environ['DOCKER_STATE'])
if args[0].startswith('--host='):
    sys.exit(int(os.environ.get('DOCKER_UNAVAILABLE', '0')))
if args[:2] == ['image', 'inspect']:
    if not state.exists():
        sys.exit(1)
    image = json.loads(state.read_text())
    print(image['id'] if args[3] == '{{.Id}}' else image['hash'])
elif args[:2] == ['image', 'rm']:
    pass
elif args[0] == 'compose':
    operation = args[3]
    if operation == 'config':
        print(json.dumps({'ha': os.environ.get('HA_VERSION', '2026.10.0')}))
    elif operation == 'build':
        fingerprint = os.environ['HA_GROQ_TEST_ENVIRONMENT_HASH']
        state.write_text(json.dumps({'id': fingerprint, 'hash': fingerprint}))
    elif operation not in ('up', 'down', 'exec'):
        raise AssertionError(args)
else:
    raise AssertionError(args)
""")
    docker.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    env = os.environ.copy()
    env.update(
        PATH=f"{tmp_path}{os.pathsep}{env['PATH']}",
        DOCKER_CALLS=str(calls),
        DOCKER_STATE=str(tmp_path / "image.json"),
    )

    def run(action, **overrides):
        subprocess.run(
            [str(ROOT / "scripts/cloud-environment"), action],
            env={**env, **overrides},
            check=True,
            capture_output=True,
            text=True,
        )
        result = [json.loads(line) for line in calls.read_text().splitlines()]
        calls.unlink()
        return result

    return run


def test_install_caches_build_and_does_not_run_tests_or_manual_ha(cloud_docker):
    first = cloud_docker("install")
    second = cloud_docker("install")

    assert any("build" in call for call in first)
    assert not any("build" in call for call in second)
    for calls in (first, second):
        assert not any("pytest" in " ".join(call) for call in calls)
        assert not any("ha-runtime" in call for call in calls)
        assert any(call[-4:] == ["python", "-m", "pip", "check"] for call in calls)
        assert any(
            "/opt/groq-test/verify_test_environment.py" in call for call in calls
        )


def test_start_recreates_container_without_rebuilding_image(cloud_docker):
    cloud_docker("install")
    calls = cloud_docker("start")

    assert not any("build" in call for call in calls)
    assert any("--force-recreate" in call for call in calls)
    assert any(call[-4:] == ["python", "-m", "pip", "check"] for call in calls)


def test_changed_core_build_argument_invalidates_cached_image(cloud_docker):
    cloud_docker("install")
    calls = cloud_docker("install", HA_VERSION="2026.11.0")

    assert any("build" in call for call in calls)
    assert any(call[:2] == ["image", "rm"] for call in calls)


def test_reset_removes_container_and_explicitly_rebuilds_without_cache(cloud_docker):
    cloud_docker("install")
    calls = cloud_docker("reset")

    down = next(i for i, call in enumerate(calls) if "down" in call)
    build = next(i for i, call in enumerate(calls) if "build" in call)
    assert down < build
    assert "--no-cache" in calls[build]
    assert "--pull" in calls[build]
    assert any("--force-recreate" in call for call in calls)


def test_failed_daemon_check_stops_before_mutating_docker(cloud_docker):
    with pytest.raises(subprocess.CalledProcessError):
        cloud_docker("start", DOCKER_UNAVAILABLE="1")
