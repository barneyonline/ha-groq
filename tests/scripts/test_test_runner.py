"""Regression tests for the Docker test runner lifecycle."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]


def test_environment_fingerprint_covers_docker_build_inputs() -> None:
    """Rebuild when files or Compose-resolved build arguments change."""
    runner = (REPOSITORY_ROOT / "scripts/test").read_text()

    for build_input in (
        ".dockerignore",
        "devtools/docker/Dockerfile",
        "devtools/docker/docker-compose.yml",
        "devtools/docker/requirements-dev.txt",
        "devtools/docker/requirements-min-ha.txt",
        "scripts/verify_test_environment.py",
    ):
        assert build_input in runner

    assert 'docker compose -f "$compose_file" config --format json ha-dev' in runner
    assert "HA_GROQ_TEST_ENVIRONMENT_HASH=fingerprint" in runner
    assert "HA_GROQ_TEST_IMAGE=ha-groq-dev:fingerprint" in runner
    assert '"${HA_IMAGE:-}" "${PYTHON_VERSION:-}"' not in runner


def test_runner_replaces_one_stable_image_per_worktree() -> None:
    """Avoid retaining a permanently tagged image for every environment hash."""
    runner = (REPOSITORY_ROOT / "scripts/test").read_text()

    assert 'HA_GROQ_TEST_IMAGE="ha-groq-dev:$project_hash"' in runner
    assert 'HA_GROQ_TEST_IMAGE="ha-groq-dev:$environment_hash"' not in runner
    assert 'docker image rm "$previous_image_id"' in runner
    assert "io.ha-groq.test-environment-hash" in runner
