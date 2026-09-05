"""Tests for immutable executable references in repository automation."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).parents[2]
IMMUTABLE_ACTION = re.compile(r"^[\w.-]+/[\w.-]+(?:/[\w.-]+)?@[0-9a-f]{40}$")
HOME_ASSISTANT_IMAGE = re.compile(
    r"ghcr\.io/home-assistant/home-assistant:[\w.-]+" r"(?:@sha256:[0-9a-fA-F]+)?"
)
IMMUTABLE_HOME_ASSISTANT_IMAGE = re.compile(
    r"^ghcr\.io/home-assistant/home-assistant:[\w.-]+@sha256:[0-9a-f]{64}$"
)


def test_github_actions_are_pinned_to_full_commit_shas() -> None:
    """Every external GitHub Action reference uses an immutable commit SHA."""
    mutable: list[str] = []
    for workflow in sorted((REPOSITORY_ROOT / ".github/workflows").glob("*.*ml")):
        for line_number, line in enumerate(workflow.read_text().splitlines(), 1):
            value = line.strip().removeprefix("- ").removeprefix("uses: ")
            if "uses:" not in line or value.startswith("./"):
                continue
            reference = value.split(" #", 1)[0].strip().strip('"')
            if not IMMUTABLE_ACTION.fullmatch(reference):
                mutable.append(f"{workflow.name}:{line_number}: {reference}")

    assert mutable == []


def test_consolidated_checks_do_not_persist_write_credentials() -> None:
    """Keep third-party hooks isolated from write-scoped checkout credentials."""
    workflow = yaml.safe_load(
        (REPOSITORY_ROOT / ".github/workflows/tests.yml").read_text()
    )
    checks = workflow["jobs"]["checks"]

    # OIDC authenticates Codecov; repository contents remain read-only.
    assert checks["permissions"] == {"contents": "read", "id-token": "write"}

    caller = yaml.safe_load((REPOSITORY_ROOT / ".github/workflows/ci.yml").read_text())[
        "jobs"
    ]["checks"]
    assert caller["permissions"] == checks["permissions"]

    checkout = next(
        step
        for step in checks["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"]["persist-credentials"] is False


def test_codecov_uploads_require_completed_pytest_step() -> None:
    """Do not attempt uploads when an earlier consolidated check skips pytest."""
    workflow = yaml.safe_load(
        (REPOSITORY_ROOT / ".github/workflows/tests.yml").read_text()
    )
    codecov_steps = [
        step
        for step in workflow["jobs"]["checks"]["steps"]
        if step.get("uses", "").startswith("codecov/codecov-action@")
    ]

    assert len(codecov_steps) == 2
    for step in codecov_steps:
        # Dependabot runs cannot access ordinary Actions upload secrets.
        assert step["with"]["use_oidc"] is True
        assert "token" not in step["with"]
        condition = step["if"]
        assert "steps.pytest.outcome == 'success'" in condition
        assert "steps.pytest.outcome == 'failure'" in condition
        assert "always()" not in condition


def test_home_assistant_container_defaults_are_digest_pinned() -> None:
    """Default development/runtime images use immutable OCI digests."""
    compose = (REPOSITORY_ROOT / "devtools/docker/docker-compose.yml").read_text()
    dockerfile = (REPOSITORY_ROOT / "devtools/docker/Dockerfile").read_text()

    compose_images = HOME_ASSISTANT_IMAGE.findall(compose)
    dockerfile_images = HOME_ASSISTANT_IMAGE.findall(dockerfile)

    assert len(compose_images) == 2
    assert len(dockerfile_images) == 1
    assert all(
        IMMUTABLE_HOME_ASSISTANT_IMAGE.fullmatch(image)
        for image in [*compose_images, *dockerfile_images]
    )
