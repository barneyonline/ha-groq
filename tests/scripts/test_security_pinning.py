"""Tests for immutable executable references in repository automation."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).parents[2]
IMMUTABLE_ACTION = re.compile(r"^[\w.-]+/[\w.-]+(?:/[\w.-]+)*@[0-9a-f]{40}$")
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
            if not line.strip().removeprefix("- ").startswith(
                "uses:"
            ) or value.startswith("./"):
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

    # Candidate checks use read access; protected reporting owns Codecov uploads.
    assert checks["permissions"] == {"contents": "read"}

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


def test_candidate_checks_have_no_codecov_or_inherited_secrets() -> None:
    """Mutable test programs cannot share a runner with upload credentials."""
    for filename in ("tests.yml", "ci.yml"):
        source = (REPOSITORY_ROOT / ".github/workflows" / filename).read_text()
        workflow = yaml.safe_load(source)
        assert workflow["permissions"] == {"contents": "read"}
        assert "codecov/codecov-action@" not in source
        assert "CODECOV_TOKEN" not in source
        assert "secrets:" not in source
        for job in workflow["jobs"].values():
            assert job.get("permissions", {"contents": "read"}) == {"contents": "read"}
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("actions/checkout@"):
                    assert step["with"]["persist-credentials"] is False


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


@pytest.mark.parametrize("symlink", [None, ".test-cache", ".test-cache/pre-commit"])
def test_cache_ownership_cleanup_rejects_external_symlinks(tmp_path, symlink) -> None:
    """Runner cleanup must never change ownership through a cache-path symlink."""
    workflow = yaml.safe_load(
        (REPOSITORY_ROOT / ".github/workflows/tests.yml").read_text()
    )
    cleanup = next(
        step
        for step in workflow["jobs"]["checks"]["steps"]
        if step.get("name") == "Restore pre-commit cache ownership for the runner"
    )
    assert cleanup["if"] == "${{ always() }}"
    cache = tmp_path / ".test-cache/pre-commit"
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "marker"
    marker.write_text("leave this file alone")
    if symlink is None:
        cache.mkdir(parents=True)
    else:
        link = tmp_path / symlink
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(outside, target_is_directory=True)

    # Record the privileged command while exercising the actual workflow script.
    binaries = tmp_path / "bin"
    binaries.mkdir()
    sudo = binaries / "sudo"
    sudo.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$SUDO_ARGUMENTS"\n')
    sudo.chmod(0o755)
    recorded = tmp_path / "sudo-arguments"
    result = subprocess.run(
        ["bash", "-eu", "-c", cleanup["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
            "SUDO_ARGUMENTS": str(recorded),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    if symlink is None:
        assert result.returncode == 0
        assert recorded.read_text().splitlines() == [
            "chown",
            "-RhP",
            "--",
            f"{os.getuid()}:{os.getgid()}",
            ".test-cache/pre-commit",
        ]
    else:
        assert result.returncode == 1
        assert "symbolic link" in result.stderr
        assert not recorded.exists()
    assert marker.read_text() == "leave this file alone"
