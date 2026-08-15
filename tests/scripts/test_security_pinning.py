"""Tests for immutable executable references in repository automation."""

from __future__ import annotations

import re
from pathlib import Path

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
