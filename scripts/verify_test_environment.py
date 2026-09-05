#!/usr/bin/env python3
"""Verify the actual Home Assistant test environment, including helper pins."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Sequence

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def environment_errors(
    expected_core: str,
    expected_python: str,
    actual_core: str,
    actual_python: str,
    helper_requirements: Sequence[str],
) -> list[str]:
    """Identify mismatched image, interpreter, and pytest helper contracts."""
    errors = []
    if actual_core != expected_core:
        errors.append(f"Expected Home Assistant {expected_core}, got {actual_core}")
    if actual_python.split(".")[:2] != expected_python.split(".")[:2]:
        errors.append(f"Expected Python {expected_python}, got {actual_python}")
    core_requirements = [
        requirement
        for value in helper_requirements
        if canonicalize_name((requirement := Requirement(value)).name)
        == "homeassistant"
        and (requirement.marker is None or requirement.marker.evaluate())
    ]
    if not core_requirements:
        errors.append("The pytest helper does not declare its Home Assistant contract")
    for requirement in core_requirements:
        if actual_core not in requirement.specifier:
            errors.append(f"Pytest helper requires {requirement}, got {actual_core}")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Report exact versions and fail before tests on a mislabeled environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    expected = parser.add_mutually_exclusive_group(required=True)
    expected.add_argument("--expected-core")
    expected.add_argument("--expected-core-file", type=Path)
    parser.add_argument("--expected-python", default="3.14")
    parser.add_argument("--check-dependencies", action="store_true")
    args = parser.parse_args(argv)
    expected_core = (
        args.expected_core_file.read_text().strip()
        if args.expected_core_file
        else args.expected_core
    )
    helper = "pytest-homeassistant-custom-component"
    try:
        core = metadata.version("homeassistant")
        helper_version = metadata.version(helper)
        requirements = metadata.requires(helper) or []
    except metadata.PackageNotFoundError as err:
        print(f"Incomplete test environment: {err}", file=sys.stderr)
        return 1
    python = platform.python_version()
    print(json.dumps({"homeassistant": core, "python": python, helper: helper_version}))
    errors = environment_errors(
        expected_core, args.expected_python, core, python, requirements
    )
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    if args.check_dependencies:
        return subprocess.run(
            [sys.executable, "-m", "pip", "check"], check=False
        ).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
