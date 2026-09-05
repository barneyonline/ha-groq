"""Regression tests for mislabeled Home Assistant compatibility environments."""

from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import verify_test_environment as verifier


@pytest.mark.parametrize("core", ["2026.6.0", "2026.9.0"])
def test_matched_helper_and_image_are_accepted(core):
    assert (
        verifier.environment_errors(
            core, "3.14", core, "3.14.6", [f"homeassistant=={core}", "pytest>=9"]
        )
        == []
    )


def test_helper_cannot_silently_downgrade_image():
    errors = verifier.environment_errors(
        "2026.9.0", "3.14", "2026.7.2", "3.14.6", ["homeassistant==2026.7.2"]
    )
    assert errors == ["Expected Home Assistant 2026.9.0, got 2026.7.2"]


def test_no_deps_install_cannot_hide_incompatible_helper():
    errors = verifier.environment_errors(
        "2026.9.0", "3.14", "2026.9.0", "3.14.6", ["homeassistant==2026.7.2"]
    )
    assert errors == ["Pytest helper requires homeassistant==2026.7.2, got 2026.9.0"]


def test_missing_contract_and_wrong_interpreter_fail():
    errors = verifier.environment_errors(
        "2026.9.0", "3.14", "2026.9.0", "3.13.9", ["pytest>=9"]
    )
    assert len(errors) == 2
    assert "Python" in errors[0]
    assert "does not declare" in errors[1]


def test_cli_checks_dependencies_and_reports_versions(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(verifier.metadata, "version", lambda name: "2026.9.0")
    monkeypatch.setattr(
        verifier.metadata, "requires", lambda name: ["homeassistant==2026.9.0"]
    )
    monkeypatch.setattr(verifier.platform, "python_version", lambda: "3.14.6")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(verifier.subprocess, "run", run)
    expected = tmp_path / "core-version"
    expected.write_text("2026.9.0\n")
    assert (
        verifier.main(["--expected-core-file", str(expected), "--check-dependencies"])
        == 1
    )
    assert calls == [[verifier.sys.executable, "-m", "pip", "check"]]
    assert '"homeassistant": "2026.9.0"' in capsys.readouterr().out
    assert verifier.main(["--expected-core", "2026.9.0"]) == 0
    assert verifier.main(["--expected-core", "2026.6.0"]) == 1
    assert "Expected Home Assistant" in capsys.readouterr().err


def test_cli_rejects_missing_helper(monkeypatch, capsys):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(verifier.metadata, "version", missing)
    assert verifier.main(["--expected-core", "2026.9.0"]) == 1
    assert "Incomplete test environment" in capsys.readouterr().err


def test_image_and_hacs_minimum_pins_match_test_environments():
    """Catch drift between declared support, requirements and image defaults."""
    import json
    import re

    root = Path(__file__).parents[2]
    dockerfile = (root / "devtools/docker/Dockerfile").read_text()
    compose = (root / "devtools/docker/docker-compose.yml").read_text()
    image = dockerfile.splitlines()[0].removeprefix("ARG HA_IMAGE=")
    assert image in compose
    current = re.search(r"home-assistant:([0-9.]+)@", image)[1]
    assert current == "2026.9.0"
    minimum = json.loads((root / "hacs.json").read_text())["homeassistant"]
    assert (
        f"homeassistant=={minimum}"
        in (root / "devtools/docker/requirements-min-ha.txt").read_text()
    )
    assert f"home-assistant:{minimum}@" in (root / "scripts/test").read_text()
