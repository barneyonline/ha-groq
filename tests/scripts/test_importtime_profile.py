from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _load_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "scripts" / "importtime_profile.py"
    spec = importlib.util.spec_from_file_location("importtime_profile", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


importtime_profile = _load_module()


def test_discover_modules_returns_integration_package_modules(tmp_path: Path) -> None:
    package_root = tmp_path / "custom_components" / "groq"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "api.py").write_text("", encoding="utf-8")
    (package_root / "tts.py").write_text("", encoding="utf-8")
    (package_root / "README.md").write_text("ignored", encoding="utf-8")

    modules = importtime_profile.discover_modules(tmp_path)

    assert modules == (
        "custom_components.groq",
        "custom_components.groq.api",
        "custom_components.groq.tts",
    )


def test_build_import_runner_imports_each_module() -> None:
    runner = importtime_profile.build_import_runner(
        ("package", "package.module"),
        preload_modules=("preload.package",),
        warning_module_pattern=r"package(\.|$)",
    )

    assert "importlib.import_module(module)" in runner
    assert "importlib.import_module(preload_module)" in runner
    assert '"package", "package.module"' in runner
    assert '"preload.package"' in runner
    assert "warnings.filterwarnings" in runner
    assert "DeprecationWarning" in runner
    assert "preloaded {len(preload_modules)} modules" in runner
    assert "imported {len(modules)} modules" in runner
    assert "integration import duration ms" in runner


def test_run_importtime_uses_importtime(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return subprocess.CompletedProcess(args[0], 0, "imported 1 modules\n", "")

    monkeypatch.setattr(importtime_profile.subprocess, "run", fake_run)

    result = importtime_profile.run_importtime(
        tmp_path,
        ("custom_components.groq",),
        python="python3.14",
        preload_modules=("homeassistant.bootstrap",),
        strict_warnings=True,
    )

    assert result.returncode == 0
    command = calls[0]["args"][0]
    assert command[:3] == ["python3.14", "-X", "importtime=2"]
    assert "custom_components\\\\.groq" in command[4]
    assert "homeassistant.bootstrap" in command[4]
    assert calls[0]["kwargs"]["cwd"] == tmp_path
    assert calls[0]["kwargs"]["text"] is True
    assert calls[0]["kwargs"]["capture_output"] is True
    assert str(tmp_path) in calls[0]["kwargs"]["env"]["PYTHONPATH"].split(os.pathsep)


def test_render_reports_includes_median_duration() -> None:
    results = tuple(
        subprocess.CompletedProcess(
            ["python"],
            0,
            f"integration import duration ms: {duration}\n",
            "",
        )
        for duration in (30, 10, 20)
    )

    report = importtime_profile.render_reports(results)

    assert "run 1/3" in report
    assert "run 3/3" in report
    assert "median integration import duration ms: 20.000" in report


def test_parse_args_rejects_nonpositive_runs() -> None:
    with pytest.raises(SystemExit):
        importtime_profile.parse_args(["--runs", "0"])


def test_main_writes_output(monkeypatch, tmp_path: Path) -> None:
    def fake_run_importtime(
        _root,
        _modules,
        *,
        python,
        preload_modules,
        strict_warnings,
        warning_module_pattern,
    ):
        assert strict_warnings is True
        assert preload_modules == ("homeassistant.bootstrap",)
        assert warning_module_pattern == r"custom_components\.groq(\.|$)"
        return subprocess.CompletedProcess(
            [python], 0, "imported 1 modules\n", "import time: test\n"
        )

    monkeypatch.setattr(
        importtime_profile,
        "run_importtime",
        fake_run_importtime,
    )
    output = tmp_path / "importtime.log"

    exit_code = importtime_profile.main(
        [
            "--repo-root",
            str(tmp_path),
            "--module",
            "custom_components.groq",
            "--preload-home-assistant",
            "--strict-integration-warnings",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert output.read_text(encoding="utf-8") == (
        "imported 1 modules\n\nimport time: test\n"
    )
