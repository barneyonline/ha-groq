#!/usr/bin/env python3
"""Profile Groq integration import time with Python importtime output."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

DEFAULT_PACKAGE = "custom_components.groq"
IMPORT_DURATION_PATTERN = re.compile(
    r"integration import duration ms: (?P<duration>[0-9.]+)"
)


def discover_modules(root: Path, package: str = DEFAULT_PACKAGE) -> tuple[str, ...]:
    """Return top-level modules in the integration package."""

    package_path = root / Path(*package.split("."))
    if not package_path.is_dir():
        raise ValueError(f"Package path not found: {package_path}")

    modules = [package]
    modules.extend(
        f"{package}.{path.stem}"
        for path in sorted(package_path.glob("*.py"))
        if path.name != "__init__.py"
    )
    return tuple(modules)


def build_import_runner(
    modules: Sequence[str],
    *,
    preload_modules: Sequence[str] = (),
    warning_module_pattern: str | None = None,
) -> str:
    """Return Python code that imports every requested module."""

    modules_json = json.dumps(list(modules))
    preload_modules_json = json.dumps(list(preload_modules))
    warning_lines = ""
    if warning_module_pattern is not None:
        warning_lines = (
            "import warnings\n"
            f"_warning_module = {warning_module_pattern!r}\n"
            "for _category in (\n"
            "    DeprecationWarning,\n"
            "    PendingDeprecationWarning,\n"
            "    FutureWarning,\n"
            "    SyntaxWarning,\n"
            "):\n"
            "    warnings.filterwarnings(\n"
            '        "error", category=_category, module=_warning_module\n'
            "    )\n"
        )
    return (
        "import importlib\n"
        "import time\n"
        f"{warning_lines}"
        f"preload_modules = {preload_modules_json}\n"
        "for preload_module in preload_modules:\n"
        "    importlib.import_module(preload_module)\n"
        f"modules = {modules_json}\n"
        "import_start = time.perf_counter()\n"
        "for module in modules:\n"
        "    importlib.import_module(module)\n"
        "integration_import_ms = (time.perf_counter() - import_start) * 1000\n"
        "if preload_modules:\n"
        "    print(f'preloaded {len(preload_modules)} modules')\n"
        "print(f'imported {len(modules)} modules')\n"
        "print(f'integration import duration ms: {integration_import_ms:.3f}')\n"
    )


def run_importtime(
    root: Path,
    modules: Sequence[str],
    *,
    python: str = sys.executable,
    preload_modules: Sequence[str] = (),
    strict_warnings: bool = False,
    warning_module_pattern: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the import-time profiler for the requested modules."""

    if strict_warnings and warning_module_pattern is None:
        warning_module_pattern = rf"{re.escape(DEFAULT_PACKAGE)}(\.|$)"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(root), env.get("PYTHONPATH", "")) if part
    )
    return subprocess.run(
        [
            python,
            "-X",
            "importtime=2",
            "-c",
            build_import_runner(
                modules,
                preload_modules=preload_modules,
                warning_module_pattern=warning_module_pattern,
            ),
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def render_report(result: subprocess.CompletedProcess[str]) -> str:
    """Combine importtime stderr with the import runner summary."""

    sections: list[str] = []
    if result.stdout:
        sections.append(result.stdout.rstrip())
    if result.stderr:
        sections.append(result.stderr.rstrip())
    return "\n\n".join(sections) + "\n"


def render_reports(results: Sequence[subprocess.CompletedProcess[str]]) -> str:
    """Render one or more import reports with a median duration summary."""
    if len(results) == 1:
        return render_report(results[0])
    reports = [
        f"run {index}/{len(results)}\n{render_report(result).rstrip()}"
        for index, result in enumerate(results, start=1)
    ]
    durations = [
        float(match.group("duration"))
        for result in results
        if (match := IMPORT_DURATION_PATTERN.search(result.stdout)) is not None
    ]
    if durations:
        reports.append(
            "median integration import duration ms: "
            f"{statistics.median(durations):.3f}"
        )
    return "\n\n".join(reports) + "\n"


def _positive_int(value: str) -> int:
    """Return a positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run '-X importtime=2' against Groq integration modules."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to import from.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to run with '-X importtime=2'.",
    )
    parser.add_argument(
        "--package",
        default=DEFAULT_PACKAGE,
        help="Package to discover modules from when --module is not provided.",
    )
    parser.add_argument(
        "--module",
        action="append",
        dest="modules",
        help="Explicit module to import. May be supplied more than once.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the combined importtime report.",
    )
    parser.add_argument(
        "--runs",
        type=_positive_int,
        default=1,
        help="Run the profile in multiple fresh processes and report median duration.",
    )
    parser.add_argument(
        "--preload-home-assistant",
        action="store_true",
        help=(
            "Import homeassistant.bootstrap before the integration to profile "
            "incremental import cost in normal Home Assistant startup order."
        ),
    )
    parser.add_argument(
        "--strict-integration-warnings",
        action="store_true",
        help=(
            "Treat import-time DeprecationWarning, PendingDeprecationWarning, "
            "FutureWarning, and SyntaxWarning from the integration package as errors."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.repo_root.resolve()
    modules = tuple(args.modules or discover_modules(root, args.package))
    warning_module_pattern = (
        rf"{re.escape(args.package)}(\.|$)"
        if args.strict_integration_warnings
        else None
    )
    results = tuple(
        run_importtime(
            root,
            modules,
            python=args.python,
            preload_modules=(
                ("homeassistant.bootstrap",) if args.preload_home_assistant else ()
            ),
            strict_warnings=args.strict_integration_warnings,
            warning_module_pattern=warning_module_pattern,
        )
        for _ in range(args.runs)
    )
    report = render_reports(results)

    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")

    returncode = next((result.returncode for result in results if result.returncode), 0)
    if returncode != 0 and args.output:
        print(report, file=sys.stderr, end="")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
