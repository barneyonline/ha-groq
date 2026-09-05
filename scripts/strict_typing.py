#!/usr/bin/env python3
"""Run strict integration checks against the installed Home Assistant source."""

from __future__ import annotations

import importlib.util
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    """Expose editable Core source to mypy without suppressing its interface types."""
    spec = importlib.util.find_spec("homeassistant")
    if spec is None or spec.origin is None:
        print("Home Assistant source is not installed", file=sys.stderr)
        return 1
    source = Path(spec.origin).resolve().parent.parent
    env = os.environ.copy()
    env.setdefault(
        "MYPY_CACHE_DIR",
        f".mypy_cache/strict/{version('homeassistant')}/{version('mypy')}",
    )
    env["MYPYPATH"] = os.pathsep.join(
        value for value in (str(source), env.get("MYPYPATH", "")) if value
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--follow-imports=silent",
            "custom_components/groq",
        ],
        env=env,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
