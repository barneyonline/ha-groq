#!/usr/bin/env python3
"""Check real HA mapping and callback types, including a negative control."""

from __future__ import annotations

import importlib.util
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

POSITIVE = """from typing import Any, assert_type
from types import MappingProxyType
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.components.conversation import ChatLog, AssistantContent

async def listener(hass: HomeAssistant, entry: ConfigEntry[Any]) -> None:
    pass

def check(entry: ConfigEntry[Any], subentry: ConfigSubentry, log: ChatLog) -> None:
    assert_type(entry.data, MappingProxyType[str, Any])
    assert_type(subentry.data, MappingProxyType[str, Any])
    entry.add_update_listener(listener)
    log.async_add_assistant_content(AssistantContent(agent_id="groq", content="ok"))
"""

NEGATIVE = """from typing import Any
from homeassistant.config_entries import ConfigEntry
from homeassistant.components.conversation import ChatLog

def check(entry: ConfigEntry[Any], log: ChatLog) -> None:
    entry.data["invalid_mutation"] = True
    entry.add_update_listener(lambda: 42)
    log.async_add_assistant_content("invalid content")
"""


def valid_control_results(
    positive: subprocess.CompletedProcess[str],
    negative: subprocess.CompletedProcess[str],
) -> bool:
    """Reject an accidentally permissive checker as well as incompatible APIs."""
    return (
        positive.returncode == 0
        and negative.returncode == 1
        and "[index]" in negative.stdout
        and negative.stdout.count("[arg-type]") >= 2
    )


def main() -> int:
    """Locate the installed version's source instead of relying on editable paths."""
    spec = importlib.util.find_spec("homeassistant")
    if spec is None or spec.origin is None:
        print("Home Assistant source is not installed", file=sys.stderr)
        return 1
    source_root = Path(spec.origin).resolve().parent.parent
    env = os.environ.copy()
    # Both Docker versions mount this workspace. Their HA source paths are
    # identical, so concurrent checks must not overwrite each other's caches.
    env.setdefault(
        "MYPY_CACHE_DIR",
        f".mypy_cache/contracts/{version('homeassistant')}/{version('mypy')}",
    )
    env["MYPYPATH"] = os.pathsep.join(
        part for part in (str(source_root), env.get("MYPYPATH", "")) if part
    )
    with TemporaryDirectory(prefix="groq-ha-types-") as directory:
        root = Path(directory)
        config = root / "mypy.ini"
        config.write_text(
            "[mypy]\nstrict = True\nfollow_imports = silent\nignore_missing_imports = True\n"
        )
        results = []
        for name, code in (("valid", POSITIVE), ("invalid", NEGATIVE)):
            probe = root / f"{name}.py"
            probe.write_text(code)
            results.append(
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "mypy",
                        "--config-file",
                        str(config),
                        str(probe),
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            )
        if valid_control_results(*results):
            print(
                "HA source contracts verified; immutable-data and callback negative controls rejected."
            )
            return 0
        for result in results:
            print(result.stdout, end="")
            print(result.stderr, end="", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
