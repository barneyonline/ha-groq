#!/usr/bin/env python3
"""Validate Home Assistant quality scale evidence for this custom integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    print("ERROR: PyYAML is required. Install dev requirements.", file=sys.stderr)
    raise

DOMAIN = "groq"
QUALITY_LEVEL_ORDER = ("bronze", "silver", "gold", "platinum")
NA_ALLOWED_RULES = {
    "appropriate-polling",
    "discovery",
    "discovery-update-info",
    "entity-category",
    "entity-device-class",
    "entity-disabled-by-default",
    "entity-event-setup",
    "entity-unavailable",
    "icon-translations",
    "repair-issues",
    "stale-devices",
}
VALID_STATUSES = {"done", "n/a", "todo"}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _manifest(root: Path) -> dict[str, Any]:
    path = root / "custom_components" / DOMAIN / "manifest.json"
    if not path.exists():
        raise ValueError(f"{path} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _claimed_level(root: Path) -> str | None:
    raw_level = _manifest(root).get("quality_scale")
    if raw_level is None:
        return None
    level = str(raw_level).strip().lower()
    if level not in QUALITY_LEVEL_ORDER:
        raise ValueError(f"Unsupported manifest quality_scale value: {level!r}")
    return level


def _required_levels_for_claim(level: str | None) -> tuple[str, ...]:
    if level is None:
        return ()
    claimed_index = QUALITY_LEVEL_ORDER.index(level)
    return QUALITY_LEVEL_ORDER[: claimed_index + 1]


def _status_for_rule(entry: object) -> str:
    if isinstance(entry, dict):
        return str(entry.get("status") or "").strip().lower()
    if isinstance(entry, str):
        return entry.strip().lower()
    return ""


def _references_for_rule(entry: object) -> dict[str, list[str]]:
    if not isinstance(entry, dict):
        return {}
    references = entry.get("references")
    if not isinstance(references, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for key in ("code", "tests", "docs"):
        values = references.get(key)
        if isinstance(values, str):
            normalized[key] = [values]
        elif isinstance(values, list):
            normalized[key] = [str(value) for value in values if str(value).strip()]
    return normalized


def _reference_path_exists(root: Path, reference: str) -> bool:
    path_text = reference.split("#", 1)[0].strip()
    return bool(path_text) and (root / path_text).exists()


def validate_quality_scale(root: Path) -> tuple[int, list[str]]:
    """Return exit code and validation messages."""

    messages: list[str] = []
    try:
        claimed = _claimed_level(root)
        quality = _load_yaml(root / "quality_scale.yaml")
    except (OSError, ValueError, json.JSONDecodeError) as err:
        return 1, [f"ERROR: {err}"]

    levels = quality.get("levels") or {}
    rules = quality.get("rules") or {}
    required_rules: list[str] = []
    for level in _required_levels_for_claim(claimed):
        level_entry = levels.get(level) or {}
        required_rules.extend(str(rule) for rule in level_entry.get("required") or [])

    missing_rules = [rule for rule in required_rules if rule not in rules]
    if missing_rules:
        messages.append(f"ERROR: Missing rule evidence: {', '.join(missing_rules)}")

    incomplete_rules = []
    unknown_status_rules = []
    bad_na_rules = []
    missing_na_comments = []
    broken_references = []

    for rule, entry in rules.items():
        status = _status_for_rule(entry)
        if status not in VALID_STATUSES:
            unknown_status_rules.append(str(rule))

        if rule in required_rules and status not in {"done", "n/a"}:
            incomplete_rules.append(rule)
            continue
        if status == "n/a":
            if rule not in NA_ALLOWED_RULES:
                bad_na_rules.append(rule)
            if (
                not isinstance(entry, dict)
                or not str(entry.get("comment") or "").strip()
            ):
                missing_na_comments.append(rule)
        for refs in _references_for_rule(entry).values():
            for reference in refs:
                if not _reference_path_exists(root, reference):
                    broken_references.append(f"{rule}: {reference}")

    if unknown_status_rules:
        messages.append(
            "ERROR: Rules have an unsupported status: "
            + ", ".join(unknown_status_rules)
        )
    if incomplete_rules:
        messages.append(
            f"ERROR: Rules not marked done or n/a: {', '.join(incomplete_rules)}"
        )
    if bad_na_rules:
        messages.append(
            "ERROR: Rules marked n/a without an allowlist exception: "
            + ", ".join(bad_na_rules)
        )
    if missing_na_comments:
        messages.append(
            "ERROR: n/a rules missing explanatory comments: "
            + ", ".join(missing_na_comments)
        )
    if broken_references:
        messages.append(
            "ERROR: Broken quality scale references: " + ", ".join(broken_references)
        )

    return (1 if messages else 0), messages


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    exit_code, messages = validate_quality_scale(args.repo_root.resolve())
    for message in messages:
        print(message, file=sys.stderr)
    if exit_code == 0:
        print("Quality scale evidence is valid.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
