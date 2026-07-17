#!/usr/bin/env python3
"""Validate Home Assistant quality scale evidence for this custom integration."""

from __future__ import annotations

import argparse
from datetime import date
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
RULES_BY_LEVEL: dict[str, tuple[str, ...]] = {
    "bronze": (
        "action-setup",
        "appropriate-polling",
        "brands",
        "common-modules",
        "config-flow-test-coverage",
        "config-flow",
        "dependency-transparency",
        "docs-actions",
        "docs-triggers",
        "docs-conditions",
        "docs-high-level-description",
        "docs-installation-instructions",
        "docs-removal-instructions",
        "entity-event-setup",
        "entity-unique-id",
        "has-entity-name",
        "runtime-data",
        "test-before-configure",
        "test-before-setup",
        "unique-config-entry",
    ),
    "silver": (
        "action-exceptions",
        "config-entry-unloading",
        "docs-configuration-parameters",
        "docs-installation-parameters",
        "entity-unavailable",
        "integration-owner",
        "log-when-unavailable",
        "parallel-updates",
        "reauthentication-flow",
        "test-coverage",
    ),
    "gold": (
        "devices",
        "diagnostics",
        "discovery-update-info",
        "discovery",
        "docs-data-update",
        "docs-examples",
        "docs-known-limitations",
        "docs-supported-devices",
        "docs-supported-functions",
        "docs-troubleshooting",
        "docs-use-cases",
        "dynamic-devices",
        "entity-category",
        "entity-device-class",
        "entity-disabled-by-default",
        "entity-translations",
        "exception-translations",
        "icon-translations",
        "reconfiguration-flow",
        "repair-issues",
        "stale-devices",
    ),
    "platinum": (
        "async-dependency",
        "inject-websession",
        "strict-typing",
    ),
}
OFFICIAL_RULES_SOURCE = (
    "https://developers.home-assistant.io/docs/core/integration-quality-scale/rules/"
)
MAX_RULE_CATALOG_AGE_DAYS = 180
EXEMPT_ALLOWED_RULES = {
    "appropriate-polling",
    "discovery",
    "discovery-update-info",
    "docs-conditions",
    "docs-triggers",
    "entity-category",
    "entity-device-class",
    "entity-disabled-by-default",
    "entity-event-setup",
    "entity-unavailable",
    "icon-translations",
    "stale-devices",
}
VALID_STATUSES = {"done", "exempt", "todo"}


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


def _required_rules_for_claim(level: str | None) -> list[str]:
    return [
        rule
        for required_level in _required_levels_for_claim(level)
        for rule in RULES_BY_LEVEL[required_level]
    ]


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

    metadata = quality.get("metadata") or {}
    rules = quality.get("rules") or {}
    if not isinstance(rules, dict):
        return 1, ["ERROR: Quality scale rules must be a mapping"]
    required_rules = _required_rules_for_claim(claimed)

    if "levels" in quality:
        messages.append(
            "ERROR: quality_scale.yaml must not redefine tier rules; the validator "
            "tracks the current official rule catalog"
        )

    if claimed is not None:
        if not isinstance(metadata, dict):
            messages.append("ERROR: Quality scale metadata must be a mapping")
        else:
            if metadata.get("assessment") != "self-assessed":
                messages.append(
                    "ERROR: Custom integration quality claims must be marked "
                    "self-assessed"
                )
            if metadata.get("home_assistant_reviewed") is not False:
                messages.append(
                    "ERROR: Custom integration quality evidence must state that it "
                    "has not been reviewed by Home Assistant"
                )
            if metadata.get("rules_source") != OFFICIAL_RULES_SOURCE:
                messages.append(
                    "ERROR: Quality evidence has an unsupported rules source"
                )
            verified = metadata.get("rules_last_verified")
            try:
                verified_date = date.fromisoformat(str(verified))
            except ValueError:
                messages.append(
                    "ERROR: rules_last_verified must be an ISO date (YYYY-MM-DD)"
                )
            else:
                age_days = (date.today() - verified_date).days
                if age_days < 0:
                    messages.append(
                        "ERROR: rules_last_verified cannot be in the future"
                    )
                elif age_days > MAX_RULE_CATALOG_AGE_DAYS:
                    messages.append(
                        "ERROR: Official quality rules must be rechecked at least "
                        f"every {MAX_RULE_CATALOG_AGE_DAYS} days"
                    )

    missing_rules = [rule for rule in required_rules if rule not in rules]
    if missing_rules:
        messages.append(f"ERROR: Missing rule evidence: {', '.join(missing_rules)}")

    incomplete_rules = []
    unknown_status_rules = []
    unknown_rules = []
    bad_exempt_rules = []
    missing_exempt_comments = []
    missing_references = []
    broken_references = []

    known_rules = {
        rule for level_rules in RULES_BY_LEVEL.values() for rule in level_rules
    }

    for rule, entry in rules.items():
        if rule not in known_rules:
            unknown_rules.append(str(rule))
        status = _status_for_rule(entry)
        if status not in VALID_STATUSES:
            unknown_status_rules.append(str(rule))

        if rule in required_rules and status not in {"done", "exempt"}:
            incomplete_rules.append(rule)
            continue
        if status == "exempt":
            if rule not in EXEMPT_ALLOWED_RULES:
                bad_exempt_rules.append(rule)
            if (
                not isinstance(entry, dict)
                or not str(entry.get("comment") or "").strip()
            ):
                missing_exempt_comments.append(rule)
        references = _references_for_rule(entry)
        if status in {"done", "exempt"} and not any(references.values()):
            missing_references.append(rule)
        for refs in references.values():
            for reference in refs:
                if not _reference_path_exists(root, reference):
                    broken_references.append(f"{rule}: {reference}")

    dependency_entry = rules.get("dependency-transparency")
    if dependency_entry is not None or claimed is not None:
        evidence_requirements = (
            dependency_entry.get("requirements")
            if isinstance(dependency_entry, dict)
            else None
        )
        manifest_requirements = _manifest(root).get("requirements") or []
        if evidence_requirements != manifest_requirements:
            messages.append(
                "ERROR: dependency-transparency requirements do not match manifest: "
                f"evidence={evidence_requirements!r}, "
                f"manifest={manifest_requirements!r}"
            )

    if unknown_rules:
        messages.append(
            "ERROR: Evidence contains unknown quality scale rules: "
            + ", ".join(unknown_rules)
        )
    if unknown_status_rules:
        messages.append(
            "ERROR: Rules have an unsupported status: "
            + ", ".join(unknown_status_rules)
        )
    if incomplete_rules:
        messages.append(
            f"ERROR: Rules not marked done or exempt: {', '.join(incomplete_rules)}"
        )
    if bad_exempt_rules:
        messages.append(
            "ERROR: Rules marked exempt without an allowlist exception: "
            + ", ".join(bad_exempt_rules)
        )
    if missing_exempt_comments:
        messages.append(
            "ERROR: exempt rules missing explanatory comments: "
            + ", ".join(missing_exempt_comments)
        )
    if missing_references:
        messages.append(
            "ERROR: Completed/exempt rules missing evidence references: "
            + ", ".join(missing_references)
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
