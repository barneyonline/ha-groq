from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import yaml


def _load_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "scripts" / "validate_quality_scale.py"
    spec = importlib.util.spec_from_file_location("validate_quality_scale", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validate_quality_scale = _load_module()


def _write_manifest(
    root: Path,
    quality_scale: str | None = "bronze",
    requirements: list[str] | None = None,
) -> None:
    manifest_dir = root / "custom_components" / "groq"
    manifest_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "domain": "groq",
        "requirements": requirements or [],
    }
    if quality_scale is not None:
        manifest["quality_scale"] = quality_scale
    (manifest_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _write_reference(root: Path, path: str = "README.md") -> None:
    reference = root / path
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text("ok", encoding="utf-8")


def _metadata() -> dict[str, object]:
    return {
        "assessment": "self-assessed",
        "home_assistant_reviewed": False,
        "rules_source": validate_quality_scale.OFFICIAL_RULES_SOURCE,
        "rules_last_verified": "2026-07-17",
    }


def _rules_for_claim(
    level: str,
    *,
    requirements: list[str] | None = None,
) -> dict[str, dict[str, object]]:
    rules = {
        rule: {
            "status": "done",
            "references": {"docs": ["README.md"]},
        }
        for rule in validate_quality_scale._required_rules_for_claim(level)
    }
    rules["dependency-transparency"]["requirements"] = requirements or []
    return rules


def _write_quality_scale(
    root: Path,
    rules: dict[str, object],
    *,
    metadata: dict[str, object] | None = None,
    levels: dict[str, object] | None = None,
) -> None:
    body: dict[str, object] = {"rules": rules}
    if metadata is not None:
        body["metadata"] = metadata
    if levels is not None:
        body["levels"] = levels
    (root / "quality_scale.yaml").write_text(
        yaml.safe_dump(body, sort_keys=False),
        encoding="utf-8",
    )


def test_repository_quality_scale_matches_manifest_claim() -> None:
    root = Path(__file__).resolve().parents[2]

    exit_code, messages = validate_quality_scale.validate_quality_scale(root)

    assert exit_code == 0, "\n".join(messages)


def test_current_bronze_catalog_includes_trigger_and_condition_docs() -> None:
    bronze = validate_quality_scale.RULES_BY_LEVEL["bronze"]

    assert "docs-triggers" in bronze
    assert "docs-conditions" in bronze


def test_claim_requires_all_current_rules_for_level(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    _write_reference(tmp_path)
    rules = _rules_for_claim("bronze")
    del rules["docs-triggers"]
    _write_quality_scale(tmp_path, rules, metadata=_metadata())

    exit_code, messages = validate_quality_scale.validate_quality_scale(tmp_path)

    assert exit_code == 1
    assert "docs-triggers" in "\n".join(messages)


def test_unclaimed_manifest_allows_tracked_todo_rules(tmp_path: Path) -> None:
    _write_manifest(tmp_path, None)
    _write_reference(tmp_path)
    _write_quality_scale(
        tmp_path,
        {
            "config-flow": {
                "status": "todo",
                "comment": "Tracked gap before a manifest quality claim is made.",
                "references": {"docs": ["README.md"]},
            }
        },
    )

    exit_code, messages = validate_quality_scale.validate_quality_scale(tmp_path)

    assert exit_code == 0, "\n".join(messages)


def test_exempt_rules_need_explanatory_comments(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    _write_reference(tmp_path)
    rules = _rules_for_claim("bronze")
    rules["docs-triggers"] = {
        "status": "exempt",
        "references": {"docs": ["README.md"]},
    }
    _write_quality_scale(tmp_path, rules, metadata=_metadata())

    exit_code, messages = validate_quality_scale.validate_quality_scale(tmp_path)

    assert exit_code == 1
    assert "exempt rules missing explanatory comments" in "\n".join(messages)


def test_documented_integration_exceptions_may_be_exempt(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    _write_reference(tmp_path)
    rules = _rules_for_claim("bronze")
    rules["docs-conditions"] = {
        "status": "exempt",
        "comment": "The integration does not provide custom conditions.",
        "references": {"docs": ["README.md"]},
    }
    _write_quality_scale(tmp_path, rules, metadata=_metadata())

    exit_code, messages = validate_quality_scale.validate_quality_scale(tmp_path)

    assert exit_code == 0, "\n".join(messages)


def test_exempt_status_is_restricted_to_allowlisted_rules(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    _write_reference(tmp_path)
    rules = _rules_for_claim("bronze")
    rules["config-flow"] = {
        "status": "exempt",
        "comment": "Not acceptable for this rule.",
        "references": {"docs": ["README.md"]},
    }
    _write_quality_scale(tmp_path, rules, metadata=_metadata())

    exit_code, messages = validate_quality_scale.validate_quality_scale(tmp_path)

    assert exit_code == 1
    assert "marked exempt without an allowlist exception" in "\n".join(messages)


def test_unknown_status_is_reported(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    _write_reference(tmp_path)
    rules = _rules_for_claim("bronze")
    rules["config-flow"]["status"] = "partial"
    _write_quality_scale(tmp_path, rules, metadata=_metadata())

    exit_code, messages = validate_quality_scale.validate_quality_scale(tmp_path)

    assert exit_code == 1
    assert "unsupported status" in "\n".join(messages)


def test_unknown_rule_is_reported(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    _write_reference(tmp_path)
    rules = _rules_for_claim("bronze")
    rules["invented-rule"] = {
        "status": "done",
        "references": {"docs": ["README.md"]},
    }
    _write_quality_scale(tmp_path, rules, metadata=_metadata())

    exit_code, messages = validate_quality_scale.validate_quality_scale(tmp_path)

    assert exit_code == 1
    assert "unknown quality scale rules" in "\n".join(messages)


def test_completed_rules_need_evidence_references(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    _write_reference(tmp_path)
    rules = _rules_for_claim("bronze")
    rules["config-flow"] = {"status": "done"}
    _write_quality_scale(tmp_path, rules, metadata=_metadata())

    exit_code, messages = validate_quality_scale.validate_quality_scale(tmp_path)

    assert exit_code == 1
    assert "missing evidence references" in "\n".join(messages)


def test_broken_references_are_reported(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    _write_reference(tmp_path)
    rules = _rules_for_claim("bronze")
    rules["config-flow"]["references"] = {"docs": ["MISSING.md"]}
    _write_quality_scale(tmp_path, rules, metadata=_metadata())

    exit_code, messages = validate_quality_scale.validate_quality_scale(tmp_path)

    assert exit_code == 1
    assert "MISSING.md" in "\n".join(messages)


def test_custom_claim_requires_honest_self_assessment_metadata(
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path)
    _write_reference(tmp_path)
    _write_quality_scale(tmp_path, _rules_for_claim("bronze"))

    exit_code, messages = validate_quality_scale.validate_quality_scale(tmp_path)

    assert exit_code == 1
    message = "\n".join(messages)
    assert "self-assessed" in message
    assert "not been reviewed by Home Assistant" in message


def test_rule_catalog_verification_must_be_current(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    _write_reference(tmp_path)
    metadata = _metadata()
    metadata["rules_last_verified"] = "2000-01-01"
    _write_quality_scale(
        tmp_path,
        _rules_for_claim("bronze"),
        metadata=metadata,
    )

    exit_code, messages = validate_quality_scale.validate_quality_scale(tmp_path)

    assert exit_code == 1
    assert "must be rechecked" in "\n".join(messages)


def test_dependency_evidence_must_match_manifest(tmp_path: Path) -> None:
    _write_manifest(tmp_path, requirements=["jsonschema==4.26.0"])
    _write_reference(tmp_path)
    rules = _rules_for_claim("bronze", requirements=["jsonschema==4.25.0"])
    _write_quality_scale(tmp_path, rules, metadata=_metadata())

    exit_code, messages = validate_quality_scale.validate_quality_scale(tmp_path)

    assert exit_code == 1
    assert "requirements do not match manifest" in "\n".join(messages)


def test_evidence_cannot_redefine_official_tier_rules(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    _write_reference(tmp_path)
    _write_quality_scale(
        tmp_path,
        _rules_for_claim("bronze"),
        metadata=_metadata(),
        levels={"bronze": {"required": ["config-flow"]}},
    )

    exit_code, messages = validate_quality_scale.validate_quality_scale(tmp_path)

    assert exit_code == 1
    assert "must not redefine tier rules" in "\n".join(messages)
