from __future__ import annotations

import json
from pathlib import Path

from custom_components.groq.const import DOMAIN


def test_manifest_metadata_is_consistent() -> None:
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads(
        (root / "custom_components" / "groq" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["domain"] == "groq"
    assert manifest["domain"] == DOMAIN
    assert manifest["config_flow"] is True
    assert manifest["integration_type"] == "service"
    assert manifest["iot_class"] == "cloud_polling"
    assert manifest["requirements"] == []
    assert "single_config_entry" not in manifest
    assert manifest["documentation"].endswith("ha-groq")
    assert manifest["issue_tracker"].endswith("ha-groq/issues")


def test_hacs_minimum_homeassistant_version_is_declared() -> None:
    root = Path(__file__).resolve().parents[3]
    hacs = json.loads((root / "hacs.json").read_text(encoding="utf-8"))

    assert hacs["name"] == "Groq"
    assert hacs["render_readme"] is True
    assert hacs["homeassistant"] == "2026.5.1"


def test_translation_domain_matches_component_domain() -> None:
    """Guard against requesting translations for display name 'Groq'."""
    root = Path(__file__).resolve().parents[3]
    component_dir = root / "custom_components" / DOMAIN
    manifest = json.loads((component_dir / "manifest.json").read_text(encoding="utf-8"))

    assert component_dir.name == DOMAIN
    assert manifest["domain"] == DOMAIN
    assert manifest["domain"].islower()
    assert (component_dir / "strings.json").is_file()
    assert (component_dir / "translations" / "en.json").is_file()
