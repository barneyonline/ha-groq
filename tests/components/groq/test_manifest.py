from __future__ import annotations

import json
from pathlib import Path
import re

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
    assert manifest["after_dependencies"] == ["assist_pipeline", "camera", "intent"]
    assert manifest["dependencies"] == ["conversation", "media_source"]
    assert manifest["integration_type"] == "service"
    assert manifest["iot_class"] == "cloud_polling"
    assert manifest["loggers"] == []
    assert manifest["quality_scale"] == "platinum"
    assert manifest["requirements"] == ["jsonschema==4.26.0"]
    assert manifest["version"] == "1.1.0"
    assert "single_config_entry" not in manifest
    assert manifest["documentation"].endswith("ha-groq")
    assert manifest["issue_tracker"].endswith("ha-groq/issues")


def test_hacs_minimum_homeassistant_version_is_declared() -> None:
    root = Path(__file__).resolve().parents[3]
    hacs = json.loads((root / "hacs.json").read_text(encoding="utf-8"))

    assert hacs["name"] == "Groq"
    assert hacs["render_readme"] is True
    assert hacs["homeassistant"] == "2026.3.0"
    assert hacs["content_in_root"] is False


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


def _leaf_paths(value: object, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        paths: set[str] = set()
        for key, item in value.items():
            paths |= _leaf_paths(item, f"{prefix}.{key}" if prefix else key)
        return paths
    return {prefix}


def _leaf_values(value: object, prefix: str = "") -> dict[str, str]:
    if isinstance(value, dict):
        values: dict[str, str] = {}
        for key, item in value.items():
            values.update(_leaf_values(item, f"{prefix}.{key}" if prefix else key))
        return values
    return {prefix: str(value)}


def _placeholders(value: str) -> set[str]:
    return set(re.findall(r"\{[^{}]+\}", value))


def test_supported_translation_locales_are_declared() -> None:
    root = Path(__file__).resolve().parents[3]
    translation_dir = root / "custom_components" / DOMAIN / "translations"

    assert {path.stem for path in translation_dir.glob("*.json")} == {
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "en-AU",
        "en-CA",
        "en-IE",
        "en-NZ",
        "en-US",
        "es",
        "et",
        "fi",
        "fr",
        "hu",
        "it",
        "lt",
        "lv",
        "nb",
        "nl",
        "pl",
        "pt-BR",
        "ro",
        "sv",
    }


def test_supported_translations_match_strings_json() -> None:
    root = Path(__file__).resolve().parents[3]
    component_dir = root / "custom_components" / DOMAIN
    source = json.loads((component_dir / "strings.json").read_text(encoding="utf-8"))
    source_paths = _leaf_paths(source)
    source_values = _leaf_values(source)

    translation_files = sorted((component_dir / "translations").glob("*.json"))

    assert translation_files
    for translation_file in translation_files:
        translated = json.loads(translation_file.read_text(encoding="utf-8"))
        translated_paths = _leaf_paths(translated)
        translated_values = _leaf_values(translated)

        assert translated_paths == source_paths, translation_file.name
        for path, source_value in source_values.items():
            assert _placeholders(translated_values[path]) == _placeholders(
                source_value
            ), f"{translation_file.name}:{path}"
