from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "scripts" / "issue_labeling.py"
    spec = importlib.util.spec_from_file_location("issue_labeling", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


issue_labeling = _load_module()


def _issue(title: str, body: str, labels: list[str] | None = None) -> dict[str, object]:
    return {
        "number": 42,
        "title": title,
        "body": body,
        "labels": [{"name": label} for label in (labels or [])],
    }


def test_bug_issue_uses_parser_area_install_method_and_diagnostics() -> None:
    decision = issue_labeling.desired_labels_for_issue(
        _issue(
            "[Bug]: Voice fails",
            "### Primary affected area\n\nOther / unsure\n",
            labels=["bug", "status/needs-triage"],
        ),
        action="opened",
        parser_outputs={
            "primary_area": "Text-to-speech or voice selection",
            "install_method": "HACS",
            "diagnostics_status": "Not yet, but I can upload them",
        },
    )

    assert decision.desired_labels == {
        "area/tts",
        "source/hacs",
        "status/needs-diagnostics",
        "status/needs-triage",
    }
    assert decision.to_add == [
        "area/tts",
        "source/hacs",
        "status/needs-diagnostics",
    ]


def test_feature_issue_falls_back_to_raw_body_when_parser_area_missing() -> None:
    body = """
### Area

Documentation

### Problem statement

Current docs are incomplete.
""".strip()

    decision = issue_labeling.desired_labels_for_issue(
        _issue("[Feature]: Improve docs", body, labels=["enhancement"]),
        action="edited",
        parser_outputs={"area": ""},
    )

    assert decision.desired_labels == {"area/docs", "status/needs-triage"}
    assert decision.to_add == ["area/docs", "status/needs-triage"]


def test_managed_labels_are_removed_when_no_longer_desired() -> None:
    decision = issue_labeling.desired_labels_for_issue(
        _issue(
            "[Feature]: Improve docs",
            "### Area\n\nDocumentation",
            labels=["status/needs-triage", "area/text-generation", "source/hacs"],
        ),
        action="edited",
        parser_outputs={},
    )

    assert decision.to_add == ["area/docs"]
    assert decision.to_remove == ["area/text-generation", "source/hacs"]


def test_validate_template_mappings_for_repository() -> None:
    root = Path(__file__).resolve().parents[2]
    assert issue_labeling.validate_template_mappings(root) == []
