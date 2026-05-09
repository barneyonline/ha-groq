"""Issue labeling helpers and GitHub workflow entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from urllib import parse, request

TITLE_PREFIX_TO_KIND = {
    "[Bug]:": "bug",
    "[Feature]:": "feature",
}

NO_AREA_SELECTIONS = {"Other", "Other / unsure"}

AREA_LABEL_MAP = {
    "Setup, API key, reauthentication": "area/setup-auth",
    "Model or voice selection": "area/models-voices",
    "Speech generation API request": "area/api",
    "Audio normalization": "area/audio-processing",
    "Diagnostics": "area/diagnostics",
    "Documentation": "area/docs",
    "Developer workflow / repository tooling": "area/tooling",
}

NO_SOURCE_SELECTIONS = {"Other"}

SOURCE_LABEL_MAP = {
    "HACS": "source/hacs",
    "Manual install in custom_components": "source/manual-install",
    "Development checkout": "source/dev-checkout",
}

BUG_DIAGNOSTICS_NEEDS_LABEL = {
    "Not yet, but I can upload them",
    "I cannot capture diagnostics",
}

LABEL_DEFINITIONS = {
    "status/needs-triage": {
        "color": "FBCA04",
        "description": "New or reopened issue awaiting maintainer triage",
    },
    "status/needs-diagnostics": {
        "color": "D93F0B",
        "description": "More diagnostics or evidence are needed before triage can finish",
    },
    "area/setup-auth": {
        "color": "1D76DB",
        "description": "Setup, API key, MFA, or reauthentication",
    },
    "area/models-voices": {
        "color": "1D76DB",
        "description": "Groq model and voice selection",
    },
    "area/api": {
        "color": "1D76DB",
        "description": "Groq speech API request or response behavior",
    },
    "area/audio-processing": {
        "color": "1D76DB",
        "description": "ffmpeg or normalization behavior",
    },
    "area/diagnostics": {
        "color": "1D76DB",
        "description": "Diagnostics and troubleshooting output",
    },
    "area/docs": {
        "color": "1D76DB",
        "description": "Documentation and setup guidance",
    },
    "area/tooling": {
        "color": "1D76DB",
        "description": "Developer workflow or repository tooling",
    },
    "source/hacs": {
        "color": "5319E7",
        "description": "Installed via HACS",
    },
    "source/manual-install": {
        "color": "5319E7",
        "description": "Installed manually in custom_components",
    },
    "source/dev-checkout": {
        "color": "5319E7",
        "description": "Running from a development checkout",
    },
}

MANAGED_PREFIXES = ("area/", "source/")
MANAGED_LABELS = set(LABEL_DEFINITIONS) | {
    "status/needs-diagnostics",
    "status/needs-triage",
}

BODY_FIELD_LABELS = {
    "bug": {
        "area": "Primary affected area",
        "install_method": "Installation method",
        "diagnostics_status": "Diagnostics attached?",
    },
    "feature": {
        "area": "Area",
    },
}


@dataclass(frozen=True)
class LabelDecision:
    """Desired label state for one issue."""

    desired_labels: set[str]
    managed_labels: set[str]
    to_add: list[str]
    to_remove: list[str]


def normalize(value: object) -> str:
    """Return a clean single-line string."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def issue_kind_from_title(title: str) -> str | None:
    """Infer issue kind from the title prefix."""
    for prefix, kind in TITLE_PREFIX_TO_KIND.items():
        if title.startswith(prefix):
            return kind
    return None


def parse_issue_form_section(body: str | None, heading: str) -> str:
    """Extract the first logical line below a rendered issue form heading."""
    if not body:
        return ""
    pattern = rf"(?ms)^### {re.escape(heading)}\s*\n(.*?)(?=^### |\Z)"
    match = re.search(pattern, body)
    if not match:
        return ""
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line or line == "_No response_" or line.startswith("<!--"):
            continue
        return normalize(line.lstrip("- ").strip())
    return ""


def _mapped_area_label(value: str) -> str | None:
    normalized = normalize(value)
    if not normalized or normalized in NO_AREA_SELECTIONS:
        return None
    return AREA_LABEL_MAP.get(normalized)


def _mapped_source_label(value: str) -> str | None:
    normalized = normalize(value)
    if not normalized or normalized in NO_SOURCE_SELECTIONS:
        return None
    return SOURCE_LABEL_MAP.get(normalized)


def _resolved_field(parser_value: str, body_value: str, resolver: Any) -> str | None:
    parser_label = resolver(parser_value)
    if parser_label:
        return parser_label
    return resolver(body_value)


def desired_labels_for_issue(
    issue: dict[str, Any],
    *,
    action: str,
    parser_outputs: dict[str, str],
) -> LabelDecision:
    """Return the desired managed labels for one issue."""

    kind = issue_kind_from_title(str(issue.get("title") or ""))
    if kind is None:
        return LabelDecision(set(), set(), [], [])

    body = str(issue.get("body") or "")
    desired: set[str] = {"status/needs-triage"}
    fields = BODY_FIELD_LABELS[kind]

    if kind == "bug":
        area = _resolved_field(
            parser_outputs.get("primary_area", ""),
            parse_issue_form_section(body, fields["area"]),
            _mapped_area_label,
        )
        source = _resolved_field(
            parser_outputs.get("install_method", ""),
            parse_issue_form_section(body, fields["install_method"]),
            _mapped_source_label,
        )
        if area:
            desired.add(area)
        if source:
            desired.add(source)
        diagnostics_status = normalize(
            parser_outputs.get("diagnostics_status")
            or parse_issue_form_section(body, fields["diagnostics_status"])
        )
        if diagnostics_status in BUG_DIAGNOSTICS_NEEDS_LABEL:
            desired.add("status/needs-diagnostics")

    if kind == "feature":
        area = _resolved_field(
            parser_outputs.get("area", ""),
            parse_issue_form_section(body, fields["area"]),
            _mapped_area_label,
        )
        if area:
            desired.add(area)

    if action == "opened":
        desired.add("status/needs-triage")

    existing = {
        label.get("name") if isinstance(label, dict) else str(label)
        for label in issue.get("labels", [])
    }
    managed_existing = {
        label
        for label in existing
        if label in MANAGED_LABELS or label.startswith(MANAGED_PREFIXES)
    }
    managed = managed_existing | desired
    to_add = sorted(desired - existing)
    to_remove = sorted(managed_existing - desired)
    return LabelDecision(desired, managed, to_add, to_remove)


def _github_request(
    method: str,
    path: str,
    *,
    token: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    body = None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "groq-issue-labeling",
    }
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(
        f"https://api.github.com{path}", data=body, headers=headers, method=method
    )
    with request.urlopen(req, timeout=20) as response:
        if response.status == 204:
            return None
        return json.loads(response.read().decode("utf-8"))


def _template_dropdown_options(
    repo_root: Path, template_name: str, field_id: str
) -> list[str] | None:
    import yaml

    path = repo_root / ".github" / "ISSUE_TEMPLATE" / template_name
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for item in data.get("body") or []:
        if item.get("type") == "dropdown" and item.get("id") == field_id:
            return list((item.get("attributes") or {}).get("options") or [])
    return None


def validate_template_mappings(repo_root: Path) -> list[str]:
    """Validate issue-form dropdown options against label maps."""

    checks = [
        ("bug_report.yml", "primary_area", AREA_LABEL_MAP, NO_AREA_SELECTIONS),
        ("bug_report.yml", "install_method", SOURCE_LABEL_MAP, NO_SOURCE_SELECTIONS),
        ("feature_request.yml", "area", AREA_LABEL_MAP, NO_AREA_SELECTIONS),
    ]
    errors: list[str] = []
    for template, field_id, mapping, ignored in checks:
        options = _template_dropdown_options(repo_root, template, field_id)
        if options is None:
            errors.append(f"{template}:{field_id} dropdown not found")
            continue
        unmapped = [
            option
            for option in options
            if option not in mapping and option not in ignored
        ]
        if unmapped:
            errors.append(f"{template}:{field_id} has unmapped options: {unmapped}")
    return errors


def _parser_outputs_from_env() -> dict[str, str]:
    return {
        "primary_area": os.environ.get("PRIMARY_AREA", ""),
        "area": os.environ.get("FEATURE_AREA", ""),
        "install_method": os.environ.get("INSTALL_METHOD", ""),
        "diagnostics_status": os.environ.get("DIAGNOSTICS_STATUS", ""),
    }


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    repository = os.environ.get("GITHUB_REPOSITORY")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not token or not repository or not event_path:
        print("Missing GitHub Actions environment.", file=sys.stderr)
        return 2

    event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    issue = event.get("issue")
    if not isinstance(issue, dict):
        print("No issue payload to label.")
        return 0

    owner, repo = repository.split("/", 1)
    decision = desired_labels_for_issue(
        issue,
        action=str(event.get("action") or ""),
        parser_outputs=_parser_outputs_from_env(),
    )
    issue_number = int(issue["number"])
    if decision.to_add:
        _github_request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{issue_number}/labels",
            token=token,
            payload={"labels": decision.to_add},
        )
    for label in decision.to_remove:
        encoded = parse.quote(label, safe="")
        _github_request(
            "DELETE",
            f"/repos/{owner}/{repo}/issues/{issue_number}/labels/{encoded}",
            token=token,
        )
    print(
        json.dumps(
            {
                "desired": sorted(decision.desired_labels),
                "add": decision.to_add,
                "remove": decision.to_remove,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
