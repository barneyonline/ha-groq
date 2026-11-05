"""Map isolated reports to source metadata and verify Codecov processing.

Only the default-branch workflow's copy runs in the credentialed reporter. The
producer tree supplies names and modes, never file contents or executable code.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any

MAX_RESPONSE = 16 * 1024 * 1024


def repository_path(value: str) -> str:
    """Accept a canonical relative path without control or Git metadata names."""
    path = PurePosixPath(value)
    if (
        not value
        or len(value) > 1024
        or len(path.parts) > 32
        or path.is_absolute()
        or path.as_posix() != value
        or value == "."
        or any(part in {"..", ".git"} for part in path.parts)
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
        or "\\" in value
    ):
        raise ValueError("Unsafe repository path")
    return value


def report_path(value: str) -> str:
    """Remove only the known Docker workspace, never guess a basename match."""
    return repository_path(value.removeprefix("/workspace/"))


def read_xml(path: Path) -> ET.Element:
    data = path.read_bytes()
    if b"\x00" in data or b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("Report XML declarations are not allowed")
    return ET.fromstring(data)


def write_xml(path: Path, xml: ET.Element) -> None:
    data = ET.tostring(xml, encoding="utf-8", xml_declaration=True)
    if len(data) > MAX_RESPONSE:
        raise ValueError("Report file budget exceeded")
    path.write_bytes(data)


def prepare(root: Path, sha: str) -> None:
    """Materialize inert files for the CLI's network inventory, outside Git."""
    tree = json.loads((root / "producer-tree.json").read_text())
    if tree["sha"] != sha or not 0 < len(tree["tree"]) <= 10000:
        raise ValueError("Unexpected producer inventory")
    files: set[str] = set()
    for entry in tree["tree"]:
        path = repository_path(entry["path"])
        # Never materialize symlinks or submodules, even when named like source.
        if entry["type"] == "blob" and entry["mode"] in {"100644", "100755"}:
            if path in files:
                raise ValueError("Duplicate source path")
            files.add(path)
    (root / "codecov-work").mkdir(mode=0o700)
    source = root / "codecov-source"
    source.mkdir(mode=0o700)
    for name in sorted(files):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600, exist_ok=False)

    coverage_path = root / "reports/coverage.xml"
    coverage = read_xml(coverage_path)
    if coverage.tag != "coverage":
        raise ValueError("Expected Cobertura coverage")
    expected: dict[str, dict[str, int]] = {}
    for item in coverage.findall("./packages/package/classes/class"):
        name = report_path(item.attrib["filename"])
        if name not in files or not name.startswith("custom_components/groq/"):
            raise ValueError("Coverage path is absent from producer source")
        item.set("filename", name)
        lines: dict[int, bool] = {}
        for line in item.findall("./lines/line"):
            number, hits = int(line.attrib["number"]), int(line.attrib["hits"])
            if number <= 0 or hits < 0 or number in lines:
                raise ValueError("Invalid coverage line")
            lines[number] = hits > 0
        if name in expected:
            raise ValueError("Duplicate coverage file")
        if lines:
            expected[name] = {"lines": len(lines), "hits": sum(lines.values())}
    if not expected:
        raise ValueError("Empty coverage report")
    sources = coverage.find("sources")
    if sources is not None:
        coverage.remove(sources)
    # Filenames now identify exact repository paths; no absolute source hints.
    sources = ET.Element("sources")
    ET.SubElement(sources, "source").text = "."
    coverage.insert(0, sources)
    write_xml(coverage_path, coverage)

    junit_path = root / "reports/junit.xml"
    junit = read_xml(junit_path)
    cases = list(junit.iter("testcase"))
    if junit.tag not in {"testsuites", "testsuite"} or not cases:
        raise ValueError("Empty test results")
    for case in cases:
        if "file" in case.attrib:
            name = report_path(case.attrib["file"])
            if name not in files:
                raise ValueError("Test path is absent from producer source")
            case.set("file", name)
    write_xml(junit_path, junit)
    if sum(path.stat().st_size for path in (root / "reports").iterdir()) > 33554432:
        raise ValueError("Report aggregate budget exceeded")
    (root / "codecov-expected.json").write_text(
        json.dumps({"sha": sha, "files": expected})
    )
    print(f"Prepared {len(files)} source paths and {len(expected)} covered files")


def read_json(url: str) -> dict[str, Any]:
    """Read the anonymous public API without forwarding reporting credentials."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=15) as response:
        data = response.read(MAX_RESPONSE + 1)
    if len(data) > MAX_RESPONSE:
        raise ValueError("Codecov response budget exceeded")
    return json.loads(data)


def uploads_for_commit(base: str) -> list[dict[str, Any]]:
    url = base + "uploads/?format=json&page_size=100"
    uploads = []
    for _ in range(10):
        page = read_json(url)
        uploads.extend(page["results"])
        url = page.get("next")
        if not url:
            return uploads
        if not url.startswith(base + "uploads/?"):
            raise ValueError("Unexpected Codecov pagination URL")
    raise ValueError("Codecov upload pagination budget exceeded")


def processed(
    commit: dict[str, Any],
    uploads: list[dict[str, Any]],
    expected: dict[str, Any],
    run: str,
    attempt: str,
) -> bool:
    """Require real project coverage plus independently processed JUnit."""
    if commit.get("commitid") != expected["sha"]:
        raise ValueError("Codecov returned a different commit")
    selected = {}
    # Test-result sessions store job_code, with name/build_code/totals null.
    # Coverage sessions use name. Both are scoped by the exact commit endpoint.
    for kind, field, value, state in (
        ("coverage", "name", f"coverage-{run}-{attempt}", "merged"),
        ("junit", "job_code", f"junit-{run}-{attempt}", "processed"),
    ):
        matches = [item for item in uploads if item.get(field) == value]
        if any(item.get("state") in {"error", "failed"} for item in matches):
            raise ValueError(f"Codecov rejected this attempt's {kind} report")
        ready = [item for item in matches if item.get("state") == state]
        if not ready:
            return False
        selected[kind] = ready[-1]
    junit = selected["junit"]
    if junit.get("job_code") != f"junit-{run}-{attempt}" or not str(
        junit.get("storage_path", "")
    ).startswith("test_results/v1/raw/"):
        raise ValueError("Processed JUnit attribution is missing")
    if commit.get("state") == "error":
        raise ValueError("Codecov project report is unusable")
    if commit.get("state") != "complete":
        return False
    totals = commit.get("totals") or {}
    upload_totals = selected["coverage"].get("totals") or {}
    for value in (totals, upload_totals):
        if value.get("files", 0) <= 0 or value.get("lines", 0) <= 0:
            raise ValueError("Codecov processed empty coverage")
    reported = {
        item["name"]: item["totals"]
        for item in (commit.get("report") or {}).get("files", [])
    }
    for name, counts in expected["files"].items():
        actual = reported.get(name) or {}
        if any(actual.get(key) != count for key, count in counts.items()):
            raise ValueError("Codecov project coverage differs from producer report")
    if upload_totals["lines"] != sum(
        item["lines"] for item in expected["files"].values()
    ) or upload_totals.get("hits") != sum(
        item["hits"] for item in expected["files"].values()
    ):
        raise ValueError("Codecov attempt coverage differs from producer report")
    return True


def verify(root: Path, sha: str, repo: str, run: str, attempt: str) -> None:
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)
        or not run.isdecimal()
        or not attempt.isdecimal()
    ):
        raise ValueError("Invalid reporting attribution")
    expected = json.loads((root / "codecov-expected.json").read_text())
    if expected["sha"] != sha:
        raise ValueError("Unexpected expected coverage commit")
    owner, name = repo.split("/")
    base = f"https://api.codecov.io/api/v2/github/{owner}/repos/{name}/commits/{sha}/"
    deadline = time.monotonic() + 180
    retries = 0
    for poll in range(12):
        if time.monotonic() >= deadline:
            break
        try:
            commit = read_json(base + "?format=json")
            uploads = uploads_for_commit(base)
            if processed(commit, uploads, expected, run, attempt):
                print(
                    f"Verified processed project coverage and JUnit for {sha}, producer {run}.{attempt}"
                )
                return
        except urllib.error.HTTPError as error:
            error.close()
            if error.code != 404:
                if error.code not in {429, 500, 502, 503, 504} or retries >= 2:
                    raise
                retries += 1
        if poll < 11:
            time.sleep(10)
    raise ValueError(
        "Timed out waiting for this attempt's processed coverage and JUnit"
    )


def main() -> None:
    sha = os.environ["SOURCE_SHA"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Invalid source commit")
    root = Path(os.environ["RUNNER_TEMP"])
    if sys.argv[1:] == ["prepare"]:
        prepare(root, sha)
    elif sys.argv[1:] == ["verify"]:
        verify(
            root,
            sha,
            os.environ["GITHUB_REPOSITORY"],
            os.environ["SOURCE_RUN_ID"],
            os.environ["SOURCE_RUN_ATTEMPT"],
        )
    else:
        raise ValueError("Expected prepare or verify")


if __name__ == "__main__":
    main()
