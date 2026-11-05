"""Regression checks for Codecov's unusable source inventory and empty reports."""

from __future__ import annotations

import copy
import json
import stat
import urllib.error
from pathlib import Path
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import pytest
import yaml

from scripts import codecov_reporting as reporter

SHA = "a" * 40
SOURCE = "custom_components/groq/api.py"
TEST = "tests/components/groq/test_api.py"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def reports(tmp_path):
    tree = [
        {"path": path, "type": "blob", "mode": "100644"}
        for path in (SOURCE, TEST, "scripts/malicious.py", "codecov.yml")
    ]
    tree.extend(
        [
            {
                "path": "custom_components/groq/link.py",
                "type": "blob",
                "mode": "120000",
            },
            {"path": "submodule", "type": "commit", "mode": "160000"},
        ]
    )
    (tmp_path / "producer-tree.json").write_text(json.dumps({"sha": SHA, "tree": tree}))
    directory = tmp_path / "reports"
    directory.mkdir()
    (directory / "coverage.xml").write_text(
        f"""<coverage lines-valid="2" lines-covered="1">
      <sources><source>/workspace</source></sources>
      <packages><package><classes><class filename="{SOURCE}"><lines>
        <line number="3" hits="7"/><line number="9" hits="0"/>
      </lines></class></classes></package></packages></coverage>"""
    )
    (directory / "junit.xml").write_text(f"""<testsuites><testsuite tests="1">
      <testcase name="test_api" file="/workspace/{TEST}"/>
    </testsuite></testsuites>""")
    return tmp_path


def test_inventory_and_exact_paths_preserve_covered_and_uncovered_lines(reports):
    # The old root contained only XML; no source filename could match its network.
    assert SOURCE not in {p.name for p in (reports / "reports").iterdir()}
    reporter.prepare(reports, SHA)
    # The action downloads these tools into its CWD. Keep them out of root_dir.
    work = reports / "codecov-work"
    assert list(work.iterdir()) == []
    for name in ("codecov", "codecov.SHA256SUM", "codecov.SHA256SUM.sig"):
        (work / name).write_text("wrapper tool")
    inventory = reports / "codecov-source"
    names = {
        p.relative_to(inventory).as_posix() for p in inventory.rglob("*") if p.is_file()
    }
    assert names == {SOURCE, TEST, "scripts/malicious.py", "codecov.yml"}
    for path in inventory.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == b""
            assert not path.stat().st_mode & stat.S_IXUSR
    coverage = ET.parse(reports / "reports/coverage.xml")
    assert coverage.findtext("sources/source") == "."
    item = coverage.find("./packages/package/classes/class")
    assert item.attrib["filename"] == SOURCE
    assert [
        (line.attrib["number"], line.attrib["hits"])
        for line in item.findall("lines/line")
    ] == [("3", "7"), ("9", "0")]
    assert (
        ET.parse(reports / "reports/junit.xml").find(".//testcase").attrib["file"]
        == TEST
    )
    assert json.loads((reports / "codecov-expected.json").read_text()) == {
        "sha": SHA,
        "files": {SOURCE: {"lines": 2, "hits": 1}},
    }


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "/absolute",
        "a/../../escape",
        "a//b",
        "./a",
        ".",
        "a\\b",
        "a\nb",
        ".git/config",
    ],
)
def test_unsafe_inventory_paths_fail_closed(reports, path):
    tree = {"sha": SHA, "tree": [{"path": path, "type": "blob", "mode": "100644"}]}
    (reports / "producer-tree.json").write_text(json.dumps(tree))
    with pytest.raises(ValueError, match="Unsafe repository path"):
        reporter.prepare(reports, SHA)


@pytest.mark.parametrize(
    "filename",
    [
        "api.py",
        "/unknown/custom_components/groq/api.py",
        "custom_components/groq/link.py",
        "custom_components/groq/missing.py",
    ],
)
def test_ambiguous_untracked_or_symlink_coverage_is_rejected(reports, filename):
    path = reports / "reports/coverage.xml"
    path.write_text(path.read_text().replace(SOURCE, filename))
    with pytest.raises(ValueError):
        reporter.prepare(reports, SHA)


def test_absolute_workspace_paths_are_normalized(reports):
    path = reports / "reports/coverage.xml"
    path.write_text(path.read_text().replace(SOURCE, "/workspace/" + SOURCE))
    reporter.prepare(reports, SHA)
    assert ET.parse(path).find(".//class").attrib["filename"] == SOURCE


@pytest.mark.parametrize(
    "file,content",
    [
        ("coverage.xml", "<coverage><packages/></coverage>"),
        ("junit.xml", "<testsuites/>"),
        ("coverage.xml", '<!DOCTYPE coverage [<!ENTITY data "bad">]><coverage/>'),
    ],
)
def test_empty_or_entity_reports_are_rejected(reports, file, content):
    (reports / "reports" / file).write_text(content)
    with pytest.raises(ValueError):
        reporter.prepare(reports, SHA)


@pytest.fixture
def delivery():
    expected = {"sha": SHA, "files": {SOURCE: {"lines": 2, "hits": 1}}}
    totals = {"files": 1, "lines": 2, "hits": 1, "misses": 1}
    commit = {
        "commitid": SHA,
        "state": "complete",
        "totals": totals,
        "report": {"files": [{"name": SOURCE, "totals": {**totals, "files": 0}}]},
    }
    uploads = [
        {
            "name": "coverage-123-2",
            "build_code": None,
            "state": "merged",
            "state_id": 6,
            "state_name": "MERGED",
            "totals": totals,
        },
        {
            "name": None,
            "job_code": "junit-123-2",
            "build_code": None,
            "totals": None,
            "state": "processed",
            "state_name": "UPLOADED",
            "state_id": 1,
            "build_url": None,
            "storage_path": "test_results/v1/raw/file",
        },
    ]
    return commit, uploads, expected


def check(delivery):
    return reporter.processed(*delivery, "123", "2")


def test_processed_project_and_separately_processed_junit_pass(delivery):
    assert check(delivery)


@pytest.mark.parametrize("state", ["created", "uploaded", "processing", "processed"])
def test_transport_or_queued_upload_does_not_pass(delivery, state):
    delivery[1][0]["state"] = state
    assert not check(delivery)


@pytest.mark.parametrize("index", [0, 1])
def test_exact_attempt_processing_error_is_visible(delivery, index):
    delivery[1][index]["state"] = "error"
    with pytest.raises(ValueError, match="rejected this attempt"):
        check(delivery)


def test_previous_attempt_does_not_satisfy_current_delivery(delivery):
    delivery[1][0]["name"] = "coverage-123-1"
    assert not check(delivery)


@pytest.mark.parametrize(
    "case",
    [
        "empty_commit",
        "empty_upload",
        "empty_files",
        "wrong_path",
        "wrong_hits",
        "wrong_commit",
        "unusable",
        "wrong_junit",
    ],
)
def test_noop_patch_and_empty_or_unusable_reports_cannot_pass(delivery, case):
    commit, uploads, expected = copy.deepcopy(delivery)
    if case == "empty_commit":
        commit["totals"] = {"files": 0, "lines": 0}
    elif case == "empty_upload":
        uploads[0]["totals"] = {"files": 0, "lines": 0}
    elif case == "empty_files":
        commit["report"]["files"] = []
    elif case == "wrong_path":
        commit["report"]["files"][0]["name"] = "api.py"
    elif case == "wrong_hits":
        commit["report"]["files"][0]["totals"]["hits"] = 2
    elif case == "wrong_commit":
        commit["commitid"] = "b" * 40
    elif case == "unusable":
        commit["state"] = "error"
    else:
        uploads[1]["storage_path"] = "v4/raw/coverage"
    with pytest.raises(ValueError):
        check((commit, uploads, expected))


def test_pagination_is_complete_and_rejects_foreign_urls(monkeypatch):
    base = "https://api.codecov.io/api/v2/github/owner/repos/repo/commits/sha/"
    read = Mock(
        side_effect=[
            {"results": [1], "next": base + "uploads/?page=2"},
            {"results": [2], "next": None},
        ]
    )
    monkeypatch.setattr(reporter, "read_json", read)
    assert reporter.uploads_for_commit(base) == [1, 2]
    read.side_effect = [{"results": [], "next": "https://other.invalid/"}]
    with pytest.raises(ValueError, match="pagination URL"):
        reporter.uploads_for_commit(base)


def test_verification_polls_boundedly_and_preserves_exact_attribution(
    reports, delivery, monkeypatch
):
    reporter.prepare(reports, SHA)
    commit, uploads, _ = delivery
    read = Mock(
        side_effect=[{"commitid": SHA}, {"results": []}, commit, {"results": uploads}]
    )
    monkeypatch.setattr(reporter, "read_json", read)
    sleep = Mock()
    monkeypatch.setattr(reporter.time, "sleep", sleep)
    reporter.verify(reports, SHA, "barneyonline/ha-groq", "123", "2")
    sleep.assert_called_once_with(10)
    assert all(SHA in call.args[0] for call in read.call_args_list)
    read.side_effect = None
    read.return_value = {"commitid": SHA, "results": []}
    with pytest.raises(ValueError, match="Timed out"):
        reporter.verify(reports, SHA, "barneyonline/ha-groq", "123", "2")
    assert read.call_count == 28


def test_infrastructure_retries_are_limited_to_two(reports, monkeypatch):
    reporter.prepare(reports, SHA)
    read = Mock(side_effect=urllib.error.HTTPError("url", 503, "Unavailable", {}, None))
    monkeypatch.setattr(reporter, "read_json", read)
    monkeypatch.setattr(reporter.time, "sleep", Mock())
    with pytest.raises(urllib.error.HTTPError):
        reporter.verify(reports, SHA, "barneyonline/ha-groq", "123", "2")
    assert read.call_count == 3


def test_workflow_preserves_trust_boundary_and_checks_processing():
    workflow = yaml.load(
        (ROOT / ".github/workflows/coverage-reporting.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["upload"]
    steps = job["steps"]
    checkout = next(step for step in steps if "checkout@" in step.get("uses", ""))
    assert checkout["with"]["ref"] == "${{ github.workflow_sha }}"
    assert checkout["with"]["persist-credentials"] == "false"
    metadata = next(
        step
        for step in steps
        if step["name"] == "Read the exact producer source inventory"
    )
    assert "commit_sha: sha" in metadata["with"]["script"]
    assert "tree.truncated" in metadata["with"]["script"]
    for step in steps:
        if "codecov-action@" in step.get("uses", ""):
            options = step["with"]
            assert options["root_dir"] == "${{ runner.temp }}/codecov-source"
            assert options["working-directory"] == "${{ runner.temp }}/codecov-work"
            assert options["plugins"] == "noop"
            assert options["disable_file_fixes"] == "true"
            assert options["use_oidc"] == "true"
    assert steps[-1]["run"].endswith("codecov_reporting.py verify")
    assert "coverage_upload.outcome == 'success'" in steps[-1]["if"]
    assert "junit_upload.outcome == 'success'" in steps[-1]["if"]


def test_utf16_entity_xml_is_rejected_before_parsing(reports):
    (reports / "reports/coverage.xml").write_bytes(
        '<!DOCTYPE coverage [<!ENTITY data "bad">]><coverage/>'.encode("utf-16")
    )
    with pytest.raises(ValueError, match="declarations"):
        reporter.prepare(reports, SHA)
