"""Exercise the actual reporter workflow and coverage.py path configuration."""

from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 40
SOURCE = "custom_components/groq/api.py"


def workflow():
    return yaml.load(
        (ROOT / ".github/workflows/coverage-reporting.yml").read_text(),
        Loader=yaml.BaseLoader,
    )


def step(name):
    return next(
        item for item in workflow()["jobs"]["upload"]["steps"] if item["name"] == name
    )


def test_git_inventory_never_materializes_candidate_files(tmp_path):
    repo, inventory = tmp_path / "producer", tmp_path / "inventory"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True
        ).strip()

    git("init", "--initial-branch=base")
    git("config", "user.name", "Coverage regression")
    git("config", "user.email", "coverage@example.invalid")
    git("config", "uploadpack.allowFilter", "true")
    source = repo / SOURCE
    source.parent.mkdir(parents=True)
    source.write_text("raise RuntimeError('candidate code must never run')\n")
    (repo / "codecov").symlink_to(tmp_path / "outside")
    (repo / "codecov.yml").write_text("candidate configuration\n")
    git("add", ".")
    git("commit", "-m", "Add source")
    git("checkout", "-b", "pr-head")
    source.write_text("raise RuntimeError('changed candidate must never run')\n")
    git("commit", "-am", "Change source")
    head = git("rev-parse", "HEAD")
    git("checkout", "base")
    (repo / "base_only.py").write_text("# added after the PR branched\n")
    git("add", ".")
    git("commit", "-m", "Add base source")
    git("merge", "--no-ff", "pr-head", "-m", "Synthetic PR merge")
    assert "base_only.py" in git("ls-files").splitlines()
    script = step("Prepare the verified Git inventory without a source checkout")["run"]
    subprocess.run(
        ["bash", "-eu", "-c", script],
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "SOURCE_SHA": head,
            "SOURCE_URL": repo.as_uri(),
            "CODECOV_SOURCE": str(inventory),
        },
    )
    assert {path.name for path in inventory.iterdir()} == {".git"}
    work = tmp_path / "codecov-work"
    assert list(work.iterdir()) == []
    for name in ("codecov", "codecov.SHA256SUM", "codecov.SHA256SUM.sig"):
        (work / name).write_text("wrapper artifact")
    env = {
        **os.environ,
        "GIT_DIR": str(inventory / ".git"),
        "GIT_WORK_TREE": str(inventory),
    }
    actual = subprocess.check_output(
        ["git", "-C", str(inventory), "ls-files"], env=env, text=True
    ).splitlines()
    assert set(actual) == {SOURCE, "codecov", "codecov.yml"}
    assert not (tmp_path / "outside").exists()
    assert not (inventory / SOURCE).exists()
    assert subprocess.check_output(
        ["git", "-C", str(work), "rev-parse", "--show-toplevel"], env=env, text=True
    ).strip() == str(inventory)
    for item in workflow()["jobs"]["upload"]["steps"]:
        assert "checkout@" not in item.get("uses", "")
        if "codecov-action@" in item.get("uses", ""):
            options = item["with"]
            assert options["working-directory"] == "${{ runner.temp }}/codecov-work"
            assert options["root_dir"] == "${{ runner.temp }}/codecov-source"
            assert item["env"]["GIT_DIR"] == "${{ runner.temp }}/codecov-source/.git"
            assert item["env"]["GIT_WORK_TREE"] == options["root_dir"]
            assert options["plugins"] == "noop"
            assert options["disable_file_fixes"] == options["disable_search"] == "true"
            assert options["use_oidc"] == "true"
    tests = yaml.load(
        (ROOT / ".github/workflows/tests.yml").read_text(), Loader=yaml.BaseLoader
    )
    for name in ("checks", "minimum-homeassistant"):
        checkout = next(
            item
            for item in tests["jobs"][name]["steps"]
            if "checkout@" in item.get("uses", "")
        )
        assert (
            checkout["with"]["ref"]
            == "${{ github.event.pull_request.head.sha || github.sha }}"
        )
        assert "allow-unsafe-pr-checkout" not in checkout["with"]


def test_coverage_py_emits_repository_paths_and_preserves_uncovered_lines(tmp_path):
    source = tmp_path / SOURCE
    source.parent.mkdir(parents=True)
    (source.parent / "__init__.py").touch()
    source.write_text(
        "def covered():\n    return 1\n\ndef uncovered():\n    return 2\n"
    )
    (tmp_path / "exercise.py").write_text(
        "from custom_components.groq.api import covered\ncovered()\n"
    )
    env = {**os.environ, "COVERAGE_FILE": str(tmp_path / ".coverage")}
    for args in (
        ["run", f"--rcfile={ROOT / 'pyproject.toml'}", "exercise.py"],
        ["xml", f"--rcfile={ROOT / 'pyproject.toml'}", "-o", "coverage.xml"],
    ):
        subprocess.run(
            [sys.executable, "-m", "coverage", *args],
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
        )
    xml = ET.parse(tmp_path / "coverage.xml")
    classes = {item.attrib["filename"]: item for item in xml.findall(".//class")}
    assert SOURCE in classes
    hits = [int(line.attrib["hits"]) for line in classes[SOURCE].findall("lines/line")]
    assert 0 in hits and any(hit > 0 for hit in hits)
    assert all(not Path(name).is_absolute() for name in classes)
    assert all(
        not (item.text or "").startswith("/") for item in xml.findall("sources/source")
    )


@pytest.fixture
def delivery():
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
            "build_code": None,
            "build_url": None,
            "job_code": "junit-123-2",
            "state": "processed",
            "state_id": 1,
            "state_name": "UPLOADED",
            "totals": None,
            "storage_path": "test_results/v1/raw/example",
        },
    ]
    return commit, uploads


def verify(monkeypatch, delivery, pages=None):
    commit, uploads = delivery
    base = f"https://api.codecov.io/api/v2/github/barneyonline/repos/ha-groq/commits/{SHA}/"
    reads = []

    def read(url, timeout):
        assert timeout == 15
        reads.append(url)
        data = (
            commit
            if url == base + "?format=json"
            else (pages[url] if pages else {"results": uploads, "next": None})
        )
        return io.StringIO(json.dumps(data))

    monkeypatch.setattr(urllib.request, "urlopen", read)
    sleep = Mock()
    monkeypatch.setattr("time.sleep", sleep)
    for key, value in {
        "GITHUB_REPOSITORY": "barneyonline/ha-groq",
        "SOURCE_SHA": SHA,
        "SOURCE_RUN_ID": "123",
        "SOURCE_RUN_ATTEMPT": "2",
    }.items():
        monkeypatch.setenv(key, value)
    script = step("Verify processed coverage and test results for this attempt")["run"]
    code = script.split("<<'PYTHON'\n", 1)[1].rsplit("\nPYTHON", 1)[0]
    exec(compile(code, "coverage-reporting.yml", "exec"), {})
    return reads, sleep


def test_real_success_states_and_nullable_junit_fields_pass(monkeypatch, delivery):
    reads, sleep = verify(monkeypatch, delivery)
    assert len(reads) == 2
    sleep.assert_not_called()


@pytest.mark.parametrize(
    "case",
    [
        "unusable",
        "empty_project",
        "empty_upload",
        "noop_report",
        "queued",
        "stale_coverage",
        "stale_junit",
        "wrong_commit",
    ],
)
def test_empty_unusable_or_other_attempt_cannot_pass(monkeypatch, delivery, case):
    commit, uploads = copy.deepcopy(delivery)
    if case == "unusable":
        uploads[0]["state"] = "error"
    elif case == "empty_project":
        commit["totals"] = {"files": 0, "lines": 0}
    elif case == "empty_upload":
        uploads[0]["totals"] = None
    elif case == "noop_report":
        commit["report"]["files"] = []
    elif case == "queued":
        uploads[0]["state"] = "processed"
    elif case == "stale_coverage":
        uploads[0]["name"] = "coverage-123-1"
    elif case == "stale_junit":
        uploads[1]["job_code"] = "junit-123-1"
    else:
        commit["commitid"] = "b" * 40
    with pytest.raises(SystemExit):
        verify(monkeypatch, (commit, uploads))


def test_upload_pagination_finds_this_attempt(monkeypatch, delivery):
    base = f"https://api.codecov.io/api/v2/github/barneyonline/repos/ha-groq/commits/{SHA}/uploads/?"
    pages = {
        base + "format=json&page_size=100": {"results": [], "next": base + "page=2"},
        base + "page=2": {"results": delivery[1], "next": None},
    }
    reads, _ = verify(monkeypatch, delivery, pages)
    assert len(reads) == 3
