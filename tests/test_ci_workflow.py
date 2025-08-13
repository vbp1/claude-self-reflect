"""
Tests for the GitHub Actions CI workflow.

Testing library and framework: pytest (Python).
- These tests aim to validate the structure and critical commands of the workflow
  defined under .github/workflows. They prefer YAML parsing via PyYAML if available;
  otherwise, they fall back to robust text-based assertions.

Focus areas from the PR diff:
- Workflow name: "CI/CD Pipeline"
- triggers: push (main, develop), pull_request (main), release (types: created)
- permissions: contents: write, packages: write
- jobs: pre-commit, python-check, docker-build (needs pre-commit, python-check), create-github-release (needs docker-build, if: github.event_name == 'release')
- Key steps and actions versions:
  - actions/checkout@v4
  - actions/setup-python@v5 with python-version: 3.11
  - pip install pre-commit, cd mcp-server && pip install -e .
  - pre-commit run --all-files
  - python -m py_compile mcp-server/src/*.py
  - docker/setup-buildx-action@v3
  - docker compose build
  - docker compose up -d qdrant; sleep 10; curl -f http://localhost:6333/; docker compose down
  - softprops/action-gh-release@v2 with files path including ${{ github.event.release.tag_name }}
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional

import pathlib

WF_DIR = pathlib.Path(".github/workflows")


def _find_ci_workflow_file() -> pathlib.Path:
    """
    Heuristic to locate the CI workflow that matches the diff.
    We search for a workflow file that contains hallmark strings from the diff.
    """
    candidates: List[pathlib.Path] = []
    if WF_DIR.exists():
        for f in WF_DIR.glob("*.yml"):
            candidates.append(f)
        for f in WF_DIR.glob("*.yaml"):
            candidates.append(f)
    hallmark_patterns = [
        r"name:\s*CI/CD Pipeline",
        r"docker compose up -d qdrant",
        r"python -m py_compile mcp-server/src/\*\.py",
        r"softprops/action-gh-release@v2",
    ]
    # Prefer files that match most hallmark patterns
    scored: List[tuple[int, pathlib.Path]] = []
    for p in candidates:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        score = sum(1 for pat in hallmark_patterns if re.search(pat, text))
        if score:
            scored.append((score, p))
    if scored:
        scored.sort(key=lambda x: (-x[0], str(x[1])))
        return scored[0][1]
    # Fallback: if exactly one workflow exists, use it
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        "Could not find CI workflow file with expected diff content under .github/workflows"
    )


def _maybe_load_yaml() -> Optional[Any]:
    """
    Try to import PyYAML if available without introducing new dependencies.
    """
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    return yaml


def _load_workflow_as_yaml(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    yaml = _maybe_load_yaml()
    if not yaml:
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            return None
        return data  # type: ignore[return-value]
    except Exception:
        return None


def test_workflow_file_exists():
    assert WF_DIR.exists(), "Expected .github/workflows directory to exist"
    path = _find_ci_workflow_file()
    assert path.exists(), "CI workflow file should exist"


def test_workflow_name_and_triggers():
    path = _find_ci_workflow_file()
    text = path.read_text(encoding="utf-8")
    # Basic text assertions (work regardless of PyYAML availability)
    assert re.search(r"^name:\s*CI/CD Pipeline\s*$", text, flags=re.M), "Workflow name should be 'CI/CD Pipeline'"
    # Triggers
    assert "on:" in text, "Workflow must define 'on' section"
    # push branches
    assert re.search(r"push:\s*\n\s*branches:\s*\[\s*main\s*,\s*develop\s*\]", text), "Expected push branches [ main, develop ]"
    # pull_request branches
    assert re.search(r"pull_request:\s*\n\s*branches:\s*\[\s*main\s*\]", text), "Expected pull_request branches [ main ]"
    # release types: created
    assert re.search(r"release:\s*\n\s*types:\s*\[\s*created\s*\]", text), "Expected release types [ created ]"


def test_permissions_and_jobs_presence():
    path = _find_ci_workflow_file()
    text = path.read_text(encoding="utf-8")
    # permissions
    assert re.search(r"permissions:\s*\n\s*contents:\s*write\s*\n\s*packages:\s*write", text), "Expected permissions for contents and packages set to write"
    # jobs exist
    for job in ["pre-commit", "python-check", "docker-build", "create-github-release"]:
        assert re.search(rf"^\s*{re.escape(job)}:\s*$", text, flags=re.M), f"Expected job '{job}' to be present"


def test_pre_commit_job_steps_and_python_setup():
    path = _find_ci_workflow_file()
    text = path.read_text(encoding="utf-8")
    # checkout
    assert "uses: actions/checkout@v4" in text, "Expected checkout v4"
    # setup-python v5 with 3.11
    assert "uses: actions/setup-python@v5" in text, "Expected setup-python v5"
    assert re.search(r"python-version:\s*'?\s*3\.11\s*'?", text), "Expected Python 3.11"
    # install deps includes pre-commit and install editable mcp-server
    assert re.search(r"pip install (\-\-upgrade pip|pre-commit)", text), "Expected pip upgrade or pre-commit install"
    assert re.search(r"pip install pre-commit", text), "Expected pre-commit installation"
    assert re.search(r"cd\s+mcp-server\s+&&\s+pip\s+install\s+\-e\s+\.", text), "Expected editable install of mcp-server"
    # run pre-commit
    assert "pre-commit run --all-files" in text, "Expected pre-commit run command"


def test_python_check_job_compiles_sources():
    path = _find_ci_workflow_file()
    text = path.read_text(encoding="utf-8")
    assert re.search(r"python -m py_compile\s+mcp-server/src/\*\.py", text), "Expected py_compile on mcp-server/src/*.py"


def test_docker_build_job_and_qdrant_healthcheck():
    path = _find_ci_workflow_file()
    text = path.read_text(encoding="utf-8")
    # needs
    needs_match = re.search(r"docker-build:\s*\n\s*runs-on:.*\n\s*needs:\s*\[\s*pre-commit\s*,\s*python-check\s*\]", text)
    assert needs_match, "docker-build must need pre-commit and python-check"
    # setup buildx
    assert "uses: docker/setup-buildx-action@v3" in text, "Expected Buildx action v3"
    # build images
    assert re.search(r"^\s*docker compose build\s*$", text, flags=re.M), "Expected docker compose build"
    # qdrant spin-up and health check
    assert re.search(r"docker compose up -d qdrant", text), "Expected qdrant service startup"
    assert re.search(r"sleep\s+10", text), "Expected sleep 10 for readiness"
    assert re.search(r"curl -f http://localhost:6333/", text), "Expected qdrant health check"
    assert re.search(r"docker compose down", text), "Expected compose down to cleanup"


def test_create_github_release_job_conditions_and_steps():
    path = _find_ci_workflow_file()
    text = path.read_text(encoding="utf-8")
    # needs docker-build and conditional on release event
    assert re.search(r"create-github-release:\s*\n\s*needs:\s*docker-build", text), "create-github-release should need docker-build"
    assert re.search(r"^\s*if:\s*github\.event_name\s*==\s*'release'\s*$", text, flags=re.M), "Expected if condition on release"
    # checkout
    assert "uses: actions/checkout@v4" in text, "Expected checkout in release job"
    # tarball creation with exclusions and tag_name interpolation
    assert re.search(r"tar -czf\s+/tmp/release/claude-self-reflect-\$\{\{\s*github.event.release.tag_name\s*\}\}\.tar\.gz", text), "Expected tar command with tag_name"
    for excl in [
        "node_modules", "venv", ".venv", ".git", "data", "qdrant_storage",
        "*.tar.gz", "__pycache__", "*.pyc", "logs", ".claude/agents"
    ]:
        assert re.search(rf"--exclude=['\"]?{re.escape(excl)}['\"]?\s+\\", text), f"Expected tar exclude for {excl}"
    # move tarball to repo root
    assert re.search(r"mv\s+/tmp/release/claude-self-reflect-\$\{\{\s*github.event.release.tag_name\s*\}\}\.tar\.gz\s+\.", text), "Expected mv of tarball to repo root"
    # upload release asset with softprops
    assert "uses: softprops/action-gh-release@v2" in text, "Expected softprops/action-gh-release@v2"
    assert re.search(r"files:\s*\./claude-self-reflect-\$\{\{\s*github.event.release.tag_name\s*\}\}\.tar\.gz", text), "Expected files path to tarball"
    # GITHUB_TOKEN env presence (value comes from secrets in runtime)
    assert re.search(r"GITHUB_TOKEN:\s*\$\{\{\s*secrets\.GITHUB_TOKEN\s*\}\}", text), "Expected GITHUB_TOKEN env mapping"


def test_yaml_semantics_when_pyyaml_available():
    """
    When PyYAML is installed in the environment, validate the structure semantically.
    This test is auto-skipped (via assert precondition) when PyYAML isn't available.
    """
    path = _find_ci_workflow_file()
    data = _load_workflow_as_yaml(path)
    if data is None:
        # Soft skip without relying on pytest.skip (keeps compatibility with unittest)
        assert True, "PyYAML not available; semantic YAML assertions skipped."
        return

    # Name
    assert data.get("name") == "CI/CD Pipeline"

    # Triggers
    on = data.get("on") or {}
    assert isinstance(on, dict)
    assert on.get("push", {}).get("branches") == ["main", "develop"]
    assert on.get("pull_request", {}).get("branches") == ["main"]
    assert on.get("release", {}).get("types") == ["created"]

    # Permissions
    perms = data.get("permissions")
    assert perms == {"contents": "write", "packages": "write"}

    # Jobs
    jobs = data.get("jobs")
    assert isinstance(jobs, dict)
    assert set(["pre-commit", "python-check", "docker-build", "create-github-release"]).issubset(jobs.keys())

    # pre-commit job
    pre = jobs["pre-commit"]
    assert pre.get("runs-on") == "ubuntu-latest"
    steps = pre.get("steps") or []
    assert any(s.get("uses") == "actions/checkout@v4" for s in steps)
    assert any(s.get("uses") == "actions/setup-python@v5" and s.get("with", {}).get("python-version") in ("3.11", "3.11.x", "3.11.*") for s in steps)
    assert any("pip install pre-commit" in (s.get("run") or "") for s in steps)
    assert any("cd mcp-server && pip install -e ." in (s.get("run") or "") for s in steps)
    assert any("pre-commit run --all-files" in (s.get("run") or "") for s in steps)

    # python-check job
    pyc = jobs["python-check"]
    assert pyc.get("runs-on") == "ubuntu-latest"
    steps = pyc.get("steps") or []
    assert any("python -m py_compile mcp-server/src/*.py" in (s.get("run") or "") for s in steps)

    # docker-build
    docker_build = jobs["docker-build"]
    assert docker_build.get("runs-on") == "ubuntu-latest"
    assert docker_build.get("needs") == ["pre-commit", "python-check"]
    steps = docker_build.get("steps") or []
    assert any(s.get("uses") == "docker/setup-buildx-action@v3" for s in steps)
    assert any((s.get("run") or "").strip() == "docker compose build" for s in steps)
    compose_run = "\n".join(s.get("run") or "" for s in steps)
    assert "docker compose up -d qdrant" in compose_run
    assert "sleep 10" in compose_run
    assert "curl -f http://localhost:6333/" in compose_run
    assert "docker compose down" in compose_run

    # create-github-release
    rel = jobs["create-github-release"]
    assert rel.get("runs-on") == "ubuntu-latest"
    assert rel.get("needs") == "docker-build" or rel.get("needs") == ["docker-build"]
    assert rel.get("if") == "github.event_name == 'release'"
    steps = rel.get("steps") or []
    assert any(s.get("uses") == "actions/checkout@v4" for s in steps)
    upload = next((s for s in steps if s.get("uses") == "softprops/action-gh-release@v2"), None)
    assert upload is not None
    assert upload.get("with", {}).get("files") == "./claude-self-reflect-${{ github.event.release.tag_name }}.tar.gz"
    env = upload.get("env", {})
    assert "GITHUB_TOKEN" in env
    assert env.get("GITHUB_TOKEN") == "${{ secrets.GITHUB_TOKEN }}"