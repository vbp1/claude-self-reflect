import os
import re
import unittest

WORKFLOW_PATH = os.path.join(".github", "workflows", "ci.yml")

def _read_workflow_text():
    if not os.path.isfile(WORKFLOW_PATH):
        raise FileNotFoundError(f"Expected workflow file not found at {WORKFLOW_PATH}")
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        return f.read()

def _load_yaml_or_skip(text):
    """
    Attempt to parse YAML using PyYAML if available.
    If PyYAML is not installed, skip tests that require structured parsing
    but still run some textual sanity checks.
    """
    try:
        import yaml  # PyYAML
    except Exception:
        return None  # Signal to caller that YAML isn't available
    try:
        return yaml.safe_load(text)
    except Exception as e:
        raise AssertionError(f"Failed to parse {WORKFLOW_PATH} as YAML: {e}") from e

class TestGithubWorkflowStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = _read_workflow_text()
        cls.data = _load_yaml_or_skip(cls.text)

    def test_workflow_file_exists_and_nonempty(self):
        self.assertTrue(os.path.exists(WORKFLOW_PATH), "Workflow file should exist")
        self.assertGreater(len(self.text.strip()), 0, "Workflow file should not be empty")

    def test_top_level_name_present(self):
        # Text-level check
        self.assertRegex(self.text, r"^name:\s*CI/CD Pipeline\\s*$", "Workflow name should be 'CI/CD Pipeline'")
        if self.data is not None:
            self.assertEqual(self.data.get("name"), "CI/CD Pipeline")

    def test_permissions_block(self):
        # Ensure permissions exist and include contents: write and packages: write
        if self.data is not None:
            perms = self.data.get("permissions")
            self.assertIsInstance(perms, dict, "permissions should be a mapping")
            self.assertEqual(perms.get("contents"), "write", "permissions.contents should be 'write'")
            self.assertEqual(perms.get("packages"), "write", "permissions.packages should be 'write'")
        else:
            # Fallback textual checks
            self.assertIn("permissions:", self.text)
            self.assertRegex(self.text, r"\\bcontents:\\s*write\\b")
            self.assertRegex(self.text, r"\\bpackages:\\s*write\\b")

    def test_on_triggers(self):
        if self.data is not None:
            on = self.data.get("on")
            self.assertIsInstance(on, dict)
            # push to branches main and develop
            push = on.get("push")
            self.assertIsInstance(push, dict)
            self.assertIn("branches", push)
            self.assertCountEqual(push.get("branches"), ["main", "develop"])
            # pull_request branches main
            pr = on.get("pull_request")
            self.assertIsInstance(pr, dict)
            self.assertEqual(pr.get("branches"), ["main"])
            # release types created
            rel = on.get("release")
            self.assertIsInstance(rel, dict)
            self.assertEqual(rel.get("types"), ["created"])
        else:
            # Textual checks
            self.assertRegex(self.text, r"on:\\s*\\n\\s*push:\\s*\\n\\s*branches:\\s*\\[\\s*main\\s*,\\s*develop\\s*\\]")
            self.assertRegex(self.text, r"pull_request:\\s*\\n\\s*branches:\\s*\\[\\s*main\\s*\\]")
            self.assertRegex(self.text, r"release:\\s*\\n\\s*types:\\s*\\[\\s*created\\s*\\]")

    def test_jobs_exist(self):
        # pre-commit, python-check, docker-build, create-github-release
        expected_jobs = ["pre-commit", "python-check", "docker-build", "create-github-release"]
        if self.data is not None:
            jobs = self.data.get("jobs")
            self.assertIsInstance(jobs, dict)
            for job in expected_jobs:
                self.assertIn(job, jobs, f"Expected job '{job}' not found")
        else:
            for job in expected_jobs:
                self.assertIn(job + ":", self.text)

    def test_pre_commit_job(self):
        if self.data is None:
            # Textual checks
            self.assertIn("pre-commit:", self.text)
            self.assertRegex(self.text, r"runs-on:\\s*ubuntu-latest")
            self.assertIn("actions/checkout@v4", self.text)
            self.assertIn("actions/setup-python@v5", self.text)
            self.assertRegex(self.text, r"python-version:\\s*'3\\.11'")
            self.assertIn("pip install pre-commit", self.text)
            self.assertIn("pre-commit run --all-files", self.text)
            self.assertIn("cd mcp-server && pip install -e .", self.text)
            return

        jobs = self.data["jobs"]
        job = jobs.get("pre-commit")
        self.assertIsInstance(job, dict)
        self.assertEqual(job.get("runs-on"), "ubuntu-latest")
        steps = job.get("steps", [])
        # step: checkout
        self.assertTrue(any(s.get("uses","").startswith("actions/checkout@v4") for s in steps))
        # step: setup-python with 3.11
        setup_py = [s for s in steps if s.get("uses","").startswith("actions/setup-python@v5")]
        self.assertTrue(setup_py, "setup-python step should be present")
        if setup_py:
            self.assertEqual(setup_py[0].get("with", {}).get("python-version"), "3.11")
        # step: install pre-commit and editable mcp-server
        self.assertTrue(any("pip install pre-commit" in (s.get("run") or "") for s in steps))
        self.assertTrue(any("cd mcp-server && pip install -e ." in (s.get("run") or "") for s in steps))
        # step: run pre-commit
        self.assertTrue(any(re.search(r"pre-commit\\s+run\\s+--all-files", (s.get("run") or "")) for s in steps))

    def test_python_check_job(self):
        if self.data is None:
            self.assertIn("python-check:", self.text)
            self.assertIn("actions/checkout@v4", self.text)
            self.assertIn("actions/setup-python@v5", self.text)
            self.assertRegex(self.text, r"python-version:\\s*'3\\.11'")
            self.assertIn("python -m py_compile mcp-server/src/*.py", self.text)
            return

        job = self.data["jobs"]["python-check"]
        self.assertEqual(job.get("runs-on"), "ubuntu-latest")
        steps = job.get("steps", [])
        self.assertTrue(any(s.get("uses","").startswith("actions/checkout@v4") for s in steps))
        setup_py = [s for s in steps if s.get("uses","").startswith("actions/setup-python@v5")]
        self.assertTrue(setup_py)
        if setup_py:
            self.assertEqual(setup_py[0].get("with", {}).get("python-version"), "3.11")
        self.assertTrue(any("python -m py_compile mcp-server/src/*.py" in (s.get("run") or "") for s in steps))

    def test_docker_build_job(self):
        if self.data is None:
            self.assertIn("docker-build:", self.text)
            self.assertRegex(self.text, r"needs:\\s*\\[\\s*pre-commit\\s*,\\s*python-check\\s*\\]")
            self.assertIn("docker/setup-buildx-action@v3", self.text)
            self.assertIn("docker compose build", self.text)
            self.assertIn("docker compose up -d qdrant", self.text)
            self.assertIn("curl -f http://localhost:6333/", self.text)
            self.assertIn("docker compose down", self.text)
            return

        job = self.data["jobs"]["docker-build"]
        self.assertEqual(job.get("runs-on"), "ubuntu-latest")
        self.assertEqual(sorted(job.get("needs", [])), ["pre-commit", "python-check"])
        steps = job.get("steps", [])
        self.assertTrue(any(s.get("uses","").startswith("actions/checkout@v4") for s in steps))
        self.assertTrue(any(s.get("uses","").startswith("docker/setup-buildx-action@v3") for s in steps))
        self.assertTrue(any("docker compose build" in (s.get("run") or "") for s in steps))
        # The test stack should bring up qdrant, wait, health-check, and tear down
        run_cmds = "n/a"
        for s in steps:
            if "run" in s:
                run_cmds = s["run"]
                if "docker compose up -d qdrant" in run_cmds:
                    self.assertIn("sleep 10", run_cmds, "Should include a delay allowing qdrant to start")
                    self.assertIn("curl -f http://localhost:6333/", run_cmds, "Should health-check qdrant")
                    self.assertIn("docker compose down", run_cmds, "Should tear down the stack")
                    break
        else:
            self.fail("Did not find qdrant test run commands in docker-build job")

    def test_create_github_release_job(self):
        # Note: job runs only on release event; verify conditional, archive creation and upload
        if self.data is None:
            self.assertIn("create-github-release:", self.text)
            self.assertRegex(self.text, r"^\\s*if:\\s*github\\.event_name\\s*==\\s*'release'\\s*$", "Missing 'if' guard for release event")
            self.assertIn("softprops/action-gh-release@v2", self.text)
            self.assertRegex(self.text, r"claude-self-reflect-\\$\\{\\{\\s*github\\.event\\.release\\.tag_name\\s*\\}\\}\\.tar\\.gz")
            return

        job = self.data["jobs"]["create-github-release"]
        self.assertEqual(job.get("runs-on"), "ubuntu-latest")
        self.assertEqual(job.get("needs"), "docker-build")
        self.assertEqual(job.get("if"), "github.event_name == 'release'")
        steps = job.get("steps", [])
        self.assertTrue(any(s.get("uses","").startswith("actions/checkout@v4") for s in steps))
        # Archive creation step checks
        archive_steps = [s for s in steps if "run" in s and "Create release archive" in s.get("name","")]
        self.assertTrue(archive_steps, "Expected a 'Create release archive' step")
        if archive_steps:
            run_cmd = archive_steps[0]["run"]
            self.assertIn("mkdir -p /tmp/release", run_cmd)
            self.assertRegex(run_cmd, r"tar -czf /tmp/release/claude-self-reflect-\\$\\{\\{\\s*github\\.event\\.release\\.tag_name\\s*\\}\\}\\.tar\\.gz")
            # Ensure critical excludes are present
            for excl in ["node_modules", "venv", ".venv", ".git", "data", "qdrant_storage", "*.tar.gz", "__pycache__", "*.pyc", "logs", ".claude/agents"]:
                self.assertIn(f"--exclude='{excl}'", run_cmd, f"Expected to exclude '{excl}' from archive")
            self.assertIn("mv /tmp/release/claude-self-reflect-${{ github.event.release.tag_name }}.tar.gz .", run_cmd)

        # Upload step
        upload_steps = [s for s in steps if s.get("uses","").startswith("softprops/action-gh-release@v2")]
        self.assertTrue(upload_steps, "Expected a 'softprops/action-gh-release@v2' upload step")
        if upload_steps:
            with_cfg = upload_steps[0].get("with", {})
            self.assertEqual(with_cfg.get("files"), "./claude-self-reflect-${{ github.event.release.tag_name }}.tar.gz")
            env_cfg = upload_steps[0].get("env", {})
            self.assertIn("GITHUB_TOKEN", env_cfg, "Upload step should define GITHUB_TOKEN in env")

    def test_no_unexpected_mutations(self):
        # Sanity checks that core expected strings remain present
        expected_strings = [
            "actions/checkout@v4",
            "actions/setup-python@v5",
            "python-version: '3.11'",
            "python -m py_compile mcp-server/src/*.py",
            "docker/setup-buildx-action@v3",
            "docker compose build",
            "docker compose up -d qdrant",
            "curl -f http://localhost:6333/",
            "docker compose down",
            "softprops/action-gh-release@v2",
        ]
        for s in expected_strings:
            with self.subTest(s=s):
                self.assertIn(s, self.text, f"Expected to find: {s}")

if __name__ == "__main__":
    # Allow running via python tests/test_github_ci_workflow.py directly
    unittest.main()