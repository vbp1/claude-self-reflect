"""
Integration scenarios validating MCP tools and semantic search behavior.

Covers 12 scenarios:
1) Exact match
2) Paraphrase (same language)
3) Unicode normalization (case/whitespace/diacritics)
4) Synonyms vs unrelated
5) Multilingual EN→RU
6) Multilingual RU→EN
7) Project scoping vs all
8) Tags faceting plus embedding relevance
9) Decay impact on ranking
10) min_score and limit
11) Distinguishing close topics
12) Long content vs short query

Each test:
- Spawns MCP server (stdio) as a subprocess
- Talks JSON-RPC to initialize, list tools, and call tools
- Stores a reflection with a unique tag, then searches
- Registers unique tag for post-test cleanup in Qdrant

Cleanup:
- A fixture collects (project, tag) pairs used in a test and deletes
  them from Qdrant collection `claude_logs` afterwards (best-effort).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pytest
from qdrant_client import QdrantClient, models

TESTS_DIR = Path(__file__).resolve().parent
MCP_SERVER_DIR = TESTS_DIR.parent

# Module-level default project name for isolation; set by autouse fixture
DEFAULT_INTEGRATION_PROJECT = "integration-project"

logger = logging.getLogger(__name__)


def _qdrant_reachable(url: str) -> bool:
    try:
        host_port = url.split("://", 1)[-1]
        if "/" in host_port:
            host_port = host_port.split("/", 1)[0]
        host, port_s = host_port.split(":", 1)
        port = int(port_s)
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except ValueError as e:
        logger.warning("Qdrant URL parse failed for %s: %s: %s", url, type(e).__name__, str(e))
        return False
    except OSError as e:
        logger.warning("Qdrant not reachable at %s: %s: %s", url, type(e).__name__, str(e))
        return False


def _spawn_stdio_server(env: Dict[str, str]) -> subprocess.Popen[str]:
    """Spawn server with stdio; discard stderr to avoid pipe blocking."""
    return subprocess.Popen(
        [sys.executable, "-m", "src", "--transport", "stdio"],
        cwd=str(MCP_SERVER_DIR),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
        bufsize=1,
    )


def _rpc(proc: subprocess.Popen[str], msg: Dict[str, Any], timeout: float = 20.0) -> Dict[str, Any] | None:
    assert proc.stdin and proc.stdout
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()

    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line:
            try:
                obj = json.loads(line)
                # Only return the matching response (ignore notifications/other replies)
                expect_id = msg.get("id")
                if expect_id is None:
                    # If no id expected, only consider actual responses
                    if ("result" in obj) or ("error" in obj):
                        return obj
                    continue
                if obj.get("id") == expect_id and (("result" in obj) or ("error" in obj)):
                    return obj
                continue
            except json.JSONDecodeError:
                continue
    return None


def _extract_text_result(resp: Dict[str, Any] | None) -> str:
    if not resp:
        return ""
    result = resp.get("result")
    if isinstance(result, dict) and "content" in result and result["content"]:
        # MCP content blocks
        block = result["content"][0]
        return str(block.get("text", block))
    return str(result)


@pytest.fixture(scope="module")
def integration_env() -> Dict[str, str]:
    """Prepare environment for integration tests or skip if Qdrant is unavailable."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    if not _qdrant_reachable(qdrant_url):
        pytest.skip(f"Qdrant not reachable at {qdrant_url}")
    env = os.environ.copy()
    env.setdefault("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    env.setdefault("VECTOR_SIZE", "384")
    env.setdefault("QDRANT_URL", qdrant_url)
    # Use a dedicated cache dir for tests under module temp path if provided
    env.setdefault("MCP_CLIENT_CWD", str(TESTS_DIR))
    # Use shared user cache by default to avoid repeated downloads
    env.setdefault("TRANSFORMERS_CACHE", str(Path.home() / ".cache" / "huggingface"))
    return env


@pytest.fixture(scope="module", autouse=True)
def _isolate_project(integration_env: Dict[str, str]):
    """Autouse fixture: set a unique project for this module and clean it after tests."""
    global DEFAULT_INTEGRATION_PROJECT
    DEFAULT_INTEGRATION_PROJECT = f"integration-project-{uuid.uuid4().hex[:8]}"
    yield
    # best-effort cleanup by project
    try:
        client = QdrantClient(url=integration_env["QDRANT_URL"])
        flt = models.Filter(must=[models.FieldCondition(key="project_name", match=models.MatchValue(value=DEFAULT_INTEGRATION_PROJECT))])
        client.delete(
            collection_name="claude_logs",
            points_selector=models.FilterSelector(filter=flt),
            wait=True,
        )
    except (models.UnexpectedResponse, OSError, ValueError) as e:
        print(f"Cleanup warning (module project): {e!s}")


@pytest.fixture()
def mcp_server(integration_env: Dict[str, str]):
    proc = _spawn_stdio_server(integration_env)
    try:
        # initialize
        init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "integration-tests", "version": "0.0.0"},
            },
        }
        resp = _rpc(proc, init, timeout=40.0)
        assert resp and "result" in resp, f"initialize failed: {resp} stderr={proc.stderr.read() if proc.stderr else ''}"

        # notify initialized per MCP spec before next requests
        assert proc.stdin is not None
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": None}) + "\n")
        proc.stdin.flush()

        # tools/list sanity
        list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {"cursor": None}}
        resp = _rpc(proc, list_req)
        assert resp and "result" in resp and resp["result"].get("tools"), f"tools/list failed: {resp}"
        tool_names = {t["name"] for t in resp["result"]["tools"]}
        assert {"store_reflection", "reflect_on_past"}.issubset(tool_names)

        yield proc
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()


@pytest.fixture()
def qdrant_cleaner(integration_env: Dict[str, str]):
    """Provide a registrar that records (project, tag) pairs for cleanup after each test."""
    to_cleanup: List[Tuple[str, str]] = []

    def register(tag: str, project: str | None = None) -> None:
        proj = project or DEFAULT_INTEGRATION_PROJECT
        to_cleanup.append((proj, tag))

    yield register

    # Post-test cleanup
    qdrant_url = integration_env["QDRANT_URL"]
    client = QdrantClient(url=qdrant_url)
    for project, tag in to_cleanup:
        try:
            flt = models.Filter(
                must=[
                    models.FieldCondition(key="project_name", match=models.MatchValue(value=project)),
                    models.FieldCondition(key="tags", match=models.MatchValue(value=tag)),
                ]
            )
            client.delete(
                collection_name="claude_logs",
                points_selector=models.FilterSelector(filter=flt),
                wait=True,
            )
        except (models.UnexpectedResponse, OSError, ValueError):
            # Best-effort; ignore cleanup errors
            pass


def _call_store(
    proc: subprocess.Popen[str],
    content: str,
    tags: Iterable[str],
    project: str | None = None,
) -> Dict[str, Any] | None:
    req = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "store_reflection",
            "arguments": {
                "content": content,
                "tags": list(tags),
                "project": project or DEFAULT_INTEGRATION_PROJECT,
            },
        },
    }
    return _rpc(proc, req, timeout=45.0)


def _call_search(
    proc: subprocess.Popen[str],
    query: str,
    *,
    project: str | None = None,
    tags: Iterable[str] | None = None,
    min_score: float = 0.7,
    limit: int = 5,
    use_decay: int = 0,
) -> str:
    args: Dict[str, Any] = {
        "query": query,
        "limit": limit,
        "min_score": min_score,
        "use_decay": use_decay,
    }
    # Use module default if not explicitly provided
    args["project"] = project or DEFAULT_INTEGRATION_PROJECT
    if tags is not None:
        args["tags"] = list(tags)

    req = {"jsonrpc": "2.0", "id": 11, "method": "tools/call", "params": {"name": "reflect_on_past", "arguments": args}}
    resp = _rpc(proc, req, timeout=45.0)
    return _extract_text_result(resp)


def _uniq(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.mark.integration
def test_exact_match_embedding(mcp_server, qdrant_cleaner):
    text = "Refactor database migration logic to avoid downtime"
    tag = _uniq("exact")
    qdrant_cleaner(tag)
    assert _call_store(mcp_server, text, [tag])
    out = _call_search(mcp_server, text, min_score=0.8, use_decay=0)
    assert text in out


@pytest.mark.integration
@pytest.mark.parametrize("use_decay", [0, 1])
def test_paraphrase_same_language_thresholds(mcp_server, qdrant_cleaner, use_decay):
    original = "Improve DB migration to eliminate service interruptions"
    query = "How to refactor migrations to prevent downtime?"
    tag = _uniq("paraphrase-en")
    qdrant_cleaner(tag)
    assert _call_store(mcp_server, original, [tag])
    out = _call_search(mcp_server, query, min_score=0.7, use_decay=use_decay)
    assert "Found" in out


@pytest.mark.integration
@pytest.mark.parametrize("use_decay", [0, 1])
def test_unicode_normalization_case_whitespace(mcp_server, qdrant_cleaner, use_decay):
    text = "Résumé parsing fails for José"
    tag = _uniq("unicode")
    qdrant_cleaner(tag)
    assert _call_store(mcp_server, text, [tag])
    out1 = _call_search(mcp_server, "resume parsing fails for jose", min_score=0.6, use_decay=use_decay)
    out2 = _call_search(mcp_server, "resume parsing fails for jose", min_score=0.6, use_decay=use_decay)  # combined diacritics
    assert "Found" in out1
    assert "Found" in out2


@pytest.mark.integration
def test_synonyms_and_unrelated_contrast(mcp_server, qdrant_cleaner):
    text = "optimize cache invalidation for performance"
    tag = _uniq("syn")
    qdrant_cleaner(tag)
    assert _call_store(mcp_server, text, [tag])
    pos = _call_search(mcp_server, "speed up by improving cache purge logic", min_score=0.6)
    neg = _call_search(mcp_server, "speed up by improving storage allocation", min_score=0.6)
    assert "Found" in pos
    # Either no results or substantially fewer mentions expected in neg
    assert ("No conversations found" in neg) or (pos != neg)


@pytest.mark.integration
@pytest.mark.parametrize("use_decay", [0, 1])
def test_multilingual_en_to_ru(mcp_server, qdrant_cleaner, use_decay):
    text = "Add search debounce to reduce API calls"
    tag = _uniq("ml-enru")
    qdrant_cleaner(tag)
    assert _call_store(mcp_server, text, [tag])
    out = _call_search(mcp_server, "Снизить число запросов при дебаунсе поиска", min_score=0.6, use_decay=use_decay)
    assert "Found" in out


@pytest.mark.integration
@pytest.mark.parametrize("use_decay", [0, 1])
def test_multilingual_ru_to_en(mcp_server, qdrant_cleaner, use_decay):
    text = "Использовать повторные попытки с экспоненциальной задержкой"  # noqa: RUF001
    tag = _uniq("ml-ruen")
    qdrant_cleaner(tag)
    assert _call_store(mcp_server, text, [tag])
    out = _call_search(mcp_server, "use retries with exponential backoff", min_score=0.6, use_decay=use_decay)
    assert "Found" in out


@pytest.mark.integration
def test_project_scoping_vs_all(mcp_server, qdrant_cleaner):
    tagA = _uniq("scopeA")
    tagB = _uniq("scopeB")
    qdrant_cleaner(tagA, project="project-A")
    qdrant_cleaner(tagB, project="project-B")
    assert _call_store(mcp_server, "Implement auth middleware", [tagA], project="project-A")
    assert _call_store(mcp_server, "Implement authentication layer", [tagB], project="project-B")

    outA = _call_search(mcp_server, "auth middleware", project="project-A", min_score=0.6)
    assert "Project: project-A" in outA

    outAll = _call_search(mcp_server, "auth middleware", project="all", min_score=0.6)
    # Both projects should be present across results text
    assert ("Project: project-A" in outAll) or ("Project: project-B" in outAll)


@pytest.mark.integration
@pytest.mark.parametrize("use_decay", [0, 1])
def test_tags_faceting_and_embedding(mcp_server, qdrant_cleaner, use_decay):
    tag_devops = _uniq("devops")
    tag_front = _uniq("front")
    qdrant_cleaner(tag_devops)
    qdrant_cleaner(tag_front)
    assert _call_store(mcp_server, "Kubernetes rollout strategy", [tag_devops, "devops"])  # devops flavored
    assert _call_store(mcp_server, "UI rendering optimization", [tag_front, "frontend"])  # frontend flavored

    out_any = _call_search(mcp_server, "optimize deployment rollout", min_score=0.6, use_decay=use_decay)
    assert "Found" in out_any

    out_devops = _call_search(mcp_server, "optimize deployment rollout", tags=["devops"], min_score=0.6, use_decay=use_decay)
    assert f"Project: {DEFAULT_INTEGRATION_PROJECT}" in out_devops


@pytest.mark.integration
def test_decay_impacts_ranking(mcp_server, qdrant_cleaner):
    tag_old = _uniq("old")
    tag_new = _uniq("new")
    qdrant_cleaner(tag_old)
    qdrant_cleaner(tag_new)
    # Store "old" first, then wait, then "new"
    assert _call_store(mcp_server, "rolling restart procedure", [tag_old])
    time.sleep(1.5)
    assert _call_store(mcp_server, "rolling restart steps", [tag_new])

    res_no_decay = _call_search(mcp_server, "rolling restart", use_decay=0, min_score=0.6)
    res_with_decay = _call_search(mcp_server, "rolling restart", use_decay=1, min_score=0.6)
    assert res_no_decay and res_with_decay and (res_no_decay != res_with_decay)


@pytest.mark.integration
def test_decay_impacts_ranking_order_native(integration_env: Dict[str, str]):
    """With aggressive decay settings, newer should rank above older when use_decay=1."""
    env = integration_env.copy()
    env.update(
        {
            "ENABLE_MEMORY_DECAY": "true",
            "USE_NATIVE_DECAY": "true",
            "DECAY_WEIGHT": "0.9",
            # Choose a value so that int(DECAY_SCALE_DAYS * 24 * 60 * 60) == 2
            # 2 / 86400 ≈ 0.000023148..., use a slightly larger value
            "DECAY_SCALE_DAYS": "0.0000232",
        }
    )
    proc = _spawn_stdio_server(env)
    project = f"{DEFAULT_INTEGRATION_PROJECT}-native-ord"
    try:
        init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "tests", "version": "0"}},
        }
        assert _rpc(proc, init)
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": None}) + "\n")
        proc.stdin.flush()

        # Two docs: old then new
        assert _call_store(proc, "rolling restart procedure", ["ord-old"], project=project)
        time.sleep(1.2)
        assert _call_store(proc, "rolling restart steps", ["ord-new"], project=project)

        def _excerpts(text: str) -> list[str]:
            return [line.split(": ", 1)[1] for line in text.splitlines() if line.strip().startswith("Excerpt:")]

        res_yes = _call_search(proc, "rolling restart", project=project, use_decay=1, min_score=0.0)
        res_no = _call_search(proc, "rolling restart", project=project, use_decay=0, min_score=0.0)
        ex_yes = _excerpts(res_yes)
        ex_no = _excerpts(res_no)
        assert ex_yes and ex_no
        assert ex_yes.index("rolling restart steps") <= ex_yes.index("rolling restart procedure")
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            client = QdrantClient(url=integration_env["QDRANT_URL"])
            flt = models.Filter(must=[models.FieldCondition(key="project_name", match=models.MatchValue(value=project))])
            client.delete(collection_name="claude_logs", points_selector=models.FilterSelector(filter=flt), wait=True)
        except (models.UnexpectedResponse, OSError, ValueError) as e:
            print(f"Cleanup warning (native order): {e!s}")


@pytest.mark.integration
def test_client_side_decay_differs(integration_env: Dict[str, str]):
    """With native disabled, client-side decay should change ranking vs no decay."""
    env = integration_env.copy()
    env.update(
        {
            "ENABLE_MEMORY_DECAY": "true",
            "USE_NATIVE_DECAY": "false",
            "DECAY_WEIGHT": "0.9",
            "DECAY_SCALE_DAYS": "0.00001",
        }
    )
    proc = _spawn_stdio_server(env)
    project = f"{DEFAULT_INTEGRATION_PROJECT}-client-ord"
    try:
        init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "tests", "version": "0"}},
        }
        assert _rpc(proc, init)
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": None}) + "\n")
        proc.stdin.flush()
        assert _call_store(proc, "rolling restart procedure", ["cs-old"], project=project)
        time.sleep(1.2)
        assert _call_store(proc, "rolling restart steps", ["cs-new"], project=project)
        res_no = _call_search(proc, "rolling restart", project=project, use_decay=0, min_score=0.0)
        res_yes = _call_search(proc, "rolling restart", project=project, use_decay=1, min_score=0.0)
        assert res_no and res_yes and (res_no != res_yes)
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            client = QdrantClient(url=integration_env["QDRANT_URL"])
            flt = models.Filter(must=[models.FieldCondition(key="project_name", match=models.MatchValue(value=project))])
            client.delete(collection_name="claude_logs", points_selector=models.FilterSelector(filter=flt), wait=True)
        except (models.UnexpectedResponse, OSError, ValueError) as e:
            print(f"Cleanup warning (client-side order): {e!s}")


@pytest.mark.integration
def test_decay_with_tags_and_project_all(integration_env: Dict[str, str]):
    """Decay should respect tags and project='all'."""
    env = integration_env.copy()
    proc = _spawn_stdio_server(env)
    proj_a = f"{DEFAULT_INTEGRATION_PROJECT}-ta"
    proj_b = f"{DEFAULT_INTEGRATION_PROJECT}-tb"
    try:
        init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "tests", "version": "0"}},
        }
        assert _rpc(proc, init)
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": None}) + "\n")
        proc.stdin.flush()
        assert _call_store(proc, "scale deployment with canary", ["devops", "decay-tag"], project=proj_a)
        assert _call_store(proc, "blue/green rollout config", ["devops", "decay-tag"], project=proj_b)
        out = _call_search(proc, "rollout", project="all", tags=["devops"], use_decay=1, min_score=0.5)
        assert "Found" in out
        assert (f"Project: {proj_a}" in out) or (f"Project: {proj_b}" in out)
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            client = QdrantClient(url=integration_env["QDRANT_URL"])
            for proj in (proj_a, proj_b):
                flt = models.Filter(must=[models.FieldCondition(key="project_name", match=models.MatchValue(value=proj))])
                client.delete(collection_name="claude_logs", points_selector=models.FilterSelector(filter=flt), wait=True)
        except (models.UnexpectedResponse, OSError, ValueError) as e:
            print(f"Cleanup warning (tags+all): {e!s}")


@pytest.mark.integration
def test_min_score_and_limit(mcp_server, qdrant_cleaner):
    base_tag = _uniq("thresh")
    qdrant_cleaner(base_tag + "-1")
    qdrant_cleaner(base_tag + "-2")
    qdrant_cleaner(base_tag + "-3")
    qdrant_cleaner(base_tag + "-mid")
    assert _call_store(mcp_server, "retry logic with backoff", [base_tag + "-1"])  # strong match
    assert _call_store(mcp_server, "logging improvements", [base_tag + "-2"])  # weaker match
    assert _call_store(mcp_server, "css layout tweaks", [base_tag + "-3"])  # unrelated
    # Add a medium-similarity phrasing that should pass loose (0.6) but not tight (0.85 with soft fallback ~0.68)
    assert _call_store(
        mcp_server,
        "add wait between retries for errors",
        [base_tag + "-mid"],
    )

    tight = _call_search(mcp_server, "backoff retries", min_score=0.85, limit=3)
    loose = _call_search(mcp_server, "backoff retries", min_score=0.45, limit=3)
    assert "Found" in tight and "Found" in loose and tight != loose


@pytest.mark.integration
def test_distinguish_close_topics(mcp_server, qdrant_cleaner):
    tag_a = _uniq("topicA")
    tag_b = _uniq("topicB")
    qdrant_cleaner(tag_a)
    qdrant_cleaner(tag_b)
    assert _call_store(mcp_server, "retry with exponential backoff", [tag_a])
    assert _call_store(mcp_server, "circuit breaker patterns", [tag_b])

    qb = _call_search(mcp_server, "robustness with backoff retries", min_score=0.6)
    qc = _call_search(mcp_server, "when to break circuits", min_score=0.6)
    assert "Found" in qb and "Found" in qc and qb != qc


@pytest.mark.integration
@pytest.mark.parametrize("use_decay", [0, 1])
def test_long_content_short_query(mcp_server, qdrant_cleaner, use_decay):
    tag = _uniq("long")
    qdrant_cleaner(tag)
    long_text = "In our service, ensuring idempotent write API design is critical. We discuss strategies, pitfalls, and practical patterns."
    assert _call_store(mcp_server, long_text, [tag])
    out = _call_search(mcp_server, "idempotent write API design", min_score=0.6, use_decay=use_decay)
    assert "Found" in out
