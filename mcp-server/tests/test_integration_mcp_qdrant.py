"""
Integration tests for MCP server over stdio with real Qdrant and local embedding model.

These tests:
- Spawn the server with `--transport stdio` and talk JSON-RPC (initialize, tools/list, tools/call)
- Use a real Qdrant at QDRANT_URL (defaults to http://localhost:6333)
- Positive scenario: store_reflection then reflect_on_past finds it
- Negative scenario: reflect_on_past for unknown query returns no results

Skips automatically if:
- Qdrant is not reachable at the configured URL
- Required model env vars are not set

Note: running these tests may download the embedding model on first run.
"""

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import pytest
from qdrant_client import QdrantClient, models

TESTS_DIR = Path(__file__).resolve().parent
MCP_SERVER_DIR = TESTS_DIR.parent
DEFAULT_INTEGRATION_PROJECT = "integration-project"


def _qdrant_reachable(url: str) -> bool:
    """Return True if Qdrant seems reachable on host:port from the URL string."""
    try:
        # crude parse like http://host:port
        host_port = url.split("://", 1)[-1]
        if "/" in host_port:
            host_port = host_port.split("/", 1)[0]
        host, port_s = host_port.split(":", 1)
        port = int(port_s)
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False


def _spawn_stdio_server(env: Dict[str, str]) -> subprocess.Popen[str]:
    """Spawn the MCP server with stdio transport and return the process.

    Use DEVNULL for stderr to prevent blocking on unconsumed stderr pipe.
    """
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


def _rpc(proc: subprocess.Popen[str], msg: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any] | None:
    """Send a JSON-RPC message and read one response line with a timeout; return parsed JSON or None."""
    assert proc.stdin and proc.stdout
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()

    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


@pytest.mark.integration
def test_stdio_store_then_search_positive_and_negative(tmp_path: Path):
    """End-to-end: store_reflection then reflect_on_past finds it; unknown query yields no results.

    Steps:
    - Skip if Qdrant unreachable
    - Start server with stdio
    - initialize -> ok
    - tools/list -> check tools present
    - tools/call store_reflection to write a unique content
    - tools/call reflect_on_past query by unique token -> expect results
    - tools/call reflect_on_past with unknown query -> expect no results message
    """

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    if not _qdrant_reachable(qdrant_url):
        pytest.skip(f"Qdrant not reachable at {qdrant_url}")

    # Ensure required env vars; use a tmp cache to avoid polluting user cache
    env = os.environ.copy()
    env.setdefault("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    env.setdefault("VECTOR_SIZE", "384")
    env.setdefault("QDRANT_URL", qdrant_url)
    env.setdefault("MCP_CLIENT_CWD", str(tmp_path))
    # Use shared user cache by default to avoid repeated model downloads
    env.setdefault("TRANSFORMERS_CACHE", str(Path.home() / ".cache" / "huggingface"))

    proc = _spawn_stdio_server(env)
    # Isolate project per test run
    project = f"integration-project-{uuid.uuid4().hex[:8]}"
    unique = f"integration-{int(time.time())}"
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
        resp = _rpc(proc, init, timeout=20.0)
        assert resp and "result" in resp, f"initialize failed: {resp} stderr={proc.stderr.read() if proc.stderr else ''}"

        # notify initialized per MCP spec before issuing further requests
        assert proc.stdin is not None
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": None}) + "\n")
        proc.stdin.flush()

        # tools/list
        list_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {"cursor": None},
        }
        resp = _rpc(proc, list_req)
        assert resp and "result" in resp and resp["result"].get("tools"), f"tools/list failed: {resp}"
        tool_names = {t["name"] for t in resp["result"]["tools"]}
        assert {"store_reflection", "reflect_on_past"}.issubset(tool_names)

        # store_reflection
        store_req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "store_reflection",
                "arguments": {
                    "content": f"Content for {unique}",
                    "tags": ["integration", unique],
                    "project": project,
                },
            },
        }
        resp = _rpc(proc, store_req, timeout=30.0)
        assert resp and "result" in resp, f"store_reflection failed: {resp}"

        # reflect_on_past positive
        search_req = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "reflect_on_past",
                "arguments": {
                    "query": unique,
                    "limit": 5,
                    "min_score": 0.0,
                    "use_decay": 0,
                    "project": project,
                },
            },
        }
        resp = _rpc(proc, search_req, timeout=20.0)
        assert resp and "result" in resp, f"reflect_on_past failed: {resp}"
        result_text = resp["result"]["content"][0]["text"] if isinstance(resp["result"], dict) else resp["result"]
        assert unique in str(result_text)

        # reflect_on_past negative
        search_bad = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "reflect_on_past",
                "arguments": {
                    "query": "definitely-not-present-xyz",
                    "limit": 3,
                    "min_score": 0.8,
                    "use_decay": 0,
                    "project": project,
                },
            },
        }
        resp = _rpc(proc, search_bad, timeout=20.0)
        assert resp and "result" in resp
        result_text2 = resp["result"]["content"][0]["text"] if isinstance(resp["result"], dict) else resp["result"]
        assert "No conversations found" in str(result_text2)

    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        # Cleanup test data in Qdrant (best-effort): delete points with the unique project
        try:
            client = QdrantClient(url=qdrant_url)
            flt = models.Filter(must=[models.FieldCondition(key="project_name", match=models.MatchValue(value=project))])
            client.delete(
                collection_name="claude_logs",
                points_selector=models.FilterSelector(filter=flt),
                wait=True,
            )
        except (models.UnexpectedResponse, OSError, ValueError):
            # best-effort cleanup; ignore common failures (collection missing, invalid filter, network)
            pass
