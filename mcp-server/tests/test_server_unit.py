"""
Unit tests for the MCP server tools and helpers.

Covers:
- Qdrant interactions via a lightweight in-memory fake client
- MCP tools: store_reflection and reflect_on_past
- Helper utilities: normalize_text, build_search_filter
- Tool registration on the FastMCP instance

Notes:
- The tests avoid real network/embedding downloads by monkeypatching
  the global qdrant client and the embedding generator.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qdrant_client.http.exceptions import UnexpectedResponse  # noqa: E402

import server  # noqa: E402


class DummyCtx:
    """Minimal async context stub for MCP tools.

    Provides `error()` used by tool implementations.
    """

    def __init__(self) -> None:
        self.errors: List[str] = []

    async def error(self, msg: str) -> None:
        self.errors.append(msg)


class FakePoint:
    """Simple point structure to emulate Qdrant response points."""

    def __init__(self, pid: Any, payload: Dict[str, Any], score: float = 0.9) -> None:
        self.id = pid
        self.payload = payload
        self.score = score


class FakeQueryResult:
    def __init__(self, points: List[FakePoint]) -> None:
        self.points = points


class FakeAsyncQdrantClient:
    """In-memory fake for AsyncQdrantClient used by tests.

    Stores points per collection and implements the minimal surface used by the server:
    - get_collection: raises UnexpectedResponse if not found
    - create_collection: creates an empty collection
    - upsert: appends points to collection storage
    - query_points: returns matching points with simple filtering by payload
    """

    def __init__(self) -> None:
        self._collections: Dict[str, List[FakePoint]] = {}
        self.closed = False

    async def get_collection(self, name: str) -> Dict[str, Any]:
        if name not in self._collections:
            # Mimic Qdrant behavior: raise when collection is missing
            raise UnexpectedResponse("Collection not found")
        return {"status": "ok", "name": name}

    async def create_collection(self, collection_name: str, vectors_config: Any) -> Dict[str, Any]:  # noqa: ARG002
        self._collections.setdefault(collection_name, [])
        return {"status": "created", "name": collection_name}

    async def upsert(self, collection_name: str, points: List[Any]) -> Dict[str, Any]:
        bucket = self._collections.setdefault(collection_name, [])
        for p in points:
            # Preserve IDs as strings to match Qdrant UUID usage
            pid = str(p.id)
            payload = dict(p.payload)
            # Ensure timestamp exists for scoring fallback
            payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            bucket.append(FakePoint(pid=pid, payload=payload))
        return {"status": "ok", "count": len(points)}

    async def query_points(self, collection_name: str, **kwargs: Any) -> FakeQueryResult:
        # Extract stored points
        bucket = list(self._collections.get(collection_name, []))

        # Apply simple filter by payload if provided
        query_filter = kwargs.get("query_filter")
        if query_filter and isinstance(query_filter, dict) and query_filter.get("must"):

            def match(point: FakePoint) -> bool:
                for cond in query_filter["must"]:
                    key = cond.get("key")
                    wanted = cond.get("match", {}).get("value")
                    value = point.payload.get(key)
                    if isinstance(value, list):
                        if wanted not in value:
                            return False
                    else:
                        if value != wanted:
                            return False
                return True

            bucket = [p for p in bucket if match(p)]

        # Apply limit
        limit = int(kwargs.get("limit", 10))
        points = bucket[:limit]
        # Ensure each point has a score attribute
        for p in points:
            if not hasattr(p, "score") or p.score is None:
                p.score = 0.9
        return FakeQueryResult(points)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _env_setup(monkeypatch):
    """Ensure required environment vars are set for tests and patch embedding.

    - Sets EMBEDDING_MODEL and VECTOR_SIZE
    - Patches server.generate_embedding to a deterministic vector
    - Replaces global qdrant_client with an in-memory fake
    - Ensures model readiness won't block calls
    """

    monkeypatch.setenv("EMBEDDING_MODEL", "test/model")
    monkeypatch.setenv("VECTOR_SIZE", "8")

    # Patch embedding generator to a deterministic vector of the requested size
    async def fake_generate_embedding(text: str) -> List[float]:
        size = int(os.getenv("VECTOR_SIZE", "8"))
        # Simple pattern based on text length for variety
        val = float((len(text) % 3) + 1) / 10.0
        return [val] * size

    monkeypatch.setattr(server, "generate_embedding", fake_generate_embedding)

    # Replace qdrant client
    fake = FakeAsyncQdrantClient()
    monkeypatch.setattr(server, "qdrant_client", fake)

    yield


@pytest.mark.asyncio
async def test_store_reflection_creates_collection_and_upserts():
    """store_reflection: creates collection if missing and upserts one point with expected payload.

    Steps:
    - Use FakeAsyncQdrantClient (empty state)
    - Call store_reflection with a specific project and tags
    - Verify collection creation and that a point with required payload fields is stored
    """

    ctx = DummyCtx()
    project = "unit-project"
    content = "Test reflection content"
    tags = ["unit", "reflection"]

    msg = await server.store_reflection(ctx, content=content, tags=tags, project=project)
    assert "Reflection stored successfully" in msg

    # Inspect fake storage
    fake: FakeAsyncQdrantClient = server.qdrant_client  # type: ignore[assignment]
    stored = fake._collections.get(server.MAIN_COLLECTION, [])
    assert len(stored) == 1
    payload = stored[0].payload

    assert payload["text"] == content
    assert payload["tags"] == tags
    assert payload["project_name"] == project
    assert payload["type"] == "reflection"
    assert payload["role"] == "user_reflection"


@pytest.mark.asyncio
async def test_reflect_on_past_filters_by_project_and_tags():
    """reflect_on_past: returns only reflections matching project and ALL specified tags.

    Steps:
    - Insert three reflections across two projects with varying tags
    - Query with project A and tag 'keep' -> only matching entries returned
    - Ensure formatted text contains expected counts and fields
    """

    ctx = DummyCtx()

    # Seed reflections
    await server.store_reflection(ctx, content="alpha", tags=["keep"], project="projA")
    await server.store_reflection(ctx, content="beta", tags=["skip"], project="projA")
    await server.store_reflection(ctx, content="gamma", tags=["keep", "extra"], project="projB")

    # Search within projA for tag=keep; only "alpha" should match
    result = await server.reflect_on_past(
        ctx,
        query="al",
        limit=5,
        min_score=0.0,
        use_decay=0,
        project="projA",
        tags=["keep"],
    )

    assert "Found" in result
    assert "Project: projA" in result
    assert "alpha" in result
    assert "beta" not in result
    assert "gamma" not in result


def test_build_search_filter_and_normalize_text():
    """Helper functions: build_search_filter constructs AND filters; normalize_text folds case/whitespace.

    Steps:
    - build_search_filter with project and two tags builds a `must` list of 3 conditions
    - normalize_text collapses whitespace and lowercases text with unicode normalization
    """

    f = server.build_search_filter(project_name="p", tags=["a", "b"])
    assert f == {
        "must": [
            {"key": "project_name", "match": {"value": "p"}},
            {"key": "tags", "match": {"value": "a"}},
            {"key": "tags", "match": {"value": "b"}},
        ]
    }

    assert server.normalize_text("  HéLLo\n  WORLD  ") == "hello world"


def test_convert_point_score_priority_payload_over_attr():
    """convert_point_to_search_result should prefer payload['score'] over point.score.

    - When payload contains 'score', it must override base point.score
    - When payload lacks 'score', it should fall back to point.score
    """

    # Case 1: payload['score'] present -> take it over point.score
    p1 = FakePoint(pid=1, payload={"text": "x", "score": 0.42}, score=0.9)
    res1 = server.convert_point_to_search_result(point=p1, min_score=0.0)
    assert res1 is not None
    assert abs(res1.score - 0.42) < 1e-9

    # Case 2: payload['score'] missing -> fall back to point.score
    p2 = FakePoint(pid=2, payload={"text": "y"}, score=0.77)
    res2 = server.convert_point_to_search_result(point=p2, min_score=0.0)
    assert res2 is not None
    assert abs(res2.score - 0.77) < 1e-9


@pytest.mark.asyncio
async def test_mcp_tools_registered():
    """FastMCP instance exposes both tools.

    Steps:
    - server.mcp.get_tools() returns mapping containing 'reflect_on_past' and 'store_reflection'
    """

    tools = await server.mcp.get_tools()
    assert "reflect_on_past" in tools
    assert "store_reflection" in tools
