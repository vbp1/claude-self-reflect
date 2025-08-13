#!/usr/bin/env python3
"""Test script for tags filtering in reflection search."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "mcp-server" / "src"))

from server import (
    qdrant_client,
    generate_embedding,
    MAIN_COLLECTION,
    DEFAULT_PROJECT_NAME,
    perform_qdrant_search,
)
from qdrant_client.models import PointStruct, VectorParams, Distance


async def create_test_reflections():
    """Create test reflections with different tags."""
    
    # Ensure collection exists
    try:
        await qdrant_client.get_collection(MAIN_COLLECTION)
        print(f"Collection {MAIN_COLLECTION} exists")
    except Exception:
        print(f"Creating collection {MAIN_COLLECTION}")
        await qdrant_client.create_collection(
            collection_name=MAIN_COLLECTION,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    
    # Test data
    test_reflections = [
        {
            "content": "Important bug fix: Fixed memory leak in the vector database connection pool",
            "tags": ["bug", "memory", "database"],
        },
        {
            "content": "Performance improvement: Optimized embedding generation with batch processing",
            "tags": ["performance", "optimization", "embedding"],
        },
        {
            "content": "New feature: Added support for filtering search results by tags",
            "tags": ["feature", "search", "filtering"],
        },
        {
            "content": "Documentation update: Added examples for using the MCP tools",
            "tags": ["documentation", "mcp", "examples"],
        },
        {
            "content": "Critical security fix: Updated dependencies to patch vulnerability",
            "tags": ["security", "dependencies", "critical"],
        },
    ]
    
    points = []
    for i, reflection in enumerate(test_reflections):
        # Generate embedding
        embedding = await generate_embedding(reflection["content"])
        
        # Create point
        point_id = int(datetime.now(timezone.utc).timestamp() * 1000) + i
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "text": reflection["content"],
                "tags": reflection["tags"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "reflection",
                "role": "user_reflection",
                "start_role": "user_reflection",
                "project_name": DEFAULT_PROJECT_NAME,
                "conversation_id": f"test_reflection_{point_id}",
                "field": "text",
                "source": "test_script",
            },
        )
        points.append(point)
        print(f"Created reflection: {reflection['content'][:50]}... with tags: {reflection['tags']}")
    
    # Store all points
    await qdrant_client.upsert(collection_name=MAIN_COLLECTION, points=points)
    print(f"\nStored {len(points)} test reflections")
    return len(points)


async def test_search_with_tags():
    """Test searching with and without tag filters."""
    
    print("\n" + "=" * 60)
    print("TESTING SEARCH WITH TAG FILTERS")
    print("=" * 60)
    
    # Test 1: Search without tags (should return all relevant results)
    print("\n1. Search for 'fix' WITHOUT tag filter:")
    query_embedding = await generate_embedding("fix")
    
    results = await perform_qdrant_search(
        query_embedding=query_embedding,
        search_filter=None,
        limit=10,
        min_score=0.5,
        should_use_decay=False,
    )
    
    print(f"   Found {len(results)} results")
    for result in results[:3]:
        tags = result.payload.get("tags", [])
        print(f"   - Score: {result.score:.3f}, Tags: {tags}")
        print(f"     Text: {result.payload.get('text', '')[:80]}...")
    
    # Test 2: Search with single tag filter
    print("\n2. Search for 'fix' WITH tag filter ['bug']:")
    
    search_filter = {"must": [{"key": "tags", "match": {"any": ["bug"]}}]}
    
    results = await perform_qdrant_search(
        query_embedding=query_embedding,
        search_filter=search_filter,
        limit=10,
        min_score=0.5,
        should_use_decay=False,
    )
    
    print(f"   Found {len(results)} results")
    for result in results:
        tags = result.payload.get("tags", [])
        print(f"   - Score: {result.score:.3f}, Tags: {tags}")
        print(f"     Text: {result.payload.get('text', '')[:80]}...")
    
    # Test 3: Search with multiple tag filter (OR condition)
    print("\n3. Search for 'improvement' WITH tag filter ['performance', 'feature']:")
    
    query_embedding = await generate_embedding("improvement")
    search_filter = {"must": [{"key": "tags", "match": {"any": ["performance", "feature"]}}]}
    
    results = await perform_qdrant_search(
        query_embedding=query_embedding,
        search_filter=search_filter,
        limit=10,
        min_score=0.5,
        should_use_decay=False,
    )
    
    print(f"   Found {len(results)} results")
    for result in results:
        tags = result.payload.get("tags", [])
        print(f"   - Score: {result.score:.3f}, Tags: {tags}")
        print(f"     Text: {result.payload.get('text', '')[:80]}...")
    
    # Test 4: Search with project + tag filter
    print("\n4. Search with BOTH project AND tag filters:")
    
    search_filter = {
        "must": [
            {"key": "project_name", "match": {"value": DEFAULT_PROJECT_NAME}},
            {"key": "tags", "match": {"any": ["security", "critical"]}},
        ]
    }
    
    query_embedding = await generate_embedding("update")
    results = await perform_qdrant_search(
        query_embedding=query_embedding,
        search_filter=search_filter,
        limit=10,
        min_score=0.5,
        should_use_decay=False,
    )
    
    print(f"   Found {len(results)} results")
    for result in results:
        tags = result.payload.get("tags", [])
        project = result.payload.get("project_name", "unknown")
        print(f"   - Score: {result.score:.3f}, Project: {project}, Tags: {tags}")
        print(f"     Text: {result.payload.get('text', '')[:80]}...")


async def main():
    """Main test function."""
    print("Starting tags filtering test...")
    print(f"Using collection: {MAIN_COLLECTION}")
    print(f"Default project: {DEFAULT_PROJECT_NAME}")
    
    # Create test data
    num_created = await create_test_reflections()
    
    # Run search tests
    await test_search_with_tags()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)
    print("\nNote: Test reflections remain in the database for manual testing.")
    print("You can now test via MCP tools with commands like:")
    print('  mcp__claude-self-reflect__reflect_on_past("fix", tags=["bug"])')
    print('  mcp__claude-self-reflect__reflect_on_past("improvement", tags=["performance", "feature"])')


if __name__ == "__main__":
    asyncio.run(main())