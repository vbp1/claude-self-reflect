#!/usr/bin/env python3
"""Test MCP server debug output directly."""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mcp-server', 'src'))

from server import reflect_on_past

class MockContext:
    """Mock context for testing."""
    
    async def debug(self, message: str):
        """Print debug messages."""
        print(f"[DEBUG] {message}")
    
    async def error(self, message: str):
        """Print error messages."""
        print(f"[ERROR] {message}")

async def test_search():
    """Test search with debug output."""
    ctx = MockContext()
    
    # Test 1: Search without working_directory (should use server's cwd)
    print("Test 1: Search without working_directory")
    print("=" * 60)
    result = await reflect_on_past(
        ctx=ctx,
        query="test database",
        limit=2,
        project=None,
        working_directory=None
    )
    print(f"\nResult:\n{result[:200]}...")
    
    # Test 2: Search with working_directory
    print("\n\nTest 2: Search with working_directory=/home/vbponomarev/xagent")
    print("=" * 60)
    result = await reflect_on_past(
        ctx=ctx,
        query="test database",
        limit=2,
        project=None,
        working_directory="/home/vbponomarev/xagent"
    )
    print(f"\nResult:\n{result[:200]}...")
    
    # Test 3: Search with project="all"
    print("\n\nTest 3: Search with project='all'")
    print("=" * 60)
    result = await reflect_on_past(
        ctx=ctx,
        query="test database",
        limit=2,
        project="all",
        working_directory=None
    )
    print(f"\nResult:\n{result[:200]}...")

if __name__ == "__main__":
    asyncio.run(test_search())