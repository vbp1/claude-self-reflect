#!/usr/bin/env python3
"""Test the claude-reflect MCP server."""

import asyncio
import sys
from src.server import mcp

async def test_server():
    """Test basic server functionality."""
    print("Testing claude-reflect MCP server...")
    
    # Test get_local_collections
    server = mcp
    try:
        collections = await server.get_local_collections()
        print(f"\nFound {len(collections)} local collections")
        if collections:
            print(f"First 3: {collections[:3]}")
        
        # Test embedding generation
        if collections:
            print("\nTesting embedding generation...")
            try:
                embedding = await server.generate_embedding("test query")
                print(f"Generated embedding with {len(embedding)} dimensions")
            except Exception as e:
                print(f"Embedding generation failed: {e}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nServer test complete!")

if __name__ == "__main__":
    asyncio.run(test_server())