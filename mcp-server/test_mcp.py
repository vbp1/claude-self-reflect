#!/usr/bin/env python3
"""Test MCP server functionality."""

import asyncio
import json
from src.server import mcp

async def test_tools():
    """Test that tools are registered."""
    print(f"Server name: {mcp.name}")
    print(f"Server has {len(mcp._tools)} tools")
    
    # Test if we can manually call the function
    from src.server import ClaudeReflectServer
    server = ClaudeReflectServer()
    
    # Test manual search
    try:
        collections = await server.get_local_collections()
        print(f"Found {len(collections)} local collections")
        
        if collections:
            # Test without decay
            print("\nTesting without decay (manual)...")
            embedding = await server.generate_embedding("test query")
            print(f"Generated embedding with {len(embedding)} dimensions")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_tools())