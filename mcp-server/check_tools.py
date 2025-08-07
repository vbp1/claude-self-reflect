#!/usr/bin/env python3
"""Check registered MCP tools."""

import asyncio
import sys
sys.path.insert(0, '.')

async def check_tools():
    from src.server import mcp
    tools = await mcp.get_tools()
    print(f"Tools type: {type(tools)}")
    print(f"Tools: {tools}")
    if hasattr(tools, '__iter__'):
        for t in tools:
            print(f"Tool: {t}, type: {type(t)}")
    
if __name__ == "__main__":
    asyncio.run(check_tools())