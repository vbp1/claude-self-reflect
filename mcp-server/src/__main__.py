"""Main entry point for claude-reflect MCP server."""

import argparse
import asyncio


async def run_server_with_background_init(transport: str):
    """Run server with model initialization in background."""
    from .server import mcp, start_model_initialization

    # Start model initialization in background (non-blocking)
    await start_model_initialization()

    # Run the server immediately (model will initialize in background)
    # The server will handle waiting for model when needed
    await mcp.run_async(transport=transport, show_banner=False)


def main():
    """Main entry point for the claude-reflect script."""
    # Parse the command-line arguments to determine the transport protocol.
    parser = argparse.ArgumentParser(description="claude-reflect")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
    )
    args = parser.parse_args()

    # Run server with background initialization
    asyncio.run(run_server_with_background_init(args.transport))


if __name__ == "__main__":
    main()
