"""Main entry point for claude-reflect MCP server."""

import argparse
import asyncio
import os
import signal
from contextlib import suppress


async def run_server_with_background_init(transport: str) -> bool:
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

    # Install simple process-level handlers that also close stdin for stdio
    def _force_quit(_signum, _frame):
        with suppress(Exception):
            os.write(2, b"\nServer stopped by user.\n")
        os._exit(0)

    with suppress(Exception):
        signal.signal(signal.SIGTERM, _force_quit)
        signal.signal(signal.SIGINT, _force_quit)

    # Run server with background initialization
    asyncio.run(run_server_with_background_init(args.transport))


if __name__ == "__main__":
    main()
