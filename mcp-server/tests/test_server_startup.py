"""Test that model initialization starts immediately when server starts."""

import asyncio
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# Set up environment
os.environ["EMBEDDING_MODEL"] = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
os.environ["VECTOR_SIZE"] = "384"
os.environ["QDRANT_URL"] = "http://localhost:6333"

# Use temporary directories instead of hardcoded paths
with tempfile.TemporaryDirectory() as tmpdir:
    os.environ["MCP_CLIENT_CWD"] = tmpdir
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(tmpdir, ".cache", "huggingface")

# Add the server path to sys.path relative to this file
# We're now in mcp-server/tests/, so we need to go up one level to find src/
TESTS_DIR = Path(__file__).resolve().parent
MCP_SERVER_DIR = TESTS_DIR.parent
SERVER_PATH = MCP_SERVER_DIR / "src"
sys.path.insert(0, str(SERVER_PATH))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_startup_initialization():
    """Test that initialization starts at server startup, not at first request."""

    print("=== Testing Server Startup Initialization ===")

    # Import the server module - this should trigger background initialization
    start_time = time.time()
    print(f"Importing server module at: {start_time:.3f}")

    import server

    import_time = time.time()
    print(f"Server module imported in: {import_time - start_time:.3f} seconds")

    # Check the state immediately after import
    print("\nState after import:")
    print(f"  local_embedding_model: {server.local_embedding_model is not None}")
    print(f"  model_ready event: {server.model_ready}")
    print(
        f"  model_initialization_task: {server.model_initialization_task is not None}"
    )

    # Now create the server which should start initialization
    print("\nCreating server (this starts model initialization)...")
    create_start = time.time()
    await server.create_server()
    create_end = time.time()
    print(f"Server created in: {create_end - create_start:.3f} seconds")

    if server.model_initialization_task is None:
        print(
            "❌ ERROR: model_initialization_task was not created - initialization not started!"
        )
        return False

    print("✅ Model initialization was started at server startup")

    # Check that initialization task is actually running
    if server.model_initialization_task.done():
        # If it's done this quickly, it either failed or there was no actual work
        try:
            _ = server.model_initialization_task.result()
            print(
                "⚠️  WARNING: Initialization task completed immediately (possibly cached model)"
            )
        except (ImportError, RuntimeError, ValueError) as e:
            print(f"❌ ERROR: Initialization task failed: {e}")
            return False
    else:
        print("✅ Initialization task is running in background")

    # Wait a bit to see if initialization progresses
    print("\nWaiting 5 seconds to check initialization progress...")
    await asyncio.sleep(5)

    if server.local_embedding_model is not None:
        print("✅ Model initialization completed during startup period")
    else:
        print("⏳ Model still initializing (normal for slow downloads)")

    # Now test that first call works correctly
    print("\nTesting first embedding call...")
    call_start = time.time()

    try:
        embedding = await server.generate_embedding("test")
        call_end = time.time()

        print(
            f"✅ First embedding call succeeded in: {call_end - call_start:.3f} seconds"
        )
        print(f"✅ Embedding dimensions: {len(embedding)}")

        # Second call should be very fast
        second_start = time.time()
        _ = await server.generate_embedding("test2")
        second_end = time.time()

        print(f"✅ Second call took: {second_end - second_start:.3f} seconds")

        return True

    except (RuntimeError, ValueError, ImportError) as e:
        print(f"❌ ERROR: First embedding call failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    try:
        success = await test_startup_initialization()
        if success:
            print("\n✅ All startup tests passed!")
            return 0
        print("\n❌ Startup tests failed!")
        return 1
    except (RuntimeError, ValueError, ImportError, KeyboardInterrupt) as e:
        print(f"\n💥 Test crashed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
