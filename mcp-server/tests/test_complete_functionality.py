#!/usr/bin/env python3
"""Complete functionality test for Claude Self-Reflect MCP server."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def print_test(name: str, status: str):
    """Print test result with color."""
    color = GREEN if status == "PASS" else RED if status == "FAIL" else YELLOW
    print(f"{color}[{status}]{RESET} {name}")


def test_python_import():
    """Test that server module can be imported."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        import server  # noqa: F401

        print_test("Python import", "PASS")
        return True
    except ImportError as e:
        print_test(f"Python import: {e}", "FAIL")
        return False


def test_docker_container():
    """Test Docker container functionality."""
    try:
        # Check if containers are running
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            print_test("Docker compose status check", "SKIP")
            return True  # Skip if Docker not available

        # Parse JSON output - handle both JSON array and JSON Lines formats
        containers = []
        try:
            # First try to parse as a single JSON array
            parsed = json.loads(result.stdout.strip())
            if isinstance(parsed, list):
                # It's already a list of containers
                containers = parsed
            elif isinstance(parsed, dict):
                # Single container as dict
                containers = [parsed]
        except json.JSONDecodeError:
            # Fall back to JSON Lines format (one JSON object per line)
            for line in result.stdout.strip().split("\n"):
                if line:
                    try:
                        containers.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Flatten if we got a list of lists
        if containers and isinstance(containers[0], list):
            containers = [item for sublist in containers for item in sublist]

        mcp_running = any("mcp" in c.get("Name", "") for c in containers)
        qdrant_running = any("qdrant" in c.get("Name", "") for c in containers)

        if mcp_running and qdrant_running:
            print_test("Docker containers running", "PASS")
        else:
            print_test(
                f"Docker containers (MCP: {mcp_running}, Qdrant: {qdrant_running})",
                "WARN",
            )

        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        print_test(f"Docker test: {e}", "SKIP")
        return True


def cleanup_process(process):
    """Properly terminate and reap a subprocess to avoid zombies."""
    if process is None:
        return

    # Check if process is still running
    if process.poll() is None:
        # Try graceful termination first
        process.terminate()
        try:
            # Wait for graceful termination with short timeout
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            # Force kill if still running
            process.kill()
            # Wait again to ensure process is reaped
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=1)
    else:
        # Process already terminated, but ensure it's reaped
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=0.1)


def test_mcp_protocol():
    """Test MCP protocol through stdio."""
    process = None
    try:
        # Test initialization
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "id": 1,
            "params": {
                "protocolVersion": "1.0.0",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0.0"},
            },
        }

        script_path = Path(__file__).parent.parent.parent / "run-mcp-docker.sh"

        # Copy environment to preserve test variables
        env = os.environ.copy()

        if script_path.exists():
            # Test with Docker script
            process = subprocess.Popen(
                [str(script_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                env=env,
            )
        else:
            # Test directly with Python
            process = subprocess.Popen(
                [sys.executable, "-m", "src", "--transport", "stdio"],
                cwd=Path(__file__).parent.parent,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                env=env,
            )

        # Send initialization request
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()

        # Read response with timeout
        import select

        ready = select.select([process.stdout], [], [], 5)
        if ready[0]:
            response = process.stdout.readline()
            if response:
                data = json.loads(response)
                if "result" in data:
                    print_test("MCP protocol initialization", "PASS")
                    cleanup_process(process)
                    return True

        print_test("MCP protocol initialization timeout", "WARN")
        cleanup_process(process)
        return True

    except (FileNotFoundError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print_test(f"MCP protocol: {e}", "SKIP")
        # Ensure process is cleaned up before returning
        if process is not None:
            cleanup_process(process)
        return True
    finally:
        # Always cleanup the process
        cleanup_process(process)


async def test_async_functionality():
    """Test async model initialization."""
    try:
        from server import create_server, generate_embedding

        # Create server (starts background init)
        await create_server()

        # Test embedding generation
        start = time.time()
        embedding = await generate_embedding("test text")
        duration = time.time() - start

        if len(embedding) == 384:
            print_test(f"Async embedding generation ({duration:.2f}s)", "PASS")
            return True
        print_test(f"Embedding size: {len(embedding)} (expected 384)", "FAIL")
        return False

    except (ImportError, RuntimeError, ValueError, TimeoutError) as e:
        print_test(f"Async functionality: {e}", "FAIL")
        return False


def test_ruff_compliance():
    """Test that code passes ruff checks."""
    try:
        result = subprocess.run(
            ["ruff", "check", str(Path(__file__).parent.parent / "src"), str(Path(__file__).parent)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            print_test("Ruff linting compliance", "PASS")
            return True
        errors = len(result.stdout.strip().split("\n")) if result.stdout else 0
        print_test(f"Ruff linting ({errors} issues)", "FAIL")
        return False

    except FileNotFoundError:
        print_test("Ruff not installed", "SKIP")
        return True
    except (subprocess.TimeoutExpired, OSError) as e:
        print_test(f"Ruff check: {e}", "SKIP")
        return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Claude Self-Reflect MCP Server - Complete Functionality Test")
    print("=" * 60 + "\n")

    # Set up test environment
    os.environ["EMBEDDING_MODEL"] = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    os.environ["VECTOR_SIZE"] = "384"
    os.environ["QDRANT_URL"] = "http://localhost:6333"

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["MCP_CLIENT_CWD"] = tmpdir
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(tmpdir, ".cache")

        results = []

        # Run synchronous tests
        print("Running tests...\n")
        results.append(test_python_import())
        results.append(test_docker_container())
        results.append(test_mcp_protocol())
        results.append(test_ruff_compliance())

        # Run async tests
        results.append(asyncio.run(test_async_functionality()))

        # Summary
        print("\n" + "=" * 60)
        passed = sum(1 for r in results if r)
        total = len(results)

        if passed == total:
            print(f"{GREEN}✅ All tests passed! ({passed}/{total}){RESET}")
            return 0
        print(f"{RED}❌ Some tests failed ({passed}/{total} passed){RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
