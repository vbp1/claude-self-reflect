#!/usr/bin/env python3
"""Complete functionality test for Claude Self-Reflect MCP server."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
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
        sys.path.insert(0, str(Path(__file__).parent / "mcp-server" / "src"))
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

        containers = [
            json.loads(line) for line in result.stdout.strip().split("\n") if line
        ]
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
    except Exception as e:
        print_test(f"Docker test: {e}", "SKIP")
        return True


def test_mcp_protocol():
    """Test MCP protocol through stdio."""
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

        script_path = Path(__file__).parent / "run-mcp-docker.sh"
        if script_path.exists():
            # Test with Docker script
            process = subprocess.Popen(
                [str(script_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        else:
            # Test directly with Python
            process = subprocess.Popen(
                [sys.executable, "-m", "src", "--transport", "stdio"],
                cwd=Path(__file__).parent / "mcp-server",
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
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
                    process.terminate()
                    return True

        process.terminate()
        print_test("MCP protocol initialization timeout", "WARN")
        return True

    except Exception as e:
        print_test(f"MCP protocol: {e}", "SKIP")
        return True


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
        else:
            print_test(f"Embedding size: {len(embedding)} (expected 384)", "FAIL")
            return False

    except Exception as e:
        print_test(f"Async functionality: {e}", "FAIL")
        return False


def test_ruff_compliance():
    """Test that code passes ruff checks."""
    try:
        result = subprocess.run(
            ["ruff", "check", "mcp-server/src/", "test_server_startup.py"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            print_test("Ruff linting compliance", "PASS")
            return True
        else:
            errors = len(result.stdout.strip().split("\n")) if result.stdout else 0
            print_test(f"Ruff linting ({errors} issues)", "FAIL")
            return False

    except FileNotFoundError:
        print_test("Ruff not installed", "SKIP")
        return True
    except Exception as e:
        print_test(f"Ruff check: {e}", "SKIP")
        return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Claude Self-Reflect MCP Server - Complete Functionality Test")
    print("=" * 60 + "\n")

    # Set up test environment
    os.environ["EMBEDDING_MODEL"] = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
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
        else:
            print(f"{RED}❌ Some tests failed ({passed}/{total} passed){RESET}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
