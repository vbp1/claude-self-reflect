#!/usr/bin/env python3
"""
Comprehensive validation tool for Claude Self-Reflection setup.
Checks all prerequisites, connections, and configurations.
"""

import os
import sys
import json
import time
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests

# Check for required imports
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import CollectionInfo
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_LOGS_DIR = os.path.expanduser("~/.claude/projects")
MCP_CONFIG_PATH = os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json")

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")

def print_status(name: str, status: bool, message: str = ""):
    """Print a status line with color coding."""
    icon = f"{Colors.GREEN}✅{Colors.END}" if status else f"{Colors.RED}❌{Colors.END}"
    status_text = f"{Colors.GREEN}PASS{Colors.END}" if status else f"{Colors.RED}FAIL{Colors.END}"
    print(f"{icon} {name:<30} [{status_text}] {message}")

def print_warning(name: str, message: str):
    """Print a warning line."""
    print(f"{Colors.YELLOW}⚠️{Colors.END}  {name:<30} {Colors.YELLOW}[WARN]{Colors.END} {message}")

def check_environment_variables() -> Dict[str, bool]:
    """Check required environment variables."""
    print_header("Environment Variables")
    
    results = {}
    
    # Check API keys (optional for local embeddings)
    if OPENAI_API_KEY:
        print_status("OpenAI API Key", True, "Found (optional)")
        results["openai_key"] = True
    else:
        print_status("API Keys", True, "Using local embeddings (FastEmbed)")
        results["local_embeddings"] = True
    
    # Check Qdrant URL
    if QDRANT_URL:
        print_status("Qdrant URL", True, QDRANT_URL)
        results["qdrant_url"] = True
    else:
        print_status("Qdrant URL", False, "Not configured")
        results["qdrant_url"] = False
    
    return results

def check_python_dependencies() -> Dict[str, bool]:
    """Check required Python packages."""
    print_header("Python Dependencies")
    
    results = {}
    required_packages = {
        "qdrant-client": "Qdrant client library",
        "requests": "HTTP requests",
        "tqdm": "Progress bars",
        "humanize": "Human-readable output",
        "backoff": "Retry logic"
    }
    
    for package, description in required_packages.items():
        try:
            __import__(package.replace("-", "_"))
            print_status(f"{package}", True, description)
            results[package] = True
        except ImportError:
            print_status(f"{package}", False, f"Missing - install with: pip install {package}")
            results[package] = False
    
    return results

def check_docker() -> Dict[str, bool]:
    """Check Docker installation and running containers."""
    print_header("Docker Status")
    
    results = {}
    
    # Check if Docker is installed
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
        print_status("Docker", True, "Installed")
        results["docker_installed"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_status("Docker", False, "Not installed or not in PATH")
        results["docker_installed"] = False
        return results
    
    # Check if Docker daemon is running
    try:
        subprocess.run(["docker", "ps"], capture_output=True, check=True)
        print_status("Docker Daemon", True, "Running")
        results["docker_running"] = True
    except subprocess.CalledProcessError:
        print_status("Docker Daemon", False, "Not running - start Docker Desktop")
        results["docker_running"] = False
        return results
    
    # Check for running containers
    try:
        result = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], 
                              capture_output=True, text=True, check=True)
        containers = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        # Check for Qdrant container
        qdrant_running = any('qdrant' in c.lower() for c in containers)
        if qdrant_running:
            print_status("Qdrant Container", True, "Running")
            results["qdrant_container"] = True
        else:
            print_warning("Qdrant Container", "Not running - run: docker compose up -d")
            results["qdrant_container"] = False
            
    except subprocess.CalledProcessError:
        print_status("Container Check", False, "Failed to list containers")
        results["container_check"] = False
    
    return results

def check_qdrant_connection() -> Dict[str, bool]:
    """Check Qdrant database connection and collections."""
    print_header("Qdrant Database")
    
    results = {}
    
    if not QDRANT_AVAILABLE:
        print_status("Qdrant Client", False, "Library not installed")
        results["qdrant_client"] = False
        return results
    
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=5)
        
        # Test connection
        collections = client.get_collections()
        print_status("Qdrant Connection", True, f"Connected to {QDRANT_URL}")
        results["qdrant_connection"] = True
        
        # Check collections
        local_collections = [c for c in collections.collections if c.name.endswith("_local")]
        if local_collections:
            total_vectors = 0
            for col in local_collections:
                info = client.get_collection(col.name)
                total_vectors += info.points_count
            
            print_status("Collections", True, 
                        f"{len(local_collections)} collections, {total_vectors:,} vectors total")
            results["collections"] = True
        else:
            print_warning("Collections", "No collections found - run import first")
            results["collections"] = False
            
    except Exception as e:
        print_status("Qdrant Connection", False, f"Failed: {str(e)}")
        results["qdrant_connection"] = False
    
    return results

def check_claude_logs() -> Dict[str, bool]:
    """Check Claude conversation logs."""
    print_header("Claude Conversation Logs")
    
    results = {}
    
    if not os.path.exists(CLAUDE_LOGS_DIR):
        print_status("Claude Logs Directory", False, f"Not found: {CLAUDE_LOGS_DIR}")
        results["logs_dir"] = False
        return results
    
    print_status("Claude Logs Directory", True, CLAUDE_LOGS_DIR)
    results["logs_dir"] = True
    
    # Count projects and files
    projects = []
    total_files = 0
    total_size = 0
    
    try:
        for project in os.listdir(CLAUDE_LOGS_DIR):
            project_path = os.path.join(CLAUDE_LOGS_DIR, project)
            if os.path.isdir(project_path) and not project.startswith('.'):
                jsonl_files = [f for f in os.listdir(project_path) if f.endswith('.jsonl')]
                if jsonl_files:
                    projects.append(project)
                    total_files += len(jsonl_files)
                    for f in jsonl_files:
                        total_size += os.path.getsize(os.path.join(project_path, f))
        
        if projects:
            size_mb = total_size / (1024 * 1024)
            print_status("Conversation Files", True, 
                        f"{len(projects)} projects, {total_files} files, {size_mb:.1f} MB")
            results["conversation_files"] = True
            
            # Show sample projects
            print(f"\n  Sample projects:")
            for project in projects[:5]:
                print(f"    • {project}")
            if len(projects) > 5:
                print(f"    • ... and {len(projects) - 5} more")
        else:
            print_warning("Conversation Files", "No conversation files found")
            results["conversation_files"] = False
            
    except Exception as e:
        print_status("Log Scan", False, f"Error scanning logs: {e}")
        results["log_scan"] = False
    
    return results

def check_mcp_configuration() -> Dict[str, bool]:
    """Check MCP server configuration in Claude Desktop."""
    print_header("Claude Desktop MCP Configuration")
    
    results = {}
    
    if not os.path.exists(MCP_CONFIG_PATH):
        print_warning("MCP Config", f"Not found - Claude Desktop may not be installed")
        results["mcp_config"] = False
        return results
    
    try:
        with open(MCP_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        print_status("MCP Config File", True, "Found")
        results["mcp_config_file"] = True
        
        # Check for our MCP server
        mcp_servers = config.get("mcpServers", {})
        reflection_configured = any(
            'reflection' in name.lower() or 'qdrant' in name.lower() 
            for name in mcp_servers.keys()
        )
        
        if reflection_configured:
            print_status("Self-Reflection MCP", True, "Configured")
            results["mcp_configured"] = True
        else:
            print_warning("Self-Reflection MCP", "Not configured - run install.sh")
            results["mcp_configured"] = False
            
    except Exception as e:
        print_status("MCP Config", False, f"Error reading config: {e}")
        results["mcp_config"] = False
    
    return results

def test_api_connection() -> Dict[str, bool]:
    """Test API connections for embeddings."""
    print_header("API Connectivity")
    
    results = {}
    
    # Test local embeddings (FastEmbed)
    try:
        from fastembed import TextEmbedding
        embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = list(embedding_model.embed(["test"]))
        if embeddings and len(embeddings[0]) == 384:
            print_status("FastEmbed (Local)", True, "384-dimensional embeddings working")
            results["local_embeddings"] = True
        else:
            print_status("FastEmbed (Local)", False, "Unexpected embedding format")
            results["local_embeddings"] = False
    except Exception as e:
        print_status("FastEmbed (Local)", False, f"Failed: {e}")
        results["local_embeddings"] = False
    
    if OPENAI_API_KEY:
        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": ["test"],
                    "model": "text-embedding-3-small"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                print_status("OpenAI API", True, "Connected and authenticated")
                results["openai_api"] = True
            else:
                print_status("OpenAI API", False, f"Error {response.status_code}")
                results["openai_api"] = False
                
        except Exception as e:
            print_status("OpenAI API", False, f"Connection failed: {e}")
            results["openai_api"] = False
    else:
        print_warning("API Test", "No API key configured")
        results["api_test"] = False
    
    return results

def check_disk_space() -> Dict[str, bool]:
    """Check available disk space."""
    print_header("System Resources")
    
    results = {}
    
    try:
        stat = shutil.disk_usage("/")
        free_gb = stat.free / (1024 ** 3)
        total_gb = stat.total / (1024 ** 3)
        used_percent = ((stat.total - stat.free) / stat.total) * 100
        
        if free_gb > 5:
            print_status("Disk Space", True, 
                        f"{free_gb:.1f} GB free of {total_gb:.1f} GB ({used_percent:.1f}% used)")
            results["disk_space"] = True
        elif free_gb > 1:
            print_warning("Disk Space", 
                         f"Low: {free_gb:.1f} GB free ({used_percent:.1f}% used)")
            results["disk_space"] = True
        else:
            print_status("Disk Space", False, 
                        f"Very low: {free_gb:.1f} GB free ({used_percent:.1f}% used)")
            results["disk_space"] = False
            
    except Exception as e:
        print_status("Disk Space", False, f"Could not check: {e}")
        results["disk_space"] = False
    
    # Check memory (if psutil available)
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        total_gb = memory.total / (1024 ** 3)
        
        if available_gb > 2:
            print_status("Memory", True, 
                        f"{available_gb:.1f} GB available of {total_gb:.1f} GB")
            results["memory"] = True
        else:
            print_warning("Memory", 
                         f"Low: {available_gb:.1f} GB available of {total_gb:.1f} GB")
            results["memory"] = True
    except ImportError:
        print_warning("Memory", "psutil not installed - cannot check memory")
        results["memory"] = None
    
    return results

def generate_report(all_results: Dict[str, Dict[str, bool]]) -> Tuple[int, int, int]:
    """Generate final report and recommendations."""
    print_header("Validation Summary")
    
    # Count results
    passed = sum(1 for section in all_results.values() 
                 for result in section.values() if result is True)
    failed = sum(1 for section in all_results.values() 
                 for result in section.values() if result is False)
    warnings = sum(1 for section in all_results.values() 
                   for result in section.values() if result is None)
    
    total = passed + failed + warnings
    
    print(f"\n{Colors.BOLD}Results:{Colors.END}")
    print(f"  {Colors.GREEN}✅ Passed:{Colors.END} {passed}/{total}")
    print(f"  {Colors.RED}❌ Failed:{Colors.END} {failed}/{total}")
    print(f"  {Colors.YELLOW}⚠️  Warnings:{Colors.END} {warnings}/{total}")
    
    # Overall status
    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✨ System is ready for use!{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  Some issues need to be resolved.{Colors.END}")
    
    # Recommendations
    print(f"\n{Colors.BOLD}Recommendations:{Colors.END}")
    
    if not all_results.get("environment", {}).get("local_embeddings", True):
        print(f"  1. FastEmbed should work out of the box (no API key needed)")
        print(f"     Optionally set OpenAI key:")
        print(f"     export OPENAI_API_KEY='your-openai-api-key'")
    
    if not all_results.get("docker", {}).get("qdrant_container", True):
        print(f"  2. Start Qdrant database:")
        print(f"     docker compose up -d")
    
    if not all_results.get("mcp", {}).get("mcp_configured", True):
        print(f"  3. Configure Claude Desktop:")
        print(f"     ./install.sh --configure-claude")
    
    if not all_results.get("qdrant", {}).get("collections", True):
        print(f"  4. Import your conversations:")
        print(f"     python scripts/import-conversations-unified.py")
    
    return passed, failed, warnings

def main():
    """Run all validation checks."""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("🔍 Claude-Self-Reflect (CSR) Setup Validator")
    print("=" * 60)
    print(f"{Colors.END}")
    
    all_results = {}
    
    # Run all checks
    all_results["environment"] = check_environment_variables()
    all_results["python"] = check_python_dependencies()
    all_results["docker"] = check_docker()
    all_results["qdrant"] = check_qdrant_connection()
    all_results["logs"] = check_claude_logs()
    all_results["mcp"] = check_mcp_configuration()
    all_results["api"] = test_api_connection()
    all_results["system"] = check_disk_space()
    
    # Generate report
    passed, failed, warnings = generate_report(all_results)
    
    # Exit code based on results
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()