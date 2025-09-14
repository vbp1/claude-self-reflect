#!/bin/bash
# Run the Python MCP server using FastMCP

# Fail fast
set -euo pipefail

# This script now reads configuration from the environment instead of -e flags.
# Example usage:
#   QDRANT_URL="http://localhost:6333" PROJECT_ID="xagent" ./run-mcp.sh

# Determine whether PROJECT_ID was explicitly provided; if so, do NOT set MCP_CLIENT_CWD
if [[ -n "${PROJECT_ID:-}" ]]; then
  echo "[run-mcp] PROJECT_ID provided; not setting MCP_CLIENT_CWD" 1>&2
else
  if [[ -z "${MCP_CLIENT_CWD:-}" ]]; then
    if root=$(git -C "$(pwd)" rev-parse --show-toplevel 2>/dev/null); then
      export MCP_CLIENT_CWD="${root}"
    else
      MCP_CLIENT_CWD="$(pwd)"
      export MCP_CLIENT_CWD
    fi
  fi
  echo "[run-mcp] PROJECT_ID not set; using MCP_CLIENT_CWD=$(printf '%q' "${MCP_CLIENT_CWD}")" 1>&2
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the mcp-server directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    # shellcheck disable=SC1091
    source venv/bin/activate
    pip install -e .
else
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

# Log selected project identifier for visibility
if [[ -n "${PROJECT_ID:-}" ]]; then
  echo "[run-mcp] Starting with PROJECT_ID='${PROJECT_ID}'" 1>&2
else
  echo "[run-mcp] Starting with MCP_CLIENT_CWD='${MCP_CLIENT_CWD}'" 1>&2
fi

# Set cache directory for models
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HOME/.cache/huggingface}"

# Run the MCP server
exec env PYTHONUNBUFFERED=1 python -m src