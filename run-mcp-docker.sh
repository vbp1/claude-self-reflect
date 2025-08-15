#!/bin/bash
# Docker wrapper for MCP server - ensures container is running and executes MCP via stdio

# Fail fast on errors and unset vars
set -euo pipefail

# This script now reads configuration from the environment instead of -e flags.
# Example usage:
#   QDRANT_URL="http://qdrant:6333" PROJECT_ID="xagent" ./run-mcp-docker.sh

# Determine whether PROJECT_ID was explicitly provided; if so, do NOT set MCP_CLIENT_CWD
if [[ -n "${PROJECT_ID:-}" ]]; then
  echo "[run-mcp-docker] PROJECT_ID provided; not setting MCP_CLIENT_CWD" 1>&2
else
  if [[ -z "${MCP_CLIENT_CWD:-}" ]]; then
    if root=$(git -C "$(pwd)" rev-parse --show-toplevel 2>/dev/null); then
      export MCP_CLIENT_CWD="${root}"
    else
      MCP_CLIENT_CWD="$(pwd)"
      export MCP_CLIENT_CWD
    fi
  fi
  echo "[run-mcp-docker] PROJECT_ID not set; using MCP_CLIENT_CWD=$(printf '%q' "${MCP_CLIENT_CWD}")" 1>&2
fi

# Get the directory of this script
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# Ensure MCP server container is running
docker compose -f "$SCRIPT_DIR/docker-compose.yaml" --profile mcp up -d mcp-server

# Wait for container to be ready by checking if Python module can be imported
echo "Waiting for MCP server container to be ready..." 1>&2
for i in {1..30}; do
    if docker exec claude-reflection-mcp python -c "import src" 2>/dev/null; then
        echo "MCP server container is ready" 1>&2
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "Timeout waiting for MCP server container to be ready" 1>&2
        exit 1
    fi
    sleep 1
done

# Execute MCP server in the running container via stdio
EXEC_CMD=(docker exec -i)

# Select environment variables to forward into the container
VARS_TO_FWD=(
  PROJECT_ID
  MCP_CLIENT_CWD
  QDRANT_URL
  LOG_LEVEL
  LOG_FILE
  EMBEDDING_MODEL
  VECTOR_SIZE
  ENABLE_MEMORY_DECAY
  DECAY_WEIGHT
  DECAY_SCALE_DAYS
  MODEL_CACHE_DAYS
  TRANSFORMERS_CACHE
  TRANSFORMERS_OFFLINE
  HF_HUB_OFFLINE
)

for var in "${VARS_TO_FWD[@]}"; do
  if [[ -n "${!var:-}" ]]; then
    EXEC_CMD+=("-e" "${var}=${!var}")
  fi
done

# Always ensure unbuffered output for timely logs
EXEC_CMD+=("-e" "PYTHONUNBUFFERED=1")

# Command to run inside the container
EXEC_CMD+=(claude-reflection-mcp python -m src)

exec "${EXEC_CMD[@]}"
