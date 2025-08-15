#!/bin/bash
# Docker wrapper for MCP server - ensures container is running and executes MCP via stdio

# Fail fast on errors and unset vars
set -euo pipefail

# Collect environment variables passed via -e/--env and forward them to docker exec
# Supports: `-e KEY=VALUE`, `-e KEY`, `--env KEY=VALUE`, `--env=KEY=VALUE`
ENV_LIST=()

while [[ ${#} -gt 0 ]]; do
  case "${1}" in
    -e|--env)
      # Next token should be VAR or VAR=VAL
      if [[ ${#} -lt 2 ]]; then
        echo "Missing value for ${1}" >&2
        exit 2
      fi
      ENV_LIST+=("${2}")
      shift 2
      ;;
    -e*)
      # Handle `-eVAR=VAL` form
      ENV_LIST+=("${1#-e}")
      shift 1
      ;;
    --env=*)
      # Handle `--env=VAR=VAL` form
      ENV_LIST+=("${1#--env=}")
      shift 1
      ;;
    --)
      shift 1
      break
      ;;
    *)
      # Ignore unknown args (the Claude CLI may pass only -e flags)
      shift 1
      ;;
  esac
done

# Determine whether PROJECT_ID was explicitly provided; if so, do NOT force MCP_CLIENT_CWD
HAS_PROJECT_ID=false
for kv in "${ENV_LIST[@]:-}"; do
  if [[ "$kv" == PROJECT_ID* || "$kv" == "PROJECT_ID" ]]; then
    HAS_PROJECT_ID=true
    break
  fi
done

# Ensure MCP_CLIENT_CWD is available only if PROJECT_ID is not provided
HAS_MCP_CWD=false
for kv in "${ENV_LIST[@]:-}"; do
  if [[ "$kv" == MCP_CLIENT_CWD* || "$kv" == "MCP_CLIENT_CWD" ]]; then
    HAS_MCP_CWD=true
    break
  fi
done
if [[ "$HAS_PROJECT_ID" == true ]]; then
  echo "[run-mcp-docker] PROJECT_ID provided; not setting MCP_CLIENT_CWD" 1>&2
else
  if [[ "$HAS_MCP_CWD" == false ]]; then
    if root=$(git -C "$(pwd)" rev-parse --show-toplevel 2>/dev/null); then
      ENV_LIST+=("MCP_CLIENT_CWD=$root")
    else
      ENV_LIST+=("MCP_CLIENT_CWD=$(pwd)")
    fi
  fi
  echo "[run-mcp-docker] PROJECT_ID not set; using MCP_CLIENT_CWD=$(printf '%q' "${ENV_LIST[-1]#MCP_CLIENT_CWD=}")" 1>&2
fi

# Get the directory of this script
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# Ensure MCP server container is running
docker compose -f "$SCRIPT_DIR/docker-compose.yaml" --profile mcp up -d mcp-server

# Wait for container to be ready by checking if Python module can be imported
echo "Waiting for MCP server container to be ready..." >&2
for i in {1..30}; do
    if docker exec claude-reflection-mcp python -c "import src" 2>/dev/null; then
        echo "MCP server container is ready" >&2
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "Timeout waiting for MCP server container to be ready" >&2
        exit 1
    fi
    sleep 1
done

# Execute MCP server in the running container via stdio
EXEC_CMD=(docker exec -i)

# Forward all collected env vars into container
for kv in "${ENV_LIST[@]:-}"; do
  EXEC_CMD+=("-e" "$kv")
done

# Always ensure unbuffered output for timely logs
EXEC_CMD+=("-e" "PYTHONUNBUFFERED=1")

EXEC_CMD+=(claude-reflection-mcp python -m src)

exec "${EXEC_CMD[@]}"