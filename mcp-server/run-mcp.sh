#!/bin/bash
# Run the Python MCP server using FastMCP

# Fail fast
set -euo pipefail

# Collect environment variables passed via -e/--env and export them
# Supports forms: `-e KEY=VALUE`, `-e KEY`, `-eKEY=VALUE`, `--env KEY=VALUE`, `--env=KEY=VALUE`
ENV_LIST=()

while [[ ${#} -gt 0 ]]; do
  case "${1}" in
    -e|--env)
      if [[ ${#} -lt 2 ]]; then
        echo "Missing value for ${1}" >&2
        exit 2
      fi
      ENV_LIST+=("${2}")
      shift 2
      ;;
    -e*)
      ENV_LIST+=("${1#-e}")
      shift 1
      ;;
    --env=*)
      ENV_LIST+=("${1#--env=}")
      shift 1
      ;;
    --)
      shift 1
      break
      ;;
    *)
      # Ignore unknown args
      shift 1
      ;;
  esac
done

# Ensure MCP_CLIENT_CWD is set unless user passed it explicitly
HAS_MCP_CWD=false
for kv in "${ENV_LIST[@]:-}"; do
  if [[ "$kv" == MCP_CLIENT_CWD* || "$kv" == "MCP_CLIENT_CWD" ]]; then
    HAS_MCP_CWD=true
    break
  fi
done
if [[ "$HAS_MCP_CWD" == false ]]; then
  if root=$(git -C "$(pwd)" rev-parse --show-toplevel 2>/dev/null); then
    ENV_LIST+=("MCP_CLIENT_CWD=$root")
  else
    ENV_LIST+=("MCP_CLIENT_CWD=$(pwd)")
  fi
fi

# Export all collected envs
for kv in "${ENV_LIST[@]:-}"; do
  if [[ "$kv" == *"="* ]]; then
    export "$kv"
  else
    # Export existing variable as-is (may be empty if not set by caller)
    export "$kv"
  fi
done

# Export current working directory
export MCP_CLIENT_CWD=${MCP_CLIENT_CWD:-$(pwd)}

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the mcp-server directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -e .
else
    source venv/bin/activate
fi

# Run the MCP server
exec env PYTHONUNBUFFERED=1 python -m src