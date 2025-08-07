#!/bin/bash
# Docker wrapper for MCP server - ensures container is running and executes MCP via stdio

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
    if [ $i -eq 30 ]; then
        echo "Timeout waiting for MCP server container to be ready" >&2
        exit 1
    fi
    sleep 1
done

# Execute MCP server in the running container via stdio
exec docker exec -i claude-reflection-mcp python -m src