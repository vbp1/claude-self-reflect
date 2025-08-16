#!/bin/bash
# Test script for MCP server search functionality via Docker
# Usage: ./test-mcp-search.sh --query "search query" [options]

set -euo pipefail

# Default values
QUERY=""
LIMIT=5
MIN_SCORE=0.7
PROJECT=""
TAGS=""
USE_DECAY=-1
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --query|-q)
            QUERY="$2"
            shift 2
            ;;
        --limit|-l)
            LIMIT="$2"
            shift 2
            ;;
        --min-score|-s)
            MIN_SCORE="$2"
            shift 2
            ;;
        --project|-p)
            PROJECT="$2"
            shift 2
            ;;
        --tags|-t)
            TAGS="$2"
            shift 2
            ;;
        --use-decay|-d)
            USE_DECAY="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 --query \"search query\" [options]"
            echo "Options:"
            echo "  --query, -q       Search query (required)"
            echo "  --limit, -l       Maximum number of results (default: 5)"
            echo "  --min-score, -s   Minimum similarity score 0-1 (default: 0.7)"
            echo "  --project, -p     Search specific project only (default: current project)"
            echo "  --tags, -t        Comma-separated tags to filter by"
            echo "  --use-decay, -d   Time decay: 1=enable, 0=disable, -1=default"
            echo "  --verbose, -v     Show detailed output"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required parameter
if [ -z "$QUERY" ]; then
    echo "Error: --query is required"
    echo "Use --help for usage information"
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# Ensure MCP server container is running
if [ "$VERBOSE" = true ]; then
    echo "Starting MCP server container..." >&2
fi
docker compose -f "$SCRIPT_DIR/docker-compose.yaml" --profile mcp up -d mcp-server >&2

# Wait for container to be ready
if [ "$VERBOSE" = true ]; then
    echo "Waiting for MCP server to be ready..." >&2
fi
for i in {1..30}; do
    if docker exec claude-reflection-mcp python -c "import src" 2>/dev/null; then
        if [ "$VERBOSE" = true ]; then
            echo "MCP server is ready" >&2
        fi
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Timeout waiting for MCP server" >&2
        exit 1
    fi
    sleep 1
done

# Build the JSON request for reflect_on_past
build_json_request() {
    # Build arguments object first, then the full request
    local args_json
    args_json=$(jq -n \
        --arg query "$QUERY" \
        --argjson limit "$LIMIT" \
        --argjson min_score "$MIN_SCORE" \
        --arg project "$PROJECT" \
        --arg tags "$TAGS" \
        --argjson use_decay "$USE_DECAY" \
        '{
            query: $query
        } |
        if $limit != 5 then . + {limit: $limit} else . end |
        if $min_score != 0.7 then . + {min_score: $min_score} else . end |
        if $project != "" then . + {project: $project} else . end |
        if $tags != "" then . + {tags: $tags} else . end |
        if $use_decay != -1 then . + {use_decay: ($use_decay | tonumber | . == 1)} else . end'
    )
    
    # Build the complete JSON-RPC request
    jq -n \
        --argjson arguments "$args_json" \
        '{
            jsonrpc: "2.0",
            id: 1,
            method: "tools/call",
            params: {
                name: "reflect_on_past",
                arguments: $arguments
            }
        }'
}

# Build the request
REQUEST=$(build_json_request)

if [ "$VERBOSE" = true ]; then
    echo "Sending request:" >&2
    echo "$REQUEST" | jq . >&2
fi

# Determine MCP_CLIENT_CWD for project scoping
if [ -n "$PROJECT" ]; then
    # If specific project provided, use it as-is
    MCP_CLIENT_CWD="$PROJECT"
else
    # Use git root or current directory
    if root=$(git -C "$(pwd)" rev-parse --show-toplevel 2>/dev/null); then
        MCP_CLIENT_CWD="$root"
    else
        MCP_CLIENT_CWD="$(pwd)"
    fi
fi

# Send request to MCP server and process response
RESPONSE=$(echo "$REQUEST" | docker exec -i \
    -e "QDRANT_URL=http://qdrant:6333" \
    -e "MCP_CLIENT_CWD=$MCP_CLIENT_CWD" \
    -e "PYTHONUNBUFFERED=1" \
    claude-reflection-mcp python -c "
import sys
import json
import asyncio
from src.server import reflect_on_past
from fastmcp import Context

async def main():
    # Read the JSON-RPC request
    request = json.loads(sys.stdin.read())
    args = request['params']['arguments']
    
    try:
        # Create a mock context object
        class MockContext:
            async def error(self, msg):
                print(f\"Error: {msg}\", file=sys.stderr)
        
        ctx = MockContext()
        
        # Call the underlying function directly (reflect_on_past is a FunctionTool wrapper)
        result = await reflect_on_past.fn(
            ctx=ctx,
            query=args.get('query', ''),
            limit=args.get('limit', 5),
            min_score=args.get('min_score', 0.7),
            use_decay=args.get('use_decay', -1),
            project=args.get('project'),
            tags=args.get('tags')  # Pass tags as-is from the request
        )
        
        # Parse the result string to JSON if it's a formatted result
        if result.startswith('Found'):
            # The function returns formatted text, try to extract data
            result_text = result
        else:
            result_text = result
        
        # Format the response
        response = {
            'jsonrpc': '2.0',
            'id': request['id'],
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': result_text
                    }
                ]
            }
        }
        print(json.dumps(response))
    except Exception as e:
        error_response = {
            'jsonrpc': '2.0',
            'id': request['id'],
            'error': {
                'code': -32603,
                'message': str(e)
            }
        }
        print(json.dumps(error_response))

asyncio.run(main())
")

# Parse and display results
if [ "$VERBOSE" = true ]; then
    echo -e "\n=== Full Response ===" >&2
    echo "$RESPONSE" | jq . >&2
    echo -e "\n=== Search Results ===" >&2
fi

# Extract and format the results
echo "$RESPONSE" | jq -r '.result.content[0].text'