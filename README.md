# Claude Self-Reflect

Claude forgets everything. This fixes that.

## Overview

Claude Self-Reflect gives Claude persistent, project-scoped memory via local embeddings and Qdrant search. It integrates with Claude Code through MCP (Model Context Protocol).

Core features:
- Semantic search over past conversations and reflections
- Project-aware memory (PROJECT_ID preferred; working directory as fallback)
- Optional time-based relevance decay
- 100% local embeddings (no external API calls)

## Components
- MCP server (Python, FastMCP) in mcp-server/
- Qdrant vector database

## Requirements
- Docker and Docker Compose (recommended), or Python 3.11+
- Claude Code (latest)
- 4GB+ RAM recommended for embeddings

## Quick Start (Docker)
```bash
# 1) Clone
git clone https://github.com/vbp1/claude-self-reflect.git
cd claude-self-reflect

# 2) Start services
docker compose up -d

# 3) Add MCP server to Claude Code (set your project id)
claude-self-reflect-path=$(pwd)
cd /path/to/your-project-dir
claude mcp add claude-self-reflect "$(claude-self-reflect-path)/run-mcp-docker.sh" \
  -e QDRANT_URL="http://localhost:6333" \
  -e PROJECT_ID="my-project-id"
```

## Configuration (env)
Minimal variables you may want to set:
- QDRANT_URL: URL of Qdrant API (e.g., http://localhost:6333)
- EMBEDDING_MODEL: embedding model name (required)
- VECTOR_SIZE: embedding dimension (required; must match model)
- ENABLE_MEMORY_DECAY, DECAY_WEIGHT, DECAY_SCALE_DAYS: decay configuration (optional)
- TRANSFORMERS_CACHE, MODEL_CACHE_DAYS: local model caching (optional)
- LOG_LEVEL, LOG_FILE: logging (optional)

Example .env entries:
```bash
QDRANT_URL=http://localhost:6333
PROJECT_ID=my-project-id
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
VECTOR_SIZE=384
ENABLE_MEMORY_DECAY=true
DECAY_WEIGHT=0.3
DECAY_SCALE_DAYS=90
MODEL_CACHE_DAYS=7
TRANSFORMERS_CACHE=/home/mcpuser/.cache/huggingface
LOG_LEVEL=INFO
```

## Default project resolution
- If PROJECT_ID is set when adding the MCP server, it is used.
- Otherwise the working directory (MCP_CLIENT_CWD) is used as fallback (converted to a dash-separated id).

## Add MCP server (local alternative)
If you prefer not to use Docker, the local script creates a venv and runs the server:
```bash
claude mcp add claude-self-reflect "$(claude-self-reflect-path)/mcp-server/run-mcp.sh" \
  -e QDRANT_URL="http://localhost:6333" \
  -e PROJECT_ID="my-project-id"
```

## MCP tools
1) reflect_on_past
- Inputs: query (str), limit (int, default 5), min_score (float, default 0.7),
  project (omit to use default; "all" to search across all), tags (list or comma-separated), use_decay (1/0/-1)

2) store_reflection
- Inputs: content (str), tags (list[str]), project (omit to use default)

## Notes
- EMBEDDING_MODEL and VECTOR_SIZE are mandatory; ensure VECTOR_SIZE matches the model’s output dimension.
- Client-side decay fetches extra candidates; tune min_score and limit if needed.
- On Ubuntu, system modules like python3-apt must be installed via apt, not pip.

## Running Tests

Prerequisites
- Python 3.11+
- Qdrant reachable at `http://localhost:6333` for integration tests

Setup (once)
- Create a virtualenv and install deps (including test extras):
  - `cd mcp-server`
  - `python -m venv venv && . venv/bin/activate`
  - `pip install -e .[test]`

Start Qdrant for integration tests
- Option A: Docker (local)
  - `docker run --rm -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest`
- Option B: docker-compose (from repo root)
  - `docker compose up qdrant`
- The integration tests will be skipped if Qdrant is not reachable.

Run tests
- Unit tests (no network; use fakes/mocks):
  - `pytest -q mcp-server/tests/test_server_unit.py`
- Integration tests (spawn MCP server via stdio):
  - `pytest -q mcp-server/tests/test_integration_search_scenarios.py`
- Single integration test example:
  - `pytest -q mcp-server/tests/test_integration_search_scenarios.py::test_decay_impacts_ranking_order_native`

Notes
- Integration fixtures set required env vars for the spawned server (e.g., `EMBEDDING_MODEL`, `VECTOR_SIZE`, `TRANSFORMERS_CACHE`). You typically do not need to export them manually when running tests.
- Override `QDRANT_URL` if needed: `QDRANT_URL=http://host:port pytest ...`
- Tests communicate with the MCP server via STDIO per FastMCP; the server itself does not open network ports during tests.

Add this to your project's CLAUDE.md or global Claude instructions:

```markdown
## Memory and Context Tools

You have access to semantic memory tools via MCP. Use them proactively:

### When to Search Past Conversations
- User asks about previous discussions or decisions
- You need context about the project's history
- Before implementing something that might have been discussed
- To check for existing solutions to similar problems

### When to Store Reflections
- After solving a complex problem
- When making important architecture decisions
- After discovering useful patterns or techniques
- When the user explicitly asks to remember something
```

## Project-Scoped Search

Conversations are **project-aware by default**. The server resolves the default project as follows:
- If PROJECT_ID is set when adding the MCP server, it will be used
- Otherwise, the working directory (MCP_CLIENT_CWD) is used as fallback (converted to a dash-separated id)

This keeps results focused and relevant to your project context.

### How It Works

# Example: Working in ~/projects/ShopifyMCPMockShop
You: "What authentication method did we implement?"
Claude: [Searches ONLY ShopifyMCPMockShop conversations]
        "Found 3 conversations about JWT authentication..."

# To search everywhere
You: "Search all projects for WebSocket implementations"
Claude: [Searches across ALL your projects]
        "Found implementations in 5 projects: ..."

# To search a specific project
You: "Find Docker setup in claude-self-reflect project"
Claude: [Searches only claude-self-reflect conversations]

## Memory Decay

Recent conversations matter more. Old ones fade. Like your brain, but reliable.

## For the Skeptics

**"Just use grep"** - Sure, enjoy your 10,000 matches for "database"  
**"Overengineered"** - Two functions: store_reflection, reflect_on_past  
**"Another vector DB"** - Yes, because semantic > string matching

---

Stop reading. Start installing. Your future self will thank you.
