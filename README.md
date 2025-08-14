# Claude Self-Reflect

Claude forgets everything. This fixes that.

**Based on https://github.com/ramakay/claude-self-reflect**

## Purpose and Overview

Claude Self-Reflect is a semantic memory system that gives Claude persistent context across conversations through vector embeddings and intelligent search. Built specifically for Claude Code users who need their AI assistant to remember past discussions, decisions, and solutions.

### Core Features

- **Semantic Search**: Find relevant past conversations using natural language queries
- **Project-Scoped Memory**: Automatically filters conversations by current project context
- **Memory Decay**: Time-based relevance scoring (90-day half-life) prevents information overload
- **Local-First Architecture**: All embeddings generated locally, no external API calls
- **MCP Integration**: Native Claude Code integration via Model Context Protocol

### Why It Matters

Every Claude conversation starts from zero. Previous discussions about architecture decisions, bug fixes, and implementation details are lost. Claude Self-Reflect solves this by creating a searchable semantic memory that persists across sessions, making Claude a true collaborator who remembers your project's history.

## Architecture

### System Overview

Claude Self-Reflect uses a modern microservices architecture with three main components:

1. **MCP Server** - Python async server providing semantic search tools to Claude
2. **Qdrant Vector Database** - High-performance vector storage and similarity search
3. **Import Pipeline** - Scripts for conversation processing and embedding generation

### MCP Server Architecture

The MCP server is built with FastMCP, a lightweight async framework designed for Claude integrations:

```python
# Core Server Components
mcp-server/
├── src/
│   ├── server.py         # Main server implementation (600+ lines)
│   │   ├── reflect_on_past()    # Semantic search with decay
│   │   └── store_reflection()   # Store insights and learnings
│   ├── __main__.py       # Entry point for stdio communication
│   └── __init__.py       # Package initialization
├── pyproject.toml        # Dependencies and metadata
├── run-mcp.sh           # Shell wrapper for Claude Code
└── Dockerfile           # Container configuration
```

#### Key Design Decisions:

1. **Async Architecture**: Built on Python's asyncio for high-performance concurrent operations
2. **FastMCP Framework**: Lightweight MCP implementation optimized for stdio communication
3. **Qdrant Integration**: Native async client for vector operations
4. **FastEmbed Models**: Local embedding generation with sentence-transformers
5. **Memory Decay Formula**: Exponential decay with configurable half-life
6. **Project Scoping**: Automatic filtering based on working directory

## Technology Stack

### Core Technologies

#### MCP Server Stack
- **FastMCP 0.0.7+**: Async Model Context Protocol server framework
  - Stdio-based communication with Claude Code
  - Tool registration and parameter validation
  - Built-in error handling and logging

- **Qdrant Client 1.7.0+**: Async vector database client
  - High-performance similarity search
  - Native support for metadata filtering
  - Formula-based scoring with decay expressions

- **FastEmbed 0.4.0+**: Local embedding generation
  - Sentence-transformers models
  - Offline mode with model caching
  - Batch processing for efficiency

- **Pydantic 2.9.2+**: Data validation and settings
  - Type-safe configuration
  - Automatic parameter validation
  - Environment variable parsing

- **Python 3.11+**: Modern async support
  - Native asyncio for concurrent operations
  - Type hints for better IDE support
  - Performance optimizations

#### Infrastructure Stack

- **Qdrant v1.15.1**: Vector database
  - Stores conversation embeddings
  - Supports complex filtering and scoring
  - REST API for cross-platform access
  - Memory-efficient storage (1GB default limit)

- **Docker & Docker Compose**: Container orchestration
  - Service profiles for flexible deployment
  - Named volumes for data persistence
  - Health checks and auto-restart
  - Network isolation for security

#### Embedding Models

| Model | Dimensions | Languages | Use Case |
|-------|------------|-----------|----------|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 50+ | Multilingual with good performance |

All models run 100% locally with automatic caching for offline use.

### Design Philosophy

#### Privacy-First Architecture
- **Local Embeddings**: All vector generation happens on your machine
- **No External APIs**: Zero dependencies on cloud services
- **Data Sovereignty**: Your conversations never leave your infrastructure

#### Intelligent Memory Management
- **Memory Decay**: Exponential decay (90-day half-life) mimics human memory
- **Project Scoping**: Automatic context filtering based on git repository
- **Semantic Deduplication**: Similar conversations are grouped, not duplicated

#### Production-Ready Design
- **Async Architecture**: Non-blocking I/O for high concurrency
- **Container Isolation**: Each service runs in its own security context
- **Graceful Degradation**: System continues working even if components fail
- **Smart Caching**: Model files cached for 7 days, then refreshed

## Installation and Configuration

### Prerequisites

- **Docker Desktop** (macOS/Windows) or **Docker Engine** (Linux)
- **Claude Code** (latest version)
- **Git** (for cloning the repository)
- **4GB+ RAM** recommended for embedding models

### Quick Start with Docker

```bash
# 1. Clone the repository
git clone https://github.com/vbp1/claude-self-reflect.git
cd claude-self-reflect

# 2. Create environment file
cp .env.example .env
# Edit .env to customize settings (optional)

# 3. Start all services
docker compose up -d

# 4. Wait for services to be ready
docker compose ps  # Should show all services as "healthy"

# 5. Add MCP server to Claude Code
claude mcp add claude-self-reflect "$(pwd)/run-mcp-docker.sh" \
  -e QDRANT_URL="http://localhost:6333"

# 6. Restart Claude Code to load the new MCP server
```

### Configuration Options

#### Environment Variables (.env file)

```bash
# Vector Database
QDRANT_PORT=6333                    # Qdrant API port
QDRANT_MEMORY=1g                     # Memory limit for Qdrant

# Embedding Configuration
EMBEDDING_MODEL=intfloat/multilingual-e5-small  # Model name
VECTOR_SIZE=384                      # Must match model dimensions

# Memory Decay Settings
ENABLE_MEMORY_DECAY=true             # Enable time-based decay
DECAY_WEIGHT=0.3                     # Weight of decay in final score
DECAY_SCALE_DAYS=90                  # Half-life in days

# Model Caching
MODEL_CACHE_DAYS=7                   # Days before model refresh

# Logging
LOG_LEVEL=INFO                       # DEBUG, INFO, WARNING, ERROR
LOG_FILE=/logs/mcp-server.log        # Optional log file path
```

## Connecting to Claude Code

### Adding the MCP Server

After Docker services are running, connect the MCP server to Claude Code:

```bash
# Using the Docker wrapper script (recommended)
claude mcp add claude-self-reflect "$(pwd)/run-mcp-docker.sh" \
  -e QDRANT_URL="http://localhost:6333"

# Alternative: Direct Python execution (requires local setup)
claude mcp add claude-self-reflect \
  "cd /path/to/claude-self-reflect && source venv/bin/activate && python -m mcp-server" \
  -e QDRANT_URL="http://localhost:6333"
```

### Available MCP Tools

#### 1. Search Past Conversations

```python
mcp__claude-self-reflect__reflect_on_past(
    query="database optimization techniques",  # Natural language search
    limit=5,                                    # Max results to return
    min_score=0.7,                             # Minimum similarity (0-1)
    project="current",                         # Project scope
    tags=["optimization", "postgresql"],       # Filter by tags
    use_decay=1                                # Apply time decay
)
```

**Parameters:**
- `query` (required): Natural language search query
- `limit`: Maximum results (default: 5)
- `min_score`: Minimum similarity threshold 0-1 (default: 0.7)
- `project`: "current" (default), "all", or specific project name
- `tags`: List of tags to filter results
- `use_decay`: 1=enable, 0=disable, -1=use env default

#### 2. Store Important Insights

```python
mcp__claude-self-reflect__store_reflection(
    content="Use connection pooling with pgbouncer for high-traffic APIs",
    tags=["postgresql", "performance", "scaling"],
    project="current"
)
```

**Parameters:**
- `content` (required): The insight or learning to store
- `tags`: List of tags for categorization (default: [])
- `project`: Target project (default: current working directory)

### Smart Model Caching

The containerized MCP server includes intelligent model caching:

- **First run**: Downloads model from Hugging Face (slow)
- **Subsequent runs**: Uses cached model in offline mode (fast startup)
- **Auto-refresh**: Checks for model updates after 7 days (configurable)
- **Environment variable**: Set `MODEL_CACHE_DAYS=30` for custom refresh interval

### Usage Prompt for Claude

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

## Project-Scoped Search

Conversations are **project-aware by default**. When you ask about past conversations, Claude automatically searches within your current project directory, keeping results focused and relevant.

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
