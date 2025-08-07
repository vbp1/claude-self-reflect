# Installation Guide

Complete setup guide for Claude Self-Reflect - memory system for Claude conversations.

## Quick Start (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/ramakay/claude-self-reflect.git
cd claude-self-reflect

# 2. Start all services
docker compose --profile watch --profile mcp up -d

# 3. Configure MCP in Claude Code
claude mcp add claude-self-reflect "$(pwd)/run-mcp-docker.sh" -e QDRANT_URL="http://qdrant:6333"

# 4. Restart Claude Code and start using
```

## Prerequisites

### Required
- **Docker Desktop** (macOS/Windows) or **Docker Engine** (Linux)
- **Claude Code** app with MCP support

### System Requirements
- 4GB RAM minimum (8GB recommended)
- 2GB disk space for Docker images and data
- Internet connection for initial model download

## Detailed Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/ramakay/claude-self-reflect.git
cd claude-self-reflect
```

### Step 2: Choose Deployment Option

#### Option A: Full Docker (Recommended)

All services run in containers:

```bash
# Start Qdrant, Watcher, and MCP Server
docker compose --profile watch --profile mcp up -d

# Configure containerized MCP
claude mcp add claude-self-reflect "$(pwd)/run-mcp-docker.sh" -e QDRANT_URL="http://qdrant:6333"
```

**Benefits:**
- No local Python setup needed
- Consistent environment
- Smart model caching
- Easy updates

#### Option B: Local MCP Server

Only Qdrant and Watcher in Docker, MCP server runs locally:

```bash
# Start Qdrant and Watcher only
docker compose --profile watch up -d

# Set up local Python environment
cd mcp-server
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Configure local MCP
claude mcp add claude-self-reflect "$(pwd)/mcp-server/run-mcp.sh" -e QDRANT_URL="http://localhost:6333"
```

**Requirements for Option B:**
- Python 3.11+
- Virtual environment management

### Step 3: Restart Claude Code

Close and reopen Claude Code for MCP changes to take effect.

### Step 4: Test Installation

In Claude Code, ask:
```
"Remember this: Installation completed successfully"
```

Then test search:
```
"What did we just remember about installation?"
```

## Service Management

### Starting Services
```bash
# All services
docker compose --profile watch --profile mcp up -d

# Just database + import
docker compose --profile watch up -d

# Just database + MCP
docker compose --profile mcp up -d

# Just database
docker compose up -d
```

### Stopping Services
```bash
# Stop all
docker compose down

# Stop specific service
docker compose down mcp-server
```

### Viewing Logs
```bash
# All services
docker compose logs

# Specific service
docker logs claude-reflection-watcher
docker logs claude-reflection-mcp
docker logs claude-reflection-qdrant
```

## Configuration

### Environment Variables

Create `.env` file for customization:

```bash
# Model caching (default: 7 days)
MODEL_CACHE_DAYS=30

# Memory decay (default: enabled)
ENABLE_MEMORY_DECAY=true
DECAY_WEIGHT=0.3
DECAY_SCALE_DAYS=90

# Watcher settings
IMPORT_INTERVAL=60
BATCH_SIZE=100

# Resource limits
QDRANT_MEMORY=1g
```

### Claude Code Paths

The system automatically finds Claude Code conversations in:
- **macOS**: `~/Library/Application Support/Claude/projects/`
- **Windows**: `%APPDATA%/Claude/projects/`
- **Linux**: `~/.config/Claude/projects/`

Override with:
```bash
CLAUDE_LOGS_PATH=/custom/path docker compose --profile watch up -d
```

## Data Storage

All data is stored in Docker volumes:

| Volume | Purpose | Size |
|--------|---------|------|
| `qdrant_data` | Vector embeddings | Grows with conversations |
| `watcher_state` | Import tracking | ~1MB |
| `huggingface_cache` | ML models | ~500MB |

### Backup Data
```bash
# Backup Qdrant data
docker run --rm -v claude-self-reflect_qdrant_data:/data -v $(pwd):/backup alpine tar czf /backup/qdrant-backup.tar.gz /data

# Restore from backup
docker run --rm -v claude-self-reflect_qdrant_data:/data -v $(pwd):/backup alpine tar xzf /backup/qdrant-backup.tar.gz -C /
```

## Troubleshooting

### Common Issues

1. **Docker not running**
   ```bash
   docker info  # Should show Docker info, not error
   ```

2. **MCP not connecting**
   ```bash
   # Check MCP list
   claude mcp list
   
   # Remove and re-add
   claude mcp remove claude-self-reflect
   claude mcp add claude-self-reflect "$(pwd)/run-mcp-docker.sh" -e QDRANT_URL="http://qdrant:6333"
   ```

3. **No conversations imported**
   ```bash
   # Check watcher logs
   docker logs claude-reflection-watcher
   
   # Check Claude Code path
   ls ~/.config/Claude/projects/  # or macOS/Windows equivalent
   ```

See [Troubleshooting Guide](troubleshooting.md) for detailed solutions.

## Next Steps

- [Architecture Details](architecture-details.md) - How it works
- [Advanced Usage](advanced-usage.md) - Power features
- [Components Guide](components.md) - Technical deep dive