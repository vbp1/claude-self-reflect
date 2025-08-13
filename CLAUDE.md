# Claude Self Reflect - Conversation Memory for Claude

## Overview
Claude Self Reflect provides semantic search across all Claude conversations with built-in memory decay capabilities, using a vector database for efficient similarity matching.

## Architecture
- **Vector Database**: Qdrant with per-project collections
- **Embeddings**: FastEmbed local embeddings (all-MiniLM-L6-v2, 384 dimensions)
- **Search**: Cross-collection semantic search with time-based decay
- **Import**: Continuous file watcher for automatic updates
- **MCP Server**: Python-based using FastMCP (located in `mcp-server/`)

## Memory Decay Philosophy

### Why Decay?
Digital memory systems face a fundamental challenge: perfect recall creates imperfect utility. As conversations accumulate, finding relevant information becomes harder. Memory decay solves this by:

1. **Prioritizing Recency**: Recent conversations are more relevant
2. **Reducing Noise**: Old, outdated information fades gracefully
3. **Mimicking Human Memory**: Aligning with natural cognitive patterns
4. **Maintaining Performance**: Preventing unbounded growth issues

### MCP Management in Claude Code

#### CORRECT Commands (Use These):
```bash
# List all MCPs and their connection status
claude mcp list

# Add the MCP with required environment variables (name, then command, then env vars)
claude mcp add claude-self-reflect "/Users/ramakrishnanannaswamy/projects/claude-self-reflect/mcp-server/run-mcp.sh" -e QDRANT_URL="http://localhost:6333"

# Optional: Add with custom logging configuration
# LOG_LEVEL options: DEBUG, INFO (default), WARNING, ERROR
# LOG_FILE: Path to log file (if not set, logs only to console)
claude mcp add claude-self-reflect "/path/to/mcp-server/run-mcp.sh" -e QDRANT_URL="http://localhost:6333" -e LOG_LEVEL="DEBUG" -e LOG_FILE="/tmp/mcp.log"

# Remove an MCP (useful when needing to restart)
claude mcp remove claude-self-reflect

# Restart MCP (remove then re-add with env vars)
claude mcp restart claude-self-reflect
```

#### INCORRECT Commands (Never Use):
```bash
# ❌ These commands DO NOT exist:
claude mcp status claude-self-reflect  # NO SUCH COMMAND
claude mcp logs claude-self-reflect    # NO SUCH COMMAND
claude mcp add claude-self-reflect     # MISSING required commandOrUrl argument
```

#### Important Notes:
- The `claude mcp add` command REQUIRES both a name AND a commandOrUrl
- Environment variables must be passed with `-e` flag
- After adding MCP, you may need to restart Claude Code for tools to be available
- No API keys required - uses local embeddings only

### Search & Reflection
```bash
# Use MCP tools in Claude
mcp__claude-self-reflect__reflect_on_past
mcp__claude-self-reflect__store_reflection
```

### Import Commands
```bash
# Always use the virtual environment
cd claude-self-reflect
source .venv/bin/activate  # or source venv/bin/activate

# Import all projects
python scripts/import-conversations-unified.py

# Check collections
python scripts/check-collections.py
```

## Specialized Sub-Agents for Claude Self Reflect

### Overview
This project includes 6 specialized sub-agents that Claude will PROACTIVELY use when working on different aspects of the system. Each agent has focused expertise and will automatically activate when their domain is encountered.

**IMPORTANT**: Agents are located in `.claude/agents/` directory (NOT in a random /agents folder). They are automatically installed via npm postinstall script.

### Available Sub-Agents

1. **qdrant-specialist** - Vector database expert
   - **When to use**: Qdrant operations, collection management, embedding issues
   - **Expertise**: Collection health, search troubleshooting, dimension mismatches

2. **import-debugger** - Import pipeline specialist
   - **When to use**: Import failures, JSONL processing, zero messages issues
   - **Expertise**: JQ filters, Python scripts, conversation chunking

3. **docker-orchestrator** - Container management expert
   - **When to use**: Service failures, container restarts, compose issues
   - **Expertise**: Multi-container orchestration, health monitoring, networking

4. **mcp-integration** - MCP server developer
   - **When to use**: Claude Code integration, tool development, TypeScript
   - **Expertise**: MCP protocol, tool implementation, connection debugging

5. **search-optimizer** - Search quality expert
   - **When to use**: Poor search results, tuning thresholds, comparing models
   - **Expertise**: Semantic search, embedding quality, A/B testing

6. **reflection-specialist** - Conversation memory expert
   - **When to use**: Searching past conversations, storing insights, self-reflection
   - **Expertise**: Semantic search, insight storage, knowledge continuity

### Proactive Usage Examples

When you mention any of these scenarios, Claude will automatically engage the appropriate sub-agent:

```
"The import is showing 0 messages again"
→ import-debugger will investigate JQ filters and JSONL parsing

"Search results seem irrelevant"
→ search-optimizer will analyze similarity thresholds and embedding quality

"Find conversations about debugging this issue"
→ reflection-specialist will search past conversations and insights

"Remember this solution for next time"
→ reflection-specialist will store the insight with appropriate tags
```

## Folder Structure

```
claude-self-reflect/
├── mcp-server/           # Python MCP server using FastMCP
│   ├── src/              # Main server source code
│   ├── pyproject.toml    # Python package configuration
│   └── run-mcp.sh        # MCP startup script
├── scripts/              # Import and utility scripts
│   ├── import-*.py       # Various import scripts
│   └── test-*.py         # Test scripts
├── .claude/agents/       # Claude sub-agents for specialized tasks
├── config/               # Configuration files
├── data/                 # Qdrant data storage
├── docs/                 # Documentation
└── archived/             # Archived code (TypeScript implementation)
```

## Project Rules
- Always activate venv before running Python scripts
- Use reflection-specialist agent for testing search functionality
- Never commit without running tests first
- Memory decay is opt-in (disabled by default)
- Test files belong in organized directories, not root
- **CRITICAL**: All agents MUST follow [MCP_REFERENCE.md](./MCP_REFERENCE.md) for MCP operations

## Pre-commit Hook Setup (ОБЯЗАТЕЛЬНО для разработчиков)

### Установка
```bash
# 1. Установить pre-commit
pip install pre-commit

# 2. Установить git hooks в проект
cd claude-self-reflect
pre-commit install

# 3. (Опционально) Запустить на всех файлах
pre-commit run --all-files
```

### Как работает
- **Автоматически запускается** при каждом `git commit`
- **Блокирует коммит** если есть ошибки линтера
- Проверяет только Python файлы в `mcp-server/src/` и `scripts/`
- Использует ruff для линтинга и форматирования

### Что проверяется
1. **Ruff линтер** - ошибки E, W, F (синтаксис, стиль, логика)
2. **Ruff форматирование** - правильное форматирование кода
3. **Python синтаксис** - проверка на синтаксические ошибки
4. **Trailing whitespace** и **EOF** - чистота файлов

### Пример работы
```bash
$ git commit -m "test commit"

Ruff Linter (блокирует коммит при ошибках)................................Failed
- hook id: ruff
- exit code: 1

mcp-server/src/server.py:45:80: E501 Line too long (89 > 79 characters)

# Коммит заблокирован! Нужно исправить ошибки
```

### Обход (только в крайних случаях)
```bash
# Обойти pre-commit hooks (НЕ РЕКОМЕНДУЕТСЯ)
git commit -m "message" --no-verify
```

## File Organization
- Claude automatically organizes .md files based on content (see parent project's CLAUDE.md)
- **Organization Log**: If you can't find a created .md file, check `docs/organization-log.json`
  - This log tracks where files have been moved by the auto-organization system
  - It's in .gitignore so won't appear in git status
  - Future agents should consult this log when files seem to be missing

## Upgrade Guide for Existing Users

### Key Changes in v2.3.7+
1. **Local Embeddings Only**: FastEmbed provides privacy and eliminates API dependencies
2. **Setup Wizard Improvements**: Better handling of existing installations
3. **Security Enhancements**: Automated scanning and vulnerability checks

### Common Upgrade Issues & Solutions

#### 1. Python Virtual Environment Conflicts
**Problem**: Setup wizard fails with "Unable to symlink python3.13" or similar
**Solution**: The setup wizard now includes health checks for existing venvs:
- Detects if venv exists but is broken/incomplete
- Checks if dependencies (fastmcp, qdrant_client) are installed
- Automatically reinstalls missing dependencies

#### 2. MCP Connection Issues After Upgrade
**Problem**: Tools not accessible after upgrade
**Solution**: 
```bash
# Remove and re-add the MCP server
claude mcp remove claude-self-reflect
claude mcp add claude-self-reflect "/path/to/mcp-server/run-mcp.sh" -e QDRANT_URL="http://localhost:6333"
# Restart Claude Code for changes to take effect
```

#### 3. Collection Management
All collections now use `_local` suffix for FastEmbed embeddings:
- Collections are automatically created during import
- Only local embeddings are supported

#### 4. Import Script Changes
**Script Changes**: 
- Import script: `import-conversations-unified.py`
- Uses only local FastEmbed embeddings

### Best Practices for v2.4.x
1. **Use Docker profiles**: `--profile mcp` for full setup
2. **Monitor containers**: `docker compose logs -f watcher`
3. **Check volume permissions**: All services run as uid=1000
4. **Backup volumes**: Named volumes contain all persistent data
5. **Use container networking**: `qdrant:6333` for inter-container communication

## Quick Start (v2.4.x)

### 1. Start Services
```bash
cd claude-self-reflect
docker compose --profile mcp up -d
```

### 2. Configure MCP in Claude Code
```bash
# Recommended: Use containerized MCP
claude mcp add claude-self-reflect "$(pwd)/run-mcp-docker.sh" -e QDRANT_URL="http://qdrant:6333"
```

### 3. Verify Setup
```bash
# Check services
docker compose ps

# Check MCP connection
claude mcp list

# Test tools (restart Claude Code if needed)
# Use: mcp__claude-self-reflect__reflect_on_past
# Use: mcp__claude-self-reflect__store_reflection
```