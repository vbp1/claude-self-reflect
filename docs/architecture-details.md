# Architecture Details

Technical deep dive into Claude Self-Reflect's system design and data flow.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Claude Code/Desktop                          │
│                       (MCP Client)                             │
└────────────────────────────┬───────────────────────────────────┘
                             │ MCP Protocol (stdio)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MCP Server                                │
│                   (Python FastMCP)                             │
│  ┌─────────────────┐        ┌──────────────────────┐          │
│  │ store_reflection │        │  reflect_on_past     │          │
│  └─────────────────┘        └──────────────────────┘          │
└────────────────────────────┬───────────────────────────────────┘
                             │ HTTP API
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Qdrant Vector Database                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Collections: conv_<project_hash>_local                │  │
│  │  - Conversation embeddings (384 dims)                  │  │
│  │  - Metadata (timestamp, project, context)              │  │
│  │  - Memory decay scoring                                 │  │
│  └─────────────────────────────────────────────────────────┘  │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▲
┌─────────────────────────────────────────────────────────────────┐
│                    Import Pipeline                              │
│                   (Watcher Service)                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  Watch   │→ │  Chunk   │→ │ FastEmbed│→ │ Store in     │  │
│  │  JSONL   │  │ Messages │  │ (Local)  │  │ Qdrant       │  │
│  │  Changes │  │ (500     │  │ 384-dim  │  │ Collections  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Claude Code/Desktop
The MCP client that communicates with our memory system:
- **Role**: User interface and MCP client
- **Communication**: stdio-based MCP protocol
- **Integration**: Automatic sub-agent spawning for memory operations

### 2. MCP Server (Python FastMCP)
Core service providing memory operations:

**Tools Provided**:
- `reflect_on_past` - Semantic search with optional time decay
- `store_reflection` - Store important insights and decisions

**Key Features**:
- **Smart Caching**: Downloads models once, runs offline afterwards  
- **Project Awareness**: Searches current project by default
- **Memory Decay**: Recent conversations weighted higher
- **Local Embeddings**: FastEmbed with sentence-transformers

### 3. Qdrant Vector Database
High-performance vector similarity search:
- **Storage**: Conversation embeddings + metadata
- **Collections**: Per-project isolation (`conv_<hash>_local`)
- **Features**: HNSW indexing, cosine similarity, persistence
- **Performance**: Sub-100ms search across thousands of conversations

### 4. Watcher Service
Continuous import pipeline:
- **Monitoring**: File system changes (60s interval)
- **Processing**: JSONL parsing, message chunking, deduplication
- **State Tracking**: Hash-based change detection
- **Resilience**: Incremental imports, error recovery

## Data Flow

### Import Pipeline

```
Claude Conversation (JSONL)
    ↓
File System Watcher
    ↓
Change Detection (SHA-256)
    ↓
Message Chunking (500 tokens)
    ↓
FastEmbed Encoding (384-dim)
    ↓
Qdrant Storage (with metadata)
    ↓
Collection Metadata Update
```

**Chunking Strategy**:
- **Size**: ~500 tokens per chunk
- **Overlap**: Minimal to avoid duplication
- **Metadata**: Timestamp, project, conversation ID, chunk index

### Search Pipeline

```
User Query
    ↓
MCP Tool Invocation
    ↓
FastEmbed Query Encoding
    ↓
Project Scope Resolution
    ↓
Qdrant Vector Search
    ↓
Memory Decay Application (optional)
    ↓
Result Ranking & Formatting
    ↓
Claude Code Display
```

## Memory System Design

### Project-Scoped Collections
Each project gets its own collection:
```
conv_a1b2c3d4_local  # Project hash + "_local" suffix
conv_e5f6g7h8_local  # Different project
reflections_local    # User-stored reflections
```

**Benefits**:
- **Isolation**: Work and personal conversations separated
- **Performance**: Smaller search spaces (100ms vs 1s+)
- **Privacy**: Projects don't cross-contaminate

### Memory Decay Implementation

**Client-Side Exponential Decay**:
```python
# Decay formula
decay_factor = exp(-age_ms / scale_ms)
adjusted_score = original_score + (decay_weight * decay_factor)

# Default parameters
scale_ms = 90 days * 24 * 60 * 60 * 1000  # 90-day half-life
decay_weight = 0.3  # 30% boost for recent content
```

**Why Client-Side**:
- **Flexibility**: Adjustable parameters per search
- **Compatibility**: Works with any Qdrant version
- **Performance**: ~9ms overhead for 1000 points

### Embedding Strategy

**FastEmbed with all-MiniLM-L6-v2**:
- **Dimensions**: 384 (vs 1024 for larger models)
- **Performance**: Fast encoding, good semantic quality
- **Privacy**: Runs locally, no API calls
- **Cache**: Smart model caching with 7-day refresh

## Container Architecture

### Docker Services

**Qdrant** (`qdrant` service):
- **Image**: `qdrant/qdrant:v1.15.1`
- **Purpose**: Vector database
- **Data**: `qdrant_data` volume
- **Resources**: 1GB memory limit

**Watcher** (`watcher` service, profile: `watch`):
- **Build**: Custom Python container
- **Purpose**: Continuous import
- **Data**: `watcher_state` volume (tracking)
- **Resources**: 2GB memory limit

**MCP Server** (`mcp-server` service, profile: `mcp`):
- **Build**: Custom Python container  
- **Purpose**: Claude Code integration
- **Data**: `huggingface_cache` volume (models)
- **Interface**: Docker exec via `run-mcp-docker.sh`

### Volume Strategy

| Volume | Purpose | Size | Backup Priority |
|--------|---------|------|----------------|
| `qdrant_data` | Vector embeddings | ~100MB per 1000 convs | High |
| `watcher_state` | Import tracking | ~1MB | Medium |
| `huggingface_cache` | ML models | ~500MB | Low |

## Security & Privacy

### Data Privacy
- **Local Only**: No data leaves your machine
- **No API Calls**: FastEmbed runs locally
- **Encrypted Storage**: Docker volumes use host filesystem encryption
- **Project Isolation**: Conversations segregated by project

### Security Features
- **Non-Root Containers**: All services run as uid=1000
- **Minimal Attack Surface**: Only Qdrant port exposed (6333)
- **Volume Permissions**: Proper ownership and access control
- **Resource Limits**: Memory and CPU constraints prevent DoS

## Performance Characteristics

### Search Performance
- **Single Project**: ~50-100ms
- **All Projects**: ~100-200ms
- **Memory Decay**: +9ms overhead
- **Scaling**: Linear with collection size

### Import Performance
- **New Files**: ~1-2 seconds per conversation
- **Change Detection**: ~50ms per file scan
- **Embeddings**: ~100ms per 500-token chunk
- **Deduplication**: O(1) hash lookup

### Resource Usage
- **Memory**: 2-4GB total (configurable)
- **CPU**: Low (mostly I/O bound)
- **Disk**: 100MB per 1000 conversations
- **Network**: None (local embeddings)

## Scalability

### Current Limits
- **Conversations**: 10,000+ (tested)
- **Projects**: 50+ (tested)
- **Search Latency**: <200ms at scale
- **Import Rate**: ~100 conversations/minute

### Optimization Strategies
- **Index Tuning**: HNSW parameters
- **Batch Processing**: Import chunking
- **Memory Management**: Configurable limits
- **Caching**: Model and search result caching

## Development Architecture

### Code Organization
```
claude-self-reflect/
├── mcp-server/           # Python MCP server
│   ├── src/             # FastMCP implementation
│   └── pyproject.toml   # Dependencies
├── scripts/             # Import utilities
├── .claude/agents/      # Specialized sub-agents
└── docker-compose.yaml  # Service orchestration
```

### Extension Points
- **Custom Embeddings**: Swap FastEmbed models
- **Additional Tools**: Extend MCP server
- **Import Sources**: Beyond JSONL files  
- **Search Algorithms**: Custom similarity functions

## Comparison with Alternatives

### vs. Traditional Search
- **Semantic**: Understanding vs keyword matching
- **Context-Aware**: Project and time-based relevance
- **Privacy**: Local processing vs cloud services

### vs. RAG Solutions
- **Specialized**: Conversation-optimized chunking
- **Integrated**: Native Claude Code experience
- **Efficient**: Purpose-built for dialogue memory

### vs. External APIs
- **Privacy**: No data transmission
- **Cost**: No per-query charges
- **Latency**: Local processing speed
- **Reliability**: No network dependencies