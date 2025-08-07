# Scripts Directory

Utility scripts for the Claude Self Reflect system.

## Available Scripts

### ✅ `check-collections.py`
**Purpose**: List all Qdrant collections and count local collections.

**Usage**:
```bash
cd claude-self-reflect
source .venv/bin/activate
python scripts/check-collections.py
```

**Output**:
```
Qdrant Collections:
------------------------------------------------------------
- collection_metadata_local
- conv_abcd1234_local
- conv_efgh5678_local

Found 2 local collections
```

**Status**: ✅ Active - useful for diagnostics

---

### ✅ `collection-metadata-manager.py`
**Purpose**: Collection metadata management with CLI interface.

**Usage**:
```bash
cd claude-self-reflect
source .venv/bin/activate

# Auto-discover metadata for existing collections
python scripts/collection-metadata-manager.py discover

# List all metadata
python scripts/collection-metadata-manager.py list

# Get metadata for specific collection
python scripts/collection-metadata-manager.py get conv_abcd1234_local

# Update conversation counts
python scripts/collection-metadata-manager.py update

# Search by tag
python scripts/collection-metadata-manager.py tag auto-discovered
```

**Features**:
- Auto-discovery of metadata for existing collections
- Track conversation counts, projects, tags
- Search collections by tags
- Update counters

**Status**: ✅ Active - main metadata management tool

---

### ✅ `validate-setup.py`
**Purpose**: Comprehensive validation of system installation and configuration.

**Usage**:
```bash
cd claude-self-reflect
source .venv/bin/activate
python scripts/validate-setup.py
```

**Validates**:
- ✅ Environment variables
- ✅ Python dependencies 
- ✅ Docker status and containers
- ✅ Qdrant connection
- ✅ Claude conversation logs
- ✅ MCP configuration
- ✅ API connections (FastEmbed/OpenAI)
- ✅ Disk space and memory

**Status**: ✅ Active - essential for troubleshooting

---

## 🔧 Test Scripts

### 🧪 `test-mcp-debug.py`
**Purpose**: Direct MCP server testing without Claude Code.

**Usage**:
```bash
cd claude-self-reflect
source .venv/bin/activate
python scripts/test-mcp-debug.py
```

**What it does**:
- Import MCP server directly
- Test conversation search
- Show debug information
- Check various search parameters

**Status**: 🧪 Useful for MCP debugging

---

### 🧪 `test-project-search.py`
**Purpose**: Test project search logic by paths and hashes.

**Usage**:
```bash
cd claude-self-reflect
source .venv/bin/activate
python scripts/test-project-search.py
```

**What it does**:
- Test git project detection
- Check project path matching
- Analyze collection metadata
- Test hash-based search

**Status**: 🧪 Useful for project search debugging

---

## 🔧 Utilities

### 🔧 `transfer-collection.py`
**Purpose**: Transfer data between Qdrant collections.

**Usage**:
```bash
cd claude-self-reflect
source .venv/bin/activate
python scripts/transfer-collection.py source_collection target_collection
```

**What it does**:
- Copy all vectors from one collection to another
- Preserve vector configuration
- Process in batches for large collections

**Status**: 🔧 Useful for data migration

---

## 📦 Dependencies

### `requirements.txt`
Python dependencies for all scripts:

```
qdrant-client==1.15.0     # Qdrant client
openai==1.97.1            # OpenAI API (optional)
mcp-server-qdrant==0.8.0  # MCP server (deprecated)
backoff==2.2.1            # Request retries
tqdm==4.67.1              # Progress bars
humanize==4.12.3          # Readable output
fastembed==0.7.1          # Local embeddings
tenacity==9.1.2           # Error resilience
```

**Installation**:
```bash
cd claude-self-reflect
source .venv/bin/activate
pip install -r scripts/requirements.txt
```