# Migration Plan: intfloat/multilingual-e5-large

## Overview
Migration from `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) to `intfloat/multilingual-e5-large` (1024 dimensions) for improved multilingual support, especially for Russian language.

## Current State
- **Current Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimensions**: 384
- **Model Size**: 90MB
- **Language Support**: English only
- **Collections**: ~24 projects, ~10,165 chunks

## Target State
- **New Model**: intfloat/multilingual-e5-large
- **Dimensions**: 1024
- **Model Size**: 2.24GB
- **Language Support**: 100+ languages including Russian
- **Expected Performance**: 2-3x slower, but significantly better accuracy

## Migration Phases

### Phase 1: Preparation (10 minutes)
- [ ] Check current configuration in `server.py`
- [ ] Create backup of configuration files
- [ ] Verify FastEmbed support for multilingual-e5-large
- [ ] Check available disk space (need ~3GB for model + increased DB size)
- [ ] Document current collection names and counts

### Phase 2: Code Updates (15 minutes)
- [ ] Update `EMBEDDING_MODEL` in `mcp-server/src/server.py`
  ```python
  # From:
  EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
  # To:
  EMBEDDING_MODEL = 'intfloat/multilingual-e5-large'
  ```
- [ ] Update vector dimensions if hardcoded (384 → 1024)
- [ ] Update import scripts in `scripts/` directory
- [ ] Commit configuration changes

### Phase 3: Service Shutdown (2 minutes)
- [ ] Stop Docker containers
  ```bash
  docker compose stop watcher mcp-server
  ```
- [ ] Save list of current collections
  ```bash
  python scripts/check-collections.py > collections-backup.txt
  ```

### Phase 4: Data Cleanup (5 minutes)
- [ ] Create cleanup script for old collections
- [ ] Execute cleanup script
- [ ] Verify Qdrant is empty
  ```bash
  curl http://localhost:6333/collections
  ```

### Phase 5: Re-import (30-45 minutes)
- [ ] Start re-import process
  ```bash
  cd claude-self-reflect
  source .venv/bin/activate
  python scripts/import-conversations-unified.py
  ```
- [ ] Monitor progress (expecting ~10,165 chunks)
- [ ] Check for errors in import logs

### Phase 6: Service Restart (5 minutes)
- [ ] Restart Docker containers
  ```bash
  docker compose --profile mcp up -d
  ```
- [ ] Remove and re-add MCP in Claude Code
  ```bash
  claude mcp remove claude-self-reflect
  claude mcp add claude-self-reflect "$(pwd)/mcp-server/run-mcp.sh" -e QDRANT_URL="http://localhost:6333"
  ```
- [ ] Restart Claude Code

### Phase 7: Testing (10 minutes)
- [ ] Test English search query
- [ ] Test Russian search query
- [ ] Verify response times
- [ ] Check memory usage
- [ ] Test cross-collection search

## Rollback Plan
If issues occur:
1. Stop all services
2. Restore original `EMBEDDING_MODEL` configuration
3. Clear Qdrant collections
4. Re-import with original model
5. Restart services

## Expected Changes

### Resource Usage
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Model Size | 90MB | 2.24GB | +2400% |
| Vector Dimensions | 384 | 1024 | +267% |
| DB Size per chunk | ~1.5KB | ~4KB | +267% |
| Import Speed | ~20 chunks/sec | ~7 chunks/sec | -65% |
| Search Speed | ~50ms | ~100ms | +100% |

### Quality Improvements
- ✅ Full Russian language support
- ✅ Better semantic understanding
- ✅ Improved cross-lingual search
- ✅ Higher accuracy for technical terms

## Commands Reference

### Check Current Setup
```bash
# Check current model
grep EMBEDDING_MODEL mcp-server/src/server.py

# Check collections
python scripts/check-collections.py

# Check Docker status
docker compose ps
```

### Migration Commands
```bash
# Stop services
docker compose stop

# Clean collections
python scripts/cleanup-collections.py  # Need to create this

# Re-import
python scripts/import-conversations-unified.py

# Restart
docker compose --profile mcp up -d
```

### Verification Commands
```bash
# Test MCP
claude mcp list

# Check logs
docker compose logs -f mcp-server

# Test search (in Claude)
# Use: mcp__claude-self-reflect__reflect_on_past
```

## Risk Mitigation

### Risk 1: FastEmbed Compatibility
- **Check**: Test model loading before migration
- **Mitigation**: Have alternative model ready (e.g., paraphrase-multilingual-mpnet-base-v2)

### Risk 2: Memory Issues
- **Check**: Monitor during first import
- **Mitigation**: Increase Docker memory limits if needed

### Risk 3: Import Failures
- **Check**: Run import with verbose logging
- **Mitigation**: Import in batches if memory constrained

## Success Criteria
- [ ] All ~10,165 chunks successfully imported
- [ ] Search returns relevant results for English queries
- [ ] Search returns relevant results for Russian queries
- [ ] Response time < 200ms for typical queries
- [ ] No memory errors during operation

## Notes
- First model download will take time (~2.24GB)
- Model will be cached for future use
- Consider running migration during low-usage period
- Keep backup of old collections for 1 week

## Timeline
**Total Expected Time**: 1.5-2 hours
- Preparation: 10 min
- Code changes: 15 min
- Migration: 45-60 min
- Testing: 15 min
- Buffer: 15 min