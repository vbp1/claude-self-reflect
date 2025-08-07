# Troubleshooting Guide

Common issues and solutions for Claude Self-Reflect.

## Docker Issues

### Docker not running
**Symptoms**: "Cannot connect to the Docker daemon" or "docker: command not found"

**Solutions**:
1. **Install Docker**:
   - [Docker Desktop](https://docker.com) (macOS/Windows)
   - Docker Engine (Linux)
2. **Start Docker**: Ensure Docker daemon is running
3. **Check status**: `docker info` should show system info
4. **Permissions** (Linux): Add user to docker group:
   ```bash
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

### Services won't start
**Symptoms**: `docker compose up` fails

**Solutions**:
```bash
# Check Docker system status
docker system df
docker system prune  # Clean up if needed

# Check for port conflicts
ss -tuln | grep 6333  # Qdrant port

# Check logs
docker compose logs qdrant
docker compose logs watcher
docker compose logs mcp-server
```

### Volume permission errors
**Symptoms**: "Permission denied" in container logs

**Solutions**:
```bash
# Fix Qdrant volume permissions
docker run --rm -v claude-self-reflect_qdrant_data:/data alpine chown -R 1000:1000 /data

# Fix watcher volume permissions  
docker run --rm -v claude-self-reflect_watcher_state:/state alpine chown -R 1000:1000 /state

# Fix model cache permissions
docker run --rm -v claude-self-reflect_huggingface_cache:/cache alpine chown -R 1000:1000 /cache
```

### Out of disk space
**Symptoms**: "No space left on device"

**Solutions**:
```bash
# Check Docker space usage
docker system df

# Clean up unused resources
docker system prune -a --volumes

# Remove specific volumes (WARNING: data loss)
docker volume rm claude-self-reflect_qdrant_data
```

## MCP Connection Issues

### MCP server not found
**Symptoms**: Tools not available in Claude Code

**Solutions**:
```bash
# Check MCP configuration
claude mcp list

# Verify script is executable
ls -la run-mcp-docker.sh
chmod +x run-mcp-docker.sh

# Check full path is correct
claude mcp remove claude-self-reflect
claude mcp add claude-self-reflect "$(pwd)/run-mcp-docker.sh" -e QDRANT_URL="http://qdrant:6333"

# Restart Claude Code completely
```

### MCP tools timeout
**Symptoms**: "Tool execution timed out" in Claude Code

**Solutions**:
```bash
# Check MCP container status
docker ps | grep claude-reflection-mcp

# Check MCP logs
docker logs claude-reflection-mcp

# Restart MCP service
docker compose --profile mcp down mcp-server
docker compose --profile mcp up -d mcp-server
```

### Environment variable issues
**Symptoms**: MCP can't connect to Qdrant

**Solutions**:
```bash
# Check environment in MCP add command
claude mcp list  # Verify QDRANT_URL is set

# For containerized MCP: use http://qdrant:6333
# For local MCP: use http://localhost:6333

# Test Qdrant connectivity from container
docker exec claude-reflection-mcp curl http://qdrant:6333
```

## Import Issues

### No conversations found
**Symptoms**: Search returns "No conversations found"

**Solutions**:
```bash
# Check watcher is running
docker ps | grep claude-reflection-watcher

# Check watcher logs
docker logs claude-reflection-watcher -f

# Verify Claude Code path exists
ls ~/.config/Claude/projects/  # Linux
ls ~/Library/Application\ Support/Claude/projects/  # macOS
dir %APPDATA%\Claude\projects  # Windows

# Force import with custom path
CLAUDE_LOGS_PATH=/custom/path docker compose --profile watch up -d
```

### Import stuck or slow
**Symptoms**: Watcher logs show no progress

**Solutions**:
```bash
# Check available resources
docker stats

# Restart watcher
docker restart claude-reflection-watcher

# Check for large files blocking import
find ~/.config/Claude/projects/ -name "*.jsonl" -size +100M

# Increase memory limits in docker-compose.yaml
```

### Conversations not updating
**Symptoms**: New conversations don't appear in search

**Solutions**:
```bash
# Check watcher interval (default: 60s)
docker logs claude-reflection-watcher | grep "Sleeping"

# Trigger manual import by restarting watcher
docker restart claude-reflection-watcher

# Check file permissions
ls -la ~/.config/Claude/projects/
```

## Search Issues

### Poor search results
**Symptoms**: Irrelevant or no search results

**Solutions**:
1. **Try different keywords**: Use specific terms from conversations
2. **Check project scope**: 
   ```
   "Search all projects for database optimization"
   ```
3. **Adjust similarity threshold**: Lower min_score in search
4. **Check memory decay**: Recent conversations rank higher

### Search too slow
**Symptoms**: Long delays for search results

**Solutions**:
```bash
# Check Qdrant performance
docker logs claude-reflection-qdrant | grep ERROR

# Check memory usage
docker stats claude-reflection-qdrant

# Increase Qdrant memory limit
echo "QDRANT_MEMORY=2g" >> .env
docker compose down && docker compose --profile watch --profile mcp up -d
```

### Memory decay issues
**Symptoms**: Recent conversations not prioritized

**Solutions**:
```bash
# Check decay configuration
docker exec claude-reflection-mcp env | grep DECAY

# Disable decay for testing
echo "ENABLE_MEMORY_DECAY=false" >> .env
docker compose --profile mcp up -d mcp-server
```

## Model Loading Issues

### Model download timeout
**Symptoms**: MCP server hangs on first start

**Solutions**:
```bash
# Check internet connection
curl -I https://huggingface.co

# Increase download timeout
echo "MODEL_CACHE_DAYS=0" >> .env  # Force re-download

# Check disk space for model cache
docker run --rm -v claude-self-reflect_huggingface_cache:/cache alpine df -h /cache
```

### Model cache issues
**Symptoms**: Slow startup even after first run

**Solutions**:
```bash
# Check model cache contents
docker run --rm -v claude-self-reflect_huggingface_cache:/cache alpine ls -la /cache

# Clear model cache (will re-download)
docker volume rm claude-self-reflect_huggingface_cache

# Check cache age setting
docker exec claude-reflection-mcp env | grep MODEL_CACHE_DAYS
```

## Performance Issues

### High memory usage
**Symptoms**: System slows down, OOM errors

**Solutions**:
```bash
# Check memory usage by service
docker stats

# Reduce batch sizes
echo "BATCH_SIZE=50" >> .env
echo "CHUNK_SIZE=5" >> .env

# Set memory limits
echo "QDRANT_MEMORY=1g" >> .env
```

### High CPU usage
**Symptoms**: Fans spinning, system hot

**Solutions**:
```bash
# Check which service is using CPU
docker stats

# Increase import interval
echo "IMPORT_INTERVAL=300" >> .env  # 5 minutes

# Limit concurrent processing
echo "MAX_PARALLEL_HASHES=2" >> .env
```

## Data Issues

### Lost conversations after restart
**Symptoms**: Previous conversations not found

**Solutions**:
```bash
# Check volume mounts
docker inspect claude-reflection-qdrant | grep Mounts

# Verify volume exists
docker volume ls | grep claude-self-reflect

# Check volume data
docker run --rm -v claude-self-reflect_qdrant_data:/data alpine ls -la /data
```

### Duplicate conversations
**Symptoms**: Same conversation appears multiple times

**Solutions**:
```bash
# Check watcher state file
docker exec claude-reflection-watcher ls -la /app/state/

# Reset watcher state (will re-import everything)
docker volume rm claude-self-reflect_watcher_state

# Check for file hash conflicts
docker logs claude-reflection-watcher | grep "duplicate\|conflict"
```

## Getting Help

### Collect Debug Information
```bash
# System info
docker version
docker compose version

# Service status
docker ps

# Service logs
docker compose logs > debug-logs.txt

# Volume info
docker volume ls | grep claude-self-reflect
docker system df
```

### Reset Everything
**WARNING**: This deletes all data
```bash
# Stop all services
docker compose down

# Remove all data
docker volume rm claude-self-reflect_qdrant_data
docker volume rm claude-self-reflect_watcher_state  
docker volume rm claude-self-reflect_huggingface_cache

# Clean rebuild
docker compose --profile watch --profile mcp up -d --build
```

### Report Issues
- [GitHub Issues](https://github.com/ramakay/claude-self-reflect/issues)
- [GitHub Discussions](https://github.com/ramakay/claude-self-reflect/discussions)

Include debug logs and system information when reporting issues.