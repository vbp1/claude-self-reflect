# Security Policy

## Supported Versions

| Version | Supported          | Notes |
| ------- | ------------------ |-------|
| 2.4.x   | :white_check_mark: | Docker-only architecture, local embeddings |
| 2.3.x   | :warning:          | Legacy support, migration to 2.4.x recommended |
| < 2.3   | :x:                | No longer supported |

## Reporting a Vulnerability

If you discover a security vulnerability, please **DO NOT** create a public issue. Instead:

1. Email the details to the maintainer through GitHub
2. Include:
   - Description of the vulnerability
   - Steps to reproduce (include Docker environment details)
   - Potential impact
   - Suggested fix (if any)

We aim to respond within 48 hours and will work with you to understand and address the issue promptly.

## Security Architecture

### Local-First Design
- **All processing local**: Conversations never leave your machine
- **No external APIs**: FastEmbed runs locally (after initial model download)
- **Docker isolation**: All services run in isolated containers
- **Volume persistence**: Data stored in named Docker volumes with proper permissions

### Container Security
- **Non-root execution**: All services run as uid=1000
- **Minimal attack surface**: Only Qdrant port (6333) exposed locally
- **Resource limits**: Memory and CPU constraints prevent DoS
- **Network isolation**: Custom Docker network with no external access

### Data Protection
- **Project isolation**: Conversations segregated by project hash
- **Local embeddings**: No conversation data sent to cloud services
- **Volume encryption**: Inherits host filesystem encryption
- **State persistence**: Import state and model cache in separate volumes

## Security Measures

### Container Security
- Non-privileged containers (uid=1000)
- Read-only mounts where possible
- Resource limits to prevent resource exhaustion
- Custom Docker network isolation

### Data Security
- Local-only processing (no cloud APIs)
- Project-scoped data isolation
- Volume-based persistence with proper permissions
- No hardcoded secrets or credentials

### Code Security
- No hardcoded API keys or secrets
- Environment variables for configuration
- Regular dependency updates via Dependabot
- Automated security scanning

### Development Security
- Branch protection on `main`
- Required PR reviews
- Automated security checks
- Secrets scanning with Gitleaks

## Docker Security Best Practices

### For Users
1. **Keep Docker updated**: Use latest stable Docker version
2. **Limit exposure**: Only expose necessary ports (6333 for Qdrant)
3. **Monitor resources**: Use `docker stats` to monitor container usage
4. **Regular cleanup**: Use `docker system prune` to remove unused resources

### For Developers
1. **Minimal base images**: Use Python slim images
2. **Multi-stage builds**: Separate build and runtime stages
3. **Security scanning**: Use `docker scan` or Trivy
4. **Volume permissions**: Set proper ownership (uid=1000)

## CI/CD Security

### Automated Checks
- Secrets scanning with Gitleaks
- Dependency vulnerability scanning
- Docker image security scanning with Trivy
- Python security checks with Bandit
- Container configuration validation

### Build Security
- Reproducible builds
- Minimal runtime dependencies
- Security context validation
- Resource limit testing

## Best Practices for Contributors

### Development
1. **Never commit secrets**: Use environment variables and `.env` files
2. **Test containers**: Verify Docker builds work correctly
3. **Check dependencies**: Run security scans before submitting PRs
4. **Volume permissions**: Ensure containers can read/write volumes
5. **Review changes**: Check commits for sensitive data

### Docker Development
1. **Use specific tags**: Pin base image versions
2. **Minimize layers**: Combine RUN commands where possible
3. **Clean package cache**: Remove package managers' cache
4. **Set proper permissions**: Use proper file ownership

## Security Configuration

### Environment Variables
```bash
# Model caching (prevents frequent downloads)
MODEL_CACHE_DAYS=7

# Memory limits (prevent resource exhaustion)
QDRANT_MEMORY=1g

# Import settings (control batch sizes)
BATCH_SIZE=100
CHUNK_SIZE=10
```

### Docker Compose Security
- Named volumes (not host mounts for sensitive data)
- Custom networks (no external connectivity)
- Resource limits on all services
- Non-root user execution

### Volume Security
```bash
# Fix volume permissions if needed
docker run --rm -v claude-self-reflect_qdrant_data:/data alpine chown -R 1000:1000 /data
```

## Recommended GitHub Settings

For repository administrators:

### Branch Protection Rules for `main`:
- ✅ Require pull request reviews before merging
- ✅ Dismiss stale pull request approvals when new commits are pushed
- ✅ Require status checks to pass before merging:
  - `docker-build`
  - `security-scan`
  - `dependency-scan`
- ✅ Require branches to be up to date before merging
- ✅ Include administrators
- ✅ Do not allow force pushes
- ✅ do not allow deletions

### Security Settings:
- ✅ Enable Dependabot security updates
- ✅ Enable secret scanning
- ✅ Enable push protection for secrets
- ✅ Enable vulnerability alerts
- ✅ Enable code scanning (CodeQL)

## Security Audit History

- **v2.4.x**: Docker-only architecture, removed Node.js dependencies, enhanced container security
- **v2.3.9**: Added gitleaks configuration, enhanced secret detection
- **v2.3.7**: Major security cleanup, removed 250+ internal files, secured permissions
- **v2.3.3**: Migrated to local embeddings by default for complete privacy
- **v2.0.0**: Complete rewrite with security-first design

## Threat Model

### Mitigated Threats
- **Data exfiltration**: All processing local, no external APIs
- **Container escape**: Non-root execution, resource limits
- **Network attacks**: Custom isolated network
- **Resource exhaustion**: Memory and CPU limits
- **Privilege escalation**: Non-privileged containers

### Residual Risks
- **Host compromise**: Containers inherit host security
- **Docker daemon compromise**: Requires host-level security
- **Physical access**: Volume data readable with host access

## Security Monitoring

### Health Checks
```bash
# Check container security
docker inspect claude-reflection-qdrant | jq '.[] | {User: .Config.User, Privileged: .HostConfig.Privileged}'

# Monitor resource usage
docker stats --no-stream

# Check volume permissions
docker run --rm -v claude-self-reflect_qdrant_data:/data alpine ls -la /data
```

### Log Monitoring
- Monitor Docker logs for unusual activity
- Watch for repeated failures or connection attempts
- Check resource usage patterns

## Incident Response

### If a vulnerability is discovered:
1. Assess impact scope
2. Develop and test fix
3. Coordinate disclosure timeline
4. Release security update
5. Notify users through GitHub releases
6. Update security documentation

### If containers are compromised:
1. Stop affected containers immediately
2. Investigate through logs and monitoring
3. Remove compromised containers and volumes if needed
4. Rebuild from clean images
5. Review and strengthen security measures

## Privacy Considerations

- **Local processing**: No conversation data transmitted externally
- **Model caching**: ML models cached locally to prevent repeated downloads
- **Project isolation**: Different projects' data separated
- **No telemetry**: No usage data collected or transmitted

This security policy reflects the current Docker-based architecture and local-first approach of Claude Self-Reflect.