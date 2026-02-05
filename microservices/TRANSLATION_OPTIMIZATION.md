# Translation Service Resource Optimization

## Problem Analysis

The translation service was causing server crashes due to:

1. **Unbounded resource consumption**: No CPU/memory limits on containers
2. **Memory leaks**: Large ML models (IndicTrans2 1B+ params) loaded without cleanup
3. **Uncontrolled concurrency**: Multiple parallel translations without rate limiting
4. **No error recovery**: Service failures cascaded without retry mechanisms
5. **Batch processing inefficiencies**: Loading entire datasets into memory at once

## Implemented Solutions

### 1. Resource Limits (Docker Compose)

**Changes in `docker-compose.app.yml`:**

```yaml
translate-indian:
  deploy:
    resources:
      limits:
        cpus: '4.0'
        memory: 8G
      reservations:
        cpus: '2.0'
        memory: 4G

translate-international:
  deploy:
    resources:
      limits:
        cpus: '4.0'
        memory: 6G
      reservations:
        cpus: '1.0'
        memory: 2G
```

**Benefits:**
- Prevents single service from consuming all server resources
- Ensures other services remain operational
- Linux kernel will enforce memory limits and prevent OOM

### 2. Memory Management (IndicTrans2 Service)

**New Features:**

#### Memory Monitoring
```python
def _get_memory_usage() -> float:
    """Returns current memory usage percentage"""
```

#### Automatic Cleanup
```python
def _cleanup_memory() -> None:
    """Triggers GC and clears CUDA cache when memory > 85%"""
```

#### Batch Processing Improvements
- Reduced default batch size: 16 → 8
- Text truncation: 4000 chars max per request
- Tensor cleanup after each batch
- Periodic GC during long operations

**Configuration Variables:**
```bash
INDIC_TRANS_BATCH_SIZE=8              # Smaller batches = less memory
INDIC_TRANS_MAX_CONCURRENT=2          # Max parallel requests
INDIC_TRANS_MEMORY_LIMIT_PCT=85.0     # Trigger cleanup threshold
INDIC_TRANS_TIMEOUT_SECONDS=120       # Request timeout
INDIC_TRANS_MAX_TEXT_LENGTH=4000      # Prevent OOM from huge texts
```

### 3. Request Rate Limiting

**Semaphore-based concurrency control:**
```python
_REQUEST_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_REQUESTS)
```

- Limits concurrent translations to 2 (configurable)
- Returns 503 "Service busy" when queue is full
- Backend automatically retries with exponential backoff

### 4. Enhanced Error Handling (Backend)

**Retry mechanism with exponential backoff:**
```python
max_retries = 3
retry_delay = 2.0

for attempt in range(max_retries):
    try:
        # Translation attempt
    except requests.exceptions.Timeout:
        if attempt < max_retries - 1:
            time.sleep(retry_delay * (attempt + 1))
            continue
```

**Handles:**
- Timeouts (120s default)
- Connection errors
- 503 Service Busy responses
- Partial failures without crashing entire job

### 5. Health Monitoring

**Enhanced `/health` endpoint:**
```json
{
  "ok": true,
  "memory_usage_pct": 67.3,
  "memory_available_gb": 12.5,
  "models_loaded": 2,
  "max_concurrent_requests": 2,
  "warning": "High memory usage"  // if > 85%
}
```

## Performance Impact

### Before Optimization
- ❌ Server crashes during large translation jobs
- ❌ SSH connections lost due to resource exhaustion
- ❌ Other services impacted
- ❌ No recovery from failures

### After Optimization
- ✅ Stable operation under load
- ✅ Graceful degradation (503 responses when busy)
- ✅ Automatic retry and recovery
- ✅ Protected from memory exhaustion
- ✅ Other services isolated from translation load

## Deployment Instructions

### 1. Rebuild Translation Service

```bash
cd /home/tarun-envid/envid-metadata/microservices
docker-compose -f docker-compose.app.yml build translate-indian
```

### 2. Restart Services

```bash
docker-compose -f docker-compose.app.yml down translate-indian translate-international
docker-compose -f docker-compose.app.yml up -d translate-indian translate-international
```

### 3. Verify Health

```bash
# Check Indian translation service
curl http://localhost:5102/health

# Check resource limits
docker stats translate-indian translate-international
```

### 4. Monitor Logs

```bash
docker logs -f translate-indian
docker logs -f backend
```

## Tuning Guidelines

### If translations are slow:
```bash
# Increase concurrency (requires more RAM)
INDIC_TRANS_MAX_CONCURRENT=3
INDIC_TRANS_BATCH_SIZE=12

# Adjust Docker memory limits accordingly
memory: 12G  # increase proportionally
```

### If still experiencing OOM:
```bash
# Reduce concurrency and batch size
INDIC_TRANS_MAX_CONCURRENT=1
INDIC_TRANS_BATCH_SIZE=4
INDIC_TRANS_MEMORY_LIMIT_PCT=75.0

# Reduce Docker memory limits
memory: 6G
```

### If getting 503 errors:
```bash
# Increase queue depth
INDIC_TRANS_MAX_CONCURRENT=3

# Backend will automatically retry
```

## Monitoring Commands

### Check memory usage:
```bash
docker stats --no-stream translate-indian
```

### Watch for 503 errors:
```bash
docker logs translate-indian 2>&1 | grep "503\|busy"
```

### Monitor translation performance:
```bash
curl http://localhost:5102/health | jq '.memory_usage_pct'
```

### Check backend retries:
```bash
docker logs backend 2>&1 | grep "retry"
```

## Additional Recommendations

1. **Server-level monitoring**: Install monitoring tools (Prometheus, Grafana)
2. **Swap configuration**: Ensure adequate swap space (16GB+ recommended)
3. **OOM killer protection**: Consider using systemd OOM score adjustment
4. **Load balancing**: For high-volume workloads, consider running multiple translation instances
5. **Resource scheduling**: Process large translation jobs during off-peak hours

## Rollback

If issues occur, revert to previous configuration:

```bash
git checkout HEAD~1 -- microservices/translate/code/indictrans2_service.py
git checkout HEAD~1 -- microservices/docker-compose.app.yml
docker-compose -f docker-compose.app.yml restart translate-indian translate-international
```

## Support

For issues or questions:
1. Check logs: `docker logs translate-indian`
2. Verify health: `curl localhost:5102/health`
3. Monitor resources: `docker stats`
4. Review this document for tuning options
