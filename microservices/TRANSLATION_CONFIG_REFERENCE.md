# Translation Service Configuration Reference

## Environment Variables

### IndicTrans2 Service (Indian Languages)

| Variable | Default | Description |
|----------|---------|-------------|
| `INDIC_TRANS_BATCH_SIZE` | 8 | Number of texts to process in parallel. Lower = less memory, slower. Higher = more memory, faster. |
| `INDIC_TRANS_MAX_TOKENS` | 512 | Maximum tokens per translation. Controls model input size. |
| `INDIC_TRANS_MAX_CONCURRENT` | 2 | Maximum concurrent translation requests. Limits parallelism. |
| `INDIC_TRANS_MEMORY_LIMIT_PCT` | 85.0 | Memory threshold (%) to trigger cleanup. Range: 70-95. |
| `INDIC_TRANS_TIMEOUT_SECONDS` | 120 | Request timeout in seconds. Adjust for large texts. |
| `INDIC_TRANS_MAX_TEXT_LENGTH` | 4000 | Maximum characters per text. Prevents OOM from huge inputs. |
| `INDIC_TRANS_CACHE_DIR` | - | Directory for model cache. Use fast SSD storage. |
| `INDIC_TRANS_WARMUP` | true | Pre-load models on startup. Set false to save memory. |

### Backend Translation Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVID_METADATA_TRANSLATE_CONCURRENCY` | 1 | How many languages to translate in parallel. |
| `ENVID_METADATA_UPLOAD_CONCURRENCY` | 2 | How many uploads to do in parallel. |
| `ENVID_INDIC_TRANS_TIMEOUT_SECONDS` | 120 | Backend timeout for translation requests. |
| `ENVID_METADATA_TRANSLATE_LANGS` | - | Target languages (comma-separated): e.g., "hi,bn,ta,te" |

## Docker Resource Limits

### translate-indian (IndicTrans2)

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'      # Maximum CPU cores
      memory: 8G       # Maximum RAM
    reservations:
      cpus: '2.0'      # Minimum guaranteed CPUs
      memory: 4G       # Minimum guaranteed RAM
```

**Tuning:**
- Small server (16GB RAM): `memory: 6G`, `INDIC_TRANS_MAX_CONCURRENT=1`
- Medium server (32GB RAM): `memory: 8G`, `INDIC_TRANS_MAX_CONCURRENT=2` (default)
- Large server (64GB+ RAM): `memory: 12G`, `INDIC_TRANS_MAX_CONCURRENT=3`

### translate-international (LibreTranslate)

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 6G
    reservations:
      cpus: '1.0'
      memory: 2G
```

## Performance Profiles

### Conservative (Low Resource Usage)
```bash
# .env settings
INDIC_TRANS_BATCH_SIZE=4
INDIC_TRANS_MAX_CONCURRENT=1
INDIC_TRANS_MEMORY_LIMIT_PCT=75.0
ENVID_METADATA_TRANSLATE_CONCURRENCY=1

# docker-compose.app.yml
memory: 6G  # translate-indian
```

**Best for:**
- Servers with < 24GB RAM
- Shared servers running many services
- High stability priority

**Trade-off:** Slower translation (60-90s per language)

### Balanced (Default)
```bash
# .env settings
INDIC_TRANS_BATCH_SIZE=8
INDIC_TRANS_MAX_CONCURRENT=2
INDIC_TRANS_MEMORY_LIMIT_PCT=85.0
ENVID_METADATA_TRANSLATE_CONCURRENCY=1

# docker-compose.app.yml
memory: 8G  # translate-indian
```

**Best for:**
- Servers with 32GB+ RAM
- Moderate translation workload
- Balance of speed and stability

**Trade-off:** Balanced performance (30-60s per language)

### Performance (High Throughput)
```bash
# .env settings
INDIC_TRANS_BATCH_SIZE=12
INDIC_TRANS_MAX_CONCURRENT=3
INDIC_TRANS_MEMORY_LIMIT_PCT=90.0
ENVID_METADATA_TRANSLATE_CONCURRENCY=2

# docker-compose.app.yml
memory: 12G  # translate-indian
```

**Best for:**
- Servers with 64GB+ RAM
- High-volume translation workload
- Speed priority

**Trade-off:** Higher resource usage (20-40s per language)

## Troubleshooting

### Problem: Server still crashes / OOM errors

**Solution:**
```bash
# Reduce all resource usage
INDIC_TRANS_BATCH_SIZE=4
INDIC_TRANS_MAX_CONCURRENT=1
INDIC_TRANS_MEMORY_LIMIT_PCT=70.0

# docker-compose.app.yml
memory: 6G
```

### Problem: Translations are too slow

**Solution:**
```bash
# Increase parallelism (requires more RAM)
INDIC_TRANS_MAX_CONCURRENT=3
ENVID_METADATA_TRANSLATE_CONCURRENCY=2

# docker-compose.app.yml
memory: 10G
```

### Problem: Getting 503 "Service busy" errors

**Solution:**
```bash
# Increase request queue
INDIC_TRANS_MAX_CONCURRENT=3

# Or reduce backend parallelism
ENVID_METADATA_TRANSLATE_CONCURRENCY=1
```

### Problem: High memory usage but not translating

**Solution:**
```bash
# Force cleanup more aggressively
INDIC_TRANS_MEMORY_LIMIT_PCT=75.0

# Disable warmup to save memory
INDIC_TRANS_WARMUP=false
```

### Problem: Timeouts on large texts

**Solution:**
```bash
# Increase timeouts
INDIC_TRANS_TIMEOUT_SECONDS=180
ENVID_INDIC_TRANS_TIMEOUT_SECONDS=180

# Or reduce text length
INDIC_TRANS_MAX_TEXT_LENGTH=3000
```

## Monitoring Commands

### Check current resource usage:
```bash
docker stats --no-stream translate-indian translate-international
```

### Check health and memory:
```bash
curl http://localhost:5102/health | jq '{ok, memory_usage_pct, memory_available_gb}'
```

### Monitor live stats:
```bash
watch -n 2 'curl -s http://localhost:5102/health | jq'
```

### Check for errors:
```bash
docker logs translate-indian --tail 100 | grep -i "error\|warning\|failed"
```

### View retry attempts:
```bash
docker logs backend --tail 100 | grep "retry"
```

## Recommended Server Specifications

### Minimum
- CPU: 8 cores
- RAM: 24GB
- Storage: 50GB SSD
- Config: Conservative profile

### Recommended
- CPU: 16 cores
- RAM: 32GB
- Storage: 100GB SSD
- Config: Balanced profile

### Optimal
- CPU: 32 cores
- RAM: 64GB
- Storage: 200GB NVMe SSD
- Config: Performance profile

## Quick Reference Card

```bash
# Check health
./check-translation-health.sh

# View logs
docker logs -f translate-indian

# Restart services
docker-compose -f docker-compose.app.yml restart translate-indian

# Monitor resources
docker stats translate-indian translate-international

# Test translation
curl -X POST http://localhost:5102/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello","source_lang":"en","target_lang":"hi"}'
```
