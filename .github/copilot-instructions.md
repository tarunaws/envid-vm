# Copilot Instructions: Envid Metadata Services

## Architecture Overview

This is a **multi-service video metadata extraction platform** with:
- **Primary Stack ("slim")**: React frontend (port 3000) + Envid Metadata Multimodal service (port 5016)
- **Optional CV Sidecars**: localLabelDetection (5083), localKeySceneBest (5085), localModerationNudenet (5081), localModerationNSFWJS (5082), localOcrPaddle (5084)
- **Legacy Services (deprecated)**: envidMetadata/ (AWS-based, no longer in default stack)

The multimodal backend ([code/envidMetadataGCP/app.py](code/envidMetadataGCP/app.py)) orchestrates GCP services (Speech-to-Text, Translate, Video Intelligence, Vertex AI) and delegates heavy CV tasks to optional local services via HTTP.

## Critical Path Requirements

### Python Version & Environment
- **Main venv requires Python 3.14+** ([code/start-backend.sh#L85-L88](code/start-backend.sh))
- **Sidecar services** (localLabelDetection, localKeySceneBest) use **Python 3.11/3.10** in separate venvs/containers due to PyTorch/Detectron2/MMDetection compatibility
- Always check `code/start-backend.sh` logic for how `VENV_PYTHON` is validated

### Startup Workflow
```bash
# From workspace root or code/ directory:
./start-all.sh          # Starts backend (5016) + frontend (3000)
./start-backend.sh      # Backend only
./stop-all.sh           # Stops all services
```

**Key environment variables** are sourced in this order:
1. `code/.env` (base config)
2. `code/.env.local` (local overrides)
3. `code/.env.multimodal.local` (multimodal-specific config, loaded only for envidMetadataGCP)
4. `code/.env.multimodal.secrets.local` (GCP credentials, not committed)

**PYTHONPATH setup**: [code/start-backend.sh#L24](code/start-backend.sh) exports `PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"` so services can import from `shared/` utilities.

### Docker for CV Services (Apple Silicon)
- **localLabelDetection** (Detectron2/MMDetection) often fails on macOS due to missing MMCV ops → use Docker
- On Apple Silicon, scripts auto-select `docker-compose.amd64.yml` for linux/amd64 emulation ([code/localLabelDetection/docker-compose.amd64.yml](code/localLabelDetection/docker-compose.amd64.yml))
- Before Docker startup, kill existing local processes: `lsof -tiTCP:5083 -sTCP:LISTEN | xargs -r kill -9`

**Startup command**:
```bash
cd code
ENVID_LOCAL_LABEL_DETECTION_RUNTIME=docker ./start-backend.sh
```

## Service Communication Patterns

### Backend → Sidecar Proxying
The multimodal backend ([code/envidMetadataGCP/app.py](code/envidMetadataGCP/app.py)) calls sidecars via env-configured URLs:
- `ENVID_METADATA_LOCAL_LABEL_DETECTION_URL` → http://localhost:5083 (Detectron2/MMDetection)
- `ENVID_METADATA_LOCAL_KEYSCENE_URL` → http://localhost:5085 (TransNetV2/CLIP)
- `ENVID_METADATA_LOCAL_MODERATION_URL` → http://localhost:5081 (NudeNet)
- `ENVID_METADATA_LOCAL_MODERATION_NSFWJS_URL` → http://localhost:5082 (NSFWJS)
- `ENVID_METADATA_LOCAL_OCR_PADDLE_URL` → http://localhost:5084 (PaddleOCR)

These are set with fallback defaults in [code/start-backend.sh#L58-L70](code/start-backend.sh).

### Frontend → Backend Routing
React dev server ([code/frontend/src/setupProxy.js](code/frontend/src/setupProxy.js)) proxies:
- `/envid-multimodal/*` → `http://localhost:5016`

API calls use `REACT_APP_ENVID_MULTIMODAL_BASE` (defaults to `/envid-multimodal`).

## Project-Specific Patterns

### GCP-First Design (envidMetadataGCP)
- Videos are stored in `gs://<GCP_GCS_BUCKET>/rawVideo/`
- Artifacts (subtitles, metadata JSON) go to `gs://<ARTIFACTS_BUCKET>/<ARTIFACTS_PREFIX>/`
- Signed URLs (`GCP_GCS_PRESIGN_SECONDS=3600`) are used for browser playback and downloads
- **Video Intelligence API** is optional (controlled by `ENVID_METADATA_ENABLE_VIDEO_INTELLIGENCE=true`)

### Config Helpers in app.py
Common pattern for environment parsing ([code/envidMetadataGCP/app.py#L337-L345](code/envidMetadataGCP/app.py)):
```python
def _env_truthy(value: str | None, *, default: bool = True) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    return default
```

### Label Categorization Logic
Scene/object/activity bucketing happens in the multimodal backend (inherited pattern from legacy sceneSummarization). When integrating new label sources (YOLO, Detectron2, MMDetection), normalize to `{"label": str, "confidence": float}` format.

### API Endpoint Patterns
Key routes in [code/envidMetadataGCP/app.py](code/envidMetadataGCP/app.py):
- `POST /upload-video` → ingests video bytes
- `POST /process-gcs-video-cloud` → processes existing GCS object
- `GET /videos` → lists all indexed videos
- `GET /video/<video_id>` → returns metadata JSON
- `GET /video/<video_id>/metadata-json?category=scenes&download=true` → category-specific export
- `GET /video/<video_id>/metadata-json.zip` → redirects to signed GCS URL (or generates on-demand)
- `GET /video-file/<video_id>` → redirects to signed GCS URL for playback

## Common Tasks

### Adding a New Service
1. Create service directory under `code/` with `app.py`, `requirements.txt`, `README.md`
2. Add startup logic to [code/start-backend.sh](code/start-backend.sh) using `start_service()` helper
3. Add stop logic to [code/stop-backend.sh](code/stop-backend.sh)
4. If backend needs to call it, add URL env var in [code/start-backend.sh#L58-L70](code/start-backend.sh)
5. Update [SERVICES_README.md](SERVICES_README.md) with port, purpose, health endpoint

### Debugging Service Startup
- Check logs: `code/<service-name>.log` (created by `nohup` in start scripts)
- Verify health: `curl -fsS http://localhost:<PORT>/health`
- PID files: `code/<service-name>.pid`
- Check `PYTHONUNBUFFERED=1` is set (ensures immediate log output)

### Working with GCS
- Service account JSON must be set: `GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json`
- Test GCS listing: `GET /gcs/rawvideo/list`
- Upload flow: `POST /upload-video` (multipart) → generates UUID → stores in `gs://.../rawVideo/<uuid>_<filename>`

### Docker Troubleshooting (CV Services)
If container name conflicts occur:
```bash
cd code/localLabelDetection
docker compose down --remove-orphans || true
docker rm -f locallabeldetection-local-label-detection-1 || true
docker compose -f docker-compose.amd64.yml up -d --build
```

## Conventions to Follow

- **No AWS in slim stack**: Legacy AWS-based services (envidMetadata/) are deprecated; focus on GCP-first implementations
- **Type hints**: Use `from __future__ import annotations` and modern type syntax (`str | None` not `Optional[str]`)
- **Error handling**: Services should return `jsonify({"error": "..."}), 4xx` for user errors; log exceptions to Sentry if `SENTRY_DSN` is set
- **Health endpoints**: Always implement `@app.route("/health")` returning `{"status": "healthy", ...}` with service-specific diagnostics
- **Env-first config**: No hardcoded paths/URLs; use `os.getenv()` with sensible defaults
- **PATH management**: [code/start-backend.sh#L26-L41](code/start-backend.sh) uses `maybe_prepend_path()` to avoid clobbering user PATH (especially important for NVM/Node)

## Files to Reference When

- **Architecture questions**: [SERVICES_README.md](SERVICES_README.md), [code/ENVID_METADATA_IMPLEMENTATION_CHECKLIST.md](code/ENVID_METADATA_IMPLEMENTATION_CHECKLIST.md)
- **Service startup logic**: [code/start-backend.sh](code/start-backend.sh), [start-all.sh](start-all.sh)
- **Frontend API integration**: [code/frontend/src/setupProxy.js](code/frontend/src/setupProxy.js)
- **Docker configs**: [code/localLabelDetection/docker-compose.amd64.yml](code/localLabelDetection/docker-compose.amd64.yml)
- **Main backend implementation**: [code/envidMetadataGCP/app.py](code/envidMetadataGCP/app.py) (6400+ lines; search for `@app.route` to find endpoints)
- **Deprecated patterns (avoid)**: [code/envidMetadata/](code/envidMetadata/) (AWS Rekognition/Transcribe, no longer maintained)
