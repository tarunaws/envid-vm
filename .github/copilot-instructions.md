# Copilot Instructions: Envid Metadata Services

## Big picture (where to start)
- Primary stack is **React UI + multimodal backend**: [code/frontend](code/frontend) on port 3000, [code/envidMetadataGCP/app.py](code/envidMetadataGCP/app.py) on port 5016.
- Optional local CV sidecars are separate services invoked over HTTP (ports 5081–5085); see [SERVICES_README.md](SERVICES_README.md).
- Legacy AWS-based services live under [code/envidMetadata](code/envidMetadata) and are not part of the slim stack.

## Critical workflows
- Start/stop the slim stack via [start-all.sh](start-all.sh) / [stop-all.sh](stop-all.sh). Backend-only: [code/start-backend.sh](code/start-backend.sh).
- Env load order is enforced in [code/start-backend.sh](code/start-backend.sh): `code/.env` → `code/.env.local` → `code/.env.multimodal.local` → `code/.env.multimodal.secrets.local`.
- Main backend venv requires **Python 3.14+**; sidecars often need **Python 3.10/3.11** due to ML wheels.
- Logs and PIDs are created by startup scripts: `code/<service>.log` and `code/<service>.pid`.

## Service boundaries & data flow
- Backend orchestrates GCS I/O, FFmpeg preprocess, Whisper transcription, LLM summarization, and proxies to sidecars. See [code/envidMetadataGCP/app.py](code/envidMetadataGCP/app.py).
- Frontend calls backend through the proxy in [code/frontend/src/setupProxy.js](code/frontend/src/setupProxy.js), using `REACT_APP_ENVID_MULTIMODAL_BASE`.
- GCS layout: raw videos in `gs://<GCP_GCS_BUCKET>/rawVideo/`; artifacts in `gs://<ARTIFACTS_BUCKET>/<ARTIFACTS_PREFIX>/`.

## Project-specific conventions
- Fixed model policy is enforced in [code/envidMetadataGCP/app.py](code/envidMetadataGCP/app.py) (see pipeline steps in [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md)); do not re-enable deprecated model paths.
- Local translation uses LibreTranslate via `ENVID_METADATA_TRANSLATE_PROVIDER=libretranslate` and `ENVID_LIBRETRANSLATE_URL` (see [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md)).
- Env parsing helper `_env_truthy` in [code/envidMetadataGCP/app.py](code/envidMetadataGCP/app.py) is the canonical pattern for booleans.
- Health endpoints are required for services (`/health`); backend also exposes proxy health routes for sidecars.

## Integration points (URLs/ports)
- Backend → sidecars URLs are env-driven (defaults set in [code/start-backend.sh](code/start-backend.sh)).
- Frontend → backend uses `/backend/*` proxy to port 5016.

## When adding or changing services
- Add a new service under [code](code), wire it into [code/start-backend.sh](code/start-backend.sh) + [code/stop-backend.sh](code/stop-backend.sh), and document in [SERVICES_README.md](SERVICES_README.md).

## High-signal files
- Backend core: [code/envidMetadataGCP/app.py](code/envidMetadataGCP/app.py)
- Frontend entry & routing: [code/frontend/src/App.js](code/frontend/src/App.js)
- Proxy config: [code/frontend/src/setupProxy.js](code/frontend/src/setupProxy.js)
- Service catalog & ports: [SERVICES_README.md](SERVICES_README.md)
- Pipeline overview: [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md)
