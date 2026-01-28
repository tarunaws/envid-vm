# Copilot Instructions: Envid Metadata Services

## Big picture (where to start)
- Two parallel stacks: the **all-in-one monolith** in [code/envidMetadataGCP](code/envidMetadataGCP) and the **microservices migration** in [microservices](microservices).
- The monolith remains the active orchestrator; it performs GCS I/O, ffmpeg preprocessing, OpenAI Whisper transcription, LLM summarization, and optional sidecar calls (see [code/envidMetadataGCP/README.md](code/envidMetadataGCP/README.md)).
- Microservices are defined in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml) and are gradually replacing monolith steps (see [microservices/README.md](microservices/README.md)).
- Legacy AWS service under [code/envidMetadata](code/envidMetadata) is deprecated and not part of the default stack.

## Critical workflows (local)
- All-in-one stack: [start-all.sh](start-all.sh) / [stop-all.sh](stop-all.sh). Backend-only: [code/start-backend.sh](code/start-backend.sh).
- Microservices stack: [microservices/start-services.sh](microservices/start-services.sh) / [microservices/stop-services.sh](microservices/stop-services.sh).
- Env load order for backend is enforced in [code/start-backend.sh](code/start-backend.sh): `code/.env` → `code/.env.local` → `code/.env.multimodal.local` → `code/.env.multimodal.secrets.local`.
- Python version split: monolith expects Python 3.14+; sidecars like NudeNet run in separate Python 3.11/3.12 venvs (see [code/localModerationNudeNet/README.md](code/localModerationNudeNet/README.md)).
- Startup scripts create logs/PIDs under `code/<service>.log` and `code/<service>.pid`.

## Service boundaries & data flow
- Frontend (CRA) runs on port 3000 and proxies `/backend/*` to the backend on port 5016 (see [code/frontend/src/setupProxy.js](code/frontend/src/setupProxy.js)).
- Backend stores raw videos under `gs://<GCP_GCS_BUCKET>/rawVideo/` and artifacts under `gs://<ARTIFACTS_BUCKET>/<ARTIFACTS_PREFIX>/` (see [code/envidMetadataGCP/README.md](code/envidMetadataGCP/README.md)).
- Offload points are environment-driven in microservices: `ENVID_*_SERVICE_URL` values are wired in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml).

## Project-specific conventions
- Boolean env parsing uses `_env_truthy` in [code/envidMetadataGCP/app.py](code/envidMetadataGCP/app.py); follow this when adding new env flags.
- Services expose `/health` endpoints; the backend includes proxy health routes for sidecars.
- Translation provider is LibreTranslate when `ENVID_METADATA_TRANSLATE_PROVIDER=libretranslate` and `ENVID_LIBRETRANSLATE_URL` are set (see [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md)).

## When adding or changing services
- Monolith: add/wire scripts in [code/start-backend.sh](code/start-backend.sh) and [code/stop-backend.sh](code/stop-backend.sh); document in [SERVICES_README.md](SERVICES_README.md).
- Microservices: add Dockerfiles/config under [microservices](microservices) and wire URLs in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml).

## High-signal files
- Monolith backend: [code/envidMetadataGCP/app.py](code/envidMetadataGCP/app.py)
- Monolith API contract: [code/envidMetadataGCP/README.md](code/envidMetadataGCP/README.md)
- Migration map: [microservices/README.md](microservices/README.md)
- Frontend proxy: [code/frontend/src/setupProxy.js](code/frontend/src/setupProxy.js)
- Pipeline overview: [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md)
