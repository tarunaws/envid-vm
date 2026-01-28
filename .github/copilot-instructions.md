# Copilot Instructions for Envid Metadata

## Big picture (where to start)
- Two parallel stacks: the active monolith in [code/envidMetadataGCP](code/envidMetadataGCP) and the microservices migration in [microservices](microservices).
- Monolith orchestration: [code/envidMetadataGCP/envidMetadataGCP.py](code/envidMetadataGCP/envidMetadataGCP.py) handles upload → processing → artifacts → metadata APIs (see pipeline in [allInOne/CODEBASE_OVERVIEW.md](allInOne/CODEBASE_OVERVIEW.md)).
- Microservices orchestration: [microservices/backend/code/backend.py](microservices/backend/code/backend.py), with target topology in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml).
- Frontend (CRA) in [code/frontend](code/frontend) proxies `/backend/*` to port 5016 (see [code/frontend/src/setupProxy.js](code/frontend/src/setupProxy.js)).
- Legacy AWS service in [code/envidMetadata](code/envidMetadata) is deprecated.

## Critical workflows (local)
- All-in-one stack: [allInOne/start-all.sh](allInOne/start-all.sh) / [allInOne/stop-all.sh](allInOne/stop-all.sh). Backend-only: [allInOne/start-backend.sh](allInOne/start-backend.sh).
- Microservices stack: [microservices/start-services.sh](microservices/start-services.sh) / [microservices/stop-services.sh](microservices/stop-services.sh).
- Backend env load order (scripted in [allInOne/start-backend.sh](allInOne/start-backend.sh)): `code/.env` → `code/.env.local` → `code/.env.multimodal.local` → `code/.env.multimodal.secrets.local`.
- Local LibreTranslate (used by translation provider) is managed in [allInOne](allInOne) via `make up/down/logs/models` (details in [allInOne/README.md](allInOne/README.md)).

## Integration points & data flow
- Offload pipeline steps with `ENVID_*_SERVICE_URL` (ingest, ffmpeg, OCR, moderation, transcription, scenes, summarizer, export) wired in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml).
- Services expose `/health`; backend also provides proxy health routes for sidecars.
- Artifacts and raw videos are stored in GCS buckets (see [code/envidMetadataGCP/README.md](code/envidMetadataGCP/README.md)).

## Project-specific conventions
- Boolean env parsing uses `_env_truthy` in [code/envidMetadataGCP/app.py](code/envidMetadataGCP/app.py).
- New services are HTTP-based with small request/response contracts; preserve output schemas consumed by [code/frontend/src/EnvidMetadataMinimal.js](code/frontend/src/EnvidMetadataMinimal.js).
- Route new services through Envoy in [microservices/gateway](microservices/gateway) and the Nginx front door in [microservices/reverseproxy](microservices/reverseproxy).

## When adding or changing services
1. Add Dockerfile/config under [microservices](microservices) and wire the service in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml).
2. Add the matching `ENVID_*_SERVICE_URL` and update monolith integration in [code/envidMetadataGCP/envidMetadataGCP.py](code/envidMetadataGCP/envidMetadataGCP.py).
3. If the API output changes, update UI parsing in [code/frontend/src/EnvidMetadataMinimal.js](code/frontend/src/EnvidMetadataMinimal.js).

## High-signal docs
- [allInOne/CODEBASE_OVERVIEW.md](allInOne/CODEBASE_OVERVIEW.md)
- [microservices/README.md](microservices/README.md)
- [code/envidMetadataGCP/README.md](code/envidMetadataGCP/README.md)
