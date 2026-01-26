# Copilot Instructions for Envid Metadata

## Big picture
- Two stacks coexist: the active monolith in [code/envidMetadataGCP](code/envidMetadataGCP) and the microservices migration in [microservices](microservices).
- The monolith orchestrator lives in [code/envidMetadataGCP/envidMetadataGCP.py](code/envidMetadataGCP/envidMetadataGCP.py) and drives upload → processing → artifact storage → metadata APIs.
- Frontend (CRA) in [code/frontend](code/frontend) calls the backend over REST and polls job status; proxy rules live in [code/frontend/src/setupProxy.js](code/frontend/src/setupProxy.js).
- Target service boundaries are captured in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml); all service Dockerfiles/configs belong under [microservices](microservices).

## Critical workflows
- Monolith stack: [allInOne/start-all.sh](allInOne/start-all.sh) / [allInOne/stop-all.sh](allInOne/stop-all.sh). Backend-only: [allInOne/start-backend.sh](allInOne/start-backend.sh).
- Microservices stack: [microservices/start-services.sh](microservices/start-services.sh) / [microservices/stop-services.sh](microservices/stop-services.sh).
- Backend env load order (scripted): code/.env → code/.env.local → code/.env.multimodal.local → code/.env.multimodal.secrets.local (see [allInOne/.github/copilot-instructions.md](allInOne/.github/copilot-instructions.md)).
- LibreTranslate local setup lives under [allInOne](allInOne) and uses make up/down/logs + make models (see [allInOne/README.md](allInOne/README.md)).

## Integration points & data flow
- Offload steps via ENVID_*_SERVICE_URL env vars (ingest, ffmpeg, OCR, moderation, WhisperX, scenes, synopsis, export) wired in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml).
- Services expose /health; backend also provides proxy health routes for sidecars.
- Artifacts and raw videos are stored in GCS buckets as described in [code/envidMetadataGCP/README.md](code/envidMetadataGCP/README.md).

## Project-specific patterns
- Boolean env parsing uses _env_truthy in [code/envidMetadataGCP/app.py](code/envidMetadataGCP/app.py).
- New services should be HTTP-based with small request/response contracts; preserve output schemas consumed by [code/frontend/src/EnvidMetadataMinimal.js](code/frontend/src/EnvidMetadataMinimal.js).
- Wire new services through Envoy ([microservices/gateway](microservices/gateway)) and Nginx ([microservices/reverseproxy](microservices/reverseproxy)).

## When adding or changing services
1. Put Dockerfile/config in [microservices](microservices) and add the service to [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml).
2. Add the corresponding ENVID_*_SERVICE_URL and update the monolith integration in [code/envidMetadataGCP/envidMetadataGCP.py](code/envidMetadataGCP/envidMetadataGCP.py).
3. If the API output changes, update the UI in [code/frontend/src/EnvidMetadataMinimal.js](code/frontend/src/EnvidMetadataMinimal.js).

## High-signal docs
- [allInOne/CODEBASE_OVERVIEW.md](allInOne/CODEBASE_OVERVIEW.md)
- [microservices/README.md](microservices/README.md)
- [code/envidMetadataGCP/README.md](code/envidMetadataGCP/README.md)
