# Copilot Instructions for Envid Metadata

## Big picture (where to start)
- Microservices orchestration: [microservices/backend/code/backend.py](microservices/backend/code/backend.py), with target topology in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml).
- Frontend (CRA) in [code/frontend](code/frontend) proxies `/backend/*` to port 5016 (see [code/frontend/src/setupProxy.js](code/frontend/src/setupProxy.js)).
- Legacy AWS service in [code/envidMetadata](code/envidMetadata) is deprecated.

## Critical workflows (local)
- Microservices stack: [microservices/start-services.sh](microservices/start-services.sh) / [microservices/stop-services.sh](microservices/stop-services.sh).

## Integration points & data flow
- Offload pipeline steps with `ENVID_*_SERVICE_URL` (ingest, ffmpeg, OCR, moderation, transcription, scenes, summarizer, export) wired in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml).
- Services expose `/health`; backend also provides proxy health routes for sidecars.
- Artifacts and raw videos are stored in GCS buckets.

## Project-specific conventions
- New services are HTTP-based with small request/response contracts; preserve output schemas consumed by [code/frontend/src/EnvidMetadataMinimal.js](code/frontend/src/EnvidMetadataMinimal.js).
- Route new services through Envoy in [microservices/gateway](microservices/gateway) and the Nginx front door in [microservices/reverseproxy](microservices/reverseproxy).

## When adding or changing services
1. Add Dockerfile/config under [microservices](microservices) and wire the service in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml).
2. Add the matching `ENVID_*_SERVICE_URL` and update integration in [microservices/backend/code/backend.py](microservices/backend/code/backend.py).
3. If the API output changes, update UI parsing in [code/frontend/src/EnvidMetadataMinimal.js](code/frontend/src/EnvidMetadataMinimal.js).

## High-signal docs
- [microservices/README.md](microservices/README.md)
