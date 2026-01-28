# Microservices migration plan

## Current state
- The backend service is the orchestrator; the entrypoint is [backend/code/backend.py](backend/code/backend.py).
- [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml) already models the target service topology (gateway, backend, sidecars, auth, observability).
- Base Dockerfiles/configs now live under [microservices/](microservices) (gateway, backend, frontend, sidecars, auth, observability). Use this folder for all microservice assets.

## Goal (target service boundaries)
Split the pipeline into discrete services aligned with existing steps in [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md):
- ingest + storage (GCS I/O, upload, artifact management)
- preprocess (ffmpeg normalization)
- label detection
- moderation (NudeNet)
- OCR (Tesseract)
- key-scene detection (TransNetV2 + CLIP)
- transcription (OpenAI Whisper)
- summarization (LLM synopsis + scene-by-scene)
- export (JSON/zip/subtitles)

## Phase 1 (start implementation)
- Treat [microservices/](microservices) as the single home for service Dockerfiles/config.
- Use [microservices/start-services.sh](microservices/start-services.sh) and [microservices/stop-services.sh](microservices/stop-services.sh) to run [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml).
- Keep existing sidecars functional while services are extracted from the monolith.
- Uploads can be offloaded to the ingest service by setting `ENVID_INGEST_SERVICE_URL` (wired in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml)).
- Transcode normalization can be offloaded to the transcoder service via `ENVID_FFMPEG_SERVICE_URL` (wired in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml)).
- Label detection is handled inside the backend via Google Video Intelligence.
- Text-on-screen OCR can be offloaded to the OCR service via `ENVID_OCR_SERVICE_URL` (wired in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml)).
- Moderation uses the NudeNet service via `ENVID_MODERATION_SERVICE_URL` (wired in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml)).
- Transcription can be offloaded via `ENVID_TRANSCRIBE_SERVICE_URL` (wired in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml)).
- Scene detection can be offloaded to the scene service via `ENVID_SCENE_SERVICE_URL` (wired in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml)).
- LLM synopsis + scene-by-scene summarization can be offloaded via `ENVID_SUMMARIZER_SERVICE_URL` (wired in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml)).
- Export (artifact upload/zip/subtitles) can be offloaded via `ENVID_METADATA_EXPORT_SERVICE_URL` (wired in [microservices/docker-compose.app.yml](microservices/docker-compose.app.yml)).

## Phase 2 (extraction)
- Carve out each step above into an HTTP service with `/health` and a small request/response contract.
- Replace in-process calls inside [backend/code/backend.py](backend/code/backend.py) with HTTP calls to the new services.
- Version API contracts and preserve output schemas consumed by [code/frontend/src/EnvidMetadataMinimal.js](code/frontend/src/EnvidMetadataMinimal.js).

## Phase 3 (stabilization)
- Add per-service Dockerfiles/env under [microservices/](microservices).
- Wire all services through Envoy in [microservices/gateway](microservices/gateway) and the Nginx front door in [microservices/reverseproxy](microservices/reverseproxy).
