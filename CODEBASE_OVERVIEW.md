# Envid Metadata — Frontend & Backend Code Overview

## Repository layout

- code/envidMetadataGCP/: primary multimodal backend (Flask) and all pipeline logic.
- code/frontend/: React UI (Create React App) that drives the multimodal backend.
- code/localKeySceneBest/: TransNetV2 + optional CLIP clustering sidecar (key scenes).
- code/localModerationNudeNet/: NudeNet sidecar (local moderation).
- code/localOcrPaddle/: PaddleOCR sidecar (unused in current fixed pipeline).
- code/localLabelDetection/: optional Detectron2/MMDetection sidecar (unused in fixed pipeline).
- start-all.sh / start-backend.sh / stop-backend.sh: local orchestration scripts.
- SERVICES_README.md: service list and ports.

## Frontend (code/frontend)

### Entry points

- src/index.js: mounts React app into the DOM.
- src/App.js: main router, top nav, login gate, and route wiring.
- src/setupProxy.js: dev proxy for /envid-multimodal → http://localhost:5016.

### Main UX for metadata

- src/EnvidMetadataMinimal.js: main UI for upload, job status, and metadata result browsing.
  - Uses axios to call backend endpoints.
  - Uses BACKEND_URL (REACT_APP_ENVID_METADATA_BACKEND_URL or /envid-multimodal).
  - Displays the pipeline steps list that mirrors backend job steps.
  - Polls job status and renders progress.

### Routing & pages

- App.js defines routes for the use case pages and maps /envid-metadata and /envid-metadata/multimodal to EnvidMetadataMinimal.
- Several legacy demo routes are redirected to the metadata page (GCS-only frontend).

## Backend (code/envidMetadataGCP)

### Core service

- app.py: Flask service that implements upload, processing, storage, and metadata APIs.
- Handles GCS I/O, FFmpeg preprocessing, Whisper transcription, OpenRouter Llama summaries, label detection, moderation, OCR, scene detection, and exports.

### Key endpoints (selected)

- POST /upload-video: multipart upload to GCS and queue processing.
- POST /process-gcs-video-cloud: process existing GCS object.
- GET /videos: list indexed videos.
- GET /video/<video_id>: fetch metadata.
- GET /video/<video_id>/metadata-json: export category JSON.
- GET /video/<video_id>/metadata-json.zip: download a zip archive.
- GET /video-file/<video_id>: signed URL for video playback.
- GET /health: health info and dependency checks.

### Pipeline steps

Backend job steps (mirrored in the UI):

1. upload_to_cloud_storage
2. precheck_models
3. technical_metadata
4. transcode_normalize
5. label_detection
6. moderation
7. text_on_screen
8. key_scene_detection
9. transcribe
10. synopsis_generation
11. scene_by_scene_metadata
12. famous_location_detection (disabled)
13. translate_output
14. opening_closing_credit_detection (disabled)
15. celebrity_detection (not implemented)
16. celebrity_bio_image (not implemented)
17. save_as_json
18. overall

### Fixed model policy (current)

The backend is forced to use only the models below:

- Label detection: Google Video Intelligence only.
- Moderation: NudeNet only (local).
- Text on screen: Tesseract only (local).
- Key scene detection: TransNetV2 + CLIP clustering sidecar.
- Transcribe: Whisper large-v3 only.
- Synopsis generation: OpenRouter Meta Llama only.
- Scene-by-scene summaries: OpenRouter Meta Llama only.
- Translation: LibreTranslate only (no GCP fallback by default).
- Famous locations: disabled.
- Opening/closing credits: disabled.
- No Google services are used except Video Intelligence label detection.

### How the backend selects models

The model selection logic lives in app.py inside the GCS processing path. It:

- Parses UI task selection and requested models.
- Overrides model choices to the fixed policy.
- Runs prechecks (local sidecar health, required libs, keys).
- Executes each step and stores results in GCS.

### GCS & environment configuration

The backend relies on environment variables defined in code/.env.multimodal.local:

- GCP_PROJECT_ID, GCP_LOCATION
- GOOGLE_APPLICATION_CREDENTIALS
- GCP_GCS_BUCKET (raw videos)
- ENVID_METADATA_GCS_BUCKET (artifacts)
- OPENROUTER_API_KEY and OPENROUTER_* model settings
- ENVID_METADATA_TRANSLATE_PROVIDER=libretranslate
- ENVID_LIBRETRANSLATE_URL

## Sidecars (local services)

- localKeySceneBest (port 5085): TransNetV2 + optional CLIP clustering; called via ENVID_METADATA_LOCAL_KEYSCENE_URL.
- localModerationNudeNet (port 5081): NudeNet; called via ENVID_METADATA_LOCAL_MODERATION_URL.
- localOcrPaddle (port 5084): PaddleOCR sidecar; not used by the fixed policy.
- localLabelDetection (port 5083): Detectron2/MMDetection; not used by the fixed policy.

## Scripts

- code/start-backend.sh: starts multimodal backend and local sidecars.
- code/stop-backend.sh: stops all services.
- start-all.sh / stop-all.sh: wrapper scripts that include the frontend.

## Notes on removed items

- GCP Translate, GCP Language, and GCP Speech dependencies were removed from requirements.
- NSFWJS moderation Docker/Node autostart and stop hooks were removed; NudeNet is the only local moderation now.

