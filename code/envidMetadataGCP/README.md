# Envid Metadata (GCP)

GCP-first video metadata extraction service (Flask), intended to run **GCP-only**.

## What it does
- Ingests videos from **Google Cloud Storage** (`gs://<GCP_GCS_BUCKET>/rawVideo/...`) or via `POST /upload-video` (server receives bytes)
- Runs best-effort analysis to produce replayable metadata JSON (transcript, scenes/labels, text, synopses, etc.)
- Serves UI-friendly endpoints for playback (signed GCS URL) and metadata downloads

## Core endpoints
- `GET /health`
- `GET /videos`
- `GET /video/<video_id>`
- `GET /video-file/<video_id>` (redirects to a signed GCS URL for browser playback)
- `GET /video/<video_id>/metadata-json` (query params: `category`, `download=true`)
- `GET /video/<video_id>/metadata-json.zip` (redirects to a signed GCS URL when artifacts are available; otherwise generates on-demand)
- `GET /video/<video_id>/subtitles.srt|.vtt` (redirects to a signed GCS URL when artifacts are available; otherwise serves local if present)
- `GET /video/<video_id>/subtitles.en.srt|.en.vtt` (same behavior for English)
- `GET /gcs/rawvideo/list` (lists objects in GCS under `rawVideo/`)
- `POST /process-gcs-video-cloud` (process an allowed `gs://.../rawVideo/...` object)

## GCP services used (best-effort)
Depending on what you enable/grant:
- Cloud Storage
- Speech-to-Text
- Translate
- Video Intelligence (non-fatal if disabled)
- Natural Language
- Vertex AI (Gemini)

## Configuration
Typical local setup uses a service account:
- `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`
- `GCP_PROJECT_ID=...`
- `GCP_LOCATION=us-central1`
- `GCP_GCS_BUCKET=...`
- `GCP_GCS_RAWVIDEO_PREFIX=rawVideo`

Video Intelligence (shots/labels/text/moderation) is now supported when the API is enabled and IAM is granted:
- `ENVID_METADATA_ENABLE_VIDEO_INTELLIGENCE=true` (default: true)
- `ENVID_METADATA_GCP_VIDEO_INTELLIGENCE_TIMEOUT_SECONDS=3600` (optional)

Artifacts (derived outputs) are stored in GCS as well:
- `ENVID_METADATA_GCP_ARTIFACTS_BUCKET=...` (defaults to `GCP_GCS_BUCKET`)
- `ENVID_METADATA_GCP_ARTIFACTS_PREFIX=envid-metadata/artifacts`
- `GCP_GCS_PRESIGN_SECONDS=3600` (TTL for signed download/playback URLs)

## Notes
- The GCP service only exposes GCP-oriented endpoints; legacy provider-specific routes are not present (Flask `404 Not Found`).
