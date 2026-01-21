# Envid Metadata Use Case — Implementation Checklist (mapped to this repo)

This checklist turns the requirements in [AIUseCases/code/envid-metadataUsecase](AIUseCases/code/envid-metadataUsecase) into concrete deliverables using the services already present in this workspace.

## 1) Scope (what “done” means)

Deliver a working pipeline that, for a given video asset, can:

- Extract **structured metadata** (title-level + scene-level summary artifacts)
- Produce **intelligent tags** (objects, scenes, activities, topics, entities, moods/emotions)
- Detect **celebrities** (names + confidence)
- Enable **semantic search** across the extracted metadata (natural language queries)
- Expose the above via stable **HTTP APIs** that can be called by “Envid AI Studio” (external integration)

> Note: “Netflix/Prime-level metadata” is best interpreted as *coverage* (breadth of fields + consistency), not perfect accuracy on every title without a ground-truth catalog.

## 2) Existing building blocks to reuse (no reinvention)

### Semantic Search service (already does most of the platform)
- Service: [AIUseCases/code/semanticSearch/app.py](AIUseCases/code/semanticSearch/app.py)
- README: [AIUseCases/code/semanticSearch/README.md](AIUseCases/code/semanticSearch/README.md)
- Strengths:
  - Upload & index videos (`/upload-video`), list (`/videos`), detail (`/video/<id>`), query (`/search`)
  - Uses a pluggable analysis pipeline (local OCR + local transcription + embeddings)
  - Stores indices on disk under `semanticSearch/indices/*` (JSON)

### Celebrity detection logic already exists (frame-based)
- Reference implementation: [AIUseCases/code/sceneSummarization/scene_summarization_service.py](AIUseCases/code/sceneSummarization/scene_summarization_service.py)
- Historically it used Rekognition; in this repo the direction is local/GCP/OSS replacements.

### Orchestration patterns exist (good for “platform” framing)
- Service: [AIUseCases/code/mediaSupplyChain/app.py](AIUseCases/code/mediaSupplyChain/app.py)
- Blueprint example: [AIUseCases/code/mediaSupplyChain/blueprints/reference.json](AIUseCases/code/mediaSupplyChain/blueprints/reference.json)
- This is useful if you want an “end-to-end run” artifact trail (ingest → QC → metadata → package).

### Frontend already has “Semantic Search (Video)” UI
- UI: [AIUseCases/code/frontend/src/SemanticSearchVideo.js](AIUseCases/code/frontend/src/SemanticSearchVideo.js)
- It currently targets `http://localhost:5008` directly.

## 3) Target metadata model (pragmatic “Netflix/Prime baseline”)

### 3.1 Title-level fields (one JSON per asset)
Minimum recommended fields to emit:

- `assetId` (uuid)
- `title`, `synopsis` (can be derived by LLM summary + transcript)
- `runtimeSeconds`, `language` (best-effort)
- `genres` / `themes` (LLM classification + keyword/entity cues)
- `tags`:
  - `objects[]`, `scenes[]`, `activities[]` (labels grouped; source depends on selected model)
  - `onScreenText[]` (OCR text)
  - `dominantEmotions[]` (face emotions)
- `people`:
  - `celebrities[]` = `{ name, confidence }`
  - optionally `facesSummary` (age range buckets, gender mix) if needed
- `transcript` (raw + cleaned), `keyQuotes[]` (optional)
- `safety` / `contentWarnings` (optional; can reuse contentModeration patterns later)

### 3.2 Scene-level fields (lightweight)
If you need “rich metadata” beyond title-level:

- segment video into N “representative windows” (evenly spaced is ok)
- store per-window:
  - `startSeconds`, `endSeconds`
  - top labels/tags
  - emotions
  - detected celebrities
  - short natural-language summary

## 4) Recommended implementation approach (fastest path)

### Option A (recommended): extend the existing Semantic Search service
Make `semanticSearch` the “metadata platform” by adding:

1) **Celebrity detection** into the video indexing flow
- Reuse the same frame extraction that semanticSearch already does.
- Add a local/GCP celebrity/face pipeline (or keep this field best-effort until implemented).
- Persist `celebrities[]` into the stored video entry.

2) **Structured metadata endpoint**
Add one endpoint that returns “Netflix-like” metadata JSON for a video already in the index:

- `GET /metadata/video/<video_id>` → returns the structured model above.

(Optionally also: `POST /metadata/upload-video` which uploads + returns metadata immediately, but you may not need a separate endpoint if `/upload-video` already indexes and `/video/<id>` returns details.)

3) **Tag normalization + taxonomy mapping**
Create a small mapping layer:

- Labels → buckets (`objects/scenes/activities`) (sceneSummarization already has a good heuristic)
- Optional: a config file defining “house taxonomy” (genres/themes) and mapping rules

4) **Search stays the same**
Semantic search is already implemented. Ensure the metadata text that is embedded includes:

- transcript + labels + detected text + emotions + celebrities

### Option B: add a new dedicated “metadataExtraction” service
Only do this if you need a strict boundary or separate deployment. Otherwise it’s extra surface area.

## 5) Integration contract for Envid AI Studio (API-first)

Because there’s no Envid AI Studio code in this workspace right now, the safest deliverable is a stable API contract.

Minimum contract:

- `POST /upload-video` (existing) → returns `{ id }`
- `GET /video/<id>` (existing) → full details
- `GET /metadata/video/<id>` (new if needed) → structured metadata
- `POST /search` (existing) → semantic search results

Integration pattern:

- Envid AI Studio uploads asset → stores returned `id`
- Studio requests `/metadata/video/<id>` for display + export
- Studio uses `/search` to power “search across library”

## 6) Dependencies / risks to call out early

- This repo is now oriented to **GCS-first + local processing** (ffmpeg, local whisper, local OCR).
- If you enable GCP analysis, make sure application default credentials + bucket access are configured.
- Accuracy/cost scale with video length and frame sampling regardless of provider.
- If “Netflix/Prime-level” implies ground-truth cast/crew, that typically requires an external catalog (IMDb/Gracenote/etc.) which is outside this repo today
