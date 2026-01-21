# localKeySceneBest (TransNetV2 + CLIP clustering)

Optional sidecar to enable the **best-combo** key scene pipeline:
- **TransNetV2** for shot/scene boundary detection
- **CLIP** embeddings + clustering for diversity selection

## Run (Docker)
From `code/localKeySceneBest/`:

- `docker compose up --build -d`
- Health: `curl -fsS http://localhost:5085/health`

## One-command bootstrap (recommended)
From `code/`:

- `./scripts/bootstrap_keyscene_combo.sh`

This script:
- Ensures Docker daemon is reachable
- Ensures TransNetV2 weights are present under `code/localKeySceneBest/weights/transnetv2-weights/`
- Starts the sidecar and then starts the backend services

## TransNetV2 weights
The upstream TransNetV2 repo stores weights under `inference/transnetv2-weights/` and commonly uses Git LFS.
This service supports supplying weights via a mounted directory.

1) Place weights under:
- `code/localKeySceneBest/weights/transnetv2-weights/` (directory must contain `saved_model.pb` and `variables/`)

2) Set env:
- `TRANSNETV2_MODEL_DIR=/weights/transnetv2-weights`

If weights are missing, `/transnetv2/scenes` will return an error, and the backend should fall back to VI shots or PySceneDetect.

## API
- `GET /health`
- `POST /clip/cluster` JSON `{ "images_b64": ["..."], "k": 10 }`
- `POST /transnetv2/scenes` multipart form with `video=@file.mp4`
