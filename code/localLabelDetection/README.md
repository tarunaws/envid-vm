# Local Label Detection Service (Detectron2 + MMDetection)

This service exists because the main project venv is Python 3.14+, while many CV stacks (Detectron2/MMDetection) often require Python 3.11/3.12 + a matching PyTorch build.

## What it does
- Exposes a small HTTP API that the multimodal backend calls when you select **Detectron2** or **MMDetection** in the UI.
- Accepts sampled video frames (JPEG base64) + timestamps.
- Returns normalized labels in the same shape the UI expects.

## Run
```bash
cd code/localLabelDetection
PY_BIN=python3.11 PORT=5083 ./run_local_venv.sh
```

## Docker (recommended for MMDetection)
MMDetection on macOS often fails due to missing MMCV ops (`mmcv.ops`). Running the label service in a Linux container is usually the easiest fix.

If you have a local venv already listening on `5083`, Docker won’t be able to bind the port (and you might accidentally be talking to the wrong process). Free the port first:
```bash
lsof -tiTCP:5083 -sTCP:LISTEN | xargs -r kill -9
```

Known-good Docker combo for MMDetection:
- Python 3.10 base image
- Torch 2.2 CPU
- `mmcv==2.2.0` installed from the OpenMMLab wheel index for `torch2.2.0`

Note: if you’re on Apple Silicon and you need `linux/amd64` wheels, use `docker-compose.amd64.yml`.

```bash
cd code/localLabelDetection
docker compose -f docker-compose.amd64.yml up -d --build
```

If you hit a container-name conflict like:
`container name "/locallabeldetection-local-label-detection-1" is already in use`, clean up and retry:
```bash
cd code/localLabelDetection
docker compose down --remove-orphans || true
docker rm -f locallabeldetection-local-label-detection-1 || true
docker compose -f docker-compose.amd64.yml up -d --build
```

Intel macs / Linux (native):
```bash
cd code/localLabelDetection
docker compose up -d --build
```

Optional: change the host port (still maps to container port 5083):
```bash
cd code/localLabelDetection
ENVID_LOCAL_LABEL_DETECTION_PORT=5084 docker compose -f docker-compose.amd64.yml up -d --build
```

Or via the repo start scripts:
```bash
cd code
ENVID_LOCAL_LABEL_DETECTION_RUNTIME=docker ./start-backend.sh
```

On Apple Silicon, the start scripts will default to `docker-compose.amd64.yml` automatically when Docker runtime is enabled (override with `ENVID_LOCAL_LABEL_DETECTION_DOCKER_COMPOSE_FILE` if you want something else).

To force `linux/amd64` via the start scripts:
```bash
cd code
ENVID_LOCAL_LABEL_DETECTION_RUNTIME=docker \
ENVID_LOCAL_LABEL_DETECTION_DOCKER_COMPOSE_FILE=docker-compose.amd64.yml \
./start-backend.sh
```

Then in another terminal:
```bash
curl -sS http://localhost:5083/health
curl -sS http://localhost:5016/local-label-detection/health
```

The multimodal backend already calls `ENVID_METADATA_LOCAL_LABEL_DETECTION_URL` (default `http://localhost:5083`), so Docker works without extra wiring as long as you publish `5083:5083`.

Health:
- `GET http://localhost:5083/health`

## Notes (macOS)
Detectron2/MMDetection installation on macOS can be difficult depending on your exact Python and PyTorch versions.

### MMDetection: `mmcv._ext` missing
You may see the service health report:
- `has_mmdet: false`
- `mmdet_error` mentioning `mmcv ops missing/unavailable` or `mmcv.ops`

This means MMDetection imported, but MMCV ops are not available (inference will fail).

Best-effort fix (may require Xcode CLT + CMake):
```bash
cd code/localLabelDetection
VENV_DIR=./.venv-labels MMCV_BUILD_FROM_SOURCE=1 ./install_optional_engines.sh
```

Re-check:
- `GET http://localhost:5083/health`

If installs fail:
- Keep **Google Video Intelligence** (cloud) or **YOLO(Ultralytics)** (local) as the label engines.
- Or run Detectron2/MMDetection in Docker/Linux and point the backend to it via `ENVID_METADATA_LOCAL_LABEL_DETECTION_URL`.
