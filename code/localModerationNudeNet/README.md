# Local NudeNet Moderation Service

This service runs NudeNet **locally** in a separate runtime from the main Envid Multimodal backend.

Why: the repo's main venv is Python 3.14+. NudeNet/ONNXRuntime wheels are typically not available for Python 3.14 yet, so we run this in a dedicated Python 3.11/3.12 venv.

## Run (Separate Python venv on macOS, no Docker)

This is the recommended option if you want to run NudeNet **locally** without Docker.

1) Install Python 3.11 (Homebrew):

```bash
brew install python@3.11
```

2) Start the service:

```bash
cd code/localModerationNudeNet
./run_local_venv.sh
```

Health check:

```bash
curl -sS http://localhost:5081/health
```

## API

- `GET /health`
- `POST /moderate/frames`

Request:

```json
{ "frames": [ {"time": 12.3, "image_b64": "...", "image_mime": "image/jpeg"} ] }
```

Response:

```json
{ "explicit_frames": [ {"time": 12.3, "likelihood": "LIKELY", "unsafe": 0.81, "safe": 0.19} ] }
```

## Connect Multimodal Backend

Set env:

- `ENVID_METADATA_LOCAL_MODERATION_URL=http://localhost:5081`

Then in the UI select `Moderation -> NudeNet`.
