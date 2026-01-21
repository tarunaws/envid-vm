# Local NSFWJS Moderation Service

This service runs **nsfwjs** locally in a Node runtime, separate from the main Envid Multimodal backend.

It implements the same API contract as the local NudeNet service:

- `GET /health`
- `POST /moderate/frames`

## Run (macOS/local)

```bash
cd code/localModerationNSFWJS
./run_local_node.sh
```

Health check:

```bash
curl -sS http://localhost:5082/health
```

## Connect Multimodal Backend

Set env:

- `ENVID_METADATA_LOCAL_MODERATION_NSFWJS_URL=http://localhost:5082`

Then in the UI select `Moderation -> nsfwjs`.

## API

Request:

```json
{ "frames": [ {"time": 12.3, "image_b64": "...", "image_mime": "image/jpeg"} ] }
```

Response:

```json
{ "explicit_frames": [ {"time": 12.3, "likelihood": "LIKELY", "unsafe": 0.81, "safe": 0.19} ] }
```
