# MediaGenAI Services Management

This document explains how to start, stop, and manage the MediaGenAI services.

## 🚀 Quick Start

> **Important:** Ensure the required toolchain is on PATH before invoking any service. Run:
> ```bash
> export PATH="/usr/local/bin/ffmpeg/:/opt/homebrew/bin/:/Users/tarun_bhardwaj/.nvm/versions/node/v22.20.0/bin:/Users/tarun_bhardwaj/Library/Python/3.14/lib/python/site-packages/:/Users/tarun_bhardwaj/Library/Python/3.14/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Users/tarun_bhardwaj/.vscode/extensions/ms-python.debugpy-2025.14.1-darwin-arm64/bundled/scripts/noConfigScripts:/Users/tarun_bhardwaj/Library/Application\ Support/Code/User/globalStorage/github.copilot-chat/debugCommand"
> ```
> The startup scripts automatically enforce this PATH, but manual service runs must export it first.

### Start All Services
```bash
./start-all.sh
```
By default, this starts the slim stack:
- Envid Metadata (Multimodal) (Port 5016)
- React Frontend (Port 3000)

Other services exist in this repo but are disabled in the startup scripts by default.

### Stop All Services
```bash
./stop-all.sh
```

## 🔧 Individual Service Management

### Backend Services Only
```bash
# Start backend services
./start-backend.sh

# Stop backend services
./stop-backend.sh
```

### Optional: Local Key Scene Best (TransNetV2 + CLIP)
The Envid Metadata (Multimodal) backend can call a separate local key-scene sidecar (default `http://localhost:5085`) when you select:
- `transnetv2_clip_cluster`
- `pyscenedetect_clip_cluster`

One-command bootstrap (recommended):
```bash
cd code
./scripts/bootstrap_keyscene_combo.sh
```

Manual run:
```bash
cd code/localKeySceneBest
docker compose up -d --build
curl -fsS http://localhost:5085/health
```

Note: `transnetv2_clip_cluster` requires TransNetV2 SavedModel weights under:
`code/localKeySceneBest/weights/transnetv2-weights/`.

### Frontend Only
```bash
# Start frontend (from frontend directory)
cd frontend
npm start

# Stop frontend: Ctrl+C in the terminal

```

## 🌐 Service URLs

- **Frontend Application**: http://localhost:3000
- **Envid Metadata (Multimodal)**: http://localhost:5016
- **AI Subtitle Service**: http://localhost:5001
- **Image Creation Service**: http://localhost:5002
- **Synthetic Voiceover Service**: http://localhost:5003
- **Scene Summarization Service**: http://localhost:5004
- **Movie Script Creation Service**: http://localhost:5005
- **Content Moderation Service**: http://localhost:5006
- **Personalized Trailer Service**: http://localhost:5007
- **OpenAI Whisper Service**: http://localhost:5088

## 📊 Port Configuration

| Service | Port | Purpose |
|---------|------|---------|
| Frontend (React) | 3000 | Main web application |
| Envid Metadata (Multimodal) | 5016 | Envid metadata service (multimodal) |
| AI Subtitle Service | 5001 | Video subtitle generation |
| Image Creation Service | 5002 | AI image generation |
| Synthetic Voiceover Service | 5003 | SSML generation & neural speech synthesis |
| Scene Summarization Service | 5004 | Scene analysis, narrative summaries, voiceover preview |
| Movie Script Creation Service | 5005 | Feature-length screenplay generation with audience and region targeting |
| Content Moderation Service | 5006 | Video upload with automated moderation timelines |
| Personalized Trailer Service | 5007 | AI-generated trailer planning and packaging workflows |
| OpenAI Whisper Service | 5088 | GPU-first transcription |
## 🔧 Configuration

### Environment Variables
```bash
REACT_APP_SUBTITLE_API_BASE=http://localhost:5001
REACT_APP_API_BASE=http://localhost:5002
REACT_APP_VOICE_API_BASE=http://localhost:5003
REACT_APP_SCENE_API_BASE=http://localhost:5004
REACT_APP_SCRIPT_API_BASE=http://localhost:5005
REACT_APP_MODERATION_API_BASE=http://localhost:5006
REACT_APP_TRAILER_API_BASE=http://localhost:5007

# Envid Metadata UI routing (defaults use the dev proxy paths)
REACT_APP_ENVID_MULTIMODAL_BASE=/backend

```

Optional: The Movie Script Creation service reads AWS credentials and Bedrock settings (for example `BEDROCK_REGION`, `MOVIE_SCRIPT_MODEL_ID`) from the standard environment.

### Secrets & Observability

- Set `APP_SECRETS_ID` (or service-specific `AI_SUBTITLE_SECRET_ID` / `IMAGE_CREATION_SECRET_ID`) to automatically hydrate AWS credentials, SMTP passwords, and other sensitive values from AWS Secrets Manager before each service reads its configuration.
- Provide a `SENTRY_DSN` (or `AI_SUBTITLE_SENTRY_DSN`, `IMAGE_SENTRY_DSN`) to enable structured error reporting via Sentry; adjust `SENTRY_TRACES_SAMPLE_RATE` and `SENTRY_ENVIRONMENT` as needed.
- Rate limiting now protects the heaviest endpoints. Tune `DEFAULT_RATE_LIMIT`, `UPLOAD_RATE_LIMIT`, `GENERATE_RATE_LIMIT`, `IMAGE_PROMPT_RATE_LIMIT`, `CONTACT_RATE_LIMIT`, and optionally `RATE_LIMIT_STORAGE_URI` (Redis or Memcached URI) to match your deployment capacity.

## 📝 Log Files

When using the startup scripts, logs are saved to:
- `ai-subtitle.log` — AI Subtitle Service logs
- `image-creation.log` — Image Creation Service logs
- `synthetic-voiceover.log` — Synthetic Voiceover Service logs
- `scene-summarization.log` — Scene Summarization Service logs
- `movie-script.log` — Movie Script Creation Service logs
- `content-moderation.log` — Content Moderation Service logs
- `personalized-trailer.log` — Personalized Trailer Service logs

## ♻️ Artifact Cleanup Automation

- The AI Subtitle service now prunes `uploads/`, `audio/`, and `outputs/` on a rolling basis. Control retention with `ARTIFACT_RETENTION_HOURS` (default `72`) and how often the background sweeper runs via `ARTIFACT_CLEANUP_INTERVAL_MINUTES` (default `30`).
- Set `ENABLE_ARTIFACT_CLEANUP=0` if you need to pause the background janitor thread temporarily.
- For manual or cron-driven cleanup, run `python scripts/prune_media_artifacts.py --retention-hours 48` (it accepts custom paths if you also need to purge image history or other folders).

### Backend Port Configuration

You can override the default ports using environment variables:

```bash
# For AI Subtitle Service
PORT=5001 DEBUG=false RELOADER=false python aiSubtitle/aiSubtitle.py

# For Image Creation Service
PORT=5002 DEBUG=false RELOADER=false python imageCreation/app.py

# For Synthetic Voiceover Service
PORT=5003 DEBUG=false RELOADER=false python syntheticVoiceover/app.py

# For Scene Summarization Service
PORT=5004 DEBUG=false RELOADER=false python sceneSummarization/app.py

# For Personalized Trailer Service
PERSONALIZED_TRAILER_PORT=5007 DEBUG=false RELOADER=false python personalizedTrailer/app.py
```

All backend services honour these flags:

- `DEBUG=true|false` (default `false`)
- `RELOADER=true|false` (default `false`)

In production or when launching via the helper scripts, keep DEBUG and RELOADER disabled to avoid duplicate listeners and extra processes.

## 🐛 Troubleshooting

### Ports Already in Use
- 5001 — AI Subtitle Service
- 5002 — Image Creation Service
- 5003 — Synthetic Voiceover Service
- 5004 — Scene Summarization Service
- 5005 — Movie Script Creation Service
- 5006 — Content Moderation Service
- 5007 — Personalized Trailer Service
- 5016 — Envid Metadata (Multimodal)
- 3000 — React Frontend

Check and free a port if needed:

```bash
lsof -i :5003
kill <PID>
```

### Services Won't Start
1. Ensure the Python virtual environment is activated.
2. Confirm cloud credentials are available for services that call cloud providers (GCP ADC for the metadata services; AWS credentials for optional AWS-backed services).
3. Verify ports are free using the commands above.

### Check Service Status
```bash
# Check if services are running
ps aux | grep -E "(aiSubtitle|imageCreation|syntheticVoiceover|sceneSummarization|contentModeration|personalizedTrailer|react-scripts)"

# Check port usage
netstat -an | grep -E "(3000|5001|5002|5003|5004|5005|5006|5007|5016)"

```

## ☁️ Deployment to EC2

The repository ships with a helper script that keeps the production instance on
your EC2 host in sync with the local development copy.

### Prerequisites

1. Ensure the EC2 instance is reachable via the provided SSH key:
   ```bash
   ssh -i /Users/tarun_bhardwaj/mydrive/Projects/keys/useast/mediaGenAILab.pem \
	   ec2-user@ec2-34-232-17-177.compute-1.amazonaws.com hostname
   ```
2. On the EC2 host (run once), install any runtime dependencies such as Python,
   Node, FFmpeg, etc., that the services expect.

### Push-to-production workflow

The script lives at `scripts/deploy/push_to_prod.sh`. It performs these steps:

- Syncs the current repository to `/media/dev/mediaGenAI` on the EC2 instance.
- Snapshots a timestamped release in `/media/releases/mediaGenAI/<timestamp>`.
- Promotes the release to `/media/prod/mediaGenAI`.

Run it from the project root:

```bash
chmod +x scripts/deploy/push_to_prod.sh   # first time only
scripts/deploy/push_to_prod.sh
```

Environment variables allow overrides without editing the script:

| Variable | Purpose | Default |
|----------|---------|---------|
| `DEPLOY_KEY` | SSH private key path | `/Users/tarun_bhardwaj/mydrive/Projects/keys/useast/mediaGenAILab.pem` |
| `DEPLOY_HOST` | SSH host | `ec2-user@ec2-34-232-17-177.compute-1.amazonaws.com` |
| `REMOTE_BASE` | Remote base directory | `/media` |
| `PROJECT_NAME` | Folder name on remote | `mediaGenAI` |
| `LOCAL_ROOT` | Local source directory | current working directory |

Example with overrides:

```bash
DEPLOY_KEY=~/keys/mediaGenAILab.pem \
DEPLOY_HOST=ec2-user@ec2-34-232-17-177.compute-1.amazonaws.com \
REMOTE_BASE=/media \
scripts/deploy/push_to_prod.sh
```

The script is idempotent. Subsequent runs update the dev mirror, capture a new
release, and sync prod to that release. Prior releases remain in
`/media/releases/mediaGenAI` for rollback.
```

##  Development Workflow

1. **Start Development**: `./start-all.sh`
2. **Make Changes**: Edit code in respective directories
3. **Test Changes**: Services will auto-reload (Flask debug mode)
4. **Stop Services**: `./stop-all.sh` when done

## 📁 Project Structure

```
mediaGenAIUseCases/
├── frontend/              # React application
├── aiSubtitle/            # AI Subtitle service (Flask)
├── imageCreation/         # Image Creation service (Flask)
├── syntheticVoiceover/    # Synthetic voiceover service (Flask + language + speech providers)
├── sceneSummarization/    # Scene summarization service (Flask + vision + language + speech providers)
├── movieScriptCreation/   # Feature-length screenplay generator (Flask + Bedrock LLM)
├── contentModeration/     # Video content moderation timeline service
├── personalizedTrailer/   # Personalized Trailer orchestration service
├── start-all.sh           # Start all services
├── stop-all.sh            # Stop all services
├── start-backend.sh       # Start backend services only
└── stop-backend.sh        # Stop backend services only
```