#!/bin/bash

# Start All Services Script
# This script starts all backend services and the frontend

echo "🚀 Starting BornInCloud Streaming application..."

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python3"
ENV_FILE="$PROJECT_ROOT/.env"
ENV_LOCAL_FILE="$PROJECT_ROOT/.env.local"
ENV_MULTIMODAL_LOCAL_FILE="$PROJECT_ROOT/backend/.env.multimodal.local"
ENV_MULTIMODAL_SECRETS_FILE="$PROJECT_ROOT/backend/.env.multimodal.secrets.local"

# Ensure common tool locations are on PATH without clobbering the user's PATH.
# The previous implementation replaced PATH entirely, which can hide npm/node and
# cause the frontend to never start (leading to ERR_CONNECTION_REFUSED).
maybe_prepend_path() {
    local dir="$1"
    if [ -n "$dir" ] && [ -d "$dir" ] && [[ ":$PATH:" != *":$dir:"* ]]; then
        PATH="$dir:$PATH"
    fi
}

maybe_prepend_path "/usr/local/bin"
maybe_prepend_path "/opt/homebrew/bin"

# If NVM is installed, add the newest node bin directory (if any)
if [ -d "$HOME/.nvm/versions/node" ]; then
    latest_node_bin="$(ls -1d "$HOME/.nvm/versions/node"/*/bin 2>/dev/null | tail -n 1)"
    maybe_prepend_path "$latest_node_bin"
fi

# Add ffmpeg if installed via common locations
if [ -x "/usr/local/bin/ffmpeg" ]; then
    maybe_prepend_path "/usr/local/bin"
elif [ -x "/opt/homebrew/bin/ffmpeg" ]; then
    maybe_prepend_path "/opt/homebrew/bin"
fi

if [ -f "$ENV_FILE" ]; then
    echo "ℹ️  Loading environment from $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

if [ -f "$ENV_LOCAL_FILE" ]; then
    echo "ℹ️  Loading environment from $ENV_LOCAL_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_LOCAL_FILE"
    set +a
fi

# Derive default local-service URLs for the backend/UI.
# These must be set even when using default ports, because the backend expects the URL env vars.
export ENVID_LOCAL_MODERATION_PORT="${ENVID_LOCAL_MODERATION_PORT:-5081}"

export ENVID_METADATA_LOCAL_MODERATION_URL="${ENVID_METADATA_LOCAL_MODERATION_URL:-http://localhost:${ENVID_LOCAL_MODERATION_PORT}}"


# AWS/S3 has been removed from the default slim stack.

# Ensure all services can import the shared helpers without per-app sys.path hacks
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Ensure default engine selections prefer local ffmpeg processing
export SCENE_SUMMARY_VIDEO_ENGINE="${SCENE_SUMMARY_VIDEO_ENGINE:-ffmpeg}"
export SUBTITLE_VIDEO_ENGINE="${SUBTITLE_VIDEO_ENGINE:-ffmpeg}"

# Check if virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Virtual environment not found at $PROJECT_ROOT/.venv"
    echo "Please run: python3.13 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Ensure the venv is using Python 3.14+
if ! "$VENV_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 14) else 1)'; then
    echo "❌ This project requires Python 3.14+ (venv is using: $($VENV_PYTHON -V 2>&1))"
    echo "Recreate the venv with: rm -rf .venv && python3.14 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Function to start a service in the background
start_service() {
    local service_name=$1
    local service_dir=$2
    local service_file=$3
    local port=$4
    local extra_env_file="${5:-}"
    
    echo "🔄 Starting $service_name on port $port..."
    local pid=""

    if [ -n "$extra_env_file" ]; then
        (
            cd "$PROJECT_ROOT/$service_dir" || exit 1

            # Allow a comma-separated list of env files.
            IFS=',' read -r -a _env_files <<< "$extra_env_file"
            for f in "${_env_files[@]}"; do
                f="$(echo "$f" | xargs)"
                if [ -n "$f" ] && [ -f "$f" ]; then
                    echo "ℹ️  Loading service-specific env from $f (only for $service_name)"
                    set -a
                    # shellcheck disable=SC1090
                    source "$f"
                    set +a
                fi
            done

            export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
            nohup env PYTHONUNBUFFERED=1 PYTHONPATH="$PYTHONPATH" ENVID_METADATA_PORT="$port" $VENV_PYTHON $service_file > "$PROJECT_ROOT/$service_name.log" 2>&1 &
            echo $! > "$PROJECT_ROOT/$service_name.pid"
        )
        pid=$(cat "$PROJECT_ROOT/$service_name.pid" 2>/dev/null || true)
                    # Remove stray local variable declaration
                    # local extra_env_file="${5:-}"

    else
        cd "$PROJECT_ROOT/$service_dir"
        nohup env PYTHONUNBUFFERED=1 PYTHONPATH="$PYTHONPATH" ENVID_METADATA_PORT="$port" $VENV_PYTHON $service_file > "$PROJECT_ROOT/$service_name.log" 2>&1 &
        pid=$!
        echo $pid > "$PROJECT_ROOT/$service_name.pid"
    fi
    echo "✅ $service_name started with PID $pid"
}

wait_for_health() {
    local name="$1"
    local url="$2"
    local log_file="$3"
    local timeout_seconds="${4:-60}"

    echo "⏳ Waiting for $name to become healthy ($url) ..."
    local start_ts
    start_ts="$(date +%s)"
    while true; do
        if curl -fsS "$url" >/dev/null 2>&1; then
            echo "✅ $name is healthy"
            return 0
        fi

        local now_ts
        now_ts="$(date +%s)"
        if [ $((now_ts - start_ts)) -ge "$timeout_seconds" ]; then
            echo "❌ Timed out waiting for $name health after ${timeout_seconds}s"
            if [ -n "$log_file" ] && [ -f "$log_file" ]; then
                echo "--- Last 80 lines of $log_file ---"
                tail -n 80 "$log_file" || true
            fi
            return 1
        fi

        sleep 2
    done
}

ensure_docker_ready() {
    if ! command -v docker >/dev/null 2>&1; then
        return 1
    fi
    if docker info >/dev/null 2>&1; then
        return 0
    fi
    if [[ "$(uname -s)" == "Darwin" ]]; then
        if command -v open >/dev/null 2>&1; then
            open -a Docker >/dev/null 2>&1 || true
        fi
    fi
    local max_tries="${ENVID_DOCKER_WAIT_TRIES:-30}"
    local i
    for ((i=1; i<=max_tries; i++)); do
        if docker info >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    return 1
}


# Start backend services
echo "🔧 Starting backend services..."

# Optionally start local moderation service (NudeNet) in a separate Python runtime.
start_local_moderation_service() {
    local service_name="moderation"
    local runner="$PROJECT_ROOT/localModerationNudeNet/run_local_venv.sh"
    local port="${ENVID_LOCAL_MODERATION_PORT:-5081}"
    local url="${ENVID_METADATA_LOCAL_MODERATION_URL:-http://localhost:${port}}"

    if [[ "${ENVID_LOCAL_MODERATION_AUTOSTART:-1}" == "0" ]]; then
        echo "ℹ️  Local moderation autostart disabled (ENVID_LOCAL_MODERATION_AUTOSTART=0)"
        return 0
    fi
    if [[ "$url" != http://localhost:${port}* && "$url" != http://127.0.0.1:${port}* ]]; then
        echo "ℹ️  Local moderation URL is non-local ($url); skipping autostart"
        return 0
    fi
    if curl -fsS "$url/health" >/dev/null 2>&1; then
        echo "✅ $service_name already healthy ($url)"
        return 0
    fi
    if [[ ! -x "$runner" ]]; then
        echo "⚠️  Local moderation runner not found/executable: $runner"
        return 0
    fi

    local py_bin="${ENVID_LOCAL_MODERATION_PY_BIN:-}"
    if [[ -z "$py_bin" ]]; then
        if command -v python3.11 >/dev/null 2>&1; then
            py_bin="python3.11"
        elif command -v python3.12 >/dev/null 2>&1; then
            py_bin="python3.12"
        else
            echo "⚠️  python3.11/python3.12 not found; skipping local moderation service autostart"
            return 0
        fi
    fi

    echo "🔄 Starting $service_name on port $port..."
    nohup env PYTHONUNBUFFERED=1 PY_BIN="$py_bin" PORT="$port" "$runner" > "$PROJECT_ROOT/${service_name}.log" 2>&1 &
    echo $! > "$PROJECT_ROOT/${service_name}.pid"
    echo "✅ $service_name started with PID $(cat "$PROJECT_ROOT/${service_name}.pid" 2>/dev/null || echo "?")"
}

start_multimodal_backend_docker() {
    local service_name="envid-metadata-multimodal"
    local port="5016"
    local pid_file="$PROJECT_ROOT/${service_name}.pid"
    local log_file="$PROJECT_ROOT/${service_name}.log"
    local image_name="${ENVID_MULTIMODAL_DOCKER_IMAGE:-envid-metadata-multimodal:dev}"
    local repo_root
    local env_target="${ENVID_ENV_TARGET:-}"

    repo_root="$(cd "$PROJECT_ROOT/.." && pwd)"

    if [ -z "$env_target" ]; then
        if [ "$(uname -s)" = "Darwin" ]; then
            env_target="laptop"
        else
            env_target="vm"
        fi
    fi

    if ! ensure_docker_ready; then
        if command -v docker >/dev/null 2>&1; then
            echo "⚠️  Docker daemon not reachable; cannot start $service_name"
        else
            echo "⚠️  docker not found; cannot start $service_name"
        fi
        return 0
    fi

    echo "🐳 Building $service_name image..."
    docker build -t "$image_name" -f "$repo_root/code/envidMetadataGCP/Dockerfile" "$repo_root" > "$log_file" 2>&1 || true

    local env_args=()
    local env_files=(
        "$ENV_MULTIMODAL_LOCAL_FILE"
        "$ENV_MULTIMODAL_SECRETS_FILE"
        "$repo_root/backend/.env.multimodal.${env_target}.local"
        "$repo_root/backend/.env.multimodal.${env_target}.secrets.local"
    )
    for env_file in "${env_files[@]}"; do
        if [ -f "$env_file" ]; then
            env_args+=(--env-file "$env_file")
        fi
    done

    local gcp_file="${GOOGLE_APPLICATION_CREDENTIALS:-}"
    local gcp_mount=()
    local gcp_env=()
    if [ -n "$gcp_file" ] && [ -f "$gcp_file" ]; then
        gcp_mount=(-v "$gcp_file:/opt/gcp.json:ro")
        gcp_env=(-e "GOOGLE_APPLICATION_CREDENTIALS=/opt/gcp.json")
    fi

    local gpu_args=()
    if command -v nvidia-smi >/dev/null 2>&1; then
        if [[ "${ENVID_MULTIMODAL_DOCKER_GPUS:-all}" != "0" ]]; then
            gpu_args=(--gpus "${ENVID_MULTIMODAL_DOCKER_GPUS:-all}")
        fi
    fi

    docker rm -f "$service_name" >/dev/null 2>&1 || true
    echo "🐳 Starting $service_name via Docker on port $port..."
    container_id="$(docker run -d --name "$service_name" \
        -p "${port}:5016" \
        -e PYTHONUNBUFFERED=1 \
        -e PYTHONPATH="/app:/app/code" \
        -e ENVID_METADATA_PORT="${port}" \
        "${gpu_args[@]}" \
        "${env_args[@]}" \
        "${gcp_env[@]}" \
        "${gcp_mount[@]}" \
        "$image_name")"

    echo "$container_id" > "$pid_file"
    echo "✅ $service_name started (container: $container_id)"

    wait_for_health "$service_name" "http://localhost:${port}/health" "$log_file" 90 || true
}

start_local_moderation_service

# ✅ Envid Metadata (Multimodal only)
# Multimodal backend is managed by always-on Docker (no start/stop here).

# Other services intentionally disabled to run a slim stack.
# Uncomment as needed.
# start_service "ai-subtitle" "aiSubtitle" "aiSubtitle.py" "5001"
# start_service "image-creation" "imageCreation" "imageCreation.py" "5002"
# start_service "synthetic-voiceover" "syntheticVoiceover" "syntheticVoiceover.py" "5003"
# start_service "scene-summarization" "sceneSummarization" "sceneSummarization.py" "5004"
# start_service "movie-script" "movieScriptCreation" "movieScriptCreation.py" "5005"
# start_service "content-moderation" "contentModeration" "contentModeration.py" "5006"
# start_service "personalized-trailer" "personalizedTrailer" "personalizedTrailer.py" "5007"
# start_service "semantic-search" "semanticSearch" "semanticSearch.py" "5008"
# start_service "video-generation" "videoGeneration" "videoGeneration.py" "5009"
# start_service "dynamic-ad-insertion" "dynamicAdInsertion" "dynamicAdInsertion.py" "5010"
# start_service "media-supply-chain" "mediaSupplyChain" "mediaSupplyChain.py" "5011"
# start_service "interactive-shoppable" "interactiveShoppable/backend" "interactiveShoppable.py" "5055"
# start_service "usecase-visibility" "useCaseVisibility" "useCaseVisibility.py" "5012"
# start_service "highlight-trailer" "highlightTrailer" "highlightTrailer.py" "5013"

# Wait a moment for backends to initialize
echo "⏳ Waiting for backend services to initialize..."
sleep 3

# Start frontend (if npm is available)
echo ""
if ! command -v npm >/dev/null 2>&1; then
    echo "⚠️  npm is not installed or not on PATH. Skipping frontend startup."
    echo "👉 Install Node.js (which bundles npm) and rerun ./start-all.sh to launch the React app."
else
    echo "🎨 Starting frontend application..."
    cd "$PROJECT_ROOT/frontend"

    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo "📦 Installing frontend dependencies..."
        npm install
    fi

    # Start the React development server
    FRONTEND_PORT="${FRONTEND_PORT:-3000}"
    echo "🔄 Starting React development server on port $FRONTEND_PORT..."
    # VS Code tasks can terminate child process trees after the task completes.
    # To keep the dev server running reliably, start it in a new session using
    # `os.setsid()` (portable; avoids requiring an external `setsid` binary).
    nohup env PORT="$FRONTEND_PORT" BROWSER=none "$VENV_PYTHON" -c 'import os; os.setsid(); os.execvpe("npm", ["npm", "start"], os.environ)' > "../frontend.log" 2>&1 < /dev/null &

    # Wait for the dev server to start listening (up to ~30s)
    for _ in {1..30}; do
        if lsof -nP -iTCP:"$FRONTEND_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    FRONTEND_LISTENER_PID="$(lsof -nP -t -iTCP:"$FRONTEND_PORT" -sTCP:LISTEN 2>/dev/null | head -n 1)"
    if [ -n "$FRONTEND_LISTENER_PID" ]; then
        echo "$FRONTEND_LISTENER_PID" > "../frontend.pid"
    fi

    if ! lsof -nP -iTCP:"$FRONTEND_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "❌ Frontend did not start listening on port $FRONTEND_PORT."
        echo "👉 Check $PROJECT_ROOT/frontend.log for details."
    fi
fi

echo ""
echo "🎉 All services started successfully!"
echo ""
echo "🌐 Application URLs:"
if command -v npm >/dev/null 2>&1; then
    echo "   • Frontend (React App): http://localhost:3000"
else
    echo "   • Frontend (React App): Skipped (install Node/npm to enable)"
fi
echo "   • Envid Metadata (Multimodal) Service: http://localhost:5016"
echo ""
echo "📝 Log files:"
echo "   • Multimodal Envid Metadata logs: $PROJECT_ROOT/envid-metadata-multimodal.log"
echo "   • Frontend (React App) logs: $PROJECT_ROOT/frontend.log"
echo ""
echo "🛑 To stop all services, run: ./stop-all.sh"

if command -v npm >/dev/null 2>&1; then
    FRONTEND_PORT="${FRONTEND_PORT:-3000}"
    if lsof -nP -iTCP:"$FRONTEND_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "\n✅ Frontend is ok"
    else
        echo "\n⚠️  Frontend is NOT running"
    fi
else
    echo "\n⚠️  Frontend was skipped"
fi
echo "✅ Backend is ok"