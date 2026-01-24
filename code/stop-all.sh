#!/bin/bash

# Stop All Services Script
# This script stops backend services and the frontend

echo "🛑 Stopping MediaGenAI Complete Application..."

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"
ENV_LOCAL_FILE="$PROJECT_ROOT/.env.local"

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

LOCAL_LABEL_PORT="${ENVID_LOCAL_LABEL_DETECTION_PORT:-5083}"
LOCAL_LABEL_URL="${ENVID_METADATA_LOCAL_LABEL_DETECTION_URL:-http://localhost:${LOCAL_LABEL_PORT}}"
LOCAL_LABEL_RUNTIME="${ENVID_LOCAL_LABEL_DETECTION_RUNTIME:-venv}"
if [[ "${ENVID_LOCAL_LABEL_DETECTION_DOCKER:-0}" == "1" ]]; then
    LOCAL_LABEL_RUNTIME="docker"
fi

stop_local_label_detection_docker_if_enabled() {
    if [[ "$LOCAL_LABEL_RUNTIME" != "docker" ]]; then
        return 0
    fi
    if ! command -v docker >/dev/null 2>&1; then
        echo "⚠️  docker not found; cannot stop local-label-detection via docker"
        return 0
    fi
    local compose_file="${ENVID_LOCAL_LABEL_DETECTION_DOCKER_COMPOSE_FILE:-}"
    if [[ -z "$compose_file" ]]; then
        compose_file="docker-compose.yml"
        if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" && -f "$PROJECT_ROOT/localLabelDetection/docker-compose.amd64.yml" ]]; then
            compose_file="docker-compose.amd64.yml"
            echo "ℹ️  Apple Silicon detected; defaulting localLabelDetection to $compose_file (override via ENVID_LOCAL_LABEL_DETECTION_DOCKER_COMPOSE_FILE)"
        fi
    fi
    if [[ ! -f "$PROJECT_ROOT/localLabelDetection/$compose_file" ]]; then
        echo "⚠️  $compose_file not found for localLabelDetection; cannot stop via docker"
        return 0
    fi

    echo "🐳 Stopping local-label-detection via docker compose..."
    (cd "$PROJECT_ROOT/localLabelDetection" && docker compose -f "$compose_file" down --remove-orphans) || true

    # Extra cleanup for the common "container name already in use" failure mode.
    docker rm -f locallabeldetection-local-label-detection-1 >/dev/null 2>&1 || true
}

# Stop frontend
echo "🎨 Stopping frontend application..."
if [ -f "$PROJECT_ROOT/frontend.pid" ]; then
    FRONTEND_PID=$(cat "$PROJECT_ROOT/frontend.pid")
    if ps -p $FRONTEND_PID > /dev/null; then
        echo "🔄 Stopping React development server (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
        echo "✅ Frontend stopped"
    else
        echo "⚠️  Frontend process not running"
    fi
    rm -f "$PROJECT_ROOT/frontend.pid"
else
    echo "⚠️  No frontend PID file found"
fi

# Stop backend services
echo ""
echo "🔧 Stopping backend services..."

# Function to stop a service
stop_service() {
    local service_name=$1
    local silent_if_missing=${2:-0}
    local pid_file="$PROJECT_ROOT/$service_name.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null; then
            echo "🔄 Stopping $service_name (PID: $pid)..."
            kill $pid
            echo "✅ $service_name stopped"
        else
            echo "⚠️  $service_name process not running"
        fi
        rm -f "$pid_file"
    else
        if [[ "$silent_if_missing" != "1" ]]; then
            echo "⚠️  No $service_name PID file found"
        fi
    fi
}

# ✅ Local moderation service (NudeNet)
stop_service "local-moderation-nudenet" 1

# ✅ Local label detection service (Detectron2/MMDetection)
stop_service "local-label-detection" 1
stop_local_label_detection_docker_if_enabled

# ✅ Envid Metadata (Multimodal only)
stop_service "envid-metadata-multimodal"

# Other services intentionally disabled.
# Uncomment if you re-enable them in start-all.sh.
# stop_service "ai-subtitle"
# stop_service "image-creation"
# stop_service "synthetic-voiceover"
# stop_service "scene-summarization"
# stop_service "movie-script"
# stop_service "content-moderation"
# stop_service "personalized-trailer"
# stop_service "semantic-search"
# stop_service "video-generation"
# stop_service "dynamic-ad-insertion"
# stop_service "media-supply-chain"
# stop_service "interactive-shoppable"
# stop_service "highlight-trailer"
# stop_service "usecase-visibility"

# Also kill any remaining processes on the ports (backup cleanup)
echo ""
echo "🧹 Cleaning up any remaining processes..."
lsof -ti:3000 2>/dev/null | xargs kill -9 2>/dev/null || true
LOCAL_MOD_PORT="${ENVID_LOCAL_MODERATION_PORT:-5081}"
lsof -ti:"$LOCAL_MOD_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
if [[ "$LOCAL_LABEL_RUNTIME" != "docker" ]]; then
    lsof -ti:"$LOCAL_LABEL_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
else
    echo "ℹ️  Skipping port kill for local label detection (${LOCAL_LABEL_URL}) because runtime=docker"
fi
lsof -ti:5016 2>/dev/null | xargs kill -9 2>/dev/null || true

# Other ports intentionally disabled.
# lsof -ti:5001 2>/dev/null | xargs kill -9 2>/dev/null || true
# lsof -ti:5002 2>/dev/null | xargs kill -9 2>/dev/null || true
# lsof -ti:5003 2>/dev/null | xargs kill -9 2>/dev/null || true
# lsof -ti:5004 2>/dev/null | xargs kill -9 2>/dev/null || true
# lsof -ti:5005 2>/dev/null | xargs kill -9 2>/dev/null || true
# lsof -ti:5006 2>/dev/null | xargs kill -9 2>/dev/null || true
# lsof -ti:5007 2>/dev/null | xargs kill -9 2>/dev/null || true
# lsof -ti:5008 2>/dev/null | xargs kill -9 2>/dev/null || true
# lsof -ti:5009 2>/dev/null | xargs kill -9 2>/dev/null || true
# lsof -ti:5010 2>/dev/null | xargs kill -9 2>/dev/null || true
# lsof -ti:5011 2>/dev/null | xargs kill -9 2>/dev/null || true
# lsof -ti:5012 2>/dev/null | xargs kill -9 2>/dev/null || true
# lsof -ti:5013 2>/dev/null | xargs kill -9 2>/dev/null || true

echo ""
echo "✅ All services stopped!"