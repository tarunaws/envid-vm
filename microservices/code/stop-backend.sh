#!/bin/bash

# Stop Backend Services Only
# - AI Subtitle Service (5001)
# - Image Creation Service (5002)
# - Synthetic Voiceover Service (5003)
# - Scene Summarization Service (5004)
# - Movie Script Creation Service (5005)
# - Content Moderation Service (5006)
# - Personalized Trailer Service (5007)
# - Semantic Search Service (5008)

echo "🛑 Stopping backend services..."

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

stop_service() {
  local service_name=$1
  local pid_file="$PROJECT_ROOT/$service_name.pid"
  if [ -f "$pid_file" ]; then
    local pid
    pid=$(cat "$pid_file")

    # If the pid file contains a number, treat it as a process PID.
    if [[ "$pid" =~ ^[0-9]+$ ]]; then
      if ps -p "$pid" > /dev/null 2>&1; then
        echo "🔄 Stopping $service_name (PID: $pid)..."
        kill "$pid" || true
        echo "✅ $service_name stopped"
      else
        echo "⚠️  $service_name process not running"
      fi
    else
      # Otherwise, treat it as a docker container id/name.
      if command -v docker >/dev/null 2>&1; then
        echo "🐳 Stopping $service_name (container: $pid)..."
        docker rm -f "$pid" >/dev/null 2>&1 || true
        docker rm -f "$service_name" >/dev/null 2>&1 || true
        echo "✅ $service_name container stopped"
      else
        echo "⚠️  $service_name PID file is not numeric and docker not found"
      fi
    fi
    rm -f "$pid_file"
  else
    :
  fi
}

# ✅ Local moderation service (NudeNet)
stop_service "moderation"

# ✅ Local keyscene service
stop_service "keyscene"


# ✅ Envid Metadata (Multimodal only)
# Managed by always-on Docker (do not stop here)

# Other services intentionally disabled.
# Uncomment if you re-enable them in start-backend.sh.
# stop_service "ai-subtitle"
# stop_service "image-creation"
# stop_service "synthetic-voiceover"
# stop_service "scene-summarization"
# stop_service "movie-script"
# stop_service "content-moderation"
# stop_service "personalized-trailer"
# stop_service "semantic-search"
# stop_service "video-generation"
# stop_service "media-supply-chain"
# stop_service "highlight-trailer"
# stop_service "usecase-visibility"

echo "🧹 Cleaning up any remaining processes on ports..."
LOCAL_MOD_PORT="${ENVID_LOCAL_MODERATION_PORT:-5081}"
lsof -ti:"$LOCAL_MOD_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
# Do not kill port 5016; multimodal backend is managed by Docker.

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
# lsof -ti:5011 2>/dev/null | xargs kill -9 2>/dev/null || true
# lsof -ti:5012 2>/dev/null | xargs kill -9 2>/dev/null || true
# lsof -ti:5013 2>/dev/null | xargs kill -9 2>/dev/null || true

echo "✅ Backend services stopped!"
