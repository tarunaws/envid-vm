#!/bin/bash

# Ensure the multimodal backend container is always running and restarts on crash/boot.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="envid-metadata-multimodal"
IMAGE_NAME="${ENVID_MULTIMODAL_DOCKER_IMAGE:-envid-metadata-multimodal:dev}"
PORT="${ENVID_MULTIMODAL_PORT:-5016}"

ENV_TARGET="${ENVID_ENV_TARGET:-}"
if [ -z "$ENV_TARGET" ]; then
  if [ "$(uname -s)" = "Darwin" ]; then
    ENV_TARGET="laptop"
  else
    ENV_TARGET="vm"
  fi
fi

load_env_file() {
  local file="$1"
  if [ -f "$file" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$file"
    set +a
  fi
}

load_env_file "$PROJECT_ROOT/.env"
load_env_file "$PROJECT_ROOT/.env.local"
load_env_file "$PROJECT_ROOT/.env.multimodal.local"
load_env_file "$PROJECT_ROOT/.env.multimodal.${ENV_TARGET}.local"
load_env_file "$PROJECT_ROOT/.env.multimodal.secrets.local"
load_env_file "$PROJECT_ROOT/.env.multimodal.${ENV_TARGET}.secrets.local"

if ! command -v docker >/dev/null 2>&1; then
  echo "❌ docker not found"
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "❌ docker daemon not reachable"
  exit 1
fi

echo "🐳 Building $SERVICE_NAME image..."
docker build -t "$IMAGE_NAME" -f "$PROJECT_ROOT/code/envidMetadataGCP/Dockerfile" "$PROJECT_ROOT"

ENV_ARGS=()
ENV_FILES=(
  "$PROJECT_ROOT/.env.multimodal.local"
  "$PROJECT_ROOT/.env.multimodal.${ENV_TARGET}.local"
  "$PROJECT_ROOT/.env.multimodal.secrets.local"
  "$PROJECT_ROOT/.env.multimodal.${ENV_TARGET}.secrets.local"
)
for f in "${ENV_FILES[@]}"; do
  if [ -f "$f" ]; then
    ENV_ARGS+=(--env-file "$f")
  fi
done

GCP_FILE="${GOOGLE_APPLICATION_CREDENTIALS:-}"
if [ -z "$GCP_FILE" ]; then
  if [ -f "$HOME/gcp.json" ]; then
    GCP_FILE="$HOME/gcp.json"
  elif [ -f "$HOME/gcpAccess/gcp.json" ]; then
    GCP_FILE="$HOME/gcpAccess/gcp.json"
  fi
fi
GCP_MOUNT=()
GCP_ENV=()
if [ -n "$GCP_FILE" ] && [ -f "$GCP_FILE" ]; then
  GCP_MOUNT=(-v "$GCP_FILE:$GCP_FILE:ro")
  GCP_ENV=(-e "GOOGLE_APPLICATION_CREDENTIALS=$GCP_FILE")
fi

GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  if [[ "${ENVID_MULTIMODAL_DOCKER_GPUS:-all}" != "0" ]]; then
    GPU_ARGS=(--gpus "${ENVID_MULTIMODAL_DOCKER_GPUS:-all}")
  fi
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${SERVICE_NAME}$"; then
  echo "🔧 Updating restart policy for existing container..."
  docker update --restart=always "$SERVICE_NAME" >/dev/null
  echo "🔄 Restarting container..."
  docker rm -f "$SERVICE_NAME" >/dev/null 2>&1 || true
fi

echo "🚀 Starting container with restart=always..."
docker run -d --name "$SERVICE_NAME" \
  --restart=always \
  -p "${PORT}:5016" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH="/app:/app/code" \
  -e ENVID_METADATA_PORT="${PORT}" \
  "${GPU_ARGS[@]}" \
  "${ENV_ARGS[@]}" \
  "${GCP_ENV[@]}" \
  "${GCP_MOUNT[@]}" \
  "$IMAGE_NAME"

echo "✅ $SERVICE_NAME is running with restart=always"
