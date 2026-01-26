#!/bin/bash

# Start Backend Services Only
# - AI Subtitle Service (5001)
# - Image Creation Service (5002)
# - Synthetic Voiceover Service (5003)
# - Scene Summarization Service (5004)
# - Movie Script Creation Service (5005)
# - Content Moderation Service (5006)
# - Personalized Trailer Service (5007)
# - Semantic Search Service (5008)
# - Dynamic Ad Insertion Service (5010)

echo "🔧 Starting backend services..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python3"
ENV_FILE="$PROJECT_ROOT/.env"
ENV_LOCAL_FILE="$PROJECT_ROOT/.env.local"
ENV_MULTIMODAL_LOCAL_FILE="$PROJECT_ROOT/.env.multimodal.local"
ENV_MULTIMODAL_SECRETS_FILE="$PROJECT_ROOT/.env.multimodal.secrets.local"

# Allow separate env overrides per environment (laptop vs VM).
ENV_TARGET="${ENVID_ENV_TARGET:-}"
if [ -z "$ENV_TARGET" ]; then
  if [ "$(uname -s)" = "Darwin" ]; then
    ENV_TARGET="laptop"
  else
    ENV_TARGET="vm"
  fi
fi

ENV_MULTIMODAL_TARGET_FILE="$PROJECT_ROOT/.env.multimodal.${ENV_TARGET}.local"
ENV_MULTIMODAL_TARGET_SECRETS_FILE="$PROJECT_ROOT/.env.multimodal.${ENV_TARGET}.secrets.local"

# Ensure services can import shared/ utilities.
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/code:${PYTHONPATH:-}"

# Ensure common tool locations are on PATH without clobbering the user's PATH.
maybe_prepend_path() {
  local dir="$1"
  if [ -n "$dir" ] && [ -d "$dir" ] && [[ ":$PATH:" != *":$dir:"* ]]; then
    PATH="$dir:$PATH"
  fi
}

maybe_prepend_path "/usr/local/bin"
maybe_prepend_path "/opt/homebrew/bin"

if [ -d "$HOME/.nvm/versions/node" ]; then
  latest_node_bin="$(ls -1d "$HOME/.nvm/versions/node"/*/bin 2>/dev/null | tail -n 1)"
  maybe_prepend_path "$latest_node_bin"
fi

export PATH

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
export ENVID_LOCAL_KEYSCENE_PORT="${ENVID_LOCAL_KEYSCENE_PORT:-5085}"

export ENVID_METADATA_LOCAL_MODERATION_URL="${ENVID_METADATA_LOCAL_MODERATION_URL:-http://localhost:${ENVID_LOCAL_MODERATION_PORT}}"
export ENVID_METADATA_LOCAL_KEYSCENE_URL="${ENVID_METADATA_LOCAL_KEYSCENE_URL:-http://localhost:${ENVID_LOCAL_KEYSCENE_PORT}}"


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
  nohup env PYTHONUNBUFFERED=1 PY_BIN="$py_bin" PORT="$port" "$runner" > "$log_file" 2>&1 &
  echo $! > "$pid_file"
  echo "✅ $service_name started with PID $(cat "$pid_file" 2>/dev/null || echo "?")"

  wait_for_health "$service_name" "$url/health" "$log_file" 60 || true
}

start_local_keyscene_best_service() {
  local service_name="local-keyscene-best"
  local port="${ENVID_LOCAL_KEYSCENE_PORT:-5085}"
  local url="${ENVID_METADATA_LOCAL_KEYSCENE_URL:-http://localhost:${port}}"
  local pid_file="$PROJECT_ROOT/${service_name}.pid"
  local log_file="$PROJECT_ROOT/${service_name}.log"
  local autoclean="${ENVID_LOCAL_KEYSCENE_AUTOCLEAN:-1}"

  get_keyscene_clip_ok() {
    local health_url="$1"
    "$VENV_PYTHON" - "$health_url" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1] if len(sys.argv) > 1 else ""
try:
    with urllib.request.urlopen(url, timeout=5) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    data = json.loads(raw or "{}") if raw else {}
except Exception:
    print("unknown")
    raise SystemExit(0)

details = data.get("details") if isinstance(data, dict) else None
clip = details.get("clip") if isinstance(details, dict) else None
ok = None
if isinstance(clip, dict):
    ok = clip.get("ok")
if ok is True:
    print("true")
elif ok is False:
    print("false")
else:
    print("unknown")
PY
  }

  maybe_fix_keyscene_clip_cache() {
    local health_url="$1"
    local container_name="$2"
    local clip_ok
    clip_ok="$(get_keyscene_clip_ok "$health_url")"
    if [[ "$clip_ok" == "false" && "$autoclean" != "0" ]]; then
      echo "⚠️  CLIP cache appears corrupted; clearing cache and restarting $service_name"
      docker exec "$container_name" sh -lc 'rm -rf /cache/huggingface /cache/open_clip /cache/torch || true; mkdir -p /cache/huggingface' >/dev/null 2>&1 || true
      docker restart "$container_name" >/dev/null 2>&1 || true
      wait_for_health "$service_name" "$health_url" "$log_file" 240 || true
      clip_ok="$(get_keyscene_clip_ok "$health_url")"
      if [[ "$clip_ok" != "true" ]]; then
        echo "⚠️  CLIP cache cleanup did not resolve CLIP health"
      fi
    fi
  }

  if [[ "${ENVID_LOCAL_KEYSCENE_AUTOSTART:-0}" == "0" ]]; then
    # Default off because this image is large and needs extra weights for TransNetV2.
    return 0
  fi

  if [[ "$url" != http://localhost:${port}* && "$url" != http://127.0.0.1:${port}* ]]; then
    echo "ℹ️  Local keyscene URL is non-local ($url); skipping autostart"
    return 0
  fi

  if curl -fsS "$url/health" >/dev/null 2>&1; then
    maybe_fix_keyscene_clip_cache "$url/health" "$service_name"
    return 0
  fi

  if ! ensure_docker_ready; then
    if command -v docker >/dev/null 2>&1; then
      echo "⚠️  Docker is installed but the daemon is not reachable; skipping docker start for $service_name"
    else
      echo "⚠️  docker not found; cannot start $service_name"
    fi
    return 0
  fi

  echo "🐳 Starting $service_name via Docker on port $port..."
  (
    cd "$PROJECT_ROOT/code/localKeySceneBest" || exit 0

    docker rm -f "$service_name" >/dev/null 2>&1 || true
    docker build -t "$service_name:dev" .

    container_id="$(docker run -d --name "$service_name" \
      -p "${port}:5085" \
      -e PORT=5085 \
      -e CLIP_MODEL="${ENVID_CLIP_MODEL:-ViT-B-32}" \
      -e CLIP_PRETRAINED="${ENVID_CLIP_PRETRAINED:-laion2b_s34b_b79k}" \
      -e TRANSNETV2_MODEL_DIR="${TRANSNETV2_MODEL_DIR:-/weights/transnetv2-weights}" \
      -e XDG_CACHE_HOME="/cache" \
      -v "$PROJECT_ROOT/code/localKeySceneBest/weights:/weights:ro" \
      -v "$PROJECT_ROOT/code/localKeySceneBest/.cache:/cache" \
      "$service_name:dev")"
    echo "$container_id" > "$pid_file"
    echo "✅ $service_name started (container: $container_id)"
  ) > "$log_file" 2>&1 || true

  wait_for_health "$service_name" "$url/health" "$log_file" 240 || true
  maybe_fix_keyscene_clip_cache "$url/health" "$service_name"
}

start_multimodal_backend_docker() {
  local service_name="envid-metadata-multimodal"
  local port="5016"
  local pid_file="$PROJECT_ROOT/${service_name}.pid"
  local log_file="$PROJECT_ROOT/${service_name}.log"
  local image_name="${ENVID_MULTIMODAL_DOCKER_IMAGE:-envid-metadata-multimodal:dev}"

  if ! ensure_docker_ready; then
    if command -v docker >/dev/null 2>&1; then
      echo "⚠️  Docker daemon not reachable; cannot start $service_name"
    else
      echo "⚠️  docker not found; cannot start $service_name"
    fi
    return 0
  fi

  echo "🐳 Building $service_name image..."
  docker build -t "$image_name" -f "$PROJECT_ROOT/code/envidMetadataGCP/Dockerfile" "$PROJECT_ROOT" > "$log_file" 2>&1 || true

  local env_args=()
  local env_files=(
    "$ENV_MULTIMODAL_LOCAL_FILE"
    "$ENV_MULTIMODAL_TARGET_FILE"
    "$ENV_MULTIMODAL_SECRETS_FILE"
    "$ENV_MULTIMODAL_TARGET_SECRETS_FILE"
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

#  Envid Metadata (Multimodal only)
start_local_moderation_service
start_local_keyscene_best_service
# Multimodal backend is managed by always-on Docker (no start/stop here).

# Other services intentionally disabled.
# Uncomment as needed.
# start_service "ai-subtitle" "aiSubtitle" "aiSubtitle.py" "5001"
# start_service "image-creation" "imageCreation" "app.py" "5002"
# start_service "synthetic-voiceover" "syntheticVoiceover" "app.py" "5003"
# start_service "scene-summarization" "sceneSummarization" "app.py" "5004"
# start_service "movie-script" "movieScriptCreation" "app.py" "5005"
# start_service "content-moderation" "contentModeration" "app.py" "5006"
# export PERSONALIZED_TRAILER_PIPELINE_MODE=mock
# start_service "personalized-trailer" "personalizedTrailer" "app.py" "5007"
# start_service "semantic-search" "semanticSearch" "app.py" "5008"
# start_service "video-generation" "videoGeneration" "app.py" "5009"
# start_service "dynamic-ad-insertion" "dynamicAdInsertion" "app.py" "5010"
# start_service "media-supply-chain" "mediaSupplyChain" "app.py" "5011"
# start_service "usecase-visibility" "useCaseVisibility" "app.py" "5012"
# start_service "highlight-trailer" "highlightTrailer" "app.py" "5013"

echo "🎉 Backend services started!"
echo

wait_for_health "envid-metadata-multimodal" "http://localhost:5016/health" "$PROJECT_ROOT/envid-metadata-multimodal.log" 90

echo "\n✅ Backend is ok"
