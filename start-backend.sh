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
export ENVID_LOCAL_MODERATION_NSFWJS_PORT="${ENVID_LOCAL_MODERATION_NSFWJS_PORT:-5082}"
export ENVID_LOCAL_LABEL_DETECTION_PORT="${ENVID_LOCAL_LABEL_DETECTION_PORT:-5083}"
export ENVID_LOCAL_OCR_PADDLE_PORT="${ENVID_LOCAL_OCR_PADDLE_PORT:-5084}"
export ENVID_LOCAL_KEYSCENE_PORT="${ENVID_LOCAL_KEYSCENE_PORT:-5085}"

export ENVID_METADATA_LOCAL_MODERATION_URL="${ENVID_METADATA_LOCAL_MODERATION_URL:-http://localhost:${ENVID_LOCAL_MODERATION_PORT}}"
export ENVID_METADATA_LOCAL_MODERATION_NSFWJS_URL="${ENVID_METADATA_LOCAL_MODERATION_NSFWJS_URL:-http://localhost:${ENVID_LOCAL_MODERATION_NSFWJS_PORT}}"
export ENVID_METADATA_LOCAL_LABEL_DETECTION_URL="${ENVID_METADATA_LOCAL_LABEL_DETECTION_URL:-http://localhost:${ENVID_LOCAL_LABEL_DETECTION_PORT}}"
export ENVID_METADATA_LOCAL_OCR_PADDLE_URL="${ENVID_METADATA_LOCAL_OCR_PADDLE_URL:-http://localhost:${ENVID_LOCAL_OCR_PADDLE_PORT}}"
export ENVID_METADATA_LOCAL_KEYSCENE_URL="${ENVID_METADATA_LOCAL_KEYSCENE_URL:-http://localhost:${ENVID_LOCAL_KEYSCENE_PORT}}"


# AWS/S3 has been removed from the default slim stack.

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

      # Re-assert shared imports even if env files override variables.
      export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/code:${PYTHONPATH:-}"
      nohup env PYTHONUNBUFFERED=1 PYTHONPATH="$PYTHONPATH" ENVID_METADATA_PORT="$port" $VENV_PYTHON $service_file > "$PROJECT_ROOT/$service_name.log" 2>&1 &
      echo $! > "$PROJECT_ROOT/$service_name.pid"
    )
    pid=$(cat "$PROJECT_ROOT/$service_name.pid" 2>/dev/null || true)
  else
    cd "$PROJECT_ROOT/$service_dir"
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/code:${PYTHONPATH:-}"
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
  # Returns 0 when docker daemon is reachable.
  if ! command -v docker >/dev/null 2>&1; then
    return 1
  fi
  if docker info >/dev/null 2>&1; then
    return 0
  fi

  # On macOS, Docker is often installed but the daemon isn't running yet.
  # Try to start Docker Desktop and wait a bit.
  if [[ "$(uname -s)" == "Darwin" ]]; then
    if command -v open >/dev/null 2>&1; then
      open -a Docker >/dev/null 2>&1 || true
    fi
  fi

  local max_tries="${ENVID_DOCKER_WAIT_TRIES:-30}"  # 30 * 2s = 60s
  local i
  for ((i=1; i<=max_tries; i++)); do
    if docker info >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

start_local_moderation_service() {
  local service_name="local-moderation-nudenet"
  local runner="$PROJECT_ROOT/code/localModerationNudeNet/run_local_venv.sh"
  local port="${ENVID_LOCAL_MODERATION_PORT:-5081}"
  local url="${ENVID_METADATA_LOCAL_MODERATION_URL:-http://localhost:${port}}"
  local pid_file="$PROJECT_ROOT/${service_name}.pid"
  local log_file="$PROJECT_ROOT/${service_name}.log"

  if [[ "${ENVID_LOCAL_MODERATION_AUTOSTART:-1}" == "0" ]]; then
    echo "ℹ️  Local moderation autostart disabled (ENVID_LOCAL_MODERATION_AUTOSTART=0)"
    return 0
  fi

  if [[ "$url" != http://localhost:${port}* && "$url" != http://127.0.0.1:${port}* ]]; then
    echo "ℹ️  Local moderation URL is non-local ($url); skipping autostart"
    return 0
  fi

  if curl -fsS "$url/health" >/dev/null 2>&1; then
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
  nohup env PYTHONUNBUFFERED=1 PY_BIN="$py_bin" PORT="$port" "$runner" > "$log_file" 2>&1 &
  echo $! > "$pid_file"
  echo "✅ $service_name started with PID $(cat "$pid_file" 2>/dev/null || echo "?")"

  wait_for_health "$service_name" "$url/health" "$log_file" 60 || true
}

start_local_moderation_nsfwjs_service() {
  local service_name="local-moderation-nsfwjs"
  local node_runner="$PROJECT_ROOT/code/localModerationNSFWJS/run_local_node.sh"
  local port="${ENVID_LOCAL_MODERATION_NSFWJS_PORT:-5082}"
  local url="${ENVID_METADATA_LOCAL_MODERATION_NSFWJS_URL:-http://localhost:${port}}"
  local pid_file="$PROJECT_ROOT/${service_name}.pid"
  local log_file="$PROJECT_ROOT/${service_name}.log"

  if [[ "${ENVID_LOCAL_MODERATION_NSFWJS_AUTOSTART:-1}" == "0" ]]; then
    echo "ℹ️  Local moderation nsfwjs autostart disabled (ENVID_LOCAL_MODERATION_NSFWJS_AUTOSTART=0)"
    return 0
  fi

  if [[ "$url" != http://localhost:${port}* && "$url" != http://127.0.0.1:${port}* ]]; then
    echo "ℹ️  Local moderation nsfwjs URL is non-local ($url); skipping autostart"
    return 0
  fi

  if curl -fsS "$url/health" >/dev/null 2>&1; then
    echo "✅ $service_name already healthy ($url)"
    return 0
  fi

  # Prefer Docker for NSFWJS because tfjs-node is not compatible with very new Node releases
  # (e.g. Node 25 removed util.isNullOrUndefined which older tfjs-node bundles reference).
  if ensure_docker_ready; then
    echo "🐳 Starting $service_name via Docker on port $port..."

    (
      cd "$PROJECT_ROOT/code/localModerationNSFWJS" || exit 0

      # Clean up any previous container.
      docker rm -f "$service_name" >/dev/null 2>&1 || true

      # Build a local image pinned to Node 20 LTS.
      docker build -t "$service_name:dev" .

      # Run detached and record container id.
      container_id="$(docker run -d --name "$service_name" -p "${port}:5082" -e PORT=5082 "$service_name:dev")"
      echo "$container_id" > "$pid_file"
      echo "✅ $service_name started (container: $container_id)"
    ) > "$log_file" 2>&1 || true

    wait_for_health "$service_name" "$url/health" "$log_file" 120 || true
    return 0
  fi

  if command -v docker >/dev/null 2>&1; then
    echo "⚠️  Docker is installed but the daemon is not reachable; skipping docker start for $service_name"
  fi

  # Fall back to running locally via node when Docker isn't available.
  if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    echo "⚠️  docker and node/npm not found; skipping $service_name autostart"
    return 0
  fi

  # Reject known-incompatible host Node versions.
  local node_major
  node_major="$(node -p 'parseInt(process.versions.node.split(".")[0], 10)' 2>/dev/null || echo "")"
  if [[ -n "$node_major" && "$node_major" -ge 24 ]]; then
    echo "⚠️  Host Node v${node_major} is not supported for tfjs-node; install Docker or use Node 20 LTS"
    return 0
  fi

  if [[ ! -x "$node_runner" ]]; then
    echo "⚠️  Local moderation nsfwjs runner not found/executable: $node_runner"
    return 0
  fi

  echo "🔄 Starting $service_name on port $port (host node)..."
  nohup env PORT="$port" "$node_runner" > "$log_file" 2>&1 &
  echo $! > "$pid_file"
  echo "✅ $service_name started with PID $(cat "$pid_file" 2>/dev/null || echo "?")"

  wait_for_health "$service_name" "$url/health" "$log_file" 120 || true
}

start_local_label_detection_service() {
  local service_name="local-label-detection"
  local runner="$PROJECT_ROOT/code/localLabelDetection/run_local_venv.sh"
  local port="${ENVID_LOCAL_LABEL_DETECTION_PORT:-5083}"
  local url="${ENVID_METADATA_LOCAL_LABEL_DETECTION_URL:-http://localhost:${port}}"
  local pid_file="$PROJECT_ROOT/${service_name}.pid"
  local log_file="$PROJECT_ROOT/${service_name}.log"

  # Optional: run via Docker Compose (recommended for MMDetection on macOS).
  # Enable with: ENVID_LOCAL_LABEL_DETECTION_RUNTIME=docker
  local runtime="${ENVID_LOCAL_LABEL_DETECTION_RUNTIME:-venv}"
  if [[ -z "${ENVID_LOCAL_LABEL_DETECTION_RUNTIME:-}" && "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    if [[ -f "$PROJECT_ROOT/code/localLabelDetection/docker-compose.amd64.yml" || -f "$PROJECT_ROOT/code/localLabelDetection/docker-compose.yml" ]]; then
      runtime="docker"
      echo "ℹ️  Apple Silicon detected; defaulting local label detection runtime to docker (override with ENVID_LOCAL_LABEL_DETECTION_RUNTIME=venv)"
    fi
  fi
  if [[ "${ENVID_LOCAL_LABEL_DETECTION_DOCKER:-0}" == "1" ]]; then
    runtime="docker"
  fi

  if [[ "${ENVID_LOCAL_LABEL_DETECTION_AUTOSTART:-1}" == "0" ]]; then
    echo "ℹ️  Local label detection autostart disabled (ENVID_LOCAL_LABEL_DETECTION_AUTOSTART=0)"
    return 0
  fi

  if [[ "$url" != http://localhost:${port}* && "$url" != http://127.0.0.1:${port}* ]]; then
    echo "ℹ️  Local label detection URL is non-local ($url); skipping autostart"
    return 0
  fi

  if curl -fsS "$url/health" >/dev/null 2>&1; then
    echo "✅ $service_name already healthy ($url)"
    return 0
  fi

  if [[ "$runtime" == "docker" ]]; then
    if ! command -v docker >/dev/null 2>&1; then
      echo "⚠️  docker not found; cannot run $service_name via docker"
      return 0
    fi

    # If Docker was selected implicitly (e.g., Apple Silicon default) but the daemon isn't running,
    # fall back to the venv runner so the service still comes up for local dev.
    if ! docker info >/dev/null 2>&1; then
      if [[ "${ENVID_LOCAL_LABEL_DETECTION_RUNTIME:-}" == "docker" || "${ENVID_LOCAL_LABEL_DETECTION_DOCKER:-0}" == "1" ]]; then
        echo "⚠️  Docker daemon not reachable; cannot run $service_name via docker (requested explicitly)"
        return 0
      fi
      echo "⚠️  Docker daemon not reachable; falling back to venv for $service_name"
      runtime="venv"
    fi

    if [[ "$runtime" != "docker" ]]; then
      : # continue into venv path below
    else
    local compose_file="${ENVID_LOCAL_LABEL_DETECTION_DOCKER_COMPOSE_FILE:-}"
    if [[ -z "$compose_file" ]]; then
      compose_file="docker-compose.yml"
      if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" && -f "$PROJECT_ROOT/code/localLabelDetection/docker-compose.amd64.yml" ]]; then
        compose_file="docker-compose.amd64.yml"
        echo "ℹ️  Apple Silicon detected; defaulting localLabelDetection to $compose_file (override via ENVID_LOCAL_LABEL_DETECTION_DOCKER_COMPOSE_FILE)"
      fi
    fi
    if [[ ! -f "$PROJECT_ROOT/code/localLabelDetection/$compose_file" ]]; then
      echo "⚠️  $compose_file not found for localLabelDetection; cannot run via docker"
      return 0
    fi

    echo "🐳 Starting $service_name via docker compose on port $port..."
    (
      cd "$PROJECT_ROOT/code/localLabelDetection" || exit 0

      # Clean up any stale container names from previous runs (common when switching compose files/platforms).
      docker compose -f "$compose_file" down --remove-orphans >/dev/null 2>&1 || true
      docker rm -f locallabeldetection-local-label-detection-1 >/dev/null 2>&1 || true

      docker compose -f "$compose_file" up -d --build
    ) || true
    wait_for_health "$service_name" "$url/health" "$log_file" 120 || true
    return 0
    fi
  fi

  if [[ ! -x "$runner" ]]; then
    echo "⚠️  Local label detection runner not found/executable: $runner"
    return 0
  fi

  local py_bin="${ENVID_LOCAL_LABEL_DETECTION_PY_BIN:-}"
  if [[ -z "$py_bin" ]]; then
    if command -v python3.11 >/dev/null 2>&1; then
      py_bin="python3.11"
    elif command -v python3.12 >/dev/null 2>&1; then
      py_bin="python3.12"
    else
      return 0
    fi
  fi

  echo "🔄 Starting $service_name on port $port..."
  nohup env PYTHONUNBUFFERED=1 PY_BIN="$py_bin" PORT="$port" "$runner" > "$log_file" 2>&1 &
  echo $! > "$pid_file"
  echo "✅ $service_name started with PID $(cat "$pid_file" 2>/dev/null || echo "?")"

  wait_for_health "$service_name" "$url/health" "$log_file" 60 || true
}

start_local_ocr_paddle_service() {
  local service_name="local-ocr-paddle"
  local port="${ENVID_LOCAL_OCR_PADDLE_PORT:-5084}"
  local url="${ENVID_METADATA_LOCAL_OCR_PADDLE_URL:-http://localhost:${port}}"
  local pid_file="$PROJECT_ROOT/${service_name}.pid"
  local log_file="$PROJECT_ROOT/${service_name}.log"

  if [[ "${ENVID_LOCAL_OCR_PADDLE_AUTOSTART:-1}" == "0" ]]; then
    echo "ℹ️  Local PaddleOCR autostart disabled (ENVID_LOCAL_OCR_PADDLE_AUTOSTART=0)"
    return 0
  fi

  if [[ "$url" != http://localhost:${port}* && "$url" != http://127.0.0.1:${port}* ]]; then
    echo "ℹ️  Local PaddleOCR URL is non-local ($url); skipping autostart"
    return 0
  fi

  if curl -fsS "$url/health" >/dev/null 2>&1; then
    echo "✅ $service_name already healthy ($url)"
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
    cd "$PROJECT_ROOT/code/localOcrPaddle" || exit 0

    docker rm -f "$service_name" >/dev/null 2>&1 || true
    docker build -t "$service_name:dev" .

    container_id="$(docker run -d --name "$service_name" -p "${port}:5084" -e PORT=5084 "$service_name:dev")"
    echo "$container_id" > "$pid_file"
    echo "✅ $service_name started (container: $container_id)"
  ) > "$log_file" 2>&1 || true

  wait_for_health "$service_name" "$url/health" "$log_file" 240 || true
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

#  Envid Metadata (Multimodal only)
start_local_moderation_service
# local-moderation-nsfwjs is managed by systemd (always-on). Do not start here.
start_local_label_detection_service
start_local_ocr_paddle_service
start_local_keyscene_best_service
start_service "envid-metadata-multimodal" "code/envidMetadataGCP" "app.py" "5016" "$ENV_MULTIMODAL_LOCAL_FILE,$ENV_MULTIMODAL_TARGET_FILE,$ENV_MULTIMODAL_SECRETS_FILE,$ENV_MULTIMODAL_TARGET_SECRETS_FILE"

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
