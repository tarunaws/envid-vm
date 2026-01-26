#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEYSCENE_DIR="$CODE_ROOT/localKeySceneBest"
WEIGHTS_DST="$KEYSCENE_DIR/weights/transnetv2-weights"
PORT="${ENVID_LOCAL_KEYSCENE_PORT:-5085}"
HEALTH_URL="http://localhost:${PORT}/health"

say() { printf "%s\n" "$*"; }

docker_ready() {
  docker info >/dev/null 2>&1
}

ensure_docker() {
  if docker_ready; then
    return 0
  fi

  if [[ "$(uname -s)" == "Darwin" ]]; then
    if command -v docker >/dev/null 2>&1 && docker desktop start >/dev/null 2>&1; then
      :
    else
      open -a Docker >/dev/null 2>&1 || true
    fi
  fi

  # Wait up to ~2 minutes for the daemon.
  for i in $(seq 1 60); do
    if docker_ready; then
      return 0
    fi
    sleep 2
  done

  say "❌ Docker daemon is not reachable."
  say "   - On macOS: open Docker Desktop and wait until it finishes starting."
  say "   - Then re-run: $0"
  exit 1
}

ensure_git_lfs() {
  if command -v git-lfs >/dev/null 2>&1 || git lfs version >/dev/null 2>&1; then
    return 0
  fi

  if command -v brew >/dev/null 2>&1; then
    say "Installing git-lfs via Homebrew…"
    brew list git-lfs >/dev/null 2>&1 || brew install git-lfs
    git lfs install --skip-repo >/dev/null 2>&1 || true
    return 0
  fi

  say "❌ git-lfs is required to download TransNetV2 weights, but is not installed."
  say "   Install it (e.g. Homebrew: brew install git-lfs) and re-run: $0"
  exit 1
}

ensure_transnet_weights() {
  if [[ -f "$WEIGHTS_DST/saved_model.pb" && -d "$WEIGHTS_DST/variables" ]]; then
    return 0
  fi

  ensure_git_lfs

  say "Downloading TransNetV2 weights (git-lfs)…"
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT

  git clone --depth 1 https://github.com/soCzech/TransNetV2.git "$tmpdir/TransNetV2" >/dev/null
  ( cd "$tmpdir/TransNetV2" && git lfs pull --include='inference/transnetv2-weights/*' )

  mkdir -p "$WEIGHTS_DST"
  rsync -a --delete "$tmpdir/TransNetV2/inference/transnetv2-weights/" "$WEIGHTS_DST/"

  if [[ ! -f "$WEIGHTS_DST/saved_model.pb" ]]; then
    say "❌ Weights download did not produce saved_model.pb at: $WEIGHTS_DST"
    exit 1
  fi
}

start_sidecar() {
  say "Starting keyscene sidecar on :$PORT…"
  ( cd "$KEYSCENE_DIR" && docker compose up -d --build )

  for i in $(seq 1 60); do
    if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
      say "✅ Sidecar healthy: $HEALTH_URL"
      return 0
    fi
    sleep 2
  done

  say "❌ Sidecar did not become healthy: $HEALTH_URL"
  ( cd "$KEYSCENE_DIR" && docker compose ps ) || true
  exit 1
}

start_backend() {
  # Backend scripts live in code/ root.
  say "Ensuring backend is running (5016)…"
  ( cd "$CODE_ROOT" && ./start-backend.sh )
}

main() {
  ensure_docker
  ensure_transnet_weights
  start_sidecar
  start_backend

  say ""
  say "Done. Combo key-scene models should work now:"
  say "- transnetv2_clip_cluster"
  say "- pyscenedetect_clip_cluster"
}

main "$@"
