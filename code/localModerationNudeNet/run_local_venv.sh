#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY_BIN="${PY_BIN:-}"
if [[ -z "$PY_BIN" ]]; then
  # Prefer python3.11 if present.
  if command -v python3.11 >/dev/null 2>&1; then
    PY_BIN="python3.11"
  elif command -v python3.12 >/dev/null 2>&1; then
    PY_BIN="python3.12"
  else
    echo "❌ Need Python 3.11 or 3.12 installed (python3.11/python3.12 not found)."
    echo "   On macOS with Homebrew: brew install python@3.11"
    exit 1
  fi
fi

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-nudenet}"
PORT="${PORT:-5081}"
MARKER_FILE="$VENV_DIR/.deps_installed"

echo "Using PY_BIN=$PY_BIN"
echo "Using VENV_DIR=$VENV_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PY_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install -U pip setuptools wheel

# Install deps (once by default).
if [[ -f "$MARKER_FILE" && "${NUDENET_SERVICE_FORCE_INSTALL:-}" != "1" ]]; then
  echo "ℹ️  Deps already installed (marker: $MARKER_FILE). Skipping pip install."
else
  echo "📦 Installing deps from requirements.txt"
  "$VENV_DIR/bin/pip" install -r "$ROOT_DIR/requirements.txt"
  touch "$MARKER_FILE"
fi

echo "Starting local NudeNet service on http://localhost:$PORT"
export FLASK_APP=app
exec "$VENV_DIR/bin/python" -m flask run --host 0.0.0.0 --port "$PORT"
