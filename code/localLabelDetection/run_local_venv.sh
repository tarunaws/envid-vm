#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY_BIN="${PY_BIN:-}"
if [[ -z "$PY_BIN" ]]; then
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

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-labels}"
PORT="${PORT:-5083}"
MARKER_FILE="$VENV_DIR/.deps_installed"

echo "Using PY_BIN=$PY_BIN"
echo "Using VENV_DIR=$VENV_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PY_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install -U pip setuptools wheel

if [[ -f "$MARKER_FILE" && "${LABEL_SERVICE_FORCE_INSTALL:-}" != "1" ]]; then
  echo "ℹ️  Deps already installed (marker: $MARKER_FILE). Skipping pip install."
else
  echo "📦 Installing deps from requirements.txt"
    "$VENV_DIR/bin/pip" install -r "$ROOT_DIR/requirements.txt"
  touch "$MARKER_FILE"
fi

  # Optional engines: best-effort install so the service can still start.
  if [[ "${LABEL_SERVICE_INSTALL_OPTIONAL:-1}" != "0" ]]; then
    if [[ "${LABEL_SERVICE_INSTALL_OPTIONAL_ASYNC:-1}" == "1" ]]; then
      # Run in background so the service can start immediately.
      nohup env VENV_DIR="$VENV_DIR" "$ROOT_DIR/install_optional_engines.sh" > "$ROOT_DIR/optional_install.log" 2>&1 &
      echo "ℹ️  Optional engine install running in background (see $ROOT_DIR/optional_install.log)"
    else
      env VENV_DIR="$VENV_DIR" "$ROOT_DIR/install_optional_engines.sh" || true
    fi
  fi

echo "Starting local label detection service on http://localhost:$PORT"
export FLASK_APP=app
exec "$VENV_DIR/bin/python" -m flask run --host 0.0.0.0 --port "$PORT"
