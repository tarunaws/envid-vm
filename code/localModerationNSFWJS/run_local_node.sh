#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PORT="${PORT:-5082}"

if ! command -v node >/dev/null 2>&1; then
  echo "❌ node not found; install Node.js (or nvm)" >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "❌ npm not found; install Node.js (or nvm)" >&2
  exit 1
fi

if [[ ! -d node_modules ]]; then
  echo "📦 Installing dependencies (npm install)..."
  npm install
fi

export PORT
exec npm start
