#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUSH_SCRIPT="${SCRIPT_DIR}/push-images.sh"

if [[ -x "${PUSH_SCRIPT}" ]]; then
  exec "${PUSH_SCRIPT}" "$@"
fi

echo "push-images.sh not found or not executable" >&2
exit 1