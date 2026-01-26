#!/usr/bin/env bash
set -euo pipefail

# Create sanitized .sample copies of VM env files by redacting secrets.
# Usage: ./scripts/sanitize_env_samples.sh [file ...]

REDACT_KEYS_REGEX='(KEY|TOKEN|SECRET|PASSWORD)'

sanitize_file() {
  local src="$1"
  local dst="${src}.sample"

  if [[ ! -f "$src" ]]; then
    echo "skip: $src (missing)" >&2
    return 0
  fi

  awk -F= -v re="$REDACT_KEYS_REGEX" '
    {
      key=$1
      if (key ~ re) {
        print key"=REDACTED"
      } else {
        print $0
      }
    }
  ' "$src" > "$dst"
  echo "wrote: $dst"
}

if [[ $# -gt 0 ]]; then
  for f in "$@"; do
    sanitize_file "$f"
  done
  exit 0
fi

sanitize_file ".env.multimodal.vm.local"
sanitize_file "code/.env.multimodal.vm.local"
