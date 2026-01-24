#!/usr/bin/env bash
set -euo pipefail

# Create sanitized env samples and push git changes.
# Usage: ./scripts/git_push_sanitized.sh "commit message"

msg="${1:-Update sanitized env samples}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

./scripts/sanitize_env_samples.sh

git add -A
git commit -m "$msg"
git push
