#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# MediaGenAI deployment helper
#
# Syncs the current local repository to the EC2 instance dev directory,
# snapshots a release, and promotes it to the prod directory.
#
# Environment overrides:
#   DEPLOY_KEY      - path to the SSH private key (.pem)
#   DEPLOY_HOST     - SSH host (e.g. ec2-user@ec2-34-232-17-177.compute-1.amazonaws.com)
#   REMOTE_BASE     - base path on EC2 (default: /media)
#   PROJECT_NAME    - project folder name (default: mediaGenAI)
#   LOCAL_ROOT      - local project root (default: current working directory)
#   RSYNC_EXCLUDES  - additional rsync exclude patterns (space separated)
# -----------------------------------------------------------------------------

# Use relative path from project - assumes keys are in ../keys relative to mediaGenAI project
# Override with DEPLOY_KEY environment variable
DEFAULT_KEY="$HOME/mydrive/Projects/keys/useast/mediaGenAILab.pem"
DEFAULT_HOST="ec2-user@ec2-34-232-17-177.compute-1.amazonaws.com"

DEPLOY_KEY=${DEPLOY_KEY:-$DEFAULT_KEY}
DEPLOY_HOST=${DEPLOY_HOST:-$DEFAULT_HOST}
REMOTE_BASE=${REMOTE_BASE:-/media}
PROJECT_NAME=${PROJECT_NAME:-mediaGenAI}
LOCAL_ROOT=${LOCAL_ROOT:-$(pwd)}

if [[ ! -f "$DEPLOY_KEY" ]]; then
  echo "[deploy] SSH key not found at $DEPLOY_KEY" >&2
  exit 1
fi

if [[ ! -d "$LOCAL_ROOT" ]]; then
  echo "[deploy] Local root $LOCAL_ROOT does not exist" >&2
  exit 1
fi

RSYNC_COMMON_EXCLUDES=(
  --exclude '.git'
  --exclude '.venv'
  --exclude '__pycache__'
  --exclude '*.pyc'
  --exclude 'node_modules'
  --exclude '.DS_Store'
  --exclude 'frontend/build'
  --exclude 'frontend/node_modules'
  --exclude 'uploads'
  --exclude 'outputs'
  --exclude '*.log'
  --exclude '*.pid'
)

if [[ -n "${RSYNC_EXCLUDES:-}" ]]; then
  for pattern in $RSYNC_EXCLUDES; do
    RSYNC_COMMON_EXCLUDES+=(--exclude "$pattern")
  done
fi

SSH_OPTS=(-i "$DEPLOY_KEY" -o StrictHostKeyChecking=accept-new)
REMOTE_DEV="$REMOTE_BASE/dev/$PROJECT_NAME"
REMOTE_PROD="$REMOTE_BASE/prod/$PROJECT_NAME"
REMOTE_RELEASES="$REMOTE_BASE/releases/$PROJECT_NAME"

printf '[deploy] Ensuring remote directories exist...\n'
ssh "${SSH_OPTS[@]}" "$DEPLOY_HOST" "mkdir -p '$REMOTE_DEV' '$REMOTE_PROD' '$REMOTE_RELEASES'"

printf '[deploy] Syncing local repository to remote dev directory...\n'
rsync -avz --delete "${RSYNC_COMMON_EXCLUDES[@]}" -e "ssh -i $DEPLOY_KEY" "$LOCAL_ROOT/" "$DEPLOY_HOST:$REMOTE_DEV/"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
REMOTE_RELEASE="$REMOTE_RELEASES/$TIMESTAMP"

read -r -d '' REMOTE_SCRIPT <<'EOSH'
set -euo pipefail
REMOTE_DEV="$1"
REMOTE_PROD="$2"
REMOTE_RELEASE_BASE="$3"
TIMESTAMP="$4"
REMOTE_RELEASE="$REMOTE_RELEASE_BASE/$TIMESTAMP"

mkdir -p "$REMOTE_RELEASE"
rsync -a --delete "$REMOTE_DEV/" "$REMOTE_RELEASE/"
rsync -a --delete "$REMOTE_RELEASE/" "$REMOTE_PROD/"
EOSH

printf '[deploy] Creating release snapshot and promoting to prod...\n'
ssh "${SSH_OPTS[@]}" "$DEPLOY_HOST" "bash -c '$(printf "%q" "$REMOTE_SCRIPT")' -- '$REMOTE_DEV' '$REMOTE_PROD' '$REMOTE_RELEASES' '$TIMESTAMP'"

printf '[deploy] Deployment complete. Release stored at %s\n' "$REMOTE_RELEASE"
