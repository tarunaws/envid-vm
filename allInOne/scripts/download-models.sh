#!/usr/bin/env bash
set -euo pipefail

# Download Argos models into the named volume so LibreTranslate starts fast.
# This uses the LibreTranslate image's Python environment and Argos CLI.

VOLUME_NAME="libretranslate_models"
IMAGE="libretranslate/libretranslate:v1.6.0"

pairs=(
  "en hi" "hi en"
  "en ja" "ja en"
  "en ko" "ko en"
  "en zh" "zh en"
  "en ta" "ta en"
  "en te" "te en"
  "en bn" "bn en"
  "en vi" "vi en"
  "en th" "th en"
  "en id" "id en"
)

# Ensure the volume exists
if ! docker volume inspect "$VOLUME_NAME" >/dev/null 2>&1; then
  docker volume create "$VOLUME_NAME" >/dev/null
fi

# Download models into /data/.local/share/argos-translate by setting HOME
for pair in "${pairs[@]}"; do
  from="$(echo "$pair" | awk '{print $1}')"
  to="$(echo "$pair" | awk '{print $2}')"
  echo "Downloading $from -> $to"
  docker run --rm \
    --user 0:0 \
    --entrypoint sh \
    -e HOME=/data \
    -e XDG_DATA_HOME=/data/.local/share \
    -e ARGOS_TRANSLATE_DATA_DIR=/data/.local/share/argos-translate \
    -v "$VOLUME_NAME":/data \
    "$IMAGE" \
    -lc "mkdir -p /data/.local/share/argos-translate && /app/venv/bin/python -m argostranslate.cli install --from-lang '$from' --to-lang '$to'"
done

echo "Done. Models cached in volume: $VOLUME_NAME"
