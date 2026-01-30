#!/usr/bin/env bash
set -euo pipefail

REPO="${DOCKERHUB_REPO:-tarunaws/ai}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
COMPOSE_FILE="${ROOT_DIR}/microservices/docker-compose.app.yml"

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "docker compose or docker-compose is required" >&2
  exit 1
fi

detect_changed_services() {
  local base_ref="${1:-origin/main}"
  local changed_files=""

  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if git rev-parse --verify "${base_ref}" >/dev/null 2>&1; then
      changed_files=$(git diff --name-only "${base_ref}"...HEAD || true)
    else
      changed_files=$(git diff --name-only HEAD~1..HEAD || true)
    fi
  fi

  if [[ -z "${changed_files}" ]]; then
    return 1
  fi

  declare -A service_patterns
  service_patterns[backend]="microservices/backend/"
  service_patterns[frontend]="microservices/frontend/"
  service_patterns[audio-transcription]="microservices/audio-transcription/"
  service_patterns[audio-translation]="microservices/audio-translation/"
  service_patterns[transcoder]="microservices/transcoder/"
  service_patterns[moderation]="microservices/moderation/"
  service_patterns[keyscene]="microservices/keyscene/"
  service_patterns[scene-detect]="microservices/scene-detect/"
  service_patterns[text-on-video]="microservices/text-on-video/"
  service_patterns[document-extractor]="microservices/document-extractor/"
  service_patterns[translate]="microservices/translate/"
  service_patterns[reverseproxy]="microservices/reverseproxy/"
  service_patterns[gateway]="microservices/gateway/"
  service_patterns[dns]="microservices/dns/"
  service_patterns[ingest]="microservices/ingest/"
  service_patterns[keycloak]="microservices/keycloak/"
  service_patterns[tracing]="microservices/tracing/"
  service_patterns[otel-collector]="microservices/otel-collector/"
  service_patterns[metadata-export]="microservices/metadata-export/"

  local services=()
  while IFS= read -r file; do
    for svc in "${!service_patterns[@]}"; do
      if [[ "${file}" == ${service_patterns[$svc]}* ]]; then
        services+=("${svc}")
      fi
    done
  done <<< "${changed_files}"

  if [[ ${#services[@]} -eq 0 ]]; then
    return 1
  fi

  printf '%s\n' "${services[@]}" | sort -u
}

if [[ $# -eq 0 ]]; then
  echo "No images specified; auto-detecting changed services (git diff)."
  mapfile -t changed_services < <(detect_changed_services || true)
  if [[ ${#changed_services[@]} -gt 0 ]]; then
    echo "Rebuilding changed services: ${changed_services[*]}"
    (cd "${ROOT_DIR}" && "${COMPOSE_CMD[@]}" -f "${COMPOSE_FILE}" build "${changed_services[@]}")
    set -- "${changed_services[@]}"
  else
    echo "No change list detected; auto-detecting local images."
    mapfile -t auto_images < <(
      docker images --format '{{.Repository}}:{{.Tag}}' \
        | awk '$1 !~ /^<none>:/ {print $1}' \
        | grep -E "^${REPO}:" \
        | sort -u
    )
    if [[ ${#auto_images[@]} -eq 0 ]]; then
      mapfile -t auto_images < <(
        docker images --format '{{.Repository}} {{.Tag}}' \
          | awk '$2=="latest" {print $1}' \
          | grep -v '^<none>$' \
          | sort -u
      )
    fi
    if [[ ${#auto_images[@]} -eq 0 ]]; then
      echo "No local images found to push."
      exit 1
    fi
    set -- "${auto_images[@]}"
  fi
else
  echo "Using specified images: $*"
fi

echo "Pruning dangling images..."
docker image prune -f >/dev/null

for image in "$@"; do
  if [[ "${image}" == ${REPO}:* ]]; then
    if docker image inspect "${image}" >/dev/null 2>&1; then
      echo "Pushing ${image}"
      docker push "${image}"
    else
      echo "Skipping ${image}: image not found locally"
    fi
    continue
  fi

  src_repo="${REPO}:${image}"
  if docker image inspect "${src_repo}" >/dev/null 2>&1; then
    echo "Pushing ${src_repo}"
    docker push "${src_repo}"
    continue
  fi

  src="${image}:latest"
  if ! docker image inspect "${src}" >/dev/null 2>&1; then
    echo "Skipping ${image}: image ${src} not found (or pruned)"
    continue
  fi

  dst="${REPO}:${image}"
  echo "Tagging ${src} -> ${dst}"
  docker tag "${src}" "${dst}"
  echo "Pushing ${dst}"
  docker push "${dst}"
done