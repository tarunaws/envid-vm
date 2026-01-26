#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "docker compose or docker-compose is required" >&2
  exit 1
fi

cd "${ROOT_DIR}"

compose_file="${SCRIPT_DIR}/docker-compose.app.yml"

mapfile -t services < <("${COMPOSE_CMD[@]}" -f "${compose_file}" config --services)

running_names=$(docker ps --format '{{.Names}}' || true)

to_start=()
for service in "${services[@]}"; do
  container_name=$("${COMPOSE_CMD[@]}" -f "${compose_file}" config | awk -v svc="$service" '
    $0 ~ "^  "svc":" {in_svc=1}
    in_svc && $1=="container_name:" {print $2; exit}
    in_svc && $0 ~ "^  [^ ]" && $0 !~ "^  "svc":" {exit}
  ')

  if [ -z "${container_name}" ]; then
    container_name="${service}"
  fi

  if echo "${running_names}" | grep -q "^${container_name}$"; then
    echo "✅ ${container_name} already running. Skipping build."
  else
    to_start+=("${service}")
  fi
done

if [ ${#to_start[@]} -eq 0 ]; then
  echo "✅ All containers are already running."
  exit 0
fi

exec "${COMPOSE_CMD[@]}" -f "${compose_file}" up -d --build "${to_start[@]}"
