#!/usr/bin/env bash
set -euo pipefail

REPO="${DOCKERHUB_REPO:-tarunaws/envid-metadata}"
STATE_FILE="${STATE_FILE:-/home/tarun-envid/envid-metadata/docker-image-versions.txt}"

if [[ $# -eq 0 ]]; then
  echo "No images specified; auto-detecting local :latest images."
  mapfile -t auto_images < <(
    docker images --format '{{.Repository}} {{.Tag}}' \
      | awk '$2=="latest" {print $1}' \
      | grep -v '^<none>$' \
      | grep -v "^${REPO%%/*}/${REPO##*/}$" \
      | sort -u
  )
  if [[ ${#auto_images[@]} -eq 0 ]]; then
    echo "No local :latest images found to push."
    exit 1
  fi
  set -- "${auto_images[@]}"
else
  echo "Using specified images: $*"
fi

declare -A version_map
declare -A id_map

if [[ -f "${STATE_FILE}" ]]; then
  while IFS='|' read -r name ver img_id; do
    [[ -z "${name}" ]] && continue
    version_map["${name}"]="${ver}"
    id_map["${name}"]="${img_id}"
  done < "${STATE_FILE}"
fi

echo "Pruning dangling images..."
docker image prune -f >/dev/null

for image in "$@"; do
  src="${image}:latest"
  if ! docker image inspect "${src}" >/dev/null 2>&1; then
    echo "Skipping ${image}: image ${src} not found (or pruned)"
    continue
  fi

  img_id="$(docker image inspect --format '{{.Id}}' "${src}")"
  prev_id="${id_map["${image}"]-}"
  if [[ -n "${prev_id}" && "${img_id}" == "${prev_id}" ]]; then
    echo "Skipping ${image}: no changes since last push"
    continue
  fi

  prev_ver="${version_map["${image}"]-v0}"
  if [[ "${prev_ver}" =~ ^v([0-9]+)$ ]]; then
    next_ver="v$((BASH_REMATCH[1] + 1))"
  else
    next_ver="v1"
  fi

  dst="${REPO}:${image}-${next_ver}"
  echo "Tagging ${src} -> ${dst}"
  docker tag "${src}" "${dst}"
  echo "Pushing ${dst}"
  docker push "${dst}"

  version_map["${image}"]="${next_ver}"
  id_map["${image}"]="${img_id}"
done

tmp_state="${STATE_FILE}.tmp"
> "${tmp_state}"
for name in "${!version_map[@]}"; do
  echo "${name}|${version_map["${name}"]}|${id_map["${name}"]}" >> "${tmp_state}"
done
sort -o "${tmp_state}" "${tmp_state}"
mv "${tmp_state}" "${STATE_FILE}"
echo "Updated version state in ${STATE_FILE}"