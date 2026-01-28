#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
MICROSERVICES_DIR="${ROOT_DIR}/microservices"

if ! command -v docker >/dev/null 2>&1; then
	echo "❌ docker not found. Install Docker before continuing."
	exit 1
fi

if ! docker info >/dev/null 2>&1; then
	echo "❌ Docker daemon is not running. Start Docker and re-run setup."
	exit 1
fi

if docker compose version >/dev/null 2>&1; then
	COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
	COMPOSE_CMD=(docker-compose)
else
	echo "❌ docker compose or docker-compose is required." >&2
	exit 1
fi

PUSH_SCRIPT="${ROOT_DIR}/push-images.sh"
did_build=false

# text-normalizer is now hosted inside translate. Remove legacy container if present.
if docker ps -a --format '{{.Names}}' | grep -q '^text-normalizer$'; then
	echo "🧹 Removing legacy text-normalizer container"
	docker rm -f text-normalizer >/dev/null 2>&1 || true
fi

# document-extractor service removed. Remove legacy container if present.
if docker ps -a --format '{{.Names}}' | grep -q '^document-extractor$'; then
	echo "🧹 Removing legacy document-extractor container"
	docker rm -f document-extractor >/dev/null 2>&1 || true
fi

# Mongo service is now named db. Remove legacy mongo container if present.
if docker ps -a --format '{{.Names}}' | grep -q '^mongo$'; then
	echo "🧹 Removing legacy mongo container"
	docker rm -f mongo >/dev/null 2>&1 || true
fi


if [ ! -f "${MICROSERVICES_DIR}/.env" ] && [ -f "${MICROSERVICES_DIR}/.env.sample" ]; then
	cp "${MICROSERVICES_DIR}/.env.sample" "${MICROSERVICES_DIR}/.env"
	echo "✅ Created microservices/.env from microservices/.env.sample"
fi

if [ -f "${MICROSERVICES_DIR}/.env" ]; then
	cp "${MICROSERVICES_DIR}/.env" "${ROOT_DIR}/.env"
	echo "✅ Synced .env from microservices/.env to repo root"
fi

if [ -f "${MICROSERVICES_DIR}/.env" ]; then
	echo "🔁 Syncing .env to service folders..."
	for svc_dir in "${MICROSERVICES_DIR}"/*; do
		if [ -d "${svc_dir}" ] && [ -f "${svc_dir}/Dockerfile" ]; then
			cp "${MICROSERVICES_DIR}/.env" "${svc_dir}/.env"
		fi
	done
fi

if [ ! -d "/mnt/gcs" ]; then
	echo "❌ /mnt/gcs is not mounted. Mount GCS before running setup."
	exit 1
fi

mkdir -p /mnt/gcs/envid-metadata /mnt/gcs/translate/models /mnt/gcs/translate/cache /mnt/gcs/db /mnt/processing/mongo

MICROSERVICES_DIR="${MICROSERVICES_DIR}" python3 - <<'PY'
from pathlib import Path
import re

root_dir = Path((__import__("os").environ.get("MICROSERVICES_DIR") or ".")).resolve()
compose_path = root_dir / "docker-compose.app.yml"
hosts_path = root_dir / "dns" / "code" / "hosts"

text = compose_path.read_text()
lines = text.splitlines()

in_services = False
current_service = None
service_ips = []
aliases_map = {}
aliases_mode = False

def add_alias(service, alias):
	aliases_map.setdefault(service, []).append(alias)

for line in lines:
	if line.startswith("services:"):
		in_services = True
		continue
	if not in_services:
		continue
	if re.match(r"^(networks|volumes):\s*$", line):
		in_services = False
		current_service = None
		aliases_mode = False
		continue

	service_match = re.match(r"^  ([A-Za-z0-9_.-]+):\s*$", line)
	if service_match:
		current_service = service_match.group(1)
		aliases_mode = False
		continue

	if current_service is None:
		continue

	if re.match(r"^\s*aliases:\s*$", line):
		aliases_mode = True
		continue

	if aliases_mode:
		alias_match = re.match(r"^\s*-\s*([A-Za-z0-9_.-]+)\s*$", line)
		if alias_match:
			add_alias(current_service, alias_match.group(1))
			continue
		if re.match(r"^\s*\S", line):
			# Another key encountered
			aliases_mode = False

	ip_match = re.match(r"^\s*ipv4_address:\s*([0-9.]+)\s*$", line)
	if ip_match:
		service_ips.append((current_service, ip_match.group(1)))

header = "# Update these IPs if you change the docker-compose static addresses"
rows = [header]

for service, ip in service_ips:
	names = [service]
	names.extend(aliases_map.get(service, []))
	# de-dup while preserving order
	seen = set()
	unique_names = []
	for name in names:
		if name not in seen:
			seen.add(name)
			unique_names.append(name)
	rows.append(f"{ip:<12} " + " ".join(unique_names))

hosts_path.write_text("\n".join(rows) + "\n")
print("✅ Updated dns/hosts from docker-compose.app.yml")
PY

compose_file="${MICROSERVICES_DIR}/docker-compose.app.yml"

ensure_image() {
	local image_ref="$1"
	local image_full="${image_ref}"
	if [[ "${image_full}" != *:* ]]; then
		image_full="${image_full}:latest"
	fi

	if docker image inspect "${image_full}" >/dev/null 2>&1; then
		echo "✅ Image ${image_full} exists locally."
		return 0
	fi

	echo "⬇️  Pulling ${image_full}"
	if docker pull "${image_full}"; then
		echo "✅ Pulled ${image_full}"
		return 0
	fi

	echo "🛠️  Building ${image_ref} from Dockerfile"
	return 1
}

while IFS=$'\t' read -r service container_name image_name; do
	if [ -z "${service}" ]; then
		continue
	fi

	if [ "${service}" = "dns" ]; then
		continue
	fi

	if [ -z "${container_name}" ]; then
		container_name="${service}"
	fi

	if [ -z "${image_name}" ]; then
		image_name="${service}"
	fi

	if docker ps --format '{{.Names}}' | grep -q "^${container_name}$"; then
		echo "✅ ${container_name} already running. Skipping."
		continue
	fi

	if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
		echo "▶️  Starting existing ${service} (${container_name})..."
		"${COMPOSE_CMD[@]}" -f "${compose_file}" up -d "${service}"
	else
		if ! ensure_image "${image_name}"; then
			"${COMPOSE_CMD[@]}" -f "${compose_file}" build "${service}"
			did_build=true
		fi
		echo "▶️  Starting ${service} (${container_name})..."
		"${COMPOSE_CMD[@]}" -f "${compose_file}" up -d "${service}"
	fi
done < <(MICROSERVICES_DIR="${MICROSERVICES_DIR}" python3 - <<'PY'
from pathlib import Path
import re

compose_path = Path((__import__("os").environ.get("MICROSERVICES_DIR") or ".")).resolve() / "docker-compose.app.yml"
lines = compose_path.read_text().splitlines()

in_services = False
current_service = None
container_name = None
image_name = None

def emit(service, cname, image):
	if service:
		print(f"{service}\t{cname or ''}\t{image or ''}")

for line in lines:
	if line.startswith("services:"):
		in_services = True
		continue
	if not in_services:
		continue
	if re.match(r"^(networks|volumes):\s*$", line):
		emit(current_service, container_name, image_name)
		break

	svc_match = re.match(r"^  ([A-Za-z0-9_.-]+):\s*$", line)
	if svc_match:
		emit(current_service, container_name, image_name)
		current_service = svc_match.group(1)
		container_name = None
		image_name = None
		continue

	if current_service:
		cname_match = re.match(r"^    container_name:\s*\"?([^\"#]+)\"?\s*$", line)
		if cname_match:
			container_name = cname_match.group(1).strip()
		img_match = re.match(r"^    image:\s*\"?([^\"#]+)\"?\s*$", line)
		if img_match:
			image_name = img_match.group(1).strip()

emit(current_service, container_name, image_name)
PY
)

# Build dns last (depends on finalized service IPs)
dns_service="dns"
dns_container="dns"
dns_image="dns"

if docker ps --format '{{.Names}}' | grep -q "^${dns_container}$"; then
	echo "✅ ${dns_container} already running. Skipping."
else
	if docker ps -a --format '{{.Names}}' | grep -q "^${dns_container}$"; then
		echo "▶️  Starting existing ${dns_service} (${dns_container})..."
		"${COMPOSE_CMD[@]}" -f "${compose_file}" up -d "${dns_service}"
	else
		if ! ensure_image "${dns_image}"; then
			"${COMPOSE_CMD[@]}" -f "${compose_file}" build "${dns_service}"
			did_build=true
		fi
		echo "▶️  Starting ${dns_service} (${dns_container})..."
		"${COMPOSE_CMD[@]}" -f "${compose_file}" up -d "${dns_service}"
	fi
fi

	if [[ "${did_build}" == "true" ]]; then
		if [[ -x "${PUSH_SCRIPT}" ]]; then
			echo "⬆️  Pushing updated images to Docker Hub"
			"${PUSH_SCRIPT}"
		else
			echo "⚠️  ${PUSH_SCRIPT} not found or not executable; skipping push"
		fi
	fi

echo "✅ Setup complete. Review microservices/.env and microservices/*/.env before starting the stack."
