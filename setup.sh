#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
MICROSERVICES_DIR="${ROOT_DIR}/microservices"
DOCKERHUB_REPO="${DOCKERHUB_REPO:-tarunaws/envid-metadata}"

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

if [ ! -f "${MICROSERVICES_DIR}/.env" ] && [ -f "${MICROSERVICES_DIR}/.env.sample" ]; then
	cp "${MICROSERVICES_DIR}/.env.sample" "${MICROSERVICES_DIR}/.env"
	echo "✅ Created microservices/.env from microservices/.env.sample"
fi

if [ ! -d "/mnt/gcs" ]; then
	echo "❌ /mnt/gcs is not mounted. Mount GCS before running setup."
	exit 1
fi

mkdir -p /mnt/gcs/envid-metadata /mnt/gcs/translate/models /mnt/gcs/translate/cache /mnt/gcs/db

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

get_latest_tag() {
	python3 - "$DOCKERHUB_REPO" "$1" <<'PY'
import json
import sys
import urllib.request
from urllib.parse import quote

repo = sys.argv[1]
image = sys.argv[2]

tags = []
url = f"https://hub.docker.com/v2/repositories/{repo}/tags?page_size=100&name={quote(image + '-v')}"
while url:
	try:
		with urllib.request.urlopen(url, timeout=15) as resp:
			data = json.load(resp)
	except Exception:
		break
	for item in data.get("results", []) or []:
		name = str(item.get("name") or "")
		if name.startswith(f"{image}-v"):
			tags.append(name)
	url = data.get("next")

def ver(tag: str) -> int:
	try:
		return int(tag.rsplit("-v", 1)[1])
	except Exception:
		return -1

tags = sorted(tags, key=ver, reverse=True)
print(tags[0] if tags else "")
PY
}

ensure_image() {
	local image_name="$1"
	if docker images --format '{{.Repository}}' | grep -q "^${image_name}$"; then
		echo "✅ Image ${image_name} exists locally."
		return 0
	fi

	latest_tag="$(get_latest_tag "${image_name}")"
	if [ -n "${latest_tag}" ]; then
		echo "⬇️  Pulling ${DOCKERHUB_REPO}:${latest_tag}"
		if docker pull "${DOCKERHUB_REPO}:${latest_tag}"; then
			docker tag "${DOCKERHUB_REPO}:${latest_tag}" "${image_name}:latest"
			echo "✅ Tagged ${image_name}:latest from Docker Hub"
			return 0
		fi
	fi

	echo "🛠️  Building ${image_name} from Dockerfile"
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
			image_name = img_match.group(1).strip().split(":")[0]

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
		fi
		echo "▶️  Starting ${dns_service} (${dns_container})..."
		"${COMPOSE_CMD[@]}" -f "${compose_file}" up -d "${dns_service}"
	fi
fi

echo "✅ Setup complete. Review microservices/.env and microservices/*/.env before starting the stack."
