#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <job_id> [api_base] [vm_name] [project] [zone]" >&2
  exit 1
fi

JOB_ID="$1"
API_BASE="${2:-http://34.23.26.211/backend}"
VM_NAME="${3:-gpu-g2-vm}"
PROJECT_ID="${4:-envid-development}"
ZONE="${5:-us-east1-b}"

start_ts=$(date +%s)
count=0
sum_cpu=0
sum_gpu=0

format_duration() {
  local total_seconds="$1"
  local mins=$((total_seconds / 60))
  local secs=$((total_seconds % 60))
  printf "%sm%ss" "$mins" "$secs"
}

read_local_cpu_gpu() {
  local pid_file="$1"
  local alt_pid_file="$2"
  local pid
  pid=$(cat "$pid_file" 2>/dev/null || true)
  if [ -z "$pid" ] && [ -f "$alt_pid_file" ]; then
    pid=$(cat "$alt_pid_file" 2>/dev/null || true)
  fi
  if [ -z "$pid" ]; then
    pid=$(pgrep -f 'envidMetadataGCP/app.py' | head -n 1 || true)
  fi
  if [ -n "$pid" ]; then
    ps -p "$pid" -o %cpu= | awk '{printf "%.0f", $1}'
  else
    printf "0"
  fi
  printf " "
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | awk '{printf "%.0f", $1}'
  else
    printf "0"
  fi
  printf "\n"
}

while true; do
  now_ts=$(date +%s)
  elapsed=$((now_ts - start_ts))

  if ! status_json=$(curl -fsS "$API_BASE/jobs/$JOB_ID"); then
    now_ts=$(date +%s)
    elapsed=$((now_ts - start_ts))
    elapsed_label=$(format_duration "$elapsed")
    printf "overall_progress=%s%% current_task=%s %s_progress=%s%% CPU=%s%% GPU=%s%% elapsed=%s\n" \
      "?" "api_error" "api_error" "?" "0" "0" "$elapsed_label"
    sleep 5
    continue
  fi

  python3 -c 'import json,sys
try:
    payload=json.load(sys.stdin)
except Exception:
    print("unknown")
    print(0)
    print("parse_error")
    print(0)
    raise SystemExit(0)

steps=payload.get("steps",[])
current_step=None
for s in steps:
    if s.get("status") == "running":
        current_step=s
        break
if current_step is None:
    candidates=[s for s in steps if s.get("status") not in (None, "not_started")]
    current_step=candidates[-1] if candidates else (steps[-1] if steps else {})
label=current_step.get("label") or current_step.get("name") or current_step.get("id") or "unknown"
overall_progress=payload.get("progress",0) or 0
percent=current_step.get("percent")
if percent is None or (percent == 0 and overall_progress):
    percent=overall_progress
print(payload.get("status","unknown"))
print(overall_progress)
print(label)
print(percent or 0)
' <<< "$status_json" > /tmp/job_status_parsed.txt

  read -r overall_status < /tmp/job_status_parsed.txt
  read -r overall_progress < <(sed -n '2p' /tmp/job_status_parsed.txt)
  read -r current_task < <(sed -n '3p' /tmp/job_status_parsed.txt)
  read -r current_progress < <(sed -n '4p' /tmp/job_status_parsed.txt)

  pid_file="/home/tarun-envid/envid-metadata/envid-metadata-multimodal.pid"
  alt_pid_file="/home/tarun-envid/envid-metadata/code/envid-metadata-multimodal.pid"
  if [ "${ENVID_STATUS_USE_SSH:-1}" = "0" ] || ! command -v gcloud >/dev/null 2>&1; then
    if ! read -r cpu gpu < <(read_local_cpu_gpu "$pid_file" "$alt_pid_file"); then
      cpu=0
      gpu=0
    fi
  else
    if ! read -r cpu gpu < <(gcloud compute ssh "$VM_NAME" --project="$PROJECT_ID" --zone="$ZONE" --quiet --command "PID_FILE='$pid_file'; ALT_PID_FILE='$alt_pid_file'; PID=\$(cat \"\$PID_FILE\" 2>/dev/null || true); if [ -z \"\$PID\" ] && [ -f \"\$ALT_PID_FILE\" ]; then PID=\$(cat \"\$ALT_PID_FILE\" 2>/dev/null || true); fi; if [ -z \"\$PID\" ]; then PID=\$(pgrep -f 'envidMetadataGCP/app.py' | head -n 1); fi; if [ -n \"\$PID\" ]; then CPU=\$(ps -p \"\$PID\" -o %cpu= | awk '{printf \"%.0f\", \$1}'); else CPU=0; fi; if command -v nvidia-smi >/dev/null 2>&1; then GPU=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | awk '{printf \"%.0f\", \$1}'); else GPU=0; fi; printf \"%s %s\\n\" \"\$CPU\" \"\$GPU\""); then
      cpu=0
      gpu=0
    fi
  fi

  if ! [[ "$cpu" =~ ^[0-9]+$ ]]; then
    cpu=0
  fi
  if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
    gpu=0
  fi

  count=$((count+1))
  sum_cpu=$((sum_cpu+cpu))
  sum_gpu=$((sum_gpu+gpu))

  safe_task=$(echo "$current_task" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g')
  if [ -z "$safe_task" ]; then
    safe_task="task"
  fi
  elapsed_label=$(format_duration "$elapsed")
  printf "overall_progress=%s%% current_task=%s %s_progress=%s%% CPU=%s%% GPU=%s%% elapsed=%s\n" \
    "$overall_progress" "$current_task" "$safe_task" "$current_progress" "$cpu" "$gpu" "$elapsed_label"

  if [ "$overall_status" = "completed" ] || [ "$overall_status" = "failed" ] || [ "$overall_status" = "canceled" ]; then
    avg_cpu=$((sum_cpu / count))
    avg_gpu=$((sum_gpu / count))
    total_time=$((now_ts - start_ts))
    total_label=$(format_duration "$total_time")
    printf "average_cpu=%s%% average_gpu=%s%% total_time=%s\n" "$avg_cpu" "$avg_gpu" "$total_label"
    break
  fi

  sleep 5
done
