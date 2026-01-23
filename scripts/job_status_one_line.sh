#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <job_id> [api_base] [vm_name] [project] [zone]" >&2
  exit 1
fi

JOB_ID="$1"
API_BASE="${2:-http://34.23.26.211/envid-multimodal}"
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

steps={s.get("id"):s for s in payload.get("steps",[])}
current_step=None
for s in payload.get("steps",[]):
    if s.get("status") == "running":
        current_step=s
        break
if current_step is None:
    current_step=payload.get("steps",[-1])[-1] if payload.get("steps") else {}
print(payload.get("status","unknown"))
print(payload.get("progress",0) or 0)
print(current_step.get("name") or current_step.get("id") or "unknown")
print(current_step.get("percent") or 0)
' <<< "$status_json" > /tmp/job_status_parsed.txt

  read -r overall_status < /tmp/job_status_parsed.txt
  read -r overall_progress < <(sed -n '2p' /tmp/job_status_parsed.txt)
  read -r current_task < <(sed -n '3p' /tmp/job_status_parsed.txt)
  read -r current_progress < <(sed -n '4p' /tmp/job_status_parsed.txt)

  if ! read -r cpu gpu < <(gcloud compute ssh "$VM_NAME" --project="$PROJECT_ID" --zone="$ZONE" --command "PID=\$(pgrep -f '/home/tarun-envid/envid-metadata/code/envidMetadataGCP/app.py' | head -n 1); if [ -n \"\$PID\" ]; then ps -p \"\$PID\" -o %cpu --no-headers | awk '{print int(\$1)}'; else echo 0; fi; if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | awk '{print int(\$1)}'; else echo 0; fi" | tr '\n' ' '); then
    cpu=0
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
