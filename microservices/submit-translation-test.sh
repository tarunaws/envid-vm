#!/bin/bash
#
# Submit Translation Test Job
# Submits a video processing job with all Indian languages
#

set -e

VIDEO_URI="gs://envid-metadata-tarun/rawVideo/abhayPromo.mp4"
JOB_ID="abhayPromo"
BACKEND_URL="http://localhost/backend"

# All IndicTrans2 supported languages
LANGUAGES="hi,bn,gu,kn,ml,mr,or,pa,ta,te,ur,as,ne,ks,kok,doi,sd,sa,mai"

echo "=================================================="
echo "  Submitting Translation Test Job"
echo "=================================================="
echo
echo "Video URI: $VIDEO_URI"
echo "Job ID: $JOB_ID"
echo "Languages: $LANGUAGES"
echo

# Create job payload
PAYLOAD=$(cat <<EOF
{
  "gcs_uri": "$VIDEO_URI",
  "job_id": "$JOB_ID",
  "task_selection": {
    "enable_transcribe": true,
    "transcribe_language": "hi",
    "enable_translate_output": true,
    "translate_targets": ["hi","bn","gu","kn","ml","mr","or","pa","ta","te","ur","as","ne","ks","kok","doi","sd","sa","mai"]
  }
}
EOF
)

echo "Submitting job..."
echo

# Submit the job
response=$(curl -s -X POST "${BACKEND_URL}/process-gcs-video-cloud" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" 2>&1)

echo "Response:"
echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
echo

# Use job_id returned by backend when available
job_id_resp=$(echo "$response" | python3 - <<'PY'
import json, sys
try:
  data = json.load(sys.stdin)
  job_id = data.get("job_id") or (data.get("job") or {}).get("id") or ""
  print(job_id)
except Exception:
  print("")
PY
)
if [ -z "$job_id_resp" ]; then
  job_id_resp=$(echo "$response" | sed -n 's/.*"job_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n1)
fi
if [ -z "$job_id_resp" ]; then
  job_id_resp=$(echo "$response" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n1)
fi
if [ -n "$job_id_resp" ]; then
  JOB_ID="$job_id_resp"
fi

# Check if job was created successfully
if echo "$response" | grep -q "job_id\|$JOB_ID"; then
    echo "✓ Job submitted successfully!"
    echo
    echo "Monitor the job with:"
    echo "  ./monitor-translation-job.sh $JOB_ID"
    echo
  echo "Collecting job stats (job_stats.json will be written to the job folder)..."
  echo "  python3 ./job-stats-monitor.py $JOB_ID $BACKEND_URL"
  python3 ./job-stats-monitor.py "$JOB_ID" "$BACKEND_URL"
  echo
    echo "Or check job status:"
    echo "  curl -s http://localhost/backend/jobs/$JOB_ID | python3 -m json.tool"
    echo
    echo "Check translation health:"
    echo "  ./check-translation-health.sh"
else
    echo "✗ Job submission may have failed. Check the response above."
    exit 1
fi
