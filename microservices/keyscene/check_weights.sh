#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="${ROOT}/weights/transnetv2-weights"

if [[ ! -d "$W" ]]; then
  echo "❌ Missing directory: $W"
  echo "Put TransNetV2 weights here (must contain saved_model.pb + variables/)."
  exit 1
fi

if [[ ! -f "$W/saved_model.pb" ]]; then
  echo "❌ Missing: $W/saved_model.pb"
  exit 1
fi

if [[ ! -d "$W/variables" ]]; then
  echo "❌ Missing: $W/variables/"
  exit 1
fi

n_vars="$(ls -1 "$W/variables" 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$n_vars" -lt 1 ]]; then
  echo "❌ variables/ is empty: $W/variables"
  exit 1
fi

echo "✅ TransNetV2 weights look present"
echo "- $W/saved_model.pb"
echo "- $n_vars files in $W/variables/"
