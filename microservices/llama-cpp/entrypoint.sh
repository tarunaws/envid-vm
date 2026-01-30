#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/models/llama-3.1-8b-instruct.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
N_CTX="${N_CTX:-4096}"
N_THREADS="${N_THREADS:-8}"

exec python -m llama_cpp.server \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --n_ctx "$N_CTX" \
  --n_threads "$N_THREADS"
