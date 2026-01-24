# WhisperX Service

## Overview
FastAPI wrapper for WhisperX with GPU-first defaults, streaming incremental subtitles, batch directory processing, profanity filtering, punctuation/truecasing, and profiling logs.

## Endpoints
- `GET /health`
- `POST /transcribe` (multipart file)
- `POST /batch` (server-side directory processing)

## /transcribe
Form fields:
- `file` (required)
- `stream` (true/false)
- `language` (optional)
- `model` (default `large-v2`)
- `batch_size` (default 32)
- `chunk_seconds` (default 30 for stream, 3600 for non-stream)
- `vad` (default true)
- `diarize` (default false)
- `min_speech_dur` (optional)

Streaming uses SSE (`text/event-stream`) and emits partial subtitles as chunks complete.

## /batch
Form fields:
- `input_dir`
- `output_dir`
- `concurrency` (default 2)

Requires `ENVID_WHISPERX_BATCH_ROOT` to be set; `input_dir` must be under it.

## Environment
- `ENVID_WHISPERX_VAD_METHOD` (default `silero`)
- `ENVID_PUNCTUATION_ENABLE` (true/false)
- `ENVID_PUNCTUATION_MODEL` (default `kredor/punctuate-all`)
- `ENVID_PROFANITY_WORDS` (comma-separated)
- `ENVID_WHISPERX_BATCH_ROOT` (required for `/batch`)

## Docker
Build:
- `docker build -t whisperx-service -f code/whisperxService/Dockerfile .`

Run:
- `docker run --gpus all -p 5088:5088 whisperx-service`
