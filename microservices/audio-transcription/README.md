# WhisperX Service

## Overview
FastAPI wrapper for WhisperX with GPU-first defaults. Supports profanity filtering, punctuation/truecasing, and profiling logs.

## Endpoints
- `GET /health`
- `POST /transcribe` (multipart file)

## /transcribe
Form fields:
- `file` (required)
- `language` (optional)
- `model` (default `large-v3`)
- `chunk_seconds` (default 3600)

## Environment
- `ENVID_PUNCTUATION_ENABLE` (true/false)
- `ENVID_PUNCTUATION_MODEL` (default `kredor/punctuate-all`)
- `ENVID_PROFANITY_WORDS` (comma-separated)
- `ENVID_WHISPERX_MODEL` (default `large-v3`)
- `ENVID_WHISPERX_DEVICE` (`auto`, `cuda`, or `cpu`; defaults to GPU if available)
- `ENVID_WHISPERX_COMPUTE_TYPE` (e.g. `float16` or `float32`)

## Docker
Build:
- `docker build -t audio-transcription -f microservices/audio-transcription/Dockerfile .`

Run:
- `docker run --gpus all -p 5088:5088 audio-transcription`
