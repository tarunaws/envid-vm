from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

from openai_whisper_adapter import transcribe

LOGGER = logging.getLogger("openai-whisper-acceptance")


def _get_env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value else None


def _validate_timestamps(result: dict[str, Any]) -> None:
    last_end = 0.0
    for seg in result.get("segments", []) or []:
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or 0.0)
        if start < 0 or end < 0:
            raise AssertionError("Negative timestamps detected.")
        if end < start:
            raise AssertionError("Segment end precedes start.")
        if start < last_end:
            raise AssertionError("Segments are not monotonic.")
        last_end = end
        for word in seg.get("words", []) or []:
            w_start = float(word.get("start") or 0.0)
            w_end = float(word.get("end") or 0.0)
            if w_end < w_start:
                raise AssertionError("Word end precedes start.")
            if w_start < start or w_end > end:
                raise AssertionError("Word timestamps outside segment range.")


def _require_language_detected(result: dict[str, Any]) -> None:
    language = (result.get("language") or "").strip()
    if not language:
        raise AssertionError("Language not detected.")


def _check_chunking(result: dict[str, Any], chunk_seconds: int) -> None:
    if not result.get("segments"):
        raise AssertionError("No segments produced.")
    max_end = max(float(seg.get("end") or 0.0) for seg in result.get("segments", []))
    if max_end <= float(chunk_seconds):
        raise AssertionError("Chunking did not produce offset segments.")


def _run_transcribe_case(
    *,
    label: str,
    audio: str,
    device: str,
    compute_type: str,
    batch_size: int,
    model_size: str,
    vad: bool,
    min_speech_dur: float | None,
    chunk_seconds: int,
    language: str | None,
) -> dict[str, Any]:
    LOGGER.info(
        "Running %s: device=%s compute_type=%s model=%s batch_size=%s",
        label,
        device,
        compute_type,
        model_size,
        batch_size,
    )
    result = transcribe(
        audio,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        model_size=model_size,
        vad=vad,
        min_speech_dur=min_speech_dur,
        chunk_seconds=chunk_seconds,
        language=language,
    )
    if not isinstance(result, dict):
        raise AssertionError("Expected single result dictionary.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAI Whisper acceptance tests")
    parser.add_argument("--audio", default=_get_env("ENVID_OPENAI_WHISPER_TEST_AUDIO"))
    parser.add_argument("--audio-long", default=_get_env("ENVID_OPENAI_WHISPER_TEST_AUDIO_LONG"))
    parser.add_argument("--model", default=_get_env("ENVID_OPENAI_WHISPER_MODEL") or "large-v3")
    parser.add_argument("--language", default=_get_env("ENVID_OPENAI_WHISPER_TEST_LANGUAGE"))
    parser.add_argument("--run-cpu", action="store_true")
    parser.add_argument("--run-cuda", action="store_true")
    parser.add_argument("--vad", action="store_true", default=True)
    parser.add_argument("--min-speech-dur", type=float, default=0.1)
    parser.add_argument("--chunk-seconds", type=int, default=15)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.audio:
        LOGGER.error("Provide --audio or set ENVID_OPENAI_WHISPER_TEST_AUDIO.")
        return 2

    run_cpu = args.run_cpu or not args.run_cuda
    run_cuda = args.run_cuda

    try:
        if run_cpu:
            result = _run_transcribe_case(
                label="cpu",
                audio=args.audio,
                device="cpu",
                compute_type="float32",
                batch_size=8,
                model_size=args.model,
                vad=args.vad,
                min_speech_dur=args.min_speech_dur,
                chunk_seconds=3600,
                language=args.language,
            )
            _validate_timestamps(result)
            if not args.language:
                _require_language_detected(result)

        if run_cuda:
            result = _run_transcribe_case(
                label="cuda",
                audio=args.audio,
                device="cuda",
                compute_type="float16",
                batch_size=32,
                model_size=args.model,
                vad=args.vad,
                min_speech_dur=args.min_speech_dur,
                chunk_seconds=3600,
                language=args.language,
            )
            _validate_timestamps(result)
            if not args.language:
                _require_language_detected(result)

        if args.audio_long:
            result = _run_transcribe_case(
                label="chunking",
                audio=args.audio_long,
                device="cpu",
                compute_type="float32",
                batch_size=8,
                model_size=args.model,
                vad=args.vad,
                min_speech_dur=args.min_speech_dur,
                chunk_seconds=args.chunk_seconds,
                language=args.language,
            )
            _validate_timestamps(result)
            _check_chunking(result, args.chunk_seconds)

        if args.diarization:
            hf_token = _get_env("HUGGINGFACE_TOKEN") or _get_env("HF_TOKEN")
            if hf_token:
                _expect_diarization_success(args.audio, args.model, "cpu", "float32", hf_token)
            else:
                _expect_diarization_error(args.audio, args.model, "cpu", "float32")

    except AssertionError as exc:
        LOGGER.error("Acceptance test failed: %s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Acceptance test error: %s", exc)
        return 2

    LOGGER.info("All acceptance checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
