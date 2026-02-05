from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch

try:  # Ensure compatibility with newer faster-whisper signatures.
    from faster_whisper import transcribe as _fw_transcribe  # type: ignore

    class _PatchedTranscriptionOptions(_fw_transcribe.TranscriptionOptions):  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("multilingual", True)
            kwargs.setdefault("max_new_tokens", None)
            kwargs.setdefault("clip_timestamps", "0")
            kwargs.setdefault("hallucination_silence_threshold", None)
            kwargs.setdefault("hotwords", None)
            super().__init__(*args, **kwargs)

    _fw_transcribe.TranscriptionOptions = _PatchedTranscriptionOptions  # type: ignore
except Exception:
    pass

try:
    import whisperx  # type: ignore
except Exception as exc:  # pragma: no cover
    whisperx = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

LOGGER = logging.getLogger("whisperx")


@dataclass
class WhisperXOptions:
    model_size: str = "large-v3"
    language: str | None = None
    device: str | None = None
    compute_type: str | None = None
    chunk_seconds: int | None = None


def _require_whisperx() -> None:
    if whisperx is None:
        raise RuntimeError("whisperx is not installed") from _IMPORT_ERROR


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _normalize_model_name(name: str) -> str:
    cleaned = name.strip().lower()
    if cleaned in {"large3", "large-3", "large_v3"}:
        return "large-v3"
    return name.strip()


def _resolve_device(device: str | None) -> str:
    resolved = (device or "auto").strip().lower()
    if resolved in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if resolved in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but not available; retrying in 2 seconds.")
            time.sleep(2)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
        return "cuda"
    return "cpu"


def _resolve_compute_type(device: str, requested: str | None) -> str:
    raw = (requested or "").strip().lower()
    if raw:
        return raw
    return "float16" if device == "cuda" else "float32"


def _load_audio(path: str) -> Any:
    try:
        import soundfile as sf  # type: ignore

        audio, sample_rate = sf.read(path)
        if hasattr(audio, "ndim") and audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            try:
                import torchaudio  # type: ignore

                audio_tensor = torch.from_numpy(audio).float()
                audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, 16000)
                audio = audio_tensor.numpy()
            except Exception as exc:
                raise RuntimeError(f"Failed to resample audio to 16000 Hz: {exc}") from exc
        return audio.astype("float32")
    except Exception:
        return whisperx.load_audio(path)


def _segments_to_api(segments: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    output: list[dict[str, Any]] = []
    diarization = False
    for seg in segments or []:
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        words_payload: list[dict[str, Any]] = []
        words = seg.get("words") or []
        if isinstance(words, list):
            for word in words:
                if not isinstance(word, dict):
                    continue
                words_payload.append(
                    {
                        "word": str(word.get("word") or "").strip(),
                        "start": float(word.get("start") or 0.0),
                        "end": float(word.get("end") or 0.0),
                        "confidence": word.get("score") if word.get("score") is not None else word.get("confidence"),
                    }
                )
        speaker = seg.get("speaker")
        if speaker is not None:
            diarization = True
        output.append(
            {
                "start": start,
                "end": end,
                "text": str(seg.get("text") or "").strip(),
                "words": words_payload,
                "speaker": speaker,
                "confidence": seg.get("avg_logprob"),
            }
        )
    return output, diarization


def transcribe(*args: Any, **kwargs: Any) -> dict[str, Any]:
    _require_whisperx()
    if "input_path" in kwargs:
        input_path = str(kwargs.pop("input_path"))
    elif args:
        input_path = str(args[0])
    else:
        raise ValueError("input_path is required")

    opts = WhisperXOptions(**kwargs)
    device = _resolve_device(opts.device)
    compute_type = _resolve_compute_type(device, opts.compute_type)
    model_name = _normalize_model_name(opts.model_size)

    if device == "cuda":
        try:
            fraction_raw = (os.getenv("ENVID_TRANSCRIBE_GPU_MEMORY_FRACTION") or "").strip()
            fraction = float(fraction_raw) if fraction_raw else 0.5
            if 0.05 < fraction <= 1.0:
                torch.cuda.set_per_process_memory_fraction(fraction)
        except Exception:
            pass

    batch_size = _env_int("ENVID_WHISPERX_BATCH_SIZE", 16)
    align_enabled = _env_bool("ENVID_WHISPERX_ALIGN", True)

    LOGGER.warning("whisperx runtime device=%s model=%s compute_type=%s", device, model_name, compute_type)
    audio = _load_audio(input_path)
    try:
        model = whisperx.load_model(model_name, device=device, compute_type=compute_type)
        result = model.transcribe(audio, batch_size=batch_size, language=opts.language or None)
    except Exception as exc:
        LOGGER.warning("whisperx VAD load failed, falling back to faster-whisper: %s", exc)
        try:
            from faster_whisper import WhisperModel  # type: ignore

            fallback_devices = [(device, compute_type)]
            if device == "cuda":
                fallback_devices.append(("cpu", "float32"))

            last_exc: Exception | None = None
            for fw_device, fw_compute in fallback_devices:
                try:
                    LOGGER.warning(
                        "whisperx faster-whisper fallback device=%s compute_type=%s", fw_device, fw_compute
                    )
                    fw_model = WhisperModel(model_name, device=fw_device, compute_type=fw_compute)
                    fw_segments, fw_info = fw_model.transcribe(
                        input_path,
                        language=opts.language or None,
                        word_timestamps=True,
                        vad_filter=False,
                    )
                    language = (getattr(fw_info, "language", None) or opts.language or "").strip()
                    segments = []
                    for seg in fw_segments:
                        words_payload = []
                        for word in getattr(seg, "words", None) or []:
                            words_payload.append(
                                {
                                    "word": str(getattr(word, "word", "") or "").strip(),
                                    "start": float(getattr(word, "start", 0.0) or 0.0),
                                    "end": float(getattr(word, "end", 0.0) or 0.0),
                                    "score": getattr(word, "probability", None),
                                }
                            )
                        segments.append(
                            {
                                "start": float(getattr(seg, "start", 0.0) or 0.0),
                                "end": float(getattr(seg, "end", 0.0) or 0.0),
                                "text": str(getattr(seg, "text", "") or "").strip(),
                                "words": words_payload,
                                "avg_logprob": getattr(seg, "avg_logprob", None),
                            }
                        )
                    result = {"language": language, "segments": segments}
                    break
                except Exception as inner_exc:
                    last_exc = inner_exc
                    msg = str(inner_exc).lower()
                    if "out of memory" not in msg and "cuda" not in msg:
                        raise
            else:
                if last_exc is None:
                    raise RuntimeError("WhisperX fallback failed without error")
                raise last_exc
        except Exception as fw_exc:
            raise RuntimeError(f"WhisperX VAD failed and faster-whisper fallback failed: {fw_exc}") from fw_exc

    language = (result.get("language") or opts.language or "").strip()
    segments = result.get("segments") or []
    if align_enabled and language:
        try:
            align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
            aligned = whisperx.align(segments, align_model, metadata, audio, device)
            segments = aligned.get("segments") or segments
        except Exception as exc:
            LOGGER.warning("whisperx alignment failed: %s", exc)

    api_segments, diarization = _segments_to_api(segments)
    return {
        "language": language,
        "segments": api_segments,
        "diarization": diarization,
    }


def transcribe_stream(*args: Any, **kwargs: Any) -> Iterable[dict[str, Any]]:
    raise RuntimeError("Streaming transcription is not supported for WhisperX")
