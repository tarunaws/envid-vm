from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

try:
    import whisper  # type: ignore
except Exception as exc:  # pragma: no cover
    whisper = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

LOGGER = logging.getLogger("openai-whisper")


@dataclass
class WhisperOptions:
    model_size: str = "large-v3"
    language: str | None = None
    device: str | None = None
    compute_type: str | None = None
    chunk_seconds: int | None = None


def _require_whisper() -> None:
    if whisper is None:
        raise RuntimeError("openai-whisper is not installed") from _IMPORT_ERROR


def _resolve_device(device: str | None) -> str:
    resolved = device or "cuda"
    if resolved == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available; retrying in 2 seconds.")
        time.sleep(2)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
    return resolved


def _normalize_model_name(name: str) -> str:
    cleaned = name.strip().lower()
    if cleaned in {"large3", "large-3", "large_v3"}:
        return "large-v3"
    return name.strip()


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _load_audio(path: Path) -> np.ndarray:
    try:
        waveform, sample_rate = torchaudio.load(str(path))
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        return waveform.squeeze(0).numpy()
    except Exception as exc:
        try:
            import soundfile as sf  # type: ignore
        except Exception:
            raise exc

        data, sample_rate = sf.read(str(path))
        if isinstance(data, np.ndarray) and data.ndim > 1:
            data = data.mean(axis=1)
        if sample_rate != 16000:
            waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            return waveform.squeeze(0).numpy()
        return np.asarray(data, dtype=np.float32)


def _chunk_audio(audio: np.ndarray, chunk_seconds: int, sample_rate: int = 16000) -> list[np.ndarray]:
    total_seconds = len(audio) / float(sample_rate)
    if total_seconds <= chunk_seconds:
        return [audio]
    chunks: list[np.ndarray] = []
    chunk_len = int(chunk_seconds * sample_rate)
    for start in range(0, len(audio), chunk_len):
        chunks.append(audio[start : start + chunk_len])
    return chunks


def _segments_to_api(segments: list[dict[str, Any]], offset: float = 0.0) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seg in segments:
        start = float(seg.get("start") or 0.0) + offset
        end = float(seg.get("end") or 0.0) + offset
        output.append(
            {
                "start": start,
                "end": end,
                "text": str(seg.get("text") or "").strip(),
                "words": [],
                "speaker": None,
                "confidence": seg.get("avg_logprob"),
            }
        )
    return output


@lru_cache(maxsize=4)
def _load_model_cached(model_name: str, device: str) -> Any:
    LOGGER.info("openai-whisper loading model=%s device=%s", model_name, device)
    return whisper.load_model(model_name, device=device)


def warmup(*, model_size: str, device: str | None = None) -> None:
    _require_whisper()
    resolved_device = _resolve_device(device)
    model_name = _normalize_model_name(model_size)
    _load_model_cached(model_name, resolved_device)


def transcribe(*, input_path: str, **kwargs: Any) -> dict[str, Any]:
    _require_whisper()
    opts = WhisperOptions(**kwargs)
    device = _resolve_device(opts.device)
    model_name = _normalize_model_name(opts.model_size)
    fp16 = device == "cuda" and (opts.compute_type or "float16") != "float32"

    LOGGER.info("openai-whisper device=%s model=%s fp16=%s", device, model_name, fp16)
    model = _load_model_cached(model_name, device)

    audio = _load_audio(Path(input_path))
    chunk_seconds = opts.chunk_seconds or 0
    if chunk_seconds and chunk_seconds > 0:
        chunks = _chunk_audio(audio, chunk_seconds)
    else:
        chunks = [audio]

    combined_segments: list[dict[str, Any]] = []
    detected_language = opts.language or ""
    offset = 0.0
    for chunk in chunks:
        beam_size = _env_int("ENVID_WHISPER_BEAM_SIZE", 5)
        best_of = _env_int("ENVID_WHISPER_BEST_OF", 5)
        temperature = _env_float("ENVID_WHISPER_TEMPERATURE", 0.0)
        condition_on_previous_text = _env_bool("ENVID_WHISPER_CONDITION_ON_PREVIOUS_TEXT", True)
        no_speech_threshold = _env_float("ENVID_WHISPER_NO_SPEECH_THRESHOLD", 0.6)
        logprob_threshold = _env_float("ENVID_WHISPER_LOGPROB_THRESHOLD", -1.0)
        compression_ratio_threshold = _env_float("ENVID_WHISPER_COMPRESSION_RATIO_THRESHOLD", 2.4)
        result = model.transcribe(
            chunk,
            language=opts.language or None,
            fp16=fp16,
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
            condition_on_previous_text=condition_on_previous_text,
            no_speech_threshold=no_speech_threshold,
            logprob_threshold=logprob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
        )
        language = result.get("language") or ""
        if language and not detected_language:
            detected_language = language
        segments = result.get("segments") or []
        combined_segments.extend(_segments_to_api(segments, offset=offset))
        offset += len(chunk) / 16000.0

    return {
        "language": detected_language,
        "segments": combined_segments,
        "diarization": False,
    }