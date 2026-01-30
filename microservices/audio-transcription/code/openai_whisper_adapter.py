from __future__ import annotations

import logging
import os
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
    if device:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return device
    if torch.cuda.is_available():
        return "cuda"
    LOGGER.warning("CUDA not available; falling back to CPU.")
    return "cpu"


def _normalize_model_name(name: str) -> str:
    cleaned = name.strip().lower()
    if cleaned in {"large3", "large-3", "large_v3"}:
        return "large-v3"
    return name.strip()


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


def transcribe(*, input_path: str, **kwargs: Any) -> dict[str, Any]:
    _require_whisper()
    opts = WhisperOptions(**kwargs)
    device = _resolve_device(opts.device)
    model_name = _normalize_model_name(opts.model_size)
    fp16 = device == "cuda" and (opts.compute_type or "float16") != "float32"

    LOGGER.info("openai-whisper device=%s model=%s fp16=%s", device, model_name, fp16)
    model = whisper.load_model(model_name, device=device)

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
        result = model.transcribe(
            chunk,
            language=opts.language or None,
            fp16=fp16,
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