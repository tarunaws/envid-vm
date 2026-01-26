from __future__ import annotations

import logging
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

try:
    import whisperx  # type: ignore
except Exception as exc:  # pragma: no cover
    whisperx = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

LOGGER = logging.getLogger("whisperx")


@dataclass
class TranscribeOptions:
    batch_size: int = 16
    language: str | None = None
    diarize: bool = False
    hf_token: str | None = None
    model_size: str = "large-v2"
    compute_type: str | None = None
    min_speech_dur: float | None = None
    vad: bool = False
    device: str | None = None
    chunk_seconds: int = 3600
    seed: int = 1234


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _require_whisperx() -> None:
    if whisperx is None:
        raise RuntimeError(
            "WhisperX is not installed. Install with `pip install whisperx` and ensure its dependencies are present."
        ) from _IMPORT_ERROR


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg and ensure it is discoverable.")


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    LOGGER.warning("CUDA not available; falling back to CPU.")
    return "cpu"


def _hf_token_from_env() -> str | None:
    return os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")


def _vad_method_from_env() -> str | None:
    method = os.getenv("ENVID_WHISPERX_VAD_METHOD")
    return method.strip() if method else None


def _to_temp_path(data: bytes, suffix: str = ".wav") -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _download_url(url: str) -> Path:
    import urllib.request

    with urllib.request.urlopen(url) as response:
        data = response.read()
    return _to_temp_path(data, suffix=Path(url).suffix or ".wav")


def _normalize_inputs(inputs: Iterable[str | bytes | os.PathLike]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if isinstance(item, (str, os.PathLike)):
            text = str(item)
            if text.startswith("http://") or text.startswith("https://"):
                paths.append(_download_url(text))
            else:
                paths.append(Path(text))
        elif isinstance(item, (bytes, bytearray)):
            paths.append(_to_temp_path(bytes(item)))
        else:
            raise TypeError(f"Unsupported input type: {type(item)}")
    return paths


def _load_audio(path: Path) -> np.ndarray:
    _require_whisperx()
    return whisperx.load_audio(str(path))


def _chunk_audio(audio: np.ndarray, chunk_seconds: int, sample_rate: int = 16000) -> list[np.ndarray]:
    total_seconds = len(audio) / float(sample_rate)
    if total_seconds <= chunk_seconds:
        return [audio]
    chunks: list[np.ndarray] = []
    chunk_len = int(chunk_seconds * sample_rate)
    for start in range(0, len(audio), chunk_len):
        chunks.append(audio[start : start + chunk_len])
    return chunks


def _vad_parameters(min_speech_dur: float | None) -> dict[str, Any]:
    if min_speech_dur is None:
        return {}
    return {"min_speech_duration_ms": int(min_speech_dur * 1000)}


def _run_transcribe_on_audio(
    model: Any,
    audio: np.ndarray,
    *,
    batch_size: int,
    language: str | None,
    vad: bool,
    min_speech_dur: float | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"batch_size": batch_size, "language": language}
    if vad:
        kwargs["vad_filter"] = True
        vad_params = _vad_parameters(min_speech_dur)
        if vad_params:
            kwargs["vad_parameters"] = vad_params
    try:
        return model.transcribe(audio, **kwargs)
    except TypeError:
        kwargs.pop("vad_filter", None)
        kwargs.pop("vad_parameters", None)
        return model.transcribe(audio, **kwargs)


def _align(
    segments: list[dict[str, Any]],
    audio: np.ndarray,
    language: str,
    device: str,
) -> dict[str, Any]:
    _require_whisperx()
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    return whisperx.align(segments, model_a, metadata, audio, device, return_char_alignments=False)


def _diarize(audio: np.ndarray, device: str, hf_token: str) -> Any:
    _require_whisperx()
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    return diarize_model(audio)


def _merge_speakers(aligned: dict[str, Any], diarize_segments: Any) -> dict[str, Any]:
    _require_whisperx()
    return whisperx.assign_word_speakers(diarize_segments, aligned)


def _segments_to_api(result: dict[str, Any]) -> list[dict[str, Any]]:
    api_segments: list[dict[str, Any]] = []
    for seg in result.get("segments", []) or []:
        words = []
        for w in seg.get("words", []) or []:
            words.append(
                {
                    "start": float(w.get("start") or 0.0),
                    "end": float(w.get("end") or 0.0),
                    "word": str(w.get("word") or "").strip(),
                    "speaker": w.get("speaker"),
                    "confidence": w.get("confidence", w.get("score")),
                }
            )
        api_segments.append(
            {
                "start": float(seg.get("start") or 0.0),
                "end": float(seg.get("end") or 0.0),
                "text": str(seg.get("text") or "").strip(),
                "words": words,
                "speaker": seg.get("speaker"),
                "confidence": seg.get("confidence", seg.get("avg_logprob")),
            }
        )
    return api_segments


def transcribe_stream(
    inputs: str | bytes | os.PathLike,
    **kwargs: Any,
) -> Iterable[dict[str, Any]]:
    _require_whisperx()
    _require_ffmpeg()

    opts = TranscribeOptions(**kwargs)
    _set_deterministic(opts.seed)
    device = _resolve_device(opts.device)
    compute_type = opts.compute_type or ("float16" if device == "cuda" else "float32")
    vad_method = _vad_method_from_env()
    if vad_method:
        LOGGER.info("WhisperX VAD method=%s", vad_method)
    LOGGER.info("WhisperX device=%s compute_type=%s model=%s batch_size=%s", device, compute_type, opts.model_size, opts.batch_size)

    if opts.diarize:
        token = opts.hf_token or _hf_token_from_env()
        if not token:
            raise RuntimeError("Diarization requested but no HF token provided. Set --hf-token or HUGGINGFACE_TOKEN.")
        opts.hf_token = token

    model = whisperx.load_model(
        opts.model_size,
        device,
        compute_type=compute_type,
        vad_method=vad_method,
    )

    input_paths = _normalize_inputs([inputs])
    detected_language = opts.language
    diarization_enabled = bool(opts.diarize)

    for path in input_paths:
        audio = _load_audio(path)
        chunks = _chunk_audio(audio, opts.chunk_seconds)
        offset = 0.0
        for chunk in chunks:
            result = _run_transcribe_on_audio(
                model,
                chunk,
                batch_size=opts.batch_size,
                language=opts.language,
                vad=opts.vad,
                min_speech_dur=opts.min_speech_dur,
            )
            language = result.get("language") or opts.language or ""
            if not opts.language and language:
                LOGGER.info("WhisperX detected language=%s", language)
            if not detected_language:
                detected_language = language

            aligned = _align(result.get("segments", []) or [], chunk, language, device)
            aligned_result = aligned

            if opts.diarize and opts.hf_token:
                diarize_segments = _diarize(chunk, device, opts.hf_token)
                aligned_result = _merge_speakers(aligned, diarize_segments)

            segments = _segments_to_api(aligned_result)
            for seg in segments:
                seg["start"] += offset
                seg["end"] += offset
                for word in seg.get("words", []) or []:
                    word["start"] += offset
                    word["end"] += offset
            yield {
                "language": detected_language or "",
                "segments": segments,
                "diarization": diarization_enabled,
                "offset": offset,
                "duration": len(chunk) / 16000.0,
            }
            offset += len(chunk) / 16000.0


def transcribe(inputs: str | bytes | os.PathLike | Iterable[str | bytes | os.PathLike], **kwargs: Any) -> dict[str, Any] | list[dict[str, Any]]:
    _require_whisperx()
    _require_ffmpeg()

    opts = TranscribeOptions(**kwargs)
    _set_deterministic(opts.seed)
    device = _resolve_device(opts.device)
    compute_type = opts.compute_type or ("float16" if device == "cuda" else "float32")
    vad_method = _vad_method_from_env()
    if vad_method:
        LOGGER.info("WhisperX VAD method=%s", vad_method)
    LOGGER.info("WhisperX device=%s compute_type=%s model=%s batch_size=%s", device, compute_type, opts.model_size, opts.batch_size)

    if opts.diarize:
        token = opts.hf_token or _hf_token_from_env()
        if not token:
            raise RuntimeError("Diarization requested but no HF token provided. Set --hf-token or HUGGINGFACE_TOKEN.")
        opts.hf_token = token

    model = whisperx.load_model(
        opts.model_size,
        device,
        compute_type=compute_type,
        vad_method=vad_method,
    )

    if isinstance(inputs, (str, bytes, os.PathLike)):
        input_items = [inputs]
        single = True
    else:
        input_items = list(inputs)
        single = len(input_items) == 1

    results: list[dict[str, Any]] = []
    for item in input_items:
        input_paths = _normalize_inputs([item])
        combined_segments: list[dict[str, Any]] = []
        detected_language = opts.language
        diarization_enabled = bool(opts.diarize)

        for path in input_paths:
            audio = _load_audio(path)
            chunks = _chunk_audio(audio, opts.chunk_seconds)
            offset = 0.0
            for chunk in chunks:
                result = _run_transcribe_on_audio(
                    model,
                    chunk,
                    batch_size=opts.batch_size,
                    language=opts.language,
                    vad=opts.vad,
                    min_speech_dur=opts.min_speech_dur,
                )
                language = result.get("language") or opts.language or ""
                if not opts.language and language:
                    LOGGER.info("WhisperX detected language=%s", language)
                if not detected_language:
                    detected_language = language

                aligned = _align(result.get("segments", []) or [], chunk, language, device)
                aligned_result = aligned

                if opts.diarize and opts.hf_token:
                    diarize_segments = _diarize(chunk, device, opts.hf_token)
                    aligned_result = _merge_speakers(aligned, diarize_segments)

                segments = _segments_to_api(aligned_result)
                for seg in segments:
                    seg["start"] += offset
                    seg["end"] += offset
                    for word in seg.get("words", []) or []:
                        word["start"] += offset
                        word["end"] += offset
                combined_segments.extend(segments)
                offset += len(chunk) / 16000.0

        results.append(
            {
                "language": detected_language or "",
                "segments": combined_segments,
                "diarization": diarization_enabled,
            }
        )

    return results[0] if single else results
