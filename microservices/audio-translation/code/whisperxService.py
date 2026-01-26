from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from whisperx_adapter import transcribe, transcribe_stream
from writers import _format_timestamp

LOGGER = logging.getLogger("audio-translation")

app = FastAPI(title="WhisperX Service", version="1.0")


def _bool_param(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _profanity_list() -> list[str]:
    raw = (os.getenv("ENVID_PROFANITY_WORDS") or "").strip()
    if raw:
        words = [w.strip().lower() for w in raw.split(",") if w.strip()]
    else:
        words = ["fuck", "shit", "bitch", "asshole"]
    return words


def _filter_profanity(text: str) -> str:
    words = _profanity_list()
    if not words:
        return text
    pattern = re.compile(r"\\b(" + "|".join(map(re.escape, words)) + r")\\b", re.IGNORECASE)
    return pattern.sub(lambda m: "*" * len(m.group(0)), text)


def _maybe_punctuate(text: str) -> str:
    if not text:
        return text
    if not _bool_param(os.getenv("ENVID_PUNCTUATION_ENABLE"), default=True):
        return text
    try:
        from deepmultilingualpunctuation import PunctuationModel  # type: ignore
    except Exception:
        return text
    model_name = (os.getenv("ENVID_PUNCTUATION_MODEL") or "kredor/punctuate-all").strip()
    try:
        model = PunctuationModel(model_name)
        return model.restore_punctuation(text)
    except Exception:
        return text


def _segments_to_srt(segments: Iterable[dict[str, Any]], start_index: int = 1) -> str:
    lines: list[str] = []
    index = start_index
    for seg in segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        start = _format_timestamp(float(seg.get("start") or 0.0), decimal=",")
        end = _format_timestamp(float(seg.get("end") or 0.0), decimal=",")
        lines.extend([str(index), f"{start} --> {end}", text, ""])
        index += 1
    return "\n".join(lines).strip() + "\n"


def _segments_to_vtt(segments: Iterable[dict[str, Any]]) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        start = _format_timestamp(float(seg.get("start") or 0.0), decimal=".")
        end = _format_timestamp(float(seg.get("end") or 0.0), decimal=".")
        lines.extend([f"{start} --> {end}", text, ""])
    return "\n".join(lines).strip() + "\n"


def _apply_post_processing(result: dict[str, Any]) -> dict[str, Any]:
    segments = result.get("segments") or []
    if not isinstance(segments, list):
        return result
    processed_segments: list[dict[str, Any]] = []
    for seg in segments:
        text = _maybe_punctuate(str(seg.get("text") or "").strip())
        text = _filter_profanity(text)
        seg = dict(seg)
        seg["text"] = text
        processed_segments.append(seg)
    result["segments"] = processed_segments
    return result


def _rtfx_and_memory(start_t: float, audio_seconds: float) -> dict[str, Any]:
    elapsed = max(1e-6, time.perf_counter() - start_t)
    rtfx = elapsed / max(1e-6, audio_seconds)
    mem = None
    try:
        import resource

        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        pass
    return {"elapsed_sec": elapsed, "audio_sec": audio_seconds, "rtfx": rtfx, "max_rss_kb": mem}


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "healthy"})


@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    stream: str | None = Form(default=None),
    language: str | None = Form(default=None),
    model: str | None = Form(default=None),
    batch_size: int | None = Form(default=None),
    chunk_seconds: int | None = Form(default=None),
    vad: str | None = Form(default=None),
    diarize: str | None = Form(default=None),
    min_speech_dur: float | None = Form(default=None),
) -> Any:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix or ".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        input_path = Path(tmp.name)

    device_env = (os.getenv("ENVID_WHISPERX_DEVICE") or "auto").strip().lower()
    if device_env in {"", "auto"}:
        device = "cuda"
    elif device_env in {"cuda", "gpu"}:
        device = "cuda"
    else:
        device = "cpu"

    compute_type = (os.getenv("ENVID_WHISPERX_COMPUTE_TYPE") or "").strip() or None

    try:
        if _bool_param(stream, default=False):
            async def _event_stream() -> Iterable[str]:
                start_t = time.perf_counter()
                all_segments: list[dict[str, Any]] = []
                audio_seconds = 0.0
                for partial in transcribe_stream(
                    str(input_path),
                    device=device,
                    compute_type=compute_type,
                    model_size="large-v2",
                    batch_size=batch_size or 32,
                    language=language,
                    vad=_bool_param(vad, default=True),
                    min_speech_dur=min_speech_dur,
                    chunk_seconds=chunk_seconds or 30,
                    diarize=_bool_param(diarize, default=False),
                ):
                    partial = _apply_post_processing(partial)
                    chunk_segments = partial.get("segments") or []
                    all_segments.extend(chunk_segments)
                    audio_seconds += float(partial.get("duration") or 0.0)
                    payload = {
                        "segments": chunk_segments,
                        "srt": _segments_to_srt(chunk_segments, start_index=len(all_segments) - len(chunk_segments) + 1),
                        "vtt": _segments_to_vtt(chunk_segments),
                        "progress": {
                            "audio_seconds": audio_seconds,
                            "offset": partial.get("offset") or 0.0,
                        },
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                metrics = _rtfx_and_memory(start_t, audio_seconds)
                complete_payload = {"event": "complete", "metrics": metrics}
                yield f"data: {json.dumps(complete_payload, ensure_ascii=False)}\n\n"

            return StreamingResponse(_event_stream(), media_type="text/event-stream")

        start_t = time.perf_counter()
        result = transcribe(
            str(input_path),
            device=device,
            compute_type=compute_type,
            model_size="large-v2",
            batch_size=batch_size or 32,
            language=language,
            vad=_bool_param(vad, default=True),
            min_speech_dur=min_speech_dur,
            diarize=_bool_param(diarize, default=False),
            chunk_seconds=chunk_seconds or 3600,
        )
        if isinstance(result, list):
            if not result:
                raise HTTPException(status_code=500, detail="No transcription output")
            result = result[0]
        result = _apply_post_processing(result)
        segments = result.get("segments") or []
        audio_seconds = 0.0
        if isinstance(segments, list) and segments:
            audio_seconds = max(float(seg.get("end") or 0.0) for seg in segments)
        metrics = _rtfx_and_memory(start_t, max(1e-6, audio_seconds))
        response = {
            "result": result,
            "srt": _segments_to_srt(segments),
            "vtt": _segments_to_vtt(segments),
            "metrics": metrics,
        }
        return JSONResponse(response)
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/batch")
def batch_endpoint(
    input_dir: str = Form(...),
    output_dir: str = Form(...),
    concurrency: int = Form(default=2),
) -> Any:
    allow_root = (os.getenv("ENVID_WHISPERX_BATCH_ROOT") or "").strip()
    if not allow_root:
        raise HTTPException(status_code=400, detail="Batch root not configured")
    if not Path(input_dir).resolve().as_posix().startswith(Path(allow_root).resolve().as_posix()):
        raise HTTPException(status_code=403, detail="Input dir outside allowed root")

    from whisperx_batch import run_batch

    summary = run_batch(Path(input_dir), Path(output_dir), concurrency=concurrency)
    return JSONResponse(summary)


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT") or 5088))
