from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import requests
import torch

from openai_whisper_adapter import transcribe as openai_transcribe
from writers import _format_timestamp

LOGGER = logging.getLogger("audio-transcription")

app = FastAPI(title="OpenAI Whisper Service", version="1.0")

_PUNCTUATION_MODEL = None
_PUNCTUATION_MODEL_NAME = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _bool_param(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _openai_device() -> str:
    device_env = (os.getenv("ENVID_OPENAI_WHISPER_DEVICE") or "auto").strip().lower()
    if device_env in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_env in {"cuda", "gpu"}:
        return "cuda"
    return "cpu"


def _ffmpeg_service_url() -> str:
    return "http://transcoder:5091"


def _extract_audio_via_transcoder(*, filename: str, content: bytes, fmt: str = "wav") -> Path:
    service_url = _ffmpeg_service_url()
    files = {"video": (filename, content, "application/octet-stream")}
    data = {
        "format": fmt,
        "sample_rate": "16000",
        "channels": "1",
    }
    resp = requests.post(f"{service_url}/extract-audio", files=files, data=data, timeout=900)
    if resp.status_code >= 400:
        raise RuntimeError(f"transcoder audio extract failed ({resp.status_code}): {resp.text}")
    suffix = ".flac" if fmt == "flac" else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(resp.content)
        tmp.flush()
        return Path(tmp.name)


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
    if not _bool_param(os.getenv("ENVID_PUNCTUATION_ENABLED"), default=True):
        return text
    if not text:
        return text
    try:
        from deepmultilingualpunctuation import PunctuationModel  # type: ignore
    except Exception:
        return text


def _language_code(language: str | None) -> str:
    if not language:
        return "en-US"
    lang = language.strip().lower()
    if lang.startswith("hi"):
        return "hi-IN"
    if lang.startswith("en"):
        return "en-US"
    return lang


@lru_cache(maxsize=4)
def _language_tool(language: str) -> Any:
    try:
        import language_tool_python  # type: ignore
    except Exception:
        return None
    url = (os.getenv("ENVID_LANGUAGETOOL_URL") or "").strip()
    try:
        if url:
            return language_tool_python.LanguageTool(language, url=url)
        return language_tool_python.LanguageToolPublicAPI(language)
    except Exception:
        return None


def _grammar_correct(text: str, language: str | None) -> tuple[str, bool]:
    tool = _language_tool(_language_code(language))
    if tool is None or not text.strip():
        return text, False
    try:
        corrected = tool.correct(text)
    except Exception:
        return text, False
    changed = corrected.strip() != text.strip()
    return corrected, changed


def _regex_cleanup(text: str, language: str | None) -> tuple[str, bool]:
    original = text
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(\S)", r"\1 \2", text)
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    if language and language.lower().startswith("hi"):
        text = re.sub(r"([\u0900-\u097F])\s*\.\s*", r"\1। ", text)
        text = re.sub(r"\s+।", "।", text)
        text = re.sub(r"।([\u0900-\u097F])", r"। \1", text)
    return text, text != original


def _hindi_confusion_map() -> dict[str, str]:
    return {
        "हैँ": "हैं",
        "हैंं": "हैं",
        "है" : "है",
        "नही": "नहीं",
        "क्यु": "क्यू",
        "क्यूँ": "क्यों",
        "क्यो": "क्यों",
        "किया": "किया",
        "होगाा": "होगा",
        "सकताा": "सकता",
        "होगीे": "होगी",
        "कहा़": "कहा",
        "कहां": "कहाँ",
        "गयाा": "गया",
    }


def _hindi_dictionary_correct(text: str, language: str | None) -> tuple[str, bool]:
    if not language or not language.lower().startswith("hi"):
        return text, False
    try:
        from rapidfuzz import process as rf_process  # type: ignore
        from rapidfuzz import fuzz as rf_fuzz  # type: ignore
        from wordfreq import zipf_frequency  # type: ignore
    except Exception:
        return text, False

    confusion = _hindi_confusion_map()
    if not confusion:
        return text, False
    keys = list(confusion.keys())
    tokens = re.findall(r"\s+|[^\s]+", text)
    changed = False
    out: list[str] = []
    for tok in tokens:
        if tok.isspace():
            out.append(tok)
            continue
        if re.fullmatch(r"[\u0900-\u097F]+", tok):
            if tok in confusion:
                out.append(confusion[tok])
                changed = True
                continue
            freq = zipf_frequency(tok, "hi")
            if freq < 2.0:
                match = rf_process.extractOne(tok, keys, scorer=rf_fuzz.ratio)
                if match and match[1] >= 92:
                    out.append(confusion[match[0]])
                    changed = True
                    continue
        out.append(tok)
    return "".join(out), changed


def _llm_normalize(text: str, language: str | None) -> tuple[str, bool]:
    if not _bool_param(os.getenv("ENVID_TEXT_NORMALIZE_ENABLED"), default=True):
        return text, False
    if not text.strip():
        return text, False
    base_url = (os.getenv("ENVID_TEXT_NORMALIZER_URL") or "http://genai:5099/normalize_transcript").strip()
    if not base_url:
        return text, False
    payload = {
        "text": text,
        "language_code": language or "",
    }
    try:
        resp = requests.post(
            base_url,
            json=payload,
            timeout=int(float(os.getenv("ENVID_TEXT_NORMALIZE_TIMEOUT") or 30)),
        )
        if resp.status_code >= 400:
            return text, False
        data = resp.json()
        normalized = data.get("text") if isinstance(data, dict) else None
        if not isinstance(normalized, str) or not normalized.strip():
            return text, False
        normalized = normalized.strip()
        return normalized, normalized != text.strip()
    except Exception:
        return text, False


def _apply_correction_stack(text: str, language: str | None, *, use_llm: bool) -> tuple[str, list[str]]:
    steps: list[str] = []
    if use_llm:
        text, changed = _llm_normalize(text, language)
        if changed:
            steps.append("llm")
    text, changed = _grammar_correct(text, language)
    if changed:
        steps.append("languagetool")
    text, changed = _hindi_dictionary_correct(text, language)
    if changed:
        steps.append("hindi_dict")
    text, changed = _regex_cleanup(text, language)
    if changed:
        steps.append("regex")
    return text, steps


def _quality_checks(text: str, language: str | None) -> tuple[bool, dict[str, float]]:
    words = re.findall(r"[A-Za-z]+|[\u0900-\u097F]+", text)
    min_words = int(float(os.getenv("ENVID_QC_MIN_WORDS") or 3))
    if len(words) < min_words:
        return False, {"min_words": len(words)}
    stripped = re.sub(r"\s+", "", text)
    total_chars = max(1, len(stripped))
    alpha_chars = sum(1 for ch in stripped if ch.isalpha())
    alpha_ratio = alpha_chars / float(total_chars)
    alpha_thresh = float(os.getenv("ENVID_QC_ALPHA_RATIO") or 0.55)
    if alpha_ratio < alpha_thresh:
        return False, {"alphabetic_ratio": alpha_ratio}
    unique_ratio = len(set(w.lower() for w in words)) / float(max(1, len(words)))
    unique_thresh = float(os.getenv("ENVID_QC_UNIQUE_WORD_RATIO") or 0.3)
    if unique_ratio < unique_thresh:
        return False, {"unique_word_ratio": unique_ratio}
    if language and language.lower().startswith("hi"):
        dev_chars = sum(1 for ch in stripped if "\u0900" <= ch <= "\u097F")
        devanagari_ratio = dev_chars / float(max(1, alpha_chars))
        dev_thresh = float(os.getenv("ENVID_QC_DEVANAGARI_RATIO") or 0.4)
        if devanagari_ratio < dev_thresh:
            return False, {"devanagari_ratio": devanagari_ratio}
    return True, {
        "alphabetic_ratio": alpha_ratio,
        "unique_word_ratio": unique_ratio,
    }
    model_name = (os.getenv("ENVID_PUNCTUATION_MODEL") or "kredor/punctuate-all").strip()
    local_only = _bool_param(os.getenv("ENVID_PUNCTUATION_LOCAL_ONLY"), default=False)
    if local_only and model_name and not Path(model_name).exists():
        return text
    try:
        global _PUNCTUATION_MODEL, _PUNCTUATION_MODEL_NAME
        if _PUNCTUATION_MODEL is None or _PUNCTUATION_MODEL_NAME != model_name:
            _PUNCTUATION_MODEL = PunctuationModel(model_name)
            _PUNCTUATION_MODEL_NAME = model_name
        return _PUNCTUATION_MODEL.restore_punctuation(text)
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


def _apply_post_processing(result: dict[str, Any], language: str | None) -> dict[str, Any]:
    segments = result.get("segments") or []
    if not isinstance(segments, list):
        return result

    use_llm_segments = _bool_param(os.getenv("ENVID_TEXT_NORMALIZE_SEGMENTS"), default=True)
    processed_segments: list[dict[str, Any]] = []
    applied_steps: set[str] = set()
    for seg in segments:
        text = str(seg.get("text") or "").strip()
        text, steps = _apply_correction_stack(text, language, use_llm=use_llm_segments)
        if steps:
            applied_steps.update(steps)
        text = _filter_profanity(text)
        seg = dict(seg)
        seg["text"] = text
        processed_segments.append(seg)

    rebuilt_text = " ".join(s.get("text") or "" for s in processed_segments).strip()
    normalized_text, full_steps = _apply_correction_stack(rebuilt_text, language, use_llm=True)
    if full_steps:
        applied_steps.update(full_steps)

    if not applied_steps:
        raise HTTPException(status_code=422, detail="Transcript rejected: no corrections applied")

    ok, metrics = _quality_checks(rebuilt_text, language)
    if not ok:
        raise HTTPException(status_code=422, detail={"error": "Transcript rejected: quality checks failed", "metrics": metrics})

    result["segments"] = processed_segments
    result["text"] = rebuilt_text
    result["normalized_text"] = normalized_text
    result["corrections"] = sorted(applied_steps)
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
    language: str | None = Form(default=None),
    model: str | None = Form(default=None),
    chunk_seconds: int | None = Form(default=None),
) -> Any:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    content = await file.read()
    fmt = "wav"
    input_path = _extract_audio_via_transcoder(filename=file.filename, content=content, fmt=fmt)

    language_hint = (language or "").strip().lower()
    if language_hint in {"", "auto", "detect", "none"}:
        language_hint = ""

    openai_device = _openai_device()
    openai_compute_type = (os.getenv("ENVID_OPENAI_WHISPER_COMPUTE_TYPE") or "").strip() or None
    openai_model = "large-v3"

    try:
        start_t = time.perf_counter()
        result = openai_transcribe(
            input_path=str(input_path),
            device=openai_device,
            compute_type=openai_compute_type,
            model_size=openai_model,
            language=language_hint or None,
            chunk_seconds=chunk_seconds or 3600,
        )
        result = _apply_post_processing(result, language_hint or result.get("language"))
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


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT") or 5088))
