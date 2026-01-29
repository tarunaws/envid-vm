from __future__ import annotations

import base64
import difflib
import gzip
import io
import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import fcntl
import threading
import time
import uuid
import zipfile
import urllib.request
import urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from flask import Flask, Response, jsonify, redirect, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None

try:
    import language_tool_python  # type: ignore
except Exception:
    language_tool_python = None

try:
    from pymongo import MongoClient, ASCENDING, ReturnDocument  # type: ignore
except Exception:
    MongoClient = None
    ASCENDING = None
    ReturnDocument = None

try:
    from google.cloud import storage as gcs_storage  # type: ignore
except Exception:
    gcs_storage = None

try:
    from google.cloud import speech as gcp_speech  # type: ignore
except Exception:
    gcp_speech = None

try:
    from google.cloud import translate as gcp_translate  # type: ignore
except Exception:
    gcp_translate = None

try:
    from google.cloud import language_v1 as gcp_language  # type: ignore
except Exception:
    gcp_language = None

try:
    from google.cloud import videointelligence as gcp_video_intelligence  # type: ignore
except Exception:
    gcp_video_intelligence = None

try:
    from rapidfuzz import process as rapid_process, fuzz as rapid_fuzz  # type: ignore
except Exception:
    rapid_process = None
    rapid_fuzz = None

try:
    from wordfreq import zipf_frequency, top_n_list  # type: ignore
except Exception:
    zipf_frequency = None
    top_n_list = None

app = Flask(__name__)


class TranscriptVerificationError(RuntimeError):
    pass


def _gcp_project_id() -> str | None:
    return (os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip() or None


def _gcp_location() -> str:
    return (os.getenv("GCP_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1").strip() or "us-central1"


def _gcp_translate_location() -> str:
    return (os.getenv("GCP_TRANSLATE_LOCATION") or "global").strip() or "global"


def _translate_provider() -> str:
    # Translation is always routed through the translate container (LibreTranslate).
    return "libretranslate"


def _translate_targets() -> list[str]:
    raw = (os.getenv("ENVID_METADATA_TRANSLATE_LANGS") or "").strip()
    langs = [seg.strip().lower() for seg in raw.split(",") if seg.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for lang in langs:
        if lang in seen:
            continue
        seen.add(lang)
        out.append(lang)
    return out


def _translate_targets_from_selection(sel: Dict[str, Any]) -> list[str]:
    raw = None
    for key in (
        "translate_targets",
        "translate_target_languages",
        "translate_target_language",
        "target_languages",
        "target_language",
        "translation_targets",
    ):
        if isinstance(sel, dict) and key in sel:
            raw = sel.get(key)
            break
    if raw is None:
        return []
    if isinstance(raw, list):
        langs = [str(x).strip().lower() for x in raw if str(x).strip()]
    else:
        langs = [seg.strip().lower() for seg in str(raw).split(",") if seg.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for lang in langs:
        if lang in seen:
            continue
        seen.add(lang)
        out.append(lang)
    return out


_LIBRE_LANG_CACHE: tuple[set[str], float] = (set(), 0.0)


def _libretranslate_base_url() -> str:
    return "http://translate:5000"


def _libretranslate_translate(
    *,
    text: str,
    source_lang: str | None,
    target_lang: str,
) -> str:
    base_url = _libretranslate_base_url()
    timeout_s = _safe_float(
        os.getenv("ENVID_LIBRETRANSLATE_TIMEOUT_SECONDS")
        or os.getenv("ENVID_TRANSLATE_TIMEOUT_SECONDS")
        or 30,
        30.0,
    )
    payload: Dict[str, Any] = {
        "q": text,
        "source": source_lang or "auto",
        "target": target_lang,
        "format": "text",
    }
    resp = requests.post(base_url.rstrip("/") + "/translate", json=payload, timeout=float(timeout_s))
    if resp.status_code >= 400:
        raise RuntimeError(f"LibreTranslate /translate failed ({resp.status_code}): {resp.text}")
    data = resp.json() if resp.content else {}
    translated = data.get("translatedText") if isinstance(data, dict) else None
    return (str(translated).strip() if translated else text)


def _libretranslate_languages_raw(base_url: str) -> list[dict[str, Any]]:
    url = base_url.rstrip("/") + "/languages"
    resp = requests.get(url, timeout=10)
    if resp.status_code >= 400:
        raise RuntimeError(f"LibreTranslate /languages failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def _libretranslate_supported_langs(base_url: str) -> set[str]:
    global _LIBRE_LANG_CACHE
    cached, ts = _LIBRE_LANG_CACHE
    if cached and (time.time() - ts) < 600:
        return cached
    try:
        langs = _libretranslate_languages_raw(base_url)
        codes = {
            str(item.get("code") or item.get("language") or "").strip()
            for item in langs
            if isinstance(item, dict)
        }
        codes = {c for c in codes if c}
    except Exception:
        return cached
    _LIBRE_LANG_CACHE = (codes, time.time())
    return codes


def _translate_segments(
    *,
    segments: list[dict[str, Any]],
    source_lang: str | None,
    target_lang: str,
    provider: str,
    gcp_client: Any | None,
    gcp_parent: str | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in segments:
        src = source_lang or "auto"
        tgt = target_lang
        text = str(seg.get("text") or "").strip()
        if not text:
            out.append(seg)
            continue

        translated = text
        if provider == "disabled":
            translated = text
        elif provider == "libretranslate":
            translated = _libretranslate_translate(
                text=text[:4500],
                source_lang=(src if src and len(src) >= 2 else None),
                target_lang=tgt,
            )
        elif provider == "gcp_translate":
            if gcp_client is None or gcp_parent is None:
                raise RuntimeError("GCP Translate client not initialized")
            req: Dict[str, Any] = {
                "parent": gcp_parent,
                "contents": [text[:4500]],
                "mime_type": "text/plain",
                "target_language_code": tgt,
            }
            if src and len(src) >= 2:
                req["source_language_code"] = src
            resp = gcp_client.translate_text(request=req)
            if resp and resp.translations:
                translated = (resp.translations[0].translated_text or "").strip() or text

        new_seg = dict(seg)
        new_seg["translated_text"] = translated
        out.append(new_seg)
    return out


def _translate_text(
    *,
    text: str,
    source_lang: str | None,
    target_lang: str,
    provider: str,
    gcp_client: Any | None,
    gcp_parent: str | None,
) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    src = source_lang or "auto"
    tgt = target_lang
    if provider == "disabled":
        return raw
    if provider == "libretranslate":
        return _libretranslate_translate(
            text=raw[:4500],
            source_lang=(src if src and len(src) >= 2 else None),
            target_lang=tgt,
        )
    if provider == "gcp_translate":
        if gcp_client is None or gcp_parent is None:
            raise RuntimeError("GCP Translate client not initialized")
        req: Dict[str, Any] = {
            "parent": gcp_parent,
            "contents": [raw[:4500]],
            "mime_type": "text/plain",
            "target_language_code": tgt,
        }
        if src and len(src) >= 2:
            req["source_language_code"] = src
        resp = gcp_client.translate_text(request=req)
        if resp and resp.translations:
            return (resp.translations[0].translated_text or "").strip() or raw
    return raw


def _normalize_transcript_basic(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    return t.strip()


def _enhance_transcript_punctuation(text: str) -> str:
    """Lightweight, deterministic punctuation/spacing cleanup (language-agnostic, Hindi-friendly)."""

    out = _normalize_transcript_basic(text)
    if not out:
        return ""

    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"\s+([।])", r"\1", out)
    out = re.sub(r"([,.;:!?])([A-Za-z0-9\u0900-\u097F])", r"\1 \2", out)
    out = re.sub(r"(।)([A-Za-z0-9\u0900-\u097F])", r"\1 \2", out)
    out = re.sub(r"([\(\[\{])\s+", r"\1", out)
    out = re.sub(r"\s+([\)\]\}])", r"\1", out)
    out = re.sub(r"([,.;:!?])\s+", r"\1 ", out)
    out = re.sub(r"(।)\s+", r"। ", out)
    out = re.sub(r"([.!?।])\1{1,}", r"\1", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def _ensure_terminal_punctuation(text: str, language_code: str | None) -> str:
    out = (text or "").strip()
    if not out:
        return ""
    if out[-1] in ".!?।":
        return out
    return f"{out}{'।' if _is_hindi_language(language_code) else '.'}"


def _build_transcript_script(
    *,
    transcript: str,
    transcript_segments: list[dict[str, Any]],
    language_code: str | None,
) -> tuple[str, dict[str, Any]]:
    gap_seconds = _safe_float(os.getenv("ENVID_TRANSCRIPT_SCRIPT_GAP_SECONDS"), 1.2)
    max_chars = _parse_int(os.getenv("ENVID_TRANSCRIPT_SCRIPT_MAX_CHARS"), default=180, min_value=80, max_value=1000)

    def _clean(text: str) -> str:
        return _ensure_terminal_punctuation(_enhance_transcript_punctuation(text), language_code)

    lines: list[str] = []
    source = "transcript"
    if transcript_segments:
        source = "segments"
        cur = ""
        last_end: float | None = None
        safe_segments = [s for s in transcript_segments if isinstance(s, dict)]
        for seg in sorted(safe_segments, key=lambda s: float(s.get("start") or 0.0)):
            txt = _clean(str(seg.get("text") or "").strip())
            if not txt:
                continue
            try:
                st = float(seg.get("start") or 0.0)
                en = float(seg.get("end") or st)
            except Exception:
                st = None
                en = None

            gap = (st - last_end) if (st is not None and last_end is not None) else None
            if cur:
                if (gap is not None and gap >= gap_seconds) or (len(cur) + 1 + len(txt) > max_chars):
                    lines.append(cur.strip())
                    cur = txt
                else:
                    cur = f"{cur} {txt}".strip()
            else:
                cur = txt
            if en is not None:
                last_end = en

        if cur:
            lines.append(cur.strip())

    if not lines:
        cleaned = _clean(transcript or "")
        if cleaned:
            sentences = re.split(r"(?<=[.!?।])\s+", cleaned)
            cur = ""
            for sentence in sentences:
                s = sentence.strip()
                if not s:
                    continue
                if not cur:
                    cur = s
                elif len(cur) + 1 + len(s) > max_chars:
                    lines.append(cur)
                    cur = s
                else:
                    cur = f"{cur} {s}"
            if cur:
                lines.append(cur)

    script = "\n".join([ln for ln in lines if ln]).strip()
    meta = {
        "applied": bool(script),
        "source": source,
        "gap_seconds": float(gap_seconds),
        "max_chars": int(max_chars),
    }
    return script, meta


def _transcript_is_reasonable(text: str, language_code: str | None) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False

    words = [w for w in re.split(r"\s+", raw) if w]
    min_words = 6
    if len(words) < min_words:
        return False

    letters = sum(1 for ch in raw if ch.isalpha())
    alpha_ratio = letters / max(1, len(raw))
    min_alpha_ratio = 0.20
    if alpha_ratio < min_alpha_ratio:
        return False

    lang = (language_code or "").strip().lower()
    if lang.startswith("hi"):
        devanagari = sum(1 for ch in raw if "\u0900" <= ch <= "\u097F")
        min_dev_ratio = 0.30
        if devanagari / max(1, letters) < min_dev_ratio:
            return False

    uniq = len({w.lower() for w in words})
    min_unique_ratio = 0.20
    if uniq / max(1, len(words)) < min_unique_ratio:
        return False

    return True


def _apply_segment_corrections(
    *,
    text: str,
    language_code: str | None,
    grammar_enabled: bool,
    hindi_dictionary_enabled: bool,
    punctuation_enabled: bool,
) -> tuple[str, dict[str, Any]]:
    out = (text or "").strip()
    if not out:
        return "", {
            "nlp_applied": False,
            "grammar_applied": False,
            "dictionary_applied": False,
            "hindi_applied": False,
            "punctuation_applied": False,
        }

    meta = {
        "nlp_applied": False,
        "grammar_applied": False,
        "dictionary_applied": False,
        "hindi_applied": False,
        "punctuation_applied": False,
    }

    if grammar_enabled:
        corrected, _ = _languagetool_correct_text(text=out, language=language_code)
        if corrected and corrected != out:
            out = corrected
            meta["grammar_applied"] = True

    if hindi_dictionary_enabled:
        corrected = _dictionary_correct_text(text=out, language_code=language_code)
        if corrected and corrected != out:
            out = corrected
            meta["dictionary_applied"] = True
            if _is_hindi_language(language_code):
                meta["hindi_applied"] = True

    if punctuation_enabled:
        corrected = _enhance_transcript_punctuation(out)
        if corrected and corrected != out:
            out = corrected
            meta["punctuation_applied"] = True

    if punctuation_enabled and not meta["punctuation_applied"]:
        force_terminal = True
        if force_terminal and out:
            tail = out[-1]
            if tail not in ".!?।":
                out = f"{out}{'।' if _is_hindi_language(language_code) else '.'}"
                meta["punctuation_applied"] = True

    use_text_normalizer = _env_truthy(os.getenv("ENVID_TRANSCRIPT_USE_TEXT_NORMALIZER"), default=True)
    if use_text_normalizer and out:
        nlp_mode = (os.getenv("ENVID_TRANSCRIPT_TEXT_NORMALIZER_NLP_MODE") or "openrouter_llama").strip().lower()
        resp = _text_normalizer_normalize_segment(
            text=out,
            language_code=language_code,
            grammar_enabled=bool(grammar_enabled),
            dictionary_enabled=bool(hindi_dictionary_enabled),
            punctuation_enabled=bool(punctuation_enabled),
            nlp_mode=nlp_mode,
        )
        if resp:
            normalized, remote_meta = resp
            if normalized:
                meta.update({
                    "nlp_applied": bool(remote_meta.get("nlp_applied")),
                    "grammar_applied": bool(remote_meta.get("grammar_applied")) or meta.get("grammar_applied", False),
                    "dictionary_applied": bool(remote_meta.get("dictionary_applied")) or meta.get("dictionary_applied", False),
                    "hindi_applied": bool(remote_meta.get("hindi_applied")) or meta.get("hindi_applied", False),
                    "punctuation_applied": bool(remote_meta.get("punctuation_applied")) or meta.get("punctuation_applied", False),
                    "source": str(remote_meta.get("source") or "text-normalizer"),
                })
                return normalized.strip(), meta
        meta["source"] = "fallback"

    return out.strip(), meta


def _normalize_scene_segments(*, scenes: list[dict[str, Any]], duration_seconds: float | None) -> list[dict[str, Any]]:
    if not scenes:
        return []
    dur = float(duration_seconds or 0.0)
    cleaned: list[tuple[float, float]] = []
    for s in scenes:
        try:
            st = float(s.get("start") or 0.0)
            en = float(s.get("end") or st)
        except Exception:
            continue
        if en < st:
            st, en = en, st
        cleaned.append((max(0.0, st), max(0.0, en)))

    if not cleaned:
        return []

    cleaned.sort(key=lambda x: (x[0], x[1]))
    normalized: list[tuple[float, float]] = []
    for st, en in cleaned:
        if not normalized:
            normalized.append((st, en))
            continue
        last_st, last_en = normalized[-1]
        if st <= last_en + 0.05:
            normalized[-1] = (last_st, max(last_en, en))
        else:
            normalized.append((st, en))

    filled: list[tuple[float, float]] = []
    cur = 0.0
    for st, en in normalized:
        if st > cur + 0.05:
            filled.append((cur, st))
        filled.append((st, en))
        cur = max(cur, en)
    if dur > 0 and cur < dur - 0.05:
        filled.append((cur, dur))

    out: list[dict[str, Any]] = []
    min_len = _safe_float(os.getenv("ENVID_SCENE_MIN_SECONDS"), 0.4)
    for i, (st, en) in enumerate(filled):
        if dur > 0:
            st = min(max(0.0, st), dur)
            en = min(max(0.0, en), dur)
        if en - st < min_len:
            continue
        out.append({"index": i, "start": st, "end": en})
    return out


def _fallback_scene_segments(*, duration_seconds: float | None) -> list[dict[str, Any]]:
    if not duration_seconds or duration_seconds <= 0:
        return []
    min_len = _safe_float(os.getenv("ENVID_SCENE_FALLBACK_MIN_SECONDS"), 30.0)
    max_scenes = _parse_int(os.getenv("ENVID_SCENE_FALLBACK_MAX"), default=20, min_value=2, max_value=200)
    target = _parse_int(os.getenv("ENVID_SCENE_FALLBACK_TARGET"), default=8, min_value=2, max_value=max_scenes)

    if duration_seconds < min_len * 2:
        return []

    count = max(2, min(int(target), int(max_scenes)))
    seg_len = max(min_len, float(duration_seconds) / float(count))
    scenes: list[dict[str, Any]] = []
    cur = 0.0
    index = 0
    while cur < float(duration_seconds) - 0.01:
        end = min(float(duration_seconds), cur + seg_len)
        if end - cur >= min_len:
            scenes.append({"index": index, "start": cur, "end": end})
            index += 1
        cur = end
    return scenes


def _summarizer_post_json(*, service_url: str, endpoint: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    url = service_url.rstrip("/") + endpoint
    resp = requests.post(url, json=payload, timeout=float(timeout_s))
    if resp.status_code >= 400:
        raise RuntimeError(f"Summarizer request failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    return data if isinstance(data, dict) else {}


def _generate_synopsis_via_summarizer_service(
    *,
    service_url: str,
    text: str,
    language_code: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    timeout_s = _safe_float(os.getenv("OPENROUTER_TIMEOUT_SECONDS"), 20.0)
    payload = {"text": text, "language_code": language_code}
    data = _summarizer_post_json(
        service_url=service_url,
        endpoint="/synopsis",
        payload=payload,
        timeout_s=timeout_s,
    )
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {"available": True, "applied": False}
    synopsis = data.get("synopsis") if isinstance(data.get("synopsis"), dict) else None
    return synopsis, meta


def _scene_summaries_via_summarizer_service(
    *,
    service_url: str,
    scenes: list[dict[str, Any]],
    transcript_segments: list[dict[str, Any]],
    labels_src: list[dict[str, Any]] | None,
    language_code: str | None,
) -> tuple[dict[int, str] | None, dict[str, Any]]:
    timeout_s = _safe_float(os.getenv("OPENROUTER_TIMEOUT_SECONDS"), 30.0)
    payload = {
        "scenes": scenes,
        "transcript_segments": transcript_segments,
        "labels": labels_src or [],
        "language_code": language_code,
    }
    data = _summarizer_post_json(
        service_url=service_url,
        endpoint="/scene_summaries",
        payload=payload,
        timeout_s=timeout_s,
    )
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {"available": True, "applied": False}
    summaries_raw = data.get("summaries")
    if isinstance(summaries_raw, dict):
        out: dict[int, str] = {}
        for key, value in summaries_raw.items():
            try:
                idx = int(key)
            except Exception:
                continue
            summary = str(value or "").strip()
            if summary:
                out[idx] = summary
        if out:
            return out, meta
    return None, meta


def _verify_transcript_via_summarizer_service(
    *,
    service_url: str,
    text: str,
    language_code: str | None,
) -> dict[str, Any]:
    timeout_s = _safe_float(os.getenv("OPENROUTER_TIMEOUT_SECONDS"), 15.0)
    payload = {"text": text, "language_code": language_code}
    data = _summarizer_post_json(
        service_url=service_url,
        endpoint="/verify_transcript",
        payload=payload,
        timeout_s=timeout_s,
    )
    return data if isinstance(data, dict) else {}


def _openrouter_headers() -> dict[str, str] | None:
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        return None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    http_referer = (os.getenv("OPENROUTER_HTTP_REFERER") or "").strip()
    x_title = (os.getenv("OPENROUTER_X_TITLE") or "").strip()
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if x_title:
        headers["X-Title"] = x_title
    return headers


def _openrouter_base_url() -> str:
    return (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip().rstrip("/")


def _openrouter_model(default_model: str) -> str:
    return (
        os.getenv("OPENROUTER_MODEL")
        or os.getenv("OPENROUTER_TRANSCRIPT_MODEL")
        or default_model
    ).strip()


def _openrouter_chat_completion(payload: dict[str, Any], timeout_s: float) -> str | None:
    headers = _openrouter_headers()
    if not headers:
        return None
    base_url = _openrouter_base_url()
    resp = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=float(timeout_s))
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenRouter failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    return content or None


def _gemini_api_key() -> str | None:
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    return api_key or None


def _gemini_base_url() -> str:
    return (os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta").strip().rstrip("/")


def _gemini_model(default_model: str) -> str:
    model = (os.getenv("GEMINI_MODEL") or os.getenv("GEMINI_DEFAULT_MODEL") or default_model).strip()
    if not model:
        return default_model
    if model.startswith("models/"):
        return model
    return f"models/{model}"


def _gemini_generate_text(
    *,
    model: str,
    prompt: str,
    system: str | None = None,
    timeout_s: float = 20.0,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str | None:
    api_key = _gemini_api_key()
    if not api_key:
        return None
    base_url = _gemini_base_url()
    model_path = model if model.startswith("models/") else f"models/{model}"
    url = f"{base_url}/{model_path}:generateContent?key={api_key}"

    payload: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    generation_config: dict[str, Any] = {}
    if temperature is not None:
        generation_config["temperature"] = float(temperature)
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = int(max_tokens)
    if generation_config:
        payload["generationConfig"] = generation_config

    resp = requests.post(url, json=payload, timeout=float(timeout_s))
    if resp.status_code >= 400:
        raise RuntimeError(f"Gemini failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return None
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    texts = [str(p.get("text") or "").strip() for p in parts if isinstance(p, dict)]
    content = "\n".join([t for t in texts if t]).strip()
    return content or None


def _local_llm_base_url() -> str:
    return (os.getenv("ENVID_LLM_BASE_URL") or "http://localhost:8000/v1").strip().rstrip("/")


def _local_llm_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = (os.getenv("ENVID_LLM_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _local_llm_model(default_model: str, *, transcript: bool = False) -> str:
    if transcript:
        return (
            os.getenv("ENVID_LLM_TRANSCRIPT_MODEL")
            or os.getenv("ENVID_LLM_MODEL")
            or default_model
        ).strip()
    return (os.getenv("ENVID_LLM_MODEL") or default_model).strip()


def _local_llm_available() -> bool:
    base_url = _local_llm_base_url()
    try:
        resp = requests.get(f"{base_url}/models", timeout=2.0)
        return resp.status_code < 400
    except Exception:
        return False


def _local_llm_chat_completion(payload: dict[str, Any], timeout_s: float) -> str | None:
    base_url = _local_llm_base_url()
    headers = _local_llm_headers()
    resp = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=float(timeout_s))
    if resp.status_code >= 400:
        raise RuntimeError(f"Local LLM failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    return content or None


def _strip_code_fences(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned.strip())
    cleaned = cleaned.replace("```", "")
    return cleaned.strip()


def _extract_json_payload(content: str) -> dict[str, Any] | None:
    raw = (content or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def _clean_value_text(value: str) -> str:
    cleaned = value.strip()
    cleaned = cleaned.strip(" \t\n\r\"\'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _extract_synopsis_from_text(content: str) -> dict[str, str] | None:
    raw = _strip_code_fences(content)
    if not raw:
        return None

    candidates: list[str] = [raw]
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m and m.group(0) not in candidates:
        candidates.append(m.group(0))
    for candidate in candidates:
        try:
            payload_json = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload_json, dict):
            short = payload_json.get("short")
            long = payload_json.get("long")
            if isinstance(short, str) and isinstance(long, str):
                return {"short": short.strip(), "long": long.strip()}

    short_match = re.search(r"short\s*[:=]\s*(.+?)(?:(?:\n\s*long\s*[:=])|$)", raw, flags=re.IGNORECASE | re.DOTALL)
    long_match = re.search(r"long\s*[:=]\s*(.+)$", raw, flags=re.IGNORECASE | re.DOTALL)
    if short_match and long_match:
        short = _clean_value_text(short_match.group(1))
        long = _clean_value_text(long_match.group(1))
        if short and long:
            return {"short": short, "long": long}

    short_match = re.search(r"\"?short\"?\s*[:=]\s*\"(.+?)\"", raw, flags=re.IGNORECASE | re.DOTALL)
    long_match = re.search(r"\"?long\"?\s*[:=]\s*\"(.+?)\"", raw, flags=re.IGNORECASE | re.DOTALL)
    if short_match and long_match:
        short = _clean_value_text(short_match.group(1))
        long = _clean_value_text(long_match.group(1))
        if short and long:
            return {"short": short, "long": long}

    return None


def _word_list(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", (text or "").strip()) if w]


def _truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text
    words = _word_list(text)
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def _enforce_synopsis_limits(payload: dict[str, str]) -> dict[str, str]:
    short = _truncate_words(payload.get("short", ""), 50)
    long = _truncate_words(payload.get("long", ""), 150)
    return {"short": short, "long": long}


def _devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(1 for ch in text if ch.isalpha())
    dev = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
    return dev / max(1, letters)


def _needs_hindi_retry(payload: dict[str, str], language_code: str | None) -> bool:
    lang = (language_code or "").strip().lower()
    if not lang.startswith("hi"):
        return False
    sample = " ".join(filter(None, [payload.get("short", ""), payload.get("long", "")]))
    return _devanagari_ratio(sample) < 0.2


def _verify_transcript_via_llm_direct(*, text: str, language_code: str | None) -> dict[str, Any]:
    raw = _normalize_transcript_basic(text)
    if not raw:
        return {
            "ok": False,
            "score": 0.0,
            "corrected_text": None,
            "issues": ["empty_transcript"],
            "meta": {"available": False, "provider": "none", "reason": "no_transcript"},
        }

    lang = (language_code or "").strip() or "unknown"
    prompt = (
        "I am giving you a script that was generated from audio. Your job is to correct it.\n\n"
        "Instructions:\n"
        "- Remove any irrelevant or incorrect words that came from transcription errors.\n"
        "- Replace misheard or misspelled words with the correct ones so the script makes complete sense.\n"
        "- Fix grammar, punctuation, and sentence flow.\n"
        "- Do NOT change the meaning of the script.\n"
        "- Do NOT add new content that wasn’t implied.\n"
        "- Only rewrite the script in clean, correct, readable form.\n\n"
        "Return STRICT JSON with keys: ok (bool), score (0-1), corrected_text, issues (array).\n"
        "Rules:\n"
        "- corrected_text must only fix wrong words, punctuation, casing, spacing, and obvious sentence boundaries.\n"
        "- Do NOT add new content; do NOT change meaning; do NOT paraphrase.\n"
        "- ok is true only if the transcript is coherent and not gibberish.\n"
        "- score reflects overall sense/quality (1=excellent).\n"
        "- issues is a short list like [\"gibberish\", \"language_mismatch\"].\n"
        "- Output JSON only (no markdown).\n\n"
        f"Language hint: {lang}\n\n"
        f"send script as input\n{raw[:12000]}\n"
    )

    timeout_s = _safe_float(os.getenv("OPENROUTER_TIMEOUT_SECONDS"), 15.0)
    model_or = _openrouter_model("meta-llama/Meta-Llama-3.1-8B-Instruct")
    model_local = _local_llm_model("meta-llama/Meta-Llama-3.1-8B-Instruct", transcript=True)

    gemini_error = None
    for _ in range(3):
        try:
            content = _gemini_generate_text(
                model=_gemini_model("gemini-2.5-pro"),
                prompt=prompt,
                system="Return JSON only.",
                timeout_s=timeout_s,
                max_tokens=1200,
                temperature=0.2,
            ) or ""
            parsed = _extract_json_payload(content)
            if isinstance(parsed, dict):
                return {**parsed, "meta": {"available": True, "provider": "gemini", "model": _gemini_model("gemini-2.5-pro")}}
            gemini_error = "invalid_response"
        except Exception as exc:
            gemini_error = str(exc)[:240]

    data_local = {
        "model": model_local,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    local_error = None
    for _ in range(3):
        try:
            content = _local_llm_chat_completion(data_local, timeout_s) or ""
            parsed = _extract_json_payload(content)
            if isinstance(parsed, dict):
                return {**parsed, "meta": {"available": True, "provider": "local", "model": model_local}}
            local_error = "invalid_response"
        except Exception as exc:
            local_error = str(exc)[:240]

    return {
        "ok": False,
        "score": 0.0,
        "corrected_text": None,
        "issues": ["request_failed"],
        "meta": {
            "available": False,
            "provider": "gemini",
            "model": _gemini_model("gemini-2.5-pro"),
            "error": str(gemini_error or local_error or "request_failed")[:240],
        },
    }


def _openrouter_direct_generate_synopsis(
    *,
    text: str,
    language_code: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not _openrouter_headers():
        return None, {"available": False, "provider": "openrouter", "reason": "missing_api_key"}

    raw = _normalize_transcript_basic(text)
    if not raw:
        return None, {"available": False, "provider": "openrouter", "reason": "no_transcript"}

    model = _openrouter_model("meta-llama/Meta-Llama-3.1-8B-Instruct")
    timeout_s = _safe_float(os.getenv("OPENROUTER_TIMEOUT_SECONDS"), 20.0)
    lang = (language_code or "").strip() or "unknown"

    system_prompt_short = (
        "Create a SHORT movie-style synopsis (30–50 words) based strictly on the transcript I will provide.\n\n"
        "Requirements:\n"
        "- Summarize the premise and central conflict in an abstracted way.\n"
        "- Be concise, cinematic, and spoiler-free.\n"
        "- Use only facts explicitly stated in the transcript; do NOT add new details.\n"
        "- Paraphrase: do NOT copy phrases longer than 6 words from the transcript.\n"
        "- Maintain the exact language of the transcript (no translation).\n"
        "- Preserve proper nouns and names exactly.\n"
        "- If something is unclear, say only what is confirmed.\n\n"
        "Wait for the transcript.\n"
    )

    system_prompt_long = (
        "Create a LONG movie-style synopsis (70–150 words) based strictly on the transcript I will provide.\n\n"
        "Requirements:\n"
        "- Describe the setting, character motivations, and major plot developments.\n"
        "- Highlight dramatic tension without revealing endings or twists (spoiler-free).\n"
        "- Maintain a polished, cinematic tone.\n"
        "- Use only information explicitly mentioned in the transcript; do NOT add new details.\n"
        "- Paraphrase: do NOT copy phrases longer than 6 words from the transcript.\n"
        "- Do NOT translate; keep the same language.\n"
        "- Preserve proper nouns and names exactly.\n"
        "- If something is vague, state only what can be confirmed.\n\n"
        "Wait for the transcript.\n"
    )

    user_prompt = (
        f"Language hint: {lang}\n\n"
        f"Transcript:\n{raw[:12000]}\n"
    )

    def _request_synopsis(kind: str, extra_rules: str = "") -> str | None:
        if kind == "long":
            system_prompt = system_prompt_long
            max_tokens = 1200
        else:
            system_prompt = system_prompt_short
            max_tokens = 600
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + extra_rules},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        content = (_openrouter_chat_completion(data, timeout_s) or "").strip()
        return content or None

    try:
        short_text = _request_synopsis("short") or ""
        long_text = _request_synopsis("long") or ""
        synopsis_payload = _enforce_synopsis_limits({"short": short_text, "long": long_text})

        if _needs_hindi_retry(synopsis_payload, language_code):
            short_text = _request_synopsis("short", "\nRespond in Hindi only using Devanagari characters.\n") or short_text
            long_text = _request_synopsis("long", "\nRespond in Hindi only using Devanagari characters.\n") or long_text
            synopsis_payload = _enforce_synopsis_limits({"short": short_text, "long": long_text})

        if synopsis_payload.get("short") or synopsis_payload.get("long"):
            return synopsis_payload, {"available": True, "applied": True, "provider": "openrouter", "model": model}
        return None, {"available": False, "provider": "openrouter", "reason": "invalid_response", "model": model}
    except Exception as exc:
        return None, {"available": False, "provider": "openrouter", "reason": "request_failed", "error": str(exc)[:240]}


def _gemini_direct_generate_synopsis(
    *,
    text: str,
    language_code: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not _gemini_api_key():
        return None, {"available": False, "provider": "gemini", "reason": "missing_api_key"}

    raw = _normalize_transcript_basic(text)
    if not raw:
        return None, {"available": False, "provider": "gemini", "reason": "no_transcript"}

    model = _gemini_model("gemini-2.5-pro")
    timeout_s = _safe_float(os.getenv("GEMINI_TIMEOUT_SECONDS") or os.getenv("OPENROUTER_TIMEOUT_SECONDS"), 20.0)
    lang = (language_code or "").strip() or "unknown"

    system_prompt_short = (
        "Create a SHORT movie-style synopsis (30–50 words) based strictly on the transcript I will provide.\n\n"
        "Requirements:\n"
        "- Summarize the premise and central conflict in an abstracted way.\n"
        "- Be concise, cinematic, and spoiler-free.\n"
        "- Use only details that are explicitly stated in the transcript; do NOT add new details.\n"
        "- Paraphrase: do NOT copy phrases longer than 6 words from the transcript.\n"
        "- Maintain the exact language of the transcript (no translation).\n"
        "- Preserve proper nouns and names exactly.\n"
        "- If a detail is unclear or missing, state only what is confirmed.\n"
    )

    system_prompt_long = (
        "Create a LONG movie-style synopsis (70–150 words) based strictly on the transcript I will provide.\n\n"
        "Requirements:\n"
        "- Describe the setting, character motivations, and major plot developments.\n"
        "- Highlight dramatic tension without revealing endings or twists (spoiler-free).\n"
        "- Maintain a polished, cinematic tone.\n"
        "- Use only information explicitly mentioned in the transcript; do NOT add new details.\n"
        "- Paraphrase: do NOT copy phrases longer than 6 words from the transcript.\n"
        "- Do NOT translate; keep the same language.\n"
        "- Preserve proper nouns and names exactly.\n"
        "- If something is vague, state only what can be confirmed.\n"
    )

    user_prompt = (
        f"Language hint: {lang}\n\n"
        f"Transcript:\n{raw[:12000]}\n"
    )

    def _request_synopsis(kind: str, extra_rules: str = "") -> str | None:
        if kind == "long":
            system_prompt = system_prompt_long
            max_tokens = 1200
        else:
            system_prompt = system_prompt_short
            max_tokens = 600
        return _gemini_generate_text(
            model=model,
            prompt=user_prompt + extra_rules,
            system=system_prompt,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            temperature=0.2,
        )

    try:
        short_text = _request_synopsis("short") or ""
        long_text = _request_synopsis("long") or ""
        synopsis_payload = _enforce_synopsis_limits({"short": short_text, "long": long_text})

        if _needs_hindi_retry(synopsis_payload, language_code):
            short_text = _request_synopsis("short", "\nRespond in Hindi only using Devanagari characters.\n") or short_text
            long_text = _request_synopsis("long", "\nRespond in Hindi only using Devanagari characters.\n") or long_text
            synopsis_payload = _enforce_synopsis_limits({"short": short_text, "long": long_text})

        if synopsis_payload.get("short") or synopsis_payload.get("long"):
            return synopsis_payload, {"available": True, "applied": True, "provider": "gemini", "model": model}
        return None, {"available": False, "provider": "gemini", "reason": "invalid_response", "model": model}
    except Exception as exc:
        return None, {"available": False, "provider": "gemini", "reason": "request_failed", "error": str(exc)[:240]}


def _local_llm_generate_synopsis(
    *,
    text: str,
    language_code: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    raw = _normalize_transcript_basic(text)
    if not raw:
        return None, {"available": False, "provider": "local", "reason": "no_transcript"}

    model = _local_llm_model("meta-llama/Meta-Llama-3.1-8B-Instruct")
    timeout_s = _safe_float(os.getenv("OPENROUTER_TIMEOUT_SECONDS"), 20.0)
    lang = (language_code or "").strip() or "unknown"

    system_prompt_short = (
        "Create a SHORT movie-style synopsis (30–50 words) based strictly on the transcript I will provide.\n\n"
        "Requirements:\n"
        "- Summarize the premise and central conflict in an abstracted way.\n"
        "- Be concise, cinematic, and spoiler-free.\n"
        "- Use only details that are explicitly stated in the transcript; do NOT add new details.\n"
        "- Paraphrase: do NOT copy phrases longer than 6 words from the transcript.\n"
        "- Maintain the exact language of the transcript (no translation).\n"
        "- Preserve proper nouns and names exactly.\n"
        "- If a detail is unclear or missing, state only what is confirmed.\n\n"
        "Wait for the transcript.\n"
    )

    system_prompt_long = (
        "Create a LONG movie-style synopsis (70–150 words) based strictly on the transcript I will provide.\n\n"
        "Requirements:\n"
        "- Describe the setting, character motivations, and major plot developments.\n"
        "- Highlight dramatic tension without revealing endings or twists (spoiler-free).\n"
        "- Maintain a polished, cinematic tone.\n"
        "- Use only information explicitly mentioned in the transcript; do NOT add new details.\n"
        "- Paraphrase: do NOT copy phrases longer than 6 words from the transcript.\n"
        "- Do NOT translate; keep the same language.\n"
        "- Preserve proper nouns and names exactly.\n"
        "- If something is vague, state only what can be confirmed.\n\n"
        "Wait for the transcript.\n"
    )

    user_prompt = (
        f"Language hint: {lang}\n\n"
        f"Transcript:\n{raw[:12000]}\n"
    )

    def _request_synopsis(kind: str, extra_rules: str = "") -> str | None:
        if kind == "long":
            system_prompt = system_prompt_long
            max_tokens = 1200
        else:
            system_prompt = system_prompt_short
            max_tokens = 600
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + extra_rules},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        content = (_local_llm_chat_completion(data, timeout_s) or "").strip()
        return content or None

    try:
        short_text = _request_synopsis("short") or ""
        long_text = _request_synopsis("long") or ""
        synopsis_payload = _enforce_synopsis_limits({"short": short_text, "long": long_text})

        if _needs_hindi_retry(synopsis_payload, language_code):
            short_text = _request_synopsis("short", "\nRespond in Hindi only using Devanagari characters.\n") or short_text
            long_text = _request_synopsis("long", "\nRespond in Hindi only using Devanagari characters.\n") or long_text
            synopsis_payload = _enforce_synopsis_limits({"short": short_text, "long": long_text})

        if synopsis_payload.get("short") or synopsis_payload.get("long"):
            return synopsis_payload, {"available": True, "applied": True, "provider": "local", "model": model}
        return None, {"available": False, "provider": "local", "reason": "invalid_response", "model": model}
    except Exception as exc:
        return None, {"available": False, "provider": "local", "reason": "request_failed", "error": str(exc)[:240]}


def _gemini_direct_scene_summaries(
    *,
    scenes: list[dict[str, Any]],
    transcript_segments: list[dict[str, Any]],
    labels_src: list[dict[str, Any]] | None,
    objects_src: list[dict[str, Any]] | None,
    language_code: str | None,
) -> tuple[dict[int, str] | None, dict[str, Any]]:
    if not _gemini_api_key():
        return None, {"available": False, "provider": "gemini", "reason": "missing_api_key"}

    if not isinstance(scenes, list) or not scenes:
        return None, {"available": False, "provider": "gemini", "reason": "no_scenes"}

    model = _gemini_model("gemini-2.5-pro")
    timeout_s = _safe_float(os.getenv("GEMINI_TIMEOUT_SECONDS") or os.getenv("OPENROUTER_TIMEOUT_SECONDS"), 25.0)

    max_scenes = _parse_int(os.getenv("ENVID_SCENE_LLM_MAX_SCENES"), default=30, min_value=1, max_value=200)
    max_chars = _parse_int(os.getenv("ENVID_SCENE_LLM_MAX_CHARS_PER_SCENE"), default=600, min_value=120, max_value=4000)
    lang = (language_code or "").strip() or "unknown"

    def _scene_context_text(sc: dict[str, Any]) -> tuple[int, str, list[str]]:
        try:
            st = float(sc.get("start") or sc.get("start_seconds") or 0.0)
            en = float(sc.get("end") or sc.get("end_seconds") or st)
        except Exception:
            st, en = 0.0, 0.0
        if en < st:
            st, en = en, st

        idx = int(sc.get("index") or 0)
        segs: list[str] = []
        for seg in transcript_segments:
            if not isinstance(seg, dict):
                continue
            ss = _safe_float(seg.get("start"), 0.0)
            se = _safe_float(seg.get("end"), ss)
            if se <= ss:
                continue
            if _overlap_seconds(ss, se, st, en) <= 0:
                continue
            text = str(seg.get("text") or "").strip()
            if text:
                segs.append(text)

        label_names: list[str] = []
        if isinstance(labels_src, list):
            for lab in labels_src:
                if not isinstance(lab, dict):
                    continue
                if _overlap_seconds(_safe_float(lab.get("start"), 0.0), _safe_float(lab.get("end"), 0.0), st, en) <= 0:
                    continue
                name = str(lab.get("name") or lab.get("label") or "").strip()
                if name:
                    label_names.append(name)
        label_names = list(dict.fromkeys(label_names))

        object_names: list[str] = []
        if isinstance(objects_src, list):
            for obj in objects_src:
                if not isinstance(obj, dict):
                    continue
                name = str(obj.get("name") or obj.get("label") or "").strip()
                if not name:
                    continue
                for seg in (obj.get("segments") or []):
                    if not isinstance(seg, dict):
                        continue
                    if _overlap_seconds(
                        _safe_float(seg.get("start"), 0.0),
                        _safe_float(seg.get("end"), 0.0),
                        st,
                        en,
                    ) <= 0:
                        continue
                    object_names.append(name)
                    break
        object_names = list(dict.fromkeys(object_names))

        transcript_text = " ".join(segs).strip()
        if len(transcript_text) > max_chars:
            transcript_text = transcript_text[:max_chars].rsplit(" ", 1)[0].strip()

        labels_text = ", ".join(label_names[:20])
        objects_text = ", ".join(object_names[:20])
        context_parts = [f"Transcript: {transcript_text}".strip()]
        if labels_text:
            context_parts.append(f"Labels: {labels_text}")
        if objects_text:
            context_parts.append(f"Objects: {objects_text}")
        context = "\n".join([p for p in context_parts if p])
        return idx, context, object_names

    scene_inputs: list[dict[str, Any]] = []
    objects_by_index: dict[int, list[str]] = {}
    for sc in scenes[:max_scenes]:
        if not isinstance(sc, dict):
            continue
        idx, context, object_names = _scene_context_text(sc)
        if not context.replace("Transcript:", "").strip() and "Labels:" not in context:
            continue
        scene_inputs.append({"index": idx, "context": context})
        if object_names:
            objects_by_index[idx] = object_names

    if not scene_inputs:
        return None, {"available": False, "provider": "gemini", "reason": "empty_inputs", "model": model}

    prompt = (
        "You are a scene-by-scene summarizer.\n"
        "Summarize each scene in 1-2 short sentences.\n"
        "Use ONLY the provided scene context.\n"
        "Return STRICT JSON only: {\"scenes\": [{\"index\": <int>, \"summary\": <string>}]}.\n"
        "Do not add any extra keys or commentary.\n"
        "Write in the SAME LANGUAGE as the transcript. Do NOT translate.\n\n"
        f"Language hint: {lang}\n\n"
        f"Scenes:\n{json.dumps(scene_inputs, ensure_ascii=False)[:20000]}\n"
    )

    try:
        content = _gemini_generate_text(
            model=model,
            prompt=prompt,
            system="Return only JSON.",
            timeout_s=timeout_s,
            max_tokens=1400,
            temperature=0.2,
        ) or ""
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        payload_json = json.loads(m.group(0) if m else content)
        scene_items = payload_json.get("scenes") if isinstance(payload_json, dict) else None
        if isinstance(scene_items, list):
            out: dict[int, str] = {}
            for item in scene_items:
                if not isinstance(item, dict):
                    continue
                try:
                    idx = int(item.get("index"))
                except Exception:
                    continue
                summary = str(item.get("summary") or "").strip()
                if summary:
                    out[idx] = summary
            if out:
                return out, {"available": True, "applied": True, "provider": "gemini", "model": model}
        return None, {"available": False, "provider": "gemini", "reason": "invalid_response", "model": model}
    except Exception as exc:
        return None, {"available": False, "provider": "gemini", "reason": "request_failed", "error": str(exc)[:240]}


def _openrouter_direct_scene_summaries(
    *,
    scenes: list[dict[str, Any]],
    transcript_segments: list[dict[str, Any]],
    labels_src: list[dict[str, Any]] | None,
    objects_src: list[dict[str, Any]] | None,
    language_code: str | None,
) -> tuple[dict[int, str] | None, dict[str, Any]]:
    if not _openrouter_headers():
        return None, {"available": False, "provider": "openrouter", "reason": "missing_api_key"}

    if not isinstance(scenes, list) or not scenes:
        return None, {"available": False, "provider": "openrouter", "reason": "no_scenes"}

    model = _openrouter_model("meta-llama/Meta-Llama-3.1-8B-Instruct")
    timeout_s = _safe_float(os.getenv("OPENROUTER_TIMEOUT_SECONDS"), 25.0)

    max_scenes = _parse_int(os.getenv("ENVID_SCENE_LLM_MAX_SCENES"), default=30, min_value=1, max_value=200)
    max_chars = _parse_int(os.getenv("ENVID_SCENE_LLM_MAX_CHARS_PER_SCENE"), default=600, min_value=120, max_value=4000)
    lang = (language_code or "").strip() or "unknown"

    def _scene_context_text(sc: dict[str, Any]) -> tuple[int, str, list[str]]:
        try:
            st = float(sc.get("start") or sc.get("start_seconds") or 0.0)
            en = float(sc.get("end") or sc.get("end_seconds") or st)
        except Exception:
            st, en = 0.0, 0.0
        if en < st:
            st, en = en, st

        idx = int(sc.get("index") or 0)
        segs: list[str] = []
        for seg in transcript_segments:
            if not isinstance(seg, dict):
                continue
            ss = _safe_float(seg.get("start"), 0.0)
            se = _safe_float(seg.get("end"), ss)
            if se <= ss:
                continue
            if _overlap_seconds(ss, se, st, en) <= 0:
                continue
            text = str(seg.get("text") or "").strip()
            if text:
                segs.append(text)

        label_names: list[str] = []
        if isinstance(labels_src, list):
            for lab in labels_src:
                if not isinstance(lab, dict):
                    continue
                if _overlap_seconds(_safe_float(lab.get("start"), 0.0), _safe_float(lab.get("end"), 0.0), st, en) <= 0:
                    continue
                name = str(lab.get("name") or lab.get("label") or "").strip()
                if name:
                    label_names.append(name)
        label_names = list(dict.fromkeys(label_names))

        object_names: list[str] = []
        if isinstance(objects_src, list):
            for obj in objects_src:
                if not isinstance(obj, dict):
                    continue
                name = str(obj.get("name") or obj.get("label") or "").strip()
                if not name:
                    continue
                for seg in (obj.get("segments") or []):
                    if not isinstance(seg, dict):
                        continue
                    if _overlap_seconds(
                        _safe_float(seg.get("start"), 0.0),
                        _safe_float(seg.get("end"), 0.0),
                        st,
                        en,
                    ) <= 0:
                        continue
                    object_names.append(name)
                    break
        object_names = list(dict.fromkeys(object_names))

        transcript_text = " ".join(segs).strip()
        if len(transcript_text) > max_chars:
            transcript_text = transcript_text[:max_chars].rsplit(" ", 1)[0].strip()

        labels_text = ", ".join(label_names[:20])
        objects_text = ", ".join(object_names[:20])
        context_parts = [f"Transcript: {transcript_text}".strip()]
        if labels_text:
            context_parts.append(f"Labels: {labels_text}")
        if objects_text:
            context_parts.append(f"Objects: {objects_text}")
        context = "\n".join([p for p in context_parts if p])
        return idx, context, object_names

    scene_inputs: list[dict[str, Any]] = []
    objects_by_index: dict[int, list[str]] = {}
    for sc in scenes[:max_scenes]:
        if not isinstance(sc, dict):
            continue
        idx, context, object_names = _scene_context_text(sc)
        if not context.replace("Transcript:", "").strip() and "Labels:" not in context:
            continue
        scene_inputs.append({"index": idx, "context": context})
        if object_names:
            objects_by_index[idx] = object_names

    if not scene_inputs:
        return None, {"available": False, "provider": "openrouter", "reason": "empty_inputs", "model": model}

    prompt = (
        "You are a scene-by-scene summarizer.\n"
        "Summarize each scene in 1-2 short sentences.\n"
        "Use ONLY the provided scene context.\n"
        "Return STRICT JSON only: {\"scenes\": [{\"index\": <int>, \"summary\": <string>}]}\n"
        "Do not add any extra keys or commentary.\n"
        "Write in the SAME LANGUAGE as the transcript. Do NOT translate.\n\n"
        f"Language hint: {lang}\n\n"
        f"Scenes:\n{json.dumps(scene_inputs, ensure_ascii=False)[:20000]}\n"
    )

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1400,
    }

    try:
        content = _openrouter_chat_completion(data, timeout_s) or ""
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        payload_json = json.loads(m.group(0) if m else content)
        scene_items = payload_json.get("scenes") if isinstance(payload_json, dict) else None
        if isinstance(scene_items, list):
            out: dict[int, str] = {}
            for item in scene_items:
                if not isinstance(item, dict):
                    continue
                try:
                    idx = int(item.get("index"))
                except Exception:
                    continue
                summary = str(item.get("summary") or "").strip()
                if summary:
                    out[idx] = summary
            if out:
                return out, {"available": True, "applied": True, "provider": "openrouter", "model": model}
        return None, {"available": False, "provider": "openrouter", "reason": "invalid_response", "model": model}
    except Exception as exc:
        return None, {"available": False, "provider": "openrouter", "reason": "request_failed", "error": str(exc)[:240]}


    def _local_llm_scene_summaries(
        *,
        scenes: list[dict[str, Any]],
        transcript_segments: list[dict[str, Any]],
        labels_src: list[dict[str, Any]] | None,
        objects_src: list[dict[str, Any]] | None,
        language_code: str | None,
    ) -> tuple[dict[int, str] | None, dict[str, Any]]:
        if not scenes:
            return None, {"available": False, "provider": "local", "reason": "no_scenes"}

        model = _local_llm_model("meta-llama/Meta-Llama-3.1-8B-Instruct")
        timeout_s = _safe_float(os.getenv("OPENROUTER_TIMEOUT_SECONDS"), 25.0)
        max_scenes = _parse_int(os.getenv("ENVID_SCENE_LLM_MAX_SCENES"), default=30, min_value=1, max_value=200)
        max_chars = _parse_int(os.getenv("ENVID_SCENE_LLM_MAX_CHARS_PER_SCENE"), default=600, min_value=120, max_value=4000)
        lang = (language_code or "").strip() or "unknown"

        def _scene_context_text(sc: dict[str, Any]) -> tuple[int, str, list[str]]:
            try:
                st = float(sc.get("start") or sc.get("start_seconds") or 0.0)
                en = float(sc.get("end") or sc.get("end_seconds") or st)
            except Exception:
                st, en = 0.0, 0.0
            if en < st:
                st, en = en, st

            idx = int(sc.get("index") or 0)
            segs: list[str] = []
            for seg in transcript_segments:
                if not isinstance(seg, dict):
                    continue
                ss = _safe_float(seg.get("start"), 0.0)
                se = _safe_float(seg.get("end"), ss)
                if se <= ss:
                    continue
                if _overlap_seconds(ss, se, st, en) <= 0:
                    continue
                text = str(seg.get("text") or "").strip()
                if text:
                    segs.append(text)

            label_names: list[str] = []
            if isinstance(labels_src, list):
                for lab in labels_src:
                    if not isinstance(lab, dict):
                        continue
                    if _overlap_seconds(_safe_float(lab.get("start"), 0.0), _safe_float(lab.get("end"), 0.0), st, en) <= 0:
                        continue
                    name = str(lab.get("name") or lab.get("label") or "").strip()
                    if name:
                        label_names.append(name)
            label_names = list(dict.fromkeys(label_names))

            object_names: list[str] = []
            if isinstance(objects_src, list):
                for obj in objects_src:
                    if not isinstance(obj, dict):
                        continue
                    name = str(obj.get("name") or obj.get("label") or "").strip()
                    if not name:
                        continue
                    for seg in (obj.get("segments") or []):
                        if not isinstance(seg, dict):
                            continue
                        if _overlap_seconds(
                            _safe_float(seg.get("start"), 0.0),
                            _safe_float(seg.get("end"), 0.0),
                            st,
                            en,
                        ) <= 0:
                            continue
                        object_names.append(name)
                        break
            object_names = list(dict.fromkeys(object_names))

            transcript_text = " ".join(segs).strip()
            if len(transcript_text) > max_chars:
                transcript_text = transcript_text[:max_chars].rsplit(" ", 1)[0].strip()

            labels_text = ", ".join(label_names[:20])
            objects_text = ", ".join(object_names[:20])
            context_parts = [f"Transcript: {transcript_text}".strip()]
            if labels_text:
                context_parts.append(f"Labels: {labels_text}")
            if objects_text:
                context_parts.append(f"Objects: {objects_text}")
            context = "\n".join([p for p in context_parts if p])
            return idx, context, object_names

        scene_inputs: list[dict[str, Any]] = []
        for sc in scenes[:max_scenes]:
            if not isinstance(sc, dict):
                continue
            idx, context, _object_names = _scene_context_text(sc)
            if not context.replace("Transcript:", "").strip() and "Labels:" not in context:
                continue
            scene_inputs.append({"index": idx, "context": context})

        if not scene_inputs:
            return None, {"available": False, "provider": "local", "reason": "empty_inputs", "model": model}

        prompt = (
            "You are a scene-by-scene summarizer.\n"
            "Summarize each scene in 1-2 short sentences.\n"
            "Use ONLY the provided scene context.\n"
            "Return STRICT JSON only: {\"scenes\": [{\"index\": <int>, \"summary\": <string>}]}.\n"
            "Do not add any extra keys or commentary.\n"
            "Write in the SAME LANGUAGE as the transcript. Do NOT translate.\n\n"
            f"Language hint: {lang}\n\n"
            f"Scenes:\n{json.dumps(scene_inputs, ensure_ascii=False)[:20000]}\n"
        )

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1200,
        }

        try:
            content = (_local_llm_chat_completion(data, timeout_s) or "").strip()
            payload = _extract_json_payload(content)
            if not isinstance(payload, dict):
                return None, {"available": False, "provider": "local", "reason": "invalid_response", "model": model}
            scene_items = payload.get("scenes")
            if isinstance(scene_items, list):
                out: dict[int, str] = {}
                for item in scene_items:
                    if not isinstance(item, dict):
                        continue
                    try:
                        idx = int(item.get("index"))
                    except Exception:
                        continue
                    summary = str(item.get("summary") or "").strip()
                    if summary:
                        out[idx] = summary
                if out:
                    return out, {"available": True, "applied": True, "provider": "local", "model": model}
            return None, {"available": False, "provider": "local", "reason": "invalid_response", "model": model}
        except Exception as exc:
            return None, {"available": False, "provider": "local", "reason": "request_failed", "error": str(exc)[:240]}


def _openrouter_llama_generate_synopsis(*, text: str, language_code: str | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Generate short/long synopsis via optional summarizer service."""

    summarizer_service_url = _summarizer_service_url()
    if summarizer_service_url:
        try:
            synopsis, meta = _generate_synopsis_via_summarizer_service(
                service_url=summarizer_service_url,
                text=text,
                language_code=language_code,
            )
            if isinstance(synopsis, dict):
                return synopsis, {**meta, "provider": meta.get("provider") or "summarizer"}
        except Exception as exc:
            app.logger.warning("Synopsis service failed: %s", exc)

    # Prefer Gemini, fallback to local LLM when summarizer is unavailable.
    synopsis, meta = _gemini_direct_generate_synopsis(text=text, language_code=language_code)
    if isinstance(synopsis, dict):
        return synopsis, meta

    synopsis, meta = _local_llm_generate_synopsis(text=text, language_code=language_code)
    if isinstance(synopsis, dict):
        return synopsis, meta
    return None, meta


def _openrouter_llama_scene_summaries(
    *,
    scenes: list[dict[str, Any]],
    transcript_segments: list[dict[str, Any]],
    labels_src: list[dict[str, Any]] | None,
    objects_src: list[dict[str, Any]] | None,
    language_code: str | None,
) -> tuple[dict[int, str] | None, dict[str, Any]]:
    """Generate scene-by-scene summaries via optional summarizer service."""

    summarizer_service_url = _summarizer_service_url()
    if summarizer_service_url:
        try:
            summaries, meta = _scene_summaries_via_summarizer_service(
                service_url=summarizer_service_url,
                scenes=scenes,
                transcript_segments=transcript_segments,
                labels_src=labels_src,
                language_code=language_code,
            )
            if summaries:
                return summaries, {**meta, "provider": meta.get("provider") or "summarizer"}
        except Exception as exc:
            app.logger.warning("Synopsis service scene summaries failed: %s", exc)

    summaries, meta = _gemini_direct_scene_summaries(
        scenes=scenes,
        transcript_segments=transcript_segments,
        labels_src=labels_src,
        objects_src=objects_src,
        language_code=language_code,
    )
    if summaries:
        return summaries, meta
    summaries, meta = _local_llm_scene_summaries(
        scenes=scenes,
        transcript_segments=transcript_segments,
        labels_src=labels_src,
        objects_src=objects_src,
        language_code=language_code,
    )
    if summaries:
        return summaries, meta
    return None, meta


def _synopsis_is_reasonable(text: str, language_code: str | None, *, min_words: int, max_words: int) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False

    words = [w for w in re.split(r"\s+", raw) if w]
    if len(words) < min_words or len(words) > max_words:
        return False

    letters = sum(1 for ch in raw if ch.isalpha())
    alpha_ratio = letters / max(1, len(raw))
    if alpha_ratio < 0.15:
        return False

    lang = (language_code or "").strip().lower()
    if lang.startswith("hi"):
        devanagari = sum(1 for ch in raw if "\u0900" <= ch <= "\u097F")
        if devanagari / max(1, letters) < 0.2:
            return False

    # Avoid pathological repetition (e.g., same word repeated).
    uniq = len({w.lower() for w in words})
    if uniq / max(1, len(words)) < 0.2:
        return False

    return True


def _synopsis_too_similar_to_transcript(synopsis_text: str, transcript_text: str) -> bool:
    syn = _normalize_transcript_basic(synopsis_text)
    src = _normalize_transcript_basic(transcript_text)
    if not syn or not src:
        return False
    syn_words = [w.lower() for w in re.split(r"\s+", syn) if w]
    src_words = [w.lower() for w in re.split(r"\s+", src) if w]
    if not syn_words or not src_words:
        return False
    syn_set = set(syn_words)
    src_set = set(src_words)
    if not syn_set or not src_set:
        return False
    overlap = len(syn_set & src_set) / max(1, len(syn_set))
    return overlap >= 0.9


def _validate_synopsis_payload(payload: dict[str, Any], language_code: str | None) -> bool:
    if not isinstance(payload, dict):
        return False
    short = payload.get("short")
    long = payload.get("long")
    if not _synopsis_is_reasonable(str(short or ""), language_code, min_words=4, max_words=120):
        return False
    if not _synopsis_is_reasonable(str(long or ""), language_code, min_words=10, max_words=300):
        return False
    return True


def _fallback_synopsis_from_text(text: str, language_code: str | None) -> dict[str, Any] | None:
    """Best-effort fallback synopsis using transcript text only.

    Keeps the original language by reusing transcript words.
    """

    raw = _normalize_transcript_basic(text)
    if not raw:
        return None

    words = [w for w in re.split(r"\s+", raw) if w]
    if len(words) < 20:
        return None

    def _slice(min_words: int, max_words: int) -> str:
        count = min(max_words, max(min_words, len(words)))
        return " ".join(words[:count]).strip()

    short = _slice(12, 50)
    long = _slice(35, 180)
    payload = {
        "kids": {"short": short, "long": long},
        "teens": {"short": short, "long": long},
        "adults": {"short": short, "long": long},
    }
    if _validate_synopsis_payload(payload, language_code):
        return payload
    return None


def _max_upload_bytes() -> int | None:
    raw_bytes = os.getenv("ENGRO_MAX_UPLOAD_BYTES")
    if raw_bytes:
        try:
            v = int(raw_bytes)
            return v if v > 0 else None
        except ValueError:
            pass

    raw_gb = os.getenv("ENGRO_MAX_UPLOAD_GB", "25")
    try:
        gb = float(raw_gb)
    except ValueError:
        gb = 25.0
    if gb <= 0:
        return None
    return int(gb * 1024 * 1024 * 1024)


app.config["MAX_CONTENT_LENGTH"] = _max_upload_bytes()
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.errorhandler(RequestEntityTooLarge)
def _handle_request_too_large(_: RequestEntityTooLarge) -> Any:
    limit = app.config.get("MAX_CONTENT_LENGTH")
    if isinstance(limit, int) and limit > 0:
        limit_gb = limit / (1024 * 1024 * 1024)
        return jsonify({"error": f"Upload too large. Max allowed is {limit_gb:.1f} GB."}), 413
    return jsonify({"error": "Upload too large."}), 413


def _gcs_bucket_name() -> str:
    bucket = (
        os.getenv("ENVID_METADATA_GCS_BUCKET")
        or os.getenv("GCP_GCS_BUCKET")
        or os.getenv("GCS_BUCKET")
        or ""
    ).strip()
    if not bucket:
        raise ValueError(
            "Missing GCS bucket. Set ENVID_METADATA_GCS_BUCKET (or GCP_GCS_BUCKET) in .env.local/.env.multimodal.local"
        )
    return bucket


def _gcs_working_bucket_name() -> str:
    bucket = (os.getenv("ENVID_METADATA_WORKING_BUCKET") or "envid-metadata-tarun").strip()
    if not bucket:
        raise ValueError("Missing working GCS bucket. Set ENVID_METADATA_WORKING_BUCKET.")
    return bucket


def _gcs_working_prefix() -> str:
    prefix = (os.getenv("ENVID_METADATA_WORKING_PREFIX") or "envid/raw/").strip()
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix


def _gcs_mount_root() -> Path:
    prefix = (os.getenv("ENVID_GCS_ROOT") or "envid").strip().strip("/")
    base = Path("/mnt/gcs")
    return base / prefix if prefix else base


def _gcs_job_raw_dir(job_id: str) -> Path:
    return _gcs_mount_root() / "raw" / str(job_id)


def _gcs_job_work_dir(job_id: str) -> Path:
    return _gcs_mount_root() / "work" / str(job_id)


def _gcs_job_id_counter_path() -> Path:
    return _gcs_mount_root() / "_counters" / "job_id.txt"


def _upload_to_working_bucket(*, job_id: str, local_path: Path, filename: str) -> str:
    working_bucket = _gcs_working_bucket_name()
    working_prefix = _gcs_working_prefix()
    safe_name = Path(filename).name or local_path.name
    obj = f"{working_prefix}{job_id}/{safe_name}"
    uri = f"gs://{working_bucket}/{obj}"
    client = _gcs_client()
    client.bucket(working_bucket).blob(obj).upload_from_filename(str(local_path))
    _db_file_insert(job_id, kind="gcs_working", path=None, gcs_uri=uri)
    return uri


def _gcs_rawvideo_prefix() -> str:
    prefix = (os.getenv("GCP_GCS_RAWVIDEO_PREFIX") or "envid/raw/").strip()
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix


def _gcs_enforce_prefix() -> bool:
    return _env_truthy(os.getenv("ENVID_METADATA_GCS_ENFORCE_PREFIX"), default=True)


def _allowed_gcs_buckets() -> set[str] | None:
    raw = (
        os.getenv("ENVID_METADATA_ALLOWED_GCS_BUCKETS")
        or os.getenv("GCP_GCS_ALLOWED_BUCKETS")
        or ""
    ).strip()
    if not raw:
        return None
    return {b.strip() for b in raw.split(",") if b.strip()}


def _gcs_artifacts_bucket(fallback: str) -> str:
    return (
        (os.getenv("ENVID_METADATA_GCS_ARTIFACTS_BUCKET")
        or os.getenv("ENVID_METADATA_GCP_ARTIFACTS_BUCKET")
        or fallback)
        .strip()
        or fallback
    )


def _gcs_artifacts_prefix() -> str:
    return (
        os.getenv("ENVID_METADATA_GCS_ARTIFACTS_PREFIX")
        or os.getenv("ENVID_METADATA_GCP_ARTIFACTS_PREFIX")
        or "envid-metadata/artifacts"
    ).strip().strip("/")


def _gcs_client() -> Any:
    if gcs_storage is None:
        raise RuntimeError("google-cloud-storage is not installed. Run: pip install -r microservices/backend/code/requirements.txt")
    return gcs_storage.Client()


def _ingest_service_url() -> str | None:
    return "http://ingest:5090"


def _transcribe_service_url() -> str | None:
    return (os.getenv("ENVID_TRANSCRIBE_SERVICE_URL") or "http://audio-transcription:5088").strip()


def _default_transcribe_mode() -> str:
    raw = (os.getenv("ENVID_TRANSCRIBE_BACKEND") or os.getenv("ENVID_TRANSCRIBE_ENGINE") or "").strip().lower()
    if raw in {"gcp", "gcp_speech", "speech", "speech_to_text"}:
        return "gcp_speech"
    return "openai-whisper"


def _transcribe_service_available() -> bool:
    service_url = _transcribe_service_url()
    if service_url:
        try:
            resp = requests.get(f"{service_url}/health", timeout=3)
            if resp.status_code < 400:
                return True
        except Exception:
            pass
    return False


def _upload_via_ingest_service(
    *,
    ingest_url: str,
    video_path: Path,
    filename: str,
    job_id: str,
    title: str,
    description: str,
) -> Dict[str, Any]:
    endpoint = f"{ingest_url}/upload-video"
    total_bytes = 0
    try:
        total_bytes = int(video_path.stat().st_size)
    except Exception:
        total_bytes = 0
    last_percent = -1
    last_tick = 0.0

    def _report_progress(sent: int) -> None:
        nonlocal last_percent, last_tick
        if total_bytes <= 0:
            return
        pct = int((sent * 100) / total_bytes) if total_bytes else 0
        if pct <= last_percent:
            return
        now = time.time()
        if pct < 100 and (now - last_tick) < 0.4:
            return
        last_percent = pct
        last_tick = now
        _job_step_update(job_id, "upload_to_cloud_storage", status="running", percent=pct, message=f"Uploading {pct}%")

    class _ProgressReader:
        def __init__(self, fp):
            self._fp = fp
            self._sent = 0

        def read(self, amt: int | None = None):
            data = self._fp.read(amt)
            if data:
                self._sent += len(data)
                _report_progress(self._sent)
            return data

        def tell(self):
            return self._fp.tell()

        def seek(self, offset, whence=0):
            return self._fp.seek(offset, whence)

        def close(self):
            return self._fp.close()

        @property
        def closed(self):
            return self._fp.closed

    with video_path.open("rb") as handle:
        files = {"video": (filename, _ProgressReader(handle), "application/octet-stream")}
        data = {
            "job_id": job_id,
            "title": title,
            "description": description,
        }
        resp = requests.post(endpoint, files=files, data=data, timeout=300)
    if resp.status_code >= 400:
        raise RuntimeError(f"Ingest upload failed ({resp.status_code}): {resp.text}")
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid ingest response")
    return payload


def _ffmpeg_service_url() -> str | None:
    return "http://transcoder:5091"


def _summarizer_service_url() -> str | None:
    url = (os.getenv("ENVID_SUMMARIZER_SERVICE_URL") or "").strip()
    if not url:
        return None
    return url.rstrip("/")


def _upload_gcs_with_progress(*, bucket: str, obj: str, video_path: Path, job_id: str) -> None:
    total_bytes = 0
    try:
        total_bytes = int(video_path.stat().st_size)
    except Exception:
        total_bytes = 0
    last_percent = -1
    last_tick = 0.0

    def _report_progress(sent: int) -> None:
        nonlocal last_percent, last_tick
        if total_bytes <= 0:
            return
        pct = int((sent * 100) / total_bytes) if total_bytes else 0
        if pct <= last_percent:
            return
        now = time.time()
        if pct < 100 and (now - last_tick) < 0.4:
            return
        last_percent = pct
        last_tick = now
        _job_step_update(job_id, "upload_to_cloud_storage", status="running", percent=pct, message=f"Uploading {pct}%")

    client = _gcs_client()
    blob = client.bucket(bucket).blob(obj)
    blob.chunk_size = 8 * 1024 * 1024
    sent = 0
    with video_path.open("rb") as handle, blob.open("wb") as writer:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            writer.write(chunk)
            sent += len(chunk)
            _report_progress(sent)


def _probe_via_ffmpeg_service(*, service_url: str, video_path: Path, filename: str) -> dict[str, Any]:
    endpoint = f"{service_url}/probe"
    with video_path.open("rb") as handle:
        files = {"video": (filename, handle, "application/octet-stream")}
        resp = requests.post(endpoint, files=files, timeout=300)
    if resp.status_code >= 400:
        raise RuntimeError(f"FFmpeg probe failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("Invalid ffprobe response")
    return data


def _normalize_via_ffmpeg_service(
    *,
    service_url: str,
    video_path: Path,
    filename: str,
    video_bitrate_k: str = "1500",
    audio_bitrate_k: str = "128",
) -> bytes:
    endpoint = f"{service_url}/normalize"
    with video_path.open("rb") as handle:
        files = {"video": (filename, handle, "application/octet-stream")}
        data = {
            "video_bitrate_k": video_bitrate_k,
            "audio_bitrate_k": audio_bitrate_k,
        }
        resp = requests.post(endpoint, files=files, data=data, timeout=900)
    if resp.status_code >= 400:
        raise RuntimeError(f"FFmpeg service failed ({resp.status_code}): {resp.text}")
    return resp.content


def _blackdetect_via_ffmpeg_service(
    *,
    service_url: str,
    video_path: Path,
    filename: str,
    min_black_seconds: float,
    picture_black_threshold: float,
    pixel_black_threshold: float,
    max_seconds: int,
) -> list[dict[str, Any]]:
    endpoint = f"{service_url}/blackdetect"
    with video_path.open("rb") as handle:
        files = {"video": (filename, handle, "application/octet-stream")}
        data = {
            "min_black_seconds": str(min_black_seconds),
            "picture_black_threshold": str(picture_black_threshold),
            "pixel_black_threshold": str(pixel_black_threshold),
            "max_seconds": str(max_seconds),
        }
        resp = requests.post(endpoint, files=files, data=data, timeout=float(max_seconds) + 30.0)
    if resp.status_code >= 400:
        raise RuntimeError(f"FFmpeg blackdetect failed ({resp.status_code}): {resp.text}")
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid blackdetect response")
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return []
    return [x for x in segments if isinstance(x, dict)]


def _text_on_video_service_url() -> str | None:
    return "http://text-on-video:5083"


def _text_normalizer_service_url() -> str | None:
    return "http://translate:5098"


def _export_service_url() -> str | None:
    return "http://metadata-export:5096"


def _extract_audio_via_ffmpeg_service(
    *,
    service_url: str,
    video_path: Path,
    filename: str,
    sample_rate: int = 16000,
    channels: int = 1,
    fmt: str = "flac",
) -> bytes:
    endpoint = f"{service_url}/extract-audio"
    with video_path.open("rb") as handle:
        files = {"video": (filename, handle, "application/octet-stream")}
        data = {
            "sample_rate": str(sample_rate),
            "channels": str(channels),
            "format": fmt,
        }
        resp = requests.post(endpoint, files=files, data=data, timeout=900)
    if resp.status_code >= 400:
        raise RuntimeError(f"FFmpeg audio extract failed ({resp.status_code}): {resp.text}")
    return resp.content


def _extract_frame_via_ffmpeg_service(
    *,
    service_url: str,
    video_path: Path,
    filename: str,
    timestamp: float,
    scale: int = 224,
    quality: int = 3,
) -> bytes:
    endpoint = f"{service_url}/extract-frame"
    with video_path.open("rb") as handle:
        files = {"video": (filename, handle, "application/octet-stream")}
        data = {
            "timestamp": f"{max(0.0, float(timestamp)):.3f}",
            "scale": str(scale),
            "quality": str(quality),
        }
        resp = requests.post(endpoint, files=files, data=data, timeout=300)
    if resp.status_code >= 400:
        raise RuntimeError(f"FFmpeg frame extract failed ({resp.status_code}): {resp.text}")
    return resp.content


def _fetch_ocr_via_text_on_video(
    *,
    service_url: str,
    video_path: Path,
    interval_seconds: float,
    max_frames: int,
    job_id: str,
) -> dict[str, Any]:
    endpoint = f"{service_url}/ocr"
    payload = {
        "interval_seconds": float(interval_seconds),
        "max_frames": int(max_frames),
        "job_id": job_id,
        "output_kind": "processed_local",
    }
    resp = requests.post(endpoint, json=payload, timeout=600)
    if resp.status_code >= 400:
        raise RuntimeError(f"Text-on-video service failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("Invalid text-on-video service response")
    return data


def _moderation_service_url() -> str:
    return "http://moderation:5081"



def _frame_extractor_url() -> str | None:
    return "http://text-on-video:5083"


def _parse_allowed_gcs_video_source(raw: str, *, enforce_prefix: bool | None = None) -> Tuple[str, str]:
    v = (raw or "").strip()
    if not v:
        raise ValueError("Missing gcs_object (or gcs_uri)")

    bucket = _gcs_bucket_name()
    allowed = _allowed_gcs_buckets()
    prefix = _gcs_rawvideo_prefix()
    if enforce_prefix is None:
        enforce_prefix = _gcs_enforce_prefix()

    if v.lower().startswith("gs://"):
        parts = v.split("/", 3)
        if len(parts) < 4 or not parts[2] or not parts[3]:
            raise ValueError("Invalid gs:// URI")
        uri_bucket = parts[2]
        obj = parts[3]
        if allowed is not None and uri_bucket not in allowed:
            raise ValueError(f"Bucket not allowed: {uri_bucket}. Allowed: {', '.join(sorted(allowed))}")
        bucket = uri_bucket
    else:
        obj = v.lstrip("/")

    if enforce_prefix and not obj.startswith(prefix):
        raise ValueError(f"Object must be under {prefix}")
    if obj.endswith("/"):
        raise ValueError("Object cannot be a folder")

    ext = (Path(obj).suffix or "").lower()
    if ext not in {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mxf"}:
        raise ValueError("Only video files are allowed (.mp4, .mov, .m4v, .mkv, .avi, .webm, .mxf)")
    return bucket, obj


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    raw = (uri or "").strip()
    if not raw.lower().startswith("gs://"):
        raise ValueError("Invalid gs:// URI")
    parts = raw.split("/", 3)
    if len(parts) < 4 or not parts[2] or not parts[3]:
        raise ValueError("Invalid gs:// URI")
    return parts[2], parts[3]


def _parse_any_gcs_video_source(raw: str) -> Tuple[str, str]:
    v = (raw or "").strip()
    if not v:
        raise ValueError("Missing gcs_object (or gcs_uri)")

    bucket = _gcs_bucket_name()
    if v.lower().startswith("gs://"):
        parts = v.split("/", 3)
        if len(parts) < 4 or not parts[2] or not parts[3]:
            raise ValueError("Invalid gs:// URI")
        bucket = parts[2]
        obj = parts[3]
    else:
        obj = v.lstrip("/")

    if obj.endswith("/"):
        raise ValueError("Object cannot be a folder")

    ext = (Path(obj).suffix or "").lower()
    if ext not in {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mxf"}:
        raise ValueError("Only video files are allowed (.mp4, .mov, .m4v, .mkv, .avi, .webm, .mxf)")
    return bucket, obj


def _looks_like_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except Exception:
        return False


def _looks_like_short_job_id(value: str) -> bool:
    v = str(value or "").strip()
    if not v.isdigit() or len(v) != 6:
        return False
    try:
        return int(v) >= 100001
    except Exception:
        return False


def _looks_like_job_id(value: str) -> bool:
    return _looks_like_uuid(value) or _looks_like_short_job_id(value)


def _db_next_job_id() -> int | None:
    db = _db_pool()
    if db is None:
        return None
    try:
        doc = db["counters"].find_one_and_update(
            {"_id": "job_id"},
            {"$inc": {"seq": 1}, "$setOnInsert": {"seq": 100000}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        if isinstance(doc, dict):
            seq = doc.get("seq")
            if isinstance(seq, int):
                return max(seq, 100001)
    except Exception as exc:
        try:
            app.logger.warning("Failed to generate job id: %s", exc)
        except Exception:
            pass
    return None


def _gcs_next_job_id() -> int | None:
    try:
        counter_path = _gcs_job_id_counter_path()
        counter_path.parent.mkdir(parents=True, exist_ok=True)
        with open(counter_path, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            raw = handle.read().strip()
            current = int(raw) if raw.isdigit() else 100000
            next_id = max(current + 1, 100001)
            handle.seek(0)
            handle.truncate()
            handle.write(str(next_id))
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except Exception:
                pass
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return next_id
    except Exception as exc:
        try:
            app.logger.warning("Failed to generate job id from GCS counter: %s", exc)
        except Exception:
            pass
        return None


def _next_job_id() -> str:
    seq = _db_next_job_id()
    if seq is not None:
        return f"{seq:06d}"
    seq = _gcs_next_job_id()
    if seq is not None:
        return f"{seq:06d}"
    global JOB_ID_COUNTER
    with JOB_ID_LOCK:
        JOB_ID_COUNTER += 1
        return f"{JOB_ID_COUNTER:06d}"


def _persist_local_video_copy() -> bool:
    return _env_truthy(os.getenv("ENVID_METADATA_PERSIST_LOCAL_VIDEO"), default=True)


def _format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round((seconds - int(seconds)) * 1000.0))
    total = int(seconds)
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    idx = 1
    for seg in segments:
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        out.append(str(idx))
        out.append(f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}")
        out.append(text)
        out.append("")
        idx += 1
    return "\n".join(out).strip() + "\n"


def _segments_to_vtt(segments: List[Dict[str, Any]]) -> str:
    def _fmt(seconds: float) -> str:
        return _format_srt_timestamp(seconds).replace(",", ".")

    out: List[str] = ["WEBVTT", ""]
    for seg in segments:
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        out.append(f"{_fmt(start)} --> {_fmt(end)}")
        out.append(text)
        out.append("")
    return "\n".join(out).strip() + "\n"


SERVICE_DIR = Path(__file__).parent
STORAGE_BASE_DIR = Path(
    (os.getenv("ENVID_STORAGE_BASE_DIR") or os.getenv("ENVID_GCS_MOUNT_PATH") or "/mnt/gcs/envid-metadata").strip()
)
DOCUMENTS_DIR = STORAGE_BASE_DIR / "documents"
VIDEOS_DIR = STORAGE_BASE_DIR / "videos"
INDICES_DIR = STORAGE_BASE_DIR / "indices"
SUBTITLES_DIR = INDICES_DIR / "subtitles"

for d in [DOCUMENTS_DIR, VIDEOS_DIR, INDICES_DIR, SUBTITLES_DIR]:
    d.mkdir(exist_ok=True)

VIDEO_INDEX_FILE = INDICES_DIR / "video_index.json"
DOCUMENT_INDEX_FILE = INDICES_DIR / "document_index.json"


VIDEO_INDEX_LOCK = threading.Lock()
VIDEO_INDEX: List[Dict[str, Any]] = []
DOCUMENT_INDEX_LOCK = threading.Lock()
DOCUMENT_INDEX: List[Dict[str, Any]] = []


def _safe_json_load(path: Path, default_value: Any) -> Any:
    try:
        if not path.exists():
            return default_value
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_value


def _safe_json_save(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, default=str, separators=(",", ":"))
    os.replace(tmp, path)


def _load_indices() -> None:
    global VIDEO_INDEX, DOCUMENT_INDEX
    VIDEO_INDEX = _safe_json_load(VIDEO_INDEX_FILE, [])
    if not isinstance(VIDEO_INDEX, list):
        VIDEO_INDEX = []
    if not VIDEO_INDEX:
        gcs_videos = _load_video_index_from_gcs()
        if gcs_videos:
            VIDEO_INDEX = gcs_videos
            _safe_json_save(VIDEO_INDEX_FILE, VIDEO_INDEX)
    DOCUMENT_INDEX = _safe_json_load(DOCUMENT_INDEX_FILE, [])
    if not isinstance(DOCUMENT_INDEX, list):
        DOCUMENT_INDEX = []


def _save_video_index() -> None:
    with VIDEO_INDEX_LOCK:
        snapshot = list(VIDEO_INDEX)
    _safe_json_save(VIDEO_INDEX_FILE, snapshot)
    _upload_video_index_to_gcs()


def _save_document_index() -> None:
    with DOCUMENT_INDEX_LOCK:
        _safe_json_save(DOCUMENT_INDEX_FILE, DOCUMENT_INDEX)


def _history_index_gcs_location() -> tuple[str, str]:
    bucket = _gcs_artifacts_bucket(_gcs_bucket_name())
    prefix = _gcs_artifacts_prefix()
    obj = f"{prefix}/history/video_index.json".strip("/")
    return bucket, obj


def _upload_video_index_to_gcs() -> None:
    if _env_truthy(os.getenv("ENVID_DISABLE_GCS_INDEX_UPLOAD"), default=False):
        return
    try:
        bucket, obj = _history_index_gcs_location()
        payload = json.dumps(VIDEO_INDEX, indent=2, ensure_ascii=False, default=str)
        _gcs_client().bucket(bucket).blob(obj).upload_from_string(
            payload,
            content_type="application/json",
            timeout=60,
        )
    except Exception as exc:
        try:
            app.logger.warning("Failed to upload video history index to GCS: %s", exc)
        except Exception:
            pass


def _load_video_index_from_gcs() -> list[dict[str, Any]] | None:
    try:
        bucket, obj = _history_index_gcs_location()
        client = _gcs_client()
        blob = client.bucket(bucket).blob(obj)
        if not blob.exists(client):
            return None
        raw = blob.download_as_bytes()
        parsed = json.loads(raw.decode("utf-8"))
        if isinstance(parsed, list):
            return parsed
    except Exception as exc:
        try:
            app.logger.warning("Failed to load video history index from GCS: %s", exc)
        except Exception:
            pass
    return None


def _subtitles_local_path(video_id: str, *, lang: str, fmt: str) -> Path:
    vid = str(video_id or "").strip()
    lang_norm = (lang or "orig").strip().lower() or "orig"
    fmt_norm = (fmt or "vtt").strip().lower().lstrip(".")
    if fmt_norm not in {"srt", "vtt"}:
        fmt_norm = "vtt"
    name = f"{vid}.{lang_norm}.{fmt_norm}" if lang_norm != "orig" else f"{vid}.{fmt_norm}"
    return SUBTITLES_DIR / name


JOBS_LOCK = threading.Lock()
JOBS: Dict[str, Dict[str, Any]] = {}


class StopJob(RuntimeError):
    pass
JOB_ID_LOCK = threading.Lock()
JOB_ID_COUNTER = 100000

_LANGTOOL_LOCAL: Any | None = None
_LANGTOOL_LOCAL_LANG: str | None = None
_LANG_DICTIONARY_WORDS: dict[str, list[str]] = {}
_LANG_DICTIONARY_SET: dict[str, set[str]] = {}
_LANG_CONFUSION_MAP: dict[str, dict[str, str]] = {}


def _text_normalizer_url() -> str | None:
    url = (os.getenv("ENVID_TEXT_NORMALIZER_URL") or "http://translate:5098").strip()
    return url or None


def _text_normalizer_normalize_segment(
    *,
    text: str,
    language_code: str | None,
    grammar_enabled: bool,
    dictionary_enabled: bool,
    punctuation_enabled: bool,
    nlp_mode: str | None,
) -> tuple[str, dict[str, Any]] | None:
    url = _text_normalizer_url()
    if not url:
        return None
    endpoint = url.rstrip("/") + "/normalize/segment"
    timeout_s = _safe_float(os.getenv("ENVID_TRANSCRIPT_TEXT_NORMALIZER_TIMEOUT_SECONDS"), 12.0)
    payload = {
        "text": text,
        "language_code": language_code,
        "grammar_enabled": bool(grammar_enabled),
        "dictionary_enabled": bool(dictionary_enabled),
        "punctuation_enabled": bool(punctuation_enabled),
        "nlp_mode": (nlp_mode or "").strip().lower(),
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout_s)
        if resp.status_code >= 400:
            return None
        data = resp.json()
        if not isinstance(data, dict):
            return None
        out = str(data.get("text") or "").strip()
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        meta["source"] = "text-normalizer"
        return out, meta
    except Exception:
        return None


def _languagetool_local_check(*, text: str, language: str) -> dict[str, Any] | None:
    if language_tool_python is None:
        return None

    global _LANGTOOL_LOCAL, _LANGTOOL_LOCAL_LANG
    try:
        lang = (language or "auto").strip() or "auto"
        if _LANGTOOL_LOCAL is None or _LANGTOOL_LOCAL_LANG != lang:
            # LanguageTool runs locally (open-source). Requires Java.
            _LANGTOOL_LOCAL = language_tool_python.LanguageTool(lang)
            _LANGTOOL_LOCAL_LANG = lang
        matches = _LANGTOOL_LOCAL.check(text)
    except Exception:
        return None

    match_dicts: list[dict[str, Any]] = []
    for m in matches[:250]:
        try:
            repls = [{"value": r} for r in (m.replacements or [])[:5]]
        except Exception:
            repls = []
        match_dicts.append(
            {
                "offset": int(getattr(m, "offset", 0) or 0),
                "length": int(getattr(m, "errorLength", 0) or 0),
                "replacements": repls,
                "message": str(getattr(m, "message", "") or "")[:240],
                "rule": {"id": str(getattr(getattr(m, "rule", None), "id", "") or "")},
            }
        )

    return {"matches": match_dicts}


def _languagetool_remote_check(*, text: str, language: str) -> dict[str, Any] | None:
    url = "http://translate:8010"

    if url.endswith("/"):
        url = url[:-1]
    if not url.endswith("/v2/check") and not url.endswith("/check"):
        url = f"{url}/v2/check"

    lang = (language or "auto").strip() or "auto"
    timeout_s = _safe_float(os.getenv("ENVID_GRAMMAR_CORRECTION_TIMEOUT_SECONDS"), 10.0)
    try:
        resp = requests.post(url, data={"text": text, "language": lang}, timeout=timeout_s)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    return data if isinstance(data, dict) else None


def _apply_languagetool_matches(text: str, matches: list[dict[str, Any]]) -> str:
    if not matches:
        return text

    # Apply replacements from the end to preserve offsets.
    out = text
    ordered = sorted(matches, key=lambda m: int(m.get("offset", 0) or 0), reverse=True)
    for m in ordered:
        try:
            offset = int(m.get("offset", 0) or 0)
            length = int(m.get("length", 0) or 0)
            repls = m.get("replacements") or []
            replacement = ""
            if repls:
                rep0 = repls[0]
                if isinstance(rep0, dict):
                    replacement = str(rep0.get("value") or "")
                else:
                    replacement = str(rep0 or "")
            if length <= 0:
                continue
            out = out[:offset] + replacement + out[offset + length :]
        except Exception:
            continue
    return out


def _languagetool_correct_text(*, text: str, language: str | None) -> tuple[str, dict[str, Any]]:
    if not text:
        return "", {"matches": [], "source": "none"}

    lang = (language or "auto").strip() or "auto"
    data = _languagetool_remote_check(text=text, language=lang)
    source = "remote"
    if data is None:
        data = _languagetool_local_check(text=text, language=lang)
        source = "local"

    if not isinstance(data, dict):
        return text, {"matches": [], "source": "none"}

    matches = data.get("matches")
    if not isinstance(matches, list):
        matches = []

    corrected = _apply_languagetool_matches(text, matches)
    return corrected, {"matches": matches, "source": source}


def _is_hindi_language(language_code: str | None) -> bool:
    if not language_code:
        return False
    code = _normalize_language_code(language_code)
    return code == "hi"


def _normalize_language_code(language_code: str | None) -> str:
    if not language_code:
        return ""
    code = str(language_code).strip().lower()
    if code in {"hindi", "hin"}:
        return "hi"
    if code in {"urdu", "urd"}:
        return "ur"
    if code.startswith("zh"):
        return "zh"
    if code.startswith("ja"):
        return "ja"
    if code.startswith("ko"):
        return "ko"
    for sep in ("-", "_"):
        if sep in code:
            return code.split(sep, 1)[0]
    return code


def _dictionary_languages_enabled() -> set[str]:
    langs = {
        "hi",
        "bn",
        "ta",
        "te",
        "ml",
        "mr",
        "gu",
        "kn",
        "pa",
        "ur",
        "ne",
        "as",
        "or",
        "sa",
        "kok",
        "mai",
        "bho",
        "doi",
        "mni",
        "sat",
        "ks",
        "si",
        "th",
        "id",
        "ms",
        "vi",
        "zh",
        "ja",
        "ko",
        "ar",
    }
    return langs


def _dictionary_language_name(lang: str) -> str:
    names = {
        "hi": "Hindi",
        "bn": "Bengali",
        "ta": "Tamil",
        "te": "Telugu",
        "ml": "Malayalam",
        "mr": "Marathi",
        "gu": "Gujarati",
        "kn": "Kannada",
        "pa": "Punjabi",
        "ur": "Urdu",
        "ne": "Nepali",
        "as": "Assamese",
        "or": "Odia",
        "sa": "Sanskrit",
        "kok": "Konkani",
        "mai": "Maithili",
        "bho": "Bhojpuri",
        "doi": "Dogri",
        "mni": "Manipuri (Meitei)",
        "sat": "Santali",
        "ks": "Kashmiri",
        "si": "Sinhala",
        "th": "Thai",
        "id": "Indonesian",
        "ms": "Malay",
        "vi": "Vietnamese",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "ar": "Arabic",
    }
    return names.get(lang, lang)


def _language_word_pattern(lang: str) -> str:
    if lang == "hi":
        return r"[\u0900-\u097F]+"
    if lang == "ar":
        return r"[\u0600-\u06FF]+"
    if lang in {"zh", "ja", "ko"}:
        return r"[\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]+"
    return r"[^\W\d_]+"


def _load_language_dictionary(lang: str) -> tuple[list[str], set[str]]:
    if lang in _LANG_DICTIONARY_WORDS and lang in _LANG_DICTIONARY_SET:
        return _LANG_DICTIONARY_WORDS[lang], _LANG_DICTIONARY_SET[lang]

    words: list[str] = []
    raw_paths = (os.getenv("ENVID_DICTIONARY_PATHS_JSON") or "").strip()
    path = ""
    if raw_paths:
        try:
            parsed = json.loads(raw_paths)
            if isinstance(parsed, dict):
                path = str(parsed.get(lang) or "").strip()
        except Exception:
            path = ""

    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if not w or w.startswith("#"):
                        continue
                    if lang == "hi" and not re.search(r"[\u0900-\u097F]", w):
                        continue
                    words.append(w)
        except Exception:
            words = []
    elif top_n_list is not None:
        max_words = 5000
        try:
            words = [w for w in top_n_list(lang, max_words) if w]
        except Exception:
            words = []

    _LANG_DICTIONARY_WORDS[lang] = words
    _LANG_DICTIONARY_SET[lang] = set(words)
    return words, set(words)


def _load_language_confusions(lang: str) -> dict[str, str]:
    if lang in _LANG_CONFUSION_MAP:
        return _LANG_CONFUSION_MAP[lang]

    conf: dict[str, str] = {"सा": "ऐसा"} if lang == "hi" else {}

    _LANG_CONFUSION_MAP[lang] = conf
    return conf


def _dictionary_correct_text(*, text: str, language_code: str | None) -> str:
    if zipf_frequency is None and rapid_process is None:
        return text

    words, wordset = _load_language_dictionary(lang)
    confusions = _load_language_confusions(lang)

    min_zipf = 2.5
    fuzz_cutoff = 92
    confusions_strict = False

    def _score(w: str) -> float:
        if zipf_frequency is None:
            return 0.0
        try:
            return float(zipf_frequency(w, lang))
        except Exception:
            return 0.0

    def _fix_word(w: str) -> str:
        if not w:
            return w
        if w in confusions:
            cand = confusions[w]
            if cand:
                if not confusions_strict:
                    return cand
                if _score(cand) >= (_score(w) + 0.4):
                    return cand

        if words and rapid_process is not None and rapid_fuzz is not None:
            if w in wordset:
                return w
            try:
                match = rapid_process.extractOne(w, words, scorer=rapid_fuzz.QRatio, score_cutoff=fuzz_cutoff)
            except Exception:
                match = None
            if match:
                cand = str(match[0])
                if cand and _score(cand) >= max(min_zipf, _score(w)):
                    return cand
        return w

    pattern = _language_word_pattern(lang)

    def _repl(m: re.Match) -> str:
        return _fix_word(m.group(0))

    return re.sub(pattern, _repl, text)


def _load_hindi_dictionary() -> tuple[list[str], set[str]]:
    return _load_language_dictionary("hi")


def _load_hindi_confusions() -> dict[str, str]:
    return _load_language_confusions("hi")


def _hindi_dictionary_correct_text(*, text: str, language_code: str | None) -> str:
    return _dictionary_correct_text(text=text, language_code=language_code)


def _job_steps_default() -> List[Dict[str, Any]]:
    return [
        {"id": "upload_to_cloud_storage", "label": "Ingest", "status": "not_started", "percent": 0, "message": None},
        {"id": "technical_metadata", "label": "Technical Metadata", "status": "not_started", "percent": 0, "message": None},
        {"id": "transcode_normalize", "label": "Transcode to Normalize @ 1.5 mbps", "status": "not_started", "percent": 0, "message": None},
        {"id": "label_detection", "label": "Label detection", "status": "not_started", "percent": 0, "message": None},
        {"id": "moderation", "label": "Moderation", "status": "not_started", "percent": 0, "message": None},
        {"id": "text_on_screen", "label": "Text on Screen", "status": "not_started", "percent": 0, "message": None},
        {"id": "key_scene_detection", "label": "Key scene & high point detection", "status": "not_started", "percent": 0, "message": None},
        {"id": "transcribe", "label": "Audio Transcription", "status": "not_started", "percent": 0, "message": None},
        {"id": "synopsis_generation", "label": "Synopsis Generation", "status": "not_started", "percent": 0, "message": None},
        {"id": "scene_by_scene_metadata", "label": "Scene by scene metadata", "status": "not_started", "percent": 0, "message": None},
        {"id": "translate_output", "label": "Translation of Metadata", "status": "not_started", "percent": 0, "message": None},
        {"id": "famous_location_detection", "label": "Famous location detection", "status": "not_started", "percent": 0, "message": None},
        {"id": "opening_closing_credit_detection", "label": "Opening/Closing credit detection", "status": "not_started", "percent": 0, "message": None},
        {"id": "celebrity_detection", "label": "Celebrity detection", "status": "not_started", "percent": 0, "message": None},
        {"id": "celebrity_bio_image", "label": "Celebrity bio & Image", "status": "not_started", "percent": 0, "message": None},
        {"id": "upload_artifacts", "label": "Upload artifacts", "status": "not_started", "percent": 0, "message": None},
        {"id": "save_as_json", "label": "Save as Json", "status": "not_started", "percent": 0, "message": None},
        {"id": "overall", "label": "Overall", "status": "not_started", "percent": 0, "message": None},
    ]


@dataclass(slots=True)
class OrchestratorInputs:
    job_id: str
    task_selection: Dict[str, Any]
    requested_models: Dict[str, Any]


@dataclass(slots=True)
class Selection:
    enable_label_detection: bool
    enable_text_on_screen: bool
    enable_moderation: bool
    enable_transcribe: bool
    enable_translate_output: bool
    enable_famous_locations: bool
    enable_scene_by_scene: bool
    enable_key_scene: bool
    enable_high_point: bool
    enable_synopsis_generation: bool
    enable_opening_closing: bool
    enable_celebrity_detection: bool
    enable_celebrity_bio_image: bool
    requested_label_model_raw: str
    requested_label_model: str
    requested_text_model: str
    requested_moderation_model: str
    requested_key_scene_model_raw: str
    requested_key_scene_model: str
    label_engine: str
    use_vi_label_detection: bool
    use_local_ocr: bool
    use_local_moderation: bool
    allow_moderation_fallback: bool
    local_moderation_url_override: str
    use_transnetv2_for_scenes: bool
    use_pyscenedetect_for_scenes: bool
    use_clip_cluster_for_key_scenes: bool
    want_shots: bool
    want_vi_shots: bool
    want_any_vi: bool


@dataclass(slots=True)
class PreflightResult:
    selection: Selection
    precheck: Dict[str, Any]


def _orchestrator_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def _orchestrator_str(value: Any, default: str) -> str:
    raw = str(value).strip() if value is not None else ""
    return raw or default


def _build_selection(inputs: OrchestratorInputs) -> Selection:
    sel = inputs.task_selection or {}
    requested = inputs.requested_models or {}

    partial_reprocess = _orchestrator_bool(sel.get("partial_reprocess"), False)

    enable_label_detection = _orchestrator_bool(sel.get("enable_label_detection"), False)
    enable_text_on_screen = _orchestrator_bool(sel.get("enable_text_on_screen"), False)
    enable_moderation = _orchestrator_bool(sel.get("enable_moderation"), False)
    enable_transcribe = _orchestrator_bool(sel.get("enable_transcribe"), False)
    enable_translate_output = _orchestrator_bool(sel.get("enable_translate_output"), False)
    enable_famous_locations = _orchestrator_bool(sel.get("enable_famous_locations"), False)
    enable_scene_by_scene = _orchestrator_bool(sel.get("enable_scene_by_scene"), False)
    enable_key_scene = _orchestrator_bool(sel.get("enable_key_scene"), False)
    enable_high_point = _orchestrator_bool(sel.get("enable_high_point"), False)
    enable_synopsis_generation = _orchestrator_bool(sel.get("enable_synopsis_generation"), False)
    enable_opening_closing = _orchestrator_bool(sel.get("enable_opening_closing"), False)
    enable_celebrity_detection = _orchestrator_bool(sel.get("enable_celebrity_detection"), False)
    enable_celebrity_bio_image = _orchestrator_bool(sel.get("enable_celebrity_bio_image"), False)

    # Always-on tasks (UI hides toggles for these) unless partial reprocess requested.
    if not partial_reprocess:
        enable_transcribe = True
        enable_synopsis_generation = True
        enable_text_on_screen = True
        enable_moderation = True
        enable_key_scene = True
        enable_scene_by_scene = True
        enable_high_point = True
        enable_translate_output = True

    requested_label_model_raw = _orchestrator_str(
        requested.get("label_detection_model") or sel.get("label_detection_model"), "auto"
    )
    requested_label_model = _orchestrator_str(sel.get("label_detection_model_normalized"), requested_label_model_raw)

    # Force transcription on for all runs.
    enable_transcribe = True

    requested_text_model = _orchestrator_str(requested.get("text_model") or sel.get("text_model"), "tesseract")
    requested_moderation_model = _orchestrator_str(
        requested.get("moderation_model") or sel.get("moderation_model"), "nudenet"
    )

    requested_key_scene_model_raw = _orchestrator_str(
        requested.get("key_scene_detection_model") or sel.get("key_scene_detection_model"),
        "transnetv2_clip_cluster",
    )
    requested_key_scene_model = _orchestrator_str(
        sel.get("key_scene_detection_model_normalized"), requested_key_scene_model_raw
    )

    label_engine = _orchestrator_str(sel.get("label_engine"), "gcp_video_intelligence")
    use_vi_label_detection = _orchestrator_bool(
        sel.get("use_vi_label_detection"), label_engine.lower() in {"gcp_video_intelligence", "vi"}
    )

    use_local_ocr = _orchestrator_bool(sel.get("use_local_ocr"), True)
    use_local_moderation = _orchestrator_bool(sel.get("use_local_moderation"), True)
    allow_moderation_fallback = _orchestrator_bool(sel.get("allow_moderation_fallback"), True)
    local_moderation_url_override = _orchestrator_str(sel.get("local_moderation_url_override"), "")

    use_transnetv2_for_scenes = True
    use_pyscenedetect_for_scenes = False
    use_clip_cluster_for_key_scenes = _orchestrator_bool(sel.get("use_clip_cluster_for_key_scenes"), True)

    want_vi_shots = False
    want_any_vi = _orchestrator_bool(sel.get("want_any_vi"), want_vi_shots)
    want_shots = _orchestrator_bool(
        sel.get("want_shots"),
        enable_scene_by_scene or enable_key_scene or enable_high_point,
    )

    return Selection(
        enable_label_detection=enable_label_detection,
        enable_text_on_screen=enable_text_on_screen,
        enable_moderation=enable_moderation,
        enable_transcribe=enable_transcribe,
        enable_translate_output=enable_translate_output,
        enable_famous_locations=enable_famous_locations,
        enable_scene_by_scene=enable_scene_by_scene,
        enable_key_scene=enable_key_scene,
        enable_high_point=enable_high_point,
        enable_synopsis_generation=enable_synopsis_generation,
        enable_opening_closing=enable_opening_closing,
        enable_celebrity_detection=enable_celebrity_detection,
        enable_celebrity_bio_image=enable_celebrity_bio_image,
        requested_label_model_raw=requested_label_model_raw,
        requested_label_model=requested_label_model,
        requested_text_model=requested_text_model,
        requested_moderation_model=requested_moderation_model,
        requested_key_scene_model_raw=requested_key_scene_model_raw,
        requested_key_scene_model=requested_key_scene_model,
        label_engine=label_engine,
        use_vi_label_detection=use_vi_label_detection,
        use_local_ocr=use_local_ocr,
        use_local_moderation=use_local_moderation,
        allow_moderation_fallback=allow_moderation_fallback,
        local_moderation_url_override=local_moderation_url_override,
        use_transnetv2_for_scenes=use_transnetv2_for_scenes,
        use_pyscenedetect_for_scenes=use_pyscenedetect_for_scenes,
        use_clip_cluster_for_key_scenes=use_clip_cluster_for_key_scenes,
        want_shots=want_shots,
        want_vi_shots=want_vi_shots,
        want_any_vi=want_any_vi,
    )


def orchestrate_preflight(
    *,
    inputs: OrchestratorInputs,
    precheck_models: Callable[..., Dict[str, Any]] | None = None,
    job_update: Callable[..., Any] | None = None,
    job_step_update: Callable[..., Any] | None = None,
) -> PreflightResult:
    selection = _build_selection(inputs)

    precheck: Dict[str, Any] = {}
    if precheck_models is not None:
        precheck = precheck_models(
            enable_transcribe=selection.enable_transcribe,
            enable_synopsis_generation=selection.enable_synopsis_generation,
            enable_label_detection=selection.enable_label_detection,
            enable_moderation=selection.enable_moderation,
            enable_text_on_screen=selection.enable_text_on_screen,
            enable_key_scene=selection.enable_key_scene,
            enable_scene_by_scene=selection.enable_scene_by_scene,
            enable_famous_locations=selection.enable_famous_locations,
            requested_key_scene_model=selection.requested_key_scene_model,
            requested_label_model=selection.requested_label_model,
            requested_text_model=selection.requested_text_model,
            requested_moderation_model=selection.requested_moderation_model,
            use_local_moderation=selection.use_local_moderation,
            allow_moderation_fallback=selection.allow_moderation_fallback,
            use_local_ocr=selection.use_local_ocr,
            want_any_vi=selection.want_any_vi,
            local_moderation_url_override=selection.local_moderation_url_override,
        )

    if job_update is not None:
        job_update(inputs.job_id, status="preflight", progress=0, message="Preflight completed")
    if job_step_update is not None:
        job_step_update(inputs.job_id, "preflight", status="completed", percent=100, message="Preflight completed")

    return PreflightResult(selection=selection, precheck=precheck)


def _db_config() -> Dict[str, Any]:
    return {
        "host": (os.getenv("ENVID_METADATA_DB_HOST") or "db").strip(),
        "port": int(os.getenv("ENVID_METADATA_DB_PORT") or "27017"),
        "database": (os.getenv("ENVID_METADATA_DB_NAME") or "envid_metadata").strip(),
        "user": (os.getenv("ENVID_METADATA_DB_USER") or "").strip(),
        "password": (os.getenv("ENVID_METADATA_DB_PASSWORD") or "").strip(),
    }


def _db_enabled() -> bool:
    cfg = _db_config()
    return bool(cfg["host"] and cfg["database"] and MongoClient)


def _db_init_schema(db: Any) -> None:
    if db is None:
        return
    try:
        db["jobs"].create_index([("id", ASCENDING)], unique=True)
        db["job_steps"].create_index([("job_id", ASCENDING), ("step_id", ASCENDING)], unique=True)
        db["job_outputs"].create_index([("job_id", ASCENDING), ("kind", ASCENDING)], unique=True)
    except Exception:
        pass


@lru_cache(maxsize=1)
def _db_pool() -> Any:
    if not _db_enabled():
        return None
    cfg = _db_config()
    kwargs: dict[str, Any] = {
        "host": cfg["host"],
        "port": int(cfg["port"]),
        "serverSelectionTimeoutMS": 3000,
    }
    if cfg.get("user") and cfg.get("password"):
        kwargs.update({"username": cfg["user"], "password": cfg["password"], "authSource": cfg["database"]})
    client = MongoClient(**kwargs)
    db = client[cfg["database"]]
    _db_init_schema(db)
    return db


def _db_fetch_all(query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    _ = (query, params)
    return []


def _db_fetch_one(query: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
    _ = (query, params)
    return None


def _db_job_upsert(job_id: str, **fields: Any) -> None:
    db = _db_pool()
    if db is None:
        return
    now = fields.get("updated_at") or datetime.utcnow()
    set_fields = {
        "id": job_id,
        "updated_at": now,
    }
    if fields.get("title") is not None:
        set_fields["title"] = fields.get("title")
    if fields.get("status") is not None:
        set_fields["status"] = fields.get("status")
    if fields.get("progress") is not None:
        set_fields["progress"] = fields.get("progress")
    if fields.get("message") is not None:
        set_fields["message"] = fields.get("message")
    if fields.get("gcs_video_uri") is not None:
        set_fields["gcs_video_uri"] = fields.get("gcs_video_uri")
    if fields.get("gcs_working_uri") is not None:
        set_fields["gcs_working_uri"] = fields.get("gcs_working_uri")
    if fields.get("temp_dir") is not None:
        set_fields["temp_dir"] = fields.get("temp_dir")
    if "stop_requested" in fields and fields.get("stop_requested") is not None:
        set_fields["stop_requested"] = fields.get("stop_requested")
    if "task_selection" in fields and fields.get("task_selection") is not None:
        set_fields["task_selection"] = fields.get("task_selection")
    if "task_selection_effective" in fields and fields.get("task_selection_effective") is not None:
        set_fields["task_selection_effective"] = fields.get("task_selection_effective")
    if "task_selection_requested_models" in fields and fields.get("task_selection_requested_models") is not None:
        set_fields["task_selection_requested_models"] = fields.get("task_selection_requested_models")
    db["jobs"].update_one(
        {"id": job_id},
        {
            "$set": set_fields,
            "$setOnInsert": {"created_at": fields.get("created_at") or now},
        },
        upsert=True,
    )


def _db_step_upsert(job_id: str, step_id: str, **fields: Any) -> None:
    db = _db_pool()
    if db is None:
        return
    now = fields.get("updated_at") or datetime.utcnow()
    set_fields = {
        "job_id": job_id,
        "step_id": step_id,
        "status": fields.get("status"),
        "percent": fields.get("percent"),
        "message": fields.get("message"),
        "started_at": fields.get("started_at"),
        "completed_at": fields.get("completed_at"),
        "updated_at": now,
    }
    db["job_steps"].update_one(
        {"job_id": job_id, "step_id": step_id},
        {
            "$set": set_fields,
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )


def _db_file_insert(job_id: str, *, kind: str, path: str | None, gcs_uri: str | None = None) -> None:
    db = _db_pool()
    if db is None:
        return
    now = datetime.utcnow()
    db["job_outputs"].update_one(
        {"job_id": job_id, "kind": kind},
        {
            "$set": {"path": path, "gcs_uri": gcs_uri, "payload": None, "updated_at": now},
            "$setOnInsert": {"job_id": job_id, "kind": kind, "created_at": now},
        },
        upsert=True,
    )


def _db_file_upsert(job_id: str, *, kind: str, path: str | None, gcs_uri: str | None = None) -> None:
    db = _db_pool()
    if db is None:
        return
    now = datetime.utcnow()
    db["job_outputs"].update_one(
        {"job_id": job_id, "kind": kind},
        {
            "$set": {"path": path, "gcs_uri": gcs_uri, "updated_at": now},
            "$setOnInsert": {"job_id": job_id, "kind": kind, "payload": None, "created_at": now},
        },
        upsert=True,
    )


def _db_payload_insert(job_id: str, *, kind: str, payload: Dict[str, Any]) -> None:
    db = _db_pool()
    if db is None:
        return
    now = datetime.utcnow()
    db["job_outputs"].update_one(
        {"job_id": job_id, "kind": kind},
        {
            "$set": {"payload": payload, "updated_at": now},
            "$setOnInsert": {"job_id": job_id, "kind": kind, "created_at": now},
        },
        upsert=True,
    )


def _db_delete_job(job_id: str) -> None:
    db = _db_pool()
    if db is None:
        return
    db["job_outputs"].delete_many({"job_id": job_id})
    db["job_steps"].delete_many({"job_id": job_id})
    db["jobs"].delete_many({"id": job_id})


def _db_get_job(job_id: str) -> dict[str, Any] | None:
    db = _db_pool()
    if db is None:
        return None
    row = db["jobs"].find_one({"id": job_id}, {"_id": 0})
    return dict(row) if isinstance(row, dict) else None


def _db_list_jobs(*, statuses: list[str] | None = None, limit: int = 50) -> list[dict[str, Any]]:
    db = _db_pool()
    if db is None:
        return []
    query: dict[str, Any] = {}
    if statuses:
        query["status"] = {"$in": statuses}
    rows = db["jobs"].find(query, {"_id": 0}).sort("updated_at", -1).limit(int(limit))
    return [dict(r) for r in rows if isinstance(r, dict)]


def _db_get_job_steps(job_id: str) -> list[dict[str, Any]]:
    db = _db_pool()
    if db is None:
        return []
    rows = db["job_steps"].find({"job_id": job_id}, {"_id": 0}).sort("updated_at", 1)
    return [dict(r) for r in rows if isinstance(r, dict)]


def _db_get_job_outputs(job_id: str) -> list[dict[str, Any]]:
    db = _db_pool()
    if db is None:
        return []
    rows = db["job_outputs"].find({"job_id": job_id}, {"_id": 0})
    return [dict(r) for r in rows if isinstance(r, dict)]


def _cleanup_job_artifacts(job_id: str, job_row: dict[str, Any] | None = None) -> None:
    row = job_row or _db_get_job(job_id) or {}
    temp_dir = str(row.get("temp_dir") or row.get("tempDir") or "").strip()
    if temp_dir:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    for out in _db_get_job_outputs(job_id):
        try:
            path = str(out.get("path") or "").strip()
            if path and Path(path).exists():
                Path(path).unlink()
        except Exception:
            pass

    for lang in ["orig", "en", "ar", "id"]:
        for fmt in ["srt", "vtt"]:
            try:
                p = _subtitles_local_path(job_id, lang=lang, fmt=fmt)
                if p.exists():
                    p.unlink()
            except Exception:
                pass


def _db_get_job_steps_for_jobs(job_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    db = _db_pool()
    if db is None or not job_ids:
        return {}
    rows = db["job_steps"].find({"job_id": {"$in": job_ids}}, {"_id": 0}).sort("updated_at", 1)
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        job_id = str(row.get("job_id") or "").strip()
        if not job_id:
            continue
        out.setdefault(job_id, []).append(dict(row))
    return out


def _db_get_job_output(job_id: str, kind: str) -> dict[str, Any] | None:
    db = _db_pool()
    if db is None:
        return None
    row = db["job_outputs"].find_one({"job_id": job_id, "kind": kind}, {"_id": 0})
    return dict(row) if isinstance(row, dict) else None


def _task_selection_for_failed_steps(failed_steps: set[str]) -> dict[str, Any]:
    sel: dict[str, Any] = {}

    if "label_detection" in failed_steps:
        sel["enable_label_detection"] = True
    if "text_on_screen" in failed_steps:
        sel["enable_text_on_screen"] = True
    if "moderation" in failed_steps:
        sel["enable_moderation"] = True
    if "transcribe" in failed_steps:
        sel["enable_transcribe"] = True
    if "famous_location_detection" in failed_steps:
        sel["enable_famous_locations"] = True
    if "scene_by_scene_metadata" in failed_steps:
        sel["enable_scene_by_scene"] = True
    if "key_scene_detection" in failed_steps:
        sel["enable_key_scene"] = True
        sel["enable_high_point"] = True
    if "synopsis_generation" in failed_steps:
        sel["enable_synopsis_generation"] = True
    if "opening_closing_credit_detection" in failed_steps:
        sel["enable_opening_closing"] = True
    if "celebrity_detection" in failed_steps:
        sel["enable_celebrity_detection"] = True
    if "celebrity_bio_image" in failed_steps:
        sel["enable_celebrity_bio_image"] = True
    if "translate_output" in failed_steps:
        sel["translate_targets"] = _translate_targets()

    return sel


def _job_init(job_id: str, *, title: str | None = None) -> None:
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "title": title,
            "status": "queued",
            "progress": 0,
            "message": None,
            "steps": _job_steps_default(),
            "stop_requested": False,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
    _db_job_upsert(
        job_id,
        title=title,
        status="queued",
        progress=0,
        message=None,
        stop_requested=False,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


def _job_stop_requested(job_id: str) -> bool:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job and job.get("stop_requested"):
            return True
    row = _db_get_job(job_id)
    return bool(row and row.get("stop_requested"))


def _job_update(job_id: str, **fields: Any) -> None:
    if not fields.get("stop_requested") and fields.get("status") not in {"stopping", "stopped"}:
        if _job_stop_requested(job_id):
            raise StopJob("Stop requested")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            job.update(fields)
            job["updated_at"] = datetime.utcnow().isoformat()
    db_fields: Dict[str, Any] = {}
    for key in ("task_selection", "task_selection_effective", "task_selection_requested_models"):
        if key in fields and fields.get(key) is not None:
            db_fields[key] = fields.get(key)
    _db_job_upsert(
        job_id,
        title=fields.get("title"),
        status=fields.get("status"),
        progress=fields.get("progress"),
        message=fields.get("message"),
        gcs_video_uri=fields.get("gcs_video_uri"),
        gcs_working_uri=fields.get("gcs_working_uri"),
        temp_dir=fields.get("temp_dir"),
        stop_requested=fields.get("stop_requested"),
        **db_fields,
        updated_at=datetime.utcnow(),
    )


def _job_step_update(job_id: str, step_id: str, *, status: str | None = None, percent: int | None = None, message: str | None = None) -> None:
    if status == "running" and _job_stop_requested(job_id):
        raise StopJob("Stop requested")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            steps = job.get("steps")
            if not isinstance(steps, list):
                steps = []
                job["steps"] = steps

            step_obj: Dict[str, Any] | None = None
            for s in steps:
                if isinstance(s, dict) and s.get("id") == step_id:
                    step_obj = s
                    break
            if step_obj is None:
                step_obj = {"id": step_id, "label": step_id, "status": "not_started", "percent": 0, "message": None}
                steps.append(step_obj)

            prev_status = str(step_obj.get("status") or "not_started")
            next_status = status or prev_status
            if next_status == "running" and not step_obj.get("started_at"):
                step_obj["started_at"] = datetime.utcnow().isoformat()
            if next_status in {"completed", "failed", "skipped"} and not step_obj.get("completed_at"):
                step_obj["completed_at"] = datetime.utcnow().isoformat()

            step_obj["status"] = next_status
            if percent is not None:
                step_obj["percent"] = max(0, min(100, int(percent)))
            if message is not None:
                step_obj["message"] = message

            job["updated_at"] = datetime.utcnow().isoformat()

    _db_step_upsert(
        job_id,
        step_id,
        status=status,
        percent=percent,
        message=message,
        started_at=datetime.utcnow() if status == "running" else None,
        completed_at=datetime.utcnow() if status in {"completed", "failed", "skipped"} else None,
        updated_at=datetime.utcnow(),
    )


def _parse_iso_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        raw = str(value).strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def _job_step_durations(job_id: str) -> tuple[dict[str, float], float]:
    with JOBS_LOCK:
        job = JOBS.get(job_id) or {}
    steps = job.get("steps") if isinstance(job, dict) else None
    if not isinstance(steps, list):
        return {}, 0.0

    durations: dict[str, float] = {}
    total = 0.0
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or "").strip()
        if not step_id:
            continue
        st = _parse_iso_ts(step.get("started_at"))
        en = _parse_iso_ts(step.get("completed_at"))
        if not st or not en:
            continue
        dur = (en - st).total_seconds()
        if dur <= 0:
            continue
        durations[step_id] = dur
        total += dur
    return durations, total


def _probe_technical_metadata(video_path: Path) -> dict[str, Any]:
    ffmpeg_service_url = _ffmpeg_service_url()
    if not ffmpeg_service_url:
        return {"available": False, "error": "FFmpeg service not configured"}
    try:
        return _probe_via_ffmpeg_service(service_url=ffmpeg_service_url, video_path=video_path, filename=video_path.name)
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _check_service_health(url: str, *, timeout_seconds: float = 3.0) -> dict[str, Any]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        data = json.loads(raw or "{}") if raw else {}
        ok = None
        if isinstance(data, dict):
            if "ok" in data:
                ok = bool(data.get("ok"))
            elif "status" in data:
                ok = str(data.get("status") or "").lower() in {"ok", "healthy"}
            elif "ready" in data:
                ok = bool(data.get("ready"))
        return {"ok": bool(ok) if ok is not None else True, "raw": data}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _precheck_models(
    *,
    enable_transcribe: bool,
    enable_synopsis_generation: bool,
    enable_label_detection: bool,
    enable_moderation: bool,
    enable_text_on_screen: bool,
    enable_key_scene: bool,
    enable_scene_by_scene: bool,
    enable_famous_locations: bool,
    requested_key_scene_model: str,
    requested_label_model: str,
    requested_text_model: str,
    requested_moderation_model: str,
    use_local_moderation: bool,
    allow_moderation_fallback: bool,
    use_local_ocr: bool,
    want_any_vi: bool,
    local_moderation_url_override: str,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    errors: list[str] = []
    warnings: list[str] = []
    summarizer_service_url = _summarizer_service_url()

    def _require(ok: bool, name: str, msg: str) -> None:
        checks[name] = {"ok": bool(ok), "message": msg}
        if not ok:
            errors.append(msg)

    def _warn(ok: bool, name: str, msg: str) -> None:
        checks[name] = {"ok": bool(ok), "message": msg}
        if not ok:
            warnings.append(msg)

    # ffmpeg/ffprobe availability (core pipeline tools)
    ffmpeg_service_url = _ffmpeg_service_url()
    if ffmpeg_service_url:
        health = _check_service_health(f"{ffmpeg_service_url}/health")
        _require(bool(health.get("ok")), "ffmpeg_service", f"FFmpeg service unhealthy: {health.get('error') or health.get('raw')}")
    else:
        _require(False, "ffmpeg_service", "FFmpeg service URL is not set")

    # Text normalizer for transcript corrections
    if enable_transcribe:
        text_norm_url = _text_normalizer_service_url()
        if text_norm_url:
            health = _check_service_health(f"{text_norm_url}/health")
            _require(bool(health.get("ok")), "text_normalizer", f"text normalizer unhealthy: {health.get('error') or health.get('raw')}")
        else:
            _require(False, "text_normalizer", "Text normalizer service URL is not set")

    # Audio transcription availability
    if enable_transcribe:
        _require(
            _transcribe_service_available(),
            "audio_transcription",
            "Audio transcription service is not available",
        )

    # Synopsis / LLM availability
    if enable_synopsis_generation:
        gemini_ok = bool(_gemini_api_key())
        local_ok = _local_llm_available()
        summarizer_ok = bool(summarizer_service_url)
        if not (summarizer_ok or gemini_ok or local_ok):
            _require(False, "synopsis", "No LLM available for synopsis generation")

    if enable_scene_by_scene and _env_truthy(os.getenv("ENVID_SCENE_BY_SCENE_LLM"), default=False):
        gemini_ok = bool(_gemini_api_key())
        local_ok = _local_llm_available()
        summarizer_ok = bool(summarizer_service_url)
        if not (summarizer_ok or gemini_ok or local_ok):
            _require(False, "synopsis", "No LLM available for scene-by-scene summaries")

    # Local moderation service (NudeNet only)
    if enable_moderation and use_local_moderation:
        service_url_default = _moderation_service_url()
        service_url = (local_moderation_url_override or service_url_default).strip()
        if service_url:
            health = _check_service_health(f"{service_url.rstrip('/')}/health")
            _require(bool(health.get("ok")), "local_moderation", f"local moderation unhealthy: {health.get('error') or health.get('raw')}")
    if enable_moderation and (not use_local_moderation or allow_moderation_fallback) and want_any_vi:
        _warn(False, "gcp_video_intelligence", "gcp_video_intelligence is not used for moderation")

    # Text-on-screen OCR (text-on-video service only)
    if enable_text_on_screen:
        text_on_video_url = _text_on_video_service_url()
        _require(bool(text_on_video_url), "text_on_video_service", "Text-on-video service URL is not set")
        if text_on_video_url:
            health = _check_service_health(f"{text_on_video_url.rstrip('/')}/health")
            _require(bool(health.get("ok")), "text_on_video_service", f"Text-on-video service unhealthy: {health.get('error') or health.get('raw')}")

    # Key scene sidecar (CLIP clustering optional, sidecar required for transnetv2 scenes)
    if enable_key_scene and requested_key_scene_model in {"transnetv2_clip_cluster", "pyscenedetect_clip_cluster", "clip_cluster"}:
        base = "http://keyscene:5085"
        health_timeout = float(os.getenv("ENVID_METADATA_LOCAL_KEYSCENE_HEALTH_TIMEOUT_SECONDS") or 10.0)
        health = _check_service_health(f"{base}/health", timeout_seconds=health_timeout)
        if not health.get("ok"):
            _require(False, "keyscene_sidecar", f"key scene sidecar unhealthy: {health.get('error') or health.get('raw')}")
        else:
            # If CLIP is degraded, only warn (optional clustering)
            raw = health.get("raw") if isinstance(health.get("raw"), dict) else {}
            clip_ok = bool(((raw or {}).get("details") or {}).get("clip", {}).get("ok")) if isinstance(raw, dict) else True
            if not clip_ok:
                _warn(False, "clip_model", "CLIP model unavailable")

    # LibreTranslate availability (when translation is enabled)
    enable_translate = _env_truthy(os.getenv("ENVID_METADATA_ENABLE_TRANSLATE"), default=True)
    translate_provider = _translate_provider()
    if enable_translate and translate_provider == "libretranslate":
        base_url = "http://translate:5000"
        try:
            langs = _libretranslate_languages_raw(base_url)
            _require(bool(langs), "libretranslate", "LibreTranslate /languages returned no data")
        except Exception as exc:
            _require(False, "libretranslate", f"LibreTranslate unavailable: {exc}")

    # GCP Language (locations)
    if enable_famous_locations and gcp_language is None:
        _require(False, "gcp_language", "google-cloud-language is not installed")

    return {"ok": not errors, "errors": errors, "warnings": warnings, "checks": checks}


def _normalize_segments_for_ui(segments: Any, *, fps: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(segments, list):
        return out
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start = seg.get("start_seconds")
        if start is None:
            start = seg.get("start")
        end = seg.get("end_seconds")
        if end is None:
            end = seg.get("end")
        try:
            start_s = float(start)
        except Exception:
            start_s = 0.0
        try:
            end_s = float(end)
        except Exception:
            end_s = start_s
        if end_s < start_s:
            end_s = start_s
        item: dict[str, Any] = {"start_seconds": start_s, "end_seconds": end_s}
        if fps:
            item["start_timecode"] = _seconds_to_timecode(start_s, fps=int(fps))
            item["end_timecode"] = _seconds_to_timecode(end_s, fps=int(fps))
        if "confidence" in seg and seg.get("confidence") is not None:
            item["confidence"] = seg.get("confidence")
        out.append(item)
    return out


def _first_last_from_segments(segments: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    if not segments:
        return None, None
    starts: list[float] = []
    ends: list[float] = []
    for seg in segments:
        try:
            starts.append(float(seg.get("start_seconds") or 0.0))
        except Exception:
            pass
        try:
            ends.append(float(seg.get("end_seconds") or 0.0))
        except Exception:
            pass
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def _normalize_video_intelligence_to_detected_content(
    *, video_intel: dict[str, Any], technical: dict[str, Any]
) -> dict[str, Any]:
    """Normalize GCP video intelligence output into the UI's detected_content shape.

    UI expects:
    - labels: [{name, segments:[{start_seconds,end_seconds}], first_seen_seconds?, last_seen_seconds?}, ...]
    - on_screen_text: [{text, segments:[...], first_seen_seconds?, last_seen_seconds?}, ...]
    - moderation: [{name, segments:[...], first_seen_seconds?, last_seen_seconds?}, ...]
    """

    out: dict[str, Any] = {}

    fps = _video_fps_from_ffprobe(technical)

    labels_in = video_intel.get("labels")
    if isinstance(labels_in, list) and labels_in:
        labels_out: list[dict[str, Any]] = []
        for it in labels_in:
            if not isinstance(it, dict):
                continue
            name = (it.get("name") or it.get("label") or "").strip() if isinstance(it.get("name") or it.get("label"), str) else None
            segments = _normalize_segments_for_ui(it.get("segments"), fps=fps)
            first_seen, last_seen = _first_last_from_segments(segments)
            item: dict[str, Any] = {"name": name, "segments": segments}
            if first_seen is not None and last_seen is not None:
                item["first_seen_seconds"] = first_seen
                item["last_seen_seconds"] = last_seen
            if isinstance(it.get("categories"), list) and it.get("categories"):
                item["categories"] = it.get("categories")
            labels_out.append(item)
        if labels_out:
            out["labels"] = labels_out

    # Optional: treat tracked objects/logos/people as additional labels for UI purposes.
    extra_labels: list[dict[str, Any]] = []
    objects_in = video_intel.get("objects")
    if isinstance(objects_in, list) and objects_in:
        for it in objects_in:
            if not isinstance(it, dict):
                continue
            name = (it.get("name") or it.get("label") or "").strip() if isinstance(it.get("name") or it.get("label"), str) else None
            segments = _normalize_segments_for_ui(it.get("segments"), fps=fps)
            if not name or not segments:
                continue
            first_seen, last_seen = _first_last_from_segments(segments)
            item: dict[str, Any] = {"name": name, "segments": segments, "source": "objects"}
            if first_seen is not None and last_seen is not None:
                item["first_seen_seconds"] = first_seen
                item["last_seen_seconds"] = last_seen
            extra_labels.append(item)

    logos_in = video_intel.get("logos")
    if isinstance(logos_in, list) and logos_in:
        for it in logos_in:
            if not isinstance(it, dict):
                continue
            name = (it.get("name") or it.get("label") or "").strip() if isinstance(it.get("name") or it.get("label"), str) else None
            segments = _normalize_segments_for_ui(it.get("segments"))
            if not name or not segments:
                continue
            first_seen, last_seen = _first_last_from_segments(segments)
            item = {"name": name, "segments": segments, "source": "logos"}
            if first_seen is not None and last_seen is not None:
                item["first_seen_seconds"] = first_seen
                item["last_seen_seconds"] = last_seen
            extra_labels.append(item)

    people_in = video_intel.get("people")
    if isinstance(people_in, list) and people_in:
        for it in people_in:
            if not isinstance(it, dict):
                continue
            name = (it.get("name") or it.get("label") or "person").strip() if isinstance(it.get("name") or it.get("label") or "person", str) else "person"
            segments = _normalize_segments_for_ui(it.get("segments"), fps=fps)
            if not segments:
                continue
            first_seen, last_seen = _first_last_from_segments(segments)
            item = {"name": name, "segments": segments, "source": "people"}
            if first_seen is not None and last_seen is not None:
                item["first_seen_seconds"] = first_seen
                item["last_seen_seconds"] = last_seen
            extra_labels.append(item)

    if extra_labels:
        existing = out.get("labels")
        if isinstance(existing, list):
            out["labels"] = existing + extra_labels
        else:
            out["labels"] = extra_labels

    text_in = video_intel.get("text")
    if isinstance(text_in, list) and text_in:
        text_out: list[dict[str, Any]] = []
        for it in text_in:
            if not isinstance(it, dict):
                continue
            text = (it.get("text") or "").strip() if isinstance(it.get("text"), str) else None
            segments = _normalize_segments_for_ui(it.get("segments"), fps=fps)
            first_seen, last_seen = _first_last_from_segments(segments)
            item = {"text": text, "segments": segments}
            if first_seen is not None and last_seen is not None:
                item["first_seen_seconds"] = first_seen
                item["last_seen_seconds"] = last_seen
            text_out.append(item)
        if text_out:
            # UI variants expect either `text` or `on_screen_text`.
            # Emit both to avoid "blank" content in the frontend.
            out["text"] = text_out
            out["on_screen_text"] = text_out

    moderation_in = video_intel.get("moderation")
    if isinstance(moderation_in, dict):
        frames = moderation_in.get("explicit_frames")
        if isinstance(frames, list) and frames:
            duration = _video_duration_seconds_from_ffprobe(technical) or None
            frame_interval = _parse_int(os.getenv("GCP_EXPLICIT_FRAME_INTERVAL_SECONDS"), default=1, min_value=0, max_value=10)
            frame_len = float(frame_interval) if frame_interval > 0 else 0.5

            groups: dict[str, list[float]] = {}
            for f in frames:
                if not isinstance(f, dict):
                    continue
                severity = str(f.get("severity") or "")
                if not severity:
                    severity = _severity_from_likelihood(str(f.get("provider_likelihood") or f.get("likelihood") or ""))
                if not severity:
                    severity = "unknown"
                try:
                    t = float(f.get("time") or 0.0)
                except Exception:
                    t = 0.0
                groups.setdefault(severity, []).append(t)

            moderation_out: list[dict[str, Any]] = []
            for severity, times in sorted(groups.items(), key=lambda kv: kv[0]):
                segments: list[dict[str, Any]] = []
                for t in sorted(set(times)):
                    end = t + frame_len
                    if duration is not None:
                        end = min(end, duration)
                    segments.append({"start_seconds": t, "end_seconds": end})
                first_seen, last_seen = _first_last_from_segments(segments)
                item = {"name": f"explicit_severity_{severity}", "segments": segments}
                if first_seen is not None and last_seen is not None:
                    item["first_seen_seconds"] = first_seen
                    item["last_seen_seconds"] = last_seen
                moderation_out.append(item)

            if moderation_out:
                out["moderation"] = moderation_out

    return out


def _build_categorized_metadata_json(video: dict[str, Any]) -> dict[str, Any]:
    categories: dict[str, Any] = {}
    combined: dict[str, Any] = {}
    detected: dict[str, Any] = {}

    synopsis = video.get("synopsis") or {}
    if synopsis:
        categories.setdefault("synopsis", synopsis)
        combined["synopsis"] = synopsis

    translations = video.get("translations") or {}
    if translations:
        categories.setdefault("translations", translations)
        combined["translations"] = translations

    subs = video.get("subtitles") or {}
    if subs:
        categories.setdefault("subtitles", subs)
        combined["subtitles"] = subs

    locations = video.get("locations") or []
    if locations:
        categories.setdefault("famous_locations", {"locations": locations})
        combined["famous_locations"] = {"locations": locations}

    technical = video.get("technical_ffprobe") or {}
    categories.setdefault("technical", technical)
    categories.setdefault("technical_metadata", technical)
    combined["technical"] = technical
    combined["technical_metadata"] = technical

    transcript = (video.get("transcript") or "").strip()
    transcript_script = (video.get("transcript_script") or "").strip()
    transcript_raw = (video.get("transcript_raw") or "").strip()
    transcript_raw_script = (video.get("transcript_raw_script") or "").strip()
    transcript_raw_segments = video.get("transcript_raw_segments")
    if not isinstance(transcript_raw_segments, list):
        transcript_raw_segments = []
    transcript_segments = video.get("transcript_segments")
    if not isinstance(transcript_segments, list):
        transcript_segments = []
    if transcript or transcript_segments or transcript_script:
        payload = {
            "text": transcript,
            "script": transcript_script,
            "language_code": video.get("language_code"),
            "segments": transcript_segments,
            "meta": video.get("transcript_meta") or {},
            "raw_text": transcript_raw,
            "raw_script": transcript_raw_script,
            "raw_segments": transcript_raw_segments,
        }
        categories.setdefault("transcript", payload)
        combined["transcript"] = payload

    video_intel = video.get("video_intelligence") or {}
    if video_intel:
        categories.setdefault("video_intelligence", video_intel)
        combined["video_intelligence"] = video_intel

        detected = _normalize_video_intelligence_to_detected_content(video_intel=video_intel, technical=technical)
        if detected:
            categories.setdefault("detected_content", detected)
            combined["detected_content"] = detected

    # Key scenes / high points (derived).
    key_scenes = video.get("key_scenes")
    if isinstance(key_scenes, list) and key_scenes:
        categories.setdefault("key_scenes", key_scenes)
        combined["key_scenes"] = key_scenes

    high_points = video.get("high_points")
    if isinstance(high_points, list) and high_points:
        categories.setdefault("high_points", high_points)
        combined["high_points"] = high_points

    scenes = video.get("scenes")
    if isinstance(scenes, list) and scenes:
        fps = _video_fps_from_ffprobe(video.get("technical_ffprobe") or {})
        scene_out: list[dict[str, Any]] = []
        labels_src = detected.get("labels") if isinstance(detected, dict) else None
        if not isinstance(labels_src, list):
            labels_src = []
        objects_src = detected.get("objects") if isinstance(detected, dict) else None
        if not isinstance(objects_src, list):
            objects_src = []
        transcript_segments = video.get("transcript_segments")
        if not isinstance(transcript_segments, list):
            transcript_segments = []

        for idx, sc in enumerate(scenes):
            if not isinstance(sc, dict):
                continue
            try:
                st = float(sc.get("start") or sc.get("start_seconds") or 0.0)
                en = float(sc.get("end") or sc.get("end_seconds") or st)
            except Exception:
                continue
            if en < st:
                st, en = en, st

            scene_index = int(sc.get("index") or idx)
            item = {
                "index": scene_index,
                "scene_index": scene_index,
                "start_seconds": st,
                "end_seconds": en,
                "start_timecode": _seconds_to_timecode(st, fps=fps),
                "end_timecode": _seconds_to_timecode(en, fps=fps),
                "duration_seconds": max(0.0, en - st),
            }

            timestamp = f"{_seconds_to_timecode(st, fps=fps)}-{_seconds_to_timecode(en, fps=fps)}"

            # Attach overlapping transcript segments (original language).
            if transcript_segments:
                segs: list[dict[str, Any]] = []
                for seg in transcript_segments:
                    if not isinstance(seg, dict):
                        continue
                    ss = _safe_float(seg.get("start"), 0.0)
                    se = _safe_float(seg.get("end"), ss)
                    if se <= ss:
                        continue
                    if _overlap_seconds(ss, se, st, en) > 0:
                        segs.append(seg)
                if segs:
                    item["transcript_segments"] = segs
                    txt = " ".join([str(s.get("text") or "").strip() for s in segs if str(s.get("text") or "").strip()]).strip()
                    if txt:
                        item["summary_text_transcript"] = txt[:320]
                        item["summary_text"] = txt[:320]

            llm_summary = str(sc.get("summary_llm") or sc.get("summary_text_llm") or "").strip()
            if llm_summary:
                item["summary_text_llm"] = llm_summary
                item["summary_text"] = llm_summary

            summary_value = str(item.get("summary_text") or "").strip()

            # Attach overlapping objects for scene context.
            scene_objects: list[str] = []
            if objects_src:
                for obj in objects_src:
                    if not isinstance(obj, dict):
                        continue
                    name = str(obj.get("name") or obj.get("label") or "").strip()
                    if not name:
                        continue
                    for seg in (obj.get("segments") or []):
                        if not isinstance(seg, dict):
                            continue
                        if _overlap_seconds(
                            _safe_float(seg.get("start"), 0.0),
                            _safe_float(seg.get("end"), 0.0),
                            st,
                            en,
                        ) <= 0:
                            continue
                        scene_objects.append(name)
                        break
            if scene_objects:
                item["objects"] = list(dict.fromkeys(scene_objects))

            if timestamp:
                item["timestamp"] = timestamp
            if scene_objects:
                item["objects_detected"] = ", ".join(list(dict.fromkeys(scene_objects)))
            if summary_value:
                item["summary"] = summary_value

            # Attach overlapping labels for scene context.
            if labels_src:
                scene_labels: list[dict[str, Any]] = []
                for lab in labels_src:
                    if not isinstance(lab, dict):
                        continue
                    if _count_overlapping_segments([lab], st, en) <= 0:
                        continue
                    name = str(lab.get("name") or lab.get("label") or "").strip()
                    if not name:
                        continue
                    scene_labels.append({"name": name})
                if scene_labels:
                    item["labels"] = scene_labels

            scene_out.append(item)

        if scene_out:
            payload = {
                "scenes": scene_out,
                "source": video.get("scenes_source") or "unknown",
            }
            categories.setdefault("scene_by_scene_metadata", payload)
            combined["scene_by_scene_metadata"] = payload

    return {"categories": categories, "combined": combined}


def _build_combined_metadata_for_language(video: dict[str, Any], lang: str | None) -> dict[str, Any]:
    categorized = _build_categorized_metadata_json(video)
    combined = categorized.get("combined") if isinstance(categorized.get("combined"), dict) else {}
    lang_norm = (lang or "").strip().lower()
    if not lang_norm or lang_norm in {"orig", "original"}:
        return combined

    translations = video.get("translations") if isinstance(video, dict) else None
    by_lang = None
    if isinstance(translations, dict):
        by_lang = translations.get("by_language")
        if isinstance(by_lang, dict):
            by_lang = by_lang.get(lang_norm)
        else:
            by_lang = None
    if not isinstance(by_lang, dict):
        return combined

    out = deepcopy(combined)

    translated_transcript = by_lang.get("transcript")
    if isinstance(translated_transcript, dict):
        meta = {}
        if isinstance(out.get("transcript"), dict):
            meta = out.get("transcript", {}).get("meta") or {}
        out["transcript"] = {
            "text": str(translated_transcript.get("text") or ""),
            "segments": translated_transcript.get("segments") or [],
            "language_code": lang_norm,
            "meta": meta,
        }

    for key in ("synopsis", "scene_by_scene_metadata", "key_scenes", "high_points"):
        value = by_lang.get(key)
        if value is not None:
            out[key] = value

    on_screen_text = by_lang.get("on_screen_text")
    if on_screen_text is not None:
        out["on_screen_text"] = on_screen_text

    return out


def _apply_translated_combined(video: dict[str, Any], combined: dict[str, Any], lang: str | None) -> dict[str, Any]:
    lang_norm = (lang or "").strip().lower()
    if not lang_norm or lang_norm in {"orig", "original"}:
        return combined

    translations = video.get("translations") if isinstance(video, dict) else None
    by_lang = None
    if isinstance(translations, dict):
        by_lang = translations.get("by_language")
        if isinstance(by_lang, dict):
            by_lang = by_lang.get(lang_norm)
        else:
            by_lang = None
    if not isinstance(by_lang, dict):
        return combined

    out = deepcopy(combined)

    translated_transcript = by_lang.get("transcript")
    if isinstance(translated_transcript, dict):
        meta = {}
        if isinstance(out.get("transcript"), dict):
            meta = out.get("transcript", {}).get("meta") or {}
        out["transcript"] = {
            "text": str(translated_transcript.get("text") or ""),
            "segments": translated_transcript.get("segments") or [],
            "language_code": lang_norm,
            "meta": meta,
        }

    for key in ("synopsis", "scene_by_scene_metadata", "key_scenes", "high_points"):
        value = by_lang.get(key)
        if value is not None:
            out[key] = value

    on_screen_text = by_lang.get("on_screen_text")
    if on_screen_text is not None:
        out["on_screen_text"] = on_screen_text

    return out


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    st = max(float(a_start or 0.0), float(b_start or 0.0))
    en = min(float(a_end or 0.0), float(b_end or 0.0))
    return max(0.0, en - st)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _parse_int(
    value: Any,
    *,
    default: int = 0,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    try:
        parsed = int(str(value).strip())
    except Exception:
        parsed = int(default)
    if min_value is not None:
        parsed = max(int(min_value), parsed)
    if max_value is not None:
        parsed = min(int(max_value), parsed)
    return parsed


def _video_duration_seconds_from_ffprobe(technical: dict[str, Any]) -> float | None:
    try:
        raw = (technical or {}).get("raw") if isinstance(technical, dict) else None
        fmt = (raw or {}).get("format") if isinstance(raw, dict) else None
        dur = (fmt or {}).get("duration") if isinstance(fmt, dict) else None
        if dur is None:
            return None
        v = float(dur)
        return v if v > 0 else None
    except Exception:
        return None


def _parse_ffprobe_ratio(value: Any) -> float | None:
    """Parse ffprobe ratio strings like '30000/1001' or numeric strings."""
    if value is None:
        return None
    try:
        s = str(value).strip()
    except Exception:
        return None
    if not s:
        return None
    try:
        if "/" in s:
            num_s, den_s = s.split("/", 1)
            num = float(num_s)
            den = float(den_s)
            if den == 0:
                return None
            v = num / den
            return v if v > 0 else None
        v = float(s)
        return v if v > 0 else None
    except Exception:
        return None


def _video_fps_from_ffprobe(technical: dict[str, Any]) -> int:
    """Best-effort fps (integer) for hh:mm:ss:ff timecodes."""
    try:
        raw = (technical or {}).get("raw") if isinstance(technical, dict) else None
        streams = (raw or {}).get("streams") if isinstance(raw, dict) else None
        if isinstance(streams, list):
            for st in streams:
                if not isinstance(st, dict):
                    continue
                if str(st.get("codec_type") or "") != "video":
                    continue
                fps = _parse_ffprobe_ratio(st.get("avg_frame_rate")) or _parse_ffprobe_ratio(st.get("r_frame_rate"))
                if fps:
                    fps_i = int(round(float(fps)))
                    return max(1, min(240, fps_i))
    except Exception:
        pass
    return 30


def _seconds_to_timecode(seconds: float | None, *, fps: int) -> str:
    try:
        s = float(seconds or 0.0)
    except Exception:
        s = 0.0
    if s < 0:
        s = 0.0
    fps_i = max(1, int(fps))
    total_frames = int(round(s * fps_i))
    frames_per_hour = fps_i * 3600
    frames_per_minute = fps_i * 60

    hh = total_frames // frames_per_hour
    rem = total_frames % frames_per_hour
    mm = rem // frames_per_minute
    rem = rem % frames_per_minute
    ss = rem // fps_i
    ff = rem % fps_i

    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _env_truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    try:
        text = str(value).strip().lower()
    except Exception:
        return bool(default)
    if not text:
        return bool(default)
    return text in {"1", "true", "yes", "y", "on"}


def _count_overlapping_segments(items: Any, start_s: float, end_s: float) -> int:
    """Counts how many nested segments overlap a time window."""

    if not isinstance(items, list) or not items:
        return 0
    cnt = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        segs = it.get("segments")
        if not isinstance(segs, list):
            continue
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            st = seg.get("start_seconds")
            en = seg.get("end_seconds")
            if st is None or en is None:
                st = seg.get("start")
                en = seg.get("end")
            ss = _safe_float(st, 0.0)
            se = _safe_float(en, ss)
            if se <= ss:
                continue
            if _overlap_seconds(ss, se, start_s, end_s) > 0:
                cnt += 1
    return cnt


def _extract_keyframe_jpg(*, video_path: Path, at_seconds: float, out_path: Path, max_seconds: int = 60) -> bool:
    """Extract a single JPEG frame near `at_seconds` (small resolution for speed)."""

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    ffmpeg_service_url = _ffmpeg_service_url()
    if not ffmpeg_service_url:
        return False
    try:
        frame_bytes = _extract_frame_via_ffmpeg_service(
            service_url=ffmpeg_service_url,
            video_path=video_path,
            filename=video_path.name,
            timestamp=float(at_seconds or 0.0),
            scale=224,
            quality=3,
        )
        out_path.write_bytes(frame_bytes)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def _ahash64_from_jpg(jpg_path: Path) -> str | None:
    """Simple 64-bit average-hash (aHash) from a JPEG."""

    if Image is None:
        return None
    try:
        img = Image.open(str(jpg_path)).convert("L").resize((8, 8))
        pixels = list(img.getdata())
        if not pixels:
            return None
        avg = sum(pixels) / float(len(pixels))
        bits = ["1" if p >= avg else "0" for p in pixels]
        return "".join(bits) if len(bits) == 64 else None
    except Exception:
        return None


def _hamming01(a: str | None, b: str | None) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0
    diff = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            diff += 1
    return float(diff) / float(len(a))


@dataclass
class _SceneScore:
    index: int
    start: float
    end: float
    score: float
    reasons: list[str]
    emb: str | None = None


def _select_key_scenes_eventful(
    *,
    scenes: List[Dict[str, Any]],
    scenes_source: str,
    video_intelligence: Dict[str, Any],
    transcript_segments: List[Dict[str, Any]],
    local_path: Path,
    temp_dir: Path,
    top_k: int = 10,
    use_clip_cluster: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Ranks scenes as 'key' using a keyscene microservice."""

    if not scenes:
        return ([], [])

    service_url = _keyscene_service_url()
    if not service_url:
        raise RuntimeError("Keyscene service URL is not configured")

    payload = {
        "scenes": scenes,
        "scenes_source": scenes_source,
        "video_intelligence": video_intelligence or {},
        "transcript_segments": transcript_segments or [],
        "video_path": str(local_path),
        "top_k": int(top_k or 10),
        "use_clip_cluster": bool(use_clip_cluster),
    }
    timeout_s = float(os.getenv("ENVID_METADATA_KEY_SCENE_TIMEOUT_SECONDS") or 180.0)
    resp = requests.post(f"{service_url}/keyscene/select", json=payload, timeout=timeout_s)
    if resp.status_code >= 400:
        raise RuntimeError(f"Keyscene service failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("Invalid keyscene service response")
    key_scenes = data.get("key_scenes")
    high_points = data.get("high_points")
    if not isinstance(key_scenes, list):
        key_scenes = []
    if not isinstance(high_points, list):
        high_points = []
    return (key_scenes, high_points)


def _download_gcs_object_to_file(*, bucket: str, obj: str, dest_path: Path, job_id: str) -> None:
    src_uri = f"gs://{bucket}/{obj}"
    dest_path = dest_path.resolve()
    gcs_root = Path("/mnt/gcs").resolve()
    try:
        rel = dest_path.relative_to(gcs_root)
    except ValueError:
        rel = None

    if rel is not None:
        dest_obj = str(rel).replace(os.sep, "/")
        dest_bucket = _gcs_working_bucket_name() or bucket
        dest_uri = f"gs://{dest_bucket}/{dest_obj}"
        _job_step_update(
            job_id,
            "upload_to_cloud_storage",
            status="running",
            percent=0,
            message=f"Copying {src_uri} -> {dest_uri}",
        )
        _job_update(job_id, progress=2, message="Copying within cloud storage")
        client = _gcs_client()
        src_bucket = client.bucket(bucket)
        dst_bucket = client.bucket(dest_bucket)
        src_blob = src_bucket.blob(obj)
        dst_blob = dst_bucket.blob(dest_obj)
        token = None
        last_percent = -1
        last_tick = 0.0
        copy_timeout_s = _safe_float(os.getenv("ENVID_GCS_COPY_TIMEOUT_SECONDS"), 900.0)
        copy_start = time.monotonic()
        while True:
            token, bytes_rewritten, total_bytes = dst_blob.rewrite(src_blob, token=token)
            if total_bytes:
                pct = int((bytes_rewritten * 100) / total_bytes)
                if pct > last_percent:
                    now = time.time()
                    if pct == 100 or (now - last_tick) >= 0.4:
                        last_percent = pct
                        last_tick = now
                        _job_step_update(
                            job_id,
                            "upload_to_cloud_storage",
                            status="running",
                            percent=pct,
                            message=f"Copying {src_uri} -> {dest_uri} ({pct}%)",
                        )
            if token is None:
                break
            if (time.monotonic() - copy_start) > copy_timeout_s:
                app.logger.warning("GCS rewrite timed out after %.1fs; falling back to download", copy_timeout_s)
                break
        # Wait briefly for gcsfuse to reflect the object
        for _ in range(10):
            if dest_path.exists():
                break
            time.sleep(0.5)
        if not dest_path.exists():
            # Fallback: download into the mounted path to ensure availability
            src_bucket.blob(obj).download_to_filename(str(dest_path))
        _job_step_update(job_id, "upload_to_cloud_storage", status="completed", percent=100, message="Copied")
        _db_file_upsert(job_id, kind="local_video", path=str(dest_path), gcs_uri=dest_uri)
        return

    _job_step_update(
        job_id,
        "upload_to_cloud_storage",
        status="running",
        percent=0,
        message=f"Downloading {src_uri} -> {dest_path}",
    )
    _job_update(job_id, progress=2, message="Downloading from cloud storage")
    client = _gcs_client()
    blob = client.bucket(bucket).blob(obj)
    try:
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.utcnow() + timedelta(minutes=30),
            method="GET",
        )
        with requests.get(url, stream=True, timeout=(10, 300)) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as handle:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    except Exception:
        blob.download_to_filename(str(dest_path))
    _job_step_update(job_id, "upload_to_cloud_storage", status="completed", percent=100, message="Downloaded")
    _db_file_upsert(job_id, kind="local_video", path=str(dest_path), gcs_uri=src_uri)


def _parse_task_selection(raw: Any) -> Dict[str, Any]:
    """Best-effort parser for the UI's task_selection payload.

    - Accepts dict (already parsed) or JSON string.
    - Returns an empty dict for missing/invalid payloads.
    """

    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
            return dict(obj) if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _ffmpeg_blackdetect_segments(
    *,
    video_path: Path,
    min_black_seconds: float = 0.7,
    picture_black_threshold: float = 0.98,
    pixel_black_threshold: float = 0.10,
    max_seconds: int = 900,
) -> List[Dict[str, Any]]:
    """Run FFmpeg blackdetect via service and return parsed segments.

    Returns: [{"start": float, "end": float, "duration": float}, ...]
    """

    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    service_url = _ffmpeg_service_url()
    if not service_url:
        raise RuntimeError("FFmpeg service URL is not configured")

    segments = _blackdetect_via_ffmpeg_service(
        service_url=service_url,
        video_path=video_path,
        filename=video_path.name,
        min_black_seconds=min_black_seconds,
        picture_black_threshold=picture_black_threshold,
        pixel_black_threshold=pixel_black_threshold,
        max_seconds=max_seconds,
    )
    segments.sort(key=lambda x: (float(x.get("start") or 0.0), float(x.get("end") or 0.0)))
    return segments


def _parse_hhmmss_to_seconds(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return 0.0
    # Supports HH:MM:SS[.mmm]
    m = re.match(r"^(?P<h>\d+):(?P<m>\d{2}):(?P<sec>\d{2})(?:\.(?P<ms>\d{1,6}))?$", s)
    if not m:
        return 0.0
    h = int(m.group("h"))
    mi = int(m.group("m"))
    sec = int(m.group("sec"))
    ms_raw = m.group("ms")
    if not ms_raw:
        frac = 0.0
    else:
        # Treat digits after the dot as fractional seconds (supports ms or us).
        frac = float(int(ms_raw)) / (10.0 ** len(ms_raw))
    return float(h * 3600 + mi * 60 + sec) + frac


def _scene_service_url() -> str | None:
    return "http://scene-detect:5094"


def _keyscene_service_url() -> str | None:
    return "http://keyscene:5085"


def _pyscenedetect_list_scenes_via_service(
    *,
    service_url: str,
    video_path: Path,
    max_seconds: int,
) -> List[Dict[str, Any]]:
    threshold = float(os.getenv("ENVID_METADATA_SCENEDETECT_THRESHOLD") or 27.0)
    with video_path.open("rb") as handle:
        files = {"video": (video_path.name, handle, "application/octet-stream")}
        data = {
            "threshold": str(threshold),
            "max_seconds": str(max_seconds),
        }
        resp = requests.post(f"{service_url}/pyscenedetect/scenes", files=files, data=data, timeout=float(max_seconds) + 30.0)
    if resp.status_code >= 400:
        raise RuntimeError(f"Scene service failed ({resp.status_code}): {resp.text}")
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid scene service response")
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        return []
    return [x for x in scenes if isinstance(x, dict)]


def _pyscenedetect_list_scenes(*, video_path: Path, temp_dir: Path, max_seconds: int = 900) -> List[Dict[str, Any]]:
    """Run PySceneDetect CLI and return scenes as [{start,end,index}, ...]."""
    scene_service_url = _scene_service_url()
    if not scene_service_url:
        raise RuntimeError("Scene service URL is not configured")
    return _pyscenedetect_list_scenes_via_service(
        service_url=scene_service_url,
        video_path=video_path,
        max_seconds=max_seconds,
    )


def _transnetv2_list_scenes(*, video_path: Path, temp_dir: Path, max_seconds: int = 900) -> List[Dict[str, Any]]:
    """TransNetV2 scene/shot detection via optional sidecar.

    This backend runs on Python 3.14+; TransNetV2 commonly needs a different runtime.
    We support a Docker sidecar (see microservices/keyscene/) and call it here.
    """

    _ = temp_dir
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    base = "http://keyscene:5085"

    url = f"{base}/transnetv2/scenes"
    timeout_seconds = float(os.getenv("ENVID_METADATA_LOCAL_KEYSCENE_TIMEOUT_SECONDS") or 120.0)

    # Guardrails: avoid trying to upload extremely large files.
    try:
        size_mb = float(video_path.stat().st_size) / (1024.0 * 1024.0)
        max_mb = float(os.getenv("ENVID_METADATA_LOCAL_KEYSCENE_MAX_UPLOAD_MB") or 350.0)
        if size_mb > max_mb:
            raise RuntimeError(f"video too large for transnetv2 sidecar upload ({size_mb:.1f}MB > {max_mb:.1f}MB)")
    except Exception:
        pass

    data = video_path.read_bytes()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Filename": video_path.name,
            "X-Max-Seconds": str(int(max_seconds)),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""
        raise RuntimeError(f"transnetv2 sidecar HTTP {exc.code}: {body[:240]}") from exc
    except Exception as exc:
        raise RuntimeError(f"transnetv2 sidecar request failed: {exc}") from exc

    scenes_in = resp.get("scenes")
    if not isinstance(scenes_in, list):
        raise RuntimeError("transnetv2 sidecar returned invalid scenes")

    scenes: List[Dict[str, Any]] = []
    for it in scenes_in:
        if not isinstance(it, dict):
            continue
        st = _safe_float(it.get("start"), 0.0)
        en = _safe_float(it.get("end"), st)
        if en <= st:
            continue
        scenes.append({"index": int(it.get("index") or len(scenes)), "start": float(st), "end": float(en)})
    return scenes


def _local_keyscene_best_clip_cluster(
    *,
    images_b64: List[str],
    k: int,
    timeout_seconds: int = 60,
    required: bool = False,
) -> List[int] | None:
    """Call CLIP clustering sidecar.

    Returns a list of cluster IDs aligned with images_b64, or None if unavailable.
    """

    base = "http://keyscene:5085"
    if not images_b64:
        if required:
            raise RuntimeError("CLIP clustering requested but no images were provided")
        return None

    url = f"{base}/clip/cluster"

    # Send explicit CLIP config to avoid any reliance on ambient defaults.
    # These are optional; the sidecar will fall back to its own env defaults.
    clip_model = (os.getenv("ENVID_METADATA_LOCAL_KEYSCENE_CLIP_MODEL") or os.getenv("CLIP_MODEL") or "").strip() or None
    clip_pretrained = (
        os.getenv("ENVID_METADATA_LOCAL_KEYSCENE_CLIP_PRETRAINED")
        or os.getenv("CLIP_PRETRAINED")
        or ""
    ).strip() or None
    clip_device = (os.getenv("ENVID_METADATA_LOCAL_KEYSCENE_CLIP_DEVICE") or os.getenv("CLIP_DEVICE") or "").strip() or None

    payload: Dict[str, Any] = {"images_b64": images_b64, "k": int(k)}
    if clip_model:
        payload["clip_model"] = clip_model
    if clip_pretrained:
        payload["clip_pretrained"] = clip_pretrained
    if clip_device:
        payload["clip_device"] = clip_device
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_seconds)) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        if required:
            raise RuntimeError(f"CLIP clustering sidecar request failed: {exc}") from exc
        return None

    ids = resp.get("cluster_ids")
    if not isinstance(ids, list) or len(ids) != len(images_b64):
        if required:
            raise RuntimeError("CLIP clustering sidecar returned invalid cluster_ids")
        return None
    out: List[int] = []
    for x in ids:
        try:
            out.append(int(x))
        except Exception:
            out.append(-1)
    return out


def _repo_root() -> Path:
    # backend.py is in microservices/backend/code/backend.py
    return Path(__file__).resolve().parents[2]


def _likelihood_from_score(score: float) -> str:
    """Map a [0..1] unsafe score to a GCP-like likelihood string."""
    try:
        s = float(score)
    except Exception:
        s = 0.0
    if s >= 0.90:
        return "VERY_LIKELY"
    if s >= 0.70:
        return "LIKELY"
    if s >= 0.40:
        return "POSSIBLE"
    if s >= 0.20:
        return "UNLIKELY"
    return "VERY_UNLIKELY"


def _severity_from_score(score: float) -> str:
    """Map a [0..1] unsafe score to a neutral, model-agnostic severity string."""
    try:
        s = float(score)
    except Exception:
        s = 0.0
    if s >= 0.90:
        return "critical"
    if s >= 0.70:
        return "high"
    if s >= 0.40:
        return "medium"
    if s >= 0.20:
        return "low"
    return "minimal"


def _severity_from_likelihood(likelihood: str) -> str:
    """Best-effort mapping from provider likelihood buckets to neutral severity."""
    lk = (likelihood or "").strip().upper()
    if lk in {"VERY_LIKELY"}:
        return "critical"
    if lk in {"LIKELY"}:
        return "high"
    if lk in {"POSSIBLE"}:
        return "medium"
    if lk in {"UNLIKELY"}:
        return "low"
    if lk in {"VERY_UNLIKELY"}:
        return "minimal"
    return "unknown"


def _moderation_extractor_explicit_frames_from_video(
    *,
    video_path: Path,
    interval_seconds: float,
    max_frames: int,
    service_base_url: str,
    timeout_seconds: int | None,
    job_id: str,
    timestamps: list[float] | None = None,
) -> Dict[str, Any]:
    base = service_base_url.rstrip("/")
    url = f"{base}/moderate/path"
    timeout_val = None
    if timeout_seconds is not None:
        try:
            timeout_val = float(timeout_seconds)
        except Exception:
            timeout_val = None
    if timeout_val is not None and timeout_val <= 0:
        timeout_val = None
    payload_req: dict[str, Any] = {
        "video_path": str(video_path),
        "interval_seconds": float(interval_seconds),
        "max_frames": int(max_frames),
        "job_id": job_id,
        "output_kind": "processed_local",
        "force_interval": True,
    }
    if timestamps:
        payload_req["timestamps"] = [float(t) for t in timestamps]

    resp = requests.post(
        url,
        json=payload_req,
        timeout=timeout_val,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"moderation extractor failed ({resp.status_code}): {resp.text}")
    payload = resp.json()
    explicit_frames = payload.get("explicit_frames")
    if not isinstance(explicit_frames, list):
        raise RuntimeError("moderation extractor returned invalid explicit_frames")
    out: List[Dict[str, Any]] = []
    for f in explicit_frames:
        if not isinstance(f, dict):
            continue
        provider_likelihood = str(f.get("likelihood") or f.get("provider_likelihood") or "")
        severity = str(f.get("severity") or "")
        if not severity:
            severity = _severity_from_likelihood(provider_likelihood)
        reasons = f.get("reasons") if isinstance(f.get("reasons"), list) else None
        top_label = f.get("top_label") if isinstance(f.get("top_label"), str) else None
        top_score = f.get("top_score")
        try:
            top_score_val = float(top_score) if top_score is not None else None
        except Exception:
            top_score_val = None
        out.append(
            {
                "time": float(f.get("time") or 0.0),
                "severity": severity,
                **({"provider_likelihood": provider_likelihood} if provider_likelihood else {}),
                **({"unsafe": float(f.get("unsafe") or 0.0)} if f.get("unsafe") is not None else {}),
                **({"safe": float(f.get("safe") or 0.0)} if f.get("safe") is not None else {}),
                **({"reasons": reasons} if reasons else {}),
                **({"top_label": top_label} if top_label else {}),
                **({"top_score": top_score_val} if top_score_val is not None else {}),
            }
        )
    return {
        "explicit_frames": out,
        "output_path": payload.get("output_path"),
        "frames_dir": payload.get("frames_dir"),
    }


def _aggregate_text_segments(
    *,
    hits: List[Tuple[str, float, float, float]],
    max_items: int = 250,
    merge_gap_seconds: float | None = None,
) -> List[Dict[str, Any]]:
    """Aggregate (text, start, end, conf) into [{text, segments:[...]}]"""
    min_conf = _safe_float(os.getenv("ENVID_METADATA_OCR_MIN_CONF"), 0.6)
    by_text: Dict[str, List[Dict[str, Any]]] = {}
    for txt, st, en, conf in hits:
        key = (txt or "").strip()
        if not key:
            continue
        if conf < min_conf:
            continue
        by_text.setdefault(key, []).append({"start": float(st), "end": float(en), "confidence": float(conf)})

    out: List[Dict[str, Any]] = []
    gap = float(merge_gap_seconds or 0.0)
    for txt, segs in by_text.items():
        segs.sort(key=lambda x: (float(x.get("start") or 0.0), float(x.get("end") or 0.0)))
        if not segs:
            continue
        if gap <= 0:
            out.append({"text": txt, "segments": segs})
            continue
        merged: List[Dict[str, Any]] = []
        cur_st = float(segs[0].get("start") or 0.0)
        cur_en = float(segs[0].get("end") or 0.0)
        cur_conf = float(segs[0].get("confidence") or 0.0)
        for seg in segs[1:]:
            st = float(seg.get("start") or 0.0)
            en = float(seg.get("end") or 0.0)
            conf = float(seg.get("confidence") or 0.0)
            if st <= cur_en + gap:
                cur_en = max(cur_en, en)
                cur_conf = max(cur_conf, conf)
            else:
                merged.append({"start": cur_st, "end": cur_en, "confidence": cur_conf})
                cur_st, cur_en, cur_conf = st, en, conf
        merged.append({"start": cur_st, "end": cur_en, "confidence": cur_conf})
        out.append({"text": txt, "segments": merged})
    out.sort(key=lambda x: (len(x.get("segments") or []), len(str(x.get("text") or ""))), reverse=True)
    return out[:max_items]



def _gcs_presign_get_url(
    *,
    bucket: str,
    obj: str,
    seconds: int | None = None,
    response_type: str | None = None,
    response_disposition: str | None = None,
) -> str:
    b = (bucket or "").strip()
    o = (obj or "").strip().lstrip("/")
    if not b or not o:
        raise ValueError("Missing bucket/object")

    ttl = seconds
    if ttl is None:
        ttl = _parse_int(os.getenv("GCP_GCS_PRESIGN_SECONDS"), default=3600, min_value=60, max_value=86400)

    return (
        _gcs_client()
        .bucket(b)
        .blob(o)
        .generate_signed_url(
            version="v4",
            expiration=datetime.utcnow() + timedelta(seconds=int(ttl)),
            method="GET",
            response_type=response_type,
            response_disposition=response_disposition,
        )
    )


def _upload_metadata_artifacts_to_gcs(*, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Best-effort upload of derived artifacts to GCS.

    Uploads:
    - combined.json
    - categories/*.json
    - metadata_json.zip (single JSON file inside)
    - subtitles (if present locally)
    """

    artifacts_bucket = _gcs_artifacts_bucket(_gcs_bucket_name())
    artifacts_prefix = _gcs_artifacts_prefix()
    base = f"{artifacts_prefix}/{job_id}".strip("/")

    client = _gcs_client()
    bkt = client.bucket(artifacts_bucket)

    out: dict[str, Any] = {
        "bucket": artifacts_bucket,
        "base_prefix": base,
        "combined": None,
        "categories": {},
        "zip": None,
        "subtitles": {},
    }

    combined_json = json.dumps(payload.get("combined") or {}, indent=2, ensure_ascii=False)
    combined_obj = f"{base}/combined.json.gz"
    bkt.blob(combined_obj).upload_from_string(
        gzip.compress(combined_json.encode("utf-8")),
        content_type="application/json; charset=utf-8",
        content_encoding="gzip",
    )
    out["combined"] = {"object": combined_obj, "uri": f"gs://{artifacts_bucket}/{combined_obj}"}

    cats = payload.get("categories") if isinstance(payload.get("categories"), dict) else {}
    for name, cat_payload in (cats or {}).items():
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name or "category").strip()) or "category"
        cat_json = json.dumps(cat_payload, indent=2, ensure_ascii=False)
        obj = f"{base}/categories/{safe}.json.gz"
        bkt.blob(obj).upload_from_string(
            gzip.compress(cat_json.encode("utf-8")),
            content_type="application/json; charset=utf-8",
            content_encoding="gzip",
        )
        out["categories"][str(name)] = {"object": obj, "uri": f"gs://{artifacts_bucket}/{obj}"}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{job_id}.metadata.json", json.dumps(payload, indent=2, ensure_ascii=False))
    buf.seek(0)
    zip_obj = f"{base}/metadata_json.zip"
    bkt.blob(zip_obj).upload_from_file(buf, rewind=True, content_type="application/zip")
    out["zip"] = {"object": zip_obj, "uri": f"gs://{artifacts_bucket}/{zip_obj}"}

    subtitle_specs = [
        ("orig", "srt", "application/x-subrip"),
        ("orig", "vtt", "text/vtt"),
        ("en", "srt", "application/x-subrip"),
        ("en", "vtt", "text/vtt"),
        ("ar", "srt", "application/x-subrip"),
        ("ar", "vtt", "text/vtt"),
        ("id", "srt", "application/x-subrip"),
        ("id", "vtt", "text/vtt"),
    ]
    for lang, fmt, content_type in subtitle_specs:
        local_path = _subtitles_local_path(job_id, lang=lang, fmt=fmt)
        if not local_path.exists():
            continue
        obj = f"{base}/subtitles/{lang}.{fmt}.gz"
        try:
            raw = local_path.read_bytes()
        except Exception:
            continue
        bkt.blob(obj).upload_from_string(
            gzip.compress(raw),
            content_type=content_type,
            content_encoding="gzip",
        )
        out["subtitles"][f"{lang}.{fmt}"] = {"object": obj, "uri": f"gs://{artifacts_bucket}/{obj}"}

    return out


def _gather_subtitles_payload(job_id: str) -> dict[str, dict[str, str]]:
    subtitle_specs = [
        ("orig", "srt", "application/x-subrip"),
        ("orig", "vtt", "text/vtt"),
        ("en", "srt", "application/x-subrip"),
        ("en", "vtt", "text/vtt"),
        ("ar", "srt", "application/x-subrip"),
        ("ar", "vtt", "text/vtt"),
        ("id", "srt", "application/x-subrip"),
        ("id", "vtt", "text/vtt"),
    ]
    out: dict[str, dict[str, str]] = {}
    for lang, fmt, content_type in subtitle_specs:
        local_path = _subtitles_local_path(job_id, lang=lang, fmt=fmt)
        if not local_path.exists():
            continue
        try:
            content = local_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not content.strip():
            continue
        out[f"{lang}.{fmt}"] = {"content": content, "content_type": content_type}
    return out


def _upload_metadata_artifacts_via_service(*, service_url: str, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    endpoint = f"{service_url}/upload_artifacts"
    subtitles_payload = _gather_subtitles_payload(job_id)
    request_payload = {
        "job_id": job_id,
        "payload": payload,
        "subtitles": subtitles_payload,
    }
    timeout_s = _safe_float(os.getenv("ENVID_EXPORT_SERVICE_TIMEOUT_SECONDS"), 60.0)
    resp = requests.post(endpoint, json=request_payload, timeout=float(timeout_s))
    if resp.status_code >= 400:
        raise RuntimeError(f"Export service failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("Invalid export service response")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("Invalid export service artifacts")
    return artifacts


def _upload_metadata_artifacts(*, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    export_url = _export_service_url()
    if export_url:
        return _upload_metadata_artifacts_via_service(service_url=export_url, job_id=job_id, payload=payload)
    return _upload_metadata_artifacts_to_gcs(job_id=job_id, payload=payload)


def _record_job_artifacts_outputs(job_id: str, artifacts: dict[str, Any] | None) -> None:
    if not artifacts or not isinstance(artifacts, dict):
        return
    bucket = str(artifacts.get("bucket") or "").strip()
    base_prefix = str(artifacts.get("base_prefix") or "").strip()
    if bucket and base_prefix:
        _db_file_upsert(job_id, kind="gcs_artifacts_base", path=None, gcs_uri=f"gs://{bucket}/{base_prefix}")

    combined = artifacts.get("combined") if isinstance(artifacts.get("combined"), dict) else None
    if combined:
        _db_file_upsert(job_id, kind="metadata_combined", path=None, gcs_uri=str(combined.get("uri") or "").strip() or None)

    zip_info = artifacts.get("zip") if isinstance(artifacts.get("zip"), dict) else None
    if zip_info:
        _db_file_upsert(job_id, kind="metadata_json_zip", path=None, gcs_uri=str(zip_info.get("uri") or "").strip() or None)

    categories = artifacts.get("categories") if isinstance(artifacts.get("categories"), dict) else {}
    for name, info in categories.items():
        if not isinstance(info, dict):
            continue
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name or "category").strip()) or "category"
        _db_file_upsert(job_id, kind=f"metadata_category_{safe}", path=None, gcs_uri=str(info.get("uri") or "").strip() or None)

    subtitles = artifacts.get("subtitles") if isinstance(artifacts.get("subtitles"), dict) else {}
    for key, info in subtitles.items():
        if not isinstance(info, dict):
            continue
        safe_key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(key or "subtitle").strip()) or "subtitle"
        kind = f"subtitle_{safe_key.replace('.', '_')}"
        _db_file_upsert(job_id, kind=kind, path=None, gcs_uri=str(info.get("uri") or "").strip() or None)


def _task_selection_requested_models(sel: dict[str, Any]) -> dict[str, Any]:
    if isinstance(sel.get("requested_models"), dict):
        return dict(sel.get("requested_models") or {})
    return {
        "label_detection_model": sel.get("label_detection_model"),
        "moderation_model": sel.get("moderation_model"),
        "text_model": sel.get("text_model"),
        "key_scene_detection_model": sel.get("key_scene_detection_model"),
        "transcribe_model": sel.get("transcribe_model"),
        "famous_location_detection_model": sel.get("famous_location_detection_model"),
        "opening_closing_credit_detection_model": sel.get("opening_closing_credit_detection_model"),
        "celebrity_detection_model": sel.get("celebrity_detection_model"),
        "celebrity_bio_image_model": sel.get("celebrity_bio_image_model"),
        "scene_by_scene_metadata_model": sel.get("scene_by_scene_metadata_model"),
    }


def _process_gcs_video_job_cloud_only(
    *,
    job_id: str,
    gcs_bucket: str,
    gcs_object: str,
    video_title: str,
    video_description: str,
    frame_interval_seconds: int,
    max_frames_to_analyze: int,
    face_recognition_mode: Optional[str],
    task_selection: Optional[Dict[str, Any]] = None,
) -> None:
    temp_dir = _gcs_job_work_dir(job_id)
    temp_dir.mkdir(parents=True, exist_ok=True)
    gcs_uri = f"gs://{gcs_bucket}/{gcs_object}"
    gcs_label_uri = gcs_uri
    _db_file_upsert(job_id, kind="source_gcs", path=None, gcs_uri=gcs_uri)
    critical_failures: list[str] = []
    video_intelligence: Dict[str, Any] | None = None
    effective_models: dict[str, Any] = {}
    try:
        _job_update(job_id, status="processing", progress=1, message="Processing", gcs_video_uri=gcs_uri, temp_dir=str(temp_dir))

        sel: Dict[str, Any] = task_selection if isinstance(task_selection, dict) else {}
        requested_models = _task_selection_requested_models(sel)
        preflight = orchestrate_preflight(
            inputs=OrchestratorInputs(job_id=job_id, task_selection=sel, requested_models=requested_models),
            precheck_models=None,
            job_update=_job_update,
            job_step_update=_job_step_update,
        )
        selection = preflight.selection

        _job_update(
            job_id,
            task_selection=sel,
            task_selection_effective={
                "enable_label_detection": selection.enable_label_detection,
                "enable_text_on_screen": selection.enable_text_on_screen,
                "enable_moderation": selection.enable_moderation,
                "enable_transcribe": selection.enable_transcribe,
                "enable_scene_by_scene": selection.enable_scene_by_scene,
                "enable_key_scene": selection.enable_key_scene,
                "enable_high_point": selection.enable_high_point,
                "enable_synopsis_generation": selection.enable_synopsis_generation,
                "enable_translate_output": selection.enable_translate_output,
                "enable_famous_locations": selection.enable_famous_locations,
                "enable_opening_closing": selection.enable_opening_closing,
                "enable_celebrity_detection": selection.enable_celebrity_detection,
                "enable_celebrity_bio_image": selection.enable_celebrity_bio_image,
            },
        )

        enable_label_detection = selection.enable_label_detection
        enable_text_on_screen = selection.enable_text_on_screen
        enable_moderation = selection.enable_moderation
        enable_transcribe = selection.enable_transcribe
        enable_famous_locations = selection.enable_famous_locations
        enable_scene_by_scene = selection.enable_scene_by_scene
        enable_key_scene = selection.enable_key_scene
        enable_high_point = selection.enable_high_point
        enable_synopsis_generation = selection.enable_synopsis_generation
        enable_translate_output = selection.enable_translate_output
        enable_opening_closing = selection.enable_opening_closing
        enable_celebrity_detection = selection.enable_celebrity_detection
        enable_celebrity_bio_image = selection.enable_celebrity_bio_image

        requested_label_model_raw = selection.requested_label_model_raw
        requested_label_model = selection.requested_label_model
        requested_text_model = selection.requested_text_model
        requested_moderation_model = selection.requested_moderation_model
        requested_key_scene_model_raw = selection.requested_key_scene_model_raw
        requested_key_scene_model = selection.requested_key_scene_model
        requested_transcribe_model_raw = (requested_models.get("transcribe_model") or "").strip()
        requested_transcribe_model = requested_transcribe_model_raw.strip().lower() if requested_transcribe_model_raw else ""
        default_transcribe_mode = _default_transcribe_mode()
        # Force Whisper for all transcribe runs.
        if enable_transcribe:
            transcribe_effective_mode = "openai-whisper"
        else:
            transcribe_effective_mode = default_transcribe_mode

        label_engine = selection.label_engine
        use_vi_label_detection = True
        use_local_ocr = True
        use_local_moderation = True
        allow_moderation_fallback = False
        if not requested_text_model or requested_text_model in {"", "auto", "default"}:
            requested_text_model = "text-on-video"
        if requested_moderation_model not in {"nudenet"}:
            requested_moderation_model = "nudenet"
        local_moderation_url_override = selection.local_moderation_url_override

        key_scene_step_finalized = False
        allowed_key_scene_models = {
            "transnetv2_clip_cluster",
            "pyscenedetect_clip_cluster",
            "clip_cluster",
        }
        if enable_key_scene and requested_key_scene_model not in allowed_key_scene_models:
            _job_step_update(
                job_id,
                "key_scene_detection",
                status="failed",
                percent=100,
                message=(
                    "key_scene_detection_model must be one of: transnetv2_clip_cluster, "
                    "pyscenedetect_clip_cluster (no fallback)."
                ),
            )
            key_scene_step_finalized = True
            enable_key_scene = False
            enable_high_point = False

        use_transnetv2_for_scenes = selection.use_transnetv2_for_scenes
        use_pyscenedetect_for_scenes = selection.use_pyscenedetect_for_scenes
        use_clip_cluster_for_key_scenes = selection.use_clip_cluster_for_key_scenes
        want_shots = selection.want_shots
        want_vi_shots = False
        want_any_vi = True
        enable_vi = True

        vi_label_mode = (os.getenv("ENVID_METADATA_GCP_VI_LABEL_MODE") or "frame").strip().lower() or "frame"
        if vi_label_mode not in {"segment", "shot", "frame"}:
            vi_label_mode = "frame"

        labels: List[Dict[str, Any]] = []
        label_service_used = False
        clip_ok = True
        if not clip_ok:
            use_clip_cluster_for_key_scenes = False

        ext = Path(gcs_object).suffix or ".mp4"
        original_dir = temp_dir / "transcode" / "original"
        original_dir.mkdir(parents=True, exist_ok=True)
        local_path = original_dir / f"{job_id}{ext}"
        _download_gcs_object_to_file(bucket=gcs_bucket, obj=gcs_object, dest_path=local_path, job_id=job_id)
        _db_file_upsert(job_id, kind="transcode_original", path=str(local_path), gcs_uri=gcs_uri)

        _job_step_update(job_id, "technical_metadata", status="running", percent=0, message="Probing")
        ffmpeg_service_url = _ffmpeg_service_url()
        if ffmpeg_service_url:
            technical_ffprobe = _probe_via_ffmpeg_service(
                service_url=ffmpeg_service_url,
                video_path=local_path,
                filename=local_path.name,
            )
        else:
            technical_ffprobe = _probe_technical_metadata(local_path)
        _db_payload_insert(job_id, kind="technical_metadata", payload=technical_ffprobe or {})
        duration_seconds = _video_duration_seconds_from_ffprobe(technical_ffprobe)
        _job_step_update(job_id, "technical_metadata", status="completed", percent=100, message="Completed")

        # Transcode normalize (required). If this fails, stop the job.
        normalize_enabled = _env_truthy(os.getenv("ENVID_METADATA_ENABLE_TRANSCODE_NORMALIZE"), default=True)
        normalized_dir = temp_dir / "transcode" / "normalize"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        normalized_path = normalized_dir / f"{job_id}{ext}"
        if normalize_enabled:
            try:
                ffmpeg_service_url = _ffmpeg_service_url()
                if ffmpeg_service_url:
                    _job_step_update(
                        job_id,
                        "transcode_normalize",
                        status="running",
                        percent=1,
                        message="Running (service)",
                    )
                    _job_update(job_id, progress=6, message="Transcode normalize")

                    normalized_bytes = _normalize_via_ffmpeg_service(
                        service_url=ffmpeg_service_url,
                        video_path=local_path,
                        filename=local_path.name,
                    )
                    normalized_path.write_bytes(normalized_bytes)
                    if normalized_path.exists() and normalized_path.stat().st_size > 0:
                        local_path = normalized_path
                        _db_file_upsert(job_id, kind="local_normalized", path=str(local_path))
                        _db_file_upsert(job_id, kind="transcode_normalized", path=str(local_path), gcs_uri=None)
                    _job_step_update(job_id, "transcode_normalize", status="completed", percent=100, message="Completed")
                else:
                    _job_step_update(
                        job_id,
                        "transcode_normalize",
                        status="failed",
                        percent=100,
                        message="FFmpeg service not configured",
                    )
                    raise RuntimeError("Transcode normalize failed: FFmpeg service not configured")
            except Exception as exc:
                app.logger.warning("Transcode normalize failed: %s", exc)
                _job_step_update(job_id, "transcode_normalize", status="failed", percent=100, message="Failed")
                raise
        else:
            _job_step_update(job_id, "transcode_normalize", status="failed", percent=100, message="Disabled")
            raise RuntimeError("Transcode normalize is disabled; stopping job")

        # Upload transcoded (or original if no transcode) to working GCS for label detection.
        try:
            working_label_uri = _upload_to_working_bucket(job_id=job_id, local_path=local_path, filename=local_path.name)
            gcs_label_uri = working_label_uri
            _job_update(job_id, gcs_working_uri=working_label_uri)
            _db_file_upsert(job_id, kind="processed_gcs", path=None, gcs_uri=working_label_uri)
        except Exception as exc:
            app.logger.warning("Failed to upload normalized video to working bucket: %s", exc)

        _db_file_upsert(job_id, kind="processed_local", path=str(local_path))

        # Opening/closing credit detection is disabled (not implemented).
        effective_models["opening_closing_credit_detection"] = "not_implemented"
        _job_step_update(
            job_id,
            "opening_closing_credit_detection",
            status="skipped",
            percent=100,
            message="Not implemented",
        )

        # Placeholders for steps not implemented in this backend yet.
        _job_step_update(
            job_id,
            "celebrity_detection",
            status="skipped",
            percent=100,
            message=(
                f"Not implemented (requested: {requested_models.get('celebrity_detection_model')})"
                if enable_celebrity_detection
                else "Disabled"
            ),
        )
        _job_step_update(
            job_id,
            "celebrity_bio_image",
            status="skipped",
            percent=100,
            message=(
                f"Not implemented (requested: {requested_models.get('celebrity_bio_image_model')})"
                if enable_celebrity_bio_image
                else "Disabled"
            ),
        )

        # vi_label_mode is already set above; it may be overridden by explicit label model values.

        # Honor label detection mode override when provided.
        # (Only applies to Google Video Intelligence modes.)

        if requested_label_model not in {"", "auto"}:
            # Accept explicit VI-mode overrides when the UI uses them.
            if requested_label_model in {"vi_segment", "segment"}:
                vi_label_mode = "segment"
            elif requested_label_model in {"vi_shot", "shot"}:
                vi_label_mode = "shot"
            elif requested_label_model in {"vi_frame", "frame"}:
                vi_label_mode = "frame"
        effective_models: dict[str, Any] = {}
        effective_models["label_detection"] = label_engine if enable_label_detection else "disabled"
        effective_models["label_detection_model_requested"] = requested_label_model_raw
        effective_models["label_detection_model_normalized"] = requested_label_model
        effective_models["label_detection_mode"] = vi_label_mode
        effective_models["text_on_screen"] = requested_text_model if (enable_text_on_screen and use_local_ocr) else "disabled"
        effective_models["moderation"] = requested_moderation_model if (enable_moderation and use_local_moderation) else "disabled"
        translate_provider = _translate_provider()
        if translate_provider == "libretranslate":
            translate_label = "libretranslate"
        elif translate_provider == "gcp_translate":
            translate_label = "gcp_translate"
        else:
            translate_label = "disabled"
        effective_models["famous_location_detection"] = (
            f"{translate_label}+gcp_language" if (translate_label != "disabled" and gcp_language is not None) else "disabled"
        )
        effective_models["key_scene_detection"] = requested_key_scene_model if enable_key_scene else "disabled"

        vi_min_label_conf = float(os.getenv("ENVID_METADATA_GCP_VI_MIN_LABEL_CONFIDENCE") or 0.0)
        vi_min_object_conf = float(os.getenv("ENVID_METADATA_GCP_VI_MIN_OBJECT_CONFIDENCE") or 0.95)
        vi_max_labels = _parse_int(os.getenv("ENVID_METADATA_GCP_VI_MAX_LABELS"), default=80, min_value=1, max_value=2000)
        vi_max_segments_per_label = _parse_int(os.getenv("ENVID_METADATA_GCP_VI_MAX_SEGMENTS_PER_LABEL"), default=8, min_value=1, max_value=500)
        vi_max_frames_per_label = _parse_int(os.getenv("ENVID_METADATA_GCP_VI_MAX_FRAMES_PER_LABEL"), default=200, min_value=1, max_value=20000)
        enable_objects = _env_truthy(os.getenv("ENVID_METADATA_GCP_VI_ENABLE_OBJECT_TRACKING"), default=False)

        parallel_futures: Dict[str, Any] = {}
        parallel_executor: ThreadPoolExecutor | None = None
        local_ocr_started = False
        local_moderation_started = False
        whisper_started = False
        scenes_started = False
        sequential_scene_then_whisper = _env_truthy(
            os.getenv("ENVID_METADATA_SEQUENTIAL_SCENE_BEFORE_WHISPER"),
            default=False,
        )
        run_all_sequential = _env_truthy(
            os.getenv("ENVID_METADATA_RUN_ALL_SEQUENTIAL"),
            default=False,
        )

        local_labels: list[dict[str, Any]] = []
        local_text: list[dict[str, Any]] = []
        local_moderation_frames: list[dict[str, Any]] | None = None
        local_moderation_source: str | None = None
        transcript = ""
        transcript_language_code = ""
        languages_detected: list[str] = []
        transcript_words: list[dict[str, Any]] = []
        transcript_segments: list[dict[str, Any]] = []
        transcript_meta: dict[str, Any] = {}
        transcript_script = ""
        transcript_raw = ""
        transcript_raw_segments: list[dict[str, Any]] = []
        transcript_raw_script = ""
        transcribe_completed = False
        transcribe_failed = False
        precomputed_scenes: list[dict[str, Any]] | None = None
        precomputed_scenes_source: str | None = None

        def _selection_language_hint(selection: dict[str, Any] | None) -> str:
            if not isinstance(selection, dict):
                return ""
            return str(
                selection.get("transcribe_language")
                or selection.get("transcribe_source_language")
                or selection.get("source_language")
                or selection.get("source_language_code")
                or selection.get("sourceLanguage")
                or ""
            ).strip().lower()

        def _ensure_parallel_executor() -> ThreadPoolExecutor:
            nonlocal parallel_executor
            if parallel_executor is None:
                parallel_executor = ThreadPoolExecutor(max_workers=5)
            return parallel_executor

        def _run_local_ocr_task() -> None:
            nonlocal local_text, effective_models
            try:
                _job_update(job_id, progress=26, message="Text on Screen (OCR)")
                interval = 1.0
                max_frames = 20000
                _job_step_update(job_id, "text_on_screen", status="running", percent=0, message=f"Running ({requested_text_model})")
                text_on_video_url = _text_on_video_service_url()
                if not text_on_video_url:
                    raise RuntimeError("Text-on-video service URL is not set")
                payload = _fetch_ocr_via_text_on_video(
                    service_url=text_on_video_url,
                    video_path=local_path,
                    interval_seconds=interval,
                    max_frames=max_frames,
                    job_id=job_id,
                )
                local_text = payload.get("text") if isinstance(payload, dict) else []
                effective_models["text_on_screen"] = "text-on-video"
                output_path = payload.get("output_path") if isinstance(payload, dict) else None
                if isinstance(output_path, str) and output_path:
                    _db_file_insert(job_id, kind="ocr_output", path=output_path)
                _job_step_update(job_id, "text_on_screen", status="completed", percent=100, message=f"{len(local_text or [])} text entries ({effective_models.get('text_on_screen')})")
            except Exception as exc:
                app.logger.warning("Local OCR failed: %s", exc)
                _job_step_update(job_id, "text_on_screen", status="failed", message=str(exc)[:240])

        def _run_local_moderation_task(*, timestamps: list[float] | None = None, sampling_strategy: str | None = None) -> None:
            nonlocal local_moderation_frames, local_moderation_source, effective_models
            try:
                _job_update(job_id, progress=28, message="Moderation (local)")
                interval = 0.0
                max_frames = 200000
                service_url_default = _moderation_service_url()
                service_url = (local_moderation_url_override or service_url_default).strip()
                service_timeout = int(_parse_int(os.getenv("ENVID_METADATA_LOCAL_MODERATION_TIMEOUT_SECONDS"), default=0, min_value=0, max_value=3600) or 0)

                _job_step_update(job_id, "moderation", status="running", percent=0, message=f"Running ({requested_moderation_model})")
                if requested_moderation_model == "nudenet":
                    if not service_url:
                        raise RuntimeError("Moderation service URL is not set")
                    moderation_payload = _moderation_extractor_explicit_frames_from_video(
                        video_path=local_path,
                        interval_seconds=interval,
                        max_frames=max_frames,
                        service_base_url=service_url,
                        timeout_seconds=None if service_timeout <= 0 else service_timeout,
                        job_id=job_id,
                        timestamps=timestamps,
                    )
                    local_moderation_frames = moderation_payload.get("explicit_frames") if isinstance(moderation_payload, dict) else []
                    if isinstance(local_moderation_frames, list) and local_moderation_frames:
                        deduped: list[dict[str, Any]] = []
                        seen: set[tuple[float, str]] = set()
                        for fr in local_moderation_frames:
                            if not isinstance(fr, dict):
                                continue
                            try:
                                t = float(fr.get("time") or 0.0)
                            except Exception:
                                t = 0.0
                            sev = str(fr.get("severity") or "")
                            key = (round(t, 3), sev)
                            if key in seen:
                                continue
                            seen.add(key)
                            deduped.append(fr)
                        local_moderation_frames = deduped
                    local_moderation_source = "nudenet_extractor"
                    effective_models["moderation"] = "nudenet_extractor"
                    if isinstance(moderation_payload, dict):
                        output_path = moderation_payload.get("output_path")
                        if isinstance(output_path, str) and output_path:
                            _db_file_insert(job_id, kind="moderation_output", path=output_path)
                    _db_payload_insert(
                        job_id,
                        kind="moderation",
                        payload={
                            "source": local_moderation_source,
                            "model": effective_models.get("moderation"),
                            "video_path": str(local_path),
                            "interval_seconds": interval,
                            "max_frames": max_frames,
                            **({"sampling_strategy": sampling_strategy} if sampling_strategy else {}),
                            **({"timestamps": timestamps} if timestamps else {}),
                            **({"timestamps_count": len(timestamps)} if timestamps else {}),
                            "explicit_frames": local_moderation_frames or [],
                        },
                    )
                    _job_step_update(job_id, "moderation", status="completed", percent=100, message=f"{len(local_moderation_frames or [])} frames ({local_moderation_source})")
                else:
                    raise RuntimeError(f"Unsupported moderation model: {requested_moderation_model}")
            except Exception as exc:
                app.logger.warning("Local moderation failed: %s", exc)
                _job_step_update(job_id, "moderation", status="failed", message=str(exc)[:240])

        def _run_whisper_transcription() -> None:
            nonlocal transcript, transcript_language_code, languages_detected, transcript_words, transcript_segments, transcript_meta, effective_models
            nonlocal transcript_raw, transcript_raw_segments, transcript_raw_script, transcribe_completed, transcribe_failed
            service_url = _transcribe_service_url()
            if not service_url:
                _job_step_update(job_id, "transcribe", status="skipped", percent=100, message="Audio transcription service not available")
                return
            try:
                provider_label = "OpenAI Whisper"
                app.logger.info("%s transcription started", provider_label)
                _job_update(job_id, progress=32, message=provider_label)
                model_name = (os.getenv("ENVID_OPENAI_WHISPER_MODEL") or "large-v3").strip() or "large-v3"
                _job_step_update(job_id, "transcribe", status="running", percent=5, message=f"Running ({provider_label}/{model_name})")

                whisper_device = (os.getenv("ENVID_OPENAI_WHISPER_DEVICE") or "auto").strip().lower()
                transcript_meta["openai_whisper"] = {
                    "model": model_name,
                    "device": whisper_device or "auto",
                }

                selection_language_hint = ""
                if isinstance(sel, dict):
                    selection_language_hint = _selection_language_hint(sel)
                raw_language_hint = selection_language_hint or (os.getenv("ENVID_OPENAI_WHISPER_LANGUAGE") or "hi").strip().lower()
                whisper_language = None if raw_language_hint in {"", "auto", "detect", "none"} else raw_language_hint
                fallback_auto = _env_truthy(os.getenv("ENVID_OPENAI_WHISPER_LANGUAGE_FALLBACK_AUTO"), default=True)

                def _call_transcribe_service(language_hint: str | None) -> dict[str, Any]:
                    with local_path.open("rb") as handle:
                        files = {"file": (local_path.name, handle, "application/octet-stream")}
                        data_payload = {"model": model_name}
                        if language_hint:
                            data_payload["language"] = language_hint
                        resp = requests.post(
                            f"{service_url}/transcribe",
                            files=files,
                            data=data_payload,
                            timeout=5400,
                        )
                    if resp.status_code >= 400:
                        raise RuntimeError(f"Audio transcription service failed ({resp.status_code}): {resp.text}")
                    payload = resp.json()
                    result = payload.get("result") if isinstance(payload, dict) else None
                    if not isinstance(result, dict):
                        raise RuntimeError("Invalid transcription service response")
                    return result

                data = _call_transcribe_service(whisper_language)
                segments_probe = data.get("segments") if isinstance(data, dict) else None
                if (
                    fallback_auto
                    and whisper_language
                    and (not isinstance(segments_probe, list) or not segments_probe)
                ):
                    data = _call_transcribe_service(None)
                    transcript_meta["openai_whisper"]["language_fallback"] = "auto"

                segments = data.get("segments") if isinstance(data, dict) else None
                if not isinstance(segments, list):
                    segments = []

                transcript_words = []
                transcript_segments = []
                seg_quality_issues: List[Dict[str, Any]] = []
                for seg in segments[:50000]:
                    if not isinstance(seg, dict):
                        continue
                    try:
                        st = float(seg.get("start") or 0.0)
                        en = float(seg.get("end") or st)
                        txt = str(seg.get("text") or "").strip()
                    except Exception:
                        continue
                    if not txt:
                        continue
                    words = seg.get("words") if isinstance(seg.get("words"), list) else []
                    for w in words:
                        if not isinstance(w, dict):
                            continue
                        transcript_words.append(
                            {
                                "word": str(w.get("word") or "").strip(),
                                "start": _safe_float(w.get("start"), 0.0),
                                "end": _safe_float(w.get("end"), 0.0),
                            }
                        )
                    transcript_segments.append({"start": st, "end": en, "text": txt})

                transcript = " ".join(
                    [str(s.get("text") or "").strip() for s in transcript_segments if (s.get("text") or "").strip()]
                ).strip()
                transcript_language_code = str((data.get("language") if isinstance(data, dict) else None) or whisper_language or "").strip()
                languages_detected = [transcript_language_code] if transcript_language_code else []

                transcript_meta.setdefault("openai_whisper", {})
                transcript_meta["openai_whisper"]["language_hint"] = whisper_language or "auto"
                transcript_meta["openai_whisper"]["source"] = "service"

                # Preserve raw transcript/segments for diagnostics before any corrections.
                raw_transcript = transcript
                raw_transcript_segments = [dict(s) for s in transcript_segments] if transcript_segments else []
                transcript_raw = raw_transcript
                transcript_raw_segments = raw_transcript_segments

                # Correction settings (applied to both transcript and segments).
                # Only use LLM correction; disable grammar/dictionary/punctuation helpers.
                grammar_enabled = False
                grammar_url = "http://translate:8010"
                grammar_local = False
                dictionary_enabled = False
                punctuation_enabled = False

                def _word_similarity(a: str, b: str) -> float:
                    a_words = re.findall(r"[\w\u0900-\u097F]+", a or "")
                    b_words = re.findall(r"[\w\u0900-\u097F]+", b or "")
                    if not a_words or not b_words:
                        return 0.0
                    return difflib.SequenceMatcher(None, a_words, b_words).ratio()

                min_similarity = _safe_float(os.getenv("ENVID_TRANSCRIPT_CORRECTION_MIN_SIMILARITY"), 0.75)

                # Apply full correction pipeline to the full transcript (for metadata/visibility).
                if transcript:
                    before = transcript
                    transcript, corr_meta = _apply_segment_corrections(
                        text=transcript,
                        language_code=(whisper_language or transcript_language_code or "hi"),
                        grammar_enabled=bool(grammar_enabled),
                        hindi_dictionary_enabled=bool(dictionary_enabled),
                        punctuation_enabled=bool(punctuation_enabled),
                    )
                    transcript_meta["grammar_correction"] = {
                        "enabled": True,
                        "available": bool(grammar_url) or (bool(grammar_local) and language_tool_python is not None),
                        "applied": bool(corr_meta.get("grammar_applied")),
                    }
                    lang_norm = _normalize_language_code(whisper_language or transcript_language_code or "hi")
                    transcript_meta["dictionary_correction"] = {
                        "enabled": bool(dictionary_enabled),
                        "language": lang_norm,
                        "language_name": _dictionary_language_name(lang_norm),
                        "applied": bool(corr_meta.get("dictionary_applied")),
                    }
                    transcript_meta["hindi_dictionary_correction"] = {
                        "enabled": bool(dictionary_enabled and lang_norm == "hi"),
                        "applied": bool(corr_meta.get("hindi_applied")),
                    }
                    transcript_meta["punctuation_enhance"] = {
                        "enabled": bool(punctuation_enabled),
                        "applied": bool(corr_meta.get("punctuation_applied")),
                    }

                # Keep time-band segments consistent with corrected transcript.
                segment_correction_enabled = True
                segment_mode = "llm_only"
                segment_max = 5000
                if segment_correction_enabled and transcript_segments:
                    corrected_segments: list[dict[str, Any]] = []
                    corrected_count = 0
                    skipped_low_similarity = 0
                    for seg in transcript_segments[:segment_max]:
                        if not isinstance(seg, dict):
                            continue
                        txt = str(seg.get("text") or "").strip()
                        if not txt:
                            corrected_segments.append(seg)
                            continue

                        seg_grammar = grammar_enabled and segment_mode == "full"
                        seg_punct = punctuation_enabled or segment_mode in {"full", "punctuation"}

                        corrected_txt, _seg_meta = _apply_segment_corrections(
                            text=txt,
                            language_code=(transcript_language_code or whisper_language or "hi"),
                            grammar_enabled=bool(seg_grammar),
                            hindi_dictionary_enabled=bool(dictionary_enabled),
                            punctuation_enabled=bool(seg_punct),
                        )
                        if corrected_txt and corrected_txt != txt:
                            similarity = _word_similarity(txt, corrected_txt)
                            if similarity < min_similarity:
                                skipped_low_similarity += 1
                            else:
                                seg = {**seg, "text": corrected_txt}
                                corrected_count += 1
                        corrected_segments.append(seg)

                    # Preserve any remaining segments without heavy correction.
                    if len(transcript_segments) > segment_max:
                        corrected_segments.extend(transcript_segments[segment_max:])

                    transcript_segments = corrected_segments
                    transcript_meta.setdefault("segment_correction", {})
                    transcript_meta["segment_correction"].update(
                        {
                            "enabled": True,
                            "mode": segment_mode,
                            "applied_count": int(corrected_count),
                            "skipped_low_similarity": int(skipped_low_similarity),
                            "max_segments": int(segment_max),
                            "min_similarity": float(min_similarity),
                        }
                    )

                    # Rebuild the raw transcript from corrected segments for consistency.
                    rebuilt = " ".join(
                        [
                            str(s.get("text") or "").strip()
                            for s in transcript_segments
                            if isinstance(s, dict) and (s.get("text") or "").strip()
                        ]
                    ).strip()
                    if rebuilt:
                        transcript = rebuilt
                        transcript_meta["correction_source"] = "segment_corrected"

                # If corrections significantly diverged, revert to raw transcript.
                if raw_transcript and transcript:
                    similarity = _word_similarity(raw_transcript, transcript)
                    transcript_meta["correction_similarity"] = float(similarity)
                    if similarity < min_similarity:
                        transcript = raw_transcript
                        if raw_transcript_segments:
                            transcript_segments = raw_transcript_segments
                        transcript_meta["correction_reverted"] = True

                # LLM verification + correction (OpenRouter primary, local fallback).
                try:
                    llm_data = _verify_transcript_via_llm_direct(
                        text=transcript,
                        language_code=transcript_language_code or whisper_language,
                    )
                    llm_meta = llm_data.get("meta") if isinstance(llm_data.get("meta"), dict) else {}
                    if not llm_meta.get("available"):
                        transcript_meta["llm_verification"] = {
                            "enabled": False,
                            "available": False,
                            "provider": llm_meta.get("provider") or "none",
                            **({"error": llm_meta.get("error")} if llm_meta.get("error") else {}),
                        }
                    else:
                        llm_ok = bool(llm_data.get("ok"))
                        llm_score = _safe_float(llm_data.get("score"), 0.0)
                        llm_min_score = _safe_float(os.getenv("ENVID_TRANSCRIPT_LLM_MIN_SCORE"), 0.6)
                        llm_provider = str(llm_meta.get("provider") or "").strip().lower()
                        llm_corrected = str(llm_data.get("corrected_text") or "").strip()
                        llm_issues = llm_data.get("issues") if isinstance(llm_data.get("issues"), list) else []
                        transcript_meta["llm_verification"] = {
                            "enabled": True,
                            "input_source": "final_transcript",
                            "score": float(llm_score),
                            "min_score": float(llm_min_score),
                            "ok": bool(llm_ok),
                            "issues": [str(x) for x in llm_issues if str(x).strip()],
                            "provider": llm_provider,
                            "model": llm_meta.get("model"),
                        }
                        if llm_corrected:
                            min_llm_similarity = _safe_float(os.getenv("ENVID_TRANSCRIPT_LLM_MIN_SIMILARITY"), 0.85)
                            similarity = _word_similarity(transcript, llm_corrected)
                            transcript_meta["llm_verification"]["corrected_similarity"] = float(similarity)
                            transcript_meta["llm_verification"]["min_similarity"] = float(min_llm_similarity)
                            if similarity >= min_llm_similarity:
                                transcript = llm_corrected
                                transcript_raw = llm_corrected
                                if transcript_segments:
                                    transcript_raw_segments = [dict(s) for s in transcript_segments]
                                transcript_meta["llm_verification"]["corrected_applied"] = True
                        if not (llm_ok and llm_score >= llm_min_score):
                            if llm_provider == "local":
                                transcript_meta["llm_verification"]["relaxed_for_local"] = True
                            else:
                                raise TranscriptVerificationError("Transcript verification failed: LLM check failed")
                except TranscriptVerificationError:
                    raise
                except Exception as exc:
                    app.logger.warning("LLM transcript verification failed: %s", exc)
                    transcript_meta["llm_verification"] = {
                        "enabled": True,
                        "ok": False,
                        "error": str(exc)[:240],
                    }

                # Strict verification: fail the job if transcript looks invalid.
                verify_strict = _env_truthy(os.getenv("ENVID_TRANSCRIPT_VERIFY_STRICT"), default=True)
                require_correction = _env_truthy(os.getenv("ENVID_TRANSCRIPT_VERIFY_REQUIRE_CORRECTION"), default=True)
                if verify_strict:
                    lang_for_check = transcript_language_code or whisper_language or "hi"
                    is_reasonable = _transcript_is_reasonable(transcript, lang_for_check)
                    transcript_meta["verification"] = {
                        "strict": True,
                        "require_correction": bool(require_correction),
                        "reasonable": bool(is_reasonable),
                    }
                    if not is_reasonable:
                        raise TranscriptVerificationError("Transcript verification failed: low-quality output")
                    if require_correction:
                        dictionary_applied = transcript_meta.get("dictionary_correction", {}).get("applied")
                        if dictionary_applied is None:
                            dictionary_applied = transcript_meta.get("hindi_dictionary_correction", {}).get("applied")
                        applied_flags = [
                            transcript_meta.get("grammar_correction", {}).get("applied"),
                            transcript_meta.get("punctuation_enhance", {}).get("applied"),
                            dictionary_applied,
                        ]
                        applied_any = any(bool(x) for x in applied_flags)
                        transcript_meta["verification"]["corrections_applied"] = bool(applied_any)
                        if not applied_any:
                            raise TranscriptVerificationError("Transcript verification failed: corrections not applied")

                if seg_quality_issues:
                    transcript_meta.setdefault("quality", {})
                    transcript_meta["quality"]["segment_flags"] = seg_quality_issues[:500]

                transcript_meta.setdefault("raw_transcript", raw_transcript)
                if raw_transcript_segments:
                    transcript_meta.setdefault("raw_segments", raw_transcript_segments)

                # Persist raw transcript in DB for reference.
                try:
                    _db_payload_insert(
                        job_id,
                        kind="transcript_raw",
                        payload={
                            "text": raw_transcript,
                            "language_code": transcript_language_code or whisper_language,
                            "segments": raw_transcript_segments,
                            "source": "openai_whisper",
                        },
                    )
                except Exception:
                    pass

                effective_models["transcribe"] = f"openai-whisper/{model_name}"
                _job_step_update(job_id, "transcribe", status="completed", percent=100, message=f"Completed ({provider_label}/{model_name})")
                transcribe_completed = True
                app.logger.info("%s transcription completed", provider_label)
            except TranscriptVerificationError as exc:
                app.logger.warning("Transcript verification failed: %s", exc)
                _job_step_update(job_id, "transcribe", status="failed", message=str(exc)[:240])
                transcribe_failed = True
                raise
            except Exception as exc:
                app.logger.warning("%s failed: %s", provider_label, exc)
                _job_step_update(job_id, "transcribe", status="failed", percent=100, message=str(exc)[:240])
                transcribe_failed = True
                # Fall through to the GCP path below.
                transcript = ""
                transcript_language_code = ""
                languages_detected = []
                transcript_words = []
                transcript_segments = []

        def _run_scene_detect_task() -> None:
            nonlocal precomputed_scenes, precomputed_scenes_source
            try:
                force_transnetv2 = _env_truthy(os.getenv("ENVID_METADATA_FORCE_TRANSNETV2_SCENES"), default=False)
                if enable_key_scene:
                    _job_step_update(job_id, "key_scene_detection", status="running", percent=5, message="Running (scene detection)")
                app.logger.info("Scene detection started")
                if use_transnetv2_for_scenes:
                    try:
                        precomputed_scenes = _transnetv2_list_scenes(video_path=local_path, temp_dir=temp_dir)
                        precomputed_scenes_source = "transnetv2"
                    except Exception as exc:
                        app.logger.warning("TransNetV2 failed: %s", exc)
                        precomputed_scenes = []
                        precomputed_scenes_source = "transnetv2_failed"
                if (not precomputed_scenes) and (
                    use_pyscenedetect_for_scenes or (precomputed_scenes_source == "transnetv2_failed" and not force_transnetv2)
                ):
                    try:
                        precomputed_scenes = _pyscenedetect_list_scenes(video_path=local_path, temp_dir=temp_dir)
                        precomputed_scenes_source = "pyscenedetect"
                    except Exception as exc:
                        app.logger.warning("PySceneDetect failed: %s", exc)
                        precomputed_scenes = []
                        precomputed_scenes_source = "pyscenedetect_failed"
                if enable_key_scene:
                    scene_count = len(precomputed_scenes or [])
                    source = precomputed_scenes_source or "unknown"
                    _job_step_update(
                        job_id,
                        "key_scene_detection",
                        status="completed",
                        percent=100,
                        message=f"Completed ({scene_count} scenes; {source})",
                    )
                app.logger.info("Scene detection completed")
            except Exception as exc:
                app.logger.warning("Scene detection failed: %s", exc)
                if enable_key_scene:
                    _job_step_update(job_id, "key_scene_detection", status="failed", percent=100, message=str(exc)[:240])

        use_vi_label_detection_effective = enable_label_detection and use_vi_label_detection

        def _run_vi_label_detection() -> None:
            nonlocal labels, video_intelligence, label_service_used
            try:
                _job_update(job_id, progress=12, message="Video Intelligence")
                _job_step_update(
                    job_id,
                    "label_detection",
                    status="running",
                    percent=0,
                    message=f"Running (Google Video Intelligence; mode={vi_label_mode})",
                )

                vi_client = gcp_video_intelligence.VideoIntelligenceServiceClient()
                features: List[Any] = []
                if use_vi_label_detection_effective:
                    features.append(gcp_video_intelligence.Feature.LABEL_DETECTION)
                if enable_objects:
                    try:
                        features.append(gcp_video_intelligence.Feature.OBJECT_TRACKING)
                    except Exception:
                        pass

                req: Dict[str, Any] = {"input_uri": gcs_label_uri, "features": features}

                # Optional label detection mode: segment (default) / shot / frame.
                try:
                    if vi_label_mode in {"shot", "frame"}:
                        vc = req.get("video_context") if isinstance(req.get("video_context"), dict) else {}
                        cfg = vc.get("label_detection_config") if isinstance(vc.get("label_detection_config"), dict) else {}
                        cfg["label_detection_mode"] = "SHOT_MODE" if vi_label_mode == "shot" else "FRAME_MODE"
                        vc["label_detection_config"] = cfg
                        req["video_context"] = vc
                except Exception:
                    pass

                timeout = _parse_int(
                    os.getenv("ENVID_METADATA_GCP_VIDEO_INTELLIGENCE_TIMEOUT_SECONDS"),
                    default=3600,
                    min_value=60,
                    max_value=21600,
                )
                operation = vi_client.annotate_video(request=req)
                response = operation.result(timeout=timeout)

                def _dur_seconds(d) -> float:
                    try:
                        return float(d.seconds) + float(d.nanos) / 1e9
                    except Exception:
                        return 0.0

                results = list(getattr(response, "annotation_results", None) or [])
                ar = results[0] if results else None

                labels_vi: List[Dict[str, Any]] = []
                objects: List[Dict[str, Any]] = []
                if ar is not None:
                    label_annotations: list[Any] = []
                    if vi_label_mode == "shot":
                        label_annotations = list(getattr(ar, "shot_label_annotations", None) or [])
                    elif vi_label_mode == "frame":
                        label_annotations = list(getattr(ar, "frame_label_annotations", None) or [])
                    else:
                        label_annotations = list(getattr(ar, "segment_label_annotations", None) or [])

                    for ann in label_annotations[:vi_max_labels]:
                        entity = getattr(ann, "entity", None)
                        desc = (getattr(entity, "description", None) or "").strip()
                        cats: List[str] = []
                        for c in (getattr(ann, "category_entities", None) or [])[:5]:
                            cd = (getattr(c, "description", None) or "").strip()
                            if cd:
                                cats.append(cd)

                        segs: List[Dict[str, Any]] = []
                        if vi_label_mode == "frame":
                            frames = list(getattr(ann, "frames", None) or [])[:vi_max_frames_per_label]
                            frame_len = float(
                                _parse_int(
                                    os.getenv("ENVID_METADATA_GCP_VI_FRAME_SEGMENT_SECONDS"),
                                    default=1,
                                    min_value=0,
                                    max_value=10,
                                )
                                or 1
                            )
                            for fr in frames:
                                t = _dur_seconds(getattr(fr, "time_offset", None))
                                conf = float(getattr(fr, "confidence", 0.0) or 0.0)
                                if conf < vi_min_label_conf:
                                    continue
                                segs.append({"start": t, "end": t + frame_len, "confidence": conf})
                            segs = segs[:vi_max_segments_per_label]
                        else:
                            for seg in (getattr(ann, "segments", None) or [])[:vi_max_segments_per_label]:
                                seg_obj = getattr(seg, "segment", None)
                                conf = float(getattr(seg, "confidence", 0.0) or 0.0)
                                if conf < vi_min_label_conf:
                                    continue
                                segs.append(
                                    {
                                        "start": _dur_seconds(getattr(seg_obj, "start_time_offset", None)),
                                        "end": _dur_seconds(getattr(seg_obj, "end_time_offset", None)),
                                        "confidence": conf,
                                    }
                                )

                        if desc and segs:
                            labels_vi.append({"label": desc, "categories": cats, "segments": segs})

                    if enable_objects:
                        try:
                            for oa in (getattr(ar, "object_annotations", None) or [])[:2000]:
                                ent = getattr(oa, "entity", None)
                                name = (getattr(ent, "description", None) or "").strip()
                                if not name:
                                    continue
                                segs: List[Dict[str, Any]] = []
                                for fr in (getattr(oa, "frames", None) or [])[:vi_max_frames_per_label]:
                                    t = _dur_seconds(getattr(fr, "time_offset", None))
                                    conf = float(getattr(fr, "confidence", 0.0) or 0.0)
                                    if conf < vi_min_object_conf:
                                        continue
                                    segs.append({"start": t, "end": t + 1.0, "confidence": conf})
                                segs = segs[:vi_max_segments_per_label]
                                if segs:
                                    objects.append({"name": name, "segments": segs})
                        except Exception:
                            pass

                if labels_vi:
                    labels = labels_vi

                video_intelligence = {
                    "source": "gcp_video_intelligence",
                    "config": {
                        "label_detection_mode": vi_label_mode,
                        "transcribe_mode": transcribe_effective_mode,
                        "requested_models": requested_models,
                    },
                    "labels": labels,
                }
                if objects:
                    video_intelligence["objects"] = objects

                _job_step_update(job_id, "label_detection", status="completed", percent=100, message=f"{len(labels)} labels")
            except Exception as exc:
                app.logger.warning("Video Intelligence failed: %s", exc)
                if use_vi_label_detection_effective:
                    _job_step_update(job_id, "label_detection", status="failed", message=str(exc))

        if not run_all_sequential:
            if enable_text_on_screen and use_local_ocr:
                local_ocr_started = True
                parallel_futures["local_ocr"] = _ensure_parallel_executor().submit(_run_local_ocr_task)

            if enable_moderation and use_local_moderation:
                local_moderation_started = True
                parallel_futures["local_moderation"] = _ensure_parallel_executor().submit(_run_local_moderation_task)

            if (enable_scene_by_scene or enable_key_scene or enable_high_point) and (use_transnetv2_for_scenes or use_pyscenedetect_for_scenes):
                scenes_started = True
                if sequential_scene_then_whisper:
                    _run_scene_detect_task()
                else:
                    parallel_futures["scene_detect"] = _ensure_parallel_executor().submit(_run_scene_detect_task)

            if enable_transcribe and (transcribe_effective_mode == "openai-whisper") and not _transcribe_service_available():
                _job_step_update(job_id, "transcribe", status="skipped", percent=100, message="Audio transcription service not available")
            elif enable_transcribe and (transcribe_effective_mode == "openai-whisper") and _transcribe_service_available():
                whisper_started = True
                if sequential_scene_then_whisper:
                    _run_whisper_transcription()
                else:
                    parallel_futures["whisper"] = _ensure_parallel_executor().submit(_run_whisper_transcription)

        if enable_text_on_screen and not use_local_ocr:
            _job_step_update(job_id, "text_on_screen", status="skipped", percent=100, message="Requires local OCR")

        if enable_moderation and not use_local_moderation:
            _job_step_update(job_id, "moderation", status="skipped", percent=100, message="Requires local moderation")

        if enable_vi and gcp_video_intelligence is not None and want_any_vi and use_vi_label_detection_effective:
            if run_all_sequential:
                _run_vi_label_detection()
            else:
                parallel_futures["label_detection"] = _ensure_parallel_executor().submit(_run_vi_label_detection)
        else:
            msg = "Disabled" if not enable_vi else ("Library not installed" if gcp_video_intelligence is None else "Disabled")
            if use_vi_label_detection_effective:
                _job_step_update(job_id, "label_detection", status="skipped", percent=100, message=msg)
            else:
                _job_step_update(job_id, "label_detection", status="skipped", percent=100, message="Disabled")

        if label_service_used:
            _job_step_update(job_id, "label_detection", status="completed", percent=100, message=f"{len(labels)} labels")

        if label_service_used and not isinstance(video_intelligence, dict):
            video_intelligence = {
                "source": "label_service",
                "config": {
                    "label_detection_mode": vi_label_mode,
                    "requested_models": requested_models,
                },
                "labels": labels,
            }

        if run_all_sequential:
            if enable_moderation and use_local_moderation:
                local_moderation_started = True
                _run_local_moderation_task()

            if enable_text_on_screen and use_local_ocr:
                local_ocr_started = True
                _run_local_ocr_task()

            if (enable_scene_by_scene or enable_key_scene or enable_high_point) and (use_transnetv2_for_scenes or use_pyscenedetect_for_scenes):
                scenes_started = True
                _run_scene_detect_task()

            if enable_transcribe and (transcribe_effective_mode == "openai-whisper") and not _transcribe_service_available():
                _job_step_update(job_id, "transcribe", status="skipped", percent=100, message="Audio transcription service not available")
            elif enable_transcribe and (transcribe_effective_mode == "openai-whisper") and _transcribe_service_available():
                whisper_started = True
                _run_whisper_transcription()

        if parallel_executor is not None:
            whisper_timeout = 5400.0
            scene_timeout = float(os.getenv("ENVID_METADATA_SCENE_DETECT_TIMEOUT_SECONDS") or 1200.0)
            default_timeout = float(os.getenv("ENVID_METADATA_PARALLEL_TASK_TIMEOUT_SECONDS") or 1200.0)
            for name, fut in list(parallel_futures.items()):
                timeout = default_timeout
                if name == "whisper":
                    timeout = whisper_timeout
                elif name == "scene_detect":
                    timeout = scene_timeout
                try:
                    fut.result(timeout=timeout if timeout and timeout > 0 else None)
                except FutureTimeoutError:
                    app.logger.warning("Parallel task %s timed out", name)
                    if name == "whisper":
                        _job_step_update(job_id, "transcribe", status="failed", percent=100, message="Transcription timed out")
                    elif name == "scene_detect":
                        _job_step_update(job_id, "key_scene_detection", status="failed", percent=100, message="Scene detect timed out")
                    elif name == "local_ocr":
                        _job_step_update(job_id, "text_on_screen", status="failed", percent=100, message="OCR timed out")
                    elif name == "local_moderation":
                        _job_step_update(job_id, "moderation", status="failed", percent=100, message="Moderation timed out")
                except TranscriptVerificationError:
                    raise
                except Exception as exc:
                    app.logger.warning("Parallel task %s failed: %s", name, exc)
            parallel_executor.shutdown(wait=True)

        if local_moderation_frames is not None:
            if not isinstance(video_intelligence, dict):
                video_intelligence = {}
            video_intelligence["moderation"] = {
                "explicit_frames": local_moderation_frames,
                "source": local_moderation_source or "local",
            }

        # Local Text-on-Screen OCR (when requested).
        if enable_text_on_screen and use_local_ocr and not local_ocr_started:
            _run_local_ocr_task()

        # Local moderation (when requested).
        if enable_moderation and use_local_moderation and not local_moderation_started:
            _run_local_moderation_task()

        enable_speech = _env_truthy(os.getenv("ENVID_METADATA_ENABLE_SPEECH"), default=True)
        if enable_transcribe and (transcribe_effective_mode == "gcp_speech") and enable_speech and gcp_speech is not None and not transcript:
            try:
                _job_update(job_id, progress=35, message="Speech-to-Text")
                _job_step_update(job_id, "transcribe", status="running", percent=0, message=f"Running (requested: {requested_models.get('transcribe_model')})")

                audio_path = temp_dir / "audio.flac"
                ffmpeg_service_url = _ffmpeg_service_url()
                if not ffmpeg_service_url:
                    raise RuntimeError("FFmpeg service URL is not set")
                audio_bytes = _extract_audio_via_ffmpeg_service(
                    service_url=ffmpeg_service_url,
                    video_path=local_path,
                    filename=local_path.name,
                    sample_rate=16000,
                    channels=1,
                    fmt="flac",
                )
                audio_path.write_bytes(audio_bytes)

                artifacts_bucket = _gcs_artifacts_bucket(gcs_bucket)
                artifacts_prefix = _gcs_artifacts_prefix()
                audio_obj = f"{artifacts_prefix}/{job_id}/audio.flac"
                client = _gcs_client()
                client.bucket(artifacts_bucket).blob(audio_obj).upload_from_filename(str(audio_path), content_type="audio/flac")
                audio_uri = f"gs://{artifacts_bucket}/{audio_obj}"

                speech_client = gcp_speech.SpeechClient()
                sel_lang = _selection_language_hint(sel)
                primary_lang = (
                    sel_lang
                    or (os.getenv("GCP_SPEECH_LANGUAGE_CODE") or os.getenv("SPEECH_LANGUAGE_CODE") or "hi-IN")
                ).strip() or "hi-IN"
                alt_langs_raw = (os.getenv("GCP_SPEECH_ALTERNATIVE_LANGUAGE_CODES") or "").strip()
                alt_langs = [x.strip() for x in alt_langs_raw.split(",") if x.strip()]
                alt_langs = [x for x in alt_langs if x != primary_lang][:4]

                config = gcp_speech.RecognitionConfig(
                    encoding=gcp_speech.RecognitionConfig.AudioEncoding.FLAC,
                    sample_rate_hertz=16000,
                    language_code=primary_lang,
                    alternative_language_codes=alt_langs,
                    enable_word_time_offsets=True,
                    enable_automatic_punctuation=True,
                    model=(os.getenv("GCP_SPEECH_MODEL") or "latest_long").strip() or "latest_long",
                )
                audio = gcp_speech.RecognitionAudio(uri=audio_uri)
                max_wait = _parse_int(os.getenv("ENVID_METADATA_GCP_SPEECH_TIMEOUT_SECONDS"), default=3600, min_value=60, max_value=21600)
                operation = speech_client.long_running_recognize(config=config, audio=audio)
                response = operation.result(timeout=max_wait)

                parts: List[str] = []
                result_segments: List[Dict[str, Any]] = []
                segment_cursor = 0.0

                def _dur_seconds(d) -> float:
                    try:
                        return float(d.seconds) + float(d.nanos) / 1e9
                    except Exception:
                        return 0.0

                for r in (response.results or []):
                    alt = r.alternatives[0] if (r.alternatives or []) else None
                    if not alt:
                        continue
                    if getattr(alt, "transcript", None):
                        txt = str(alt.transcript).strip()
                        if txt:
                            parts.append(txt)
                            # Fallback segmentation: some STT responses omit per-word timestamps.
                            # When available, use result_end_time to build coarse, time-banded segments.
                            end_t = _dur_seconds(getattr(r, "result_end_time", None))
                            if end_t and end_t >= segment_cursor:
                                result_segments.append({"start": float(segment_cursor), "end": float(end_t), "text": txt})
                                segment_cursor = float(end_t)
                    for w in (getattr(alt, "words", None) or []):
                        transcript_words.append({"word": (getattr(w, "word", None) or "").strip(), "start": _dur_seconds(getattr(w, "start_time", None)), "end": _dur_seconds(getattr(w, "end_time", None))})

                transcript = " ".join([p for p in parts if p]).strip()
                transcript_language_code = primary_lang
                languages_detected = [primary_lang] + alt_langs
                transcript_raw = transcript
                transcript_raw_segments = [dict(s) for s in transcript_segments] if transcript_segments else []

                effective_models["transcribe"] = "gcp_speech"
                # Prefer word-based segments when they have real timing.
                # If word timestamps are missing/zero, fall back to result-based segments.
                has_real_word_timing = any((float(w.get("end") or 0.0) > 0.0) for w in transcript_words)

                if transcript_words and has_real_word_timing:
                    seg: Dict[str, Any] = {"start": transcript_words[0].get("start") or 0.0, "end": 0.0, "text": ""}
                    for w in transcript_words:
                        st = float(w.get("start") or 0.0)
                        en = float(w.get("end") or st)
                        if (en - float(seg.get("start") or 0.0)) > 6.0 and (seg.get("text") or "").strip():
                            seg["end"] = float(seg.get("end") or en)
                            transcript_segments.append(seg)
                            seg = {"start": st, "end": en, "text": ""}
                        seg["end"] = max(float(seg.get("end") or 0.0), en)
                        seg["text"] = (str(seg.get("text") or "") + " " + str(w.get("word") or "")).strip()
                    if (seg.get("text") or "").strip():
                        transcript_segments.append(seg)
                elif result_segments:
                    transcript_segments = result_segments

                # Persist raw transcript in DB for reference.
                try:
                    _db_payload_insert(
                        job_id,
                        kind="transcript_raw",
                        payload={
                            "text": transcript_raw,
                            "language_code": transcript_language_code,
                            "segments": transcript_raw_segments,
                            "source": "gcp_speech",
                        },
                    )
                except Exception:
                    pass
                _job_step_update(job_id, "transcribe", status="completed", percent=100, message="Completed")
                transcribe_completed = True
            except Exception as exc:
                app.logger.warning("Speech-to-Text failed: %s", exc)
                _job_step_update(job_id, "transcribe", status="failed", message=str(exc))
                transcribe_failed = True
        else:
            if transcript:
                # Already completed via Video Intelligence.
                transcribe_completed = True
            else:
                if not enable_transcribe:
                    _job_step_update(job_id, "transcribe", status="skipped", percent=100, message="Disabled")
                else:
                    if transcribe_effective_mode == "openai-whisper":
                        if whisper_started:
                            _job_step_update(
                                job_id,
                                "transcribe",
                                status="failed",
                                percent=100,
                                message="Transcription produced no output",
                            )
                            transcribe_failed = True
                        else:
                            msg = (
                                "Audio transcription service not available"
                                if not _transcribe_service_available()
                                else "Disabled"
                            )
                            _job_step_update(job_id, "transcribe", status="skipped", percent=100, message=msg)
                    else:
                        msg = "Disabled" if not enable_speech else "Library not installed"
                        _job_step_update(job_id, "transcribe", status="skipped", percent=100, message=msg)

        # If we ran local labels and VI didn't populate label details, attach them.
        if local_labels:
            if not isinstance(video_intelligence, dict) or not video_intelligence:
                video_intelligence = {"source": "local", "config": {"requested_models": requested_models}}
            video_intelligence["labels"] = local_labels

        if local_text:
            if not isinstance(video_intelligence, dict) or not video_intelligence:
                video_intelligence = {"source": "local", "config": {"requested_models": requested_models}}
            video_intelligence["text"] = local_text

        subtitles: Dict[str, Any] = {}
        translations: Dict[str, Any] = {}
        if transcript and not transcript_segments:
            try:
                transcript_segments = [
                    {
                        "start": 0.0,
                        "end": float(duration_seconds or 0.0),
                        "text": str(transcript).strip(),
                    }
                ]
            except Exception:
                transcript_segments = [{"start": 0.0, "end": 0.0, "text": str(transcript).strip()}]

        if transcript or transcript_segments:
            transcript_script, script_meta = _build_transcript_script(
                transcript=transcript,
                transcript_segments=transcript_segments,
                language_code=transcript_language_code or None,
            )
            transcript_meta["script_processing"] = script_meta
        if transcript_raw or transcript_raw_segments:
            transcript_raw_script, _raw_script_meta = _build_transcript_script(
                transcript=transcript_raw,
                transcript_segments=transcript_raw_segments,
                language_code=transcript_language_code or None,
            )

        if transcript_segments:
            subtitles = {"language_code": transcript_language_code or ""}
            try:
                _subtitles_local_path(job_id, lang="orig", fmt="srt").write_text(_segments_to_srt(transcript_segments), encoding="utf-8")
                _subtitles_local_path(job_id, lang="orig", fmt="vtt").write_text(_segments_to_vtt(transcript_segments), encoding="utf-8")
            except Exception:
                pass
            source_lang = (transcript_language_code or "").strip().lower()
            if source_lang:
                subtitles.setdefault("orig", {"language_code": source_lang})
                try:
                    _subtitles_local_path(job_id, lang=source_lang, fmt="srt").write_text(
                        _segments_to_srt(transcript_segments),
                        encoding="utf-8",
                    )
                    _subtitles_local_path(job_id, lang=source_lang, fmt="vtt").write_text(
                        _segments_to_vtt(transcript_segments),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
                if source_lang.startswith("en"):
                    subtitles.setdefault("en", {"language_code": "en"})
                    try:
                        _subtitles_local_path(job_id, lang="en", fmt="srt").write_text(
                            _segments_to_srt(transcript_segments),
                            encoding="utf-8",
                        )
                        _subtitles_local_path(job_id, lang="en", fmt="vtt").write_text(
                            _segments_to_vtt(transcript_segments),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass

        translated_en = ""
        en_segments: List[Dict[str, Any]] = []

        locations: List[Dict[str, Any]] = []

        summary_executor: ThreadPoolExecutor | None = None
        synopsis_future = None

        def _ensure_summary_executor() -> ThreadPoolExecutor:
            nonlocal summary_executor
            if summary_executor is None:
                summary_executor = ThreadPoolExecutor(max_workers=2)
            return summary_executor

        synopsis: Dict[str, Any] = {}

        def _run_synopsis_generation() -> None:
            nonlocal synopsis, effective_models, critical_failures
            try:
                if enable_transcribe and not transcribe_completed:
                    effective_models["synopsis_generation"] = "transcribe_incomplete"
                    _job_step_update(job_id, "synopsis_generation", status="failed", percent=100, message="Transcript not ready")
                    critical_failures.append("synopsis_generation: transcript not ready")
                    return

                _job_update(job_id, progress=70, message="Synopsis")
                _job_step_update(job_id, "synopsis_generation", status="running", percent=0, message="Running")
                last_validation_failed = False

                synopsis_input = transcript_script or transcript or transcript_raw_script or transcript_raw or ""
                if not synopsis_input.strip():
                    effective_models["synopsis_generation"] = "no_transcript"
                    _job_step_update(job_id, "synopsis_generation", status="failed", percent=100, message="No transcript input")
                    critical_failures.append("synopsis_generation: no transcript input")
                    return
                def _try_synopsis_payload(payload: dict[str, Any] | None, model_label: str) -> bool:
                    nonlocal synopsis, effective_models, last_validation_failed
                    if isinstance(payload, dict):
                        if _validate_synopsis_payload(payload, transcript_language_code or "hi"):
                            if _synopsis_too_similar_to_transcript(
                                str(payload.get("short") or ""), synopsis_input
                            ) or _synopsis_too_similar_to_transcript(
                                str(payload.get("long") or ""), synopsis_input
                            ):
                                last_validation_failed = True
                                return False
                            synopsis = payload
                            effective_models["synopsis_generation"] = model_label
                            return True
                        last_validation_failed = True
                    return False
                max_attempts = _parse_int(
                    os.getenv("ENVID_SYNOPSIS_UNAVAILABLE_RETRIES"),
                    default=3,
                    min_value=1,
                    max_value=10,
                )
                gemini_attempts = 3
                def _log_provider_issue(provider: str, meta: dict[str, Any] | None, attempt_no: int) -> None:
                    if not isinstance(meta, dict):
                        return
                    if meta.get("available") is False:
                        reason = meta.get("reason") or meta.get("error") or meta.get("openrouter_error") or "unknown"
                        model_name = meta.get("model") or "unknown"
                        app.logger.warning(
                            "Synopsis provider unavailable (job=%s attempt=%s provider=%s model=%s reason=%s)",
                            job_id,
                            attempt_no,
                            provider,
                            model_name,
                            str(reason)[:240],
                        )
                gemini_unavailable = False
                for attempt in range(1, gemini_attempts + 1):
                    data_gem, meta_gem = _gemini_direct_generate_synopsis(
                        text=synopsis_input,
                        language_code=(transcript_language_code or "hi"),
                    )
                    _log_provider_issue("gemini", meta_gem if isinstance(meta_gem, dict) else None, attempt)
                    gemini_unavailable = isinstance(meta_gem, dict) and meta_gem.get("available") is False
                    if _try_synopsis_payload(
                        data_gem,
                        (meta_gem.get("model") if isinstance(meta_gem, dict) else None) or "gemini",
                    ):
                        _job_step_update(job_id, "synopsis_generation", status="completed", percent=100, message="Completed")
                        return

                for attempt in range(1, max_attempts + 1):
                    data_local, meta_local = _local_llm_generate_synopsis(
                        text=synopsis_input,
                        language_code=(transcript_language_code or "hi"),
                    )
                    _log_provider_issue("local", meta_local if isinstance(meta_local, dict) else None, attempt)
                    local_unavailable = isinstance(meta_local, dict) and meta_local.get("available") is False
                    local_ok = _try_synopsis_payload(
                        data_local,
                        (meta_local.get("model") if isinstance(meta_local, dict) else None) or "local",
                    )
                    if local_ok:
                        _job_step_update(job_id, "synopsis_generation", status="completed", percent=100, message="Completed")
                        return
                    if not local_unavailable and isinstance(meta_local, dict) and meta_local.get("available") is True:
                        app.logger.warning(
                            "Synopsis local response invalid (job=%s attempt=%s model=%s)",
                            job_id,
                            attempt,
                            meta_local.get("model") or "unknown",
                        )

                if gemini_unavailable and local_unavailable:
                    effective_models["synopsis_generation"] = "unavailable"
                    _job_step_update(job_id, "synopsis_generation", status="failed", percent=100, message="Unavailable")
                    critical_failures.append("synopsis_generation: unavailable")
                    return

                msg = "Invalid synopsis" if last_validation_failed else "Empty response"
                effective_models["synopsis_generation"] = "validation_failed" if last_validation_failed else "empty_response"
                _job_step_update(job_id, "synopsis_generation", status="failed", percent=100, message=msg)
                critical_failures.append(f"synopsis_generation: {msg}")
                return
            except Exception as exc:
                app.logger.warning("Synopsis failed: %s", exc)
                msg = str(exc)
                effective_models["synopsis_generation"] = "failed"
                _job_step_update(job_id, "synopsis_generation", status="failed", message=msg[:240])
                critical_failures.append(f"synopsis_generation: {msg[:200]}")

        if enable_synopsis_generation and (transcript_raw_script or transcript_raw or transcript_script or transcript):
            if run_all_sequential:
                _run_synopsis_generation()
            else:
                synopsis_future = _ensure_summary_executor().submit(_run_synopsis_generation)
        else:
            if not enable_synopsis_generation:
                effective_models["synopsis_generation"] = "disabled"
                msg = "Disabled"
            elif not (transcript_raw_script or transcript_raw or transcript_script or transcript):
                effective_models["synopsis_generation"] = "no_transcript"
                msg = "No transcript"
            else:
                effective_models["synopsis_generation"] = "not_configured"
                msg = "Not configured"
            _job_step_update(job_id, "synopsis_generation", status="skipped", percent=100, message=msg)
        scenes: List[Dict[str, Any]] = list(precomputed_scenes) if precomputed_scenes else []
        scenes_source = precomputed_scenes_source if precomputed_scenes else "none"
        key_scenes: list[dict[str, Any]] = []
        high_points: list[dict[str, Any]] = []
        if enable_scene_by_scene or enable_key_scene or enable_high_point:
            # Prefer local engines when explicitly selected.
            if (not scenes) and use_transnetv2_for_scenes:
                try:
                    scenes = _transnetv2_list_scenes(video_path=local_path, temp_dir=temp_dir)
                    scenes_source = "transnetv2"
                except Exception as exc:
                    app.logger.warning("TransNetV2 failed: %s", exc)
                    scenes = []
                    scenes_source = "transnetv2_failed"

            if (not scenes) and use_pyscenedetect_for_scenes:
                try:
                    scenes = _pyscenedetect_list_scenes(video_path=local_path, temp_dir=temp_dir)
                    scenes_source = "pyscenedetect"
                except Exception as exc:
                    app.logger.warning("PySceneDetect failed: %s", exc)
                    scenes = []
                    scenes_source = "pyscenedetect_failed"

            # Otherwise, derive from VI shots (only when explicitly requested).
            if (not scenes) and want_vi_shots:
                try:
                    shots = (video_intelligence.get("shots") or []) if isinstance(video_intelligence, dict) else []
                    for i, s in enumerate(shots[:200]):
                        st = float(s.get("start") or 0.0)
                        en = float(s.get("end") or st)
                        scenes.append({"index": i, "start": st, "end": en})
                    scenes_source = "gcp_video_intelligence_shots" if shots else scenes_source
                except Exception:
                    scenes = []

            if scenes:
                scenes = _normalize_scene_segments(scenes=scenes, duration_seconds=duration_seconds)

            # Optional fallback: if scene detection yields a single segment on longer videos,
            # generate uniform segments (controlled via env).
            allow_scene_fallback = _env_truthy(os.getenv("ENVID_SCENE_FALLBACK_ENABLED"), default=True)
            if allow_scene_fallback and (not scenes or len(scenes) <= 1):
                fallback_duration = float(duration_seconds) if duration_seconds and float(duration_seconds) > 0 else None
                if fallback_duration is None and scenes:
                    try:
                        fallback_duration = max(float(s.get("end") or s.get("end_seconds") or 0.0) for s in scenes)
                    except Exception:
                        fallback_duration = None
                if fallback_duration and fallback_duration > 0:
                    fallback = _fallback_scene_segments(duration_seconds=float(fallback_duration))
                    if fallback:
                        scenes = fallback
                        scenes_source = "fallback_uniform"

            # Force uniform segmentation over full duration when requested.
            force_uniform = _env_truthy(os.getenv("ENVID_SCENE_FORCE_UNIFORM"), default=False)
            if force_uniform:
                fallback_duration = float(duration_seconds) if duration_seconds and float(duration_seconds) > 0 else None
                if fallback_duration and fallback_duration > 0:
                    fallback = _fallback_scene_segments(duration_seconds=float(fallback_duration))
                    if fallback:
                        scenes = fallback
                        scenes_source = "fallback_uniform_forced"

            # Optional: OpenRouter (Meta Llama) scene-by-scene summaries.
            scene_llm_mode = (requested_models.get("scene_by_scene_metadata_model") or "").strip().lower()
            use_scene_llm = scene_llm_mode in {"openrouter_llama", "openrouter", "llama"} or _env_truthy(
                os.getenv("ENVID_SCENE_BY_SCENE_LLM"),
                default=False,
            )
            if enable_scene_by_scene and use_scene_llm and scenes:
                try:
                    labels_src = None
                    if isinstance(video_intelligence, dict):
                        labels_src = video_intelligence.get("labels")
                        if not isinstance(labels_src, list):
                            labels_src = None
                    objects_src = None
                    if isinstance(video_intelligence, dict):
                        objects_src = video_intelligence.get("objects")
                        if not isinstance(objects_src, list):
                            objects_src = None
                    summaries, meta = _openrouter_llama_scene_summaries(
                        scenes=scenes,
                        transcript_segments=transcript_segments,
                        labels_src=labels_src,
                        objects_src=objects_src,
                        language_code=transcript_language_code or "",
                    )
                    if summaries:
                        for sc in scenes:
                            if not isinstance(sc, dict):
                                continue
                            try:
                                idx = int(sc.get("index") or 0)
                            except Exception:
                                continue
                            summary = summaries.get(idx)
                            if summary:
                                sc["summary_llm"] = summary
                    if objects_src:
                        for sc in scenes:
                            if not isinstance(sc, dict):
                                continue
                            try:
                                st = float(sc.get("start") or sc.get("start_seconds") or 0.0)
                                en = float(sc.get("end") or sc.get("end_seconds") or st)
                            except Exception:
                                st, en = 0.0, 0.0
                            if en < st:
                                st, en = en, st
                            obj_names: list[str] = []
                            for obj in objects_src:
                                if not isinstance(obj, dict):
                                    continue
                                name = str(obj.get("name") or obj.get("label") or "").strip()
                                if not name:
                                    continue
                                for seg in (obj.get("segments") or []):
                                    if not isinstance(seg, dict):
                                        continue
                                    if _overlap_seconds(
                                        _safe_float(seg.get("start"), 0.0),
                                        _safe_float(seg.get("end"), 0.0),
                                        st,
                                        en,
                                    ) <= 0:
                                        continue
                                    obj_names.append(name)
                                    break
                            if obj_names:
                                sc["objects"] = list(dict.fromkeys(obj_names))
                    if meta.get("applied"):
                        effective_models["scene_by_scene_metadata"] = meta.get("model") or "openrouter"
                    else:
                        effective_models["scene_by_scene_metadata"] = "openrouter_unavailable"
                except Exception as exc:
                    app.logger.warning("Scene-by-scene LLM summary failed: %s", exc)

            if scenes:
                ranking_error: str | None = None
                if enable_key_scene or enable_high_point:
                    if not key_scene_step_finalized and enable_key_scene:
                        _job_step_update(job_id, "key_scene_detection", status="running", percent=60, message="Ranking key scenes")
                    key_scene_all_scenes = _env_truthy(os.getenv("ENVID_METADATA_KEY_SCENE_ALL_SCENES"), default=False)
                    if key_scene_all_scenes:
                        key_scenes = [dict(sc) for sc in scenes if isinstance(sc, dict)]
                        for sc in key_scenes:
                            sc.setdefault("reason", "full_video")
                            sc.setdefault("score", 1.0)
                        high_points = []
                    else:
                        try:
                            top_k = _parse_int(os.getenv("ENVID_METADATA_KEY_SCENE_TOP_K"), default=10, min_value=1, max_value=50)
                        except Exception:
                            top_k = 10
                        try:
                            key_scenes, high_points = _select_key_scenes_eventful(
                                scenes=scenes,
                                scenes_source=scenes_source,
                                video_intelligence=video_intelligence,
                                transcript_segments=transcript_segments,
                                local_path=local_path,
                                temp_dir=temp_dir,
                                top_k=int(top_k),
                                use_clip_cluster=bool(use_clip_cluster_for_key_scenes),
                            )
                        except Exception as exc:
                            app.logger.warning("Key scene ranking failed: %s", exc)
                            key_scenes = []
                            high_points = []
                            ranking_error = str(exc)[:240]

                if enable_scene_by_scene:
                    _job_step_update(
                        job_id,
                        "scene_by_scene_metadata",
                        status="completed",
                        percent=100,
                        message=f"{len(scenes)} scenes ({scenes_source}; requested: {requested_models.get('scene_by_scene_metadata_model')})",
                    )
                else:
                    _job_step_update(job_id, "scene_by_scene_metadata", status="skipped", percent=100, message="Disabled")

                if enable_key_scene and (not key_scene_step_finalized):
                    hp_msg = "none"
                    try:
                        if high_points:
                            best = high_points[0]
                            hp_msg = f"{float(best.get('start_seconds') or 0.0):.2f}-{float(best.get('end_seconds') or 0.0):.2f}s"
                    except Exception:
                        hp_msg = "none"
                    if ranking_error:
                        _job_step_update(job_id, "key_scene_detection", status="failed", percent=100, message=ranking_error)
                        critical_failures.append(f"key_scene_detection: {ranking_error}")
                    else:
                        if key_scene_all_scenes:
                            _job_step_update(
                                job_id,
                                "key_scene_detection",
                                status="completed",
                                percent=100,
                                message=f"{len(key_scenes)} scenes (full video; {scenes_source})",
                            )
                        else:
                            _job_step_update(
                                job_id,
                                "key_scene_detection",
                                status="completed",
                                percent=100,
                                message=(
                                    f"{min(len(key_scenes) or 0, 10)} key scenes; high point: {hp_msg} "
                                    f"({scenes_source}; requested: {requested_models.get('key_scene_detection_model')})"
                                ),
                            )
                else:
                    if not key_scene_step_finalized:
                        _job_step_update(job_id, "key_scene_detection", status="skipped", percent=100, message="Disabled")
            else:
                local_model_requested = requested_key_scene_model in {
                    "transnetv2",
                    "pyscenedetect",
                    "transnetv2_clip_cluster",
                    "pyscenedetect_clip_cluster",
                }
                if enable_scene_by_scene:
                    if (use_transnetv2_for_scenes or use_pyscenedetect_for_scenes) and (not want_vi_shots):
                        _job_step_update(job_id, "scene_by_scene_metadata", status="skipped", percent=100, message=f"No scenes ({scenes_source})")
                    else:
                        _job_step_update(job_id, "scene_by_scene_metadata", status="skipped", percent=100, message="No scenes")
                else:
                    _job_step_update(job_id, "scene_by_scene_metadata", status="skipped", percent=100, message="Disabled")
                if enable_key_scene and (not key_scene_step_finalized):
                    if local_model_requested and (use_transnetv2_for_scenes or use_pyscenedetect_for_scenes):
                        _job_step_update(job_id, "key_scene_detection", status="skipped", percent=100, message=f"No scenes ({scenes_source})")
                    else:
                        _job_step_update(job_id, "key_scene_detection", status="skipped", percent=100, message="No scenes")
                else:
                    if not key_scene_step_finalized:
                        _job_step_update(job_id, "key_scene_detection", status="skipped", percent=100, message="Disabled")
        else:
            _job_step_update(job_id, "scene_by_scene_metadata", status="skipped", percent=100, message="Disabled")
            if not key_scene_step_finalized:
                _job_step_update(job_id, "key_scene_detection", status="skipped", percent=100, message="Disabled")

        if synopsis_future is not None:
            try:
                synopsis_future.result(timeout=float(os.getenv("ENVID_METADATA_SYNOPSIS_TIMEOUT_SECONDS") or 900.0))
            except FutureTimeoutError:
                _job_step_update(job_id, "synopsis_generation", status="failed", percent=100, message="Synopsis timed out")
                critical_failures.append("synopsis_generation: timed out")
            except Exception as exc:
                _job_step_update(job_id, "synopsis_generation", status="failed", percent=100, message=str(exc)[:240])
                critical_failures.append(f"synopsis_generation: {str(exc)[:200]}")
        if summary_executor is not None:
            summary_executor.shutdown(wait=True)

        # Famous locations are not implemented in the slim stack; always skip.
        enable_famous_locations = False
        _job_step_update(job_id, "famous_location_detection", status="skipped", percent=100, message="Disabled")

        opening_closing: Dict[str, Any] | None = None

        metadata_text = " ".join([p for p in [video_title, video_description, transcript] if (p or "").strip()])

        thumbnail_base64: str | None = None
        try:
            def _best_scene_timestamp(items: list[dict[str, Any]]) -> float | None:
                if not isinstance(items, list) or not items:
                    return None
                best = None
                best_score = None
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    score = item.get("score")
                    try:
                        score_val = float(score) if score is not None else None
                    except Exception:
                        score_val = None
                    if best is None or (score_val is not None and (best_score is None or score_val > best_score)):
                        best = item
                        best_score = score_val
                if not isinstance(best, dict):
                    return None
                st = best.get("start_seconds") or best.get("start")
                en = best.get("end_seconds") or best.get("end") or st
                try:
                    st_f = float(st or 0.0)
                    en_f = float(en or st_f)
                except Exception:
                    return None
                if en_f < st_f:
                    st_f, en_f = en_f, st_f
                return st_f + max(0.0, (en_f - st_f) * 0.5)

            thumb_ts = _best_scene_timestamp(high_points) or _best_scene_timestamp(key_scenes)
            if thumb_ts is None:
                thumb_ts = float(duration_seconds or 0.0) / 2.0 if duration_seconds else 0.0

            out_path = temp_dir / "thumb.jpg"
            ffmpeg_service_url = _ffmpeg_service_url()
            if ffmpeg_service_url:
                thumb_bytes = _extract_frame_via_ffmpeg_service(
                    service_url=ffmpeg_service_url,
                    video_path=local_path,
                    filename=local_path.name,
                    timestamp=float(thumb_ts),
                    scale=224,
                    quality=2,
                )
                out_path.write_bytes(thumb_bytes)
                if out_path.exists():
                    thumbnail_base64 = base64.b64encode(out_path.read_bytes()).decode("utf-8")
        except Exception:
            pass

        stored_filename = ""
        if _persist_local_video_copy():
            file_extension = Path(gcs_object).suffix or ".mp4"
            stored_filename = f"{job_id}{file_extension}"
            shutil.copy2(local_path, VIDEOS_DIR / stored_filename)
            _db_file_upsert(
                job_id,
                kind="history_video",
                path=str(VIDEOS_DIR / stored_filename),
                gcs_uri=gcs_uri,
            )

        file_size_bytes: int | None = None
        try:
            file_size_bytes = int(local_path.stat().st_size)
        except Exception:
            file_size_bytes = None

        # Add stable timecodes + one-line summaries for key scenes/high points.
        fps_for_timecode = _video_fps_from_ffprobe(technical_ffprobe)
        try:
            for item in key_scenes:
                if not isinstance(item, dict):
                    continue
                st = float(item.get("start_seconds") or 0.0)
                en = float(item.get("end_seconds") or st)
                reasons = item.get("reasons")
                if isinstance(reasons, list):
                    reasons_str = ", ".join([str(r) for r in reasons if str(r).strip()])
                else:
                    reasons_str = str(item.get("reason") or "").strip()
                score = item.get("score")
                score_str = ""
                try:
                    score_str = f" score={float(score):.2f}" if score is not None else ""
                except Exception:
                    score_str = ""

                st_tc = _seconds_to_timecode(st, fps=fps_for_timecode)
                en_tc = _seconds_to_timecode(en, fps=fps_for_timecode)
                item["start_timecode"] = st_tc
                item["end_timecode"] = en_tc
                # Example: 00:00:10:11-00:00:14:27 key_scene reasons=explicit, text, scene_change score=6.11
                tail = f" reasons={reasons_str}" if reasons_str else ""
                item["summary"] = f"{st_tc}-{en_tc} key_scene{tail}{score_str}".strip()

            for item in high_points:
                if not isinstance(item, dict):
                    continue
                st = float(item.get("start_seconds") or 0.0)
                en = float(item.get("end_seconds") or st)
                reason = str(item.get("reason") or "").strip()
                score = item.get("score")
                score_str = ""
                try:
                    score_str = f" score={float(score):.2f}" if score is not None else ""
                except Exception:
                    score_str = ""

                st_tc = _seconds_to_timecode(st, fps=fps_for_timecode)
                en_tc = _seconds_to_timecode(en, fps=fps_for_timecode)
                item["start_timecode"] = st_tc
                item["end_timecode"] = en_tc
                tail = f" reason={reason}" if reason else ""
                item["summary"] = f"{st_tc}-{en_tc} high_point{tail}{score_str}".strip()
        except Exception:
            # Best-effort only; never fail the job for summary formatting.
            pass

        enable_translate = _env_truthy(os.getenv("ENVID_METADATA_ENABLE_TRANSLATE"), default=True)
        translate_provider = _translate_provider()
        has_text_to_translate = bool(transcript_segments) or bool((transcript or "").strip())
        requested_targets = _translate_targets_from_selection(sel)
        translate_targets = requested_targets or _translate_targets()
        translations_meta: Dict[str, Any] = {
            "enabled": bool(enable_translate),
            "provider": translate_provider,
            "targets": translate_targets,
        }
        translations_by_lang: Dict[str, Any] = {}
        gcp_translate_client = None
        gcp_translate_parent = None
        if enable_translate and translate_provider != "disabled" and has_text_to_translate:
            try:
                _job_step_update(job_id, "translate_output", status="running", percent=0, message="Running")
                _job_update(job_id, progress=85, message="Translate outputs")
                src = transcript_language_code.strip() or None

                if translate_provider == "gcp_translate":
                    gcp_translate_client, gcp_translate_parent = _init_translate_client("gcp_translate")

                segments_for_translate = transcript_segments
                if not segments_for_translate:
                    segments_for_translate = [{"start": 0.0, "end": float(duration_seconds or 0.0), "text": (transcript or "").strip()}]

                for lang in translate_targets:
                    if not lang:
                        continue
                    translated_segments = _translate_segments(
                        segments=segments_for_translate,
                        source_lang=src,
                        target_lang=lang,
                        provider=translate_provider,
                        gcp_client=gcp_translate_client,
                        gcp_parent=gcp_translate_parent,
                    )
                    if not translated_segments:
                        continue
                    translated_text = " ".join(
                        [str(s.get("text") or "").strip() for s in translated_segments if str(s.get("text") or "").strip()]
                    )
                    translations_by_lang[lang] = {
                        "transcript": {
                            "text": translated_text,
                            "segments": translated_segments,
                            "language_code": lang,
                        }
                    }
                    _subtitles_local_path(job_id, lang=lang, fmt="srt").write_text(_segments_to_srt(translated_segments), encoding="utf-8")
                    _subtitles_local_path(job_id, lang=lang, fmt="vtt").write_text(_segments_to_vtt(translated_segments), encoding="utf-8")
                    subtitles[lang] = {"language_code": lang}

                source_lang = (transcript_language_code or "").strip().lower()
                if source_lang and source_lang not in translations_by_lang:
                    translations_by_lang[source_lang] = {
                        "transcript": {
                            "text": str(transcript or "").strip(),
                            "segments": transcript_segments,
                            "language_code": source_lang,
                        }
                    }
                if translations_by_lang:
                    _job_step_update(job_id, "translate_output", status="completed", percent=100, message=f"{len(translations_by_lang)} languages")
            except Exception as exc:
                app.logger.warning("Translate (for locations) failed: %s", exc)
                _job_step_update(job_id, "translate_output", status="failed", message=str(exc)[:240])
        elif enable_translate:
            translations_meta["enabled"] = False
            _job_step_update(job_id, "translate_output", status="skipped", percent=100, message="No text to translate")
        else:
            _job_step_update(job_id, "translate_output", status="skipped", percent=100, message="Disabled")

        source_lang = (transcript_language_code or "").strip().lower()
        if transcript_segments:
            if source_lang and source_lang not in translations_by_lang:
                translations_by_lang[source_lang] = {
                    "transcript": {
                        "text": str(transcript or "").strip(),
                        "segments": transcript_segments,
                        "language_code": source_lang,
                    }
                }
            subtitles.setdefault("orig", {"language_code": source_lang or ""})

        # Build translated metadata payload (best-effort).
        if translations_by_lang:
            translations = {"meta": translations_meta, "languages": list(translations_by_lang.keys()), "by_language": {}}
            for lang, payload in translations_by_lang.items():
                by_lang: Dict[str, Any] = dict(payload)

                # Translate synopsis.
                if synopsis:
                    syn_tr: Dict[str, Any] = {}
                    short = _translate_text(
                        text=str(synopsis.get("short") or ""),
                        source_lang=transcript_language_code,
                        target_lang=lang,
                        provider=translate_provider,
                        gcp_client=gcp_translate_client,
                        gcp_parent=gcp_translate_parent,
                    )
                    long = _translate_text(
                        text=str(synopsis.get("long") or ""),
                        source_lang=transcript_language_code,
                        target_lang=lang,
                        provider=translate_provider,
                        gcp_client=gcp_translate_client,
                        gcp_parent=gcp_translate_parent,
                    )
                    syn_tr = {"short": short, "long": long}
                    if syn_tr:
                        by_lang["synopsis"] = syn_tr

                # Translate scene-by-scene summaries and transcript snippets.
                if isinstance(scenes, list) and scenes and isinstance(transcript_segments, list):
                    scene_out: list[dict[str, Any]] = []
                    for idx, sc in enumerate(scenes):
                        if not isinstance(sc, dict):
                            continue
                        try:
                            st = float(sc.get("start") or sc.get("start_seconds") or 0.0)
                            en = float(sc.get("end") or sc.get("end_seconds") or st)
                        except Exception:
                            continue
                        if en < st:
                            st, en = en, st
                        segs: list[dict[str, Any]] = []
                        for seg in transcript_segments:
                            if not isinstance(seg, dict):
                                continue
                            ss = _safe_float(seg.get("start"), 0.0)
                            se = _safe_float(seg.get("end"), ss)
                            if se <= ss:
                                continue
                            if _overlap_seconds(ss, se, st, en) > 0:
                                segs.append(seg)
                        translated_segs = _translate_segments(
                            segments=segs,
                            source_lang=transcript_language_code,
                            target_lang=lang,
                            provider=translate_provider,
                            gcp_client=gcp_translate_client,
                            gcp_parent=gcp_translate_parent,
                        )
                        summary_text = " ".join(
                            [str(s.get("text") or "").strip() for s in translated_segs if str(s.get("text") or "").strip()]
                        ).strip()
                        scene_out.append(
                            {
                                "index": int(sc.get("index") or idx),
                                "scene_index": int(sc.get("index") or idx),
                                "start_seconds": st,
                                "end_seconds": en,
                                "summary_text": summary_text[:320] if summary_text else "",
                                "transcript_segments": translated_segs,
                            }
                        )
                    if scene_out:
                        by_lang["scene_by_scene_metadata"] = {
                            "scenes": scene_out,
                            "source": scenes_source or "unknown",
                        }

                # Translate key scene/high point summaries.
                if isinstance(key_scenes, list) and key_scenes:
                    ks_out: list[dict[str, Any]] = []
                    for item in key_scenes:
                        if not isinstance(item, dict):
                            continue
                        summary = str(item.get("summary") or "").strip()
                        if summary:
                            ks_out.append(
                                {
                                    "summary": _translate_text(
                                        text=summary,
                                        source_lang="en",
                                        target_lang=lang,
                                        provider=translate_provider,
                                        gcp_client=gcp_translate_client,
                                        gcp_parent=gcp_translate_parent,
                                    )
                                }
                            )
                    if ks_out:
                        by_lang["key_scenes"] = ks_out

                if isinstance(high_points, list) and high_points:
                    hp_out: list[dict[str, Any]] = []
                    for item in high_points:
                        if not isinstance(item, dict):
                            continue
                        summary = str(item.get("summary") or "").strip()
                        if summary:
                            hp_out.append(
                                {
                                    "summary": _translate_text(
                                        text=summary,
                                        source_lang="en",
                                        target_lang=lang,
                                        provider=translate_provider,
                                        gcp_client=gcp_translate_client,
                                        gcp_parent=gcp_translate_parent,
                                    )
                                }
                            )
                    if hp_out:
                        by_lang["high_points"] = hp_out

                # Translate on-screen text (OCR).
                if isinstance(video_intelligence, dict) and isinstance(video_intelligence.get("text"), list):
                    text_items: list[dict[str, Any]] = []
                    for t in video_intelligence.get("text") or []:
                        if not isinstance(t, dict):
                            continue
                        raw_text = str(t.get("text") or "").strip()
                        if not raw_text:
                            continue
                        text_items.append(
                            {
                                "text": _translate_text(
                                    text=raw_text,
                                    source_lang=transcript_language_code,
                                    target_lang=lang,
                                    provider=translate_provider,
                                    gcp_client=gcp_translate_client,
                                    gcp_parent=gcp_translate_parent,
                                )
                            }
                        )
                    if text_items:
                        by_lang["on_screen_text"] = text_items

                translations["by_language"][lang] = by_lang

        def _step_map(job_id: str) -> dict[str, dict[str, Any]]:
            with JOBS_LOCK:
                job = JOBS.get(job_id) or {}
                steps = job.get("steps") or []
            out: dict[str, dict[str, Any]] = {}
            for s in steps:
                if isinstance(s, dict) and s.get("id"):
                    out[str(s.get("id"))] = s
            return out

        # Fail fast if any required step was skipped or failed.
        step_map = _step_map(job_id)
        required_steps: list[str] = [
            "upload_to_cloud_storage",
            "technical_metadata",
            "transcode_normalize",
        ]
        if enable_label_detection:
            required_steps.append("label_detection")
        if enable_moderation:
            required_steps.append("moderation")
        if enable_text_on_screen:
            required_steps.append("text_on_screen")
        if enable_key_scene or enable_high_point:
            required_steps.append("key_scene_detection")
        if enable_scene_by_scene:
            required_steps.append("scene_by_scene_metadata")
        if enable_transcribe:
            required_steps.append("transcribe")
        if enable_synopsis_generation:
            required_steps.append("synopsis_generation")
        if enable_translate_output:
            required_steps.append("translate_output")

        step_failures: list[str] = []
        allowed_skip_messages = {
            "audio transcription service not available",
            "no transcript",
            "no text to translate",
            "unavailable",
            "disabled",
            "not implemented",
        }
        for step_id in required_steps:
            step = step_map.get(step_id) or {}
            status = str(step.get("status") or "").strip().lower()
            msg = str(step.get("message") or "").strip()
            msg_lower = msg.lower()
            if status == "skipped" and msg_lower in allowed_skip_messages:
                continue
            if status in {"failed", "skipped"}:
                if msg:
                    step_failures.append(f"{step_id}:{status}:{msg}")
                else:
                    step_failures.append(f"{step_id}:{status}")
        if step_failures:
            raise RuntimeError("Required steps failed or skipped: " + "; ".join(step_failures)[:400])

        _job_step_update(job_id, "save_as_json", status="running", percent=0, message="Saving")
        task_durations, task_duration_total = _job_step_durations(job_id)
        video_entry: Dict[str, Any] = {
            "id": job_id,
            "title": video_title,
            "description": video_description,
            "original_filename": Path(gcs_object).name,
            "stored_filename": stored_filename,
            "file_path": f"videos/{stored_filename}" if stored_filename else "",
            "gcs_video_uri": gcs_uri,
            "transcript": transcript,
            "transcript_script": transcript_script,
            "transcript_raw": transcript_raw,
            "transcript_raw_script": transcript_raw_script,
            "transcript_raw_segments": transcript_raw_segments,
            "language_code": transcript_language_code,
            "languages_detected": languages_detected,
            "transcript_words": transcript_words,
            "transcript_segments": transcript_segments,
            "transcript_meta": transcript_meta,
            "video_intelligence": video_intelligence,
            "thumbnail": thumbnail_base64,
            "metadata_text": metadata_text,
            "duration_seconds": duration_seconds,
            "uploaded_at": datetime.utcnow().isoformat(),
            "technical_ffprobe": technical_ffprobe,
            "file_size_bytes": file_size_bytes,
            "subtitles": subtitles,
            "locations": locations,
            "famous_locations": {"locations": locations},
            "scenes": scenes,
            "scenes_source": scenes_source,
            "key_scenes": key_scenes,
            "high_points": high_points,
            "synopsis": synopsis,
            "translations": translations,
            "output_profile": "multimodal",
            "task_selection": sel,
            "task_selection_requested_models": requested_models,
            "task_selection_effective": {
                "label_detection_mode": vi_label_mode,
                "transcribe_mode": transcribe_effective_mode,
                "effective_models": effective_models,
            },
            "opening_closing_credit_detection": opening_closing,
            "task_durations": task_durations,
            "task_duration_total_seconds": float(task_duration_total),
        }

        categorized = _build_categorized_metadata_json(video_entry)
        video_entry["metadata_categories"] = categorized.get("categories")
        video_entry["metadata_combined"] = categorized.get("combined")

        with VIDEO_INDEX_LOCK:
            existing_idx = next((i for i, v in enumerate(VIDEO_INDEX) if str(v.get("id")) == str(job_id)), None)
            if existing_idx is not None:
                VIDEO_INDEX[existing_idx] = video_entry
            else:
                VIDEO_INDEX.append(video_entry)
        _save_video_index()

        _job_step_update(job_id, "save_as_json", status="completed", percent=100, message="Completed")

        # Best-effort: persist derived artifacts to GCS so this service can be run cloud-only
        # (no reliance on local subtitle/zip generation).
        try:
            _job_update(job_id, progress=92, message="Uploading artifacts to GCS")
            _job_step_update(job_id, "upload_artifacts", status="running", percent=10, message="Preparing")
            payload = {"id": job_id, "categories": video_entry.get("metadata_categories") or {}, "combined": video_entry.get("metadata_combined") or {}}
            _job_step_update(job_id, "upload_artifacts", status="running", percent=50, message="Uploading")
            video_entry["gcs_artifacts"] = _upload_metadata_artifacts(job_id=job_id, payload=payload)
            _record_job_artifacts_outputs(job_id, video_entry.get("gcs_artifacts"))
            _job_step_update(job_id, "upload_artifacts", status="completed", percent=100, message="Completed")
        except Exception as exc:
            app.logger.warning("Failed to upload artifacts to GCS for %s: %s", job_id, exc)
            _job_step_update(job_id, "upload_artifacts", status="failed", percent=100, message=str(exc)[:240])
        if critical_failures:
            err = "; ".join(critical_failures)[:400]
            _job_update(job_id, status="failed", progress=100, message="Failed", error=err)
        else:
            _job_update(job_id, status="completed", progress=100, message="Completed", result={"id": job_id, "title": video_title, "gcs_video_uri": gcs_uri})
    except StopJob as exc:
        _job_update(job_id, status="stopped", message="Stopped by user", error=str(exc))
        _job_step_update(job_id, "save_as_json", status="failed", message="Stopped by user")
    except Exception as exc:
        app.logger.error("GCS job %s failed: %s", job_id, exc)
        _job_update(job_id, status="failed", message="Failed", error=str(exc))
        _job_step_update(job_id, "save_as_json", status="failed", message=str(exc))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _gcp_auth_status() -> dict[str, Any]:
    adc_path = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip() or None
    return {
        "has_storage": gcs_storage is not None,
        "has_speech": gcp_speech is not None,
        "has_translate": gcp_translate is not None,
        "has_language": gcp_language is not None,
        "has_video_intelligence": gcp_video_intelligence is not None,
        "adc": {"path": adc_path},
    }


_CPU_SAMPLE_LOCK = threading.Lock()
_CPU_LAST_SAMPLE: tuple[float, float] | None = None


def _read_cpu_sample() -> tuple[float, float] | None:
    try:
        with open("/proc/stat", "r", encoding="utf-8") as handle:
            line = handle.readline()
        if not line.startswith("cpu "):
            return None
        parts = [float(x) for x in line.strip().split()[1:]]
        if len(parts) < 4:
            return None
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0.0)
        total = sum(parts)
        return total, idle
    except Exception:
        return None


def _cpu_percent() -> float | None:
    global _CPU_LAST_SAMPLE
    sample_1 = _read_cpu_sample()
    if not sample_1:
        return None
    time.sleep(0.12)
    sample_2 = _read_cpu_sample()
    if not sample_2:
        return None
    total_1, idle_1 = sample_1
    total_2, idle_2 = sample_2
    delta_total = total_2 - total_1
    delta_idle = idle_2 - idle_1
    if delta_total <= 0:
        return None
    used = 100.0 * (1.0 - (delta_idle / delta_total))
    return max(0.0, min(100.0, used))


def _gpu_percent() -> float | None:
    try:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            pynvml.nvmlShutdown()
            value = float(util.gpu)
            return max(0.0, min(100.0, value))
        except Exception:
            pass
        if not (shutil.which("nvidia-smi") or Path("/usr/bin/nvidia-smi").exists()):
            return None
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode != 0:
            return None
        raw = (proc.stdout or "").strip().splitlines()
        if not raw:
            return None
        value = float(raw[0].strip())
        return max(0.0, min(100.0, value))
    except Exception:
        return None


def _memory_stats() -> tuple[float, float, float] | None:
    try:
        mem_total_kb = None
        mem_available_kb = None
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    mem_total_kb = float(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_available_kb = float(line.split()[1])
                if mem_total_kb is not None and mem_available_kb is not None:
                    break
        if mem_total_kb is None or mem_available_kb is None:
            return None
        total_gb = mem_total_kb / 1024.0 / 1024.0
        available_gb = mem_available_kb / 1024.0 / 1024.0
        used_gb = max(0.0, total_gb - available_gb)
        percent = 0.0 if total_gb <= 0 else (used_gb / total_gb) * 100.0
        return (used_gb, total_gb, max(0.0, min(100.0, percent)))
    except Exception:
        return None


@app.route("/health", methods=["GET"])
def health() -> Any:
    resolved_bucket = None
    try:
        resolved_bucket = _gcs_bucket_name()
    except Exception:
        resolved_bucket = None

    return jsonify(
        {
            "status": "ok",
            "service": os.getenv("ENVID_SERVICE_NAME", "backend"),
            "gcp_auth": _gcp_auth_status(),
            "gcp_location": _gcp_location(),
            "gcs_bucket": resolved_bucket,
            "gcp_gcs_bucket": resolved_bucket,
        }
    )


@app.route("/system/stats", methods=["GET"])
def system_stats() -> Any:
    cpu_value = None
    with _CPU_SAMPLE_LOCK:
        cpu_value = _cpu_percent()
    gpu_value = _gpu_percent()
    mem_stats = _memory_stats()
    return jsonify(
        {
            "status": "ok",
            "cpu_percent": cpu_value,
            "gpu_percent": gpu_value,
            "memory_used_gb": mem_stats[0] if mem_stats else None,
            "memory_total_gb": mem_stats[1] if mem_stats else None,
            "memory_percent": mem_stats[2] if mem_stats else None,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


@app.route("/gcs/rawvideo/list", methods=["GET"])
def list_gcs_rawvideo_objects() -> Any:
    try:
        bucket = (request.args.get("bucket") or "").strip() or _gcs_bucket_name()
        raw_prefix = request.args.get("prefix")
        if raw_prefix is None:
            prefix = _gcs_rawvideo_prefix()
        else:
            prefix = (raw_prefix or "").strip()
            if prefix.lower() in {"*", "all", "__all__"}:
                prefix = ""
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        max_results = _parse_int(request.args.get("max_results") or request.args.get("max_keys") or "200", default=200, min_value=1, max_value=2000)
        page_token = (request.args.get("page_token") or "").strip() or None
        allowed = _allowed_gcs_buckets()
        if allowed is not None and bucket not in allowed:
            return jsonify({"error": f"Bucket not allowed: {bucket}. Allowed: {', '.join(sorted(allowed))}"}), 400

        client = _gcs_client()
        blobs_iter = client.list_blobs(bucket, prefix=prefix, max_results=max_results, page_token=page_token)

        objects: List[Dict[str, Any]] = []
        for blob in blobs_iter:
            name = getattr(blob, "name", "")
            if not name or name.endswith("/"):
                continue
            objects.append({"bucket": bucket, "name": name, "size": int(getattr(blob, "size", 0) or 0), "updated": getattr(blob, "updated", None).isoformat() if getattr(blob, "updated", None) else None, "uri": f"gs://{bucket}/{name}"})

        next_token = getattr(blobs_iter, "next_page_token", None)
        return jsonify({"bucket": bucket, "prefix": prefix.rstrip("/"), "objects": objects, "next_page_token": next_token})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/gcs/buckets/list", methods=["GET"])
def list_gcs_buckets() -> Any:
    try:
        project_id = _gcp_project_id()
        if not project_id:
            return jsonify({"error": "Missing GCP_PROJECT_ID"}), 400

        client = _gcs_client()
        allowed = _allowed_gcs_buckets()
        buckets = []
        for bucket in client.list_buckets(project=project_id):
            name = getattr(bucket, "name", None)
            if not name:
                continue
            if allowed is not None and name not in allowed:
                continue
            buckets.append(
                {
                    "name": name,
                    "location": getattr(bucket, "location", None),
                    "storage_class": getattr(bucket, "storage_class", None),
                }
            )
        default_bucket = None
        try:
            default_bucket = _gcs_bucket_name()
        except Exception:
            default_bucket = None
        return jsonify({"buckets": buckets, "default_bucket": default_bucket, "allowed": sorted(allowed) if allowed is not None else None})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/gcs/objects/list", methods=["GET"])
def list_gcs_objects() -> Any:
    try:
        bucket = (request.args.get("bucket") or "").strip() or _gcs_bucket_name()
        prefix = (request.args.get("prefix") or "").strip()
        if prefix.lower() in {"*", "all", "__all__"}:
            prefix = ""
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        allowed = _allowed_gcs_buckets()
        if allowed is not None and bucket not in allowed:
            return jsonify({"error": f"Bucket not allowed: {bucket}. Allowed: {', '.join(sorted(allowed))}"}), 400

        max_results = _parse_int(request.args.get("max_results") or request.args.get("max_keys") or "200", default=200, min_value=1, max_value=2000)
        page_token = (request.args.get("page_token") or "").strip() or None

        client = _gcs_client()
        blobs_iter = client.list_blobs(bucket, prefix=prefix, delimiter="/", max_results=max_results, page_token=page_token)

        objects: List[Dict[str, Any]] = []
        for blob in blobs_iter:
            name = getattr(blob, "name", "")
            if not name or name.endswith("/"):
                continue
            objects.append(
                {
                    "bucket": bucket,
                    "name": name,
                    "size": int(getattr(blob, "size", 0) or 0),
                    "updated": getattr(blob, "updated", None).isoformat() if getattr(blob, "updated", None) else None,
                    "uri": f"gs://{bucket}/{name}",
                }
            )

        prefixes = sorted(list(getattr(blobs_iter, "prefixes", []) or []))
        next_token = getattr(blobs_iter, "next_page_token", None)
        return jsonify({"bucket": bucket, "prefix": prefix, "prefixes": prefixes, "objects": objects, "next_page_token": next_token})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/upload-video", methods=["POST"])
def upload_video() -> Any:
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    video_file = request.files["video"]
    if not video_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    video_title = request.form.get("title", video_file.filename)
    video_description = request.form.get("description", "")

    job_id = _next_job_id()
    original_filename = video_file.filename
    temp_dir = _gcs_job_work_dir(job_id)
    temp_dir.mkdir(parents=True, exist_ok=True)
    frame_interval_seconds = 0
    max_frames_to_analyze = 1000
    face_recognition_mode = (request.form.get("face_recognition_mode") or "").strip() or None
    raw_task_selection = request.form.get("task_selection") or request.form.get("taskSelection") or request.form.get("selection")
    task_selection = _parse_task_selection(raw_task_selection)

    try:
        ext = Path(original_filename).suffix or ".mp4"
        upload_dir = _gcs_job_raw_dir(job_id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        video_path = upload_dir / f"{job_id}{ext}"
        video_file.save(str(video_path))
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify({"error": f"Failed to save upload: {exc}"}), 500

    _job_init(job_id, title=video_title)
    _job_update(job_id, status="processing", progress=1, message="Upload received", temp_dir=str(temp_dir))
    _db_file_insert(job_id, kind="local_upload", path=str(video_path))

    def _worker() -> None:
        try:
            working_bucket = _gcs_working_bucket_name()
            working_prefix = _gcs_working_prefix()
            ext = Path(original_filename).suffix or ".mp4"
            working_safe_name = f"{job_id}{ext}"
            working_obj = f"{working_prefix}{job_id}/{working_safe_name}"
            working_uri = f"gs://{working_bucket}/{working_obj}"
            _job_update(job_id, message="Copying to working bucket")
            client = _gcs_client()
            client.bucket(working_bucket).blob(working_obj).upload_from_filename(str(video_path))
            _job_update(job_id, gcs_working_uri=working_uri)

            ingest_url = _ingest_service_url()
            if ingest_url:
                _job_step_update(job_id, "upload_to_cloud_storage", status="running", percent=0, message="Uploading")
                _job_update(job_id, progress=2, message="Uploading to cloud storage")
                ingest_payload = _upload_via_ingest_service(
                    ingest_url=ingest_url,
                    video_path=video_path,
                    filename=original_filename,
                    job_id=job_id,
                    title=video_title,
                    description=video_description,
                )
                bucket = str(ingest_payload.get("gcs_bucket") or "").strip()
                obj = str(ingest_payload.get("gcs_object") or "").strip()
                gcs_uri = str(ingest_payload.get("gcs_uri") or "").strip()
                if not bucket or not obj:
                    raise RuntimeError("Ingest service did not return gcs_bucket/gcs_object")
                if not gcs_uri:
                    gcs_uri = f"gs://{bucket}/{obj}"
                _job_step_update(job_id, "upload_to_cloud_storage", status="completed", percent=100, message="Uploaded")
                _job_update(job_id, gcs_video_uri=gcs_uri)
            else:
                bucket = _gcs_bucket_name()
                prefix = _gcs_rawvideo_prefix()
                safe_name = Path(original_filename).name or "video.mp4"
                obj = f"{prefix}{job_id}/{safe_name}"
                gcs_uri = f"gs://{bucket}/{obj}"

                _job_step_update(job_id, "upload_to_cloud_storage", status="running", percent=0, message="Uploading")
                _job_update(job_id, progress=2, message="Uploading to cloud storage")

                _upload_gcs_with_progress(bucket=bucket, obj=obj, video_path=video_path, job_id=job_id)
                _job_step_update(job_id, "upload_to_cloud_storage", status="completed", percent=100, message="Uploaded")
                _job_update(job_id, gcs_video_uri=gcs_uri)
                _db_file_upsert(job_id, kind="source_gcs", path=None, gcs_uri=gcs_uri)

            _process_gcs_video_job_cloud_only(
                job_id=job_id,
                gcs_bucket=bucket,
                gcs_object=obj,
                video_title=video_title,
                video_description=video_description,
                frame_interval_seconds=frame_interval_seconds,
                max_frames_to_analyze=max_frames_to_analyze,
                face_recognition_mode=face_recognition_mode,
                task_selection=task_selection,
            )
        except Exception as exc:
            app.logger.error("Upload job %s failed: %s", job_id, exc)
            _job_step_update(job_id, "upload_to_cloud_storage", status="failed", message=str(exc))
            _job_update(job_id, status="failed", message="Failed", error=str(exc))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    threading.Thread(target=_worker, daemon=True).start()
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    return jsonify({"job_id": job_id, "job": job}), 202


@app.route("/jobs", methods=["GET"])
def list_jobs() -> Any:
    status_raw = str(request.args.get("status") or "").strip().lower()
    limit = _parse_int(request.args.get("limit"), default=50, min_value=1, max_value=200)
    statuses = [s.strip() for s in status_raw.split(",") if s.strip()] if status_raw else []

    if _db_enabled():
        jobs = _db_list_jobs(statuses=statuses or None, limit=limit)
        job_ids = [str(j.get("id") or "").strip() for j in jobs if j.get("id")]
        steps_map = _db_get_job_steps_for_jobs(job_ids)
        for job in jobs:
            job_id = str(job.get("id") or "").strip()
            job["steps"] = steps_map.get(job_id, [])
            status = str(job.get("status") or "").strip().lower()
            if not status:
                steps = job.get("steps") if isinstance(job.get("steps"), list) else []
                step_statuses = {str(s.get("status") or "").lower() for s in steps if isinstance(s, dict)}
                if "running" in step_statuses or "processing" in step_statuses:
                    status = "processing"
                elif "failed" in step_statuses:
                    status = "failed"
                elif step_statuses and step_statuses.issubset({"completed", "skipped"}):
                    status = "completed"
                else:
                    progress = job.get("progress")
                    if isinstance(progress, (int, float)) and progress >= 100:
                        status = "completed"
                    elif isinstance(progress, (int, float)) and progress > 0:
                        status = "processing"
                    else:
                        status = "queued"
                job["status"] = status
        return jsonify({"jobs": jobs}), 200

    with JOBS_LOCK:
        jobs = list(JOBS.values())
    if statuses:
        status_set = {s.lower() for s in statuses}
        jobs = [j for j in jobs if str(j.get("status") or "").lower() in status_set]
    jobs.sort(key=lambda j: str(j.get("updated_at") or ""), reverse=True)
    return jsonify({"jobs": jobs[:limit]}), 200


@app.route("/process-gcs-video-cloud", methods=["POST"])
def process_gcs_video_cloud() -> Any:
    payload = request.get_json(silent=True) or {}
    gcs_bucket = (payload.get("gcs_bucket") or payload.get("bucket") or "").strip()
    raw = (payload.get("gcs_object") or payload.get("gcs_uri") or "").strip()
    if raw and gcs_bucket and not raw.lower().startswith("gs://"):
        raw = f"gs://{gcs_bucket}/{raw.lstrip('/')}"
    if not raw:
        return jsonify({"error": "Missing gcs_object (or gcs_uri)"}), 400

    try:
        source_bucket, source_obj = _parse_any_gcs_video_source(raw)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    video_title = (payload.get("title") or Path(source_obj).name or "GCS Video").strip()
    video_description = (payload.get("description") or "").strip()
    raw_task_selection = payload.get("task_selection") or payload.get("taskSelection") or payload.get("selection")
    task_selection = _parse_task_selection(raw_task_selection)
    requested_job_id = (payload.get("job_id") or payload.get("id") or "").strip()
    job_id = requested_job_id if _looks_like_job_id(requested_job_id) else _next_job_id()

    _job_init(job_id, title=video_title)
    _job_update(
        job_id,
        status="processing",
        progress=1,
        message="GCS video queued",
        gcs_video_uri=f"gs://{source_bucket}/{source_obj}",
    )

    threading.Thread(
        target=_process_gcs_video_job_cloud_only,
        kwargs={
            "job_id": job_id,
            "gcs_bucket": source_bucket,
            "gcs_object": source_obj,
            "video_title": video_title,
            "video_description": video_description,
            "frame_interval_seconds": 0,
            "max_frames_to_analyze": 1000,
            "face_recognition_mode": None,
            "task_selection": task_selection,
        },
        daemon=True,
    ).start()
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    return jsonify({"job_id": job_id, "job": job}), 202


@app.route("/jobs/<job_id>", methods=["GET", "DELETE"])
def get_job(job_id: str) -> Any:
    if request.method == "DELETE":
        with JOBS_LOCK:
            removed = JOBS.pop(job_id, None)
        db_row = _db_get_job(job_id)
        _cleanup_job_artifacts(job_id, db_row if isinstance(db_row, dict) else None)
        _db_delete_job(job_id)
        with VIDEO_INDEX_LOCK:
            idx = next((i for i, v in enumerate(VIDEO_INDEX) if str(v.get("id")) == str(job_id)), None)
            if idx is not None:
                VIDEO_INDEX.pop(idx)
                _save_video_index()
        if not removed and not _db_get_job(job_id):
            return jsonify({"error": "Job not found"}), 404
        return jsonify({"ok": True, "deleted": job_id}), 200

    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        db_row = _db_get_job(job_id)
        if not db_row:
            return jsonify({"error": "Job not found"}), 404
        steps = _db_get_job_steps(job_id)
        if steps:
            db_row = {**db_row, "steps": steps}
        if _db_enabled():
            outputs = _db_get_job_outputs(job_id)
            if outputs:
                db_row = {**db_row, "outputs": outputs}
        return jsonify(db_row), 200
    if _db_enabled():
        outputs = _db_get_job_outputs(job_id)
        if outputs:
            job = {**job, "outputs": outputs}
    return jsonify(job), 200


@app.route("/jobs/<job_id>/outputs/<kind>", methods=["GET"])
def get_job_output(job_id: str, kind: str) -> Any:
    job_id = (job_id or "").strip()
    kind = (kind or "").strip()
    if not job_id or not kind:
        return jsonify({"error": "Missing job_id or kind"}), 400
    row = _db_get_job_output(job_id, kind)
    if not row:
        return jsonify({"error": "Output not found"}), 404
    return jsonify(row), 200


@app.route("/jobs/<job_id>/outputs/<kind>/download", methods=["GET"])
def download_job_output(job_id: str, kind: str) -> Any:
    job_id = (job_id or "").strip()
    kind = (kind or "").strip()
    if not job_id or not kind:
        return jsonify({"error": "Missing job_id or kind"}), 400
    row = _db_get_job_output(job_id, kind)
    if not row:
        return jsonify({"error": "Output not found"}), 404

    path = row.get("path") if isinstance(row, dict) else None
    if isinstance(path, str) and path:
        p = Path(path)
        if not p.exists():
            return jsonify({"error": "Output file missing"}), 404
        suffix = p.suffix or ".json"
        mime = "application/json" if suffix.lower() == ".json" else "application/octet-stream"
        download_name = f"{job_id}__{kind}{suffix}"
        return send_file(p, mimetype=mime, as_attachment=True, download_name=download_name)

    payload = row.get("payload") if isinstance(row, dict) else None
    if isinstance(payload, dict):
        buf = io.BytesIO()
        buf.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        buf.seek(0)
        return send_file(
            buf,
            mimetype="application/json",
            as_attachment=True,
            download_name=f"{job_id}__{kind}.json",
        )

    gcs_uri = row.get("gcs_uri") if isinstance(row, dict) else None
    if isinstance(gcs_uri, str) and gcs_uri.strip().lower().startswith("gs://"):
        try:
            bucket, obj = _parse_gcs_uri(gcs_uri)
            suffix = Path(obj).suffix or ""
            response_type = "application/octet-stream"
            if suffix.lower() in {".json", ".gz"}:
                response_type = "application/json"
            elif suffix.lower() == ".zip":
                response_type = "application/zip"
            elif suffix.lower() == ".vtt":
                response_type = "text/vtt"
            elif suffix.lower() == ".srt":
                response_type = "application/x-subrip"
            url = _gcs_presign_get_url(
                bucket=bucket,
                obj=obj,
                response_type=response_type,
                response_disposition=f'attachment; filename="{job_id}__{kind}{suffix}"',
            )
            return redirect(url, code=302)
        except Exception:
            pass

    return jsonify({"error": "Output path not found"}), 404


@app.route("/jobs/<job_id>/stop", methods=["POST"])
def stop_job(job_id: str) -> Any:
    job_id = (job_id or "").strip()
    if not job_id:
        return jsonify({"error": "Missing job_id"}), 400
    if not _db_get_job(job_id) and job_id not in JOBS:
        return jsonify({"error": "Job not found"}), 404
    _job_update(job_id, stop_requested=True, status="stopping", message="Stop requested by user")
    return jsonify({"ok": True, "job_id": job_id, "status": "stopping"}), 202


@app.route("/jobs/<job_id>/start", methods=["POST"])
def start_job(job_id: str) -> Any:
    job_id = (job_id or "").strip()
    if not job_id:
        return jsonify({"error": "Missing job_id"}), 400
    if not _db_get_job(job_id):
        return jsonify({"error": "Job not found"}), 404
    _job_update(job_id, stop_requested=False)
    return reprocess_job(job_id)


@app.route("/jobs/<job_id>/restart", methods=["POST"])
def restart_job(job_id: str) -> Any:
    return start_job(job_id)


@app.route("/videos", methods=["GET"])
def list_videos() -> Any:
    with VIDEO_INDEX_LOCK:
        videos = list(VIDEO_INDEX)
    videos.sort(key=lambda v: (v.get("uploaded_at") or "", v.get("id") or ""), reverse=True)
    return jsonify({"videos": videos, "count": len(videos)}), 200


def _get_video_entry_by_id(video_id: str) -> dict[str, Any] | None:
    video_id = str(video_id or "").strip()
    if not video_id:
        return None
    if VIDEO_INDEX_LOCK.acquire(timeout=0.2):
        try:
            found = next((x for x in VIDEO_INDEX if str(x.get("id")) == video_id), None)
            if found is not None:
                return found
        finally:
            VIDEO_INDEX_LOCK.release()
    try:
        snapshot = _safe_json_load(VIDEO_INDEX_FILE, [])
        if isinstance(snapshot, list):
            return next((x for x in snapshot if isinstance(x, dict) and str(x.get("id")) == video_id), None)
    except Exception:
        return None
    return None


def _get_video_entry_with_cached_metadata(video_id: str, entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        entry = None
    if entry and isinstance(entry.get("metadata_categories"), dict) and isinstance(entry.get("metadata_combined"), dict):
        return entry
    try:
        snapshot = _safe_json_load(VIDEO_INDEX_FILE, [])
        if isinstance(snapshot, list):
            candidate = next((x for x in snapshot if isinstance(x, dict) and str(x.get("id")) == video_id), None)
            if isinstance(candidate, dict):
                if isinstance(candidate.get("metadata_categories"), dict) and isinstance(candidate.get("metadata_combined"), dict):
                    return candidate
                if entry is None:
                    return candidate
    except Exception:
        return entry
    return entry


@app.route("/video/<video_id>", methods=["GET"])
def get_video(video_id: str) -> Any:
    v = _get_video_entry_by_id(video_id)
    if not v:
        return jsonify({"error": "Video not found"}), 404
    return jsonify(v), 200


@app.route("/video/<video_id>/metadata-json", methods=["GET"])
def get_video_metadata_json(video_id: str) -> Any:
    v = _get_video_entry_by_id(video_id)
    v = _get_video_entry_with_cached_metadata(video_id, v)
    if not v:
        return jsonify({"error": "Video not found"}), 404

    category = (request.args.get("category") or "").strip().lower()
    lang = (request.args.get("lang") or "").strip().lower() or None
    download = (request.args.get("download") or "").strip().lower() in {"1", "true", "yes", "on"}

    categories = v.get("metadata_categories") if isinstance(v.get("metadata_categories"), dict) else None
    combined_base = v.get("metadata_combined") if isinstance(v.get("metadata_combined"), dict) else None
    categorized = None
    if categories is None or combined_base is None:
        categorized = _build_categorized_metadata_json(v)
        if categories is None:
            categories = categorized.get("categories") if isinstance(categorized.get("categories"), dict) else {}
        if combined_base is None:
            combined_base = categorized.get("combined") if isinstance(categorized.get("combined"), dict) else {}
    if not isinstance(categories, dict):
        categories = {}
    if not isinstance(combined_base, dict):
        combined_base = {}
    combined = _apply_translated_combined(v, combined_base, lang)

    payload: dict[str, Any]
    if category == "combined":
        payload = {"id": video_id, "combined": combined}
    elif category == "categories":
        payload = {"id": video_id, "categories": categories}
    else:
        payload = {
            "id": video_id,
            "categories": categories,
            "combined": combined,
            "task_selection": v.get("task_selection") if isinstance(v.get("task_selection"), dict) else {},
            "task_selection_requested_models": v.get("task_selection_requested_models")
            if isinstance(v.get("task_selection_requested_models"), dict)
            else {},
            "task_selection_effective": v.get("task_selection_effective") if isinstance(v.get("task_selection_effective"), dict) else {},
        }

    if lang:
        payload["language"] = lang

    response = jsonify(payload)
    if download:
        suffix = f".{lang}" if lang else ""
        response.headers["Content-Disposition"] = f'attachment; filename="{video_id}{suffix}.metadata.json"'
    return response, 200


@app.route("/jobs/<job_id>/metadata-json", methods=["GET"])
def get_job_metadata_json(job_id: str) -> Any:
    return get_video_metadata_json(job_id)


@app.route("/video/<video_id>/metadata-json.zip", methods=["GET"])
def get_video_metadata_zip(video_id: str) -> Any:
    v = _get_video_entry_by_id(video_id)
    v = _get_video_entry_with_cached_metadata(video_id, v)
    if not v:
        return jsonify({"error": "Video not found"}), 404
    artifacts = v.get("gcs_artifacts") if isinstance(v.get("gcs_artifacts"), dict) else None
    zip_info = artifacts.get("zip") if artifacts else None
    if isinstance(zip_info, dict) and (zip_info.get("object") and artifacts and artifacts.get("bucket")):
        try:
            url = _gcs_presign_get_url(
                bucket=str(artifacts.get("bucket") or "").strip(),
                obj=str(zip_info.get("object") or "").strip(),
                response_type="application/zip",
                response_disposition=f'attachment; filename="{video_id}.metadata-json.zip"',
            )
            return redirect(url, code=302)
        except Exception:
            pass

    categories = v.get("metadata_categories") if isinstance(v.get("metadata_categories"), dict) else None
    combined = v.get("metadata_combined") if isinstance(v.get("metadata_combined"), dict) else None
    if categories is None or combined is None:
        categorized = _build_categorized_metadata_json(v)
        if categories is None:
            categories = categorized.get("categories") if isinstance(categorized.get("categories"), dict) else {}
        if combined is None:
            combined = categorized.get("combined") if isinstance(categorized.get("combined"), dict) else {}
    if not isinstance(categories, dict):
        categories = {}
    if not isinstance(combined, dict):
        combined = {}
    payload = {"id": video_id, "categories": categories, "combined": combined}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{video_id}.metadata.json", json.dumps(payload, indent=2, ensure_ascii=False))
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"{video_id}.metadata-json.zip")


@app.route("/jobs/<job_id>/metadata-json.zip", methods=["GET"])
def get_job_metadata_zip(job_id: str) -> Any:
    return get_video_metadata_zip(job_id)


def _serve_text_file(path: Path, content_type: str) -> Response:
    data = path.read_text(encoding="utf-8", errors="replace")
    return Response(data, mimetype=content_type)


@app.route("/video/<video_id>/subtitles.srt", methods=["GET"])
def subtitles_srt(video_id: str) -> Any:
    with VIDEO_INDEX_LOCK:
        v = next((x for x in VIDEO_INDEX if str(x.get("id")) == str(video_id)), None)
    artifacts = (v.get("gcs_artifacts") if isinstance(v, dict) else None) if v else None
    subs = artifacts.get("subtitles") if isinstance(artifacts, dict) else None
    info = subs.get("orig.srt") if isinstance(subs, dict) else None
    if isinstance(info, dict) and artifacts and artifacts.get("bucket") and info.get("object"):
        try:
            url = _gcs_presign_get_url(bucket=str(artifacts.get("bucket")), obj=str(info.get("object")), response_type="application/x-subrip")
            return redirect(url, code=302)
        except Exception:
            pass

    path = _subtitles_local_path(video_id, lang="orig", fmt="srt")
    if not path.exists():
        return jsonify({"error": "Subtitles not found"}), 404
    return _serve_text_file(path, "application/x-subrip")


@app.route("/video/<video_id>/subtitles.vtt", methods=["GET"])
def subtitles_vtt(video_id: str) -> Any:
    with VIDEO_INDEX_LOCK:
        v = next((x for x in VIDEO_INDEX if str(x.get("id")) == str(video_id)), None)
    artifacts = (v.get("gcs_artifacts") if isinstance(v, dict) else None) if v else None
    subs = artifacts.get("subtitles") if isinstance(artifacts, dict) else None
    info = subs.get("orig.vtt") if isinstance(subs, dict) else None
    if isinstance(info, dict) and artifacts and artifacts.get("bucket") and info.get("object"):
        try:
            url = _gcs_presign_get_url(bucket=str(artifacts.get("bucket")), obj=str(info.get("object")), response_type="text/vtt")
            return redirect(url, code=302)
        except Exception:
            pass

    path = _subtitles_local_path(video_id, lang="orig", fmt="vtt")
    if not path.exists():
        return jsonify({"error": "Subtitles not found"}), 404
    return _serve_text_file(path, "text/vtt")


@app.route("/video/<video_id>/subtitles.en.srt", methods=["GET"])
def subtitles_en_srt(video_id: str) -> Any:
    with VIDEO_INDEX_LOCK:
        v = next((x for x in VIDEO_INDEX if str(x.get("id")) == str(video_id)), None)
    artifacts = (v.get("gcs_artifacts") if isinstance(v, dict) else None) if v else None
    subs = artifacts.get("subtitles") if isinstance(artifacts, dict) else None
    info = subs.get("en.srt") if isinstance(subs, dict) else None
    if isinstance(info, dict) and artifacts and artifacts.get("bucket") and info.get("object"):
        try:
            url = _gcs_presign_get_url(bucket=str(artifacts.get("bucket")), obj=str(info.get("object")), response_type="application/x-subrip")
            return redirect(url, code=302)
        except Exception:
            pass

    path = _subtitles_local_path(video_id, lang="en", fmt="srt")
    if not path.exists():
        return jsonify({"error": "English subtitles not found"}), 404
    return _serve_text_file(path, "application/x-subrip")


@app.route("/video/<video_id>/subtitles.en.vtt", methods=["GET"])
def subtitles_en_vtt(video_id: str) -> Any:
    with VIDEO_INDEX_LOCK:
        v = next((x for x in VIDEO_INDEX if str(x.get("id")) == str(video_id)), None)
    artifacts = (v.get("gcs_artifacts") if isinstance(v, dict) else None) if v else None
    subs = artifacts.get("subtitles") if isinstance(artifacts, dict) else None
    info = subs.get("en.vtt") if isinstance(subs, dict) else None
    if isinstance(info, dict) and artifacts and artifacts.get("bucket") and info.get("object"):
        try:
            url = _gcs_presign_get_url(bucket=str(artifacts.get("bucket")), obj=str(info.get("object")), response_type="text/vtt")
            return redirect(url, code=302)
        except Exception:
            pass

    path = _subtitles_local_path(video_id, lang="en", fmt="vtt")
    if not path.exists():
        return jsonify({"error": "English subtitles not found"}), 404
    return _serve_text_file(path, "text/vtt")


@app.route("/video/<video_id>/subtitles.ar.srt", methods=["GET"])
def subtitles_ar_srt(video_id: str) -> Any:
    with VIDEO_INDEX_LOCK:
        v = next((x for x in VIDEO_INDEX if str(x.get("id")) == str(video_id)), None)
    artifacts = (v.get("gcs_artifacts") if isinstance(v, dict) else None) if v else None
    subs = artifacts.get("subtitles") if isinstance(artifacts, dict) else None
    info = subs.get("ar.srt") if isinstance(subs, dict) else None
    if isinstance(info, dict) and artifacts and artifacts.get("bucket") and info.get("object"):
        try:
            url = _gcs_presign_get_url(bucket=str(artifacts.get("bucket")), obj=str(info.get("object")), response_type="application/x-subrip")
            return redirect(url, code=302)
        except Exception:
            pass

    path = _subtitles_local_path(video_id, lang="ar", fmt="srt")
    if not path.exists():
        return jsonify({"error": "Arabic subtitles not found"}), 404
    return _serve_text_file(path, "application/x-subrip")


@app.route("/video/<video_id>/subtitles.ar.vtt", methods=["GET"])
def subtitles_ar_vtt(video_id: str) -> Any:
    with VIDEO_INDEX_LOCK:
        v = next((x for x in VIDEO_INDEX if str(x.get("id")) == str(video_id)), None)
    artifacts = (v.get("gcs_artifacts") if isinstance(v, dict) else None) if v else None
    subs = artifacts.get("subtitles") if isinstance(artifacts, dict) else None
    info = subs.get("ar.vtt") if isinstance(subs, dict) else None
    if isinstance(info, dict) and artifacts and artifacts.get("bucket") and info.get("object"):
        try:
            url = _gcs_presign_get_url(bucket=str(artifacts.get("bucket")), obj=str(info.get("object")), response_type="text/vtt")
            return redirect(url, code=302)
        except Exception:
            pass

    path = _subtitles_local_path(video_id, lang="ar", fmt="vtt")
    if not path.exists():
        return jsonify({"error": "Arabic subtitles not found"}), 404
    return _serve_text_file(path, "text/vtt")


@app.route("/video/<video_id>/subtitles.id.srt", methods=["GET"])
def subtitles_id_srt(video_id: str) -> Any:
    with VIDEO_INDEX_LOCK:
        v = next((x for x in VIDEO_INDEX if str(x.get("id")) == str(video_id)), None)
    artifacts = (v.get("gcs_artifacts") if isinstance(v, dict) else None) if v else None
    subs = artifacts.get("subtitles") if isinstance(artifacts, dict) else None
    info = subs.get("id.srt") if isinstance(subs, dict) else None
    if isinstance(info, dict) and artifacts and artifacts.get("bucket") and info.get("object"):
        try:
            url = _gcs_presign_get_url(bucket=str(artifacts.get("bucket")), obj=str(info.get("object")), response_type="application/x-subrip")
            return redirect(url, code=302)
        except Exception:
            pass

    path = _subtitles_local_path(video_id, lang="id", fmt="srt")
    if not path.exists():
        return jsonify({"error": "Indonesian subtitles not found"}), 404
    return _serve_text_file(path, "application/x-subrip")


@app.route("/video/<video_id>/subtitles.id.vtt", methods=["GET"])
def subtitles_id_vtt(video_id: str) -> Any:
    with VIDEO_INDEX_LOCK:
        v = next((x for x in VIDEO_INDEX if str(x.get("id")) == str(video_id)), None)
    artifacts = (v.get("gcs_artifacts") if isinstance(v, dict) else None) if v else None
    subs = artifacts.get("subtitles") if isinstance(artifacts, dict) else None
    info = subs.get("id.vtt") if isinstance(subs, dict) else None
    if isinstance(info, dict) and artifacts and artifacts.get("bucket") and info.get("object"):
        try:
            url = _gcs_presign_get_url(bucket=str(artifacts.get("bucket")), obj=str(info.get("object")), response_type="text/vtt")
            return redirect(url, code=302)
        except Exception:
            pass

    path = _subtitles_local_path(video_id, lang="id", fmt="vtt")
    if not path.exists():
        return jsonify({"error": "Indonesian subtitles not found"}), 404
    return _serve_text_file(path, "text/vtt")


@app.route("/video/<video_id>/subtitles/<lang>.<fmt>", methods=["GET"])
def subtitles_any(video_id: str, lang: str, fmt: str) -> Any:
    lang_norm = (lang or "").strip().lower() or "orig"
    fmt_norm = (fmt or "").strip().lower().lstrip(".")
    if fmt_norm not in {"srt", "vtt"}:
        return jsonify({"error": "Unsupported subtitle format"}), 400
    with VIDEO_INDEX_LOCK:
        v = next((x for x in VIDEO_INDEX if str(x.get("id")) == str(video_id)), None)
    artifacts = (v.get("gcs_artifacts") if isinstance(v, dict) else None) if v else None
    subs = artifacts.get("subtitles") if isinstance(artifacts, dict) else None
    key = f"{lang_norm}.{fmt_norm}"
    info = subs.get(key) if isinstance(subs, dict) else None
    if isinstance(info, dict) and artifacts and artifacts.get("bucket") and info.get("object"):
        try:
            response_type = "application/x-subrip" if fmt_norm == "srt" else "text/vtt"
            url = _gcs_presign_get_url(
                bucket=str(artifacts.get("bucket")),
                obj=str(info.get("object")),
                response_type=response_type,
            )
            return redirect(url, code=302)
        except Exception:
            pass

    path = _subtitles_local_path(video_id, lang=lang_norm, fmt=fmt_norm)
    if not path.exists():
        return jsonify({"error": "Subtitles not found"}), 404
    content_type = "application/x-subrip" if fmt_norm == "srt" else "text/vtt"
    return _serve_text_file(path, content_type)


@app.get("/translate/languages")
def translate_languages() -> Any:
    base_url = "http://translate:5000"
    try:
        langs = _libretranslate_languages_raw(base_url)
        return jsonify({"ok": True, "languages": langs}), 200
    except Exception as exc:
        return jsonify({"ok": False, "languages": [], "error": str(exc)[:240]}), 200


@app.route("/video-file/<video_id>", methods=["GET"])
def video_file(video_id: str) -> Any:
    with VIDEO_INDEX_LOCK:
        v = next((x for x in VIDEO_INDEX if str(x.get("id")) == str(video_id)), None)
    if not v:
        return jsonify({"error": "Video not found"}), 404

    stored = (v.get("stored_filename") or "").strip()
    if stored:
        p = VIDEOS_DIR / stored
        if p.exists():
            mime = mimetypes.guess_type(p.name)[0] or "video/mp4"
            return send_file(p, mimetype=mime, as_attachment=False)

    gcs_uri = (v.get("gcs_video_uri") or "").strip()
    if not gcs_uri:
        return jsonify({"error": "No video source available"}), 404

    bucket, obj = _parse_allowed_gcs_video_source(gcs_uri)
    seconds = _parse_int(request.args.get("expires") or "3600", default=3600, min_value=60, max_value=86400)
    url = _gcs_client().bucket(bucket).blob(obj).generate_signed_url(version="v4", expiration=datetime.utcnow() + timedelta(seconds=seconds), method="GET")
    return redirect(url)


@app.route("/video/<video_id>", methods=["DELETE"])
def delete_video(video_id: str) -> Any:
    deleted_gcs_objects: List[str] = []
    warnings: List[str] = []
    kept_gcs_objects: List[str] = []
    with VIDEO_INDEX_LOCK:
        v = next((x for x in VIDEO_INDEX if str(x.get("id")) == str(video_id)), None)
    if not v:
        return jsonify({"error": "Video not found"}), 404
    try:
        gcs_uri = (v.get("gcs_video_uri") or "").strip()
        if gcs_uri:
            try:
                bucket, obj = _parse_allowed_gcs_video_source(gcs_uri)
                # IMPORTANT: never delete the source/original video object.
                # Deleting a history record should only remove temporary artifacts.
                kept_gcs_objects.append(f"gs://{bucket}/{obj}")
            except Exception as exc:
                warnings.append(f"Source video retained (could not parse gcs uri): {exc}")

        try:
            artifacts_bucket = _gcs_artifacts_bucket(_gcs_bucket_name())
            ap = _gcs_artifacts_prefix()
            prefix = f"{ap}/{video_id}/"
            client = _gcs_client()
            for blob in client.list_blobs(artifacts_bucket, prefix=prefix):
                try:
                    name = getattr(blob, "name", "")
                    blob.delete()
                    if name:
                        deleted_gcs_objects.append(f"gs://{artifacts_bucket}/{name}")
                except Exception as exc:
                    warnings.append(f"Failed to delete artifact: {exc}")
        except Exception as exc:
            warnings.append(f"Artifacts delete skipped: {exc}")

        try:
            stored = (v.get("stored_filename") or "").strip()
            if stored:
                p = VIDEOS_DIR / stored
                if p.exists():
                    p.unlink()
        except Exception as exc:
            warnings.append(f"Failed to delete local file: {exc}")

        for p in [
            _subtitles_local_path(video_id, lang="orig", fmt="srt"),
            _subtitles_local_path(video_id, lang="orig", fmt="vtt"),
            _subtitles_local_path(video_id, lang="en", fmt="srt"),
            _subtitles_local_path(video_id, lang="en", fmt="vtt"),
            _subtitles_local_path(video_id, lang="ar", fmt="srt"),
            _subtitles_local_path(video_id, lang="ar", fmt="vtt"),
            _subtitles_local_path(video_id, lang="id", fmt="srt"),
            _subtitles_local_path(video_id, lang="id", fmt="vtt"),
        ]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

        with VIDEO_INDEX_LOCK:
            VIDEO_INDEX[:] = [x for x in VIDEO_INDEX if str(x.get("id")) != str(video_id)]
        _save_video_index()
        _db_delete_job(video_id)
        return jsonify({"ok": True, "message": f"Deleted video {video_id}", "deleted_gcs_objects": deleted_gcs_objects, "kept_gcs_objects": kept_gcs_objects, "gcs_warnings": warnings}), 200
    except Exception as exc:
        return jsonify({"error": str(exc), "deleted_gcs_objects": deleted_gcs_objects, "kept_gcs_objects": kept_gcs_objects, "gcs_warnings": warnings}), 500


@app.route("/reprocess-video/<video_id>", methods=["POST"])
def reprocess_video(video_id: str) -> Any:
    payload = request.get_json(silent=True) or {}
    task_selection = payload.get("task_selection") or payload.get("taskSelection") or payload.get("selection")
    if not isinstance(task_selection, dict):
        task_selection = None

    with VIDEO_INDEX_LOCK:
        v = next((x for x in VIDEO_INDEX if str(x.get("id")) == str(video_id)), None)
    if not v:
        return jsonify({"error": "Video not found"}), 404
    gcs_uri = (v.get("gcs_video_uri") or "").strip()
    if not gcs_uri:
        return jsonify({"error": "Missing gcs_video_uri for video"}), 400
    try:
        bucket, obj = _parse_allowed_gcs_video_source(gcs_uri, enforce_prefix=False)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    # Reprocess should overwrite the existing entry (stable id) so clients can
    # poll /jobs/<video_id> and the index updates in-place.
    job_id = video_id if _looks_like_job_id(video_id) else _next_job_id()
    _job_init(job_id, title=v.get("title") or "Reprocess")
    _job_update(job_id, status="processing", progress=1, message="Reprocess queued", gcs_video_uri=gcs_uri)

    frame_interval_seconds = _parse_int(request.args.get("frame_interval_seconds"), default=0, min_value=0, max_value=30)
    max_frames_to_analyze = _parse_int(request.args.get("max_frames_to_analyze"), default=1000, min_value=1, max_value=10000)
    threading.Thread(
        target=_process_gcs_video_job_cloud_only,
        kwargs={
            "job_id": job_id,
            "gcs_bucket": bucket,
            "gcs_object": obj,
            "video_title": v.get("title") or Path(obj).name,
            "video_description": v.get("description") or "",
            "frame_interval_seconds": frame_interval_seconds,
            "max_frames_to_analyze": max_frames_to_analyze,
            "face_recognition_mode": (v.get("face_recognition_mode") or None),
            "task_selection": task_selection,
        },
        daemon=True,
    ).start()
    return jsonify({"ok": True, "job_id": job_id}), 202


@app.route("/video/<video_id>/reprocess", methods=["POST"])
def reprocess_video_alias(video_id: str) -> Any:
    return reprocess_video(video_id)


@app.route("/reprocess-videos", methods=["POST"])
def reprocess_videos() -> Any:
    payload = request.get_json(silent=True) or {}
    ids = payload.get("video_ids") or payload.get("ids") or []
    if not isinstance(ids, list) or not ids:
        return jsonify({"error": "Missing video_ids"}), 400
    accepted: List[Dict[str, Any]] = []
    for vid in ids:
        try:
            _ = reprocess_video(str(vid))
            accepted.append({"video_id": str(vid), "accepted": True})
        except Exception:
            accepted.append({"video_id": str(vid), "accepted": False})
    return jsonify({"ok": True, "results": accepted}), 200


@app.route("/jobs/<job_id>/reprocess-failed", methods=["POST"])
def reprocess_failed_steps(job_id: str) -> Any:
    job_id = (job_id or "").strip()
    if not job_id:
        return jsonify({"error": "Missing job_id"}), 400

    job_row = _db_get_job(job_id)
    if not job_row:
        return jsonify({"error": "Job not found"}), 404

    payload = request.get_json(silent=True) or {}
    gcs_uri = str(job_row.get("gcs_video_uri") or "").strip()
    if not gcs_uri:
        with VIDEO_INDEX_LOCK:
            v_lookup = next((x for x in VIDEO_INDEX if str(x.get("id")) == job_id), None)
        if isinstance(v_lookup, dict):
            gcs_uri = str(v_lookup.get("gcs_video_uri") or "").strip()
    if not gcs_uri:
        gcs_uri = str(payload.get("gcs_video_uri") or payload.get("gcs_uri") or payload.get("gcs_object") or "").strip()
    if not gcs_uri:
        return jsonify({"error": "Missing gcs_video_uri for job"}), 400

    steps = _db_get_job_steps(job_id)
    failed_steps = {
        str(s.get("step_id") or "").strip()
        for s in steps
        if str(s.get("status") or "").strip().lower() == "failed"
    }
    failed_steps.discard("")
    if not failed_steps:
        return jsonify({"error": "No failed steps to reprocess"}), 400

    try:
        bucket, obj = _parse_allowed_gcs_video_source(gcs_uri)
    except Exception as exc:
        alt_uri = str(job_row.get("gcs_working_uri") or "").strip()
        if alt_uri and "Object must be under" in str(exc):
            try:
                bucket, obj = _parse_allowed_gcs_video_source(alt_uri)
                gcs_uri = alt_uri
            except Exception:
                pass
        if "Object must be under" in str(exc):
            try:
                bucket, obj = _parse_allowed_gcs_video_source(gcs_uri, enforce_prefix=False)
            except Exception:
                return jsonify({"error": str(exc)}), 400
        else:
            return jsonify({"error": str(exc)}), 400

    task_selection = _task_selection_for_failed_steps(failed_steps)

    title = str(job_row.get("title") or "Reprocess").strip() or "Reprocess"
    with VIDEO_INDEX_LOCK:
        v = next((x for x in VIDEO_INDEX if str(x.get("id")) == job_id), None)
    video_title = (v.get("title") if isinstance(v, dict) else None) or title
    video_description = (v.get("description") if isinstance(v, dict) else None) or ""

    _job_init(job_id, title=video_title)
    _job_update(job_id, status="processing", progress=1, message="Reprocess failed steps queued", gcs_video_uri=gcs_uri)

    frame_interval_seconds = _parse_int(request.args.get("frame_interval_seconds"), default=0, min_value=0, max_value=30)
    max_frames_to_analyze = _parse_int(request.args.get("max_frames_to_analyze"), default=1000, min_value=1, max_value=10000)

    threading.Thread(
        target=_process_gcs_video_job_cloud_only,
        kwargs={
            "job_id": job_id,
            "gcs_bucket": bucket,
            "gcs_object": obj,
            "video_title": video_title,
            "video_description": video_description,
            "frame_interval_seconds": frame_interval_seconds,
            "max_frames_to_analyze": max_frames_to_analyze,
            "face_recognition_mode": None,
            "task_selection": task_selection,
        },
        daemon=True,
    ).start()

    return jsonify({"ok": True, "job_id": job_id, "failed_steps": sorted(failed_steps)}), 202


@app.route("/jobs/<job_id>/reprocess", methods=["POST"])
def reprocess_job(job_id: str) -> Any:
    job_id = (job_id or "").strip()
    if not job_id:
        return jsonify({"error": "Missing job_id"}), 400

    job_row = _db_get_job(job_id)
    if not job_row:
        return jsonify({"error": "Job not found"}), 404

    payload = request.get_json(silent=True) or {}
    gcs_uri = str(job_row.get("gcs_video_uri") or "").strip()
    if not gcs_uri:
        gcs_uri = str(payload.get("gcs_video_uri") or payload.get("gcs_uri") or payload.get("gcs_object") or "").strip()
    if not gcs_uri:
        return jsonify({"error": "Missing gcs_video_uri for job"}), 400

    try:
        bucket, obj = _parse_allowed_gcs_video_source(gcs_uri)
    except Exception as exc:
        alt_uri = str(job_row.get("gcs_working_uri") or "").strip()
        if alt_uri and "Object must be under" in str(exc):
            try:
                bucket, obj = _parse_allowed_gcs_video_source(alt_uri)
                gcs_uri = alt_uri
            except Exception:
                pass
        if "Object must be under" in str(exc):
            try:
                bucket, obj = _parse_allowed_gcs_video_source(gcs_uri, enforce_prefix=False)
            except Exception:
                return jsonify({"error": str(exc)}), 400
        else:
            return jsonify({"error": str(exc)}), 400

    task_selection = payload.get("task_selection") or payload.get("taskSelection") or payload.get("selection")
    if not isinstance(task_selection, dict):
        task_selection = job_row.get("task_selection") if isinstance(job_row.get("task_selection"), dict) else None

    title = str(job_row.get("title") or "Reprocess").strip() or "Reprocess"
    _job_init(job_id, title=title)
    _job_update(job_id, status="processing", progress=1, message="Reprocess queued", gcs_video_uri=gcs_uri)

    frame_interval_seconds = _parse_int(request.args.get("frame_interval_seconds"), default=0, min_value=0, max_value=30)
    max_frames_to_analyze = _parse_int(request.args.get("max_frames_to_analyze"), default=1000, min_value=1, max_value=10000)
    threading.Thread(
        target=_process_gcs_video_job_cloud_only,
        kwargs={
            "job_id": job_id,
            "gcs_bucket": bucket,
            "gcs_object": obj,
            "video_title": title or Path(obj).name,
            "video_description": str(job_row.get("description") or ""),
            "frame_interval_seconds": frame_interval_seconds,
            "max_frames_to_analyze": max_frames_to_analyze,
            "face_recognition_mode": None,
            "task_selection": task_selection,
        },
        daemon=True,
    ).start()

    return jsonify({"ok": True, "job_id": job_id}), 202


@app.route("/jobs/<job_id>/steps/<step_id>/restart", methods=["POST"])
def restart_job_step(job_id: str, step_id: str) -> Any:
    job_id = (job_id or "").strip()
    step_id = (step_id or "").strip()
    if not job_id or not step_id:
        return jsonify({"error": "Missing job_id or step_id"}), 400

    job_row = _db_get_job(job_id)
    if not job_row:
        return jsonify({"error": "Job not found"}), 404

    payload = request.get_json(silent=True) or {}
    gcs_uri = str(payload.get("gcs_video_uri") or payload.get("gcs_uri") or payload.get("gcs_object") or "").strip()
    if not gcs_uri:
        gcs_uri = str(job_row.get("gcs_video_uri") or "").strip()
    if not gcs_uri:
        with VIDEO_INDEX_LOCK:
            v_lookup = next((x for x in VIDEO_INDEX if str(x.get("id")) == job_id), None)
        if isinstance(v_lookup, dict):
            gcs_uri = str(v_lookup.get("gcs_video_uri") or "").strip()
    if not gcs_uri:
        return jsonify({"error": "Missing gcs_video_uri for job"}), 400

    try:
        bucket, obj = _parse_allowed_gcs_video_source(gcs_uri)
    except Exception as exc:
        alt_uri = str(job_row.get("gcs_working_uri") or "").strip()
        if alt_uri and "Object must be under" in str(exc):
            try:
                bucket, obj = _parse_allowed_gcs_video_source(alt_uri)
                gcs_uri = alt_uri
            except Exception:
                pass
        if "Object must be under" in str(exc):
            try:
                bucket, obj = _parse_allowed_gcs_video_source(gcs_uri, enforce_prefix=False)
            except Exception:
                return jsonify({"error": str(exc)}), 400
        else:
            return jsonify({"error": str(exc)}), 400

    task_selection = _task_selection_for_failed_steps({step_id})
    if not task_selection:
        return jsonify({"error": f"Step cannot be restarted: {step_id}"}), 400
    task_selection["partial_reprocess"] = True

    title = str(job_row.get("title") or "Reprocess").strip() or "Reprocess"
    with VIDEO_INDEX_LOCK:
        v = next((x for x in VIDEO_INDEX if str(x.get("id")) == job_id), None)
    video_title = (v.get("title") if isinstance(v, dict) else None) or title
    video_description = (v.get("description") if isinstance(v, dict) else "") or ""

    _job_init(job_id, title=video_title)
    _job_update(job_id, status="processing", progress=1, message=f"Reprocess step queued: {step_id}", gcs_video_uri=gcs_uri)

    frame_interval_seconds = _parse_int(request.args.get("frame_interval_seconds"), default=0, min_value=0, max_value=30)
    max_frames_to_analyze = _parse_int(request.args.get("max_frames_to_analyze"), default=1000, min_value=1, max_value=10000)

    threading.Thread(
        target=_process_gcs_video_job_cloud_only,
        kwargs={
            "job_id": job_id,
            "gcs_bucket": bucket,
            "gcs_object": obj,
            "video_title": video_title,
            "video_description": video_description,
            "frame_interval_seconds": frame_interval_seconds,
            "max_frames_to_analyze": max_frames_to_analyze,
            "face_recognition_mode": None,
            "task_selection": task_selection,
        },
        daemon=True,
    ).start()

    return jsonify({"ok": True, "job_id": job_id, "step_id": step_id}), 202


@app.route("/search", methods=["POST"])
def search_videos() -> Any:
    return jsonify({"error": "Search has been removed"}), 410


@app.route("/upload-document", methods=["POST"])
def upload_document() -> Any:
    return jsonify({"error": "Document extraction is disabled"}), 410
    with DOCUMENT_INDEX_LOCK:
        DOCUMENT_INDEX.append(entry)
    _save_document_index()
    return jsonify({"ok": True, "document": {"id": doc_id, "filename": filename}}), 200


@app.route("/documents", methods=["GET"])
def list_documents() -> Any:
    with DOCUMENT_INDEX_LOCK:
        docs = list(DOCUMENT_INDEX)
    docs.sort(key=lambda d: (d.get("uploaded_at") or "", d.get("id") or ""), reverse=True)
    slim = [{"id": d.get("id"), "filename": d.get("filename"), "uploaded_at": d.get("uploaded_at")} for d in docs]
    return jsonify({"documents": slim, "count": len(slim)}), 200


@app.route("/documents/<document_id>", methods=["DELETE"])
def delete_document(document_id: str) -> Any:
    with DOCUMENT_INDEX_LOCK:
        d = next((x for x in DOCUMENT_INDEX if str(x.get("id")) == str(document_id)), None)
    if not d:
        return jsonify({"error": "Document not found"}), 404
    try:
        stored = (d.get("stored_path") or "").strip()
        if stored:
            p = DOCUMENTS_DIR / stored
            if p.exists():
                p.unlink()
    except Exception:
        pass
    with DOCUMENT_INDEX_LOCK:
        DOCUMENT_INDEX[:] = [x for x in DOCUMENT_INDEX if str(x.get("id")) != str(document_id)]
    _save_document_index()
    return jsonify({"ok": True}), 200


def _extract_match_snippet(text: str, query_terms: List[str], *, window: int = 160) -> str | None:
    if not text:
        return None
    lower = text.lower()
    hits: List[int] = []
    for t in query_terms:
        t = t.lower().strip()
        if not t:
            continue
        idx = lower.find(t)
        if idx >= 0:
            hits.append(idx)
    if not hits:
        return None
    i = min(hits)
    start = max(0, i - window)
    end = min(len(text), i + window)
    return text[start:end].strip()


@app.route("/search-text", methods=["POST"])
def search_text() -> Any:
    return jsonify({"error": "Document search has been removed"}), 410


@app.route("/ask-question", methods=["POST"])
def ask_question() -> Any:
    return jsonify({"error": "Question answering has been removed"}), 410


@app.route("/askme", methods=["POST"])
def askme() -> Any:
    return ask_question()


@app.route("/local-faces", methods=["GET"])
def local_faces_list() -> Any:
    return jsonify({"error": "Local face recognition has been removed"}), 410


@app.route("/local-faces/enroll", methods=["POST"])
def local_faces_enroll() -> Any:
    return jsonify({"error": "Local face recognition has been removed"}), 410


_load_indices()


if __name__ == "__main__":
    port = _parse_int(os.getenv("ENVID_METADATA_PORT") or "5016", default=5016, min_value=1, max_value=65535)
    host = (os.getenv("ENVID_METADATA_HOST") or "0.0.0.0").strip() or "0.0.0.0"
    debug = _env_truthy(os.getenv("FLASK_DEBUG"), default=False)
    app.run(host=host, port=port, debug=debug)
