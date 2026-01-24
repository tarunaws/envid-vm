from __future__ import annotations

import base64
import difflib
import io
import json
import mimetypes
import os
import platform
import re
import shutil
import sys
import sysconfig
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from math import exp
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error
import urllib.parse
import requests

from flask import Flask, Response, jsonify, redirect, request, send_file
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from shared.env_loader import load_environment

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:  # pragma: no cover
    PaddleOCR = None  # type: ignore

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from google.cloud import storage as gcs_storage  # type: ignore
except Exception:  # pragma: no cover
    gcs_storage = None  # type: ignore

try:
    from google.cloud import speech_v1 as gcp_speech  # type: ignore
except Exception:  # pragma: no cover
    gcp_speech = None  # type: ignore

try:
    from google.cloud import translate_v3 as gcp_translate  # type: ignore
except Exception:  # pragma: no cover
    gcp_translate = None  # type: ignore

try:
    from google.cloud import language_v1 as gcp_language  # type: ignore
except Exception:  # pragma: no cover
    gcp_language = None  # type: ignore

try:
    from google.cloud import videointelligence_v1 as gcp_video_intelligence  # type: ignore
except Exception:  # pragma: no cover
    gcp_video_intelligence = None  # type: ignore


try:
    from ultralytics import YOLO as UltralyticsYOLO  # type: ignore
except Exception:  # pragma: no cover
    UltralyticsYOLO = None  # type: ignore

_NUDENET_IMPORT_ERROR: str | None = None
try:
    # NudeNet is optional and may require tensorflow.
    from nudenet import NudeClassifier  # type: ignore
except Exception as exc:  # pragma: no cover
    NudeClassifier = None  # type: ignore
    _NUDENET_IMPORT_ERROR = str(exc)

try:
    from PyPDF2 import PdfReader
except Exception:  # pragma: no cover
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:  # pragma: no cover
        PdfReader = None  # type: ignore

try:
    from docx import Document as DocxDocument  # type: ignore
except Exception:  # pragma: no cover
    DocxDocument = None  # type: ignore



try:
    import language_tool_python  # type: ignore
except Exception:  # pragma: no cover
    language_tool_python = None  # type: ignore

try:
    from wordfreq import zipf_frequency, top_n_list  # type: ignore
except Exception:  # pragma: no cover
    zipf_frequency = None  # type: ignore
    top_n_list = None  # type: ignore

try:
    from rapidfuzz import process as rapid_process  # type: ignore
    from rapidfuzz import fuzz as rapid_fuzz  # type: ignore
except Exception:  # pragma: no cover
    rapid_process = None  # type: ignore
    rapid_fuzz = None  # type: ignore


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key:
                os.environ[key] = value
    except Exception:
        return


def _bootstrap_env() -> None:
    try:
        project_root = Path(__file__).resolve().parents[2]
    except Exception:
        return
    target = (os.getenv("ENVID_ENV_TARGET") or "").strip()
    if not target:
        target = "laptop" if platform.system().lower() == "darwin" else "vm"
    env_files = [
        project_root / ".env",
        project_root / ".env.local",
        project_root / ".env.multimodal.local",
        project_root / f".env.multimodal.{target}.local",
        project_root / ".env.multimodal.secrets.local",
        project_root / f".env.multimodal.{target}.secrets.local",
    ]
    for env_path in env_files:
        _load_env_file(env_path)


_bootstrap_env()


load_environment()

# Ensure GOOGLE_APPLICATION_CREDENTIALS points to an existing file.
_gac = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
if _gac and not Path(_gac).expanduser().exists():
    fallback_gac = Path.home() / "gcpAccess" / "gcp.json"
    if fallback_gac.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(fallback_gac)

# Only Google Video Intelligence is allowed; disable other GCP services by default.
_DISABLE_GCP_NON_VI = (os.getenv("ENVID_METADATA_ONLY_GCP_VIDEO_INTELLIGENCE") or "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
if _DISABLE_GCP_NON_VI:
    gcp_speech = None  # type: ignore
    gcp_translate = None  # type: ignore
    gcp_language = None  # type: ignore

app = Flask(__name__)
CORS(app)


class TranscriptVerificationError(RuntimeError):
    pass


def _gcp_project_id() -> str | None:
    return (os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip() or None


def _gcp_location() -> str:
    return (os.getenv("GCP_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1").strip() or "us-central1"


def _gcp_translate_location() -> str:
    # Translate v3 commonly supports "global" (and sometimes a limited set of regions).
    # Our stack uses GCP_LOCATION for other regional services, so keep
    # Translate separate to avoid invalid-location errors.
    return (os.getenv("GCP_TRANSLATE_LOCATION") or "global").strip() or "global"


def _translate_provider() -> str:
    pref = (os.getenv("ENVID_METADATA_TRANSLATE_PROVIDER") or "auto").strip().lower()
    libre_url = (os.getenv("ENVID_LIBRETRANSLATE_URL") or os.getenv("LIBRETRANSLATE_URL") or "").strip()
    has_libre = bool(libre_url)
    # Google Translate is explicitly disabled (GCP is reserved for label detection only).
    # Keep the env flag for legacy compatibility, but never allow it to enable translation.
    allow_gcp = False
    has_gcp = False

    if pref in {"libretranslate", "libre", "opus", "marian"}:
        return "libretranslate" if has_libre else "disabled"
    if pref in {"gcp", "google", "google_translate"}:
        return "disabled"

    if has_libre:
        return "libretranslate"
    return "disabled"


_LIBRE_LANG_CACHE: tuple[set[str], float] = (set(), 0.0)


def _libretranslate_supported_langs(base_url: str) -> set[str]:
    global _LIBRE_LANG_CACHE
    endpoint = base_url.rstrip("/") + "/languages"
    cached, ts = _LIBRE_LANG_CACHE
    if cached and (time.time() - ts) < 600:
        return cached
    try:
        resp = requests.get(endpoint, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        langs: set[str] = set()
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    code = str(item.get("code") or "").strip().lower()
                    if code:
                        langs.add(code)
        if langs:
            _LIBRE_LANG_CACHE = (langs, time.time())
            return langs
    except Exception:
        pass
    return cached


def _libretranslate_languages_raw(base_url: str) -> list[dict[str, Any]]:
    endpoint = base_url.rstrip("/") + "/languages"
    resp = requests.get(endpoint, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _libretranslate_translate(*, text: str, source_lang: str | None, target_lang: str) -> str:
    base_url = (os.getenv("ENVID_LIBRETRANSLATE_URL") or os.getenv("LIBRETRANSLATE_URL") or "").strip()
    if not base_url:
        raise RuntimeError("LibreTranslate URL is not configured")

    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/translate"):
        endpoint = f"{endpoint}/translate"

    api_key = (os.getenv("ENVID_LIBRETRANSLATE_API_KEY") or os.getenv("LIBRETRANSLATE_API_KEY") or "").strip()
    timeout_s = _safe_float(os.getenv("ENVID_LIBRETRANSLATE_TIMEOUT_SECONDS"), 20.0)

    supported = _libretranslate_supported_langs(base_url)
    src = (source_lang or "auto").strip().lower() or "auto"
    tgt = (target_lang or "en").strip().lower() or "en"
    if supported:
        if tgt not in supported:
            return ""
        if src != "auto" and src not in supported:
            src = "auto"

    payload: Dict[str, Any] = {
        "q": text,
        "source": src or "auto",
        "target": tgt or "en",
        "format": "text",
    }
    if api_key:
        payload["api_key"] = api_key

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as r:
        resp = json.loads(r.read().decode("utf-8"))
    translated = resp.get("translatedText") or resp.get("translated_text") or ""
    return str(translated or "").strip()


def _whisper_docker_transcribe(
    *,
    audio_path: Path,
    model_size: str,
    decode_kwargs: Dict[str, Any],
    prompt: str | None,
) -> dict[str, Any]:
    raise RuntimeError("Whisper docker is disabled; WhisperX only")


def _whisperx_command() -> list[str] | None:
    bin_path = (os.getenv("ENVID_WHISPERX_BIN") or "").strip()
    if bin_path:
        return [bin_path]
    py_path = (os.getenv("ENVID_WHISPERX_PYTHON") or "").strip()
    if py_path:
        return [py_path, "-m", "whisperx"]
    if shutil.which("whisperx"):
        return ["whisperx"]
    return None


def _whisperx_available() -> bool:
    return _whisperx_command() is not None


def _translate_targets() -> list[str]:
    raw = (os.getenv("ENVID_METADATA_TRANSLATE_LANGS") or "en,ar,id").strip()
    langs = [seg.strip().lower() for seg in raw.split(",") if seg.strip()]
    if not langs:
        langs = ["en", "ar", "id"]
    # De-duplicate while preserving order.
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
    # De-duplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for lang in langs:
        if lang in seen:
            continue
        seen.add(lang)
        out.append(lang)
    return out


def _init_translate_client(provider: str) -> tuple[Any | None, str | None]:
    if provider == "gcp_translate":
        raise RuntimeError("GCP Translate is disabled in this stack")
    if provider != "gcp_translate":
        return None, None
    if gcp_translate is None:
        raise RuntimeError("GCP Translate not available")
    pid = _gcp_project_id()
    if not pid:
        raise RuntimeError("Missing GCP_PROJECT_ID")
    parent = f"projects/{pid}/locations/{_gcp_translate_location()}"
    client = gcp_translate.TranslationServiceClient()
    return client, parent


def _translate_text(
    *,
    text: str,
    source_lang: str | None,
    target_lang: str,
    provider: str,
    gcp_client: Any | None,
    gcp_parent: str | None,
) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    if not target_lang:
        return raw
    src = (source_lang or "").strip()
    tgt = target_lang.strip().lower()
    if src and (src.lower() == tgt or src.lower().startswith(f"{tgt}-")):
        return raw
    if provider == "libretranslate":
        return _libretranslate_translate(text=raw[:4500], source_lang=(src if src and len(src) >= 2 else None), target_lang=tgt)
    if provider == "gcp_translate":
        if gcp_client is None or gcp_parent is None:
            raise RuntimeError("GCP Translate client not initialized")
        req: Dict[str, Any] = {"parent": gcp_parent, "contents": [raw[:4500]], "mime_type": "text/plain", "target_language_code": tgt}
        if src and len(src) >= 2:
            req["source_language_code"] = src
        resp = gcp_client.translate_text(request=req)
        if resp and resp.translations:
            return (resp.translations[0].translated_text or "").strip()
    return raw


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
        if not isinstance(seg, dict):
            continue
        txt = str(seg.get("text") or "").strip()
        if not txt:
            continue
        tr = _translate_text(
            text=txt,
            source_lang=source_lang,
            target_lang=target_lang,
            provider=provider,
            gcp_client=gcp_client,
            gcp_parent=gcp_parent,
        )
        item = dict(seg)
        item["text"] = tr or txt
        out.append(item)
    return out




def _env_truthy(value: str | None, *, default: bool = True) -> bool:
    if value is None:
        return default
    v = str(value).strip().lower()
    if v in {"0", "false", "no", "off"}:
        return False
    if v in {"1", "true", "yes", "on"}:
        return True
    return default


def _parse_int(value: Any, *, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        v = int(value)
    except Exception:
        v = default
    if min_value is not None:
        v = max(min_value, v)
    if max_value is not None:
        v = min(max_value, v)
    return v


def _build_ffmpeg_atempo_chain(tempo: float) -> str:
    """Build a safe ffmpeg atempo chain for a requested tempo.

    ffmpeg's atempo filter generally supports 0.5..2.0 per filter instance.
    We chain multiple atempo filters when tempo is outside that range.
    """

    try:
        t = float(tempo)
    except Exception:
        return ""
    if t <= 0:
        return ""
    if abs(t - 1.0) < 1e-6:
        return ""

    parts: list[float] = []
    while t < 0.5:
        parts.append(0.5)
        t = t / 0.5
    while t > 2.0:
        parts.append(2.0)
        t = t / 2.0
    parts.append(t)
    return ",".join([f"atempo={p:.6f}".rstrip("0").rstrip(".") for p in parts if p and abs(p - 1.0) > 1e-6])


def _languagetool_correct_text(*, text: str, language: str | None) -> tuple[str, list[dict[str, Any]]]:
    """Optional grammar correction via an external LanguageTool HTTP endpoint.

    Set ENVID_GRAMMAR_CORRECTION_URL (e.g., http://localhost:8010) to enable.
    Returns (corrected_text, suggestions).
    """

    base = (os.getenv("ENVID_GRAMMAR_CORRECTION_URL") or "").strip().rstrip("/")
    use_local = False
    if not base and not (use_local and language_tool_python is not None):
        return text, []

    lang = (language or "").strip() or "auto"
    url = f"{base}/v2/check" if base else ""
    timeout_seconds = 12.0
    max_chars = 12000

    src = (text or "").strip()
    if not src:
        return text, []
    if len(src) > max_chars:
        src = src[:max_chars]

    if base:
        form = urllib.parse.urlencode({"text": src, "language": lang}).encode("utf-8")
        req = urllib.request.Request(url, data=form, headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as r:
                resp = json.loads(r.read().decode("utf-8"))
        except Exception:
            return text, []
    else:
        resp = _languagetool_local_check(text=src, language=lang)
        if not resp:
            return text, []

    matches = resp.get("matches")
    if not isinstance(matches, list) or not matches:
        return text, []

    suggestions: list[dict[str, Any]] = []
    edits: list[tuple[int, int, str]] = []

    for m in matches[:250]:
        if not isinstance(m, dict):
            continue
        try:
            offset = int(m.get("offset") or 0)
            length = int(m.get("length") or 0)
        except Exception:
            continue
        repls = m.get("replacements")
        replacement = ""
        if isinstance(repls, list) and repls:
            first = repls[0]
            if isinstance(first, dict):
                replacement = str(first.get("value") or "")
            else:
                replacement = str(first)

        suggestions.append(
            {
                "offset": offset,
                "length": length,
                "replacement": replacement,
                "message": str(m.get("message") or "")[:240],
                "rule": (m.get("rule") or {}).get("id") if isinstance(m.get("rule"), dict) else None,
            }
        )
        if length > 0 and replacement:
            edits.append((offset, length, replacement))

    corrected = src
    # Apply from end to start to keep offsets valid.
    for offset, length, replacement in sorted(edits, key=lambda x: x[0], reverse=True):
        if offset < 0 or length <= 0:
            continue
        if offset + length > len(corrected):
            continue
        corrected = corrected[:offset] + replacement + corrected[offset + length :]

    return corrected.strip(), suggestions


def _normalize_transcript_basic(text: str) -> str:
    out = (text or "").strip()
    if not out:
        return ""
    out = re.sub(r"[\t\r]+", " ", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def _enhance_transcript_punctuation(text: str) -> str:
    """Lightweight, deterministic punctuation/spacing cleanup (language-agnostic, Hindi-friendly)."""

    out = _normalize_transcript_basic(text)
    if not out:
        return ""

    # Remove spaces before punctuation.
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"\s+([।])", r"\1", out)

    # Ensure space after punctuation when followed by a letter/number (Latin or Devanagari).
    out = re.sub(r"([,.;:!?])([A-Za-z0-9\u0900-\u097F])", r"\1 \2", out)
    out = re.sub(r"(।)([A-Za-z0-9\u0900-\u097F])", r"\1 \2", out)

    # Normalize spaces around brackets.
    out = re.sub(r"([\(\[\{])\s+", r"\1", out)
    out = re.sub(r"\s+([\)\]\}])", r"\1", out)

    # Ensure single spacing after punctuation tokens.
    out = re.sub(r"([,.;:!?])\s+", r"\1 ", out)
    out = re.sub(r"(।)\s+", r"। ", out)

    # Collapse repeated punctuation (e.g., "...." -> ".").
    out = re.sub(r"([.!?।])\1{1,}", r"\1", out)

    # Final whitespace cleanup.
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


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
    nlp_mode: str,
    punctuation_enabled: bool,
) -> tuple[str, dict[str, bool]]:
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

    if nlp_mode in {"openrouter_llama", "llama", "openrouter"}:
        normalized, _ = _openrouter_llama_normalize_transcript(text=out, language_code=language_code)
        if normalized and normalized != out:
            out = normalized
            meta["nlp_applied"] = True

    if grammar_enabled:
        corrected, _ = _languagetool_correct_text(text=out, language=language_code)
        if corrected and corrected != out:
            out = corrected
            meta["grammar_applied"] = True

    # Apply dictionary correction (language-aware) before deterministic punctuation.
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

    # Fill gaps so every moment is covered.
    filled: list[tuple[float, float]] = []
    cur = 0.0
    for st, en in normalized:
        if st > cur + 0.05:
            filled.append((cur, st))
        filled.append((st, en))
        cur = max(cur, en)
    if dur > 0 and cur < dur - 0.05:
        filled.append((cur, dur))

    # Clamp and drop tiny segments.
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


def _openrouter_llama_normalize_transcript(*, text: str, language_code: str | None) -> tuple[str | None, dict[str, Any]]:
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        return None, {"available": False}

    raw = _normalize_transcript_basic(text)
    if not raw:
        return None, {"available": True, "applied": False}

    base_url = "https://openrouter.ai/api/v1"
    model = "meta-llama/llama-3.3-70b-instruct:free"
    timeout_s = 15.0

    lang = (language_code or "").strip() or "unknown"
    strict = False
    extra = ""
    if strict:
        extra = (
            "CRITICAL: Use only facts from the transcript. If information is unclear or missing, say so briefly.\n"
            "Avoid generic filler. Be specific to transcript details.\n"
        )
    prompt = (
        "You are a transcript normalizer.\n"
        "Task: improve readability ONLY by fixing punctuation, casing, spacing, and obvious sentence boundaries.\n"
        "Hard rules:\n"
        "- Do NOT add new words or remove words.\n"
        "- Do NOT guess missing words.\n"
        "- Do NOT rewrite, paraphrase, or summarize.\n"
        "- Keep the language as-is.\n"
        "- For Hindi, prefer the danda (।) for sentence endings.\n"
        "- Output plain text only (no markdown, no quotes).\n\n"
        f"{extra}"
        f"Language hint: {lang}\n\n"
        f"Transcript:\n{raw[:12000]}\n"
    )

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    http_referer = ""
    x_title = ""
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if x_title:
        headers["X-Title"] = x_title

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Fix punctuation/casing/spacing only. Output plain text."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as r:
            resp = json.loads(r.read().decode("utf-8"))
        content = (((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        if content:
            return content, {"available": True, "applied": True, "provider": "openrouter", "model": model}
        return None, {"available": True, "applied": False, "provider": "openrouter", "model": model}
    except Exception:
        return None, {"available": True, "applied": False, "provider": "openrouter", "model": model}


def _openrouter_llama_generate_synopses(*, text: str, language_code: str | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Generate age-group synopses using OpenRouter (Meta Llama).

    Returns (synopses_or_none, meta).
    """

    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        return None, {"available": False}

    raw = _normalize_transcript_basic(text)
    if not raw:
        return None, {"available": True, "applied": False}

    base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip().rstrip("/")
    model = (
        os.getenv("OPENROUTER_SYNOPSIS_MODEL")
        or os.getenv("OPENROUTER_TRANSCRIPT_MODEL")
        or os.getenv("OPENROUTER_MODEL")
        or "meta-llama/llama-3.3-70b-instruct:free"
    ).strip()
    timeout_s = _safe_float(os.getenv("OPENROUTER_TIMEOUT_SECONDS"), 20.0)

    lang = (language_code or "").strip() or "unknown"
    prompt = (
        "You are a synopsis generator.\n"
        "Create short and long synopses for three age groups.\n"
        "Return STRICT JSON only with keys: kids, teens, adults.\n"
        "Each value must be an object with keys: short, long.\n"
        "Do not add any extra keys or commentary.\n"
        "Write in the SAME LANGUAGE as the transcript. Do NOT translate.\n\n"
        f"Language hint: {lang}\n\n"
        f"Transcript:\n{raw[:12000]}\n"
    )

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    http_referer = (os.getenv("OPENROUTER_HTTP_REFERER") or "").strip()
    x_title = (os.getenv("OPENROUTER_X_TITLE") or "").strip()
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if x_title:
        headers["X-Title"] = x_title

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 900,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as r:
            resp = json.loads(r.read().decode("utf-8"))
        content = (((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        payload_json = json.loads(m.group(0) if m else content)
        if isinstance(payload_json, dict):
            return payload_json, {"available": True, "applied": True, "provider": "openrouter", "model": model}
        return None, {"available": True, "applied": False, "provider": "openrouter", "model": model}
    except Exception:
        return None, {"available": True, "applied": False, "provider": "openrouter", "model": model}


def _openrouter_llama_scene_summaries(
    *,
    scenes: list[dict[str, Any]],
    transcript_segments: list[dict[str, Any]],
    labels_src: list[dict[str, Any]] | None,
    language_code: str | None,
) -> tuple[dict[int, str] | None, dict[str, Any]]:
    """Generate scene-by-scene summaries using OpenRouter (Meta Llama).

    Returns (scene_index_to_summary_or_none, meta).
    """

    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        return None, {"available": False}

    if not scenes:
        return None, {"available": True, "applied": False}

    base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip().rstrip("/")
    model = (
        os.getenv("OPENROUTER_SCENE_MODEL")
        or os.getenv("OPENROUTER_SYNOPSIS_MODEL")
        or os.getenv("OPENROUTER_MODEL")
        or "meta-llama/llama-3.3-70b-instruct:free"
    ).strip()
    timeout_s = _safe_float(os.getenv("OPENROUTER_TIMEOUT_SECONDS"), 25.0)

    max_scenes = _parse_int(os.getenv("ENVID_SCENE_LLM_MAX_SCENES"), default=30, min_value=1, max_value=200)
    max_chars = _parse_int(os.getenv("ENVID_SCENE_LLM_MAX_CHARS_PER_SCENE"), default=600, min_value=120, max_value=4000)

    lang = (language_code or "").strip() or "unknown"

    def _scene_context_text(sc: dict[str, Any]) -> tuple[int, str]:
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
        if labels_src:
            for lab in labels_src:
                if not isinstance(lab, dict):
                    continue
                if _count_overlapping_segments([lab], st, en) <= 0:
                    continue
                name = str(lab.get("name") or lab.get("label") or "").strip()
                if name:
                    label_names.append(name)
        label_names = list(dict.fromkeys(label_names))

        transcript_text = " ".join(segs).strip()
        if len(transcript_text) > max_chars:
            transcript_text = transcript_text[:max_chars].rsplit(" ", 1)[0].strip()

        labels_text = ", ".join(label_names[:20])
        if labels_text:
            context = f"Transcript: {transcript_text}\nLabels: {labels_text}".strip()
        else:
            context = f"Transcript: {transcript_text}".strip()
        return idx, context

    scene_inputs: list[dict[str, Any]] = []
    for sc in scenes[:max_scenes]:
        if not isinstance(sc, dict):
            continue
        idx, context = _scene_context_text(sc)
        if not context.replace("Transcript:", "").strip() and "Labels:" not in context:
            continue
        scene_inputs.append({"index": idx, "context": context})

    if not scene_inputs:
        return None, {"available": True, "applied": False, "provider": "openrouter", "model": model}

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

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    http_referer = (os.getenv("OPENROUTER_HTTP_REFERER") or "").strip()
    x_title = (os.getenv("OPENROUTER_X_TITLE") or "").strip()
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if x_title:
        headers["X-Title"] = x_title

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1400,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as r:
            resp = json.loads(r.read().decode("utf-8"))
        content = (((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
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
        return None, {"available": True, "applied": False, "provider": "openrouter", "model": model}
    except Exception:
        return None, {"available": True, "applied": False, "provider": "openrouter", "model": model}


def _synopsis_is_reasonable(text: str, language_code: str | None, *, min_words: int, max_words: int) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False

    words = [w for w in re.split(r"\s+", raw) if w]
    if len(words) < min_words or len(words) > max_words:
        return False

    letters = sum(1 for ch in raw if ch.isalpha())
    alpha_ratio = letters / max(1, len(raw))
    if alpha_ratio < 0.2:
        return False

    lang = (language_code or "").strip().lower()
    if lang.startswith("hi"):
        devanagari = sum(1 for ch in raw if "\u0900" <= ch <= "\u097F")
        if devanagari / max(1, letters) < 0.4:
            return False

    # Avoid pathological repetition (e.g., same word repeated).
    uniq = len({w.lower() for w in words})
    if uniq / max(1, len(words)) < 0.3:
        return False

    return True


def _validate_synopses_payload(payload: dict[str, Any], language_code: str | None) -> bool:
    if not isinstance(payload, dict):
        return False
    for group in ("kids", "teens", "adults"):
        item = payload.get(group)
        if not isinstance(item, dict):
            return False
        short = item.get("short")
        long = item.get("long")
        if not _synopsis_is_reasonable(str(short or ""), language_code, min_words=6, max_words=120):
            return False
        if not _synopsis_is_reasonable(str(long or ""), language_code, min_words=15, max_words=300):
            return False
    return True


def _fallback_synopses_from_text(text: str, language_code: str | None) -> dict[str, Any] | None:
    """Best-effort fallback synopses using transcript text only.

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
    if _validate_synopses_payload(payload, language_code):
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


def _gcs_rawvideo_prefix() -> str:
    prefix = (os.getenv("GCP_GCS_RAWVIDEO_PREFIX") or "rawVideo/").strip()
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
        raise RuntimeError("google-cloud-storage is not installed. Run: pip install -r code/requirements.txt")
    return gcs_storage.Client()


def _parse_allowed_gcs_video_source(raw: str) -> Tuple[str, str]:
    v = (raw or "").strip()
    if not v:
        raise ValueError("Missing gcs_object (or gcs_uri)")

    bucket = _gcs_bucket_name()
    allowed = _allowed_gcs_buckets()
    prefix = _gcs_rawvideo_prefix()

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

    if _gcs_enforce_prefix() and not obj.startswith(prefix):
        raise ValueError(f"Object must be under {prefix}")
    if obj.endswith("/"):
        raise ValueError("Object cannot be a folder")

    ext = (Path(obj).suffix or "").lower()
    if ext not in {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mxf"}:
        raise ValueError("Only video files are allowed (.mp4, .mov, .m4v, .mkv, .avi, .webm, .mxf)")
    return bucket, obj


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


def _persist_local_video_copy() -> bool:
    return _env_truthy(os.getenv("ENVID_METADATA_PERSIST_LOCAL_VIDEO"), default=False)


FFMPEG_PATH = os.environ.get("FFMPEG_PATH")
if not FFMPEG_PATH:
    for path in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg", "ffmpeg"]:
        try:
            result = subprocess.run([path, "-version"], capture_output=True, timeout=2)
            if result.returncode == 0:
                FFMPEG_PATH = path
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    if not FFMPEG_PATH:
        FFMPEG_PATH = "ffmpeg"

FFPROBE_PATH = FFMPEG_PATH.replace("ffmpeg", "ffprobe") if "ffmpeg" in FFMPEG_PATH else "ffprobe"


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
STORAGE_BASE_DIR = SERVICE_DIR
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
        json.dump(value, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _load_indices() -> None:
    global VIDEO_INDEX, DOCUMENT_INDEX
    VIDEO_INDEX = _safe_json_load(VIDEO_INDEX_FILE, [])
    if not isinstance(VIDEO_INDEX, list):
        VIDEO_INDEX = []
    DOCUMENT_INDEX = _safe_json_load(DOCUMENT_INDEX_FILE, [])
    if not isinstance(DOCUMENT_INDEX, list):
        DOCUMENT_INDEX = []


def _save_video_index() -> None:
    with VIDEO_INDEX_LOCK:
        _safe_json_save(VIDEO_INDEX_FILE, VIDEO_INDEX)


def _save_document_index() -> None:
    with DOCUMENT_INDEX_LOCK:
        _safe_json_save(DOCUMENT_INDEX_FILE, DOCUMENT_INDEX)


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

_LANGTOOL_LOCAL: Any | None = None
_LANGTOOL_LOCAL_LANG: str | None = None
_LANG_DICTIONARY_WORDS: dict[str, list[str]] = {}
_LANG_DICTIONARY_SET: dict[str, set[str]] = {}
_LANG_CONFUSION_MAP: dict[str, dict[str, str]] = {}


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
    if not text:
        return ""
    lang = _normalize_language_code(language_code)
    if not lang:
        return text
    if lang not in _dictionary_languages_enabled():
        return text
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
        {"id": "precheck_models", "label": "Precheck models", "status": "not_started", "percent": 0, "message": None},
        {"id": "upload_to_cloud_storage", "label": "Upload to cloud storage", "status": "not_started", "percent": 0, "message": None},
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
        {"id": "save_as_json", "label": "Save as Json", "status": "not_started", "percent": 0, "message": None},
        {"id": "overall", "label": "Overall", "status": "not_started", "percent": 0, "message": None},
    ]


def _job_init(job_id: str, *, title: str | None = None) -> None:
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "title": title,
            "status": "queued",
            "progress": 0,
            "message": None,
            "steps": _job_steps_default(),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }


def _job_update(job_id: str, **fields: Any) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(fields)
        job["updated_at"] = datetime.utcnow().isoformat()


def _job_step_update(job_id: str, step_id: str, *, status: str | None = None, percent: int | None = None, message: str | None = None) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
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
    if not shutil.which(FFPROBE_PATH):
        return {"available": False, "error": f"ffprobe not found (FFPROBE_PATH={FFPROBE_PATH})"}
    try:
        cmd = [FFPROBE_PATH, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(video_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return {"available": False, "error": (res.stderr or "ffprobe failed").strip()}
        data = json.loads(res.stdout or "{}")
        return {"available": True, "raw": data}
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
    use_local_label_detection_service: bool,
    use_local_moderation: bool,
    allow_moderation_fallback: bool,
    use_local_ocr: bool,
    want_any_vi: bool,
    local_label_detection_url_override: str,
    local_moderation_url_override: str,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    errors: list[str] = []
    warnings: list[str] = []

    def _require(ok: bool, name: str, msg: str) -> None:
        checks[name] = {"ok": bool(ok), "message": msg}
        if not ok:
            errors.append(msg)

    def _warn(ok: bool, name: str, msg: str) -> None:
        checks[name] = {"ok": bool(ok), "message": msg}
        if not ok:
            warnings.append(msg)

    # ffmpeg/ffprobe availability (core pipeline tools)
    _require(bool(shutil.which(FFMPEG_PATH)), "ffmpeg", f"ffmpeg not found (FFMPEG_PATH={FFMPEG_PATH})")
    _require(bool(shutil.which(FFPROBE_PATH)), "ffprobe", f"ffprobe not found (FFPROBE_PATH={FFPROBE_PATH})")

    # WhisperX availability
    if enable_transcribe:
        _require(_whisperx_available(), "whisperx", "WhisperX is not available (set ENVID_WHISPERX_BIN or ENVID_WHISPERX_PYTHON)")
        nlp_mode = "openrouter_llama"
        if nlp_mode in {"openrouter_llama", "openrouter", "llama"}:
            api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
            _require(bool(api_key), "openrouter", "OPENROUTER_API_KEY is not set (transcript correction)")

    # OpenRouter for synopses
    if enable_synopsis_generation:
        api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        _require(bool(api_key), "openrouter", "OPENROUTER_API_KEY is not set")

    # OpenRouter for scene-by-scene summaries
    if enable_scene_by_scene and _env_truthy(os.getenv("ENVID_SCENE_BY_SCENE_LLM"), default=False):
        api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        _require(bool(api_key), "openrouter", "OPENROUTER_API_KEY is not set (scene-by-scene)")

    # Local label detection service
    if enable_label_detection and use_local_label_detection_service:
        service_url = (local_label_detection_url_override or os.getenv("ENVID_METADATA_LOCAL_LABEL_DETECTION_URL") or "").strip().rstrip("/")
        if not service_url:
            _require(False, "local_label_detection", "ENVID_METADATA_LOCAL_LABEL_DETECTION_URL is not set")
        else:
            health = _check_service_health(f"{service_url}/health")
            _require(bool(health.get("ok")), "local_label_detection", f"local label detection unhealthy: {health.get('error') or health.get('raw')}")
    if enable_label_detection and requested_label_model in {"yolo_ultralytics", "yolo", "yolov8", "yolov11"}:
        _require(UltralyticsYOLO is not None, "yolo", "ultralytics is not installed")
        yolo_model = (os.getenv("ENVID_METADATA_YOLO_MODEL") or "yolov8n.pt").strip() or "yolov8n.pt"
        yolo_path = Path(yolo_model)
        if not yolo_path.exists():
            candidate = _repo_root() / yolo_model
            if candidate.exists():
                yolo_path = candidate
        _require(yolo_path.exists(), "yolo", f"YOLO model not found: {yolo_model}")

    # Local moderation service
    if enable_moderation and use_local_moderation:
        service_url_default = (os.getenv("ENVID_METADATA_LOCAL_MODERATION_URL") or "").strip()
        service_url_nsfwjs = (os.getenv("ENVID_METADATA_LOCAL_MODERATION_NSFWJS_URL") or "").strip()
        service_url = (local_moderation_url_override or (service_url_nsfwjs if requested_moderation_model == "nsfwjs" else "") or service_url_default).strip()
        if not service_url:
            if requested_moderation_model == "nudenet":
                _require(NudeClassifier is not None, "nudenet", f"nudenet is not installed: {_NUDENET_IMPORT_ERROR or 'missing dependency'}")
            else:
                _require(False, "local_moderation", "ENVID_METADATA_LOCAL_MODERATION_URL is not set")
        else:
            health = _check_service_health(f"{service_url.rstrip('/')}/health")
            _require(bool(health.get("ok")), "local_moderation", f"local moderation unhealthy: {health.get('error') or health.get('raw')}")
    if enable_moderation and (not use_local_moderation or allow_moderation_fallback) and want_any_vi:
        _require(gcp_video_intelligence is not None, "gcp_video_intelligence", "google-cloud-videointelligence is not installed")

    # Text-on-screen OCR
    if enable_text_on_screen and use_local_ocr:
        if requested_text_model == "paddleocr":
            _require(PaddleOCR is not None, "paddleocr", "paddleocr is not installed")
            if PaddleOCR is not None:
                try:
                    import paddle  # type: ignore
                    _require(True, "paddlepaddle", "paddlepaddle available")
                except Exception as exc:
                    _require(False, "paddlepaddle", f"paddlepaddle is not installed: {exc}")
        elif requested_text_model == "tesseract":
            _require(pytesseract is not None, "tesseract", "pytesseract is not installed")
            _require(bool(shutil.which("tesseract")), "tesseract", "tesseract binary is not available in PATH")
    if enable_text_on_screen and (not use_local_ocr) and want_any_vi:
        _require(gcp_video_intelligence is not None, "gcp_video_intelligence", "google-cloud-videointelligence is not installed")

    # Key scene sidecar (CLIP clustering optional, sidecar required for transnetv2 scenes)
    if enable_key_scene and requested_key_scene_model in {"transnetv2_clip_cluster", "pyscenedetect_clip_cluster", "clip_cluster"}:
        base = (os.getenv("ENVID_METADATA_LOCAL_KEYSCENE_URL") or "http://localhost:5085").strip().rstrip("/")
        health = _check_service_health(f"{base}/health")
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
        base_url = (os.getenv("ENVID_LIBRETRANSLATE_URL") or os.getenv("LIBRETRANSLATE_URL") or "").strip()
        if not base_url:
            _require(False, "libretranslate", "ENVID_LIBRETRANSLATE_URL is not set")
        else:
            try:
                langs = _libretranslate_languages_raw(base_url)
                _require(bool(langs), "libretranslate", "LibreTranslate /languages returned no data")
            except Exception as exc:
                _require(False, "libretranslate", f"LibreTranslate unavailable: {exc}")

    # GCP Language (locations)
    if enable_famous_locations and gcp_language is None:
        _require(False, "gcp_language", "google-cloud-language is not installed")

    return {"ok": not errors, "errors": errors, "warnings": warnings, "checks": checks}


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

    syn = video.get("synopses_by_age_group") or {}
    if syn:
        categories.setdefault("synopses_by_age_group", syn)
        combined["synopses_by_age_group"] = syn

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
    transcript_segments = video.get("transcript_segments")
    if not isinstance(transcript_segments, list):
        transcript_segments = []
    if transcript or transcript_segments:
        payload = {
            "text": transcript,
            "language_code": video.get("language_code"),
            "segments": transcript_segments,
            "meta": video.get("transcript_meta") or {},
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

    for key in ("synopses_by_age_group", "scene_by_scene_metadata", "key_scenes", "high_points"):
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

    cmd = [
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, float(at_seconds or 0.0)):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        "scale=224:-1",
        "-q:v",
        "3",
        "-y",
        str(out_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=max_seconds)
    return res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


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
    """Ranks scenes as 'key' using eventfulness + diversity."""

    if not scenes:
        return ([], [])
    top_k = max(1, min(int(top_k or 10), 50))

    labels = video_intelligence.get("labels") if isinstance(video_intelligence, dict) else None
    text = video_intelligence.get("text") if isinstance(video_intelligence, dict) else None
    moderation = video_intelligence.get("moderation") if isinstance(video_intelligence, dict) else None
    explicit_frames = moderation.get("explicit_frames") if isinstance(moderation, dict) else []

    def _transcript_chars_in_window(st: float, en: float) -> int:
        if not isinstance(transcript_segments, list) or not transcript_segments:
            return 0
        total = 0
        for seg in transcript_segments:
            if not isinstance(seg, dict):
                continue
            ss = _safe_float(seg.get("start"), 0.0)
            se = _safe_float(seg.get("end"), ss)
            if se <= ss:
                continue
            if _overlap_seconds(ss, se, st, en) <= 0:
                continue
            txt = seg.get("text")
            if isinstance(txt, str) and txt.strip():
                total += len(txt.strip())
        return total

    def _norm(x: float, scale: float) -> float:
        x = max(0.0, float(x))
        s = max(1e-6, float(scale))
        return 1.0 - exp(-x / s)

    scored: list[_SceneScore] = []
    prev_emb: str | None = None
    keyframes_dir = temp_dir / "keyframes"

    for i, sc in enumerate(scenes):
        st = _safe_float(sc.get("start"), 0.0)
        en = _safe_float(sc.get("end"), st)
        if en <= st:
            continue
        dur = max(0.01, en - st)

        label_hits = _count_overlapping_segments(labels, st, en)
        text_hits = _count_overlapping_segments(text, st, en)

        explicit_hits = 0
        if isinstance(explicit_frames, list) and explicit_frames:
            for f in explicit_frames:
                if not isinstance(f, dict):
                    continue
                t = _safe_float(f.get("time"), 0.0)
                if st <= t <= en:
                    explicit_hits += 1

        transcript_chars = float(_transcript_chars_in_window(st, en))
        transcript_density = transcript_chars / float(dur)

        mid = st + 0.5 * dur
        jpg = keyframes_dir / f"sc_{i:04d}.jpg"
        emb = None
        if _extract_keyframe_jpg(video_path=local_path, at_seconds=mid, out_path=jpg):
            emb = _ahash64_from_jpg(jpg)

        change = 0.0
        if emb and prev_emb:
            change = _hamming01(emb, prev_emb)
        prev_emb = emb or prev_emb

        score = 0.0
        score += 2.5 * _norm(label_hits, 8.0)
        score += 3.0 * _norm(text_hits, 4.0)
        score += 4.0 * _norm(explicit_hits, 6.0)
        score += 2.0 * _norm(transcript_density, 120.0)
        score += 2.0 * _norm(change, 0.35)
        score += 0.5 * _norm(dur, 6.0)

        reasons: list[str] = []
        if explicit_hits:
            reasons.append("explicit")
        if text_hits:
            reasons.append("text")
        if label_hits:
            reasons.append("labels")
        if transcript_chars > 0:
            reasons.append("dialogue")
        if change >= 0.25:
            reasons.append("scene_change")
        if not reasons:
            reasons = ["activity"]

        scored.append(_SceneScore(index=i, start=st, end=en, score=float(score), reasons=reasons, emb=emb))

    if not scored:
        return ([], [])

    scored.sort(key=lambda s: s.score, reverse=True)
    picked: list[_SceneScore] = []
    min_gap_seconds = float(_parse_int(os.getenv("ENVID_METADATA_KEY_SCENE_MIN_GAP_SECONDS"), default=8, min_value=0, max_value=600) or 8)

    # Optional: use CLIP clustering (sidecar) to encourage diversity.
    cluster_by_scene_index: dict[int, int] = {}
    if use_clip_cluster:
        pool_n = min(len(scored), max(30, top_k * 6))
        pool = scored[:pool_n]
        images_b64: List[str] = []
        for s in pool:
            try:
                jpg_path = (temp_dir / "keyframes") / f"sc_{int(s.index):04d}.jpg"
                if jpg_path.exists():
                    images_b64.append(base64.b64encode(jpg_path.read_bytes()).decode("ascii"))
                else:
                    images_b64.append("")
            except Exception:
                images_b64.append("")

        # Choose k for clustering based on top_k.
        k_clusters = max(2, min(int(top_k), len(images_b64)))
        cluster_ids = _local_keyscene_best_clip_cluster(images_b64=images_b64, k=k_clusters, required=False)
        if cluster_ids:
            for s, cid in zip(pool, cluster_ids):
                cluster_by_scene_index[int(s.index)] = int(cid)

    if use_clip_cluster and cluster_by_scene_index:
        # Greedy: pick best per cluster first, respecting min-gap.
        used_clusters: set[int] = set()
        remaining = list(scored)

        def _ok_gap(cand: _SceneScore) -> bool:
            if min_gap_seconds <= 0:
                return True
            for p in picked:
                if _overlap_seconds(cand.start, cand.end, p.start - min_gap_seconds, p.end + min_gap_seconds) > 0:
                    return False
            return True

        # Pass 1: prefer new clusters.
        next_remaining: list[_SceneScore] = []
        for cand in remaining:
            if len(picked) >= top_k:
                next_remaining.append(cand)
                continue
            if not _ok_gap(cand):
                next_remaining.append(cand)
                continue
            cid = cluster_by_scene_index.get(int(cand.index), -1)
            if cid >= 0 and cid in used_clusters:
                next_remaining.append(cand)
                continue
            picked.append(cand)
            if cid >= 0:
                used_clusters.add(cid)

        # Pass 2: fill remaining slots regardless of cluster.
        for cand in next_remaining:
            if len(picked) >= top_k:
                break
            if not _ok_gap(cand):
                continue
            picked.append(cand)

        # Last resort: if min-gap is too strict, fill from top scores.
        if len(picked) < top_k:
            for cand in scored:
                if len(picked) >= top_k:
                    break
                if cand in picked:
                    continue
                picked.append(cand)
    else:
        # Original behavior: MMR-like selection using aHash similarity.
        lambda_score = float(os.getenv("ENVID_METADATA_KEY_SCENE_MMR_LAMBDA") or 0.75)
        lambda_score = min(0.95, max(0.05, lambda_score))

        while scored and len(picked) < top_k:
            best_idx = None
            best_value = -1e9
            for idx, cand in enumerate(scored):
                too_close = False
                for p in picked:
                    if min_gap_seconds > 0 and _overlap_seconds(cand.start, cand.end, p.start - min_gap_seconds, p.end + min_gap_seconds) > 0:
                        too_close = True
                        break
                if too_close:
                    continue

                max_sim = 0.0
                if cand.emb and picked:
                    for p in picked:
                        if not p.emb:
                            continue
                        sim = 1.0 - _hamming01(cand.emb, p.emb)
                        if sim > max_sim:
                            max_sim = sim

                mmr = lambda_score * cand.score - (1.0 - lambda_score) * max_sim
                if mmr > best_value:
                    best_value = mmr
                    best_idx = idx

            if best_idx is None:
                picked.append(scored.pop(0))
            else:
                picked.append(scored.pop(best_idx))

    picked.sort(key=lambda s: (s.start, s.end))
    key_scenes = [
        {
            "scene_index": int(s.index),
            "start_seconds": float(s.start),
            "end_seconds": float(s.end),
            "score": float(s.score),
            "reasons": s.reasons,
            "source": scenes_source,
            **({"cluster_id": int(cluster_by_scene_index.get(int(s.index), -1))} if cluster_by_scene_index else {}),
        }
        for s in picked
    ]

    top_scored = sorted(picked, key=lambda s: s.score, reverse=True)
    high_points = [
        {
            "scene_index": int(s.index),
            "start_seconds": float(s.start),
            "end_seconds": float(s.end),
            "score": float(s.score),
            "reason": ", ".join(s.reasons) if s.reasons else "activity",
            "source": scenes_source,
            **({"cluster_id": int(cluster_by_scene_index.get(int(s.index), -1))} if cluster_by_scene_index else {}),
        }
        for s in top_scored[:3]
    ]
    return (key_scenes, high_points)


def _download_gcs_object_to_file(*, bucket: str, obj: str, dest_path: Path, job_id: str) -> None:
    _job_step_update(job_id, "upload_to_cloud_storage", status="running", percent=0, message="Downloading")
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
    """Run FFmpeg blackdetect and return parsed segments.

    Returns: [{"start": float, "end": float, "duration": float}, ...]
    """

    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    # `blackdetect` logs to stderr.
    vf = f"blackdetect=d={min_black_seconds}:pic_th={picture_black_threshold}:pix_th={pixel_black_threshold}"
    cmd = [
        FFMPEG_PATH,
        "-hide_banner",
        "-nostats",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-an",
        "-f",
        "null",
        "-",
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=max_seconds)
    text = (res.stderr or b"").decode("utf-8", errors="ignore")
    if res.returncode not in (0, 1):
        # FFmpeg sometimes returns 1 even when filter output is present; treat other codes as failure.
        raise RuntimeError(f"ffmpeg blackdetect failed (code {res.returncode})")

    segments: List[Dict[str, Any]] = []
    # Example: "black_start:0 black_end:2.08 black_duration:2.08"
    for m in re.finditer(r"black_start:(?P<st>[0-9.]+)\s+black_end:(?P<en>[0-9.]+)\s+black_duration:(?P<dur>[0-9.]+)", text):
        try:
            st = float(m.group("st"))
            en = float(m.group("en"))
            dur = float(m.group("dur"))
        except Exception:
            continue
        if dur <= 0:
            continue
        segments.append({"start": st, "end": en, "duration": dur})

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


def _pyscenedetect_list_scenes(*, video_path: Path, temp_dir: Path, max_seconds: int = 900) -> List[Dict[str, Any]]:
    """Run PySceneDetect CLI and return scenes as [{start,end,index}, ...]."""

    # Prefer the CLI installed into the same environment as this backend process.
    candidates: list[Path] = []
    try:
        scripts_dir = (sysconfig.get_path("scripts") or "").strip()
        if scripts_dir:
            candidates.append(Path(scripts_dir) / "scenedetect")
    except Exception:
        pass
    try:
        candidates.append(Path(os.getenv("VIRTUAL_ENV") or "").resolve() / "bin" / "scenedetect")
    except Exception:
        pass

    scenedetect_path: Path | None = next((p for p in candidates if str(p) and p.exists()), None)
    if scenedetect_path is None:
        found = shutil.which("scenedetect")
        if found:
            scenedetect_path = Path(found)

    if scenedetect_path is None or not scenedetect_path.exists():
        raise RuntimeError("PySceneDetect is not installed (missing `scenedetect` CLI). Install: `pip install scenedetect`")

    scenedetect_bin = str(scenedetect_path)

    out_dir = temp_dir / "scenedetect"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use detect-content (fast cuts) by default. Users can tweak sensitivity via env.
    threshold = float(os.getenv("ENVID_METADATA_SCENEDETECT_THRESHOLD") or 27.0)
    cmd = [
        scenedetect_bin,
        "-i",
        str(video_path),
        "-o",
        str(out_dir),
        "detect-content",
        "--threshold",
        str(threshold),
        "list-scenes",
        "-q",
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=max_seconds)
    if res.returncode != 0:
        err = (res.stderr or b"").decode("utf-8", errors="ignore")
        raise RuntimeError(f"scenedetect failed (code {res.returncode}): {err[:240]}")

    # Prefer reading the CSV output (file names vary by version).
    csv_files = sorted(out_dir.glob("*.csv"))
    if not csv_files:
        csv_files = sorted(out_dir.glob("*-Scenes.csv"))

    if csv_files:
        text = csv_files[0].read_text(encoding="utf-8", errors="ignore")
    else:
        # Even with -q, output may appear on stdout/stderr for some versions.
        text = ((res.stdout or b"") + b"\n" + (res.stderr or b"")).decode("utf-8", errors="ignore")
        if not text.strip():
            return []

    scenes: List[Dict[str, Any]] = []
    # CSV has headers; we parse any timecode-like tokens.
    # Typical CSV columns include Start Timecode / End Timecode.
    for line in text.splitlines():
        if not line.strip() or line.lower().startswith("scene") or "timecode" in line.lower():
            continue
        # Try CSV: split by comma.
        parts = [p.strip().strip('"') for p in line.split(",")]
        # Extract first two HH:MM:SS.mmm occurrences.
        times = []
        for p in parts:
            if re.match(r"^\d+:\d{2}:\d{2}(?:\.\d{1,6})?$", p):
                times.append(p)
        if len(times) >= 2:
            st = _parse_hhmmss_to_seconds(times[0])
            en = _parse_hhmmss_to_seconds(times[1])
        else:
            # Fallback: parse numeric seconds columns (common in PySceneDetect CSV output).
            floats: list[float] = []
            for p in parts:
                if re.match(r"^-?\d+\.\d+$", p):
                    try:
                        floats.append(float(p))
                    except Exception:
                        pass
            st = floats[0] if len(floats) >= 1 else 0.0
            en = floats[1] if len(floats) >= 2 else 0.0

        if en > st:
            scenes.append({"index": len(scenes), "start": st, "end": en})
    return scenes


def _transnetv2_list_scenes(*, video_path: Path, temp_dir: Path, max_seconds: int = 900) -> List[Dict[str, Any]]:
    """TransNetV2 scene/shot detection via optional sidecar.

    This backend runs on Python 3.14+; TransNetV2 commonly needs a different runtime.
    We support a Docker sidecar (see code/localKeySceneBest/) and call it here.
    """

    _ = temp_dir
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    base = (os.getenv("ENVID_METADATA_LOCAL_KEYSCENE_URL") or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("ENVID_METADATA_LOCAL_KEYSCENE_URL is not set (required for transnetv2)")

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

    base = (os.getenv("ENVID_METADATA_LOCAL_KEYSCENE_URL") or "").strip().rstrip("/")
    if not base:
        if required:
            raise RuntimeError("ENVID_METADATA_LOCAL_KEYSCENE_URL is not set (required for CLIP clustering)")
        return None
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
    # app.py is in code/envidMetadataGCP/app.py
    return Path(__file__).resolve().parents[2]


def _yolo_detect_labels_from_video(
    *,
    video_path: Path,
    model_name_or_path: str,
    frame_interval_seconds: float = 1.0,
    max_frames: int = 200,
    conf: float = 0.25,
    iou: float = 0.45,
) -> List[Dict[str, Any]]:
    """Run Ultralytics YOLO on sampled frames and return label segments.

    Returns: [{"label": str, "segments": [{"start": float, "end": float, "confidence": float}, ...]}]
    """

    if UltralyticsYOLO is None:
        raise RuntimeError("ultralytics is not installed")
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv is not installed") from exc

    model = UltralyticsYOLO(model_name_or_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("failed to open video")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = fps if fps > 0 else 30.0
    every_n = max(1, int(round(float(frame_interval_seconds) * fps)))

    label_hits: Dict[str, List[Tuple[float, float]]] = {}
    label_confs: Dict[str, List[float]] = {}

    frame_idx = 0
    sampled = 0
    try:
        while sampled < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % every_n != 0:
                frame_idx += 1
                continue

            t = float(frame_idx) / fps
            # Ultralytics expects BGR ndarray; OpenCV provides BGR.
            results = model.predict(frame, conf=conf, iou=iou, verbose=False)
            for r in results or []:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                cls = getattr(boxes, "cls", None)
                confs = getattr(boxes, "conf", None)
                if cls is None or confs is None:
                    continue
                try:
                    cls_list = cls.tolist()
                    conf_list = confs.tolist()
                except Exception:
                    continue

                names = getattr(r, "names", None) or {}
                for c, cf in zip(cls_list, conf_list, strict=False):
                    try:
                        ci = int(c)
                        lbl = str(names.get(ci) or ci)
                        cff = float(cf)
                    except Exception:
                        continue
                    label_hits.setdefault(lbl, []).append((t, t + float(frame_interval_seconds)))
                    label_confs.setdefault(lbl, []).append(cff)

            sampled += 1
            frame_idx += 1
    finally:
        cap.release()

    # Merge adjacent segments per label.
    merged: List[Dict[str, Any]] = []
    for lbl, segs in label_hits.items():
        segs_sorted = sorted(segs, key=lambda x: x[0])
        out: List[Dict[str, Any]] = []
        gap = float(frame_interval_seconds) * 1.5
        cur_st, cur_en = segs_sorted[0]
        for st, en in segs_sorted[1:]:
            if st <= cur_en + gap:
                cur_en = max(cur_en, en)
            else:
                out.append({"start": cur_st, "end": cur_en})
                cur_st, cur_en = st, en
        out.append({"start": cur_st, "end": cur_en})

        avg_conf = float(sum(label_confs.get(lbl) or [0.0]) / max(1, len(label_confs.get(lbl) or [])))
        for o in out:
            o["confidence"] = avg_conf
        merged.append({"label": lbl, "segments": out})

    merged.sort(key=lambda x: (len(x.get("segments") or []), str(x.get("label") or "")), reverse=True)
    return merged


_PADDLE_OCR_LOCK = threading.Lock()
_PADDLE_OCR_INSTANCE: Any = None


def _get_paddle_ocr(*, lang: str) -> Any:
    global _PADDLE_OCR_INSTANCE
    if PaddleOCR is None:
        return None
    # PaddleOCR init is expensive; keep a singleton.
    with _PADDLE_OCR_LOCK:
        if _PADDLE_OCR_INSTANCE is None:
            try:
                _PADDLE_OCR_INSTANCE = PaddleOCR(use_angle_cls=True, lang=lang)
            except ModuleNotFoundError as exc:
                # PaddleOCR depends on paddlepaddle. If it's missing, guide the user.
                if (getattr(exc, "name", None) or "") == "paddle" or "paddle" in str(exc).lower():
                    raise RuntimeError(
                        "paddleocr is installed but paddlepaddle is missing; install paddlepaddle or switch text_model to tesseract"
                    ) from exc
                raise
        return _PADDLE_OCR_INSTANCE


def _sample_video_frames(*, video_path: Path, interval_seconds: float, max_frames: int) -> List[Tuple[float, Any]]:
    """Return list of (timestamp_seconds, bgr_frame_ndarray)."""
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv is not installed") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("failed to open video")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = fps if fps > 0 else 30.0
    every_n = max(1, int(round(float(interval_seconds) * fps)))

    out: List[Tuple[float, Any]] = []
    frame_idx = 0
    try:
        while len(out) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % every_n == 0:
                t = float(frame_idx) / fps
                out.append((t, frame))
            frame_idx += 1
    finally:
        cap.release()
    return out


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


def _nudenet_moderation_explicit_frames_from_video(*, video_path: Path, interval_seconds: float, max_frames: int) -> List[Dict[str, Any]]:
    """Run NudeNet classifier over sampled frames and return explicit_frames.

    Returns: [{time: seconds, severity: str, unsafe: float, safe: float}, ...]
    """

    if NudeClassifier is None:
        # In this repo, the default venv is Python 3.14+, which currently has no tensorflow wheels.
        msg = "nudenet is not available"
        if _NUDENET_IMPORT_ERROR:
            msg += f" ({_NUDENET_IMPORT_ERROR})"
        msg += "; tensorflow is typically required and not available for Python 3.14; use moderation_model=auto for GCP fallback"
        raise RuntimeError(msg)

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv is not installed") from exc

    classifier = NudeClassifier()
    frames = _sample_video_frames(video_path=video_path, interval_seconds=interval_seconds, max_frames=max_frames)
    if not frames:
        return []

    tmp_dir = Path(tempfile.mkdtemp(prefix="envid_nudenet_frames_"))
    out: List[Dict[str, Any]] = []
    try:
        for idx, (t, bgr) in enumerate(frames):
            img_path = tmp_dir / f"frame_{idx:05d}.jpg"
            ok = cv2.imwrite(str(img_path), bgr)
            if not ok:
                continue
            res = classifier.classify(str(img_path))
            # NudeNet returns {path: {'safe': x, 'unsafe': y}} (best-effort).
            scores = res.get(str(img_path)) if isinstance(res, dict) else None
            if not isinstance(scores, dict):
                continue
            unsafe = float(scores.get("unsafe") or 0.0)
            safe = float(scores.get("safe") or 0.0)
            out.append(
                {
                    "time": float(t),
                    "severity": _severity_from_score(unsafe),
                    "unsafe": unsafe,
                    "safe": safe,
                }
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return out


def _local_moderation_service_explicit_frames_from_video(
    *,
    video_path: Path,
    interval_seconds: float,
    max_frames: int,
    service_base_url: str,
    timeout_seconds: int,
) -> List[Dict[str, Any]]:
    """Call an external local moderation service by sending sampled frames.

    Service API:
      POST {base}/moderate/frames  { frames: [{time, image_b64, image_mime}] }
      -> { explicit_frames: [...] }
    """

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv is not installed") from exc

    base = service_base_url.rstrip("/")
    url = f"{base}/moderate/frames"

    frames = _sample_video_frames(video_path=video_path, interval_seconds=interval_seconds, max_frames=max_frames)
    if not frames:
        return []

    payload_frames: List[Dict[str, Any]] = []
    for t, bgr in frames:
        ok, enc = cv2.imencode(".jpg", bgr)
        if not ok:
            continue
        payload_frames.append(
            {
                "time": float(t),
                "image_b64": base64.b64encode(enc.tobytes()).decode("ascii"),
                "image_mime": "image/jpeg",
            }
        )

    data = json.dumps({"frames": payload_frames}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""
        raise RuntimeError(f"local moderation service HTTP {exc.code}: {body[:200]}") from exc
    except Exception as exc:
        raise RuntimeError(f"local moderation service request failed: {exc}") from exc

    explicit_frames = resp.get("explicit_frames")
    if not isinstance(explicit_frames, list):
        raise RuntimeError("local moderation service returned invalid explicit_frames")
    # Normalize to expected shape.
    include_provider_details = _env_truthy(os.getenv("ENVID_METADATA_INCLUDE_PROVIDER_DETAILS"), default=False)
    out: List[Dict[str, Any]] = []
    for f in explicit_frames:
        if not isinstance(f, dict):
            continue
        provider_likelihood = str(f.get("likelihood") or f.get("provider_likelihood") or "")
        severity = str(f.get("severity") or "")
        if not severity:
            severity = _severity_from_likelihood(provider_likelihood)
        out.append(
            {
                "time": float(f.get("time") or 0.0),
                "severity": severity,
                **({"provider_likelihood": provider_likelihood} if include_provider_details and provider_likelihood else {}),
                **({"unsafe": float(f.get("unsafe") or 0.0)} if f.get("unsafe") is not None else {}),
                **({"safe": float(f.get("safe") or 0.0)} if f.get("safe") is not None else {}),
            }
        )
    return out


def _local_label_detection_service_labels_from_video(
    *,
    video_path: Path,
    interval_seconds: float,
    max_frames: int,
    service_base_url: str,
    model: str,
    timeout_seconds: int,
) -> List[Dict[str, Any]]:
    """Call an external local label-detection service by sending sampled frames.

    Service API:
      POST {base}/detect/frames  { model, frame_len, frames:[{time, image_b64, image_mime}] }
      -> { labels: [{label, segments:[{start,end,confidence}]}] }
    """

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv is not installed") from exc

    base = service_base_url.rstrip("/")
    url = f"{base}/detect/frames"

    frames = _sample_video_frames(video_path=video_path, interval_seconds=interval_seconds, max_frames=max_frames)
    if not frames:
        return []

    payload_frames: List[Dict[str, Any]] = []
    for t, bgr in frames:
        ok, enc = cv2.imencode(".jpg", bgr)
        if not ok:
            continue
        payload_frames.append(
            {
                "time": float(t),
                "image_b64": base64.b64encode(enc.tobytes()).decode("ascii"),
                "image_mime": "image/jpeg",
            }
        )

    data = json.dumps({"model": model, "frame_len": float(interval_seconds), "frames": payload_frames}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""
        raise RuntimeError(f"local label detection service HTTP {exc.code}: {body[:200]}") from exc
    except Exception as exc:
        raise RuntimeError(f"local label detection service request failed: {exc}") from exc

    labels = resp.get("labels")
    if not isinstance(labels, list):
        raise RuntimeError("local label detection service returned invalid labels")
    out: List[Dict[str, Any]] = []
    for it in labels:
        if not isinstance(it, dict):
            continue
        lbl = (it.get("label") or it.get("name") or "").strip() if isinstance(it.get("label") or it.get("name"), str) else ""
        segs_in = it.get("segments")
        if not lbl or not isinstance(segs_in, list):
            continue
        segs: List[Dict[str, Any]] = []
        for s in segs_in:
            if not isinstance(s, dict):
                continue
            segs.append(
                {
                    "start": float(s.get("start") or 0.0),
                    "end": float(s.get("end") or 0.0),
                    "confidence": float(s.get("confidence") or 0.0),
                }
            )
        if segs:
            out.append({"label": lbl, "segments": segs})
    return out


def _aggregate_text_segments(
    *,
    hits: List[Tuple[str, float, float, float]],
    max_items: int = 250,
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
    for txt, segs in by_text.items():
        segs.sort(key=lambda x: (float(x.get("start") or 0.0), float(x.get("end") or 0.0)))
        out.append({"text": txt, "segments": segs})
    out.sort(key=lambda x: (len(x.get("segments") or []), len(str(x.get("text") or ""))), reverse=True)
    return out[:max_items]


@app.get("/local-label-detection/health")
def local_label_detection_health():
    """Proxy health for the external local label-detection service.

    This lets the frontend show actionable readiness/errors (e.g. MMCV ops missing).
    """

    service_url = (os.getenv("ENVID_METADATA_LOCAL_LABEL_DETECTION_URL") or "").strip().rstrip("/")
    if not service_url:
        return jsonify(
            {
                "ok": False,
                "configured": False,
                "service_url": "",
                "error": "ENVID_METADATA_LOCAL_LABEL_DETECTION_URL is not set",
            }
        )

    url = f"{service_url}/health"
    timeout_seconds = float(os.getenv("ENVID_METADATA_LOCAL_LABEL_HEALTH_TIMEOUT_SECONDS") or 4.0)
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as r:
            body = r.read().decode("utf-8")
        data = json.loads(body)
        if not isinstance(data, dict):
            raise RuntimeError("invalid JSON")

        return jsonify(
            {
                "ok": True,
                "configured": True,
                "service_url": service_url,
                "health": data,
            }
        )
    except Exception as exc:
        return jsonify(
            {
                "ok": False,
                "configured": True,
                "service_url": service_url,
                "error": str(exc)[:200],
            }
        )


@app.get("/local-ocr-paddle/health")
def local_ocr_paddle_health():
    """Proxy health for the external PaddleOCR service.

    This keeps PaddleOCR in a separate runtime (Python 3.11) while the main backend
    runs on Python 3.14+.
    """

    service_url = (os.getenv("ENVID_METADATA_LOCAL_OCR_PADDLE_URL") or "").strip().rstrip("/")
    if not service_url:
        return jsonify(
            {
                "ok": False,
                "configured": False,
                "service_url": "",
                "error": "ENVID_METADATA_LOCAL_OCR_PADDLE_URL is not set",
            }
        )

    url = f"{service_url}/health"
    timeout_seconds = float(os.getenv("ENVID_METADATA_LOCAL_OCR_PADDLE_HEALTH_TIMEOUT_SECONDS") or 4.0)
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as r:
            body = r.read().decode("utf-8")
        data = json.loads(body)
        if not isinstance(data, dict):
            raise RuntimeError("invalid JSON")

        return jsonify(
            {
                "ok": True,
                "configured": True,
                "service_url": service_url,
                "health": data,
            }
        )
    except Exception as exc:
        return jsonify(
            {
                "ok": False,
                "configured": True,
                "service_url": service_url,
                "error": str(exc)[:200],
            }
        )


def _local_ocr_paddle_service_text_from_video(
    *,
    video_path: Path,
    interval_seconds: float,
    max_frames: int,
    service_base_url: str,
    lang: str,
    timeout_seconds: int,
) -> List[Dict[str, Any]]:
    """Call external PaddleOCR service by sending sampled frames.

    Service API:
      POST {base}/ocr/frames { lang, frame_len, frames:[{time,image_b64,image_mime}] }
      -> { text: [{text, segments:[{start,end,confidence}]}] }
    """

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv is not installed") from exc

    base = service_base_url.rstrip("/")
    url = f"{base}/ocr/frames"

    frames = _sample_video_frames(video_path=video_path, interval_seconds=interval_seconds, max_frames=max_frames)
    if not frames:
        return []

    payload_frames: List[Dict[str, Any]] = []
    for t, bgr in frames:
        ok, enc = cv2.imencode(".jpg", bgr)
        if not ok:
            continue
        payload_frames.append(
            {
                "time": float(t),
                "image_b64": base64.b64encode(enc.tobytes()).decode("ascii"),
                "image_mime": "image/jpeg",
            }
        )

    data = json.dumps({"lang": str(lang or "en"), "frame_len": float(interval_seconds), "frames": payload_frames}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""
        raise RuntimeError(f"local PaddleOCR service HTTP {exc.code}: {body[:200]}") from exc
    except Exception as exc:
        raise RuntimeError(f"local PaddleOCR service request failed: {exc}") from exc

    texts = resp.get("text")
    if not isinstance(texts, list):
        raise RuntimeError("local PaddleOCR service returned invalid text")

    out: List[Dict[str, Any]] = []
    for it in texts:
        if not isinstance(it, dict):
            continue
        txt = (it.get("text") or "").strip() if isinstance(it.get("text"), str) else ""
        segs_in = it.get("segments")
        if not txt or not isinstance(segs_in, list):
            continue
        segs: List[Dict[str, Any]] = []
        for s in segs_in:
            if not isinstance(s, dict):
                continue
            segs.append(
                {
                    "start": float(s.get("start") or 0.0),
                    "end": float(s.get("end") or 0.0),
                    "confidence": float(s.get("confidence") or 0.0),
                }
            )
        if segs:
            out.append({"text": txt, "segments": segs})
    return out


def _paddleocr_text_from_video(
    *,
    video_path: Path,
    interval_seconds: float,
    max_frames: int,
    lang: str,
) -> List[Dict[str, Any]]:
    ocr = _get_paddle_ocr(lang=lang)
    if ocr is None:
        raise RuntimeError("paddleocr is not installed")

    hits: List[Tuple[str, float, float, float]] = []
    frames = _sample_video_frames(video_path=video_path, interval_seconds=interval_seconds, max_frames=max_frames)
    for t, frame in frames:
        # PaddleOCR accepts ndarray BGR.
        try:
            res = ocr.ocr(frame, cls=True)
        except Exception:
            continue

        # Res may be a list of lines; each line: [box, (text, conf)]
        for line in res or []:
            try:
                txt = str(line[1][0] or "").strip()
                conf = float(line[1][1] or 0.0)
            except Exception:
                continue
            if not txt:
                continue
            hits.append((txt, float(t), float(t + interval_seconds), conf))
    return _aggregate_text_segments(hits=hits)


def _tesseract_text_from_video(
    *,
    video_path: Path,
    interval_seconds: float,
    max_frames: int,
) -> List[Dict[str, Any]]:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed")
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv is not installed") from exc

    hits: List[Tuple[str, float, float, float]] = []
    tesseract_lang = (os.getenv("ENVID_METADATA_OCR_TESSERACT_LANG") or "eng+hin").strip() or "eng"
    tesseract_psm = _parse_int(os.getenv("ENVID_METADATA_OCR_TESSERACT_PSM"), default=6, min_value=3, max_value=13)
    tesseract_oem = _parse_int(os.getenv("ENVID_METADATA_OCR_TESSERACT_OEM"), default=1, min_value=0, max_value=3)
    upscale = _safe_float(os.getenv("ENVID_METADATA_OCR_TESSERACT_UPSCALE"), 1.5)
    if upscale < 1.0:
        upscale = 1.0
    tesseract_config = f"--oem {tesseract_oem} --psm {tesseract_psm}"
    frames = _sample_video_frames(video_path=video_path, interval_seconds=interval_seconds, max_frames=max_frames)
    for t, frame in frames:
        try:
            # Preprocess: grayscale + denoise + upsample + adaptive threshold.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            denoised = cv2.bilateralFilter(gray, 5, 75, 75)
            if upscale > 1.01:
                denoised = cv2.resize(
                    denoised,
                    None,
                    fx=upscale,
                    fy=upscale,
                    interpolation=cv2.INTER_CUBIC,
                )
            thresh = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                9,
            )
            rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        except Exception:
            continue
        try:
            data = pytesseract.image_to_data(
                rgb,
                lang=tesseract_lang,
                config=tesseract_config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            continue

        n = len(data.get("text") or [])
        for i in range(n):
            txt = str((data.get("text") or [""])[i] or "").strip()
            if not txt:
                continue
            try:
                conf = float((data.get("conf") or [0])[i])
            except Exception:
                conf = 0.0
            # Normalize tesseract conf range.
            conf = max(0.0, min(100.0, conf)) / 100.0
            hits.append((txt, float(t), float(t + interval_seconds), conf))
    return _aggregate_text_segments(hits=hits)


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

    combined_obj = f"{base}/combined.json"
    bkt.blob(combined_obj).upload_from_string(
        json.dumps(payload.get("combined") or {}, indent=2, ensure_ascii=False),
        content_type="application/json; charset=utf-8",
    )
    out["combined"] = {"object": combined_obj, "uri": f"gs://{artifacts_bucket}/{combined_obj}"}

    cats = payload.get("categories") if isinstance(payload.get("categories"), dict) else {}
    for name, cat_payload in (cats or {}).items():
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name or "category").strip()) or "category"
        obj = f"{base}/categories/{safe}.json"
        bkt.blob(obj).upload_from_string(
            json.dumps(cat_payload, indent=2, ensure_ascii=False),
            content_type="application/json; charset=utf-8",
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
        obj = f"{base}/subtitles/{lang}.{fmt}"
        bkt.blob(obj).upload_from_filename(str(local_path), content_type=content_type)
        out["subtitles"][f"{lang}.{fmt}"] = {"object": obj, "uri": f"gs://{artifacts_bucket}/{obj}"}

    return out


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
    temp_dir = Path(tempfile.mkdtemp(prefix=f"envid_metadata_gcs_{job_id}_"))
    gcs_uri = f"gs://{gcs_bucket}/{gcs_object}"
    critical_failures: list[str] = []
    try:
        _job_update(job_id, status="processing", progress=1, message="Processing", gcs_video_uri=gcs_uri)

        sel: Dict[str, Any] = task_selection if isinstance(task_selection, dict) else {}

        def _sel_model(key: str, *, default: str = "auto") -> str:
            raw = sel.get(key)
            if raw is None:
                return default
            try:
                s = str(raw).strip()
            except Exception:
                return default
            return s or default

        def _sel_enabled(key: str, *, default: bool = True) -> bool:
            if key in sel:
                try:
                    return bool(sel.get(key))
                except Exception:
                    return default
            return default

        enable_label_detection = _sel_enabled("enable_label_detection", default=False)
        enable_text_on_screen = _sel_enabled("enable_text", default=True)
        enable_moderation = _sel_enabled("enable_moderation", default=True)
        enable_transcribe = _sel_enabled("enable_transcribe", default=True)
        enable_famous_locations = _sel_enabled("enable_famous_location_detection", default=True)
        enable_scene_by_scene = _sel_enabled("enable_scene_by_scene_metadata", default=True)
        enable_key_scene = _sel_enabled("enable_key_scene_detection", default=True)
        # New UI ties high-point to key-scene. Preserve backward compatibility
        # with older clients which send `enable_high_point` explicitly.
        enable_high_point = _sel_enabled("enable_high_point", default=enable_key_scene)
        enable_synopsis_generation = _sel_enabled("enable_synopsis_generation", default=True)
        enable_opening_closing = _sel_enabled("enable_opening_closing_credit_detection", default=False)
        enable_celebrity_detection = _sel_enabled("enable_celebrity_detection", default=True)
        enable_celebrity_bio_image = _sel_enabled("enable_celebrity_bio_image", default=True)

        # Enforce fixed product policy: no Google usage except label detection.
        enable_label_detection = True
        enable_famous_locations = False
        enable_opening_closing = False

        # Scene-by-scene metadata is designed to be "complete" by default.
        # If the client enables scene-by-scene (or relies on the default enable), we automatically
        # enable label detection unless the client explicitly provided enable_label_detection.
        # This ensures no model selection is required to get useful scene-by-scene output.
        if enable_scene_by_scene and "enable_label_detection" not in sel:
            enable_label_detection = True

        # When scene-by-scene is enabled and label detection is enabled, default to Google Video
        # Intelligence labels so clients don't need to pick a label model.
        label_model_default = "gcp_video_intelligence" if (enable_scene_by_scene and enable_label_detection) else ""

        requested_models: Dict[str, str] = {
            "celebrity_detection_model": _sel_model("celebrity_detection_model"),
            "celebrity_bio_image_model": _sel_model("celebrity_bio_image_model"),
            "label_detection_model": _sel_model("label_detection_model", default=label_model_default),
            "famous_location_detection_model": _sel_model("famous_location_detection_model"),
            "moderation_model": _sel_model("moderation_model"),
            "text_model": _sel_model("text_model"),
            "transcribe_model": _sel_model("transcribe_model"),
            "key_scene_detection_model": _sel_model("key_scene_detection_model"),
            "high_point_model": _sel_model("high_point_model"),
            "scene_by_scene_metadata_model": _sel_model("scene_by_scene_metadata_model"),
            "synopsis_generation_model": _sel_model("synopsis_generation_model"),
            "opening_closing_credit_detection_model": _sel_model("opening_closing_credit_detection_model"),
        }

        # Force fixed models per requirement.
        requested_models["label_detection_model"] = "gcp_video_intelligence"
        requested_models["moderation_model"] = "nudenet"
        requested_models["text_model"] = "tesseract"
        requested_models["key_scene_detection_model"] = "transnetv2_clip_cluster"
        requested_models["synopsis_generation_model"] = "openrouter_llama"

        effective_models: Dict[str, str] = {}

        _job_update(
            job_id,
            task_selection=sel,
            task_selection_requested_models=requested_models,
            task_selection_enabled={
                "celebrity_detection": enable_celebrity_detection,
                "celebrity_bio_image": enable_celebrity_bio_image,
                "label_detection": enable_label_detection,
                "famous_location_detection": enable_famous_locations,
                "moderation": enable_moderation,
                "text_on_screen": enable_text_on_screen,
                "transcribe": enable_transcribe,
                "scene_by_scene_metadata": enable_scene_by_scene,
                "key_scene_detection": enable_key_scene,
                "high_point": enable_high_point,
                "synopsis_generation": enable_synopsis_generation,
                "opening_closing_credit_detection": enable_opening_closing,
            },
        )

        video_intelligence: Dict[str, Any] = {}
        local_labels: List[Dict[str, Any]] = []

        transcript = ""
        transcript_language_code = ""
        languages_detected: List[str] = []
        transcript_words: List[Dict[str, Any]] = []
        transcript_segments: List[Dict[str, Any]] = []
        transcript_meta: Dict[str, Any] = {}
        local_text: List[Dict[str, Any]] = []
        precomputed_scenes: List[Dict[str, Any]] = []
        precomputed_scenes_source = "none"
        local_moderation_frames: List[Dict[str, Any]] | None = None
        local_moderation_source: str | None = None

        enable_vi = _env_truthy(os.getenv("ENVID_METADATA_ENABLE_VIDEO_INTELLIGENCE"), default=True)

        # Transcription strategy: WhisperX only.
        requested_transcribe_model = (requested_models.get("transcribe_model") or "whisperx").strip().lower() or "whisperx"
        transcribe_effective_mode = "whisperx"
        effective_models["transcribe"] = transcribe_effective_mode

        # Text-on-screen model selection (fixed to local Tesseract).
        requested_text_model = "tesseract"
        use_local_ocr = True

        # Moderation model selection (fixed to local NudeNet; no GCP fallback).
        requested_moderation_model = "nudenet"
        use_local_moderation = True
        allow_moderation_fallback = False
        local_moderation_url_override = _sel_model("local_moderation_url", default="").strip()

        # Local label detection service URL override (optional).
        local_label_detection_url_override = _sel_model("local_label_detection_url", default="").strip()

        # Default label detection mode to frame-wise for Google Video Intelligence.
        # Note: VI label detection is only used when explicitly selected.
        vi_label_mode = (os.getenv("ENVID_METADATA_GCP_VI_LABEL_MODE") or "frame").strip().lower() or "frame"
        if vi_label_mode not in {"segment", "shot", "frame"}:
            vi_label_mode = "frame"

        # Label detection engine selection (hard-coded to GCP Video Intelligence).
        requested_label_model_raw = (requested_models.get("label_detection_model") or "").strip()
        requested_label_model = "gcp_video_intelligence"
        label_engine = "gcp_video_intelligence"
        use_yolo_labels = False
        use_local_label_detection_service = False
        use_vi_label_detection = True

        # Key scene model selection (fixed to TransNetV2 + CLIP).
        # NOTE: For key scenes/high points we avoid any implicit fallback between engines.
        key_scene_step_finalized = False
        requested_key_scene_model_raw = "transnetv2_clip_cluster" if enable_key_scene else requested_models.get("key_scene_detection_model")
        requested_key_scene_model = (requested_key_scene_model_raw or "").strip().lower()
        if requested_key_scene_model in {"", "auto"}:
            requested_key_scene_model = "pyscenedetect_clip_cluster"
        elif requested_key_scene_model in {"gcp_video_intelligence"}:
            requested_key_scene_model = "gcp_video_intelligence"
        elif requested_key_scene_model in {"gcp_vi", "video_intelligence", "vi", "google_video_intelligence"}:
            requested_key_scene_model = "gcp_video_intelligence"
        elif requested_key_scene_model in {"best_combo", "transnetv2_clip_cluster", "transnetv2+clip", "transnetv2_clip"}:
            # Back-compat alias: old `best_combo` now means TransNetV2 + CLIP (strict; no fallback).
            requested_key_scene_model = "transnetv2_clip_cluster"
        elif requested_key_scene_model in {"pyscenedetect_clip_cluster", "pyscenedetect+clip", "pyscenedetect_clip"}:
            requested_key_scene_model = "pyscenedetect_clip_cluster"
        elif requested_key_scene_model in {"clip_cluster", "clip"}:
            requested_key_scene_model = "clip_cluster"
        elif requested_key_scene_model in {"pyscenedetect", "transnetv2"}:
            # Standalone models are intentionally not supported/exposed.
            # Users must choose one of the strict combos (or VI shots).
            requested_key_scene_model = "unsupported"
        else:
            # Unknown value: do not guess; require an explicit supported value.
            requested_key_scene_model = "unsupported"

        allowed_key_scene_models = {
            "gcp_video_intelligence",
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
                    "key_scene_detection_model must be one of: gcp_video_intelligence, transnetv2_clip_cluster, "
                    "pyscenedetect_clip_cluster (no fallback)."
                ),
            )
            # Disable key scene + high point so the rest of the pipeline can continue without fallback.
            key_scene_step_finalized = True
            enable_key_scene = False
            enable_high_point = False

        use_transnetv2_for_scenes = bool(enable_key_scene and requested_key_scene_model in {"transnetv2_clip_cluster"})
        use_pyscenedetect_for_scenes = bool(
            (enable_key_scene and requested_key_scene_model in {"pyscenedetect_clip_cluster"})
            or ((requested_models.get("opening_closing_credit_detection_model") or "").strip().lower() == "pyscenedetect")
        )
        use_clip_cluster_for_key_scenes = bool(
            enable_key_scene and requested_key_scene_model in {"transnetv2_clip_cluster", "pyscenedetect_clip_cluster", "clip_cluster"}
        )

        # Sidecar availability is validated in precheck; no auto-disable here.

        want_shots = bool(enable_scene_by_scene or enable_key_scene or enable_high_point)
        # For scene-by-scene, allow VI shots as the default scene source when no local scene engine is selected.
        want_vi_shots = bool(
            want_shots
            and (requested_key_scene_model in {"gcp_video_intelligence", "clip_cluster"} or enable_scene_by_scene)
            and (not use_pyscenedetect_for_scenes)
            and (not use_transnetv2_for_scenes)
        )

        # Only use Video Intelligence TEXT_DETECTION when local OCR is not selected.
        want_any_vi = bool(
            (enable_label_detection and use_vi_label_detection)
            or (enable_text_on_screen and not use_local_ocr)
            or (enable_moderation and (not use_local_moderation or allow_moderation_fallback))
            or want_vi_shots
        )

        _job_step_update(job_id, "precheck_models", status="running", percent=0, message="Checking model availability")
        precheck = _precheck_models(
            enable_transcribe=enable_transcribe,
            enable_synopsis_generation=enable_synopsis_generation,
            enable_label_detection=enable_label_detection,
            enable_moderation=enable_moderation,
            enable_text_on_screen=enable_text_on_screen,
            enable_key_scene=enable_key_scene,
            enable_scene_by_scene=enable_scene_by_scene,
            enable_famous_locations=enable_famous_locations,
            requested_key_scene_model=requested_key_scene_model,
            requested_label_model=requested_label_model,
            requested_text_model=requested_text_model,
            requested_moderation_model=requested_moderation_model,
            use_local_label_detection_service=use_local_label_detection_service,
            use_local_moderation=use_local_moderation,
            allow_moderation_fallback=allow_moderation_fallback,
            use_local_ocr=use_local_ocr,
            want_any_vi=want_any_vi,
            local_label_detection_url_override=local_label_detection_url_override,
            local_moderation_url_override=local_moderation_url_override,
        )
        _job_update(job_id, precheck=precheck)
        if not precheck.get("ok"):
            msg = "; ".join(precheck.get("errors") or [])[:240] or "Precheck failed"
            _job_step_update(job_id, "precheck_models", status="failed", percent=100, message=msg)
            _job_step_update(job_id, "overall", status="failed", percent=100, message="Precheck failed")
            _job_update(job_id, status="failed", progress=100, message="Precheck failed")
            return
        warn_msg = "; ".join(precheck.get("warnings") or [])
        _job_step_update(job_id, "precheck_models", status="completed", percent=100, message=(warn_msg[:240] if warn_msg else "OK"))

        checks = precheck.get("checks") if isinstance(precheck, dict) else {}
        clip_ok = True
        if isinstance(checks, dict):
            if "clip_model" in checks and not checks.get("clip_model", {}).get("ok", True):
                clip_ok = False
            if "keyscene_sidecar" in checks and not checks.get("keyscene_sidecar", {}).get("ok", True):
                clip_ok = False
        if not clip_ok:
            use_clip_cluster_for_key_scenes = False

        ext = Path(gcs_object).suffix or ".mp4"
        local_path = temp_dir / f"video{ext}"
        _download_gcs_object_to_file(bucket=gcs_bucket, obj=gcs_object, dest_path=local_path, job_id=job_id)

        _job_step_update(job_id, "technical_metadata", status="running", percent=0, message="Probing")
        technical_ffprobe = _probe_technical_metadata(local_path)
        duration_seconds = _video_duration_seconds_from_ffprobe(technical_ffprobe)
        _job_step_update(job_id, "technical_metadata", status="completed", percent=100, message="Completed")

        # Transcode normalize (best-effort). Keep it lightweight and optional.
        normalize_enabled = _env_truthy(os.getenv("ENVID_METADATA_ENABLE_TRANSCODE_NORMALIZE"), default=True)
        normalized_path = temp_dir / f"video_normalized{ext}"
        if normalize_enabled:
            try:
                ffmpeg_gpu_enabled = _env_truthy(os.getenv("ENVID_METADATA_ENABLE_FFMPEG_GPU"), default=True)
                ffmpeg_gpu_available = ffmpeg_gpu_enabled and (
                    shutil.which("nvidia-smi") is not None or Path("/usr/bin/nvidia-smi").exists()
                )
                progress_span = int(os.getenv("ENVID_METADATA_TRANSCODE_PROGRESS_SPAN", "4"))
                progress_base = int(os.getenv("ENVID_METADATA_TRANSCODE_PROGRESS_BASE", "6"))
                progress_span = max(1, min(20, progress_span))
                progress_base = max(0, min(90, progress_base))
                transcode_label = "Running (GPU)" if ffmpeg_gpu_available else "Running (CPU)"
                _job_step_update(
                    job_id,
                    "transcode_normalize",
                    status="running",
                    percent=1,
                    message=transcode_label,
                )
                _job_update(job_id, progress=progress_base, message="Transcode normalize")

                def _parse_out_time(value: str) -> float | None:
                    value = value.strip()
                    if not value:
                        return None
                    if value.isdigit():
                        return float(value)
                    if ":" in value:
                        try:
                            parts = value.split(":")
                            seconds = float(parts[-1])
                            minutes = int(parts[-2]) if len(parts) > 1 else 0
                            hours = int(parts[-3]) if len(parts) > 2 else 0
                            return hours * 3600 + minutes * 60 + seconds
                        except Exception:
                            return None
                    return None

                def _run_ffmpeg_with_progress(cmd: list[str]) -> None:
                    if not duration_seconds:
                        subprocess.run(cmd, capture_output=True, check=True)
                        return

                    cmd_with_progress = [*cmd[:-1], "-progress", "pipe:1", "-nostats", cmd[-1]]
                    last_pct = -1
                    pct_step = 5
                    proc = subprocess.Popen(
                        cmd_with_progress,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        out_ms = None
                        out_seconds = None
                        if line.startswith("out_time_ms="):
                            try:
                                out_ms = int(line.split("=", 1)[1].strip())
                            except ValueError:
                                out_ms = None
                        elif line.startswith("out_time_us="):
                            try:
                                out_ms = int(line.split("=", 1)[1].strip()) / 1000
                            except ValueError:
                                out_ms = None
                        elif line.startswith("out_time="):
                            out_seconds = _parse_out_time(line.split("=", 1)[1])

                        if out_ms is None and out_seconds is None:
                            continue
                        if out_ms is not None:
                            pct = int(min(99, max(1, (out_ms / (duration_seconds * 1_000_000)) * 100)))
                        else:
                            pct = int(min(99, max(1, (out_seconds / duration_seconds) * 100)))
                        if pct >= last_pct + pct_step or pct in (1, 99):
                            _job_step_update(
                                job_id,
                                "transcode_normalize",
                                status="running",
                                percent=pct,
                                message=transcode_label,
                            )
                            overall_pct = min(progress_base + progress_span, progress_base + int((pct / 100) * progress_span))
                            _job_update(job_id, progress=overall_pct, message="Transcode normalize")
                            last_pct = pct
                    return_code = proc.wait()
                    if return_code != 0:
                        raise subprocess.CalledProcessError(return_code, cmd_with_progress)

                # Target ~1.5mbps video bitrate; keep audio modest.
                cmd_cpu = [
                    FFMPEG_PATH,
                    "-y",
                    "-i",
                    str(local_path),
                    "-c:v",
                    "libx264",
                    "-b:v",
                    "1500k",
                    "-maxrate",
                    "1500k",
                    "-bufsize",
                    "3000k",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-movflags",
                    "+faststart",
                    str(normalized_path),
                ]
                cmd_gpu = [
                    FFMPEG_PATH,
                    "-y",
                    "-i",
                    str(local_path),
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p4",
                    "-b:v",
                    "1500k",
                    "-maxrate",
                    "1500k",
                    "-bufsize",
                    "3000k",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-movflags",
                    "+faststart",
                    str(normalized_path),
                ]

                cmd = cmd_gpu if ffmpeg_gpu_available else cmd_cpu
                app.logger.info("Transcode normalize using %s", "GPU" if ffmpeg_gpu_available else "CPU")
                _run_ffmpeg_with_progress(cmd)
                if normalized_path.exists() and normalized_path.stat().st_size > 0:
                    local_path = normalized_path
                _job_step_update(job_id, "transcode_normalize", status="completed", percent=100, message="Completed")
            except Exception as exc:
                if ffmpeg_gpu_available:
                    app.logger.warning("Transcode normalize (GPU) failed; retrying CPU: %s", exc)
                    try:
                        _run_ffmpeg_with_progress(cmd_cpu)
                        if normalized_path.exists() and normalized_path.stat().st_size > 0:
                            local_path = normalized_path
                        _job_step_update(job_id, "transcode_normalize", status="completed", percent=100, message="Completed")
                    except Exception as exc2:
                        app.logger.warning("Transcode normalize failed: %s", exc2)
                        _job_step_update(job_id, "transcode_normalize", status="skipped", percent=100, message="Failed; using original")
                else:
                    app.logger.warning("Transcode normalize failed: %s", exc)
                    _job_step_update(job_id, "transcode_normalize", status="skipped", percent=100, message="Failed; using original")
        else:
            _job_step_update(job_id, "transcode_normalize", status="skipped", percent=100, message="Disabled")

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
        effective_models["label_detection"] = label_engine if enable_label_detection else "disabled"
        effective_models["label_detection_model_requested"] = requested_label_model_raw
        effective_models["label_detection_model_normalized"] = requested_label_model
        effective_models["label_detection_mode"] = vi_label_mode
        effective_models["text_on_screen"] = requested_text_model if (enable_text_on_screen and use_local_ocr) else ("gcp_video_intelligence" if enable_vi and gcp_video_intelligence is not None else "disabled")
        effective_models["moderation"] = requested_moderation_model if (enable_moderation and use_local_moderation) else ("gcp_video_intelligence" if enable_vi and gcp_video_intelligence is not None else "disabled")
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
        vi_min_text_conf = float(os.getenv("ENVID_METADATA_GCP_VI_MIN_TEXT_CONFIDENCE") or 0.0)

        vi_max_labels = _parse_int(os.getenv("ENVID_METADATA_GCP_VI_MAX_LABELS"), default=80, min_value=1, max_value=2000)
        vi_max_texts = _parse_int(os.getenv("ENVID_METADATA_GCP_VI_MAX_TEXTS"), default=120, min_value=1, max_value=5000)
        vi_max_shots = _parse_int(os.getenv("ENVID_METADATA_GCP_VI_MAX_SHOTS"), default=500, min_value=1, max_value=5000)
        vi_max_explicit_frames = _parse_int(os.getenv("ENVID_METADATA_GCP_VI_MAX_EXPLICIT_FRAMES"), default=600, min_value=1, max_value=50000)
        vi_max_segments_per_label = _parse_int(os.getenv("ENVID_METADATA_GCP_VI_MAX_SEGMENTS_PER_LABEL"), default=8, min_value=1, max_value=500)
        vi_max_segments_per_text = _parse_int(os.getenv("ENVID_METADATA_GCP_VI_MAX_SEGMENTS_PER_TEXT"), default=8, min_value=1, max_value=500)
        vi_max_frames_per_label = _parse_int(os.getenv("ENVID_METADATA_GCP_VI_MAX_FRAMES_PER_LABEL"), default=200, min_value=1, max_value=20000)

        enable_objects = _env_truthy(os.getenv("ENVID_METADATA_GCP_VI_ENABLE_OBJECT_TRACKING"), default=False)
        enable_logos = _env_truthy(os.getenv("ENVID_METADATA_GCP_VI_ENABLE_LOGO_RECOGNITION"), default=False)
        enable_people = _env_truthy(os.getenv("ENVID_METADATA_GCP_VI_ENABLE_PERSON_DETECTION"), default=False)

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

        def _ensure_parallel_executor() -> ThreadPoolExecutor:
            nonlocal parallel_executor
            if parallel_executor is None:
                parallel_executor = ThreadPoolExecutor(max_workers=5)
            return parallel_executor

        def _run_local_ocr_task() -> None:
            nonlocal local_text, effective_models
            try:
                _job_update(job_id, progress=26, message="Text on Screen (OCR)")
                interval = float(_parse_int(os.getenv("ENVID_METADATA_OCR_FRAME_INTERVAL_SECONDS"), default=2, min_value=1, max_value=30) or 2)
                max_frames = int(_parse_int(os.getenv("ENVID_METADATA_OCR_MAX_FRAMES"), default=120, min_value=1, max_value=5000) or 120)
                ocr_lang = (os.getenv("ENVID_METADATA_OCR_LANG") or "en").strip() or "en"
                ocr_timeout = int(_parse_int(os.getenv("ENVID_METADATA_LOCAL_OCR_TIMEOUT_SECONDS"), default=600, min_value=10, max_value=3600) or 600)

                _job_step_update(job_id, "text_on_screen", status="running", percent=0, message=f"Running ({requested_text_model})")
                if requested_text_model == "tesseract":
                    local_text = _tesseract_text_from_video(video_path=local_path, interval_seconds=interval, max_frames=max_frames)
                    effective_models["text_on_screen"] = "tesseract"
                else:
                    paddle_service_url = (os.getenv("ENVID_METADATA_LOCAL_OCR_PADDLE_URL") or "").strip()
                    if paddle_service_url:
                        local_text = _local_ocr_paddle_service_text_from_video(
                            video_path=local_path,
                            interval_seconds=interval,
                            max_frames=max_frames,
                            service_base_url=paddle_service_url,
                            lang=ocr_lang,
                            timeout_seconds=ocr_timeout,
                        )
                        effective_models["text_on_screen"] = "paddleocr_service"
                    else:
                        local_text = _paddleocr_text_from_video(
                            video_path=local_path,
                            interval_seconds=interval,
                            max_frames=max_frames,
                            lang=ocr_lang,
                        )
                        effective_models["text_on_screen"] = "paddleocr"
                _job_step_update(job_id, "text_on_screen", status="completed", percent=100, message=f"{len(local_text)} text entries ({effective_models.get('text_on_screen')})")
            except Exception as exc:
                app.logger.warning("Local OCR failed: %s", exc)
                _job_step_update(job_id, "text_on_screen", status="failed", message=str(exc)[:240])

        def _run_local_moderation_task() -> None:
            nonlocal local_moderation_frames, local_moderation_source, effective_models
            try:
                _job_update(job_id, progress=28, message="Moderation (local)")
                interval = float(_parse_int(os.getenv("ENVID_METADATA_MODERATION_FRAME_INTERVAL_SECONDS"), default=2, min_value=1, max_value=30) or 2)
                max_frames = int(_parse_int(os.getenv("ENVID_METADATA_MODERATION_MAX_FRAMES"), default=120, min_value=1, max_value=5000) or 120)
                service_url_default = (os.getenv("ENVID_METADATA_LOCAL_MODERATION_URL") or "").strip()
                service_url_nsfwjs = (os.getenv("ENVID_METADATA_LOCAL_MODERATION_NSFWJS_URL") or "").strip()
                service_url = (local_moderation_url_override or (service_url_nsfwjs if requested_moderation_model == "nsfwjs" else "") or service_url_default).strip()
                service_timeout = int(_parse_int(os.getenv("ENVID_METADATA_LOCAL_MODERATION_TIMEOUT_SECONDS"), default=60, min_value=1, max_value=600) or 60)

                _job_step_update(job_id, "moderation", status="running", percent=0, message=f"Running ({requested_moderation_model})")
                if requested_moderation_model == "nudenet":
                    if service_url:
                        local_moderation_frames = _local_moderation_service_explicit_frames_from_video(
                            video_path=local_path,
                            interval_seconds=interval,
                            max_frames=max_frames,
                            service_base_url=service_url,
                            timeout_seconds=service_timeout,
                        )
                        local_moderation_source = "nudenet_service"
                    else:
                        local_moderation_frames = _nudenet_moderation_explicit_frames_from_video(
                            video_path=local_path,
                            interval_seconds=interval,
                            max_frames=max_frames,
                        )
                        local_moderation_source = "nudenet"
                    effective_models["moderation"] = "nudenet" if local_moderation_source == "nudenet" else "nudenet_service"
                    _job_step_update(job_id, "moderation", status="completed", percent=100, message=f"{len(local_moderation_frames or [])} frames ({local_moderation_source})")
                elif requested_moderation_model == "nsfwjs":
                    if not service_url:
                        raise RuntimeError(
                            "nsfwjs requires a local moderation service; set ENVID_METADATA_LOCAL_MODERATION_URL (e.g. http://localhost:5081)"
                        )
                    local_moderation_frames = _local_moderation_service_explicit_frames_from_video(
                        video_path=local_path,
                        interval_seconds=interval,
                        max_frames=max_frames,
                        service_base_url=service_url,
                        timeout_seconds=service_timeout,
                    )
                    local_moderation_source = "nsfwjs_service"
                    effective_models["moderation"] = local_moderation_source
                    _job_step_update(job_id, "moderation", status="completed", percent=100, message=f"{len(local_moderation_frames or [])} frames ({local_moderation_source})")
                else:
                    raise RuntimeError(f"Unsupported moderation model: {requested_moderation_model}")
            except Exception as exc:
                app.logger.warning("Local moderation failed: %s", exc)
                _job_step_update(job_id, "moderation", status="failed", message=str(exc)[:240])

        def _run_whisper_transcription() -> None:
            nonlocal transcript, transcript_language_code, languages_detected, transcript_words, transcript_segments, transcript_meta, effective_models
            whisperx_cmd = _whisperx_command()
            if not whisperx_cmd:
                _job_step_update(job_id, "transcribe", status="skipped", percent=100, message="WhisperX not available")
                return
            try:
                app.logger.info("WhisperX transcription started")
                _job_update(job_id, progress=32, message="WhisperX")
                whisperx_model = "large-v2"
                _job_step_update(job_id, "transcribe", status="running", percent=5, message=f"Running (WhisperX/{whisperx_model})")

                whisper_device = "cuda" if shutil.which("nvidia-smi") else "cpu"

                # Audio preprocessing (best-effort): extract clean mono 16kHz WAV and optionally slow down tempo.
                # This improves stability for noisy inputs and some fast-speaking clips.
                preprocess_enabled = False
                tempo = 1.0
                filters = ""

                audio_for_whisper: Path = local_path
                audio_preprocess_fallback_used = False
                if preprocess_enabled:
                    audio_path = temp_dir / "whisper_audio.wav"
                    cmd_base = [
                        FFMPEG_PATH,
                        "-i",
                        str(local_path),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-acodec",
                        "pcm_s16le",
                    ]
                    cmd = list(cmd_base)
                    if filters:
                        cmd += ["-af", filters]
                    cmd += [str(audio_path), "-y"]
                    res = subprocess.run(cmd, capture_output=True)
                    if res.returncode != 0:
                        err = (res.stderr or b"").decode("utf-8", errors="ignore")
                        raise RuntimeError(f"FFmpeg denoise failed (filters required): {err[:240]}")
                    audio_for_whisper = audio_path

                transcript_meta["whisperx"] = {
                    "model": whisperx_model,
                    "device": whisper_device or "auto",
                    "audio_preprocess": {
                        "enabled": bool(preprocess_enabled),
                        "filters": filters,
                        "atempo": float(tempo) if tempo else 1.0,
                        "format": "wav_s16le_mono_16khz" if preprocess_enabled else "source_video",
                        "fallback_used": bool(audio_preprocess_fallback_used),
                    },
                }
                whisper_language = None

                output_dir = temp_dir / "whisperx"
                output_dir.mkdir(parents=True, exist_ok=True)

                cmd = [*whisperx_cmd, str(audio_for_whisper), "--model", whisperx_model, "--output_dir", str(output_dir), "--output_format", "json"]
                if whisper_device:
                    cmd += ["--device", whisper_device]
                if whisper_language:
                    cmd += ["--language", whisper_language]
                compute_type = "float16" if whisper_device == "cuda" else "float32"
                cmd += ["--compute_type", compute_type]
                cmd += ["--batch_size", "32"]
                cmd += ["--vad_method", "silero"]

                env = os.environ.copy()
                env.setdefault("PYTHONWARNINGS", "ignore")
                if whisper_device == "cuda":
                    env.setdefault("CT2_USE_CUDA", "1")
                    env.setdefault("CT2_CUDA_ALLOW_TF32", "1")
                    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
                app.logger.info("WhisperX cmd: %s", " ".join(cmd))
                timeout_s = 5400.0
                start_ts = time.monotonic()
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                last_emit = 0.0
                while True:
                    try:
                        res_code = proc.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        elapsed = time.monotonic() - start_ts
                        if elapsed - last_emit >= 15:
                            last_emit = elapsed
                            pct = min(95, 5 + int(elapsed / 30))
                            _job_step_update(
                                job_id,
                                "transcribe",
                                status="running",
                                percent=pct,
                                message=f"Running (WhisperX/{whisperx_model}) {int(elapsed)}s",
                            )
                stdout_text, stderr_text = proc.communicate()
                stdout_text = (stdout_text or "").strip()
                stderr_text = (stderr_text or "").strip()
                log_path = output_dir / "whisperx_cli.log"
                if stderr_text or stdout_text:
                    log_path.write_text(
                        ("STDOUT:\n" + stdout_text + "\n\nSTDERR:\n" + stderr_text).strip() + "\n",
                        encoding="utf-8",
                        errors="ignore",
                    )

                json_files = sorted(output_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                if res_code != 0 and not json_files:
                    err = (stderr_text or stdout_text or "").strip()
                    raise RuntimeError(f"WhisperX failed: {err[:240]}")
                if not json_files:
                    raise RuntimeError("WhisperX output JSON not found")
                data = json.loads(json_files[0].read_text(encoding="utf-8", errors="ignore") or "{}")

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

                # Preserve raw transcript/segments for diagnostics before any corrections.
                raw_transcript = transcript
                raw_transcript_segments = [dict(s) for s in transcript_segments] if transcript_segments else []

                # Correction settings (applied to both transcript and segments).
                grammar_enabled = True
                grammar_url = (os.getenv("ENVID_GRAMMAR_CORRECTION_URL") or "").strip()
                grammar_local = False
                dictionary_enabled = True
                nlp_mode = "openrouter_llama"
                punctuation_enabled = True

                # Apply full correction pipeline to the full transcript (for metadata/visibility).
                if transcript:
                    before = transcript
                    transcript, corr_meta = _apply_segment_corrections(
                        text=transcript,
                        language_code=(whisper_language or transcript_language_code or "hi"),
                        grammar_enabled=bool(grammar_enabled),
                        hindi_dictionary_enabled=bool(dictionary_enabled),
                        nlp_mode=nlp_mode,
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
                        "applied": bool(corr_meta.get("dictionary_applied")),
                    }
                    transcript_meta["hindi_dictionary_correction"] = {
                        "enabled": bool(dictionary_enabled and lang_norm == "hi"),
                        "applied": bool(corr_meta.get("hindi_applied")),
                    }
                    transcript_meta["nlp_correction"] = {
                        "enabled": nlp_mode in {"openrouter_llama", "llama", "openrouter"},
                        "mode": nlp_mode,
                        "applied": bool(corr_meta.get("nlp_applied")),
                    }
                    transcript_meta["punctuation_enhance"] = {
                        "enabled": bool(punctuation_enabled),
                        "applied": bool(corr_meta.get("punctuation_applied")),
                    }

                # Keep time-band segments consistent with corrected transcript.
                segment_correction_enabled = True
                segment_mode = "full"
                segment_max = 5000
                if segment_correction_enabled and transcript_segments:
                    corrected_segments: list[dict[str, Any]] = []
                    corrected_count = 0
                    for seg in transcript_segments[:segment_max]:
                        if not isinstance(seg, dict):
                            continue
                        txt = str(seg.get("text") or "").strip()
                        if not txt:
                            corrected_segments.append(seg)
                            continue

                        seg_grammar = grammar_enabled and segment_mode == "full"
                        seg_nlp_mode = nlp_mode if segment_mode in {"full", "nlp"} else ""
                        seg_punct = punctuation_enabled or segment_mode in {"full", "nlp", "punctuation"}

                        corrected_txt, _seg_meta = _apply_segment_corrections(
                            text=txt,
                            language_code=(transcript_language_code or whisper_language or "hi"),
                            grammar_enabled=bool(seg_grammar),
                            hindi_dictionary_enabled=bool(dictionary_enabled),
                            nlp_mode=seg_nlp_mode,
                            punctuation_enabled=bool(seg_punct),
                        )
                        if corrected_txt and corrected_txt != txt:
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
                            "max_segments": int(segment_max),
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
                    def _word_similarity(a: str, b: str) -> float:
                        a_words = re.findall(r"[\w\u0900-\u097F]+", a or "")
                        b_words = re.findall(r"[\w\u0900-\u097F]+", b or "")
                        if not a_words or not b_words:
                            return 0.0
                        return difflib.SequenceMatcher(None, a_words, b_words).ratio()

                    min_similarity = 0.60
                    similarity = _word_similarity(raw_transcript, transcript)
                    transcript_meta["correction_similarity"] = float(similarity)
                    if similarity < min_similarity:
                        transcript = raw_transcript
                        if raw_transcript_segments:
                            transcript_segments = raw_transcript_segments
                        transcript_meta["correction_reverted"] = True

                # Strict verification: fail the job if transcript looks invalid.
                verify_strict = False
                require_correction = True
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
                            transcript_meta.get("nlp_correction", {}).get("applied"),
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

                effective_models["transcribe"] = f"whisperx/{whisperx_model}"
                _job_step_update(job_id, "transcribe", status="completed", percent=100, message=f"Completed (WhisperX/{whisperx_model})")
                app.logger.info("WhisperX transcription completed")
            except TranscriptVerificationError as exc:
                app.logger.warning("Transcript verification failed: %s", exc)
                _job_step_update(job_id, "transcribe", status="failed", message=str(exc)[:240])
                raise
            except Exception as exc:
                app.logger.warning("WhisperX failed: %s", exc)
                _job_step_update(job_id, "transcribe", status="failed", percent=100, message=str(exc)[:240])
                # Fall through to the GCP path below.
                transcript = ""
                transcript_language_code = ""
                languages_detected = []
                transcript_words = []
                transcript_segments = []

        def _run_scene_detect_task() -> None:
            nonlocal precomputed_scenes, precomputed_scenes_source
            try:
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
                if (not precomputed_scenes) and use_pyscenedetect_for_scenes:
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

            if enable_transcribe and (transcribe_effective_mode == "whisperx") and not _whisperx_available():
                _job_step_update(job_id, "transcribe", status="skipped", percent=100, message="WhisperX not available")
            elif enable_transcribe and (transcribe_effective_mode == "whisperx") and _whisperx_available():
                whisper_started = True
                if sequential_scene_then_whisper:
                    _run_whisper_transcription()
                else:
                    parallel_futures["whisper"] = _ensure_parallel_executor().submit(_run_whisper_transcription)

        if enable_vi and gcp_video_intelligence is not None and want_any_vi:
            try:
                _job_update(job_id, progress=12, message="Video Intelligence")
                sequential_vi = bool(run_all_sequential)

                # Mark VI-backed steps based on user selection.
                if enable_label_detection:
                    if use_yolo_labels:
                        _job_step_update(
                            job_id,
                            "label_detection",
                            status="not_started",
                            percent=0,
                            message="Pending YOLO(Ultralytics)",
                        )
                    elif use_local_label_detection_service:
                        _job_step_update(
                            job_id,
                            "label_detection",
                            status="not_started",
                            percent=0,
                            message=f"Pending {label_engine}",
                        )
                    else:
                        _job_step_update(
                            job_id,
                            "label_detection",
                            status="running" if not sequential_vi else "not_started",
                            percent=0,
                            message=(
                                f"Running (Google Video Intelligence; mode={vi_label_mode})"
                                if not sequential_vi
                                else "Queued (sequential)"
                            ),
                        )
                else:
                    _job_step_update(job_id, "label_detection", status="skipped", percent=100, message="Disabled")

                if enable_text_on_screen:
                    if use_local_ocr:
                        if not local_ocr_started:
                            _job_step_update(
                                job_id,
                                "text_on_screen",
                                status="not_started",
                                percent=0,
                                message=f"Pending local OCR ({requested_text_model})",
                            )
                    else:
                        _job_step_update(
                            job_id,
                            "text_on_screen",
                            status="running" if not sequential_vi else "not_started",
                            percent=0,
                            message=(
                                f"Running (via gcp_video_intelligence; requested: {requested_models.get('text_model')})"
                                if not sequential_vi
                                else "Queued (sequential)"
                            ),
                        )
                else:
                    _job_step_update(job_id, "text_on_screen", status="skipped", percent=100, message="Disabled")

                if enable_moderation:
                    if use_local_moderation:
                        if not local_moderation_started:
                            _job_step_update(
                                job_id,
                                "moderation",
                                status="not_started",
                                percent=0,
                                message=f"Pending local moderation ({requested_moderation_model})",
                            )
                    else:
                        _job_step_update(
                            job_id,
                            "moderation",
                            status="running" if not sequential_vi else "not_started",
                            percent=0,
                            message=(
                                f"Running (via gcp_video_intelligence; requested: {requested_models.get('moderation_model')})"
                                if not sequential_vi
                                else "Queued (sequential)"
                            ),
                        )
                else:
                    _job_step_update(job_id, "moderation", status="skipped", percent=100, message="Disabled")

                vi_client = gcp_video_intelligence.VideoIntelligenceServiceClient()
                features: List[Any] = []
                if enable_label_detection and use_vi_label_detection:
                    features.append(gcp_video_intelligence.Feature.LABEL_DETECTION)
                if enable_text_on_screen and not use_local_ocr:
                    features.append(gcp_video_intelligence.Feature.TEXT_DETECTION)
                if want_vi_shots:
                    features.append(gcp_video_intelligence.Feature.SHOT_CHANGE_DETECTION)
                if enable_moderation and (not use_local_moderation or allow_moderation_fallback):
                    features.append(gcp_video_intelligence.Feature.EXPLICIT_CONTENT_DETECTION)

                if enable_objects:
                    try:
                        features.append(gcp_video_intelligence.Feature.OBJECT_TRACKING)
                    except Exception:
                        pass
                if enable_logos:
                    try:
                        features.append(gcp_video_intelligence.Feature.LOGO_RECOGNITION)
                    except Exception:
                        pass
                if enable_people:
                    try:
                        features.append(gcp_video_intelligence.Feature.PERSON_DETECTION)
                    except Exception:
                        pass

                req: Dict[str, Any] = {"input_uri": gcs_uri, "features": features}


                # Optional label detection mode: segment (default) / shot / frame.
                # Server-side acceptance varies by API/version, so best-effort.
                try:
                    if vi_label_mode in {"shot", "frame"}:
                        vc = req.get("video_context") if isinstance(req.get("video_context"), dict) else {}
                        cfg = vc.get("label_detection_config") if isinstance(vc.get("label_detection_config"), dict) else {}
                        # Common enum strings accepted by backend.
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

                # Shots (optional)
                shots: List[Dict[str, Any]] = []
                if want_shots:
                    for s in (getattr(ar, "shot_annotations", None) or [])[:vi_max_shots]:
                        shots.append(
                            {
                                "start": _dur_seconds(getattr(s, "start_time_offset", None)),
                                "end": _dur_seconds(getattr(s, "end_time_offset", None)),
                            }
                        )

                # Labels (optional)
                labels: List[Dict[str, Any]] = []

                label_annotations: list[Any] = []
                if vi_label_mode == "shot":
                    label_annotations = list(getattr(ar, "shot_label_annotations", None) or [])
                elif vi_label_mode == "frame":
                    label_annotations = list(getattr(ar, "frame_label_annotations", None) or [])
                else:
                    label_annotations = list(getattr(ar, "segment_label_annotations", None) or [])

                if enable_label_detection and use_vi_label_detection:
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
                            labels.append({"label": desc, "categories": cats, "segments": segs})

                # Text (optional)
                texts: List[Dict[str, Any]] = []
                if enable_text_on_screen:
                    for t in (getattr(ar, "text_annotations", None) or [])[:vi_max_texts]:
                        txt = (getattr(t, "text", None) or "").strip()
                        if not txt:
                            continue
                        segs: List[Dict[str, Any]] = []
                        for seg in (getattr(t, "segments", None) or [])[:vi_max_segments_per_text]:
                            seg_obj = getattr(seg, "segment", None)
                            conf = float(getattr(seg, "confidence", 0.0) or 0.0)
                            if conf < vi_min_text_conf:
                                continue
                            segs.append(
                                {
                                    "start": _dur_seconds(getattr(seg_obj, "start_time_offset", None)),
                                    "end": _dur_seconds(getattr(seg_obj, "end_time_offset", None)),
                                    "confidence": conf,
                                }
                            )
                        if segs:
                            texts.append({"text": txt, "segments": segs})

                # Explicit content (moderation) (optional)
                moderation_frames: List[Dict[str, Any]] = []
                explicit = getattr(ar, "explicit_annotation", None)
                if enable_moderation and (not use_local_moderation or allow_moderation_fallback):
                    include_provider_details = _env_truthy(os.getenv("ENVID_METADATA_INCLUDE_PROVIDER_DETAILS"), default=False)
                    for f in (getattr(explicit, "frames", None) or [])[:vi_max_explicit_frames]:
                        lk = str(getattr(f, "pornography_likelihood", None) or "")
                        moderation_frames.append(
                            {
                                "time": _dur_seconds(getattr(f, "time_offset", None)),
                                "severity": _severity_from_likelihood(lk),
                                **({"provider_likelihood": lk} if include_provider_details and lk else {}),
                            }
                        )

                # Optional: objects
                objects: List[Dict[str, Any]] = []
                try:
                    for oa in (getattr(ar, "object_annotations", None) or [])[:2000]:
                        ent = getattr(oa, "entity", None)
                        name = (getattr(ent, "description", None) or "").strip()
                        if not name:
                            continue
                        segs: List[Dict[str, Any]] = []
                        # Use frames to compute coarse segments.
                        for fr in (getattr(oa, "frames", None) or [])[:vi_max_frames_per_label]:
                            t = _dur_seconds(getattr(fr, "time_offset", None))
                            segs.append({"start": t, "end": t + 1.0, "confidence": float(getattr(fr, "confidence", 0.0) or 0.0)})
                        segs = segs[:vi_max_segments_per_label]
                        if segs:
                            objects.append({"name": name, "segments": segs})
                except Exception:
                    pass

                # Optional: logos
                logos: List[Dict[str, Any]] = []
                try:
                    for la in (getattr(ar, "logo_recognition_annotations", None) or [])[:2000]:
                        ent = getattr(la, "entity", None)
                        name = (getattr(ent, "description", None) or "").strip()
                        if not name:
                            continue
                        segs: List[Dict[str, Any]] = []
                        for tr in (getattr(la, "tracks", None) or [])[:25]:
                            seg = getattr(tr, "segment", None)
                            segs.append({"start": _dur_seconds(getattr(seg, "start_time_offset", None)), "end": _dur_seconds(getattr(seg, "end_time_offset", None))})
                        segs = segs[:vi_max_segments_per_label]
                        if segs:
                            logos.append({"name": name, "segments": segs})
                except Exception:
                    pass

                # Optional: people
                people: List[Dict[str, Any]] = []
                try:
                    for pa in (getattr(ar, "person_detection_annotations", None) or [])[:2000]:
                        segs: List[Dict[str, Any]] = []
                        for tr in (getattr(pa, "tracks", None) or [])[:50]:
                            seg = getattr(tr, "segment", None)
                            segs.append(
                                {
                                    "start": _dur_seconds(getattr(seg, "start_time_offset", None)),
                                    "end": _dur_seconds(getattr(seg, "end_time_offset", None)),
                                }
                            )
                        segs = segs[:vi_max_segments_per_label]
                        if segs:
                            # No stable label description in API; emit generic "person".
                            people.append({"name": "person", "segments": segs})
                except Exception:
                    pass

                video_intelligence = {
                    "source": "gcp_video_intelligence",
                    "config": {
                        "label_detection_mode": vi_label_mode,
                        "transcribe_mode": transcribe_effective_mode,
                        "requested_models": requested_models,
                    },
                    "labels": labels,
                    "text": texts,
                    "shots": shots,
                    "moderation": {"explicit_frames": moderation_frames},
                }
                if objects:
                    video_intelligence["objects"] = objects
                if logos:
                    video_intelligence["logos"] = logos
                if people:
                    video_intelligence["people"] = people

                if enable_label_detection and use_vi_label_detection:
                    if sequential_vi:
                        _job_step_update(
                            job_id,
                            "label_detection",
                            status="running",
                            percent=50,
                            message=f"Running (Google Video Intelligence; mode={vi_label_mode})",
                        )
                    _job_step_update(job_id, "label_detection", status="completed", percent=100, message=f"{len(labels)} labels")

                if enable_moderation and not use_local_moderation:
                    if sequential_vi:
                        _job_step_update(
                            job_id,
                            "moderation",
                            status="running",
                            percent=50,
                            message=f"Running (via gcp_video_intelligence; requested: {requested_models.get('moderation_model')})",
                        )
                    _job_step_update(job_id, "moderation", status="completed", percent=100, message=f"{len(moderation_frames)} frames")

                if enable_text_on_screen and not use_local_ocr:
                    if sequential_vi:
                        _job_step_update(
                            job_id,
                            "text_on_screen",
                            status="running",
                            percent=50,
                            message=f"Running (via gcp_video_intelligence; requested: {requested_models.get('text_model')})",
                        )
                    _job_step_update(job_id, "text_on_screen", status="completed", percent=100, message=f"{len(texts)} text entries")
            except Exception as exc:
                app.logger.warning("Video Intelligence failed: %s", exc)
                for step, enabled in [
                    ("label_detection", enable_label_detection and use_vi_label_detection),
                    ("text_on_screen", enable_text_on_screen),
                    ("moderation", enable_moderation),
                ]:
                    if enabled:
                        _job_step_update(job_id, step, status="failed", message=str(exc))
        else:
            msg = "Disabled" if not enable_vi else ("Library not installed" if gcp_video_intelligence is None else "Disabled")
            for step, enabled in [
                ("label_detection", enable_label_detection and use_vi_label_detection),
                ("text_on_screen", enable_text_on_screen and not use_local_ocr),
                ("moderation", enable_moderation),
            ]:
                if enabled:
                    _job_step_update(job_id, step, status="skipped", percent=100, message=msg)
                else:
                    _job_step_update(job_id, step, status="skipped", percent=100, message="Disabled")

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

            if enable_transcribe and (transcribe_effective_mode == "whisperx") and not _whisperx_available():
                _job_step_update(job_id, "transcribe", status="skipped", percent=100, message="WhisperX not available")
            elif enable_transcribe and (transcribe_effective_mode == "whisperx") and _whisperx_available():
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
                        _job_step_update(job_id, "transcribe", status="failed", percent=100, message="WhisperX timed out")
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
        elif enable_moderation and use_local_moderation and allow_moderation_fallback:
            fallback_frames: List[Dict[str, Any]] = []
            try:
                if isinstance(video_intelligence, dict):
                    fallback_frames = ((video_intelligence.get("moderation") or {}).get("explicit_frames") or [])
            except Exception:
                fallback_frames = []
            if fallback_frames:
                effective_models["moderation"] = "gcp_video_intelligence_fallback"
                _job_step_update(
                    job_id,
                    "moderation",
                    status="completed",
                    percent=100,
                    message=f"{len(fallback_frames)} frames (fallback: gcp_video_intelligence)",
                )

        # Local label detection via Ultralytics YOLO (when requested).
        if enable_label_detection and use_yolo_labels:
            try:
                _job_update(job_id, progress=22, message="Label detection (YOLO)")
                yolo_model = (os.getenv("ENVID_METADATA_YOLO_MODEL") or "yolov8n.pt").strip() or "yolov8n.pt"
                # Default to repo-root weights if present.
                yolo_path = Path(yolo_model)
                if not yolo_path.exists():
                    candidate = _repo_root() / yolo_model
                    if candidate.exists():
                        yolo_path = candidate
                interval = float(_parse_int(os.getenv("ENVID_METADATA_YOLO_FRAME_INTERVAL_SECONDS"), default=1, min_value=1, max_value=30) or 1)
                max_frames = int(_parse_int(os.getenv("ENVID_METADATA_YOLO_MAX_FRAMES"), default=120, min_value=1, max_value=5000) or 120)
                conf = float(os.getenv("ENVID_METADATA_YOLO_CONF") or 0.25)

                _job_step_update(job_id, "label_detection", status="running", percent=0, message=f"Running (YOLO; model={yolo_path.name})")
                local_labels = _yolo_detect_labels_from_video(
                    video_path=local_path,
                    model_name_or_path=str(yolo_path),
                    frame_interval_seconds=interval,
                    max_frames=max_frames,
                    conf=conf,
                )
                if not isinstance(video_intelligence, dict):
                    video_intelligence = {}
                video_intelligence["labels"] = local_labels
                cfg = video_intelligence.get("config") if isinstance(video_intelligence.get("config"), dict) else {}
                cfg["labels_engine"] = "yolo_ultralytics"
                cfg["yolo"] = {
                    "model": str(yolo_path),
                    "frame_interval_seconds": interval,
                    "max_frames": max_frames,
                    "conf": conf,
                }
                video_intelligence["config"] = cfg
                effective_models["label_detection"] = "yolo_ultralytics"
                _job_step_update(job_id, "label_detection", status="completed", percent=100, message=f"{len(local_labels)} labels (YOLO)")
            except Exception as exc:
                app.logger.warning("YOLO label detection failed: %s", exc)
                _job_step_update(job_id, "label_detection", status="failed", message=str(exc)[:240])

        # Local label detection via external service (Detectron2/MMDetection).
        if enable_label_detection and use_local_label_detection_service:
            try:
                _job_update(job_id, progress=22, message=f"Label detection ({label_engine})")
                service_url = (
                    local_label_detection_url_override
                    or os.getenv("ENVID_METADATA_LOCAL_LABEL_DETECTION_URL")
                    or ""
                ).strip()
                if not service_url:
                    raise RuntimeError("ENVID_METADATA_LOCAL_LABEL_DETECTION_URL is not set")

                interval = float(_parse_int(os.getenv("ENVID_METADATA_LOCAL_LABEL_FRAME_INTERVAL_SECONDS"), default=1, min_value=1, max_value=30) or 1)
                max_frames = int(_parse_int(os.getenv("ENVID_METADATA_LOCAL_LABEL_MAX_FRAMES"), default=120, min_value=1, max_value=5000) or 120)
                timeout = int(_parse_int(os.getenv("ENVID_METADATA_LOCAL_LABEL_SERVICE_TIMEOUT_SECONDS"), default=120, min_value=5, max_value=3600) or 120)

                _job_step_update(job_id, "label_detection", status="running", percent=0, message=f"Running ({label_engine})")
                local_labels = _local_label_detection_service_labels_from_video(
                    video_path=local_path,
                    interval_seconds=interval,
                    max_frames=max_frames,
                    service_base_url=service_url,
                    model=label_engine,
                    timeout_seconds=timeout,
                )
                if not isinstance(video_intelligence, dict):
                    video_intelligence = {}
                video_intelligence["labels"] = local_labels
                cfg = video_intelligence.get("config") if isinstance(video_intelligence.get("config"), dict) else {}
                cfg["labels_engine"] = label_engine
                cfg["local_label_detection_service"] = {
                    "url": service_url,
                    "frame_interval_seconds": interval,
                    "max_frames": max_frames,
                    "timeout_seconds": timeout,
                }
                video_intelligence["config"] = cfg
                effective_models["label_detection"] = label_engine
                _job_step_update(job_id, "label_detection", status="completed", percent=100, message=f"{len(local_labels)} labels ({label_engine})")
            except Exception as exc:
                app.logger.warning("Local label detection (%s) failed: %s", label_engine, exc)
                _job_step_update(job_id, "label_detection", status="failed", message=str(exc)[:240])

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
                cmd = [FFMPEG_PATH, "-i", str(local_path), "-vn", "-ac", "1", "-af", "aresample=16000", "-acodec", "flac", str(audio_path), "-y"]
                subprocess.run(cmd, capture_output=True, check=True)

                artifacts_bucket = _gcs_artifacts_bucket(gcs_bucket)
                artifacts_prefix = _gcs_artifacts_prefix()
                audio_obj = f"{artifacts_prefix}/{job_id}/audio.flac"
                client = _gcs_client()
                client.bucket(artifacts_bucket).blob(audio_obj).upload_from_filename(str(audio_path), content_type="audio/flac")
                audio_uri = f"gs://{artifacts_bucket}/{audio_obj}"

                speech_client = gcp_speech.SpeechClient()
                primary_lang = (os.getenv("GCP_SPEECH_LANGUAGE_CODE") or os.getenv("SPEECH_LANGUAGE_CODE") or "hi-IN").strip() or "hi-IN"
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

                _job_step_update(job_id, "transcribe", status="completed", percent=100, message="Completed")
            except Exception as exc:
                app.logger.warning("Speech-to-Text failed: %s", exc)
                _job_step_update(job_id, "transcribe", status="failed", message=str(exc))
        else:
            if transcript:
                # Already completed via Video Intelligence.
                pass
            else:
                if not enable_transcribe:
                    _job_step_update(job_id, "transcribe", status="skipped", percent=100, message="Disabled")
                else:
                    if transcribe_effective_mode == "whisperx":
                        msg = "WhisperX not installed" if not _whisperx_available() else "Disabled"
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

        # Famous locations are not implemented in the slim stack; always skip.
        enable_famous_locations = False
        _job_step_update(job_id, "famous_location_detection", status="skipped", percent=100, message="Disabled")
        translated_en = ""
        en_segments: List[Dict[str, Any]] = []
        enable_translate = _env_truthy(os.getenv("ENVID_METADATA_ENABLE_TRANSLATE"), default=True)
        translate_provider = _translate_provider()
        has_text_to_translate = bool(transcript_segments) or bool((transcript or "").strip())
        requested_targets = _translate_targets_from_selection(sel)
        translate_targets = requested_targets or _translate_targets()
        if "en" not in translate_targets:
            translate_targets = ["en", *translate_targets]
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
                _job_update(job_id, progress=55, message="Translate outputs")
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

                if "en" in translations_by_lang:
                    en_segments = translations_by_lang["en"].get("transcript", {}).get("segments") or []
                    translated_en = translations_by_lang["en"].get("transcript", {}).get("text") or ""
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
            if "en" not in translations_by_lang:
                translations_by_lang["en"] = {
                    "transcript": {
                        "text": str(transcript or "").strip(),
                        "segments": transcript_segments,
                        "language_code": "en",
                        "fallback_copy": True,
                    }
                }
            subtitles.setdefault("orig", {"language_code": source_lang or ""})
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

        locations: List[Dict[str, Any]] = []

        synopses_by_age: Dict[str, Any] = {}
        try:
            if enable_synopsis_generation and (transcript or video_description):
                _job_update(job_id, progress=70, message="Synopses")
                _job_step_update(job_id, "synopsis_generation", status="running", percent=0, message="Running")
                data, meta = _openrouter_llama_generate_synopses(
                    text=(transcript or video_description or ""),
                    language_code=(transcript_language_code or "hi"),
                )
                if isinstance(data, dict):
                    if _validate_synopses_payload(data, transcript_language_code or "hi"):
                        synopses_by_age = data
                        effective_models["synopsis_generation"] = meta.get("model") or "openrouter"
                    else:
                        effective_models["synopsis_generation"] = "validation_failed"
                if not synopses_by_age:
                    fallback = _fallback_synopses_from_text(
                        transcript or video_description or "",
                        transcript_language_code or "hi",
                    )
                    if isinstance(fallback, dict):
                        synopses_by_age = fallback
                        effective_models["synopsis_generation"] = "fallback"
                if synopses_by_age:
                    _job_step_update(job_id, "synopsis_generation", status="completed", percent=100, message="Completed")
                else:
                    msg = "Invalid synopsis" if effective_models.get("synopsis_generation") == "validation_failed" else "Empty response"
                    _job_step_update(job_id, "synopsis_generation", status="failed", percent=100, message=msg)
                    critical_failures.append(f"synopsis_generation: {msg}")
            else:
                if not enable_synopsis_generation:
                    effective_models["synopsis_generation"] = "disabled"
                    msg = "Disabled"
                elif not (transcript or video_description):
                    effective_models["synopsis_generation"] = "no_transcript_or_description"
                    msg = "No transcript/description"
                else:
                    effective_models["synopsis_generation"] = "not_configured"
                    msg = "Not configured"
                _job_step_update(job_id, "synopsis_generation", status="skipped", percent=100, message=msg)
        except Exception as exc:
            app.logger.warning("Synopses failed: %s", exc)
            msg = str(exc)
            effective_models["synopsis_generation"] = "failed"
            _job_step_update(job_id, "synopsis_generation", status="failed", message=msg[:240])
            critical_failures.append(f"synopsis_generation: {msg[:200]}")
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

            # Fallback: if scene detection yields a single segment on longer videos,
            # generate uniform segments so scene-by-scene output is still useful.
            if (not scenes or len(scenes) <= 1) and duration_seconds and float(duration_seconds) > 0:
                fallback = _fallback_scene_segments(duration_seconds=float(duration_seconds))
                if fallback:
                    scenes = fallback
                    scenes_source = "fallback_uniform"

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
                    summaries, meta = _openrouter_llama_scene_summaries(
                        scenes=scenes,
                        transcript_segments=transcript_segments,
                        labels_src=labels_src,
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
                        _job_step_update(job_id, "scene_by_scene_metadata", status="failed", percent=100, message=f"No scenes ({scenes_source})")
                        critical_failures.append(f"scene_by_scene_metadata: No scenes ({scenes_source})")
                    else:
                        _job_step_update(job_id, "scene_by_scene_metadata", status="skipped", percent=100, message="No scenes")
                else:
                    _job_step_update(job_id, "scene_by_scene_metadata", status="skipped", percent=100, message="Disabled")
                if enable_key_scene and (not key_scene_step_finalized):
                    if local_model_requested and (use_transnetv2_for_scenes or use_pyscenedetect_for_scenes):
                        _job_step_update(job_id, "key_scene_detection", status="failed", percent=100, message=f"No scenes ({scenes_source})")
                        critical_failures.append(f"key_scene_detection: No scenes ({scenes_source})")
                    else:
                        _job_step_update(job_id, "key_scene_detection", status="skipped", percent=100, message="No scenes")
                else:
                    if not key_scene_step_finalized:
                        _job_step_update(job_id, "key_scene_detection", status="skipped", percent=100, message="Disabled")
        else:
            _job_step_update(job_id, "scene_by_scene_metadata", status="skipped", percent=100, message="Disabled")
            if not key_scene_step_finalized:
                _job_step_update(job_id, "key_scene_detection", status="skipped", percent=100, message="Disabled")

        opening_closing: Dict[str, Any] | None = None
        effective_models["opening_closing_credit_detection"] = "not_implemented"
        _job_step_update(
            job_id,
            "opening_closing_credit_detection",
            status="skipped",
            percent=100,
            message="Not implemented",
        )

        metadata_text = " ".join([p for p in [video_title, video_description, transcript] if (p or "").strip()])

        thumbnail_base64: str | None = None
        try:
            mid_ts = int(float(duration_seconds or 0.0) / 2.0) if duration_seconds else 0
            out_path = temp_dir / "thumb.jpg"
            cmd = [FFMPEG_PATH, "-ss", str(mid_ts), "-i", str(local_path), "-vframes", "1", "-q:v", "2", str(out_path), "-y"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and out_path.exists():
                thumbnail_base64 = base64.b64encode(out_path.read_bytes()).decode("utf-8")
        except Exception:
            pass

        stored_filename = ""
        if _persist_local_video_copy():
            file_extension = Path(gcs_object).suffix or ".mp4"
            stored_filename = f"{job_id}{file_extension}"
            shutil.copy2(local_path, VIDEOS_DIR / stored_filename)

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

        # Build translated metadata payload (best-effort).
        if translations_by_lang:
            translations = {"meta": translations_meta, "languages": list(translations_by_lang.keys()), "by_language": {}}
            for lang, payload in translations_by_lang.items():
                by_lang: Dict[str, Any] = dict(payload)

                # Translate synopses.
                if synopses_by_age:
                    syn_tr: Dict[str, Any] = {}
                    for group in ("kids", "teens", "adults"):
                        item = synopses_by_age.get(group) if isinstance(synopses_by_age, dict) else None
                        if not isinstance(item, dict):
                            continue
                        short = _translate_text(
                            text=str(item.get("short") or ""),
                            source_lang=transcript_language_code,
                            target_lang=lang,
                            provider=translate_provider,
                            gcp_client=gcp_translate_client,
                            gcp_parent=gcp_translate_parent,
                        )
                        long = _translate_text(
                            text=str(item.get("long") or ""),
                            source_lang=transcript_language_code,
                            target_lang=lang,
                            provider=translate_provider,
                            gcp_client=gcp_translate_client,
                            gcp_parent=gcp_translate_parent,
                        )
                        syn_tr[group] = {"short": short, "long": long}
                    if syn_tr:
                        by_lang["synopses_by_age_group"] = syn_tr

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
            "synopses_by_age_group": synopses_by_age,
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

        # Best-effort: persist derived artifacts to GCS so this service can be run cloud-only
        # (no reliance on local subtitle/zip generation).
        try:
            _job_update(job_id, progress=90, message="Uploading artifacts to GCS")
            payload = {"id": job_id, "categories": video_entry.get("metadata_categories") or {}, "combined": video_entry.get("metadata_combined") or {}}
            video_entry["gcs_artifacts"] = _upload_metadata_artifacts_to_gcs(job_id=job_id, payload=payload)
        except Exception as exc:
            app.logger.warning("Failed to upload artifacts to GCS for %s: %s", job_id, exc)

        with VIDEO_INDEX_LOCK:
            existing_idx = next((i for i, v in enumerate(VIDEO_INDEX) if str(v.get("id")) == str(job_id)), None)
            if existing_idx is not None:
                VIDEO_INDEX[existing_idx] = video_entry
            else:
                VIDEO_INDEX.append(video_entry)
        _save_video_index()

        _job_step_update(job_id, "save_as_json", status="completed", percent=100, message="Completed")
        if critical_failures:
            err = "; ".join(critical_failures)[:400]
            _job_update(job_id, status="failed", progress=100, message="Failed", error=err)
        else:
            _job_update(job_id, status="completed", progress=100, message="Completed", result={"id": job_id, "title": video_title, "gcs_video_uri": gcs_uri})
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
            "service": "envid-metadata-multimodal",
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
    return jsonify(
        {
            "status": "ok",
            "cpu_percent": cpu_value,
            "gpu_percent": gpu_value,
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

    job_id = str(uuid.uuid4())
    original_filename = video_file.filename
    temp_dir = Path(tempfile.mkdtemp(prefix=f"envid_metadata_{job_id}_"))
    frame_interval_seconds = 0
    max_frames_to_analyze = 1000
    face_recognition_mode = (request.form.get("face_recognition_mode") or "").strip() or None
    task_selection = _parse_task_selection(request.form.get("task_selection"))

    try:
        video_path = temp_dir / "video.mp4"
        video_file.save(str(video_path))
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify({"error": f"Failed to save upload: {exc}"}), 500

    _job_init(job_id, title=video_title)
    _job_update(job_id, status="processing", progress=1, message="Upload received")

    def _worker() -> None:
        try:
            bucket = _gcs_bucket_name()
            prefix = _gcs_rawvideo_prefix()
            safe_name = Path(original_filename).name or "video.mp4"
            obj = f"{prefix}{job_id}/{safe_name}"
            gcs_uri = f"gs://{bucket}/{obj}"

            _job_step_update(job_id, "upload_to_cloud_storage", status="running", percent=0, message="Uploading")
            _job_update(job_id, progress=2, message="Uploading to cloud storage")

            client = _gcs_client()
            client.bucket(bucket).blob(obj).upload_from_filename(str(video_path))
            _job_step_update(job_id, "upload_to_cloud_storage", status="completed", percent=100, message="Uploaded")
            _job_update(job_id, gcs_video_uri=gcs_uri)

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
    task_selection = _parse_task_selection(payload.get("task_selection"))
    requested_job_id = (payload.get("job_id") or payload.get("id") or "").strip()
    job_id = requested_job_id if _looks_like_uuid(requested_job_id) else str(uuid.uuid4())

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


@app.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str) -> Any:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job), 200


@app.route("/videos", methods=["GET"])
def list_videos() -> Any:
    with VIDEO_INDEX_LOCK:
        videos = list(VIDEO_INDEX)
    videos.sort(key=lambda v: (v.get("uploaded_at") or "", v.get("id") or ""), reverse=True)
    return jsonify({"videos": videos, "count": len(videos)}), 200


@app.route("/video/<video_id>", methods=["GET"])
def get_video(video_id: str) -> Any:
    with VIDEO_INDEX_LOCK:
        v = next((x for x in VIDEO_INDEX if str(x.get("id")) == str(video_id)), None)
    if not v:
        return jsonify({"error": "Video not found"}), 404
    return jsonify(v), 200


@app.route("/video/<video_id>/metadata-json", methods=["GET"])
def get_video_metadata_json(video_id: str) -> Any:
    with VIDEO_INDEX_LOCK:
        v = next((x for x in VIDEO_INDEX if str(x.get("id")) == str(video_id)), None)
    if not v:
        return jsonify({"error": "Video not found"}), 404

    category = (request.args.get("category") or "").strip().lower()
    lang = (request.args.get("lang") or "").strip().lower() or None
    download = (request.args.get("download") or "").strip().lower() in {"1", "true", "yes", "on"}

    categorized = _build_categorized_metadata_json(v)
    categories = categorized.get("categories") if isinstance(categorized.get("categories"), dict) else {}
    combined = _build_combined_metadata_for_language(v, lang)

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


@app.route("/video/<video_id>/metadata-json.zip", methods=["GET"])
def get_video_metadata_zip(video_id: str) -> Any:
    with VIDEO_INDEX_LOCK:
        v = next((x for x in VIDEO_INDEX if str(x.get("id")) == str(video_id)), None)
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

    categorized = _build_categorized_metadata_json(v)
    categories = categorized.get("categories") if isinstance(categorized.get("categories"), dict) else {}
    combined = categorized.get("combined") if isinstance(categorized.get("combined"), dict) else {}
    payload = {"id": video_id, "categories": categories, "combined": combined}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{video_id}.metadata.json", json.dumps(payload, indent=2, ensure_ascii=False))
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"{video_id}.metadata-json.zip")


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
    base_url = (os.getenv("ENVID_LIBRETRANSLATE_URL") or os.getenv("LIBRETRANSLATE_URL") or "").strip()
    if not base_url:
        return jsonify({"ok": False, "languages": [], "error": "LibreTranslate URL is not configured"}), 200
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
            bucket, obj = _parse_allowed_gcs_video_source(gcs_uri)
            # IMPORTANT: never delete the source/original video object.
            # Deleting a history record should only remove temporary artifacts.
            kept_gcs_objects.append(f"gs://{bucket}/{obj}")

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
        return jsonify({"ok": True, "deleted_gcs_objects": deleted_gcs_objects, "kept_gcs_objects": kept_gcs_objects, "gcs_warnings": warnings}), 200
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
        bucket, obj = _parse_allowed_gcs_video_source(gcs_uri)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    # Reprocess should overwrite the existing entry (stable id) so clients can
    # poll /jobs/<video_id> and the index updates in-place.
    job_id = video_id if _looks_like_uuid(video_id) else str(uuid.uuid4())
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


@app.route("/search", methods=["POST"])
def search_videos() -> Any:
    return jsonify({"error": "Search has been removed"}), 410


def _extract_text_from_file(file_content: bytes, filename: str) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError("PDF support not installed")
        reader = PdfReader(io.BytesIO(file_content))
        parts: List[str] = []
        for p in reader.pages:
            try:
                parts.append(p.extract_text() or "")
            except Exception:
                continue
        return "\n".join(parts).strip()
    if name.endswith(".docx"):
        if DocxDocument is None:
            raise RuntimeError("DOCX support not installed")
        doc = DocxDocument(io.BytesIO(file_content))
        return "\n".join([p.text for p in doc.paragraphs if p.text]).strip()
    return file_content.decode("utf-8", errors="replace").strip()


@app.route("/upload-document", methods=["POST"])
def upload_document() -> Any:
    if "file" not in request.files:
        return jsonify({"error": "file is required"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    raw = f.read()
    if not raw:
        return jsonify({"error": "Empty file"}), 400

    doc_id = str(uuid.uuid4())
    filename = Path(f.filename).name
    path = DOCUMENTS_DIR / f"{doc_id}_{filename}"
    path.write_bytes(raw)

    text = _extract_text_from_file(raw, filename)

    entry = {"id": doc_id, "filename": filename, "stored_path": str(path.name), "text": text, "uploaded_at": datetime.utcnow().isoformat()}
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
