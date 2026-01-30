from __future__ import annotations

import os
import json
import base64
import tempfile
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
import uuid
import io
import zipfile
import re
import mimetypes
from PIL import Image, ImageOps
import urllib.parse
import urllib.request

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

from flask import Flask, jsonify, request, send_file, redirect
from werkzeug.exceptions import RequestEntityTooLarge
from flask_cors import CORS
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
try:
    from PyPDF2 import PdfReader
except ImportError:  # Prefer PyPDF2, fall back to pypdf if only modern package exists
    from pypdf import PdfReader
from docx import Document as DocxDocument
from shared.env_loader import load_environment

load_environment()

app = Flask(__name__)
CORS(app)

def _max_upload_bytes() -> int | None:
    """Return max upload bytes for Flask/werkzeug.

    Default is generous for local dev demos. Set `ENGRO_MAX_UPLOAD_GB` (or
    `ENGRO_MAX_UPLOAD_BYTES`) to tune.
    """

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


# Configure for large file uploads (local dev). Set to None for unlimited.
app.config["MAX_CONTENT_LENGTH"] = _max_upload_bytes()
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.errorhandler(RequestEntityTooLarge)
def _handle_request_too_large(_: RequestEntityTooLarge) -> Any:
    limit = app.config.get("MAX_CONTENT_LENGTH")
    if isinstance(limit, int) and limit > 0:
        limit_gb = limit / (1024 * 1024 * 1024)
        return jsonify({"error": f"Upload too large. Max allowed is {limit_gb:.1f} GB."}), 413
    return jsonify({"error": "Upload too large."}), 413


JOBS_LOCK = threading.Lock()
JOBS: Dict[str, Dict[str, Any]] = {}


def _job_steps_default() -> List[Dict[str, Any]]:
    # Ordered pipeline steps for UI.
    return [
        {"id": "upload_to_s3", "label": "Upload to S3", "status": "not_started", "percent": 0, "message": None},
        {"id": "mediaconvert_proxy", "label": "MediaConvert proxy", "status": "not_started", "percent": 0, "message": None},
        {"id": "rekognition_labels", "label": "Rekognition: labels", "status": "not_started", "percent": 0, "message": None},
        {"id": "rekognition_celebrities", "label": "Rekognition: celebrities", "status": "not_started", "percent": 0, "message": None},
        {"id": "content_moderation", "label": "Moderation", "status": "not_started", "percent": 0, "message": None},
        {"id": "rekognition_text", "label": "Rekognition: text", "status": "not_started", "percent": 0, "message": None},
        {"id": "rekognition_shots", "label": "Rekognition: shots", "status": "not_started", "percent": 0, "message": None},
        {"id": "transcribe", "label": "Transcribe", "status": "not_started", "percent": 0, "message": None},
        {"id": "embedding", "label": "Embedding", "status": "not_started", "percent": 0, "message": None},
        {"id": "saving_indexing", "label": "Save & index", "status": "not_started", "percent": 0, "message": None},
    ]


def _job_step_update(
    job_id: str,
    step_id: str,
    *,
    status: str | None = None,
    percent: int | None = None,
    message: str | None = None,
) -> None:
    # status: not_started | running | completed | failed | skipped
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
            try:
                p = int(percent)
            except Exception:
                p = None
            if p is not None:
                step_obj["percent"] = max(0, min(100, p))
        if message is not None:
            step_obj["message"] = message

        job["updated_at"] = datetime.utcnow().isoformat()


def _envid_metadata_output_profile() -> str:
    """Return the output profile for envidMetadata.

    - `required` (default): emit only the UX-required artifacts (celebs timeline+bios, content list,
      subtitles, locations, technical, synopses).
    - `full`: preserve the legacy full metadata-json shape.
    """

    return (os.getenv("ENVID_METADATA_OUTPUT_PROFILE") or "required").strip().lower()


def _disable_per_frame_analysis() -> bool:
    """If true, do not run the legacy per-frame Rekognition Image pipeline.

    Rekognition analysis will run via Rekognition Video against the S3 object.
    Local decoding (ffmpeg) may still be used for thumbnails/cropping only.
    """

    return _env_truthy(os.getenv("ENVID_METADATA_DISABLE_PER_FRAME_ANALYSIS"), default=False)


def _require_s3_upload_for_rekognition() -> bool:
    return _env_truthy(os.getenv("ENVID_METADATA_REQUIRE_S3_UPLOAD_FOR_REKOGNITION"), default=True)


def _persist_local_video_copy() -> bool:
    """Whether to persist the uploaded MP4 under VIDEOS_DIR.

    When false, the service becomes "S3-only" for the video bytes: it will still
    save derived artifacts (metadata JSON, thumbnails in index, subtitles, etc.).
    Playback will redirect to a presigned S3 URL.
    """

    return _env_truthy(os.getenv("ENVID_METADATA_PERSIST_LOCAL_VIDEO"), default=False)


def _presign_s3_get_object_url(*, bucket: str, key: str) -> str:
    bucket = (bucket or "").strip()
    key = (key or "").strip().lstrip("/")
    if not bucket or not key:
        raise ValueError("Missing bucket/key for presign")

    bucket_region = _detect_bucket_region(bucket)
    s3_region = bucket_region or DEFAULT_AWS_REGION
    client = _s3_client_for_transfer(region_name=s3_region)
    expires = _parse_int_param(
        os.getenv("ENVID_METADATA_S3_PRESIGN_SECONDS"),
        default=3600,
        min_value=60,
        max_value=86400,
    )
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )


def _looks_like_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except Exception:
        return False


SUBTITLES_DIR = None  # set after INDICES_DIR is defined


def _format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round((seconds - int(seconds)) * 1000.0))
    total = int(seconds)
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_vtt_timestamp(seconds: float) -> str:
    # WebVTT uses '.' for milliseconds
    if seconds < 0:
        seconds = 0.0
    ms = int(round((seconds - int(seconds)) * 1000.0))
    total = int(seconds)
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    idx = 1
    for seg in segments:
        try:
            start = float(seg.get("start") or 0.0)
            end = float(seg.get("end") or start)
        except Exception:
            continue
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
    out: List[str] = ["WEBVTT", ""]
    for seg in segments:
        try:
            start = float(seg.get("start") or 0.0)
            end = float(seg.get("end") or start)
        except Exception:
            continue
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        out.append(f"{_format_vtt_timestamp(start)} --> {_format_vtt_timestamp(end)}")
        out.append(text)
        out.append("")
    return "\n".join(out).strip() + "\n"


def _bio_cache_path() -> Path:
    return INDICES_DIR / "celebrity_bio_cache.json"


def _bio_cache_load() -> dict[str, Any]:
    try:
        p = _bio_cache_path()
        if not p.exists():
            return {}
        with open(p, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _bio_cache_save(cache: dict[str, Any]) -> None:
    try:
        tmp = _bio_cache_path().with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        tmp.replace(_bio_cache_path())
    except Exception:
        pass


def _word_count(text: str) -> int:
    try:
        s = str(text or "")
    except Exception:
        return 0
    # Count "word-like" tokens.
    return len(re.findall(r"\b[\w'-]+\b", s))


def _truncate_to_word_count(text: str, *, max_words: int) -> str:
    try:
        s = str(text or "").strip()
    except Exception:
        return ""
    if not s:
        return ""
    if max_words <= 0:
        return ""
    words = re.findall(r"\b[\w'-]+\b", s)
    if len(words) <= max_words:
        return s
    # Approximate truncation by walking through the string.
    # We stop after max_words word matches and cut at that position.
    it = list(re.finditer(r"\b[\w'-]+\b", s))
    if len(it) <= max_words:
        return s
    end_pos = it[max_words - 1].end()
    return (s[:end_pos].rstrip() + " …").strip()


def _bio_is_too_short(text: str | None) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    return _word_count(s) < 30


def _refresh_celebrity_bios_in_payload(payload: Any, *, include_frames: bool = True) -> bool:
    """Best-effort: ensure any celebrity bios present are >= 30 words.

    - If a bio/bio_short is missing or < 30 words, attempt to fill it from the local cache;
    if still missing/short, regenerate via Wikipedia (via `_celebrity_bios_for_names`).
    - If we cannot get a >=30-word bio, we clear any too-short `bio_short` so the UI won't
      prefer it over a longer `bio`.

    Returns True if payload was modified.
    """

    if not payload:
        return False

    # Collect celebrity dict references from common shapes.
    celebs: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    def _maybe_add_celeb(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        if not (obj.get("name") or "").strip():
            return
        oid = id(obj)
        if oid in seen_ids:
            return
        seen_ids.add(oid)
        celebs.append(obj)

    def _walk(obj: Any, *, depth: int) -> None:
        if depth <= 0:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in {"celebrities", "celebrities_detailed"} and isinstance(v, list):
                    for it in v:
                        _maybe_add_celeb(it)
                    continue
                if include_frames and k == "frames" and isinstance(v, list):
                    for f in v:
                        _walk(f, depth=depth - 1)
                    continue
                _walk(v, depth=depth - 1)
        elif isinstance(obj, list):
            for it in obj:
                _walk(it, depth=depth - 1)

    _walk(payload, depth=6)
    if not celebs:
        return False

    # Determine which names need regeneration.
    names_needing: list[str] = []
    for c in celebs:
        nm = (c.get("name") or "").strip()
        if not nm:
            continue
        # UI prefers bio_short; treat either as satisfying the requirement.
        current = (c.get("bio_short") or c.get("bio") or "").strip()
        if _bio_is_too_short(current):
            names_needing.append(nm)

    names_needing = sorted(list(dict.fromkeys(names_needing)))
    if not names_needing:
        return False

    bios = _celebrity_bios_for_names(names_needing)
    changed = False

    for c in celebs:
        nm = (c.get("name") or "").strip()
        if not nm or nm not in names_needing:
            continue

        bio_obj = bios.get(nm) if isinstance(bios, dict) else None
        new_bio = (bio_obj.get("bio") or "").strip() if isinstance(bio_obj, dict) else ""

        # If we can provide a sufficiently long bio, apply it.
        if not _bio_is_too_short(new_bio):
            if (c.get("bio") or "").strip() != new_bio:
                c["bio"] = new_bio
                changed = True
            src = (bio_obj.get("source") or "").strip() if isinstance(bio_obj, dict) else ""
            if src and (c.get("bio_source") or "").strip() != src:
                c["bio_source"] = src
                changed = True

        # If we still can't get a >=30-word bio, ensure we never return a too-short one.
        if _bio_is_too_short(new_bio):
            if "bio" in c and _bio_is_too_short((c.get("bio") or "").strip()):
                c["bio"] = None
                changed = True
            if "bio_source" in c and (c.get("bio_source") is not None):
                c["bio_source"] = None
                changed = True

        # Ensure the UI doesn't prefer a too-short bio_short.
        if "bio_short" in c and _bio_is_too_short((c.get("bio_short") or "").strip()):
            c["bio_short"] = None
            changed = True

        # Opportunistically backfill portraits too (same cache object).
        if isinstance(bio_obj, dict):
            for k in [
                "portrait_url",
                "portrait_source",
                "portrait_license",
                "portrait_license_url",
                "portrait_attribution",
            ]:
                v = (bio_obj.get(k) or "").strip() if isinstance(bio_obj.get(k), str) else bio_obj.get(k)
                if v and not (c.get(k) or "").strip():
                    c[k] = v
                    changed = True

    return changed


def _wikipedia_short_bio(name: str) -> dict[str, Any] | None:
    """Best-effort: fetch a short bio from Wikipedia's REST summary endpoint.

    Returns {"bio": str, "source": str} or None.
    """

    n = (name or "").strip()
    if not n:
        return None
    title = urllib.parse.quote(n.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "envid-metadata/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
        extract = (payload.get("extract") or "").strip()
        if not extract:
            return None
        # Prefer enough content to meet a minimum word count.
        # Wikipedia extracts can be long; we add sentences until we have enough.
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", extract) if (p or "").strip()]
        bio_parts: list[str] = []
        for p in parts[:8]:
            bio_parts.append(p)
            bio = " ".join(bio_parts).strip()
            if _word_count(bio) >= 30:
                break
        bio = " ".join(bio_parts).strip()
        bio = _truncate_to_word_count(bio, max_words=90)
        if not bio:
            return None
        if _word_count(bio) < 30:
            return None
        return {"bio": bio, "source": payload.get("content_urls", {}).get("desktop", {}).get("page") or url}
    except Exception:
        return None


def _wikidata_entity_id_for_name(name: str) -> str | None:
    n = (name or "").strip()
    if not n:
        return None

    ua = {"User-Agent": "envid-metadata/1.0"}
    timeout_s = float(os.getenv("ENVID_METADATA_WIKIDATA_TIMEOUT_SECONDS") or 6.0)
    timeout_s = max(2.0, min(20.0, timeout_s))

    try:
        qs = urllib.parse.urlencode(
            {
                "action": "wbsearchentities",
                "search": n,
                "language": "en",
                "format": "json",
                "limit": 5,
            }
        )
        search_url = f"https://www.wikidata.org/w/api.php?{qs}"
        req = urllib.request.Request(search_url, headers=ua)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
        results = payload.get("search") or []
        if not isinstance(results, list) or not results:
            return None

        for r in results:
            if not isinstance(r, dict):
                continue
            rid = (r.get("id") or "").strip()
            if rid.startswith("Q"):
                return rid
        return None
    except Exception:
        return None


def _wikidata_entity_data(entity_id: str) -> dict[str, Any] | None:
    eid = (entity_id or "").strip()
    if not eid or not eid.startswith("Q"):
        return None

    ua = {"User-Agent": "envid-metadata/1.0"}
    timeout_s = float(os.getenv("ENVID_METADATA_WIKIDATA_TIMEOUT_SECONDS") or 6.0)
    timeout_s = max(2.0, min(20.0, timeout_s))

    try:
        entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{eid}.json"
        req = urllib.request.Request(entity_url, headers=ua)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            ed = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
        entities = (ed.get("entities") or {})
        ent = entities.get(eid) if isinstance(entities, dict) else None
        return ent if isinstance(ent, dict) else None
    except Exception:
        return None


def _wikidata_labels_for_entity_ids(entity_ids: list[str]) -> dict[str, str]:
    ids = [i.strip() for i in (entity_ids or []) if isinstance(i, str) and i.strip().startswith("Q")]
    ids = list(dict.fromkeys(ids))
    if not ids:
        return {}

    ua = {"User-Agent": "envid-metadata/1.0"}
    timeout_s = float(os.getenv("ENVID_METADATA_WIKIDATA_TIMEOUT_SECONDS") or 6.0)
    timeout_s = max(2.0, min(20.0, timeout_s))

    try:
        qs = urllib.parse.urlencode(
            {
                "action": "wbgetentities",
                "ids": "|".join(ids[:50]),
                "props": "labels",
                "languages": "en",
                "format": "json",
            }
        )
        url = f"https://www.wikidata.org/w/api.php?{qs}"
        req = urllib.request.Request(url, headers=ua)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
        entities = payload.get("entities") if isinstance(payload, dict) else None
        if not isinstance(entities, dict):
            return {}

        out: dict[str, str] = {}
        for qid, ent in entities.items():
            if not isinstance(ent, dict):
                continue
            labels = ent.get("labels") if isinstance(ent.get("labels"), dict) else None
            en = labels.get("en") if isinstance(labels, dict) else None
            val = (en.get("value") or "").strip() if isinstance(en, dict) else ""
            if val:
                out[str(qid)] = val
        return out
    except Exception:
        return {}


def _wikidata_short_bio(name: str) -> dict[str, Any] | None:
    """Best-effort: generate a >=30-word bio from Wikidata description + common claims.

    Returns {"bio": str, "source": str} or None.
    """

    n = (name or "").strip()
    if not n:
        return None

    entity_id = _wikidata_entity_id_for_name(n)
    if not entity_id:
        return None

    ent = _wikidata_entity_data(entity_id)
    if not ent:
        return None

    desc = ""
    descriptions = ent.get("descriptions") if isinstance(ent.get("descriptions"), dict) else None
    if isinstance(descriptions, dict):
        en = descriptions.get("en") if isinstance(descriptions.get("en"), dict) else None
        desc = (en.get("value") or "").strip() if isinstance(en, dict) else ""

    claims = ent.get("claims") if isinstance(ent.get("claims"), dict) else None
    if not isinstance(claims, dict):
        claims = {}

    def _entity_ids_from_claim(prop: str, *, limit: int) -> list[str]:
        vals: list[str] = []
        arr = claims.get(prop)
        if not isinstance(arr, list):
            return vals
        for it in arr:
            if not isinstance(it, dict):
                continue
            mainsnak = it.get("mainsnak") if isinstance(it.get("mainsnak"), dict) else None
            datavalue = mainsnak.get("datavalue") if isinstance(mainsnak, dict) else None
            v = datavalue.get("value") if isinstance(datavalue, dict) else None
            qid = (v.get("id") or "").strip() if isinstance(v, dict) else ""
            if qid.startswith("Q"):
                vals.append(qid)
            if len(vals) >= int(limit):
                break
        return vals

    occ_ids = _entity_ids_from_claim("P106", limit=4)  # occupation
    work_ids = _entity_ids_from_claim("P800", limit=3)  # notable work
    award_ids = _entity_ids_from_claim("P166", limit=2)  # award received

    labels = _wikidata_labels_for_entity_ids(occ_ids + work_ids + award_ids)
    occupations = [labels.get(q) for q in occ_ids if labels.get(q)]
    works = [labels.get(q) for q in work_ids if labels.get(q)]
    awards = [labels.get(q) for q in award_ids if labels.get(q)]

    parts: list[str] = []
    if desc:
        parts.append(f"{n} is {desc}.")
    if occupations:
        parts.append(f"Occupations associated with {n} include {', '.join(occupations[:4])}.")
    if works:
        parts.append(f"Notable works include {', '.join(works[:3])}.")
    if awards:
        parts.append(f"Awards include {', '.join(awards[:2])}.")

    bio = " ".join([p.strip() for p in parts if p.strip()]).strip()
    bio = _truncate_to_word_count(bio, max_words=90)
    if not bio or _word_count(bio) < 30:
        return None
    return {"bio": bio, "source": f"https://www.wikidata.org/wiki/{entity_id}"}


def _wikidata_commons_portrait(name: str) -> dict[str, Any] | None:
    """Best-effort: fetch a portrait image URL from Wikidata (P18) via Wikimedia Commons.

    Returns a dict like:
      {
        "portrait_url": "https://.../File.jpg",
        "portrait_source": "https://www.wikidata.org/wiki/Q...",
        "portrait_license": "CC BY-SA 4.0" (best-effort),
        "portrait_license_url": "https://..." (best-effort),
        "portrait_attribution": "..." (best-effort)
      }

    Note: LLMs can generate text, but they cannot reliably provide image
    assets; Wikidata/Commons is the stable source for public, licensed portraits.
    """

    n = (name or "").strip()
    if not n:
        return None

    ua = {"User-Agent": "envid-metadata/1.0"}
    timeout_s = float(os.getenv("ENVID_METADATA_WIKIDATA_TIMEOUT_SECONDS") or 6.0)
    timeout_s = max(2.0, min(20.0, timeout_s))

    try:
        # 1) Find a Wikidata entity by label.
        qs = urllib.parse.urlencode(
            {
                "action": "wbsearchentities",
                "search": n,
                "language": "en",
                "format": "json",
                "limit": 5,
            }
        )
        search_url = f"https://www.wikidata.org/w/api.php?{qs}"
        req = urllib.request.Request(search_url, headers=ua)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
        results = payload.get("search") or []
        if not isinstance(results, list) or not results:
            return None

        entity_id = None
        for r in results:
            if not isinstance(r, dict):
                continue
            rid = (r.get("id") or "").strip()
            if rid.startswith("Q"):
                entity_id = rid
                break
        if not entity_id:
            return None

        # 2) Pull entity JSON and read P18 (image).
        entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
        req = urllib.request.Request(entity_url, headers=ua)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            ed = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
        entities = (ed.get("entities") or {})
        ent = entities.get(entity_id) if isinstance(entities, dict) else None
        claims = ent.get("claims") if isinstance(ent, dict) else None
        p18 = (claims or {}).get("P18") if isinstance(claims, dict) else None
        if not isinstance(p18, list) or not p18:
            return None
        mainsnak = (p18[0] or {}).get("mainsnak") if isinstance(p18[0], dict) else None
        datavalue = (mainsnak or {}).get("datavalue") if isinstance(mainsnak, dict) else None
        filename = (datavalue or {}).get("value") if isinstance(datavalue, dict) else None
        if not isinstance(filename, str) or not filename.strip():
            return None

        # 3) Resolve a stable, hotlinkable URL + license metadata from Commons.
        # Commons API expects title "File:..." (spaces are okay).
        file_title = "File:" + filename
        qs2 = urllib.parse.urlencode(
            {
                "action": "query",
                "titles": file_title,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata",
                "format": "json",
            }
        )
        commons_url = f"https://commons.wikimedia.org/w/api.php?{qs2}"
        req = urllib.request.Request(commons_url, headers=ua)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            cp = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
        pages = ((cp.get("query") or {}).get("pages") or {})
        page = None
        if isinstance(pages, dict) and pages:
            page = next(iter(pages.values()))
        imageinfo = (page or {}).get("imageinfo") if isinstance(page, dict) else None
        if not isinstance(imageinfo, list) or not imageinfo:
            return None
        info0 = imageinfo[0] if isinstance(imageinfo[0], dict) else {}
        url = (info0.get("url") or "").strip()
        if not url:
            return None
        ext = info0.get("extmetadata") if isinstance(info0.get("extmetadata"), dict) else {}

        def _ext_val(key: str) -> str | None:
            v = ext.get(key) if isinstance(ext, dict) else None
            if isinstance(v, dict):
                vv = v.get("value")
                return str(vv).strip() if vv is not None else None
            return None

        license_short = _ext_val("LicenseShortName")
        license_url = _ext_val("LicenseUrl")
        artist = _ext_val("Artist")
        credit = _ext_val("Credit")
        attribution = (credit or artist or "").strip() or None

        return {
            "portrait_url": url,
            "portrait_source": f"https://www.wikidata.org/wiki/{entity_id}",
            "portrait_license": license_short,
            "portrait_license_url": license_url,
            "portrait_attribution": attribution,
        }
    except Exception:
        return None


def _celebrity_bios_for_names(names: List[str]) -> dict[str, Any]:
    cache = _bio_cache_load()
    out: dict[str, Any] = {}
    changed = False
    for raw in names:
        name = (raw or "").strip()
        if not name:
            continue
        cached = cache.get(name)
        cached_bio = (cached.get("bio") or "").strip() if isinstance(cached, dict) else ""
        cached_portrait = (cached.get("portrait_url") or "").strip() if isinstance(cached, dict) else ""

        # If we have a cached bio but it's too short, regenerate.
        if cached_bio and _word_count(cached_bio) < 30:
            cached_bio = ""

        bio_obj = None
        if not cached_bio:
            bio_obj = _wikipedia_short_bio(name) or _wikidata_short_bio(name)

        portrait_obj = None
        if not cached_portrait and _env_truthy(os.getenv("ENVID_METADATA_ENABLE_WIKIDATA_PORTRAITS"), default=True):
            portrait_obj = _wikidata_commons_portrait(name)

        if isinstance(cached, dict):
            merged = dict(cached)
        else:
            merged = {}
        if isinstance(bio_obj, dict):
            merged.update(bio_obj)
        if isinstance(portrait_obj, dict):
            merged.update(portrait_obj)

        # Never persist or return too-short bios.
        merged_bio = (merged.get("bio") or "").strip() if isinstance(merged, dict) else ""
        if merged_bio and _word_count(merged_bio) < 30:
            merged.pop("bio", None)
            merged.pop("source", None)

        if merged:
            out[name] = merged
            if merged != cached:
                cache[name] = merged
                changed = True
    if changed:
        _bio_cache_save(cache)
    return out


def _http_get_bytes(url: str, *, timeout_s: int = 12, max_bytes: int = 4_900_000) -> bytes | None:
    u = (url or "").strip()
    if not u:
        return None
    if not (u.startswith("http://") or u.startswith("https://")):
        return None
    try:
        req = urllib.request.Request(
            u,
            headers={
                "User-Agent": "MediaGenAI/1.0 (envid-metadata)",
                "Accept": "image/*,*/*;q=0.8",
            },
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=int(timeout_s)) as resp:
            # Limit read to prevent huge Wikimedia originals.
            data = resp.read(int(max_bytes) + 1)
        if not data or len(data) > int(max_bytes):
            return None
        return data
    except Exception:
        return None


def _normalize_image_bytes_for_rekognition(image_bytes: bytes, *, max_edge_px: int = 1600, quality: int = 90) -> bytes | None:
    """Best-effort: normalize arbitrary image bytes into a Rekognition-friendly JPEG.

    Rekognition APIs can fail with InvalidParameterException if the bytes aren't a supported image
    (e.g., HTML error pages) or if the encoding is unusual. This also helps cap huge images.
    """

    if not image_bytes:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            w, h = img.size
            if w <= 0 or h <= 0:
                return None
            max_edge = max(w, h)
            if max_edge > int(max_edge_px) and int(max_edge_px) > 0:
                scale = float(max_edge_px) / float(max_edge)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=int(quality), optimize=True)
            data = out.getvalue()
            return data if data else None
    except Exception:
        return None


def _rekognition_detect_largest_face_bbox(rk_client: Any, image_bytes: bytes) -> Dict[str, Any] | None:
    """Detect faces and return the largest face bounding box (normalized Rekognition format)."""

    if not image_bytes:
        return None
    try:
        resp = rk_client.detect_faces(Image={"Bytes": image_bytes}, Attributes=["DEFAULT"])
    except Exception:
        return None
    best_bbox: Dict[str, Any] | None = None
    best_area = -1.0
    for fd in (resp.get("FaceDetails") or []):
        if not isinstance(fd, dict):
            continue
        bbox = fd.get("BoundingBox") if isinstance(fd.get("BoundingBox"), dict) else None
        if not isinstance(bbox, dict) or not bbox:
            continue
        try:
            area = float(bbox.get("Width") or 0.0) * float(bbox.get("Height") or 0.0)
        except Exception:
            area = 0.0
        if area > best_area:
            best_area = area
            best_bbox = bbox
    return best_bbox


def _comprehend_locations_from_text(text: str, *, language_code: str | None) -> List[Dict[str, Any]]:
    """Extract location entities from transcript text (best-effort)."""

    t = (text or "").strip()
    if not t:
        return []
    # Comprehend limit is 5KB.
    t = t[:4800]
    lang = (language_code or "").strip() or "en"
    # If we don't have a supported code, fall back to English.
    if len(lang) != 2:
        lang = "en"
    try:
        resp = comprehend.detect_entities(Text=t, LanguageCode=lang)
        ents = resp.get("Entities") or []
        out: List[Dict[str, Any]] = []
        for e in ents:
            if (e.get("Type") or "").upper() != "LOCATION":
                continue
            txt = (e.get("Text") or "").strip()
            if not txt:
                continue
            try:
                score = float(e.get("Score") or 0.0)
            except Exception:
                score = 0.0
            out.append({"name": txt, "score": score})
        # Dedup by name, keep max score.
        best: dict[str, float] = {}
        for r in out:
            n = r.get("name")
            if not n:
                continue
            best[n] = max(best.get(n, 0.0), float(r.get("score") or 0.0))
        ranked = [{"name": k, "score": v} for k, v in best.items()]
        ranked.sort(key=lambda x: (x.get("score") or 0.0, x.get("name") or ""), reverse=True)
        return ranked[:25]
    except Exception:
        return []


def _bedrock_synopsis_by_age_group(text: str, *, title: str | None) -> Dict[str, Any]:
    """Generate synopses per age group using Bedrock (best-effort)."""

    def _word_count(s: str) -> int:
        try:
            return len([w for w in re.split(r"\s+", (s or "").strip()) if w])
        except Exception:
            return 0

    def _first_n_words(source: str, n: int) -> str:
        words = [w for w in re.split(r"\s+", (source or "").strip()) if w]
        if not words:
            return ""
        return " ".join(words[: max(0, int(n))]).strip()

    def _ensure_min_words(value: str, *, min_words: int, source_text: str) -> str:
        v = re.sub(r"\s+", " ", (value or "").strip())
        if not v:
            return _first_n_words(source_text, min_words)
        if _word_count(v) >= int(min_words):
            return v
        # Expand deterministically using only provided content.
        need = max(0, int(min_words) - _word_count(v))
        extra = _first_n_words(source_text, need + 20)  # a bit extra to overcome overlaps/punctuation
        extra = extra.strip()
        if extra:
            v = (v.rstrip() + " " + extra).strip()
        # If still short (very small source), repeat source-derived content.
        while _word_count(v) < int(min_words) and (source_text or "").strip():
            v = (v.rstrip() + " " + _first_n_words(source_text, min_words)).strip()
            if len(v) > 6000:
                break
        return v

    def _coerce_output(data: Dict[str, Any], *, source_text: str) -> Dict[str, Any]:
        required_keys = ["kids", "teens", "adults", "senior_citizen"]
        out: Dict[str, Any] = {}
        for k in required_keys:
            grp = data.get(k) if isinstance(data.get(k), dict) else {}
            short_v = str((grp or {}).get("short") or "").strip()
            long_v = str((grp or {}).get("long") or "").strip()
            out[k] = {
                "short": _ensure_min_words(short_v, min_words=30, source_text=source_text),
                "long": _ensure_min_words(long_v, min_words=70, source_text=source_text),
            }
        return out

    def _fallback() -> Dict[str, Any]:
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        if not cleaned:
            cleaned = re.sub(r"\s+", " ", (title or "").strip())
        if not cleaned:
            return {}

        base_short = _ensure_min_words("This video is about: " + cleaned, min_words=30, source_text=cleaned)
        base_long = _ensure_min_words("Synopsis: " + cleaned, min_words=70, source_text=cleaned)
        return {
            "kids": {
                "short": base_short,
                "long": base_long,
            },
            "teens": {
                "short": base_short,
                "long": base_long,
            },
            "adults": {
                "short": base_short,
                "long": base_long,
            },
            "senior_citizen": {
                "short": base_short,
                "long": base_long,
            },
        }

    if not (text or "").strip():
        return {}

    # Keep the prompt short-ish.
    src = (text or "").strip()
    src = src[:6000]

    prompt = (
        "You are generating movie/clip synopses for different age groups. "
        "Use only the provided content; do not invent character names or plot points not implied by the text. "
        "Return STRICT JSON with keys: kids, teens, adults, senior_citizen. "
        "Each contains: short (>=30 words) and long (>=70 words). "
        "Tone: kids=very simple and safe wording, teens=neutral, adults=more detailed, senior_citizen=clear and easy-to-follow.\n\n"
        f"TITLE: {(title or '').strip()}\n"
        f"CONTENT: {src}"
    )
    try:
        payload = _bedrock_summary(prompt, title=title)
        # _bedrock_summary returns {summary, ssml}; summary may be JSON or plain.
        summary = (payload.get("summary") or "").strip()
        if not summary:
            return _fallback()
        try:
            data = json.loads(summary)
            if not isinstance(data, dict):
                return _fallback()
            return _coerce_output(data, source_text=src)
        except Exception:
            return _fallback()
    except Exception:
        return _fallback()


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    try:
        s = max(float(a_start), float(b_start))
        e = min(float(a_end), float(b_end))
        return max(0.0, e - s)
    except Exception:
        return 0.0


def _scene_windows(*, duration_seconds: float | None, window_seconds: int) -> List[Dict[str, Any]]:
    if not duration_seconds or duration_seconds <= 0:
        return []
    win = max(5, int(window_seconds))
    out: List[Dict[str, Any]] = []
    t = 0
    idx = 1
    dur = int(duration_seconds)
    while t < max(1, dur):
        start = int(t)
        end = int(min(dur, t + win))
        out.append({"scene_index": idx, "start_seconds": start, "end_seconds": end})
        t += win
        idx += 1
        if idx > 2000:
            break
    return out


def _rekognition_shot_segments_to_scene_windows(
    *,
    shot_segments_raw: list[dict[str, Any]] | None,
    duration_seconds: float | None,
    target_window_seconds: int,
    max_scenes: int = 2000,
) -> List[Dict[str, Any]]:
    """Convert Rekognition Video SHOT segments into merged scene windows.

    Rekognition's SegmentDetection returns shot boundaries which can be very granular.
    We merge consecutive shots to create human-sized scene windows roughly matching
    `target_window_seconds`.
    """

    if not duration_seconds or duration_seconds <= 0:
        return []

    win = max(5, int(target_window_seconds))
    max_scenes = max(1, int(max_scenes))

    shots: List[Tuple[int, int, float]] = []
    for it in shot_segments_raw or []:
        if not isinstance(it, dict):
            continue
        seg = it.get("ShotSegment") if isinstance(it.get("ShotSegment"), dict) else None
        if not seg:
            continue
        try:
            start_ms = int(seg.get("StartTimestampMillis") or 0)
            end_ms = int(seg.get("EndTimestampMillis") or 0)
        except Exception:
            continue
        if end_ms <= start_ms:
            continue
        try:
            conf = float(seg.get("Confidence") or 0.0)
        except Exception:
            conf = 0.0
        start_s = max(0, int(start_ms // 1000))
        end_s = max(start_s + 1, int((end_ms + 999) // 1000))
        shots.append((start_s, end_s, conf))

    if not shots:
        return []

    shots.sort(key=lambda x: (x[0], x[1]))
    # Clamp to duration.
    dur = max(1, int(duration_seconds))
    shots2: List[Tuple[int, int, float]] = []
    for s0, e0, c0 in shots:
        s = max(0, min(dur - 1, int(s0)))
        e = max(s + 1, min(dur, int(e0)))
        if e > s:
            shots2.append((s, e, c0))
    if not shots2:
        return []

    out: List[Dict[str, Any]] = []
    cur_start = shots2[0][0]
    cur_end = shots2[0][1]
    idx = 1
    for s, e, _c in shots2[1:]:
        # Ensure monotonic windows.
        if s < cur_start:
            s = cur_start
        if e <= s:
            continue

        # Extend current window.
        cur_end = max(cur_end, e)

        # If the window is large enough, cut it.
        if (cur_end - cur_start) >= win:
            out.append({"scene_index": idx, "start_seconds": int(cur_start), "end_seconds": int(cur_end)})
            idx += 1
            if idx > max_scenes:
                return out
            cur_start = cur_end
            cur_end = max(cur_start + 1, e)

    # Final scene (ensure coverage to duration if close)
    if cur_start < dur:
        end_final = max(cur_end, min(dur, cur_start + 1))
        out.append({"scene_index": idx, "start_seconds": int(cur_start), "end_seconds": int(end_final)})

    # Remove any degenerate/overlapping scenes after merges.
    cleaned: List[Dict[str, Any]] = []
    last_end = -1
    for sc in out:
        try:
            s = int(sc.get("start_seconds") or 0)
            e = int(sc.get("end_seconds") or 0)
        except Exception:
            continue
        if e <= s:
            continue
        if s < last_end:
            s = last_end
        if e <= s:
            continue
        cleaned.append({"scene_index": len(cleaned) + 1, "start_seconds": s, "end_seconds": e})
        last_end = e
        if len(cleaned) >= max_scenes:
            break
    return cleaned


def _build_scene_by_scene_metadata(
    *,
    frames: List[Dict[str, Any]],
    transcript_segments: List[Dict[str, Any]],
    duration_seconds: float | None,
    scenes: List[Dict[str, Any]] | None = None,
    source: str | None = None,
    window_seconds: int = 15,
) -> Dict[str, Any]:
    """Build scene-by-scene metadata on either fixed windows or Rekognition-derived scenes.

    This is intentionally simple and deterministic: it ensures everything we emit is time-mapped,
    even without expensive shot-boundary detection.
    """

    resolved_source = (source or "fixed_windows").strip() or "fixed_windows"
    resolved_scenes = scenes if scenes is not None else _scene_windows(duration_seconds=duration_seconds, window_seconds=window_seconds)
    if not resolved_scenes:
        return {"source": resolved_source, "window_seconds": window_seconds, "scenes": [], "high_points": [], "key_scenes": []}

    def _frames_in_window(start_s: int, end_s: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for f in frames or []:
            ts = f.get("timestamp")
            if ts is None:
                continue
            try:
                tsv = int(ts)
            except Exception:
                continue
            if start_s <= tsv < end_s:
                out.append(f)
        return out

    def _segments_in_window(start_s: int, end_s: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in transcript_segments or []:
            try:
                ss = float(s.get("start") or 0.0)
                se = float(s.get("end") or ss)
            except Exception:
                continue
            if _overlap_seconds(ss, se, float(start_s), float(end_s)) > 0:
                out.append(s)
        return out

    scene_objs: List[Dict[str, Any]] = []
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for scene in resolved_scenes:
        start_s = int(scene["start_seconds"])
        end_s = int(scene["end_seconds"])
        fwin = _frames_in_window(start_s, end_s)
        swin = _segments_in_window(start_s, end_s)

        celeb_names: List[str] = []
        label_names: List[str] = []
        mod_names: List[str] = []
        text_lines: List[str] = []

        for f in fwin:
            for c in (f.get("celebrities") or []):
                nm = (c.get("name") or "").strip()
                if nm:
                    celeb_names.append(nm)
            for l in (f.get("labels") or []):
                nm = (l.get("name") or "").strip()
                if nm:
                    label_names.append(nm)
            for m in (f.get("moderation") or []):
                nm = (m.get("name") or "").strip()
                if nm:
                    mod_names.append(nm)
            for t in (f.get("text") or []):
                if (t.get("type") or "") != "LINE":
                    continue
                tx = (t.get("text") or "").strip()
                if tx:
                    text_lines.append(tx)

        celeb_unique = sorted(list(dict.fromkeys(celeb_names)))
        labels_unique = sorted(list(dict.fromkeys(label_names)))
        mods_unique = sorted(list(dict.fromkeys(mod_names)))
        text_unique = sorted(list(dict.fromkeys(text_lines)))

        transcript_snippet = " ".join([(s.get("text") or "").strip() for s in swin if (s.get("text") or "").strip()])
        transcript_snippet = re.sub(r"\s+", " ", transcript_snippet).strip()

        # Lightweight, deterministic score for picking highlights.
        score = 0.0
        score += 4.0 * float(len(celeb_unique))
        score += 0.3 * float(len(labels_unique))
        score += 1.5 * float(len(mods_unique))
        score += 0.1 * float(len(text_unique))
        score += 0.02 * float(len(transcript_snippet))

        obj = {
            "scene_index": scene["scene_index"],
            "start_seconds": start_s,
            "end_seconds": end_s,
            "summary_text": (transcript_snippet[:500] or None),
            "celebrities": celeb_unique[:25],
            "labels": labels_unique[:40],
            "moderation_labels": mods_unique[:30],
            "on_screen_text": text_unique[:30],
            "transcript_segments": [
                {
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "text": s.get("text"),
                    "speaker": s.get("speaker"),
                }
                for s in swin[:25]
            ],
        }
        scene_objs.append(obj)
        scored.append((score, obj))

    scored.sort(key=lambda x: x[0], reverse=True)
    high_points = []
    for score, obj in scored[:8]:
        reason_bits = []
        if obj.get("celebrities"):
            reason_bits.append("celebrity")
        if obj.get("moderation_labels"):
            reason_bits.append("moderation")
        if obj.get("labels"):
            reason_bits.append("labels")
        if obj.get("summary_text"):
            reason_bits.append("dialogue")
        high_points.append(
            {
                "start_seconds": obj.get("start_seconds"),
                "end_seconds": obj.get("end_seconds"),
                "scene_index": obj.get("scene_index"),
                "score": float(score),
                "reason": ", ".join(reason_bits) or "activity",
            }
        )

    key_scenes = [
        {
            "start_seconds": s.get("start_seconds"),
            "end_seconds": s.get("end_seconds"),
            "scene_index": s.get("scene_index"),
            "key_reason": "celebrity" if (s.get("celebrities") or []) else ("content" if (s.get("moderation_labels") or []) else "scene"),
        }
        for s in scene_objs
        if (s.get("celebrities") or []) or (s.get("moderation_labels") or [])
    ][:20]

    return {
        "source": resolved_source,
        "window_seconds": int(window_seconds),
        "scenes": scene_objs,
        "high_points": high_points,
        "key_scenes": key_scenes,
    }


def _attach_celebrity_timestamps(
    *,
    frames: List[Dict[str, Any]],
    celebrities_detailed: List[Dict[str, Any]],
    max_timestamps_per_celebrity: int = 60,
) -> List[Dict[str, Any]]:
    if not celebrities_detailed:
        return celebrities_detailed

    by_name: Dict[str, List[int]] = {}
    for f in frames or []:
        ts = f.get("timestamp")
        if ts is None:
            continue
        try:
            tsv = int(ts)
        except Exception:
            continue
        for c in (f.get("celebrities") or []):
            nm = (c.get("name") or "").strip()
            if not nm:
                continue
            by_name.setdefault(nm, []).append(tsv)

    out: List[Dict[str, Any]] = []
    for c in celebrities_detailed:
        nm = (c.get("name") or "").strip()
        if not nm:
            out.append(c)
            continue
        ts_list = sorted(list(dict.fromkeys(by_name.get(nm) or [])))
        c2 = dict(c)
        c2["timestamps_seconds"] = ts_list[: int(max_timestamps_per_celebrity)]
        out.append(c2)
    return out


def _normalize_synopses_by_age_group_for_output(
    synopses: Any,
    *,
    source_text: str,
    title: str | None = None,
) -> Dict[str, Any]:
    """Ensure synopses include required groups and minimum word counts.

    This is a deterministic normalization pass used when rendering metadata JSON,
    so older indexed videos can be upgraded without reprocessing.
    """

    def _word_count(s: str) -> int:
        try:
            return len([w for w in re.split(r"\s+", (s or "").strip()) if w])
        except Exception:
            return 0

    def _first_n_words(src: str, n: int) -> str:
        words = [w for w in re.split(r"\s+", (src or "").strip()) if w]
        if not words:
            return ""
        return " ".join(words[: max(0, int(n))]).strip()

    def _ensure_min_words(value: str, *, min_words: int, src: str) -> str:
        v = re.sub(r"\s+", " ", (value or "").strip())
        if not v:
            v = _first_n_words(src, min_words)
        if _word_count(v) >= int(min_words):
            return v
        need = max(0, int(min_words) - _word_count(v))
        extra = _first_n_words(src, need + 20)
        if extra:
            v = (v.rstrip() + " " + extra).strip()
        while _word_count(v) < int(min_words) and (src or "").strip():
            v = (v.rstrip() + " " + _first_n_words(src, min_words)).strip()
            if len(v) > 6000:
                break
        return v

    src = re.sub(r"\s+", " ", (source_text or "").strip())
    if not src:
        src = re.sub(r"\s+", " ", (title or "").strip())
    if not src:
        return {}

    # Defaults used when missing.
    base_short = _ensure_min_words("This video is about: " + src, min_words=30, src=src)
    base_long = _ensure_min_words("Synopsis: " + src, min_words=70, src=src)

    in_obj = synopses if isinstance(synopses, dict) else {}
    out: Dict[str, Any] = {}
    for k in ["kids", "teens", "adults", "senior_citizen"]:
        grp = in_obj.get(k) if isinstance(in_obj.get(k), dict) else {}
        short_v = str((grp or {}).get("short") or "").strip() or base_short
        long_v = str((grp or {}).get("long") or "").strip() or base_long
        out[k] = {
            "short": _ensure_min_words(short_v, min_words=30, src=src),
            "long": _ensure_min_words(long_v, min_words=70, src=src),
        }
    return out


def _timestamps_to_segments(
    timestamps_seconds: List[int],
    *,
    gap_seconds: int,
) -> List[Dict[str, Any]]:
    ts = sorted(list(dict.fromkeys([int(t) for t in (timestamps_seconds or []) if t is not None])))
    if not ts:
        return []
    gap = max(1, int(gap_seconds))
    segments: List[Dict[str, Any]] = []
    start = ts[0]
    prev = ts[0]
    for t in ts[1:]:
        if (t - prev) <= gap:
            prev = t
            continue
        segments.append({"start_seconds": int(start), "end_seconds": int(prev)})
        start = t
        prev = t
    segments.append({"start_seconds": int(start), "end_seconds": int(prev)})
    return segments


def _timestamps_ms_to_segments(
    timestamps_ms: List[int],
    *,
    gap_ms: int,
    extend_end_ms: int = 0,
) -> List[Dict[str, Any]]:
    ts = sorted(list(dict.fromkeys([int(t) for t in (timestamps_ms or []) if t is not None])))
    if not ts:
        return []
    gap = max(1, int(gap_ms))
    extend = max(0, int(extend_end_ms))
    segments: List[Dict[str, Any]] = []
    start = ts[0]
    prev = ts[0]
    for t in ts[1:]:
        if (t - prev) <= gap:
            prev = t
            continue
        segments.append({"start_ms": int(start), "end_ms": int(prev + extend)})
        start = t
        prev = t
    segments.append({"start_ms": int(start), "end_ms": int(prev + extend)})
    return segments


def _build_detected_content_timelines(
    *,
    frames: List[Dict[str, Any]],
    frame_interval_seconds: int | None,
    max_timestamps_per_item: int = 120,
) -> Dict[str, Any]:
    """Build time-mapped detected-content lists from per-frame metadata."""

    gap = int(frame_interval_seconds or 1)
    gap = max(1, min(30, gap))
    gap_seconds = max(1, int(gap * 2))

    label_map: Dict[str, Dict[str, Any]] = {}
    mod_map: Dict[str, Dict[str, Any]] = {}
    text_map: Dict[str, Dict[str, Any]] = {}

    for f in frames or []:
        ts = f.get("timestamp")
        if ts is None:
            continue
        try:
            tsv = int(ts)
        except Exception:
            continue

        for l in (f.get("labels") or []):
            name = (l.get("name") or "").strip()
            if not name:
                continue
            try:
                conf = float(l.get("confidence") or 0.0)
            except Exception:
                conf = 0.0
            obj = label_map.get(name) or {
                "name": name,
                "max_confidence": 0.0,
                "occurrences": 0,
                "timestamps_seconds": [],
            }
            obj["max_confidence"] = max(float(obj.get("max_confidence") or 0.0), conf)
            obj["occurrences"] = int(obj.get("occurrences") or 0) + 1
            obj["timestamps_seconds"].append(tsv)
            label_map[name] = obj

        for m in (f.get("moderation") or []):
            name = (m.get("name") or "").strip()
            if not name:
                continue
            try:
                conf = float(m.get("confidence") or 0.0)
            except Exception:
                conf = 0.0
            obj = mod_map.get(name) or {
                "name": name,
                "max_confidence": 0.0,
                "occurrences": 0,
                "timestamps_seconds": [],
            }
            obj["max_confidence"] = max(float(obj.get("max_confidence") or 0.0), conf)
            obj["occurrences"] = int(obj.get("occurrences") or 0) + 1
            obj["timestamps_seconds"].append(tsv)
            mod_map[name] = obj

        for t in (f.get("text") or []):
            if (t.get("type") or "") != "LINE":
                continue
            text = (t.get("text") or "").strip()
            if not text:
                continue
            obj = text_map.get(text) or {
                "text": text,
                "occurrences": 0,
                "timestamps_seconds": [],
            }
            obj["occurrences"] = int(obj.get("occurrences") or 0) + 1
            obj["timestamps_seconds"].append(tsv)
            text_map[text] = obj

    def _finalize(items: List[Dict[str, Any]], *, key_name: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for it in items:
            ts_list = sorted(list(dict.fromkeys([int(x) for x in (it.get("timestamps_seconds") or [])])))
            it2 = dict(it)
            it2["timestamps_seconds"] = ts_list[: int(max_timestamps_per_item)]
            it2["first_seen_seconds"] = ts_list[0] if ts_list else None
            it2["last_seen_seconds"] = ts_list[-1] if ts_list else None
            it2["segments"] = _timestamps_to_segments(ts_list, gap_seconds=gap_seconds)
            out.append(it2)
        # sort by occurrences, then confidence if present
        def _sort_key(x: Dict[str, Any]) -> tuple:
            occ = int(x.get("occurrences") or 0)
            conf = float(x.get("max_confidence") or 0.0) if "max_confidence" in x else 0.0
            name = str(x.get(key_name) or "")
            return (occ, conf, name)

        out.sort(key=_sort_key, reverse=True)
        return out

    labels = _finalize(list(label_map.values()), key_name="name")
    moderation_labels = _finalize(list(mod_map.values()), key_name="name")
    on_screen_text = _finalize(list(text_map.values()), key_name="text")

    return {
        "labels": labels[:200],
        "moderation_labels": moderation_labels[:200],
        "on_screen_text": on_screen_text[:200],
        "frame_interval_seconds": int(frame_interval_seconds or 0) or None,
    }


def _time_map_locations_from_segments(
    *,
    locations: List[str],
    transcript_segments: List[Dict[str, Any]],
    max_timestamps_per_location: int = 60,
) -> List[Dict[str, Any]]:
    """Best-effort time mapping of extracted locations to transcript timestamps.

    We do NOT re-run NER per segment (keeps this lightweight). Instead we mark a location
    as present in a segment if its string appears in the segment text (case-insensitive).
    """

    if not locations or not transcript_segments:
        return []

    norm_locations = []
    for loc in locations:
        s = re.sub(r"\s+", " ", str(loc or "").strip())
        if s:
            norm_locations.append(s)

    if not norm_locations:
        return []

    by_loc: Dict[str, List[int]] = {loc: [] for loc in norm_locations}
    for seg in transcript_segments or []:
        txt = re.sub(r"\s+", " ", str(seg.get("text") or "").strip())
        if not txt:
            continue
        try:
            start_s = int(float(seg.get("start") or 0.0))
        except Exception:
            continue
        txt_l = txt.lower()
        for loc in norm_locations:
            if loc.lower() in txt_l:
                by_loc[loc].append(start_s)

    out: List[Dict[str, Any]] = []
    for loc in norm_locations:
        ts_list = sorted(list(dict.fromkeys([int(x) for x in (by_loc.get(loc) or [])])))
        ts_list = ts_list[: int(max_timestamps_per_location)]
        out.append(
            {
                "name": loc,
                "timestamps_seconds": ts_list,
                "first_seen_seconds": ts_list[0] if ts_list else None,
                "last_seen_seconds": ts_list[-1] if ts_list else None,
                "segments": _timestamps_to_segments(ts_list, gap_seconds=10),
            }
        )

    # Prioritize locations with timestamps, then alphabetical.
    out.sort(key=lambda x: (0 if (x.get("timestamps_seconds") or []) else 1, str(x.get("name") or "")))
    return out


def _amazon_location_place_index_name() -> str | None:
    name = (
        os.getenv("ENVID_METADATA_LOCATION_PLACE_INDEX")
        or os.getenv("AWS_LOCATION_PLACE_INDEX")
        or os.getenv("LOCATION_PLACE_INDEX")
        or ""
    )
    name = str(name).strip()
    return name or None


def _normalize_location_name(raw: Any) -> str:
    return re.sub(r"\s+", " ", str(raw or "").strip())


def _parse_csv_set(raw: str | None) -> set[str]:
    if not raw:
        return set()
    parts = [p.strip() for p in str(raw).split(",")]
    return {p for p in parts if p}


def _geocode_with_amazon_location(name: str, *, max_results: int = 1) -> Dict[str, Any] | None:
    """Geocode a place name via Amazon Location Service (Place Index).

    Best-effort; returns None if not configured or on any error.
    """

    idx = _amazon_location_place_index_name()
    if not idx:
        return None
    q = _normalize_location_name(name)
    if not q:
        return None

    try:
        resp = location_svc.search_place_index_for_text(
            IndexName=idx,
            Text=q,
            MaxResults=int(max_results or 1),
        )
    except Exception:
        return None

    results = resp.get("Results") or []
    if not results:
        return None
    top = results[0] or {}
    place = top.get("Place") or {}
    geom = place.get("Geometry") or {}
    point = geom.get("Point")
    if not (isinstance(point, list) and len(point) == 2):
        return None
    try:
        lng = float(point[0])
        lat = float(point[1])
    except Exception:
        return None

    confidence = None
    try:
        if top.get("Relevance") is not None:
            confidence = float(top.get("Relevance") or 0.0)
    except Exception:
        confidence = None

    return {
        "provider": "amazon_location",
        "place_id": top.get("PlaceId") or None,
        "label": place.get("Label") or None,
        "lat": lat,
        "lng": lng,
        "confidence": confidence,
    }


def _time_map_landmarks_from_frames(
    *,
    frames: List[Dict[str, Any]],
    max_timestamps_per_location: int = 60,
) -> List[Dict[str, Any]]:
    """Extract and time-map landmark-like labels from Rekognition frame labels.

    Requires that per-label payloads include `parents` (best-effort).
    """

    if not frames:
        return []

    min_conf = _parse_float_param(
        os.getenv("ENVID_METADATA_LANDMARK_MIN_CONFIDENCE"),
        default=80.0,
        min_value=0.0,
        max_value=100.0,
    )
    allowlist = _parse_csv_set(os.getenv("ENVID_METADATA_LANDMARK_LABEL_ALLOWLIST"))
    denylist = _parse_csv_set(os.getenv("ENVID_METADATA_LANDMARK_LABEL_DENYLIST"))

    def _is_landmark_label(lbl: Dict[str, Any]) -> bool:
        name = _normalize_location_name(lbl.get("name"))
        if not name:
            return False
        if name in denylist:
            return False
        if allowlist and name in allowlist:
            return True
        parents = lbl.get("parents") or []
        try:
            parents_l = {str(p).strip().lower() for p in parents if str(p).strip()}
        except Exception:
            parents_l = set()
        return ("landmarks" in parents_l) or ("landmark" in parents_l)

    by_name: Dict[str, List[int]] = {}
    for f in frames:
        try:
            ts = f.get("timestamp")
            ts_s = int(float(ts)) if ts is not None else None
        except Exception:
            ts_s = None
        if ts_s is None:
            continue
        for lbl in (f.get("labels") or []):
            if not isinstance(lbl, dict):
                continue
            try:
                conf = float(lbl.get("confidence") or 0.0)
            except Exception:
                conf = 0.0
            if conf < float(min_conf):
                continue
            if not _is_landmark_label(lbl):
                continue
            name = _normalize_location_name(lbl.get("name"))
            if not name:
                continue
            by_name.setdefault(name, []).append(ts_s)

    out: List[Dict[str, Any]] = []
    for name, ts_list in by_name.items():
        ts_norm = sorted(list(dict.fromkeys([int(x) for x in ts_list if x is not None])))
        ts_norm = ts_norm[: int(max_timestamps_per_location)]
        out.append(
            {
                "name": name,
                "timestamps_seconds": ts_norm,
                "first_seen_seconds": ts_norm[0] if ts_norm else None,
                "last_seen_seconds": ts_norm[-1] if ts_norm else None,
                "segments": _timestamps_to_segments(ts_norm, gap_seconds=10),
                "sources": ["rekognition_landmark_label"],
            }
        )

    out.sort(key=lambda x: (0 if (x.get("timestamps_seconds") or []) else 1, str(x.get("name") or "")))
    return out


def _merge_time_mapped_locations(*items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for arr in items:
        for it in arr or []:
            if not isinstance(it, dict):
                continue
            name = _normalize_location_name(it.get("name") or it.get("location") or it.get("label"))
            if not name:
                continue
            cur = merged.get(name) or {"name": name, "timestamps_seconds": [], "sources": []}
            ts = it.get("timestamps_seconds") or []
            try:
                cur["timestamps_seconds"].extend([int(x) for x in ts if x is not None])
            except Exception:
                pass
            src = it.get("sources") or []
            if isinstance(src, list):
                cur["sources"].extend([str(s) for s in src if str(s).strip()])
            merged[name] = cur

    out: List[Dict[str, Any]] = []
    for name, cur in merged.items():
        ts_list = sorted(list(dict.fromkeys([int(x) for x in (cur.get("timestamps_seconds") or []) if x is not None])))
        cur["timestamps_seconds"] = ts_list
        cur["first_seen_seconds"] = ts_list[0] if ts_list else None
        cur["last_seen_seconds"] = ts_list[-1] if ts_list else None
        cur["segments"] = _timestamps_to_segments(ts_list, gap_seconds=10)
        cur["sources"] = sorted(list(dict.fromkeys([str(s) for s in (cur.get("sources") or []) if str(s).strip()])))
        out.append(cur)

    out.sort(key=lambda x: (0 if (x.get("timestamps_seconds") or []) else 1, str(x.get("name") or "")))
    return out


def _build_famous_locations_payload(
    *,
    text_locations: List[str],
    transcript_segments: List[Dict[str, Any]],
    frames: List[Dict[str, Any]],
    geocode_cache: Dict[str, Any] | None = None,
    max_geocodes: int = 30,
) -> Dict[str, Any]:
    text_locations_norm = [_normalize_location_name(x) for x in (text_locations or [])]
    text_locations_norm = [x for x in text_locations_norm if x]
    text_locations_norm = list(dict.fromkeys(text_locations_norm))

    mapped_text = _time_map_locations_from_segments(
        locations=text_locations_norm,
        transcript_segments=transcript_segments or [],
    )
    for it in mapped_text:
        if isinstance(it, dict):
            it.setdefault("sources", ["comprehend_text"])

    mapped_landmarks = _time_map_landmarks_from_frames(frames=frames or [])
    merged_time_mapped = _merge_time_mapped_locations(mapped_text, mapped_landmarks)

    all_names = [it.get("name") for it in merged_time_mapped if isinstance(it, dict) and it.get("name")]
    all_names = [_normalize_location_name(x) for x in all_names]
    all_names = [x for x in all_names if x]
    all_names = list(dict.fromkeys(all_names))

    geo: Dict[str, Any] = geocode_cache if isinstance(geocode_cache, dict) else {}
    if _amazon_location_place_index_name():
        remaining = int(max_geocodes)
        for nm in all_names:
            if remaining <= 0:
                break
            if isinstance(geo.get(nm), dict) and geo.get(nm):
                continue
            g = _geocode_with_amazon_location(nm)
            if g:
                geo[nm] = g
            remaining -= 1

    if geo:
        for it in merged_time_mapped:
            if not isinstance(it, dict):
                continue
            nm = _normalize_location_name(it.get("name"))
            if nm and isinstance(geo.get(nm), dict) and geo.get(nm):
                it["geocode"] = geo.get(nm)

    return {
        "locations": all_names,
        "time_mapped": merged_time_mapped,
        "from_transcript": text_locations_norm,
        "from_landmarks": [it.get("name") for it in mapped_landmarks if isinstance(it, dict) and it.get("name")],
        "geocode_cache": geo,
    }


def _job_update(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = datetime.utcnow().isoformat()


def _job_init(job_id: str, *, title: str) -> None:
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "title": title,
            "status": "queued",  # queued | processing | completed | failed
            "progress": 0,
            "message": "Queued",
            "error": None,
            "result": None,
            "steps": _job_steps_default(),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }


def _default_frame_interval_seconds(duration_seconds: float | None) -> int:
    """Pick a sane default frame interval for analysis.

    Short clips need denser sampling for celebrity detection; longer videos should
    stay sparse to limit Rekognition calls.
    """
    if duration_seconds is None or duration_seconds <= 0:
        return 5
    if duration_seconds <= 60:
        return 2
    if duration_seconds <= 180:
        return 5
    if duration_seconds <= 600:
        return 10
    return 15


def _pick_evenly_spaced_indices(total: int, count: int) -> set[int]:
    if total <= 0 or count <= 0:
        return set()
    if count >= total:
        return set(range(total))
    if count == 1:
        return {0}
    step = (total - 1) / float(count - 1)
    indices = {int(round(i * step)) for i in range(count)}
    return {i for i in indices if 0 <= i < total}


def _process_video_job(
    *,
    job_id: str,
    temp_dir: Path,
    video_path: Path,
    video_title: str,
    video_description: str,
    original_filename: str,
    frame_interval_seconds: int,
    max_frames_to_analyze: int,
    face_recognition_mode: Optional[str],
    collection_id: Optional[str],
    preexisting_s3_video_key: str | None = None,
    preexisting_s3_video_uri: str | None = None,
) -> None:
    try:
        duration_seconds = _probe_video_duration_seconds(video_path)

        profile = _envid_metadata_output_profile()

        if frame_interval_seconds <= 0:
            frame_interval_seconds = _default_frame_interval_seconds(duration_seconds)
        frame_interval_seconds = _parse_int_param(
            frame_interval_seconds,
            default=10,
            min_value=1,
            max_value=30,
        )

        # Ensure the source video exists in S3 before any Rekognition processing.
        # If caller already provided an S3 object (e.g. rawvideo), reuse it.
        # Else, upload the local video to S3.
        # IMPORTANT: Per product requirement, we never run analysis without first storing the video in S3.
        s3_video_key: str | None = preexisting_s3_video_key
        s3_video_uri: str | None = preexisting_s3_video_uri
        if s3_video_key and not s3_video_uri:
            # Best-effort: reconstruct uri for downstream display/use.
            try:
                b, k = _parse_allowed_s3_video_source(s3_video_key)
                s3_video_uri = f"s3://{b}/{k}"
            except Exception:
                s3_video_uri = None

        if s3_video_key and s3_video_uri:
            _job_step_update(job_id, "upload_to_s3", status="completed", percent=100, message="Using existing S3 source")

        if not (s3_video_key and s3_video_uri):
            _job_step_update(job_id, "upload_to_s3", status="running", percent=0, message="Uploading video to S3")
            _job_update(job_id, status="processing", progress=2, message="Uploading video to S3")
            upload_path = _ensure_rekognition_compatible_before_upload(
                video_path=video_path,
                temp_dir=temp_dir,
                job_id=job_id,
            )
            original_filename_for_upload = str(Path(original_filename).with_suffix(".mp4").name)
            s3_video_key, s3_video_uri = _upload_video_to_s3_or_raise(
                video_id=job_id,
                local_path=upload_path,
                original_filename=original_filename_for_upload,
                job_id=job_id,
            )

            if s3_video_key and s3_video_uri:
                _job_step_update(job_id, "upload_to_s3", status="completed", percent=100, message="Uploaded to S3")

        if not (s3_video_key and s3_video_uri):
            raise RuntimeError("Refusing to analyze video: missing S3 location (upload failed or misconfigured bucket).")

        _job_update(job_id, s3_video_key=s3_video_key, s3_video_uri=s3_video_uri)

        def _prefer_rekognition_video_s3() -> bool:
            # User requirement: Rekognition should always analyze video from S3.
            # We keep the legacy per-frame Rekognition Image pipeline for non-required profiles.
            mode = (os.getenv("ENVID_METADATA_REKOGNITION_ANALYSIS_SOURCE") or "s3_video").strip().lower()
            return mode in {"s3_video", "video", "rekognition_video"}

        force_video = bool(s3_video_key and (s3_video_uri or s3_video_key) and _prefer_rekognition_video_s3())
        use_rekognition_video = bool(force_video and (_disable_per_frame_analysis() or (profile in {"required", "minimal"})))

        # Common outputs produced by either pipeline
        all_emotions: List[str] = []
        label_stats: Dict[str, Dict[str, float]] = {}
        text_stats: Dict[str, int] = {}
        celeb_stats: Dict[str, Dict[str, Any]] = {}
        custom_face_stats: Dict[str, Dict[str, Any]] = {}
        moderation_stats: Dict[str, float] = {}
        gender_counts: Dict[str, int] = {}
        age_ranges: List[Tuple[int, int]] = []
        frames_metadata: List[Dict[str, Any]] = []

        custom_faces_detailed: List[Dict[str, Any]] = []
        custom_collection_id_norm = _rekognition_collection_id_normalize(collection_id) if face_recognition_mode == "aws_collection" else None

        thumb_input_source: str = str(video_path)

        def _extract_single_frame_jpeg_bytes(*, ts_seconds: float) -> bytes | None:
            try:
                ts_f = max(0.0, float(ts_seconds))
            except Exception:
                ts_f = 0.0
            ts_ms = int(round(ts_f * 1000.0))
            out_path = temp_dir / f"thumb_{ts_ms:09d}.jpg"

            def _run_ffmpeg(input_source: str) -> bool:
                if out_path.exists():
                    try:
                        out_path.unlink()
                    except Exception:
                        pass
                cmd = [
                    FFMPEG_PATH,
                    "-ss",
                    f"{ts_f:.3f}",
                    "-i",
                    str(input_source),
                    "-vframes",
                    "1",
                    "-q:v",
                    "2",
                    str(out_path),
                    "-y",
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return res.returncode == 0 and out_path.exists()

            if not _run_ffmpeg(thumb_input_source):
                # If presigned-url streaming fails, fall back to local input when possible.
                if thumb_input_source != str(video_path) and video_path.exists():
                    if not _run_ffmpeg(str(video_path)):
                        return None
                else:
                    return None

            if not out_path.exists():
                return None
            try:
                return out_path.read_bytes()
            except Exception:
                return None
            finally:
                try:
                    out_path.unlink()
                except Exception:
                    pass
        # Decide feature flags early (both pipelines rely on these)
        preset_default = _features_preset_default_enabled()
        if profile in {"required", "minimal"}:
            enable_labels = True
            enable_text = True
            enable_moderation = True
            enable_faces = True
            enable_celebrities = True
            enable_transcribe = True
            enable_embedding = False
        else:
            enable_labels = _feature_enabled("LABELS", default=preset_default)
            enable_text = _feature_enabled("TEXT", default=preset_default)
            enable_moderation = _feature_enabled("MODERATION", default=preset_default)
            enable_faces = _feature_enabled("FACES", default=preset_default)
            enable_celebrities = _feature_enabled("CELEBRITIES", default=preset_default)
            enable_transcribe = _feature_enabled("TRANSCRIBE", default=preset_default)
            enable_embedding = _feature_enabled("EMBEDDING", default=preset_default)

        frames: List[Path] = []
        scene_windows_override: List[Dict[str, Any]] | None = None
        scene_segmentation_source: str = "fixed_windows"
        rekognition_shot_segments_raw: list[dict[str, Any]] = []
        rekognition_shots: List[Dict[str, Any]] = []
        if use_rekognition_video:
            # Rekognition analysis path: use Rekognition Video async jobs against the S3 object.
            # We still extract frames locally ONLY for thumbnail/cropping purposes.
            _job_update(job_id, status="processing", progress=3, message="Rekognition Video: starting (S3 source)")

            raw_source = (s3_video_uri or "").strip() or (s3_video_key or "").strip()
            s3_bucket, s3_key = _parse_allowed_s3_video_source(raw_source)

            # In S3-only deployments (no local MP4 persistence) or when explicitly requested,
            # do thumbnail/cropping extraction by streaming from S3 via a presigned URL.
            thumb_source_mode = (os.getenv("ENVID_METADATA_THUMBNAIL_SOURCE") or "local").strip().lower()
            if (thumb_source_mode in {"s3", "s3_presigned", "presigned"}) or (not Path(str(video_path)).exists()):
                try:
                    thumb_input_source = _presign_s3_get_object_url(bucket=s3_bucket, key=s3_key)
                except Exception as exc:
                    app.logger.warning("Failed to presign S3 URL for thumbnails; falling back to local file: %s", exc)
            bucket_region = _detect_bucket_region(s3_bucket)
            region = bucket_region or DEFAULT_AWS_REGION
            rk = _rekognition_client(region_name=region)
            video_ref = _rekognition_video_ref(s3_bucket, s3_key)

            labels_raw: list[dict[str, Any]] = []
            if enable_labels:
                _job_update(job_id, progress=5, message="Rekognition Video: starting label detection")
                _job_step_update(job_id, "rekognition_labels", status="running", percent=0, message="Starting")
                ld = rk.start_label_detection(
                    Video=video_ref,
                    MinConfidence=float(os.getenv("ENVID_METADATA_REKOGNITION_LABEL_MIN_CONFIDENCE") or 70.0),
                )
                label_job_id = ld.get("JobId")
                if not label_job_id:
                    raise RuntimeError("Failed to start label detection")
                labels_raw = _poll_rekognition_video_job(
                    envid_job_id=job_id,
                    step_id="rekognition_labels",
                    rekognition_job_id=label_job_id,
                    get_page=rk.get_label_detection,
                    page_key="NextToken",
                    item_key="Labels",
                    progress_base=8,
                    progress_span=16,
                    message="Rekognition Video: detecting labels",
                )

            celebs_raw: list[dict[str, Any]] = []
            if enable_celebrities:
                _job_update(job_id, progress=24, message="Rekognition Video: starting celebrity recognition")
                _job_step_update(job_id, "rekognition_celebrities", status="running", percent=0, message="Starting")
                cr = rk.start_celebrity_recognition(Video=video_ref)
                celeb_job_id = cr.get("JobId")
                if not celeb_job_id:
                    raise RuntimeError("Failed to start celebrity recognition")
                celebs_raw = _poll_rekognition_video_job(
                    envid_job_id=job_id,
                    step_id="rekognition_celebrities",
                    rekognition_job_id=celeb_job_id,
                    get_page=rk.get_celebrity_recognition,
                    page_key="NextToken",
                    item_key="Celebrities",
                    progress_base=26,
                    progress_span=16,
                    message="Rekognition Video: detecting celebrities",
                )

            # Rekognition Video collection matching intentionally removed.
            # This project now relies on celebrity recognition and other metadata only.

            moderation_raw: list[dict[str, Any]] = []
            if enable_moderation:
                try:
                    _job_update(job_id, progress=42, message="Rekognition Video: starting content moderation")
                    md = rk.start_content_moderation(
                        Video=video_ref,
                        MinConfidence=float(os.getenv("ENVID_METADATA_REKOGNITION_MODERATION_MIN_CONFIDENCE") or 60.0),
                    )
                    moderation_job_id = md.get("JobId")
                    if moderation_job_id:
                        _job_step_update(job_id, "content_moderation", status="running", percent=0, message="Starting")
                        moderation_raw = _poll_rekognition_video_job(
                            envid_job_id=job_id,
                            step_id="content_moderation",
                            rekognition_job_id=moderation_job_id,
                            get_page=rk.get_content_moderation,
                            page_key="NextToken",
                            item_key="ModerationLabels",
                            progress_base=44,
                            progress_span=6,
                            message="Rekognition Video: detecting moderation",
                        )
                except Exception as exc:
                    app.logger.warning("Moderation detection failed: %s", exc)

            text_raw: list[dict[str, Any]] = []
            if enable_text:
                try:
                    _job_update(job_id, progress=50, message="Rekognition Video: starting text detection")
                    td = rk.start_text_detection(Video=video_ref)
                    text_job_id = td.get("JobId")
                    if text_job_id:
                        _job_step_update(job_id, "rekognition_text", status="running", percent=0, message="Starting")
                        text_raw = _poll_rekognition_video_job(
                            envid_job_id=job_id,
                            step_id="rekognition_text",
                            rekognition_job_id=text_job_id,
                            get_page=rk.get_text_detection,
                            page_key="NextToken",
                            item_key="TextDetections",
                            progress_base=52,
                            progress_span=6,
                            message="Rekognition Video: detecting on-screen text",
                        )
                except Exception as exc:
                    app.logger.warning("Text detection failed: %s", exc)

            # Scene/shot segmentation for scene-by-scene metadata.
            # Prefer Rekognition shot boundaries when available; fall back to fixed windows.
            seg_mode = (os.getenv("ENVID_METADATA_SCENE_SEGMENTATION") or "rekognition").strip().lower()
            enable_shots = seg_mode in {"rekognition", "rekognition_shots", "shots", "shot", "segment_detection"}
            if enable_shots:
                try:
                    _job_update(job_id, progress=56, message="Rekognition Video: starting shot detection")
                    _job_step_update(job_id, "rekognition_shots", status="running", percent=0, message="Starting")

                    shot_min_conf = _parse_float_param(
                        os.getenv("ENVID_METADATA_REKOGNITION_SHOT_MIN_CONFIDENCE"),
                        default=60.0,
                        min_value=0.0,
                        max_value=100.0,
                    )

                    sd = rk.start_segment_detection(
                        Video=video_ref,
                        SegmentTypes=["SHOT"],
                        Filters={"ShotFilter": {"MinSegmentConfidence": float(shot_min_conf)}},
                    )
                    seg_job_id = sd.get("JobId")
                    if not seg_job_id:
                        raise RuntimeError("Failed to start shot detection")

                    rekognition_shot_segments_raw = _poll_rekognition_video_job(
                        envid_job_id=job_id,
                        step_id="rekognition_shots",
                        rekognition_job_id=seg_job_id,
                        get_page=rk.get_segment_detection,
                        page_key="NextToken",
                        item_key="Segments",
                        progress_base=56,
                        progress_span=2,
                        message="Rekognition Video: detecting shots",
                    )

                    # Keep a compact, stable representation for storage.
                    for it in rekognition_shot_segments_raw or []:
                        if not isinstance(it, dict):
                            continue
                        seg = it.get("ShotSegment") if isinstance(it.get("ShotSegment"), dict) else None
                        if not seg:
                            continue
                        try:
                            sm = int(seg.get("StartTimestampMillis") or 0)
                            em = int(seg.get("EndTimestampMillis") or 0)
                        except Exception:
                            continue
                        if em <= sm:
                            continue
                        try:
                            conf = float(seg.get("Confidence") or 0.0)
                        except Exception:
                            conf = 0.0
                        rekognition_shots.append({"start_ms": sm, "end_ms": em, "confidence": conf})
                        if len(rekognition_shots) >= 5000:
                            break

                    # Build merged scenes (target roughly matches ENVID_METADATA_SCENE_WINDOW_SECONDS)
                    target_win = _parse_int_param(
                        os.getenv("ENVID_METADATA_SCENE_WINDOW_SECONDS"),
                        default=15,
                        min_value=5,
                        max_value=120,
                    )
                    scene_windows_override = _rekognition_shot_segments_to_scene_windows(
                        shot_segments_raw=rekognition_shot_segments_raw,
                        duration_seconds=duration_seconds,
                        target_window_seconds=target_win,
                    )
                    if scene_windows_override:
                        scene_segmentation_source = "rekognition_shots"
                except Exception as exc:
                    app.logger.warning("Shot/scene segmentation failed; falling back to fixed windows: %s", exc)
                    _job_step_update(job_id, "rekognition_shots", status="skipped", percent=100, message="Fallback to fixed windows")

            # Synthesize per-timestamp frame-like metadata on a fixed interval grid.
            _job_update(job_id, progress=58, message="Building time-mapped timelines")

            interval = max(1, int(frame_interval_seconds or 1))
            dur_int = int(duration_seconds or 0)
            if dur_int <= 0:
                dur_int = max(1, int(max([int((x.get("Timestamp") or 0) / 1000) for x in (labels_raw + celebs_raw + moderation_raw + text_raw)] or [1])))

            bins: Dict[int, Dict[str, Any]] = {}
            for t in range(0, max(1, dur_int + 1), interval):
                bins[int(t)] = {
                    "timestamp": int(t),
                    "labels": [],
                    "text": [],
                    "faces": [],
                    "celebrities": [],
                    "moderation": [],
                    "custom_faces": [],
                    "local_faces": [],
                }

            def _bin_for_second(sec: int) -> int:
                if sec < 0:
                    sec = 0
                b = int(sec // interval) * interval
                if b not in bins:
                    # Clamp to nearest existing bin.
                    keys = sorted(bins.keys())
                    if not keys:
                        return 0
                    if b < keys[0]:
                        return keys[0]
                    if b > keys[-1]:
                        return keys[-1]
                    return keys[-1]
                return b

            # For thumbnail crops, retain a lookup of bounding boxes by (name, bin_ts)
            celeb_bbox_by_name_and_bin: Dict[Tuple[str, int], Dict[str, Any]] = {}
            # Track best thumbnail candidate per celebrity (prefer high confidence + larger face box).
            best_celeb_thumb: Dict[str, Dict[str, Any]] = {}
            custom_bbox_by_name_and_bin: Dict[Tuple[str, int], Dict[str, Any]] = {}

            celeb_min_conf = _parse_float_param(
                os.getenv("ENVID_METADATA_CELEB_MIN_CONFIDENCE"),
                default=95.0,
                min_value=0.0,
                max_value=100.0,
            )

            # Allow lower-confidence hits for celebrities that were already strongly identified
            # earlier in the same video. This improves recall while keeping the initial
            # identification strict.
            celeb_min_conf_seen = _parse_float_param(
                os.getenv("ENVID_METADATA_CELEB_MIN_CONFIDENCE_SEEN"),
                default=85.0,
                min_value=0.0,
                max_value=100.0,
            )
            celeb_edge_margin = _parse_float_param(
                os.getenv("ENVID_METADATA_CELEB_EDGE_MARGIN"),
                default=0.02,
                min_value=0.0,
                max_value=0.2,
            )
            celeb_min_bbox = _parse_float_param(
                os.getenv("ENVID_METADATA_CELEB_MIN_BBOX"),
                default=0.06,
                min_value=0.0,
                max_value=0.5,
            )

            celeb_verify_enable = (os.getenv("ENVID_METADATA_CELEB_VERIFY_IMAGE") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            celeb_verify_min_conf = _parse_float_param(
                os.getenv("ENVID_METADATA_CELEB_VERIFY_MIN_CONFIDENCE"),
                default=97.0,
                min_value=0.0,
                max_value=100.0,
            )
            celeb_verify_max_frames = _parse_int_param(
                os.getenv("ENVID_METADATA_CELEB_VERIFY_MAX_FRAMES"),
                default=30,
                min_value=0,
                max_value=300,
            )
            celeb_verify_allow_override = (os.getenv("ENVID_METADATA_CELEB_VERIFY_ALLOW_OVERRIDE") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

            celeb_compare_enable = (os.getenv("ENVID_METADATA_CELEB_COMPAREFACES") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            celeb_compare_similarity = _parse_float_param(
                os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_SIMILARITY"),
                default=85.0,
                min_value=0.0,
                max_value=100.0,
            )
            celeb_compare_max_frames = _parse_int_param(
                os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_MAX_FRAMES"),
                default=int(celeb_verify_max_frames or 30),
                min_value=0,
                max_value=300,
            )
            celeb_compare_allow_override = (os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_ALLOW_OVERRIDE") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

            celeb_compare_extra_names = [
                n.strip()
                for n in (os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_EXTRA_NAMES") or "").split(",")
                if n.strip()
            ][:25]

            celeb_compare_debug_seconds: set[int] = set()
            debug_raw = (os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_DEBUG_SECONDS") or "").strip()
            if debug_raw:
                for part in debug_raw.split(","):
                    p = (part or "").strip()
                    if not p:
                        continue
                    try:
                        celeb_compare_debug_seconds.add(int(float(p)))
                    except Exception:
                        continue

            celeb_compare_bbox_pad = _parse_float_param(
                os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_BBOX_PAD"),
                default=0.08,
                min_value=0.0,
                max_value=0.5,
            )

            celeb_compare_debug_apply = (os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_DEBUG_APPLY") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            celeb_compare_debug_min_similarity = _parse_float_param(
                os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_DEBUG_MIN_SIMILARITY"),
                default=0.0,
                min_value=0.0,
                max_value=100.0,
            )
            celeb_compare_debug_min_delta = _parse_float_param(
                os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_DEBUG_MIN_DELTA"),
                default=0.0,
                min_value=0.0,
                max_value=100.0,
            )

            verified_names_by_second: Dict[int, set[str]] = {}
            verified_best_name_by_second: Dict[int, str] = {}
            all_video_celeb_names: set[str] = set()
            urls_by_name: Dict[str, List[str]] = {}
            celeb_ts_ms_by_name: Dict[str, List[int]] = {}
            for it in celebs_raw:
                if not isinstance(it, dict):
                    continue
                celeb_obj = it.get("Celebrity") if isinstance(it.get("Celebrity"), dict) else {}
                nm0 = (celeb_obj.get("Name") or "").strip() if isinstance(celeb_obj, dict) else ""
                if nm0:
                    all_video_celeb_names.add(nm0)
                    u0 = [u for u in (celeb_obj.get("Urls") or []) if isinstance(u, str) and u.strip()]
                    if u0 and nm0 not in urls_by_name:
                        urls_by_name[nm0] = u0[:5]

            if celeb_verify_enable and celebs_raw and int(celeb_verify_max_frames or 0) > 0:

                try:
                    candidate_seconds = sorted(
                        {
                            int(float(item.get("Timestamp") or 0.0) / 1000.0)
                            for item in celebs_raw
                            if isinstance(item, dict)
                        }
                    )
                except Exception:
                    candidate_seconds = []

                checked = 0
                for sec in candidate_seconds:
                    if checked >= int(celeb_verify_max_frames):
                        break
                    fb = _extract_single_frame_jpeg_bytes(ts_seconds=int(sec))
                    if not fb:
                        continue
                    try:
                        resp = rk.recognize_celebrities(Image={"Bytes": fb})
                    except Exception:
                        continue
                    names: set[str] = set()
                    best_name = ""
                    best_conf = -1.0
                    for face in (resp.get("CelebrityFaces") or []):
                        if not isinstance(face, dict):
                            continue
                        nm = (face.get("Name") or "").strip()
                        if not nm:
                            continue
                        try:
                            mconf = float(face.get("MatchConfidence") or 0.0)
                        except Exception:
                            mconf = 0.0
                        if mconf < float(celeb_verify_min_conf):
                            continue
                        face_obj = face.get("Face") if isinstance(face.get("Face"), dict) else {}
                        bbox_img = face_obj.get("BoundingBox") if isinstance(face_obj, dict) else None
                        if _bbox_looks_partial_or_too_small(
                            bbox_img if isinstance(bbox_img, dict) else None,
                            edge_margin=celeb_edge_margin,
                            min_width=celeb_min_bbox,
                            min_height=celeb_min_bbox,
                        ):
                            continue
                        names.add(nm)
                        if mconf > best_conf:
                            best_conf = mconf
                            best_name = nm

                    # Only record seconds where we got at least one credible image-based celebrity.
                    if names:
                        verified_names_by_second[int(sec)] = names
                        if best_name:
                            verified_best_name_by_second[int(sec)] = best_name
                    checked += 1

            # Portrait-based verification via Rekognition CompareFaces.
            comparefaces_best_name_by_second: Dict[int, str] = {}
            comparefaces_best_similarity_by_second: Dict[int, float] = {}
            comparefaces_candidate_names: List[str] = []
            if celeb_compare_enable and celebs_raw and int(celeb_compare_max_frames or 0) > 0:
                if celeb_compare_debug_seconds:
                    app.logger.warning(
                        "CompareFaces: starting (debug_seconds=%s; celebs_raw=%d)",
                        ",".join(str(x) for x in sorted(celeb_compare_debug_seconds)),
                        int(len(celebs_raw)),
                    )
                compare_source: str | None = None
                try:
                    compare_source = _presign_s3_get_object_url(bucket=analysis_bucket, key=analysis_key)
                except Exception:
                    compare_source = None

                if compare_source:
                    compare_temp_dir = Path(tempfile.mkdtemp(prefix=f"envid_metadata_celeb_compare_{job_id}_"))
                    try:
                        def _extract_compare_frame_jpeg_bytes(*, ts_seconds: float) -> bytes | None:
                            try:
                                ts_f = max(0.0, float(ts_seconds))
                            except Exception:
                                ts_f = 0.0
                            ts_ms = int(round(ts_f * 1000.0))
                            out_path = compare_temp_dir / f"compare_{ts_ms:09d}.jpg"
                            try:
                                if out_path.exists():
                                    out_path.unlink()
                            except Exception:
                                pass
                            cmd = [
                                FFMPEG_PATH,
                                "-ss",
                                f"{ts_f:.3f}",
                                "-i",
                                str(compare_source),
                                "-vframes",
                                "1",
                                "-q:v",
                                "2",
                                str(out_path),
                                "-y",
                            ]
                            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            if res.returncode != 0 or not out_path.exists():
                                return None
                            try:
                                return out_path.read_bytes()
                            except Exception:
                                return None
                            finally:
                                try:
                                    out_path.unlink()
                                except Exception:
                                    pass

                        try:
                            distinct_names = sorted(
                                {
                                    (it.get("Celebrity") or {}).get("Name")
                                    for it in celebs_raw
                                    if isinstance(it, dict) and isinstance(it.get("Celebrity"), dict)
                                }
                            )
                        except Exception:
                            distinct_names = []
                        distinct_names = [(n or "").strip() for n in distinct_names if isinstance(n, str) and (n or "").strip()]
                        comparefaces_candidate_names = sorted(set(distinct_names + celeb_compare_extra_names))[:50]

                        portrait_meta = _celebrity_bios_for_names(comparefaces_candidate_names) if comparefaces_candidate_names else {}

                        portrait_bytes_by_name: Dict[str, bytes] = {}
                        for nm in comparefaces_candidate_names:
                            url = ((portrait_meta.get(nm) or {}).get("portrait_url") or "").strip()
                            if not url:
                                continue
                            pb = _http_get_bytes(url)
                            pb_norm = _normalize_image_bytes_for_rekognition(pb) if pb else None
                            if pb_norm:
                                portrait_bytes_by_name[nm] = pb_norm

                        sec_to_best_item: Dict[int, Dict[str, Any]] = {}
                        for it in celebs_raw:
                            if not isinstance(it, dict):
                                continue
                            try:
                                sec = int(float(it.get("Timestamp") or 0.0) / 1000.0)
                            except Exception:
                                sec = 0
                            debug_sec = int(sec) in celeb_compare_debug_seconds
                            celeb_obj = it.get("Celebrity") if isinstance(it.get("Celebrity"), dict) else {}
                            nm = (celeb_obj.get("Name") or "").strip() if isinstance(celeb_obj, dict) else ""
                            if not nm:
                                continue
                            try:
                                conf0 = float(it.get("MatchConfidence") or celeb_obj.get("Confidence") or 0.0)
                            except Exception:
                                conf0 = 0.0
                            face = celeb_obj.get("Face") if isinstance(celeb_obj.get("Face"), dict) else {}
                            bbox0 = face.get("BoundingBox") if isinstance(face, dict) else None
                            # For explicit debug seconds, allow evaluation even if the raw Rekognition
                            # hit fails our strict confidence/partial-face filters (we'll still rely on
                            # CompareFaces similarity to accept/reject overrides).
                            if (not debug_sec) and conf0 < float(celeb_min_conf):
                                continue
                            if (not debug_sec) and _bbox_looks_partial_or_too_small(
                                bbox0 if isinstance(bbox0, dict) else None,
                                edge_margin=celeb_edge_margin,
                                min_width=celeb_min_bbox,
                                min_height=celeb_min_bbox,
                            ):
                                continue
                            if not isinstance(bbox0, dict) or not bbox0:
                                continue

                            try:
                                area0 = float(bbox0.get("Width") or 0.0) * float(bbox0.get("Height") or 0.0)
                            except Exception:
                                area0 = 0.0

                            prev = sec_to_best_item.get(int(sec))
                            prev_conf = float(prev.get("conf") or 0.0) if isinstance(prev, dict) else -1.0
                            prev_area = float(prev.get("area") or 0.0) if isinstance(prev, dict) else -1.0
                            if prev is None or conf0 > prev_conf or (conf0 == prev_conf and area0 > prev_area):
                                sec_to_best_item[int(sec)] = {
                                    "detected_name": nm,
                                    "bbox": bbox0,
                                    "conf": conf0,
                                    "area": area0,
                                    "ts_s": float(it.get("Timestamp") or 0.0) / 1000.0,
                                }

                        # If the user asked to debug a specific second, ensure we can evaluate that second
                        # even when Rekognition Video's millisecond timestamps fall just before/after the
                        # integer boundary.
                        #
                        # 1) Prefer: pick the best raw celeb item within a +/- window around the requested
                        #    second (based on the true float timestamp), and store it under the requested
                        #    debug second key.
                        # 2) Fallback: borrow the nearest bbox/name from an adjacent second BUT force the
                        #    frame extraction timestamp to the requested debug second (prevents "borrowed"
                        #    frame/bbox mismatches like using 9.000s for debug sec=10).
                        if celeb_compare_debug_seconds:
                            try:
                                nearby_window_s = int(
                                    os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_DEBUG_NEARBY_WINDOW_S", "2") or 2
                                )
                            except Exception:
                                nearby_window_s = 2

                            def _best_item_near_debug_second(ds: int) -> Dict[str, Any] | None:
                                # For debug seconds, prioritize *time proximity* to ds so we evaluate
                                # the intended moment, not the highest-confidence nearby hit.
                                best: Dict[str, Any] | None = None
                                best_dt = 1e9
                                best_conf = -1.0
                                best_area = -1.0
                                for it2 in celebs_raw:
                                    if not isinstance(it2, dict):
                                        continue
                                    celeb_obj2 = it2.get("Celebrity") if isinstance(it2.get("Celebrity"), dict) else {}
                                    nm2 = (celeb_obj2.get("Name") or "").strip() if isinstance(celeb_obj2, dict) else ""
                                    if not nm2:
                                        continue
                                    try:
                                        ts_s2 = float(it2.get("Timestamp") or 0.0) / 1000.0
                                    except Exception:
                                        ts_s2 = 0.0
                                    dt = abs(float(ts_s2) - float(ds))
                                    if dt > float(nearby_window_s):
                                        continue
                                    face2 = celeb_obj2.get("Face") if isinstance(celeb_obj2.get("Face"), dict) else {}
                                    bbox2 = face2.get("BoundingBox") if isinstance(face2, dict) else None
                                    if not isinstance(bbox2, dict) or not bbox2:
                                        continue
                                    try:
                                        conf2 = float(it2.get("MatchConfidence") or celeb_obj2.get("Confidence") or 0.0)
                                    except Exception:
                                        conf2 = 0.0
                                    try:
                                        area2 = float(bbox2.get("Width") or 0.0) * float(bbox2.get("Height") or 0.0)
                                    except Exception:
                                        area2 = 0.0
                                    if (
                                        best is None
                                        or dt < best_dt
                                        or (dt == best_dt and conf2 > best_conf)
                                        or (dt == best_dt and conf2 == best_conf and area2 > best_area)
                                    ):
                                        best = {
                                            "detected_name": nm2,
                                            "bbox": bbox2,
                                            "conf": conf2,
                                            "area": area2,
                                            "ts_s": ts_s2,
                                            "_debug_from_ts_s": ts_s2,
                                        }
                                        best_dt = dt
                                        best_conf = conf2
                                        best_area = area2
                                return best

                            existing_secs_sorted = sorted(sec_to_best_item.keys())
                            for ds in sorted(celeb_compare_debug_seconds):
                                if ds in sec_to_best_item:
                                    continue

                                cand = _best_item_near_debug_second(int(ds))
                                if cand:
                                    # For debug seconds we want to evaluate the exact requested second,
                                    # even when the nearest raw Rekognition hit is slightly earlier/later.
                                    # Keep the raw timestamp for logging, but force frame extraction to ds.
                                    cand["_debug_force_ts_s"] = float(ds)
                                    cand["ts_s"] = float(ds)
                                    sec_to_best_item[int(ds)] = cand
                                    app.logger.warning(
                                        "CompareFaces debug sec=%d missing; using nearby raw ts_s=%.3f (detected=%s) and forcing frame_ts_s=%.3f",
                                        int(ds),
                                        float(cand.get("_debug_from_ts_s") or 0.0),
                                        str(cand.get("detected_name") or ""),
                                        float(ds),
                                    )
                                    continue

                                if existing_secs_sorted:
                                    closest = min(existing_secs_sorted, key=lambda s: abs(int(s) - int(ds)))
                                    if abs(int(closest) - int(ds)) <= nearby_window_s:
                                        borrowed = dict(sec_to_best_item.get(int(closest)) or {})
                                        borrowed["_debug_from_sec"] = int(closest)
                                        borrowed["_debug_force_ts_s"] = float(ds)
                                        borrowed["ts_s"] = float(ds)
                                        sec_to_best_item[int(ds)] = borrowed
                                        app.logger.warning(
                                            "CompareFaces debug sec=%d missing; borrowing bbox from sec=%d (detected=%s) and forcing frame_ts_s=%.3f",
                                            int(ds),
                                            int(closest),
                                            str((sec_to_best_item.get(int(closest)) or {}).get("detected_name") or ""),
                                            float(ds),
                                        )

                        if celeb_compare_debug_seconds:
                            for ds in sorted(celeb_compare_debug_seconds):
                                if ds in sec_to_best_item:
                                    app.logger.warning(
                                        "CompareFaces debug seconds: sec=%d present (detected=%s)",
                                        int(ds),
                                        str((sec_to_best_item.get(ds) or {}).get("detected_name") or ""),
                                    )
                                else:
                                    app.logger.warning("CompareFaces debug seconds: sec=%d NOT present", int(ds))

                        frames_for_compare: Dict[int, bytes] = {}
                        seconds = sorted(sec_to_best_item.keys())[: int(celeb_compare_max_frames)]
                        for sec in seconds:
                            best_item = sec_to_best_item.get(int(sec)) or {}
                            try:
                                frame_ts_s = float(best_item.get("ts_s") or float(sec)) if isinstance(best_item, dict) else float(sec)
                            except Exception:
                                frame_ts_s = float(sec)
                            frame_key = int(round(frame_ts_s * 1000.0))
                            if int(sec) in celeb_compare_debug_seconds and isinstance(best_item, dict):
                                if best_item.get("_debug_from_sec") is not None:
                                    app.logger.warning(
                                        "CompareFaces debug sec=%d using frame_ts_s=%.3f (borrowed bbox from sec=%s)",
                                        int(sec),
                                        float(frame_ts_s),
                                        str(best_item.get("_debug_from_sec")),
                                    )
                                elif best_item.get("_debug_force_ts_s") is not None and best_item.get("_debug_from_ts_s") is not None:
                                    app.logger.warning(
                                        "CompareFaces debug sec=%d using frame_ts_s=%.3f (forced; bbox from raw ts_s=%.3f)",
                                        int(sec),
                                        float(frame_ts_s),
                                        float(best_item.get("_debug_from_ts_s") or 0.0),
                                    )
                                elif best_item.get("_debug_from_ts_s") is not None:
                                    app.logger.warning(
                                        "CompareFaces debug sec=%d using frame_ts_s=%.3f (matched nearby raw ts_s=%.3f)",
                                        int(sec),
                                        float(frame_ts_s),
                                        float(best_item.get("_debug_from_ts_s") or 0.0),
                                    )

                            if frame_key not in frames_for_compare:
                                fb = _extract_compare_frame_jpeg_bytes(ts_seconds=float(frame_ts_s))
                                if fb:
                                    frames_for_compare[frame_key] = fb
                            frame_bytes = frames_for_compare.get(frame_key)
                            if not frame_bytes:
                                if int(sec) in celeb_compare_debug_seconds:
                                    app.logger.warning("CompareFaces debug sec=%d skipped: no frame bytes", int(sec))
                                continue
                            # If we're forcing the debug frame timestamp (e.g., sec=10 uses frame at 10.000s
                            # but bbox from a nearby raw hit at 9.000s), re-detect faces on the *actual* frame
                            # so CompareFaces gets a meaningful target face crop.
                            bbox0 = best_item.get("bbox")
                            if int(sec) in celeb_compare_debug_seconds and isinstance(best_item, dict):
                                if best_item.get("_debug_force_ts_s") is not None:
                                    df_bbox = _rekognition_detect_largest_face_bbox(rk, frame_bytes)
                                    if isinstance(df_bbox, dict) and df_bbox:
                                        bbox0 = df_bbox
                                        best_item["_debug_bbox_from_detect_faces"] = True
                                        app.logger.warning(
                                            "CompareFaces debug sec=%d bbox_source=detect_faces",
                                            int(sec),
                                        )
                            if not isinstance(bbox0, dict) or not bbox0:
                                if int(sec) in celeb_compare_debug_seconds:
                                    app.logger.warning("CompareFaces debug sec=%d skipped: no bbox", int(sec))
                                continue

                            bbox_for_crop = bbox0
                            try:
                                pad = float(celeb_compare_bbox_pad)
                            except Exception:
                                pad = 0.0
                            if pad > 0.0:
                                try:
                                    l = float(bbox0.get("Left") or 0.0)
                                    t = float(bbox0.get("Top") or 0.0)
                                    w = float(bbox0.get("Width") or 0.0)
                                    h = float(bbox0.get("Height") or 0.0)
                                    l2 = max(0.0, l - w * pad)
                                    t2 = max(0.0, t - h * pad)
                                    w2 = min(1.0 - l2, w * (1.0 + 2.0 * pad))
                                    h2 = min(1.0 - t2, h * (1.0 + 2.0 * pad))
                                    bbox_for_crop = {"Left": l2, "Top": t2, "Width": w2, "Height": h2}
                                except Exception:
                                    bbox_for_crop = bbox0

                            best_name = ""
                            best_sim = -1.0
                            second_name = ""
                            second_sim = -1.0

                            debug_this_sec = int(sec) in celeb_compare_debug_seconds

                            try:
                                compare_crop_min_px = int(
                                    os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_CROP_MIN_PX", "80") or 80
                                )
                            except Exception:
                                compare_crop_min_px = 80

                            target_crop = _pil_crop_bbox_to_jpeg_bytes(
                                frame_bytes, bbox_for_crop, min_size_px=int(compare_crop_min_px)
                            )
                            # Some Rekognition APIs are picky about small target images; if the crop is too
                            # small, fall back to comparing against the full frame.
                            target_bytes_for_compare = target_crop or frame_bytes
                            if debug_this_sec:
                                app.logger.warning(
                                    "CompareFaces debug sec=%d target_image=%s",
                                    int(sec),
                                    "crop" if target_crop else "full_frame",
                                )

                            for nm in comparefaces_candidate_names:
                                pb = portrait_bytes_by_name.get(nm)
                                if not pb:
                                    if debug_this_sec:
                                        app.logger.warning(
                                            "CompareFaces debug sec=%d candidate=%s skipped: no portrait bytes",
                                            int(sec),
                                            str(nm),
                                        )
                                    continue
                                try:
                                    resp = rk.compare_faces(
                                        SourceImage={"Bytes": pb},
                                        TargetImage={"Bytes": target_bytes_for_compare},
                                        SimilarityThreshold=0.0,
                                    )
                                except Exception as exc:
                                    if debug_this_sec:
                                        app.logger.warning(
                                            "CompareFaces debug sec=%d candidate=%s compare_faces error: %s: %s",
                                            int(sec),
                                            str(nm),
                                            exc.__class__.__name__,
                                            str(exc)[:200],
                                        )
                                    continue
                                sims = []
                                for fm in (resp.get("FaceMatches") or []):
                                    if not isinstance(fm, dict):
                                        continue
                                    try:
                                        sims.append(float(fm.get("Similarity") or 0.0))
                                    except Exception:
                                        pass
                                sim = max(sims) if sims else -1.0
                                if debug_this_sec:
                                    try:
                                        match_count = int(len(resp.get("FaceMatches") or []))
                                    except Exception:
                                        match_count = 0
                                    app.logger.warning(
                                        "CompareFaces debug sec=%d candidate=%s sim=%.1f matches=%d",
                                        int(sec),
                                        str(nm),
                                        float(sim),
                                        match_count,
                                    )
                                if sim > best_sim:
                                    second_sim = best_sim
                                    second_name = best_name
                                    best_sim = sim
                                    best_name = nm
                                elif sim > second_sim:
                                    second_sim = sim
                                    second_name = nm

                            if int(sec) in celeb_compare_debug_seconds:
                                detected_nm = (best_item.get("detected_name") or "").strip() if isinstance(best_item, dict) else ""
                                app.logger.warning(
                                    "CompareFaces debug sec=%d detected=%s best=%s sim=%.1f second=%s sim2=%.1f (candidates=%d)",
                                    int(sec),
                                    detected_nm,
                                    (best_name or ""),
                                    float(best_sim),
                                    (second_name or ""),
                                    float(second_sim),
                                    int(len(comparefaces_candidate_names)),
                                )

                            if best_name:
                                record = False
                                if best_sim >= float(celeb_compare_similarity):
                                    record = True
                                elif (
                                    debug_this_sec
                                    and celeb_compare_debug_apply
                                    and best_sim >= float(celeb_compare_debug_min_similarity)
                                    and (best_sim - (second_sim if second_sim >= 0.0 else 0.0)) >= float(celeb_compare_debug_min_delta)
                                ):
                                    record = True
                                    app.logger.warning(
                                        "CompareFaces debug sec=%d APPLY best=%s sim=%.1f second=%s sim2=%.1f (debug_apply=true)",
                                        int(sec),
                                        (best_name or ""),
                                        float(best_sim),
                                        (second_name or ""),
                                        float(second_sim),
                                    )

                                if record:
                                    comparefaces_best_name_by_second[int(sec)] = best_name
                                    comparefaces_best_similarity_by_second[int(sec)] = float(best_sim)
                    finally:
                        try:
                            shutil.rmtree(compare_temp_dir, ignore_errors=True)
                        except Exception:
                            pass
                else:
                    if celeb_compare_debug_seconds:
                        app.logger.warning("CompareFaces: skipped (no compare_source URL)")

            dropped_by_verify = 0
            checked_by_verify = 0
            overridden_by_verify = 0
            dropped_by_comparefaces = 0
            checked_by_comparefaces = 0
            overridden_by_comparefaces = 0

            # Track celebrities that have been accepted at the strict threshold.
            seen_strong_celeb_names: set[str] = set()

            for item in labels_raw:
                try:
                    sec = int(float(item.get("Timestamp") or 0.0) / 1000.0)
                except Exception:
                    sec = 0
                b = _bin_for_second(sec)
                lbl = item.get("Label") or {}
                name = (lbl.get("Name") or "").strip()
                if not name:
                    continue
                try:
                    conf = float(lbl.get("Confidence") or 0.0)
                except Exception:
                    conf = 0.0
                bins[b]["labels"].append({"name": name, "confidence": conf})

            for item in moderation_raw:
                try:
                    sec = int(float(item.get("Timestamp") or 0.0) / 1000.0)
                except Exception:
                    sec = 0
                b = _bin_for_second(sec)
                ml = item.get("ModerationLabel") or {}
                name = (ml.get("Name") or "").strip()
                if not name:
                    continue
                try:
                    conf = float(ml.get("Confidence") or 0.0)
                except Exception:
                    conf = 0.0
                bins[b]["moderation"].append({"name": name, "confidence": conf})

            for item in text_raw:
                try:
                    sec = int(float(item.get("Timestamp") or 0.0) / 1000.0)
                except Exception:
                    sec = 0
                b = _bin_for_second(sec)
                td = item.get("TextDetection") or {}
                if (td.get("Type") or "") != "LINE":
                    continue
                text_val = (td.get("DetectedText") or "").strip()
                if not text_val:
                    continue
                try:
                    conf = float(td.get("Confidence") or 0.0)
                except Exception:
                    conf = 0.0
                bins[b]["text"].append({"type": "LINE", "text": text_val, "confidence": conf})

            for item in celebs_raw:
                try:
                    sec = int(float(item.get("Timestamp") or 0.0) / 1000.0)
                except Exception:
                    sec = 0
                b = _bin_for_second(sec)
                celeb = item.get("Celebrity") or {}
                name = (celeb.get("Name") or "").strip()
                if not name:
                    continue
                urls = [u for u in (celeb.get("Urls") or []) if isinstance(u, str) and u.strip()]
                try:
                    conf = float(item.get("MatchConfidence") or celeb.get("Confidence") or 0.0)
                except Exception:
                    conf = 0.0
                face = celeb.get("Face") if isinstance(celeb.get("Face"), dict) else {}
                bbox = face.get("BoundingBox") if isinstance(face, dict) else None

                min_conf_for_this = float(celeb_min_conf)
                if name in seen_strong_celeb_names:
                    try:
                        min_conf_for_this = min(float(celeb_min_conf), float(celeb_min_conf_seen))
                    except Exception:
                        min_conf_for_this = float(celeb_min_conf)

                if conf < float(min_conf_for_this):
                    continue
                if _bbox_looks_partial_or_too_small(
                    bbox if isinstance(bbox, dict) else None,
                    edge_margin=celeb_edge_margin,
                    min_width=celeb_min_bbox,
                    min_height=celeb_min_bbox,
                ):
                    continue

                if celeb_verify_enable and sec in verified_names_by_second:
                    checked_by_verify += 1
                    verified_names = verified_names_by_second.get(sec) or set()
                    if name not in verified_names:
                        if celeb_verify_allow_override:
                            best_name = (verified_best_name_by_second.get(sec) or "").strip()
                            if best_name and best_name in verified_names and best_name != name:
                                # Only override to a celebrity that Rekognition Video saw somewhere in the video.
                                # This prevents random image-only matches from introducing new names.
                                if best_name in all_video_celeb_names:
                                    name = best_name
                                    urls = urls_by_name.get(best_name) or urls
                                    overridden_by_verify += 1
                                else:
                                    dropped_by_verify += 1
                                    continue
                            else:
                                dropped_by_verify += 1
                                continue
                        else:
                            dropped_by_verify += 1
                            continue

                if celeb_compare_enable and sec in comparefaces_best_name_by_second:
                    checked_by_comparefaces += 1
                    best_name = (comparefaces_best_name_by_second.get(sec) or "").strip()
                    if best_name and best_name != name:
                        sim0 = float(comparefaces_best_similarity_by_second.get(sec) or 0.0)
                        if int(sec) in celeb_compare_debug_seconds:
                            app.logger.warning(
                                "CompareFaces override sec=%d %s -> %s (sim=%.1f)",
                                int(sec),
                                str(name),
                                str(best_name),
                                float(sim0),
                            )
                        if celeb_compare_allow_override:
                            if (best_name not in all_video_celeb_names) and (best_name not in celeb_compare_extra_names):
                                dropped_by_comparefaces += 1
                                continue
                            name = best_name
                            urls = urls_by_name.get(best_name) or urls
                            overridden_by_comparefaces += 1
                        else:
                            dropped_by_comparefaces += 1
                            continue

                # Preserve the original Rekognition Video timestamp (milliseconds) for sub-second UIs.
                try:
                    ts_ms = int(float(item.get("Timestamp") or 0.0))
                except Exception:
                    ts_ms = int(sec) * 1000
                celeb_ts_ms_by_name.setdefault(name, []).append(max(0, int(ts_ms)))

                # Mark as strongly seen only if it met the strict threshold.
                if conf >= float(celeb_min_conf):
                    seen_strong_celeb_names.add(name)

                bins[b]["celebrities"].append({"name": name, "confidence": conf, "urls": urls})
                if isinstance(bbox, dict) and bbox and (name, b) not in celeb_bbox_by_name_and_bin:
                    celeb_bbox_by_name_and_bin[(name, b)] = bbox
                if isinstance(bbox, dict) and bbox:
                    try:
                        area = float(bbox.get("Width") or 0.0) * float(bbox.get("Height") or 0.0)
                    except Exception:
                        area = 0.0
                    prev = best_celeb_thumb.get(name)
                    prev_conf = float(prev.get("confidence") or 0.0) if isinstance(prev, dict) else 0.0
                    prev_area = float(prev.get("area") or 0.0) if isinstance(prev, dict) else 0.0
                    if (prev is None) or (conf > prev_conf) or (conf == prev_conf and area > prev_area):
                        best_celeb_thumb[name] = {"timestamp": int(sec), "bin": int(b), "bbox": bbox, "confidence": conf, "area": area}

            # Supplemental pass: Rekognition Image recognize_celebrities on sampled seconds.
            # Rekognition Video can miss some celebrities entirely; for short clips, a small
            # number of image-based checks improves recall significantly.
            celeb_image_scan_enable = _env_truthy(
                os.getenv("ENVID_METADATA_CELEB_IMAGE_SCAN"),
                default=bool((duration_seconds or 0) > 0 and float(duration_seconds or 0) <= 60.0),
            )
            celeb_image_scan_min_conf = _parse_float_param(
                os.getenv("ENVID_METADATA_CELEB_IMAGE_SCAN_MIN_CONFIDENCE"),
                default=90.0,
                min_value=0.0,
                max_value=100.0,
            )
            celeb_image_scan_min_bbox = _parse_float_param(
                os.getenv("ENVID_METADATA_CELEB_IMAGE_SCAN_MIN_BBOX"),
                default=0.04,
                min_value=0.0,
                max_value=0.5,
            )
            celeb_image_scan_edge_margin = _parse_float_param(
                os.getenv("ENVID_METADATA_CELEB_IMAGE_SCAN_EDGE_MARGIN"),
                default=float(celeb_edge_margin),
                min_value=0.0,
                max_value=0.2,
            )
            celeb_image_scan_max_frames = _parse_int_param(
                os.getenv("ENVID_METADATA_CELEB_IMAGE_SCAN_MAX_FRAMES"),
                default=60,
                min_value=0,
                max_value=600,
            )

            # Offsets (milliseconds) to sample within each second. Helps when faces appear
            # briefly between integer seconds or when keyframe seeking lands slightly off.
            offsets_ms_raw = (os.getenv("ENVID_METADATA_CELEB_IMAGE_SCAN_OFFSETS_MS") or "0,500").strip()
            offsets_ms: list[int] = []
            for part in offsets_ms_raw.split(","):
                p = (part or "").strip()
                if not p:
                    continue
                try:
                    offsets_ms.append(int(float(p)))
                except Exception:
                    continue
            if not offsets_ms:
                offsets_ms = [0]

            if enable_celebrities and celeb_image_scan_enable and int(celeb_image_scan_max_frames or 0) > 0:
                # For short clips we scan every second to cover frame-like timecodes.
                # Clamp total frames to avoid excessive Rekognition calls.
                try:
                    max_sec = int(duration_seconds or 0)
                except Exception:
                    max_sec = 0
                if max_sec <= 0:
                    max_sec = int(max(bins.keys()) if bins else 0)
                duration_ms = int(round(float(max_sec) * 1000.0))
                ts_candidates_ms: list[int] = []
                for s in range(0, max_sec + 1):
                    base_ms = int(s) * 1000
                    for off in offsets_ms:
                        ts = int(base_ms + int(off))
                        if 0 <= ts <= duration_ms:
                            ts_candidates_ms.append(ts)
                # Dedupe + sort
                ts_candidates_ms = sorted(list(dict.fromkeys(ts_candidates_ms)))
                if len(ts_candidates_ms) > int(celeb_image_scan_max_frames):
                    idxs = sorted(_pick_evenly_spaced_indices(len(ts_candidates_ms), int(celeb_image_scan_max_frames)))
                    scan_ts_ms = [int(ts_candidates_ms[i]) for i in idxs]
                else:
                    scan_ts_ms = ts_candidates_ms

                image_scan_hits = 0
                image_scan_checked = 0
                for ts_ms in scan_ts_ms:
                    ts_s = float(ts_ms) / 1000.0
                    fb = _extract_single_frame_jpeg_bytes(ts_seconds=float(ts_s))
                    if not fb:
                        image_scan_checked += 1
                        continue
                    try:
                        resp = rk.recognize_celebrities(Image={"Bytes": fb})
                    except Exception:
                        image_scan_checked += 1
                        continue

                    sec_int = int(max(0.0, ts_s))
                    b = _bin_for_second(int(sec_int))
                    for face in (resp.get("CelebrityFaces") or []):
                        if not isinstance(face, dict):
                            continue
                        nm = (face.get("Name") or "").strip()
                        if not nm:
                            continue
                        try:
                            conf = float(face.get("MatchConfidence") or 0.0)
                        except Exception:
                            conf = 0.0
                        if conf < float(celeb_image_scan_min_conf):
                            continue

                        face_obj = face.get("Face") if isinstance(face.get("Face"), dict) else {}
                        bbox = face_obj.get("BoundingBox") if isinstance(face_obj, dict) else None
                        if _bbox_looks_partial_or_too_small(
                            bbox if isinstance(bbox, dict) else None,
                            edge_margin=float(celeb_image_scan_edge_margin),
                            min_width=float(celeb_image_scan_min_bbox),
                            min_height=float(celeb_image_scan_min_bbox),
                        ):
                            continue

                        urls = [u for u in (face.get("Urls") or []) if isinstance(u, str) and u.strip()]
                        if urls and nm not in urls_by_name:
                            urls_by_name[nm] = urls[:5]
                        all_video_celeb_names.add(nm)

                        celeb_ts_ms_by_name.setdefault(nm, []).append(max(0, int(ts_ms)))
                        if conf >= float(celeb_min_conf):
                            seen_strong_celeb_names.add(nm)

                        bins[b]["celebrities"].append({"name": nm, "confidence": conf, "urls": urls})
                        if isinstance(bbox, dict) and bbox and (nm, b) not in celeb_bbox_by_name_and_bin:
                            celeb_bbox_by_name_and_bin[(nm, b)] = bbox
                        if isinstance(bbox, dict) and bbox:
                            try:
                                area = float(bbox.get("Width") or 0.0) * float(bbox.get("Height") or 0.0)
                            except Exception:
                                area = 0.0
                            prev = best_celeb_thumb.get(nm)
                            prev_conf = float(prev.get("confidence") or 0.0) if isinstance(prev, dict) else 0.0
                            prev_area = float(prev.get("area") or 0.0) if isinstance(prev, dict) else 0.0
                            if (prev is None) or (conf > prev_conf) or (conf == prev_conf and area > prev_area):
                                best_celeb_thumb[nm] = {
                                    "timestamp": int(sec_int),
                                    "bin": int(b),
                                    "bbox": bbox,
                                    "confidence": conf,
                                    "area": area,
                                }
                        image_scan_hits += 1

                    image_scan_checked += 1

                if image_scan_hits:
                    app.logger.info(
                        "Celebrity image-scan added %d hits (checked=%d, min_conf=%.1f, max_frames=%d)",
                        int(image_scan_hits),
                        int(image_scan_checked),
                        float(celeb_image_scan_min_conf),
                        int(celeb_image_scan_max_frames),
                    )

            if celeb_verify_enable and dropped_by_verify:
                app.logger.info(
                    "Celebrity verify (image) dropped %d/%d Rekognition-Video hits (max_frames=%s, min_conf=%s)",
                    dropped_by_verify,
                    checked_by_verify,
                    str(celeb_verify_max_frames),
                    str(celeb_verify_min_conf),
                )
            if celeb_verify_enable and overridden_by_verify:
                app.logger.info(
                    "Celebrity verify (image) overridden %d/%d Rekognition-Video hits (allow_override=%s)",
                    overridden_by_verify,
                    checked_by_verify,
                    str(celeb_verify_allow_override),
                )
            if celeb_compare_enable:
                app.logger.warning(
                    "Celebrity comparefaces checked=%d dropped=%d overridden=%d (candidates=%d, extra=%d, debug=%d, debug_apply=%s, debug_min_sim=%.1f, debug_min_delta=%.1f, pad=%.2f, similarity>=%.1f, allow_override=%s)",
                    checked_by_comparefaces,
                    dropped_by_comparefaces,
                    overridden_by_comparefaces,
                    int(len(comparefaces_candidate_names) if isinstance(comparefaces_candidate_names, list) else 0),
                    int(len(celeb_compare_extra_names) if isinstance(celeb_compare_extra_names, list) else 0),
                    int(len(celeb_compare_debug_seconds) if isinstance(celeb_compare_debug_seconds, set) else 0),
                    str(celeb_compare_debug_apply),
                    float(celeb_compare_debug_min_similarity),
                    float(celeb_compare_debug_min_delta),
                    float(celeb_compare_bbox_pad),
                    float(celeb_compare_similarity),
                    str(celeb_compare_allow_override),
                )

            # Rekognition Video collection matching results mapping removed.

            def _dedupe_by_key(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
                seen = set()
                out: List[Dict[str, Any]] = []
                for it in items:
                    k = (it.get(key) or "").strip() if isinstance(it.get(key), str) else it.get(key)
                    if not k or k in seen:
                        continue
                    seen.add(k)
                    out.append(it)
                return out

            for t, obj in bins.items():
                # Sort by confidence where present, de-duplicate by name/text.
                obj["labels"] = _dedupe_by_key(sorted(obj["labels"], key=lambda x: float(x.get("confidence") or 0.0), reverse=True), "name")[:12]
                obj["moderation"] = _dedupe_by_key(sorted(obj["moderation"], key=lambda x: float(x.get("confidence") or 0.0), reverse=True), "name")[:12]
                obj["text"] = _dedupe_by_key(sorted(obj["text"], key=lambda x: float(x.get("confidence") or 0.0), reverse=True), "text")[:12]
                obj["celebrities"] = _dedupe_by_key(sorted(obj["celebrities"], key=lambda x: float(x.get("confidence") or 0.0), reverse=True), "name")[:12]

            frames_metadata = [bins[t] for t in sorted(bins.keys())]

            # Build aggregates similar to the per-frame pipeline.
            for f in frames_metadata:
                timestamp = f.get("timestamp")
                for label in f.get("labels", []):
                    name = (label.get("name") or "").strip()
                    if not name:
                        continue
                    confidence = float(label.get("confidence") or 0.0)
                    stat = label_stats.setdefault(name, {"count": 0.0, "max": 0.0, "sum": 0.0})
                    stat["count"] += 1.0
                    stat["sum"] += confidence
                    stat["max"] = max(stat["max"], confidence)

                for txt in f.get("text", []):
                    value = (txt.get("text") or "").strip()
                    if not value:
                        continue
                    if txt.get("type") != "LINE":
                        continue
                    text_stats[value] = text_stats.get(value, 0) + 1

                for c in (f.get("celebrities") or []):
                    name = (c.get("name") or "").strip()
                    if not name:
                        continue
                    try:
                        conf = float(c.get("confidence") or 0.0)
                    except (TypeError, ValueError):
                        conf = 0.0
                    urls = [u for u in (c.get("urls") or []) if isinstance(u, str) and u.strip()]
                    stat = celeb_stats.get(name) or {
                        "name": name,
                        "max_confidence": 0.0,
                        "occurrences": 0,
                        "first_seen_seconds": timestamp,
                        "last_seen_seconds": timestamp,
                        "urls": [],
                        "thumbnail": None,
                    }
                    stat["max_confidence"] = max(float(stat.get("max_confidence") or 0.0), conf)
                    stat["occurrences"] = int(stat.get("occurrences") or 0) + 1
                    try:
                        if timestamp is not None:
                            stat["first_seen_seconds"] = (
                                min(int(stat.get("first_seen_seconds") or timestamp), int(timestamp))
                                if stat.get("first_seen_seconds") is not None
                                else int(timestamp)
                            )
                            stat["last_seen_seconds"] = (
                                max(int(stat.get("last_seen_seconds") or timestamp), int(timestamp))
                                if stat.get("last_seen_seconds") is not None
                                else int(timestamp)
                            )
                    except Exception:
                        pass
                    if urls:
                        existing_urls = [u for u in (stat.get("urls") or []) if isinstance(u, str)]
                        merged = list(dict.fromkeys(existing_urls + urls))
                        stat["urls"] = merged[:5]
                    celeb_stats[name] = stat

                for m in (f.get("moderation") or []):
                    name = (m.get("name") or "").strip()
                    if not name:
                        continue
                    try:
                        conf = float(m.get("confidence") or 0.0)
                    except (TypeError, ValueError):
                        conf = 0.0
                    moderation_stats[name] = max(moderation_stats.get(name, 0.0), conf)

            emotions_unique = []
            labels_ranked = sorted(
                (
                    (
                        name,
                        int(stats.get("count", 0.0) or 0),
                        float(stats.get("max", 0.0) or 0.0),
                        float(stats.get("sum", 0.0) or 0.0) / max(1.0, float(stats.get("count", 0.0) or 1.0)),
                    )
                    for name, stats in label_stats.items()
                ),
                key=lambda row: (row[1], row[2], row[0]),
                reverse=True,
            )
            labels_detailed = [
                {"name": name, "occurrences": count, "max_confidence": max_conf, "avg_confidence": avg_conf}
                for (name, count, max_conf, avg_conf) in labels_ranked[:50]
            ]
            text_detailed = [
                {"text": text, "occurrences": count}
                for text, count in sorted(text_stats.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:50]
            ]
            celebs_ranked = sorted(
                list(celeb_stats.values()),
                key=lambda x: (float(x.get("max_confidence") or 0.0), str(x.get("name") or "")),
                reverse=True,
            )[:50]

            bios = _celebrity_bios_for_names([c.get("name") for c in celebs_ranked if c.get("name")])
            for c in celebs_ranked:
                nm = (c.get("name") or "").strip()
                if nm and nm in bios:
                    c["bio"] = bios[nm].get("bio")
                    c["bio_source"] = bios[nm].get("source")
                    c["portrait_url"] = bios[nm].get("portrait_url")
                    c["portrait_source"] = bios[nm].get("portrait_source")
                    c["portrait_license"] = bios[nm].get("portrait_license")
                    c["portrait_license_url"] = bios[nm].get("portrait_license_url")
                    c["portrait_attribution"] = bios[nm].get("portrait_attribution")

            # Build detailed list and attach thumbnails by cropping local frames using Rekognition Video bounding boxes.
            frames_cache: Dict[int, bytes] = {}
            celebs_detailed = []
            for c in celebs_ranked:
                nm = (c.get("name") or "").strip()
                if not nm:
                    continue
                best = best_celeb_thumb.get(nm) or {}
                ts_for_crop = int(best.get("timestamp") or 0)
                b = int(best.get("bin") or _bin_for_second(ts_for_crop))
                bbox = best.get("bbox") if isinstance(best.get("bbox"), dict) else celeb_bbox_by_name_and_bin.get((nm, b))
                thumb_b64 = None
                if isinstance(bbox, dict) and bbox:
                    if ts_for_crop not in frames_cache:
                        fb = _extract_single_frame_jpeg_bytes(ts_seconds=ts_for_crop)
                        if fb:
                            frames_cache[ts_for_crop] = fb
                    if ts_for_crop in frames_cache:
                        thumb_b64 = _pil_crop_bbox_thumbnail_base64(frames_cache[ts_for_crop], bbox)

                celebs_detailed.append(
                    {
                        "name": nm,
                        "max_confidence": float(c.get("max_confidence") or 0.0),
                        "occurrences": int(c.get("occurrences") or 0),
                        "first_seen_seconds": c.get("first_seen_seconds"),
                        "last_seen_seconds": c.get("last_seen_seconds"),
                        "urls": c.get("urls") or [],
                        "thumbnail": thumb_b64,
                        "bio": (c.get("bio") or "").strip(),
                        "bio_source": c.get("bio_source") or None,
                        "portrait_url": (c.get("portrait_url") or "").strip() or None,
                        "portrait_source": (c.get("portrait_source") or "").strip() or None,
                        "portrait_license": (c.get("portrait_license") or "").strip() or None,
                        "portrait_license_url": (c.get("portrait_license_url") or "").strip() or None,
                        "portrait_attribution": (c.get("portrait_attribution") or "").strip() or None,
                    }
                )

            celebs_detailed = _attach_celebrity_timestamps(frames=frames_metadata, celebrities_detailed=celebs_detailed)

            # Build collection matches summary and attach thumbnails (best-effort).
            for f in frames_metadata:
                ts = f.get("timestamp")
                for m in (f.get("custom_faces") or []):
                    nm = (m.get("name") or "").strip()
                    if not nm:
                        continue
                    try:
                        sim = float(m.get("similarity") or 0.0)
                    except Exception:
                        sim = 0.0
                    stat = custom_face_stats.get(nm) or {
                        "name": nm,
                        "max_similarity": 0.0,
                        "occurrences": 0,
                        "first_seen_seconds": ts,
                        "last_seen_seconds": ts,
                        "thumbnail": None,
                    }
                    stat["max_similarity"] = max(float(stat.get("max_similarity") or 0.0), sim)
                    stat["occurrences"] = int(stat.get("occurrences") or 0) + 1
                    try:
                        if ts is not None:
                            stat["first_seen_seconds"] = (
                                min(int(stat.get("first_seen_seconds") or ts), int(ts))
                                if stat.get("first_seen_seconds") is not None
                                else int(ts)
                            )
                            stat["last_seen_seconds"] = (
                                max(int(stat.get("last_seen_seconds") or ts), int(ts))
                                if stat.get("last_seen_seconds") is not None
                                else int(ts)
                            )
                    except Exception:
                        pass
                    custom_face_stats[nm] = stat

            custom_ranked = sorted(
                list(custom_face_stats.values()),
                key=lambda x: (float(x.get("max_similarity") or 0.0), str(x.get("name") or "")),
                reverse=True,
            )[:50]

            custom_faces_detailed = []
            for c in custom_ranked:
                nm = (c.get("name") or "").strip()
                if not nm:
                    continue
                first_seen = c.get("first_seen_seconds")
                ts_for_crop = int(first_seen) if first_seen is not None else 0
                b = _bin_for_second(ts_for_crop)
                bbox = custom_bbox_by_name_and_bin.get((nm, b))
                thumb_b64 = None
                if isinstance(bbox, dict) and bbox:
                    if b not in frames_cache:
                        fb = _extract_single_frame_jpeg_bytes(ts_seconds=b)
                        if fb:
                            frames_cache[b] = fb
                    if b in frames_cache:
                        thumb_b64 = _pil_crop_bbox_thumbnail_base64(frames_cache[b], bbox)

                custom_faces_detailed.append(
                    {
                        "name": nm,
                        "max_similarity": float(c.get("max_similarity") or 0.0),
                        "occurrences": int(c.get("occurrences") or 0),
                        "first_seen_seconds": c.get("first_seen_seconds"),
                        "last_seen_seconds": c.get("last_seen_seconds"),
                        "thumbnail": thumb_b64,
                    }
                )
            moderation_detailed = [
                {"name": name, "max_confidence": conf}
                for name, conf in sorted(moderation_stats.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:50]
            ]

            # Provide an overall video thumbnail (middle frame) for history cards.
            mid_ts = int(float(duration_seconds or 0.0) / 2.0) if duration_seconds else 0
            mid_bytes = _extract_single_frame_jpeg_bytes(ts_seconds=mid_ts)
            thumbnail_base64 = base64.b64encode(mid_bytes).decode("utf-8") if mid_bytes else None

            # For compatibility with the rest of the function.
            emotions_unique = []

            # Attach millisecond-level timestamps (if we captured them) for better overlay matching.
            if isinstance(celebs_detailed, list) and celebs_detailed:
                for c in celebs_detailed:
                    if not isinstance(c, dict):
                        continue
                    nm = (c.get("name") or "").strip()
                    if not nm:
                        continue
                    ts_ms = celeb_ts_ms_by_name.get(nm) or []
                    try:
                        ts_ms_norm = sorted(list(dict.fromkeys([int(x) for x in ts_ms if x is not None])))
                    except Exception:
                        ts_ms_norm = []
                    if ts_ms_norm:
                        c["timestamps_ms"] = ts_ms_norm[:240]

        else:
            if _disable_per_frame_analysis():
                raise RuntimeError(
                    "Per-frame analysis is disabled (ENVID_METADATA_DISABLE_PER_FRAME_ANALYSIS=true). "
                    "Use ENVID_METADATA_REKOGNITION_ANALYSIS_SOURCE=s3_video (or the required/minimal profile)."
                )
            _job_update(
                job_id,
                status="processing",
                progress=3,
                message=f"Extracting frames (every {frame_interval_seconds}s)",
            )

            frames_dir = temp_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            frames = _extract_video_frames(video_path, frames_dir, interval=frame_interval_seconds)

            _job_update(
                job_id,
                progress=15,
                message=f"Analyzing frames ({min(len(frames), max_frames_to_analyze)} of {len(frames)})",
            )

            frames_to_process = frames[:max_frames_to_analyze]
            total = len(frames_to_process)
            if total <= 0:
                raise RuntimeError("No frames extracted")

            # Dependencies: celebrity detection needs faces; face matching modes need faces.
            if enable_celebrities:
                enable_faces = True
            if face_recognition_mode in {"aws_collection", "local"}:
                enable_faces = True

            max_workers = _parse_int_param(
                os.getenv("ENVID_METADATA_FRAME_WORKERS"),
                default=4,
                min_value=1,
                max_value=16,
            )
        if not use_rekognition_video:
            max_label_frames = _parse_int_param(
                os.getenv("ENVID_METADATA_MAX_LABEL_FRAMES"),
                default=6,
                min_value=0,
                max_value=200,
            )
            max_text_frames = _parse_int_param(
                os.getenv("ENVID_METADATA_MAX_TEXT_FRAMES"),
                default=4,
                min_value=0,
                max_value=200,
            )
            max_moderation_frames = _parse_int_param(
                os.getenv("ENVID_METADATA_MAX_MODERATION_FRAMES"),
                default=4,
                min_value=0,
                max_value=200,
            )
            max_face_frames = _parse_int_param(
                os.getenv("ENVID_METADATA_MAX_FACE_FRAMES"),
                default=0,
                min_value=0,
                max_value=200,
            )

            if not enable_labels:
                max_label_frames = 0
            if not enable_text:
                max_text_frames = 0
            if not enable_moderation:
                max_moderation_frames = 0

            label_indices = _pick_evenly_spaced_indices(total, max_label_frames) if max_label_frames > 0 else set()
            text_indices = _pick_evenly_spaced_indices(total, max_text_frames) if max_text_frames > 0 else set()
            moderation_indices = (
                _pick_evenly_spaced_indices(total, max_moderation_frames) if max_moderation_frames > 0 else set()
            )
            if enable_faces:
                face_indices = (
                    set(range(total)) if max_face_frames == 0 else _pick_evenly_spaced_indices(total, max_face_frames)
                )
            else:
                face_indices = set()

            if face_recognition_mode == "aws_collection":
                # Collection matching depends on AWS face detection.
                face_indices = set(range(total))

                analyses: List[Dict[str, Any]] = [
                    {
                        "labels": [],
                        "text": [],
                        "faces": [],
                        "celebrities": [],
                        "moderation": [],
                        "custom_faces": [],
                        "local_faces": [],
                        "celebrity_checked": False,
                    }
                    for _ in range(total)
                ]

                def _analyze_one(idx: int, frame_path: Path) -> Dict[str, Any]:
                    return _analyze_frame_with_rekognition(
                        frame_path,
                        collection_id=collection_id if (face_recognition_mode == "aws_collection") else None,
                        local_face_mode=bool(face_recognition_mode == "local"),
                        enable_label_detection=idx in label_indices,
                        enable_text_detection=idx in text_indices,
                        enable_face_detection=idx in face_indices,
                        enable_moderation_detection=idx in moderation_indices,
                        enable_celebrity_detection=False,
                    )

                completed = 0
                with ThreadPoolExecutor(max_workers=min(max_workers, total)) as executor:
                    future_to_idx = {
                        executor.submit(_analyze_one, idx, frame_path): idx
                        for idx, frame_path in enumerate(frames_to_process)
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            analyses[idx] = future.result()
                        except Exception as exc:
                            app.logger.error("Frame analysis failed (%s): %s", frames_to_process[idx], exc)
                        completed += 1
                        if completed % 3 == 0 or completed == total:
                            pct = 15 + int((completed / max(1, total)) * 45)
                            _job_update(job_id, progress=pct, message=f"Analyzing frames ({completed}/{total})")

                max_celebrity_frames = _parse_int_param(
                    os.getenv("ENVID_METADATA_MAX_CELEBRITY_FRAMES"),
                    default=8,
                    min_value=0,
                    max_value=200,
                )
                if not enable_celebrities:
                    max_celebrity_frames = 0
                if max_celebrity_frames > 0:
                    candidate_indices = [idx for idx, a in enumerate(analyses) if (a.get("faces") or [])]
                    if candidate_indices:
                        pick = _pick_evenly_spaced_indices(
                            len(candidate_indices),
                            min(int(max_celebrity_frames), len(candidate_indices)),
                        )
                        chosen_indices = [candidate_indices[i] for i in sorted(pick)]
                        with ThreadPoolExecutor(max_workers=min(max_workers, len(chosen_indices))) as executor:
                            future_to_idx = {
                                executor.submit(_recognize_celebrities_for_frame, frames_to_process[idx]): idx
                                for idx in chosen_indices
                            }
                            for future in as_completed(future_to_idx):
                                idx = future_to_idx[future]
                                try:
                                    analyses[idx]["celebrities"] = future.result()
                                    analyses[idx]["celebrity_checked"] = True
                                except Exception as exc:
                                    app.logger.error("Celebrity detection failed (%s): %s", frames_to_process[idx], exc)

                for i, frame in enumerate(frames_to_process):
                    analysis = analyses[i]
                    timestamp = _frame_timestamp_seconds(frame)

                    # Aggregate labels with confidence
                    for label in analysis.get("labels", []):
                        name = (label.get("name") or "").strip()
                        if not name:
                            continue
                        confidence = float(label.get("confidence") or 0.0)
                        stat = label_stats.setdefault(name, {"count": 0.0, "max": 0.0, "sum": 0.0})
                        stat["count"] += 1.0
                        stat["sum"] += confidence
                        stat["max"] = max(stat["max"], confidence)

                    # Aggregate text (prefer LINE)
                    for txt in analysis.get("text", []):
                        value = (txt.get("text") or "").strip()
                        if not value:
                            continue
                        if txt.get("type") != "LINE":
                            continue
                        text_stats[value] = text_stats.get(value, 0) + 1

                    # Aggregate faces
                    for face in analysis.get("faces", []):
                        for e in (face.get("emotions") or []):
                            et = (e.get("type") or "").strip()
                            if et:
                                all_emotions.append(et)
                        gender = (face.get("gender") or "").strip()
                        if gender:
                            gender_counts[gender] = gender_counts.get(gender, 0) + 1
                        age = face.get("age_range") or {}
                        try:
                            low = int(age.get("Low"))
                            high = int(age.get("High"))
                            age_ranges.append((low, high))
                        except Exception:
                            pass

                    # Aggregate celebrities
                    for c in (analysis.get("celebrities") or []):
                        name = (c.get("name") or "").strip()
                        if not name:
                            continue
                        try:
                            conf = float(c.get("confidence") or 0.0)
                        except (TypeError, ValueError):
                            conf = 0.0
                        urls = [u for u in (c.get("urls") or []) if isinstance(u, str) and u.strip()]
                        thumb = c.get("thumbnail")
                        stat = celeb_stats.get(name) or {
                            "name": name,
                            "max_confidence": 0.0,
                            "occurrences": 0,
                            "first_seen_seconds": timestamp,
                            "last_seen_seconds": timestamp,
                            "urls": [],
                            "thumbnail": None,
                        }
                        stat["max_confidence"] = max(float(stat.get("max_confidence") or 0.0), conf)
                        stat["occurrences"] = int(stat.get("occurrences") or 0) + 1
                        try:
                            if timestamp is not None:
                                stat["first_seen_seconds"] = (
                                    min(int(stat.get("first_seen_seconds") or timestamp), int(timestamp))
                                    if stat.get("first_seen_seconds") is not None
                                    else int(timestamp)
                                )
                                stat["last_seen_seconds"] = (
                                    max(int(stat.get("last_seen_seconds") or timestamp), int(timestamp))
                                    if stat.get("last_seen_seconds") is not None
                                    else int(timestamp)
                                )
                        except Exception:
                            pass
                        if urls:
                            existing_urls = [u for u in (stat.get("urls") or []) if isinstance(u, str)]
                            merged = list(dict.fromkeys(existing_urls + urls))
                            stat["urls"] = merged[:5]
                        if thumb and not stat.get("thumbnail"):
                            stat["thumbnail"] = thumb
                        celeb_stats[name] = stat

                    # Aggregate custom collection matches
                    for c in (analysis.get("custom_faces") or []):
                        name = (c.get("name") or "").strip()
                        if not name:
                            continue
                        try:
                            sim = float(c.get("similarity") or 0.0)
                        except (TypeError, ValueError):
                            sim = 0.0
                        custom_face_stats[name] = max(custom_face_stats.get(name, 0.0), sim)

                    # Aggregate local matches
                    for c in (analysis.get("local_faces") or []):
                        name = (c.get("name") or "").strip()
                        if not name:
                            continue
                        try:
                            sim = float(c.get("similarity") or 0.0)
                        except (TypeError, ValueError):
                            sim = 0.0
                        local_face_stats[name] = max(local_face_stats.get(name, 0.0), sim)

                    # Aggregate moderation
                    for m in (analysis.get("moderation") or []):
                        name = (m.get("name") or "").strip()
                        if not name:
                            continue
                        try:
                            conf = float(m.get("confidence") or 0.0)
                        except (TypeError, ValueError):
                            conf = 0.0
                        moderation_stats[name] = max(moderation_stats.get(name, 0.0), conf)

                    frames_metadata.append(
                        {
                            "timestamp": timestamp,
                            "labels": (analysis.get("labels") or [])[:12],
                            "text": [t for t in (analysis.get("text") or []) if t.get("type") == "LINE"][:12],
                            "faces": (analysis.get("faces") or [])[:12],
                            "celebrities": (analysis.get("celebrities") or [])[:12],
                            "custom_faces": (analysis.get("custom_faces") or [])[:12],
                            "local_faces": (analysis.get("local_faces") or [])[:12],
                            "moderation": (analysis.get("moderation") or [])[:12],
                        }
                    )

                emotions_unique = sorted(list(set(all_emotions)))

                labels_ranked = sorted(
                    (
                        (
                            name,
                            int(stats.get("count", 0.0) or 0),
                            float(stats.get("max", 0.0) or 0.0),
                            float(stats.get("sum", 0.0) or 0.0) / max(1.0, float(stats.get("count", 0.0) or 1.0)),
                        )
                        for name, stats in label_stats.items()
                    ),
                    key=lambda row: (row[1], row[2], row[0]),
                    reverse=True,
                )
                labels_detailed = [
                    {"name": name, "occurrences": count, "max_confidence": max_conf, "avg_confidence": avg_conf}
                    for (name, count, max_conf, avg_conf) in labels_ranked[:50]
                ]

                text_detailed = [
                    {"text": text, "occurrences": count}
                    for text, count in sorted(text_stats.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:50]
                ]

                celebs_ranked = sorted(
                    list(celeb_stats.values()),
                    key=lambda x: (float(x.get("max_confidence") or 0.0), str(x.get("name") or "")),
                    reverse=True,
                )[:50]

                celebs_detailed = _attach_celebrity_timestamps(frames=frames_metadata, celebrities_detailed=list(celebs_ranked))

                moderation_detailed = [
                    {"name": name, "max_confidence": conf}
                    for name, conf in sorted(moderation_stats.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:50]
                ]

        transcript = ""
        transcript_words: List[Dict[str, Any]] = []
        transcript_segments: List[Dict[str, Any]] = []
        transcript_language_code: str | None = None
        languages_detected: List[str] = []
        if enable_transcribe:
            _job_update(job_id, progress=65, message="Transcribing audio")
            _job_step_update(job_id, "transcribe", status="running", percent=0, message="Transcribing")
            transcribe_rich = _extract_audio_and_transcribe_rich(video_path, temp_dir, job_id)
            transcript = transcribe_rich.get("text") or ""
            transcript_words = transcribe_rich.get("words") or []
            transcript_segments = transcribe_rich.get("segments") or []
            transcript_language_code = (transcribe_rich.get("language_code") or "").strip() or None

            # Clean up spacing/punctuation for readability. Optional normalization is best-effort
            # and MUST NOT invent missing words.
            transcript = _normalize_transcript_basic(transcript)

            # Optional deterministic user patches (e.g., fix recurring ASR misses like a missing word).
            transcript = _apply_transcript_patches(transcript, language_code=transcript_language_code)
            transcript = _apply_language_spelling_fixes(transcript, language_code=transcript_language_code)

            if transcript_segments:
                for seg in transcript_segments:
                    if isinstance(seg, dict) and seg.get("text"):
                        seg["text"] = _normalize_transcript_basic(str(seg.get("text") or ""))
                        seg["text"] = _apply_transcript_patches(str(seg.get("text") or ""), language_code=transcript_language_code)
                        seg["text"] = _apply_language_spelling_fixes(str(seg.get("text") or ""), language_code=transcript_language_code)
            languages_detected = _detect_dominant_languages_from_text(transcript)
            _job_step_update(job_id, "transcribe", status="completed", percent=100, message="Completed")
        else:
            _job_update(job_id, progress=65, message="Transcribe: disabled")
            _job_step_update(job_id, "transcribe", status="skipped", percent=100, message="Disabled")

        subtitles: Dict[str, Any] = {}
        if transcript_segments:
            try:
                srt_text = _segments_to_srt(transcript_segments)
                vtt_text = _segments_to_vtt(transcript_segments)
                subtitles = {"language_code": transcript_language_code}

                # Cloud-only: persist subtitles to S3 (original + optional English translation).
                bucket = _metadata_artifacts_s3_bucket()
                srt_key = _subtitles_s3_key(job_id, lang="orig", fmt="srt")
                vtt_key = _subtitles_s3_key(job_id, lang="orig", fmt="vtt")
                _s3_put_bytes(
                    bucket=bucket,
                    key=srt_key,
                    body=srt_text.encode("utf-8"),
                    content_type="application/x-subrip",
                )
                _s3_put_bytes(
                    bucket=bucket,
                    key=vtt_key,
                    body=vtt_text.encode("utf-8"),
                    content_type="text/vtt",
                )

                s3_block: Dict[str, Any] = {
                    "bucket": bucket,
                    "srt_key": srt_key,
                    "vtt_key": vtt_key,
                    "srt_uri": f"s3://{bucket}/{srt_key}",
                    "vtt_uri": f"s3://{bucket}/{vtt_key}",
                }

                try:
                    enable_translate = _feature_enabled("TRANSLATE", default=True)
                except Exception:
                    enable_translate = True
                if enable_translate:
                    en_segments = _translate_segments_to_english(transcript_segments, source_language_code=transcript_language_code)
                    if en_segments:
                        srt_en_text = _segments_to_srt(en_segments)
                        vtt_en_text = _segments_to_vtt(en_segments)
                        srt_en_key = _subtitles_s3_key(job_id, lang="en", fmt="srt")
                        vtt_en_key = _subtitles_s3_key(job_id, lang="en", fmt="vtt")
                        _s3_put_bytes(
                            bucket=bucket,
                            key=srt_en_key,
                            body=srt_en_text.encode("utf-8"),
                            content_type="application/x-subrip",
                        )
                        _s3_put_bytes(
                            bucket=bucket,
                            key=vtt_en_key,
                            body=vtt_en_text.encode("utf-8"),
                            content_type="text/vtt",
                        )
                        subtitles["en"] = {"language_code": "en"}
                        s3_block["en"] = {
                            "srt_key": srt_en_key,
                            "vtt_key": vtt_en_key,
                            "srt_uri": f"s3://{bucket}/{srt_en_key}",
                            "vtt_uri": f"s3://{bucket}/{vtt_en_key}",
                        }

                subtitles["s3"] = s3_block
            except Exception:
                subtitles = {}

        locations_text = _comprehend_locations_from_text(transcript, language_code=transcript_language_code)
        famous_locations = _build_famous_locations_payload(
            text_locations=locations_text if isinstance(locations_text, list) else [],
            transcript_segments=transcript_segments if isinstance(transcript_segments, list) else [],
            frames=frames_metadata if isinstance(frames_metadata, list) else [],
            geocode_cache=None,
        )
        locations = famous_locations.get("locations") if isinstance(famous_locations, dict) else (locations_text or [])
        synopses_by_age = _bedrock_synopsis_by_age_group(transcript or video_description or "", title=video_title)

        scenes_pack = _build_scene_by_scene_metadata(
            frames=frames_metadata,
            transcript_segments=transcript_segments,
            duration_seconds=duration_seconds,
            scenes=scene_windows_override,
            source=scene_segmentation_source,
            window_seconds=_parse_int_param(os.getenv("ENVID_METADATA_SCENE_WINDOW_SECONDS"), default=15, min_value=5, max_value=120),
        )

        metadata_parts = [video_title, video_description]
        if transcript:
            metadata_parts.append(f"Transcript: {transcript}")
        if labels_detailed:
            metadata_parts.append("Visual elements: " + ", ".join([l["name"] for l in labels_detailed[:50] if l.get("name")]))
        if text_detailed:
            metadata_parts.append("Text in video: " + ", ".join([t["text"] for t in text_detailed[:20] if t.get("text")]))
        if emotions_unique:
            metadata_parts.append(f"Emotions detected: {', '.join(emotions_unique)}")
        if celebs_detailed:
            metadata_parts.append("Celebrities detected: " + ", ".join([c["name"] for c in celebs_detailed[:25] if c.get("name")]))
        if custom_faces_detailed:
            metadata_parts.append(
                "Cast detected (collection): "
                + ", ".join([c["name"] for c in custom_faces_detailed[:25] if c.get("name")])
            )
        if moderation_detailed:
            metadata_parts.append("Moderation labels: " + ", ".join([m["name"] for m in moderation_detailed[:30] if m.get("name")]))

        metadata_text = " ".join(metadata_parts)
        embedding: List[float] = []
        if enable_embedding:
            _job_update(job_id, progress=78, message="Generating embedding")
            _job_step_update(job_id, "embedding", status="running", percent=0, message="Generating")
            embedding = _generate_embedding(metadata_text)
            _job_step_update(job_id, "embedding", status="completed", percent=100, message="Completed")
        else:
            _job_update(job_id, progress=78, message="Embedding: disabled")
            _job_step_update(job_id, "embedding", status="skipped", percent=100, message="Disabled")

        _job_update(job_id, progress=88, message="Saving & indexing")
        _job_step_update(job_id, "saving_indexing", status="running", percent=0, message="Saving")
        # Note: in Rekognition-Video mode we already computed `thumbnail_base64`.
        if not use_rekognition_video:
            thumbnail_path = frames[len(frames) // 2] if frames else None
            thumbnail_base64 = None
            if thumbnail_path and thumbnail_path.exists():
                with open(thumbnail_path, "rb") as f:
                    thumbnail_base64 = base64.b64encode(f.read()).decode("utf-8")

        file_extension = Path(original_filename).suffix or ".mp4"
        stored_filename = ""
        if _persist_local_video_copy():
            stored_filename = f"{job_id}{file_extension}"
            stored_video_path = VIDEOS_DIR / stored_filename
            shutil.copy2(video_path, stored_video_path)

        technical_ffprobe = _probe_technical_metadata(video_path)

        file_size_bytes: int | None = None
        try:
            file_size_bytes = int(video_path.stat().st_size)
        except Exception:
            file_size_bytes = None

        video_entry = {
            "id": job_id,
            "title": video_title,
            "description": video_description,
            "original_filename": original_filename,
            "stored_filename": stored_filename,
            "file_path": _relative_video_path(stored_filename) if stored_filename else "",
            "s3_video_key": s3_video_key,
            "s3_video_uri": s3_video_uri,
            "transcript": transcript,
            "language_code": transcript_language_code,
            "languages_detected": languages_detected,
            "transcript_words": transcript_words,
            "transcript_segments": transcript_segments,
            "labels": [l["name"] for l in labels_detailed if l.get("name")][:50],
            "labels_detailed": labels_detailed,
            "text_detected": [t["text"] for t in text_detailed if t.get("text")][:20],
            "text_detailed": text_detailed,
            "emotions": emotions_unique,
            "celebrities": [c["name"] for c in celebs_detailed if c.get("name")][:50],
            "celebrities_detailed": celebs_detailed,
            "face_recognition_mode": face_recognition_mode,
            "moderation_labels": [m["name"] for m in moderation_detailed if m.get("name")][:50],
            "moderation_detailed": moderation_detailed,
            "faces_summary": {
                "count": sum(gender_counts.values()),
                "genders": gender_counts,
                "age_ranges": age_ranges[:200],
            },
            "embedding": embedding,
            "thumbnail": thumbnail_base64,
            "metadata_text": metadata_text,
            "frame_count": int(len(frames_metadata) or 0),
            "frames_analyzed": int(len(frames_metadata) or 0),
            "frame_interval_seconds": frame_interval_seconds,
            "duration_seconds": duration_seconds,
            "frames": frames_metadata,
            "uploaded_at": datetime.utcnow().isoformat(),
            "technical_ffprobe": technical_ffprobe,
            "file_size_bytes": file_size_bytes,

            # Scene segmentation provenance (useful for debugging and replay)
            "scene_segmentation_source": scene_segmentation_source,
            "rekognition_shots": rekognition_shots,

            # Required-output extras
            "subtitles": subtitles,
            "locations": locations,
            "famous_locations": famous_locations,
            "locations_geocoded": (famous_locations.get("geocode_cache") if isinstance(famous_locations, dict) else None),
            "synopses_by_age_group": synopses_by_age,
            "scene_metadata": scenes_pack,
            "output_profile": profile,
        }

        categorized = _build_categorized_metadata_json(video_entry)
        video_entry["metadata_categories"] = categorized.get("categories")
        video_entry["metadata_combined"] = categorized.get("combined")

        _ensure_metadata_artifacts_on_s3(video_entry, save_index=False)

        existing_idx = next(
            (i for i, v in enumerate(VIDEO_INDEX) if str(v.get("id")) == str(video_entry.get("id"))),
            None,
        )
        if existing_idx is not None:
            VIDEO_INDEX[existing_idx] = video_entry
        else:
            VIDEO_INDEX.append(video_entry)
        _save_video_index()

        result = {
            "id": job_id,
            "title": video_title,
            "message": "Video uploaded and indexed successfully",
            "labels_count": len(label_stats),
            "celebrities_count": len(celeb_stats),
            "transcript_length": len(transcript),
            "frame_count": len(frames),
            "language_code": transcript_language_code,
            "s3_video_uri": s3_video_uri,
        }
        _job_step_update(job_id, "saving_indexing", status="completed", percent=100, message="Completed")
        _job_update(job_id, status="completed", progress=100, message="Completed", result=result)
    except Exception as exc:
        app.logger.error("Job %s failed: %s", job_id, exc)
        import traceback

        app.logger.error(traceback.format_exc())
        _job_step_update(job_id, "saving_indexing", status="failed", message=str(exc))
        _job_update(job_id, status="failed", message="Failed", error=str(exc))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

DEFAULT_AWS_REGION = os.environ.get("AWS_REGION")
if not DEFAULT_AWS_REGION:
    raise RuntimeError("Set AWS_REGION before starting semantic search service")

# AWS clients
rekognition = boto3.client("rekognition", region_name=DEFAULT_AWS_REGION)
transcribe = boto3.client("transcribe", region_name=DEFAULT_AWS_REGION)
comprehend = boto3.client("comprehend", region_name=DEFAULT_AWS_REGION)
bedrock_runtime = boto3.client("bedrock-runtime", region_name=DEFAULT_AWS_REGION)
s3 = boto3.client("s3", region_name=DEFAULT_AWS_REGION)
translate = boto3.client("translate", region_name=DEFAULT_AWS_REGION)
polly = boto3.client("polly", region_name=DEFAULT_AWS_REGION)
location_svc = boto3.client("location", region_name=DEFAULT_AWS_REGION)


LOCAL_FACE_GALLERY_FILE: Path | None = None
_INSIGHTFACE_ANALYZER = None


def _safe_json_load(path: Path, default_value: Any) -> Any:
    try:
        if not path.exists():
            return default_value
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default_value


def _safe_json_save(path: Path, value: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(value, f, indent=2, default=str)
    os.replace(tmp_path, path)


def _rekognition_collection_id_normalize(raw: str | None) -> str:
    return (raw or "").strip()


def _rekognition_external_image_id_normalize(raw: str | None) -> str:
    # ExternalImageId: keep simple (avoid very long ids); Rekognition accepts Unicode,
    # but we keep it compact for UI display.
    value = (raw or "").strip()
    value = re.sub(r"\s+", " ", value)
    return value[:128]


def _pil_crop_bbox_to_jpeg_bytes(image_bytes: bytes, bbox: Dict[str, Any] | None, *, min_size_px: int = 40) -> bytes | None:
    if not bbox:
        return None
    try:
        left = float(bbox.get("Left", 0.0) or 0.0)
        top = float(bbox.get("Top", 0.0) or 0.0)
        width = float(bbox.get("Width", 0.0) or 0.0)
        height = float(bbox.get("Height", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if width <= 0.0 or height <= 0.0:
        return None

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            w, h = img.size
            x1 = max(0, min(w - 1, int(left * w)))
            y1 = max(0, min(h - 1, int(top * h)))
            x2 = max(0, min(w, int((left + width) * w)))
            y2 = max(0, min(h, int((top + height) * h)))
            if x2 <= x1 or y2 <= y1:
                return None
            if (x2 - x1) < min_size_px or (y2 - y1) < min_size_px:
                return None

            crop = img.crop((x1, y1, x2, y2))
            out = io.BytesIO()
            crop.save(out, format="JPEG", quality=85)
            return out.getvalue()
    except Exception:
        return None


def _pil_crop_bbox_thumbnail_base64(image_bytes: bytes, bbox: Dict[str, Any] | None, *, size_px: int = 256) -> str | None:
    if not _env_truthy(os.getenv("ENVID_METADATA_ENABLE_THUMBNAIL_CROPPING"), default=True):
        return None

    size_px = _parse_int_param(
        os.getenv("ENVID_METADATA_THUMBNAIL_SIZE_PX"),
        default=int(size_px),
        min_value=64,
        max_value=512,
    )
    if not bbox:
        return None
    try:
        left = float(bbox.get("Left", 0.0) or 0.0)
        top = float(bbox.get("Top", 0.0) or 0.0)
        width = float(bbox.get("Width", 0.0) or 0.0)
        height = float(bbox.get("Height", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if width <= 0.0 or height <= 0.0:
        return None

    # Rekognition returns a tight face box; pad it for better UX.
    pad = _parse_float_param(
        os.getenv("ENVID_METADATA_THUMBNAIL_PAD_FRACTION"),
        default=0.25,
        min_value=0.0,
        max_value=1.0,
    )
    try:
        if pad > 0:
            cx = left + (width / 2.0)
            cy = top + (height / 2.0)
            width = width * (1.0 + pad * 2.0)
            height = height * (1.0 + pad * 2.0)
            left = cx - (width / 2.0)
            top = cy - (height / 2.0)
            left = max(0.0, min(1.0, left))
            top = max(0.0, min(1.0, top))
            width = max(0.0, min(1.0 - left, width))
            height = max(0.0, min(1.0 - top, height))
    except Exception:
        pass

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            w, h = img.size
            x1 = max(0, min(w - 1, int(left * w)))
            y1 = max(0, min(h - 1, int(top * h)))
            x2 = max(0, min(w, int((left + width) * w)))
            y2 = max(0, min(h, int((top + height) * h)))
            if x2 <= x1 or y2 <= y1:
                return None

            crop = img.crop((x1, y1, x2, y2))

            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", getattr(Image, "LANCZOS", 1))
            try:
                crop = ImageOps.fit(crop, (size_px, size_px), method=resample)
            except TypeError:
                crop = ImageOps.fit(crop, (size_px, size_px), resample=resample)
            out = io.BytesIO()
            crop.save(out, format="JPEG", quality=92)
            return base64.b64encode(out.getvalue()).decode("utf-8")
    except Exception:
        return None


def _bbox_looks_partial_or_too_small(
    bbox: Dict[str, Any] | None,
    *,
    edge_margin: float = 0.02,
    min_width: float = 0.06,
    min_height: float = 0.06,
) -> bool:
    """Best-effort heuristic to reject partial faces.

    Rekognition can sometimes return celebrity matches for occluded/partial faces.
    We treat a face as likely-partial if the bounding box is too small or touches
    the image edges (meaning the face may be cropped off).
    """
    if not isinstance(bbox, dict) or not bbox:
        return True
    try:
        left = float(bbox.get("Left") or 0.0)
        top = float(bbox.get("Top") or 0.0)
        width = float(bbox.get("Width") or 0.0)
        height = float(bbox.get("Height") or 0.0)
    except Exception:
        return True
    if width <= 0.0 or height <= 0.0:
        return True
    if width < float(min_width) or height < float(min_height):
        return True
    right = left + width
    bottom = top + height
    m = float(edge_margin)
    if left <= m or top <= m or right >= (1.0 - m) or bottom >= (1.0 - m):
        return True
    return False


def _insightface_available() -> tuple[bool, str | None]:
    if np is None:
        return False, "numpy not available"
    try:
        import insightface  # type: ignore
        from insightface.app import FaceAnalysis  # noqa: F401
    except Exception as exc:
        return False, f"insightface not installed ({exc})"
    return True, None


def _get_insightface_analyzer():
    global _INSIGHTFACE_ANALYZER
    if _INSIGHTFACE_ANALYZER is not None:
        return _INSIGHTFACE_ANALYZER

    ok, reason = _insightface_available()
    if not ok:
        raise RuntimeError(reason or "insightface not available")

    from insightface.app import FaceAnalysis  # type: ignore

    analyzer = FaceAnalysis(name=os.environ.get("INSIGHTFACE_MODEL", "buffalo_l"))
    analyzer.prepare(ctx_id=0, det_size=(640, 640))
    _INSIGHTFACE_ANALYZER = analyzer
    return analyzer


def _local_gallery_load() -> Dict[str, Any]:
    if not isinstance(LOCAL_FACE_GALLERY_FILE, Path):
        return {"version": 1, "actors": {}}
    gallery = _safe_json_load(LOCAL_FACE_GALLERY_FILE, default_value={"version": 1, "actors": {}})
    if not isinstance(gallery, dict):
        return {"version": 1, "actors": {}}
    if "actors" not in gallery or not isinstance(gallery.get("actors"), dict):
        gallery["actors"] = {}
    if "version" not in gallery:
        gallery["version"] = 1
    return gallery


def _local_gallery_save(gallery: Dict[str, Any]) -> None:
    if not isinstance(LOCAL_FACE_GALLERY_FILE, Path):
        raise RuntimeError("Local face gallery path is not configured")
    LOCAL_FACE_GALLERY_FILE.parent.mkdir(exist_ok=True)
    _safe_json_save(LOCAL_FACE_GALLERY_FILE, gallery)


def _cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    # Normalize defensively
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))  # type: ignore
    if denom <= 0:
        return -1.0
    return float(np.dot(a, b) / denom)  # type: ignore


def _local_match_embedding(embedding: "np.ndarray", gallery: Dict[str, Any], *, threshold: float) -> Dict[str, Any] | None:
    best = None
    best_score = float("-inf")
    actors = (gallery.get("actors") or {}) if isinstance(gallery, dict) else {}
    for actor_name, samples in actors.items():
        if not isinstance(samples, list):
            continue
        for sample in samples:
            vec = sample.get("embedding") if isinstance(sample, dict) else None
            if not isinstance(vec, list) or not vec:
                continue
            try:
                v = np.asarray(vec, dtype=np.float32)  # type: ignore
            except Exception:
                continue
            score = _cosine_similarity(embedding, v)
            if score > best_score:
                best_score = score
                best = {"name": str(actor_name), "similarity": score}

    if best is None:
        return None
    if best_score < float(threshold):
        return None
    return best

# S3 bucket for video storage
DEFAULT_MEDIA_BUCKET = os.environ.get("MEDIA_S3_BUCKET")
SEMANTIC_SEARCH_BUCKET = os.environ.get("SEMANTIC_SEARCH_BUCKET") or DEFAULT_MEDIA_BUCKET

if not SEMANTIC_SEARCH_BUCKET:
    raise RuntimeError("Set SEMANTIC_SEARCH_BUCKET or MEDIA_S3_BUCKET before starting semantic search service")


def _normalise_bucket_region(region: str | None) -> str:
    """Convert S3 location constraint into a concrete region name."""
    if not region:
        return "us-east-1"
    if region == "EU":
        return "eu-west-1"
    return region


def _detect_bucket_region(bucket_name: str) -> str | None:
    """Best-effort detection of the S3 bucket's region."""
    try:
        probe_client = boto3.client("s3", region_name=DEFAULT_AWS_REGION)
        response = probe_client.get_bucket_location(Bucket=bucket_name)
        return _normalise_bucket_region(response.get("LocationConstraint"))
    except Exception as exc:  # pragma: no cover - network failures tolerated
        # Fallback: HeadBucket often returns the authoritative region in headers.
        try:
            probe_client = boto3.client("s3", region_name=DEFAULT_AWS_REGION)
            probe_client.head_bucket(Bucket=bucket_name)
            return DEFAULT_AWS_REGION
        except Exception as head_exc:
            try:
                headers = getattr(head_exc, "response", {}).get("ResponseMetadata", {}).get("HTTPHeaders", {}) or {}
                hdr_region = headers.get("x-amz-bucket-region") or headers.get("x-amz-region")
                if hdr_region:
                    return _normalise_bucket_region(str(hdr_region))
            except Exception:
                pass
            app.logger.warning("Unable to determine region for bucket %s: %s", bucket_name, exc)
            return None


def _s3_head_object_or_raise(*, bucket: str, key: str) -> dict[str, Any]:
    """Preflight that the current AWS credentials can see the object metadata.

    This helps catch wrong key/region/permissions early and produces clearer errors than
    Rekognition's InvalidS3ObjectException.
    """
    b = (bucket or "").strip()
    k = (key or "").strip().lstrip("/")
    if not b or not k:
        raise ValueError("Missing bucket/key")

    bucket_region = _detect_bucket_region(b)
    s3_region = bucket_region or DEFAULT_AWS_REGION
    client = _s3_client_for_transfer(region_name=s3_region)
    try:
        return client.head_object(Bucket=b, Key=k)
    except Exception as exc:
        # Try to extract region hint for better diagnostics.
        hinted_region = None
        try:
            headers = getattr(exc, "response", {}).get("ResponseMetadata", {}).get("HTTPHeaders", {}) or {}
            hinted_region = headers.get("x-amz-bucket-region") or headers.get("x-amz-region")
        except Exception:
            pass
        hint = f" (bucket_region={bucket_region or 'unknown'}, aws_region={DEFAULT_AWS_REGION}"
        if hinted_region:
            hint += f", hinted_region={hinted_region}"
        hint += ")"
        raise RuntimeError(f"S3 HeadObject failed for s3://{b}/{k}{hint}: {exc}")


def _mediaconvert_enabled() -> bool:
    return _env_truthy(os.getenv("ENVID_METADATA_MEDIACONVERT_ENABLE"), default=False)


def _mediaconvert_role_arn() -> str:
    return (os.getenv("ENVID_METADATA_MEDIACONVERT_ROLE_ARN") or "").strip()


def _mediaconvert_queue_arn() -> str | None:
    v = (os.getenv("ENVID_METADATA_MEDIACONVERT_QUEUE_ARN") or "").strip()
    return v or None


def _mediaconvert_output_bucket_default(input_bucket: str) -> str:
    return (os.getenv("ENVID_METADATA_MEDIACONVERT_OUTPUT_BUCKET") or input_bucket or "").strip() or input_bucket


def _mediaconvert_proxy_prefix() -> str:
    return (os.getenv("ENVID_METADATA_MEDIACONVERT_PROXY_PREFIX") or "envid-metadata/proxies").strip().strip("/")


def _mediaconvert_client(*, region_name: str) -> Any:
    """Create a MediaConvert client.

    MediaConvert requires an account-specific endpoint. We support an explicit override via
    ENVID_METADATA_MEDIACONVERT_ENDPOINT, otherwise we discover it via describe_endpoints.
    """
    endpoint = (os.getenv("ENVID_METADATA_MEDIACONVERT_ENDPOINT") or "").strip()
    if endpoint:
        return boto3.client("mediaconvert", region_name=region_name, endpoint_url=endpoint)

    probe = boto3.client("mediaconvert", region_name=region_name)
    resp = probe.describe_endpoints(MaxResults=1)
    endpoints = resp.get("Endpoints") or []
    url = (endpoints[0].get("Url") if endpoints else None) or None
    if not url:
        raise RuntimeError("Could not discover MediaConvert endpoint; set ENVID_METADATA_MEDIACONVERT_ENDPOINT")
    return boto3.client("mediaconvert", region_name=region_name, endpoint_url=url)


def _poll_mediaconvert_job(
    *,
    client: Any,
    job_id: str,
    envid_job_id: str,
    message: str,
    progress_base: int,
    progress_span: int,
    step_id: str | None = None,
) -> dict[str, Any]:
    start_t = time.monotonic()
    max_seconds = _parse_int_param(os.getenv("ENVID_METADATA_MEDIACONVERT_MAX_WAIT_SECONDS"), default=7200, min_value=60, max_value=43200)
    poll_seconds = float(os.getenv("ENVID_METADATA_MEDIACONVERT_POLL_SECONDS") or 8.0)
    poll_seconds = max(2.0, min(30.0, poll_seconds))

    last_status: str | None = None
    while True:
        if time.monotonic() - start_t > max_seconds:
            raise RuntimeError(f"Timed out waiting for MediaConvert job ({max_seconds}s)")

        resp = client.get_job(Id=job_id)
        job = resp.get("Job") or {}
        status = str(job.get("Status") or "").strip().upper()
        if status and status != last_status:
            last_status = status
            _job_update(envid_job_id, progress=progress_base, message=f"{message} ({status})")
            if step_id:
                # MediaConvert doesn't provide a native 0-100 percent here; we surface a best-effort
                # status message and keep percent moving via the elapsed-time approximation below.
                if status in {"SUBMITTED", "PROGRESSING"}:
                    _job_step_update(envid_job_id, step_id, status="running", message=status.title())

        if status in {"ERROR", "CANCELED"}:
            err = job.get("ErrorMessage") or job.get("Status") or "MediaConvert job failed"
            if step_id:
                _job_step_update(envid_job_id, step_id, status="failed", message=str(err))
            raise RuntimeError(f"MediaConvert job failed: {err}")

        if status == "COMPLETE":
            _job_update(envid_job_id, progress=progress_base + progress_span, message=message)
            if step_id:
                _job_step_update(envid_job_id, step_id, status="completed", percent=100, message="Completed")
            return job

        # SUBMITTED/PROGRESSING
        elapsed = time.monotonic() - start_t
        pct = min(0.99, elapsed / max_seconds)
        _job_update(envid_job_id, progress=progress_base + int(progress_span * pct), message=message)
        if step_id:
            # Best-effort progress indicator so the UI doesn't appear stuck.
            _job_step_update(envid_job_id, step_id, status="running", percent=int(pct * 100), message="Transcoding")
        time.sleep(poll_seconds)


def _create_mediaconvert_proxy(
    *,
    envid_job_id: str,
    input_bucket: str,
    input_key: str,
    region_name: str,
    has_audio: bool = True,
) -> tuple[str, str]:
    """Transcode the input S3 object to a Rekognition-friendly MP4 proxy.

    Returns (output_bucket, output_key).
    """
    if not _mediaconvert_enabled():
        raise RuntimeError("MediaConvert is disabled (set ENVID_METADATA_MEDIACONVERT_ENABLE=1)")
    role_arn = _mediaconvert_role_arn()
    if not role_arn:
        raise RuntimeError("Missing ENVID_METADATA_MEDIACONVERT_ROLE_ARN for MediaConvert")

    out_bucket = _mediaconvert_output_bucket_default(input_bucket)
    prefix = _mediaconvert_proxy_prefix()
    # Put proxies under a per-job folder to avoid naming collisions.
    dest_prefix = f"{prefix}/{envid_job_id}/"
    destination = f"s3://{out_bucket}/{dest_prefix}"
    # MediaConvert names outputs based on the input filename plus NameModifier.
    # Keep the name deterministic so we can reference/verify it.
    stem = Path(input_key).name
    if stem.lower().endswith(".mp4"):
        stem = stem[:-4]
    else:
        stem = Path(stem).stem
    name_modifier = "_proxy"
    output_key = f"{dest_prefix}{stem}{name_modifier}.mp4"

    mc = _mediaconvert_client(region_name=region_name)

    output: dict[str, Any] = {
        "NameModifier": name_modifier,
        "ContainerSettings": {"Container": "MP4"},
        "VideoDescription": {
            "CodecSettings": {
                "Codec": "H_264",
                "H264Settings": {
                    # Conservative proxy settings to keep size down and compatibility high.
                    "RateControlMode": "QVBR",
                    "QvbrSettings": {"QvbrQualityLevel": 7},
                    "MaxBitrate": 5000000,
                    "CodecLevel": "AUTO",
                    "CodecProfile": "MAIN",
                },
            }
        },
    }

    input_settings: dict[str, Any] = {
        "FileInput": f"s3://{input_bucket}/{input_key}",
        "VideoSelector": {},
    }
    if has_audio:
        input_settings["AudioSelectors"] = {"Audio Selector 1": {"DefaultSelection": "DEFAULT"}}
        output["AudioDescriptions"] = [
            {
                "AudioSourceName": "Audio Selector 1",
                "CodecSettings": {
                    "Codec": "AAC",
                    "AacSettings": {
                        "Bitrate": 128000,
                        "CodingMode": "CODING_MODE_2_0",
                        "SampleRate": 48000,
                    },
                },
            }
        ]

    create_kwargs: dict[str, Any] = {
        "Role": role_arn,
        "Settings": {
            "Inputs": [input_settings],
            "OutputGroups": [
                {
                    "Name": "File Group",
                    "OutputGroupSettings": {
                        "Type": "FILE_GROUP_SETTINGS",
                        "FileGroupSettings": {"Destination": destination},
                    },
                    "Outputs": [output],
                }
            ],
        },
        "UserMetadata": {
            "envid_job_id": envid_job_id,
            "input": f"s3://{input_bucket}/{input_key}",
            "output": f"s3://{out_bucket}/{output_key}",
        },
    }

    queue_arn = _mediaconvert_queue_arn()
    if queue_arn:
        create_kwargs["Queue"] = queue_arn

    _job_update(envid_job_id, progress=5, message="MediaConvert: submitting proxy transcode")
    _job_step_update(envid_job_id, "mediaconvert_proxy", status="running", percent=0, message="Submitting")
    resp = mc.create_job(**create_kwargs)
    job = resp.get("Job") or {}
    mc_job_id = job.get("Id")
    if not mc_job_id:
        _job_step_update(envid_job_id, "mediaconvert_proxy", status="failed", message="Failed to start")
        raise RuntimeError("Failed to start MediaConvert job")

    _poll_mediaconvert_job(
        client=mc,
        job_id=str(mc_job_id),
        envid_job_id=envid_job_id,
        message="MediaConvert: transcoding proxy",
        progress_base=6,
        progress_span=8,
        step_id="mediaconvert_proxy",
    )

    # Sanity check the output exists.
    _s3_head_object_or_raise(bucket=out_bucket, key=output_key)
    _job_update(envid_job_id, progress=14, message="MediaConvert: proxy ready", s3_proxy_uri=f"s3://{out_bucket}/{output_key}")
    return out_bucket, output_key


def _resolve_transcribe_region(bucket_region: str | None) -> str:
    """Determine which region AWS Transcribe should use."""
    override = os.environ.get("TRANSCRIBE_REGION")
    if override:
        if bucket_region and bucket_region != override:
            app.logger.warning(
                "TRANSCRIBE_REGION override (%s) differs from bucket region %s",
                override,
                bucket_region,
            )
        return override
    return bucket_region or DEFAULT_AWS_REGION


def _env_truthy(value: str | None, *, default: bool = True) -> bool:
    if value is None:
        return default
    v = str(value).strip().lower()
    if v in {"0", "false", "no", "off"}:
        return False
    if v in {"1", "true", "yes", "on"}:
        return True
    return default


def _features_preset_default_enabled() -> bool:
    """Default feature enablement for envidMetadata.

    - Default is "all" to preserve existing behavior.
    - Set `ENVID_METADATA_FEATURES_PRESET=none` to disable everything by default,
      then selectively re-enable with `ENVID_METADATA_ENABLE_<FEATURE>=true`.
    """

    preset = (os.getenv("ENVID_METADATA_FEATURES_PRESET") or "all").strip().lower()
    if preset in {"none", "off", "disabled", "false", "0"}:
        return False
    return True


def _feature_enabled(feature: str, *, default: bool) -> bool:
    """Return whether a given feature is enabled via env.

    Feature names are upper snake-case without the `ENVID_METADATA_ENABLE_` prefix.
    Example: `ENVID_METADATA_ENABLE_CELEBRITIES=true`.
    """

    key = f"ENVID_METADATA_ENABLE_{feature.upper()}"
    if key not in os.environ:
        return default
    return _env_truthy(os.getenv(key), default=default)


# One-time startup log for operational clarity.
try:
    if _env_truthy(os.getenv("ENVID_METADATA_S3_ACCELERATE"), default=False):
        msg = "S3 Transfer Acceleration is ENABLED for uploads (bucket must have acceleration enabled)."
        print(f"[envid-metadata] {msg}")
        app.logger.info(msg)
    else:
        msg = "S3 Transfer Acceleration is disabled for uploads."
        print(f"[envid-metadata] {msg}")
        app.logger.info(msg)
except Exception:
    pass


def _video_s3_bucket() -> str:
    bucket = (os.getenv("ENVID_METADATA_VIDEO_BUCKET") or SEMANTIC_SEARCH_BUCKET or "").strip()
    if not bucket:
        raise RuntimeError(
            "Missing S3 bucket for video upload. Set ENVID_METADATA_VIDEO_BUCKET or SEMANTIC_SEARCH_BUCKET/MEDIA_S3_BUCKET."
        )
    return bucket


def _video_s3_key(video_id: str, original_filename: str) -> str:
    prefix = (os.getenv("ENVID_METADATA_VIDEO_S3_PREFIX") or "envid-metadata/videos").strip().strip("/")
    file_extension = Path(original_filename).suffix or ".mp4"
    return f"{prefix}/{video_id}{file_extension}"


def _metadata_artifacts_s3_bucket() -> str:
    """Bucket used to persist derived metadata artifacts (JSON/ZIP)."""
    bucket = (
        os.getenv("ENVID_METADATA_ARTIFACTS_BUCKET")
        or os.getenv("ENVID_METADATA_VIDEO_BUCKET")
        or SEMANTIC_SEARCH_BUCKET
        or ""
    ).strip()
    if not bucket:
        raise RuntimeError(
            "Missing S3 bucket for metadata artifacts. Set ENVID_METADATA_ARTIFACTS_BUCKET (or ENVID_METADATA_VIDEO_BUCKET / SEMANTIC_SEARCH_BUCKET)."
        )
    return bucket


def _metadata_artifacts_s3_prefix() -> str:
    return (os.getenv("ENVID_METADATA_ARTIFACTS_S3_PREFIX") or "envid-metadata/metadata-json").strip().strip("/")


def _metadata_combined_s3_key(video_id: str) -> str:
    base = _metadata_artifacts_s3_prefix()
    return f"{base}/{video_id}/combined.json"


def _metadata_category_s3_key(video_id: str, category: str) -> str:
    base = _metadata_artifacts_s3_prefix()
    safe = re.sub(r"[^a-zA-Z0-9\-_]+", "_", str(category or "").strip()).strip("_") or "category"
    return f"{base}/{video_id}/categories/{safe}.json"


def _metadata_zip_s3_key(video_id: str) -> str:
    base = _metadata_artifacts_s3_prefix()
    return f"{base}/{video_id}/metadata_json.zip"


def _subtitles_s3_key(video_id: str, *, lang: str, fmt: str) -> str:
    """Return the canonical S3 key for subtitle artifacts.

    lang:
      - "orig" for original language
      - "en" for English translation
    fmt:
      - "srt" or "vtt"
    """
    vid = str(video_id or "").strip()
    if not vid:
        raise ValueError("Missing video_id")
    fmt_norm = (fmt or "").strip().lower()
    if fmt_norm not in {"srt", "vtt"}:
        raise ValueError("Invalid subtitle fmt")

    base = _metadata_artifacts_s3_prefix()
    if (lang or "").strip().lower() in {"en", "eng", "english"}:
        return f"{base}/{vid}/subtitles/subtitles.en.{fmt_norm}"
    return f"{base}/{vid}/subtitles/subtitles.{fmt_norm}"


def _s3_object_exists(*, bucket: str, key: str) -> bool:
    b = (bucket or "").strip()
    k = (key or "").strip().lstrip("/")
    if not b or not k:
        return False
    try:
        bucket_region = _detect_bucket_region(b)
        s3_region = bucket_region or DEFAULT_AWS_REGION
        client = _s3_client_for_transfer(region_name=s3_region)
        client.head_object(Bucket=b, Key=k)
        return True
    except Exception:
        return False


def _s3_presigned_download_url(
    *,
    bucket: str,
    key: str,
    download_name: str,
    content_type: str,
    expires_seconds: int | None = None,
) -> str:
    b = (bucket or "").strip()
    k = (key or "").strip().lstrip("/")
    if not b or not k:
        raise ValueError("Missing bucket/key")
    bucket_region = _detect_bucket_region(b)
    s3_region = bucket_region or DEFAULT_AWS_REGION
    client = _s3_client_for_transfer(region_name=s3_region)
    expires = (
        int(expires_seconds)
        if expires_seconds is not None
        else _parse_int_param(os.getenv("ENVID_METADATA_S3_PRESIGN_SECONDS"), default=3600, min_value=60, max_value=86400)
    )
    return client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": b,
            "Key": k,
            "ResponseContentDisposition": f'attachment; filename="{download_name}"',
            "ResponseContentType": content_type,
        },
        ExpiresIn=expires,
    )


def _s3_put_bytes(
    *,
    bucket: str,
    key: str,
    body: bytes,
    content_type: str,
) -> None:
    bucket_region = _detect_bucket_region(bucket)
    s3_region = bucket_region or DEFAULT_AWS_REGION
    client = _s3_client_for_transfer(region_name=s3_region)

    # Ensure bucket exists in local demos (best-effort).
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        create_kwargs: Dict[str, Any] = {"Bucket": bucket}
        if s3_region != "us-east-1":
            create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": s3_region}
        client.create_bucket(**create_kwargs)

    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
    )


def _s3_delete_object_best_effort(*, bucket: str, key: str) -> bool:
    b = (bucket or "").strip()
    k = (key or "").strip().lstrip("/")
    if not b or not k:
        return False
    try:
        bucket_region = _detect_bucket_region(b)
        s3_region = bucket_region or DEFAULT_AWS_REGION
        client = _s3_client_for_transfer(region_name=s3_region)
        client.delete_object(Bucket=b, Key=k)
        return True
    except Exception:
        return False


def _s3_delete_prefix_best_effort(*, bucket: str, prefix: str) -> int:
    b = (bucket or "").strip()
    p = (prefix or "").strip().lstrip("/")
    if not b or not p:
        return 0
    # Ensure prefix ends with '/' so we don't accidentally delete sibling objects.
    if not p.endswith("/"):
        p = p + "/"
    deleted = 0
    try:
        bucket_region = _detect_bucket_region(b)
        s3_region = bucket_region or DEFAULT_AWS_REGION
        client = _s3_client_for_transfer(region_name=s3_region)

        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": b, "Prefix": p, "MaxKeys": 1000}
            if token:
                kwargs["ContinuationToken"] = token
            resp = client.list_objects_v2(**kwargs)
            contents = resp.get("Contents") or []
            keys = [obj.get("Key") for obj in contents if obj.get("Key")]
            if keys:
                # delete_objects allows up to 1000 objects per call
                del_resp = client.delete_objects(
                    Bucket=b,
                    Delete={"Objects": [{"Key": k} for k in keys], "Quiet": True},
                )
                deleted += len(keys)
                # Best-effort: ignore errors in del_resp

            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
            if not token:
                break
    except Exception:
        return deleted
    return deleted


def _metadata_zip_bytes(*, categories: dict[str, Any], combined: dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("combined.json", json.dumps(combined, ensure_ascii=False, indent=2))
        for key in sorted(categories.keys()):
            zf.writestr(f"categories/{key}.json", json.dumps(categories.get(key) or {}, ensure_ascii=False, indent=2))
    return buf.getvalue()


def _ensure_metadata_artifacts_on_s3(video: dict[str, Any], *, save_index: bool = True) -> None:
    """Persist categorized metadata artifacts to S3 and record keys on the video entry.

    Best-effort: if upload fails, we keep local/indexed data intact.
    """

    video_id = str(video.get("id") or "").strip()
    if not video_id:
        return

    categories: dict[str, Any] = video.get("metadata_categories") or {}
    combined: dict[str, Any] = video.get("metadata_combined") or {}
    if not isinstance(categories, dict) or not isinstance(combined, dict):
        return

    bucket = _metadata_artifacts_s3_bucket()
    keys: dict[str, Any] = {
        "bucket": bucket,
        "prefix": _metadata_artifacts_s3_prefix(),
        "combined": _metadata_combined_s3_key(video_id),
        "categories": {k: _metadata_category_s3_key(video_id, k) for k in sorted(categories.keys())},
        "zip": _metadata_zip_s3_key(video_id),
    }

    # Skip work if it looks like we've already recorded a complete set.
    existing = video.get("metadata_s3")
    if isinstance(existing, dict) and existing.get("combined") and existing.get("zip") and isinstance(existing.get("categories"), dict):
        video["metadata_s3"] = {**existing, **{k: v for k, v in keys.items() if k in {"bucket", "prefix"}}}
        if save_index:
            _save_video_index()
        return

    # Optional: allow callers to attach job progress messaging.
    job_id = str(video.get("id") or "").strip() or None

    try:
        if job_id:
            _job_update(job_id, message="Saving: uploading metadata combined.json")
        _s3_put_bytes(
            bucket=bucket,
            key=keys["combined"],
            body=json.dumps(combined, ensure_ascii=False, indent=2).encode("utf-8"),
            content_type="application/json",
        )

        if job_id:
            _job_update(job_id, message="Saving: uploading metadata category JSONs")
        for cat_key, s3_key in (keys.get("categories") or {}).items():
            _s3_put_bytes(
                bucket=bucket,
                key=s3_key,
                body=json.dumps(categories.get(cat_key) or {}, ensure_ascii=False, indent=2).encode("utf-8"),
                content_type="application/json",
            )

        if job_id:
            _job_update(job_id, message="Saving: uploading metadata_json.zip")
        _s3_put_bytes(
            bucket=bucket,
            key=keys["zip"],
            body=_metadata_zip_bytes(categories=categories, combined=combined),
            content_type="application/zip",
        )

        video["metadata_s3"] = keys
        if save_index:
            _save_video_index()
    except Exception as exc:
        try:
            app.logger.warning("Failed to upload metadata artifacts to S3 for video %s: %s", video_id, exc)
        except Exception:
            pass


def _rawvideo_bucket() -> str:
    return (os.getenv("ENVID_METADATA_RAWVIDEO_BUCKET") or "mediagenailab").strip()


def _rawvideo_prefix() -> str:
    return (os.getenv("ENVID_METADATA_RAWVIDEO_S3_PREFIX") or "rawvideo").strip().strip("/")


def _normalize_rawvideo_key(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        raise ValueError("Missing s3_key")
    if value.startswith("s3://"):
        # Accept s3://mediagenailab/rawvideo/...
        parts = value.replace("s3://", "", 1).split("/", 1)
        if len(parts) != 2:
            raise ValueError("Invalid s3_uri")
        bucket, key = parts[0], parts[1]
        if bucket != _rawvideo_bucket():
            raise ValueError(f"Only bucket '{_rawvideo_bucket()}' is allowed")
        value = key

    value = value.lstrip("/")
    prefix = _rawvideo_prefix()
    if not value.startswith(prefix + "/"):
        # Allow shorthand like "myfile.mp4" -> "rawvideo/myfile.mp4"
        value = f"{prefix}/{value}"

    # Enforce prefix constraint.
    if not value.startswith(prefix + "/"):
        raise ValueError(f"S3 key must be under '{prefix}/'")
    return value


def _parse_allowed_s3_video_source(raw: str) -> tuple[str, str]:
    """Parse user-provided S3 location.

    Accepts:
    - s3://<bucket>/<key>
    - <key> under rawvideo/ (rawvideo bucket)
    - <key> under ENVID_METADATA_VIDEO_S3_PREFIX (video bucket)

    For backward compatibility, bare filenames are treated as rawvideo keys.
    """

    value = (raw or "").strip()
    if not value:
        raise ValueError("Missing s3_key")

    if value.startswith("s3://"):
        parts = value.replace("s3://", "", 1).split("/", 1)
        if len(parts) != 2:
            raise ValueError("Invalid s3_uri")
        bucket, key = parts[0].strip(), parts[1].lstrip("/")
        if bucket not in {_rawvideo_bucket(), _video_s3_bucket()}:
            raise ValueError(f"Only buckets '{_rawvideo_bucket()}' or '{_video_s3_bucket()}' are allowed")
        if bucket == _rawvideo_bucket():
            # rawvideo bucket and video bucket can be the same bucket in some setups.
            # Only normalize into rawvideo/* when the key is actually a rawvideo-style key.
            rawvideo_prefix = _rawvideo_prefix().rstrip("/")
            video_prefix = (os.getenv("ENVID_METADATA_VIDEO_S3_PREFIX") or "envid-metadata/videos").strip().strip("/")
            if key.startswith(video_prefix + "/"):
                return bucket, key
            if ("/" not in key) or key.startswith(rawvideo_prefix + "/"):
                return bucket, _normalize_rawvideo_key(key)
            return bucket, key
        return bucket, key

    # Accept inputs like "bucket/key" from UI; strip known bucket prefix if present.
    v = value.lstrip("/")
    for b in (_rawvideo_bucket(), _video_s3_bucket()):
        if v.startswith(b + "/"):
            v = v[len(b) + 1 :]
            break

    rawvideo_prefix = _rawvideo_prefix().rstrip("/")
    video_prefix = (os.getenv("ENVID_METADATA_VIDEO_S3_PREFIX") or "envid-metadata/videos").strip().strip("/")

    if v.startswith(rawvideo_prefix + "/") or ("/" not in v):
        # Treat bare filenames as rawvideo.
        return _rawvideo_bucket(), _normalize_rawvideo_key(v)
    if v.startswith(video_prefix + "/"):
        return _video_s3_bucket(), v

    # Fallback: try rawvideo normalization.
    return _rawvideo_bucket(), _normalize_rawvideo_key(v)


def _s3_client_for_transfer(*, region_name: str) -> Any:
    use_accelerate = _env_truthy(os.getenv("ENVID_METADATA_S3_ACCELERATE"), default=False)

    # Timeouts/retries: prevent long hangs during uploads/downloads.
    connect_timeout_s = _parse_int_param(
        os.getenv("ENVID_METADATA_S3_CONNECT_TIMEOUT_SECONDS"),
        default=10,
        min_value=1,
        max_value=120,
    )
    read_timeout_s = _parse_int_param(
        os.getenv("ENVID_METADATA_S3_READ_TIMEOUT_SECONDS"),
        default=120,
        min_value=1,
        max_value=3600,
    )
    max_attempts = _parse_int_param(
        os.getenv("ENVID_METADATA_S3_MAX_RETRY_ATTEMPTS"),
        default=10,
        min_value=1,
        max_value=30,
    )

    if use_accelerate:
        # Acceleration uses the CloudFront-backed accelerate endpoint; enforce SigV4
        # to avoid legacy SigV2 presign behavior that can cause SignatureDoesNotMatch
        # when browsers send Content-Type.
        cfg_accel = BotoConfig(
            signature_version="s3v4",
            connect_timeout=connect_timeout_s,
            read_timeout=read_timeout_s,
            retries={"max_attempts": max_attempts, "mode": "standard"},
            s3={"use_accelerate_endpoint": True},
        )
        return boto3.client("s3", region_name=region_name, config=cfg_accel)

    # Non-accelerate client.
    cfg_no_accel = BotoConfig(
        signature_version="s3v4",
        connect_timeout=connect_timeout_s,
        read_timeout=read_timeout_s,
        retries={"max_attempts": max_attempts, "mode": "standard"},
    )
    return boto3.client("s3", region_name=region_name, config=cfg_no_accel)


def _download_s3_object_to_file(
    *,
    bucket: str,
    key: str,
    dest_path: Path,
    job_id: str,
) -> None:
    bucket_region = _detect_bucket_region(bucket)
    s3_region = bucket_region or DEFAULT_AWS_REGION
    client = _s3_client_for_transfer(region_name=s3_region)

    # Similar tuning knobs as upload.
    multipart_chunk_mb = _parse_int_param(
        os.getenv("ENVID_METADATA_S3_MULTIPART_CHUNK_MB"),
        default=16,
        min_value=5,
        max_value=512,
    )
    max_concurrency = _parse_int_param(
        os.getenv("ENVID_METADATA_S3_MAX_CONCURRENCY"),
        default=8,
        min_value=1,
        max_value=32,
    )
    transfer_config = TransferConfig(
        multipart_chunksize=multipart_chunk_mb * 1024 * 1024,
        max_concurrency=max_concurrency,
        use_threads=True,
    )

    total_bytes: int | None = None
    try:
        head = client.head_object(Bucket=bucket, Key=key)
        total_bytes = int(head.get("ContentLength") or 0) or None
    except Exception:
        total_bytes = None

    downloaded = 0
    start_t = time.monotonic()
    last_emit_t = 0.0
    last_pct = -1

    def _cb(bytes_amount: int) -> None:
        nonlocal downloaded, last_emit_t, last_pct
        downloaded += int(bytes_amount or 0)
        now = time.monotonic()
        if (now - last_emit_t) < 0.5:
            return
        last_emit_t = now
        elapsed = max(0.001, now - start_t)
        mbps = (downloaded / (1024 * 1024)) / elapsed
        if total_bytes:
            pct = int(min(100.0, (downloaded / float(total_bytes)) * 100.0))
            if pct == last_pct:
                return
            last_pct = pct
            _job_update(job_id, progress=2, message=f"Downloading video from S3 ({pct}%, {mbps:.1f} MB/s)")
        else:
            _job_update(job_id, progress=2, message=f"Downloading video from S3 ({downloaded} bytes, {mbps:.1f} MB/s)")

    diag = f"[envid-metadata] Downloading S3 object: s3://{bucket}/{key} (region={s3_region} accelerate={_env_truthy(os.getenv('ENVID_METADATA_S3_ACCELERATE'), default=False)})"
    print(diag, flush=True)
    try:
        app.logger.warning(diag)
    except Exception:
        pass

    client.download_file(
        bucket,
        key,
        str(dest_path),
        Config=transfer_config,
        Callback=_cb,
    )


@app.route("/rawvideo/list", methods=["GET"])
def list_rawvideo_objects() -> Any:
    """List objects in s3://<rawvideo_bucket>/<rawvideo_prefix>/ for UI selection."""
    try:
        bucket = _rawvideo_bucket()
        prefix = _rawvideo_prefix().strip().strip("/") + "/"

        max_keys_raw = (request.args.get("max_keys") or "200").strip()
        try:
            max_keys = int(max_keys_raw)
        except Exception:
            max_keys = 200
        max_keys = max(1, min(1000, max_keys))

        continuation_token = (request.args.get("continuation_token") or "").strip() or None

        bucket_region = _detect_bucket_region(bucket)
        s3_region = bucket_region or DEFAULT_AWS_REGION
        client = _s3_client_for_transfer(region_name=s3_region)

        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": max_keys}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        resp = client.list_objects_v2(**kwargs)
        contents = resp.get("Contents") or []

        objects: list[dict[str, Any]] = []
        for obj in contents:
            key = obj.get("Key")
            if not key or not str(key).startswith(prefix):
                continue
            if str(key).endswith("/"):
                continue
            last_modified = obj.get("LastModified")
            objects.append(
                {
                    "key": key,
                    "size": obj.get("Size"),
                    "last_modified": last_modified.isoformat() if hasattr(last_modified, "isoformat") else None,
                }
            )

        try:
            objects.sort(key=lambda x: x.get("last_modified") or "", reverse=True)
        except Exception:
            pass

        return jsonify(
            {
                "bucket": bucket,
                "prefix": prefix.rstrip("/"),
                "objects": objects,
                "is_truncated": bool(resp.get("IsTruncated")),
                "next_continuation_token": resp.get("NextContinuationToken"),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/list-s3-videos", methods=["GET"])
def list_s3_videos_compat() -> Any:
    """Backward-compatible alias for listing rawvideo objects.

    Historically some scripts called /list-s3-videos. The canonical endpoint is /rawvideo/list.
    """
    return list_rawvideo_objects()


def _process_s3_video_job(
    *,
    job_id: str,
    s3_key: str,
    video_title: str,
    video_description: str,
    frame_interval_seconds: int,
    max_frames_to_analyze: int,
    face_recognition_mode: Optional[str],
    collection_id: Optional[str],
) -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix=f"envid_metadata_s3_{job_id}_"))
    try:
        bucket = _rawvideo_bucket()
        normalized_key = _normalize_rawvideo_key(s3_key)

        _job_update(job_id, status="processing", progress=2, message="Downloading video from S3")
        local_path = temp_dir / "video"
        # Use original extension if present
        ext = Path(normalized_key).suffix
        if ext:
            local_path = local_path.with_suffix(ext)
        _download_s3_object_to_file(bucket=bucket, key=normalized_key, dest_path=local_path, job_id=job_id)

        s3_uri = f"s3://{bucket}/{normalized_key}"
        _job_update(job_id, s3_video_key=normalized_key, s3_video_uri=s3_uri)

        original_filename = Path(normalized_key).name
        _process_video_job(
            job_id=job_id,
            temp_dir=temp_dir,
            video_path=local_path,
            video_title=video_title,
            video_description=video_description,
            original_filename=original_filename,
            frame_interval_seconds=frame_interval_seconds,
            max_frames_to_analyze=max_frames_to_analyze,
            face_recognition_mode=face_recognition_mode,
            collection_id=collection_id,
            preexisting_s3_video_key=normalized_key,
            preexisting_s3_video_uri=s3_uri,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _process_s3_object_job_download_and_analyze(
    *,
    job_id: str,
    s3_bucket: str,
    s3_key: str,
    video_title: str,
    video_description: str,
    frame_interval_seconds: int,
    max_frames_to_analyze: int,
    face_recognition_mode: Optional[str],
    collection_id: Optional[str],
) -> None:
    """Download an allowed S3 video to the server, then run the full per-frame pipeline.

    This is the S3 equivalent of local uploads: it always extracts frames and stores per-frame metadata
    so UI features like Envid eye and exact timestamps work consistently.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=f"envid_metadata_s3_{job_id}_"))
    try:
        bucket = (s3_bucket or "").strip()
        if not bucket:
            raise ValueError("Missing s3_bucket")

        key = (s3_key or "").strip().lstrip("/")
        if not key:
            raise ValueError("Missing s3_key")

        if bucket == _rawvideo_bucket():
            raw_prefix = _rawvideo_prefix().rstrip("/")
            # Only normalize when the key is actually a rawvideo-style key.
            if ("/" not in key) or key.startswith(raw_prefix + "/"):
                key = _normalize_rawvideo_key(key)

        _job_update(job_id, status="processing", progress=2, message="Downloading video from S3")
        local_path = temp_dir / "video"
        ext = Path(key).suffix
        if ext:
            local_path = local_path.with_suffix(ext)
        _download_s3_object_to_file(bucket=bucket, key=key, dest_path=local_path, job_id=job_id)

        s3_uri = f"s3://{bucket}/{key}"
        _job_update(job_id, s3_video_key=key, s3_video_uri=s3_uri)

        original_filename = Path(key).name
        _process_video_job(
            job_id=job_id,
            temp_dir=temp_dir,
            video_path=local_path,
            video_title=video_title,
            video_description=video_description,
            original_filename=original_filename,
            frame_interval_seconds=frame_interval_seconds,
            max_frames_to_analyze=max_frames_to_analyze,
            face_recognition_mode=face_recognition_mode,
            collection_id=collection_id,
            preexisting_s3_video_key=key,
            preexisting_s3_video_uri=s3_uri,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _rekognition_client(*, region_name: str) -> Any:
    return boto3.client("rekognition", region_name=region_name)


def _rekognition_video_ref_for_rawvideo(key: str) -> dict[str, Any]:
    return {"S3Object": {"Bucket": _rawvideo_bucket(), "Name": _normalize_rawvideo_key(key)}}


def _rekognition_video_ref(bucket: str, key: str) -> dict[str, Any]:
    return {"S3Object": {"Bucket": bucket, "Name": key}}


def _poll_rekognition_video_job(
    *,
    envid_job_id: str,
    step_id: str | None = None,
    rekognition_job_id: str,
    get_page: Any,
    page_key: str,
    item_key: str,
    status_key: str = "JobStatus",
    progress_base: int,
    progress_span: int,
    message: str,
) -> list[dict[str, Any]]:
    """Poll a Rekognition Video async job and collect all items from paginated results."""
    items: list[dict[str, Any]] = []
    next_token: str | None = None
    last_status: str | None = None
    start_t = time.monotonic()
    max_seconds = _parse_int_param(os.getenv("ENVID_METADATA_REKOGNITION_VIDEO_MAX_WAIT_SECONDS"), default=3600, min_value=60, max_value=21600)
    poll_seconds = float(os.getenv("ENVID_METADATA_REKOGNITION_VIDEO_POLL_SECONDS") or 5.0)
    poll_seconds = max(2.0, min(20.0, poll_seconds))

    while True:
        if time.monotonic() - start_t > max_seconds:
            raise RuntimeError(f"Timed out waiting for Rekognition job ({max_seconds}s)")

        kwargs: dict[str, Any] = {"JobId": rekognition_job_id}
        if next_token:
            kwargs[page_key] = next_token

        resp = get_page(**kwargs)
        status = (resp.get(status_key) or "").strip().upper()
        if status and status != last_status:
            last_status = status
            _job_update(envid_job_id, progress=progress_base, message=f"{message} ({status})")
            if step_id:
                _job_step_update(envid_job_id, step_id, status="running", percent=0, message=f"{message} ({status})")

        if status in {"FAILED", "ERROR"}:
            status_message = str(resp.get("StatusMessage") or status)
            if step_id:
                _job_step_update(envid_job_id, step_id, status="failed", message=f"Failed: {status_message}")
            lowered = status_message.lower()
            if "unsupported codec" in lowered or "unsupported format" in lowered:
                raise RuntimeError(
                    "Rekognition Video job failed: unsupported codec/format. "
                    "Rekognition Video generally expects an MP4 container with H.264 (AVC) video and AAC audio. "
                    f"AWS message: {status_message}"
                )
            raise RuntimeError(f"Rekognition job failed: {status_message}")

        if status == "SUCCEEDED":
            batch = resp.get(item_key) or []
            if isinstance(batch, list):
                items.extend(batch)
            next_token = resp.get("NextToken")
            if not next_token:
                break
            # paginate without sleeping
            continue

        # IN_PROGRESS
        # Smooth progress tick while waiting
        elapsed = time.monotonic() - start_t
        pct = min(0.99, elapsed / max_seconds)
        _job_update(envid_job_id, progress=progress_base + int(progress_span * pct), message=message)
        if step_id:
            _job_step_update(envid_job_id, step_id, status="running", percent=int(pct * 100), message=message)
        time.sleep(poll_seconds)

    _job_update(envid_job_id, progress=progress_base + progress_span, message=message)
    if step_id:
        _job_step_update(envid_job_id, step_id, status="completed", percent=100, message=message)
    return items


def _process_s3_object_job_cloud_only(
    *,
    job_id: str,
    s3_bucket: str,
    s3_key: str,
    video_title: str,
    video_description: str,
    frame_interval_seconds: int,
    technical_ffprobe: dict[str, Any] | None = None,
    technical_mediainfo: dict[str, Any] | None = None,
    technical_verification: dict[str, Any] | None = None,
    duration_seconds: float | None = None,
) -> None:
    """Deprecated: cloud-only processing for S3 videos (no server download).

    Kept for backward compatibility, but the application now prefers frame extraction + per-frame
    analysis for all videos (local or S3) so timestamp-based UI works consistently.
    """
    try:
        profile = _envid_metadata_output_profile()

        preset_default = _features_preset_default_enabled()
        if profile in {"required", "minimal"}:
            enable_labels = True
            enable_text = True
            enable_moderation = True
            enable_celebrities = True
            enable_transcribe = True
            enable_translate = True
            enable_comprehend = True
            enable_summary = True
            enable_polly = True
            enable_embedding = False
        else:
            enable_labels = _feature_enabled("LABELS", default=preset_default)
            enable_celebrities = _feature_enabled("CELEBRITIES", default=preset_default)
            enable_moderation = _feature_enabled("MODERATION", default=preset_default)
            enable_text = _feature_enabled("TEXT", default=preset_default)
            enable_transcribe = _feature_enabled("TRANSCRIBE", default=preset_default)
            enable_translate = _feature_enabled("TRANSLATE", default=preset_default)
            enable_comprehend = _feature_enabled("COMPREHEND", default=preset_default)
            enable_summary = _feature_enabled("SUMMARY", default=preset_default)
            enable_polly = _feature_enabled("POLLY", default=preset_default)
            enable_embedding = _feature_enabled("EMBEDDING", default=preset_default)

        bucket = (s3_bucket or "").strip()
        if not bucket:
            raise ValueError("Missing s3_bucket")

        key = (s3_key or "").strip().lstrip("/")
        if not key:
            raise ValueError("Missing s3_key")
        if bucket == _rawvideo_bucket():
            raw_prefix = _rawvideo_prefix().rstrip("/")
            # Only normalize when the key is actually a rawvideo-style key.
            # NOTE: rawvideo bucket and video bucket can be the same bucket in local setups.
            if ("/" not in key) or key.startswith(raw_prefix + "/"):
                key = _normalize_rawvideo_key(key)

        s3_uri = f"s3://{bucket}/{key}"
        _job_update(job_id, status="processing", progress=2, message="Starting cloud analysis", s3_video_key=key, s3_video_uri=s3_uri)

        # We may switch Rekognition/Transcribe to use a MediaConvert proxy object.
        analysis_bucket = bucket
        analysis_key = key
        s3_proxy_uri: str | None = None
        needs_proxy = False
        has_audio = True

        # Technical metadata verification:
        # - For local uploads, mediainfo is computed before upload and passed in.
        # - For S3-only sources, we only use MediaInfo Lambda when explicitly configured.
        if technical_mediainfo is None and _mediainfo_lambda_name_or_arn():
            _job_update(job_id, progress=3, message="MediaInfo: extracting technical metadata")
            technical_mediainfo = _invoke_mediainfo_lambda(s3_bucket=bucket, s3_key=key)
        elif technical_mediainfo is None and _mediainfo_s3_partial_enabled():
            _job_update(job_id, progress=3, message="MediaInfo: probing S3 header bytes")
            technical_mediainfo = _probe_mediainfo_s3_partial(bucket=bucket, key=key)

        # If we have MediaInfo, do a lightweight codec sanity check.
        # Rekognition Video commonly fails on HEVC/H.265 or uncommon containers.
        try:
            tracks = (technical_mediainfo or {}).get("tracks") if isinstance(technical_mediainfo, dict) else None
            if isinstance(tracks, list):
                video_track = next((t for t in tracks if str(t.get("@type") or "").lower() == "video"), None)
                audio_track = next((t for t in tracks if str(t.get("@type") or "").lower() == "audio"), None)
                has_audio = bool(audio_track)
                container_fmt = None
                general_track = next((t for t in tracks if str(t.get("@type") or "").lower() == "general"), None)
                if isinstance(general_track, dict):
                    container_fmt = (general_track.get("Format") or general_track.get("Format_Profile") or "")
                vfmt = ""
                if isinstance(video_track, dict):
                    vfmt = str(video_track.get("Format") or video_track.get("CodecID") or "")
                vfmt_l = vfmt.lower()
                if any(x in vfmt_l for x in ["hevc", "h.265", "h265", "x265"]):
                    needs_proxy = True
                    _job_update(
                        job_id,
                        progress=3,
                        message=(
                            "MediaInfo: detected HEVC/H.265; will use MediaConvert proxy (MP4 H.264/AAC) if configured. "
                            f"Detected: {vfmt}. Container: {container_fmt or 'unknown'}."
                        ),
                    )
        except RuntimeError:
            raise
        except Exception:
            pass
        if technical_verification is None and technical_ffprobe is not None:
            technical_verification = _verify_technical_metadata(ffprobe=technical_ffprobe, mediainfo=technical_mediainfo)

        # Preflight object access/region before launching Rekognition Video async jobs.
        _job_update(job_id, progress=4, message="S3: checking object metadata")
        head = _s3_head_object_or_raise(bucket=bucket, key=key)

        file_size_bytes: int | None = None
        try:
            file_size_bytes = int(head.get("ContentLength") or 0) or None
        except Exception:
            file_size_bytes = None

        # Rekognition Video has a hard object size limit (commonly 10 GiB for StartLabelDetection).
        # If we exceed it, fail early with a clear message rather than surfacing a vague Rekognition error.
        try:
            content_length = int(head.get("ContentLength") or 0)
        except Exception:
            content_length = 0
        max_bytes = 10 * 1024 * 1024 * 1024  # 10 GiB
        if content_length and content_length > max_bytes:
            needs_proxy = True
            _job_update(
                job_id,
                progress=4,
                message=(
                    "S3: object is >10GB; will use MediaConvert proxy if configured (otherwise Rekognition cannot process it)."
                ),
            )

        if needs_proxy:
            bucket_region_in = _detect_bucket_region(bucket)
            region_in = bucket_region_in or DEFAULT_AWS_REGION
            if _mediaconvert_enabled() and _mediaconvert_role_arn():
                analysis_bucket, analysis_key = _create_mediaconvert_proxy(
                    envid_job_id=job_id,
                    input_bucket=bucket,
                    input_key=key,
                    region_name=region_in,
                    has_audio=has_audio,
                )
                s3_proxy_uri = f"s3://{analysis_bucket}/{analysis_key}"
                proxy_head = _s3_head_object_or_raise(bucket=analysis_bucket, key=analysis_key)
                try:
                    proxy_len = int(proxy_head.get("ContentLength") or 0)
                except Exception:
                    proxy_len = 0
                if proxy_len and proxy_len > max_bytes:
                    raise RuntimeError(
                        "MediaConvert proxy is still too large for Rekognition Video "
                        f"({proxy_len} bytes > {max_bytes} bytes). Consider a lower bitrate or shorter clip."
                    )
            else:
                if content_length and content_length > max_bytes:
                    raise RuntimeError(
                        "Rekognition Video cannot process this S3 object because it is too large "
                        f"({content_length} bytes > {max_bytes} bytes). "
                        "Enable MediaConvert proxying (ENVID_METADATA_MEDIACONVERT_ENABLE=1 + ENVID_METADATA_MEDIACONVERT_ROLE_ARN), "
                        "or upload a smaller proxy/clip (<=10GB)."
                    )
                raise RuntimeError(
                    "Rekognition Video cannot process this S3 object due to unsupported codec/format. "
                    "Enable MediaConvert proxying (ENVID_METADATA_MEDIACONVERT_ENABLE=1 + ENVID_METADATA_MEDIACONVERT_ROLE_ARN), "
                    "or upload an MP4 (H.264/AVC video + AAC audio)."
                )

        bucket_region = _detect_bucket_region(analysis_bucket)
        region = bucket_region or DEFAULT_AWS_REGION
        rk = _rekognition_client(region_name=region)
        video_ref = _rekognition_video_ref(analysis_bucket, analysis_key)

        labels_raw: list[dict[str, Any]] = []
        label_job_id: str | None = None
        if enable_labels:
            _job_update(job_id, progress=5, message="Rekognition Video: starting label detection")
            _job_step_update(job_id, "rekognition_labels", status="running", percent=0, message="Starting")
            ld = rk.start_label_detection(
                Video=video_ref,
                MinConfidence=float(os.getenv("ENVID_METADATA_REKOGNITION_LABEL_MIN_CONFIDENCE") or 70.0),
            )
            label_job_id = ld.get("JobId")
            if not label_job_id:
                raise RuntimeError("Failed to start label detection")
            labels_raw = _poll_rekognition_video_job(
                envid_job_id=job_id,
                step_id="rekognition_labels",
                rekognition_job_id=label_job_id,
                get_page=rk.get_label_detection,
                page_key="NextToken",
                item_key="Labels",
                progress_base=10,
                progress_span=20,
                message="Rekognition Video: detecting labels",
            )
        else:
            _job_update(job_id, progress=30, message="Rekognition Video: label detection disabled")
            _job_step_update(job_id, "rekognition_labels", status="skipped", percent=100, message="Disabled")

        celebs_raw: list[dict[str, Any]] = []
        celeb_job_id: str | None = None
        if enable_celebrities:
            _job_update(job_id, progress=30, message="Rekognition Video: starting celebrity recognition")
            _job_step_update(job_id, "rekognition_celebrities", status="running", percent=0, message="Starting")
            cr = rk.start_celebrity_recognition(Video=video_ref)
            celeb_job_id = cr.get("JobId")
            if not celeb_job_id:
                raise RuntimeError("Failed to start celebrity recognition")
            celebs_raw = _poll_rekognition_video_job(
                envid_job_id=job_id,
                step_id="rekognition_celebrities",
                rekognition_job_id=celeb_job_id,
                get_page=rk.get_celebrity_recognition,
                page_key="NextToken",
                item_key="Celebrities",
                progress_base=32,
                progress_span=18,
                message="Rekognition Video: detecting celebrities",
            )
        else:
            _job_update(job_id, progress=50, message="Rekognition Video: celebrity recognition disabled")
            _job_step_update(job_id, "rekognition_celebrities", status="skipped", percent=100, message="Disabled")

        # Rekognition Video: moderation
        moderation_raw: list[dict[str, Any]] = []
        if enable_moderation:
            try:
                _job_update(job_id, progress=50, message="Rekognition Video: starting content moderation")
                _job_step_update(job_id, "content_moderation", status="running", percent=0, message="Starting")
                md = rk.start_content_moderation(
                    Video=video_ref,
                    MinConfidence=float(os.getenv("ENVID_METADATA_REKOGNITION_MODERATION_MIN_CONFIDENCE") or 60.0),
                )
                moderation_job_id = md.get("JobId")
                if moderation_job_id:
                    moderation_raw = _poll_rekognition_video_job(
                        envid_job_id=job_id,
                        step_id="content_moderation",
                        rekognition_job_id=moderation_job_id,
                        get_page=rk.get_content_moderation,
                        page_key="NextToken",
                        item_key="ModerationLabels",
                        progress_base=52,
                        progress_span=8,
                        message="Rekognition Video: detecting moderation labels",
                    )
            except Exception as exc:
                app.logger.warning("Moderation detection failed: %s", exc)
        else:
            _job_update(job_id, progress=60, message="Rekognition Video: moderation disabled")
            _job_step_update(job_id, "content_moderation", status="skipped", percent=100, message="Disabled")

        # Rekognition Video: text
        text_raw: list[dict[str, Any]] = []
        if enable_text:
            try:
                _job_update(job_id, progress=60, message="Rekognition Video: starting text detection")
                _job_step_update(job_id, "rekognition_text", status="running", percent=0, message="Starting")
                td = rk.start_text_detection(Video=video_ref)
                text_job_id = td.get("JobId")
                if text_job_id:
                    text_raw = _poll_rekognition_video_job(
                        envid_job_id=job_id,
                        step_id="rekognition_text",
                        rekognition_job_id=text_job_id,
                        get_page=rk.get_text_detection,
                        page_key="NextToken",
                        item_key="TextDetections",
                        progress_base=62,
                        progress_span=8,
                        message="Rekognition Video: detecting on-screen text",
                    )
            except Exception as exc:
                app.logger.warning("Text detection failed: %s", exc)
        else:
            _job_update(job_id, progress=70, message="Rekognition Video: text detection disabled")
            _job_step_update(job_id, "rekognition_text", status="skipped", percent=100, message="Disabled")

        # Rekognition can provide basic technical metadata even in cloud-only mode.
        # This helps populate container format / resolution in the required JSON without downloading the file.
        rekognition_video_metadata: dict[str, Any] | None = None
        try:
            md_resp = None
            if label_job_id:
                md_resp = rk.get_label_detection(JobId=label_job_id, MaxResults=1)
            elif celeb_job_id:
                md_resp = rk.get_celebrity_recognition(JobId=celeb_job_id, MaxResults=1)
            if isinstance(md_resp, dict) and isinstance(md_resp.get("VideoMetadata"), dict):
                rekognition_video_metadata = md_resp.get("VideoMetadata")
        except Exception:
            rekognition_video_metadata = None

        # Build time-mapped frame-like metadata for UI overlays + timelines.
        # The UI expects `frames[*].timestamp` and `frames[*].celebrities`.
        interval_in = int(frame_interval_seconds or 0)
        if interval_in <= 0:
            interval_in = _default_frame_interval_seconds(duration_seconds)
        interval = _parse_int_param(interval_in, default=10, min_value=1, max_value=30)

        dur_int = int(duration_seconds or 0)
        if dur_int <= 0:
            try:
                max_ms = max([int(float((x.get("Timestamp") or 0) or 0.0)) for x in (labels_raw + celebs_raw + moderation_raw + text_raw)] or [0])
            except Exception:
                max_ms = 0
            dur_int = max(1, int(max_ms / 1000))
            duration_seconds = float(dur_int)

        bins: Dict[int, Dict[str, Any]] = {}
        for t in range(0, max(1, dur_int + 1), interval):
            bins[int(t)] = {
                "timestamp": int(t),
                "labels": [],
                "text": [],
                "faces": [],
                "celebrities": [],
                "moderation": [],
                "custom_faces": [],
                "local_faces": [],
            }

        def _bin_for_second(sec: int) -> int:
            if sec < 0:
                sec = 0
            b = int(sec // interval) * interval
            if b not in bins:
                keys = sorted(bins.keys())
                if not keys:
                    return 0
                if b < keys[0]:
                    return keys[0]
                if b > keys[-1]:
                    return keys[-1]
                return keys[-1]
            return b

        def _dedupe_by_key(items: List[Dict[str, Any]], key_name: str) -> List[Dict[str, Any]]:
            seen = set()
            out: List[Dict[str, Any]] = []
            for it in items:
                k = (it.get(key_name) or "").strip() if isinstance(it.get(key_name), str) else it.get(key_name)
                if not k or k in seen:
                    continue
                seen.add(k)
                out.append(it)
            return out

        # For thumbnail crops, retain a lookup of bounding boxes by (name, bin_ts)
        celeb_bbox_by_name_and_bin: Dict[Tuple[str, int], Dict[str, Any]] = {}

        celeb_min_conf = _parse_float_param(
            os.getenv("ENVID_METADATA_CELEB_MIN_CONFIDENCE"),
            default=95.0,
            min_value=0.0,
            max_value=100.0,
        )
        celeb_edge_margin = _parse_float_param(
            os.getenv("ENVID_METADATA_CELEB_EDGE_MARGIN"),
            default=0.02,
            min_value=0.0,
            max_value=0.2,
        )
        celeb_min_bbox = _parse_float_param(
            os.getenv("ENVID_METADATA_CELEB_MIN_BBOX"),
            default=0.06,
            min_value=0.0,
            max_value=0.5,
        )

        celeb_verify_enable = (os.getenv("ENVID_METADATA_CELEB_VERIFY_IMAGE") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        celeb_verify_min_conf = _parse_float_param(
            os.getenv("ENVID_METADATA_CELEB_VERIFY_MIN_CONFIDENCE"),
            default=97.0,
            min_value=0.0,
            max_value=100.0,
        )
        celeb_verify_max_frames = _parse_int_param(
            os.getenv("ENVID_METADATA_CELEB_VERIFY_MAX_FRAMES"),
            default=30,
            min_value=0,
            max_value=300,
        )
        celeb_verify_allow_override = (os.getenv("ENVID_METADATA_CELEB_VERIFY_ALLOW_OVERRIDE") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        celeb_compare_enable = (os.getenv("ENVID_METADATA_CELEB_COMPAREFACES") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        celeb_compare_similarity = _parse_float_param(
            os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_SIMILARITY"),
            default=85.0,
            min_value=0.0,
            max_value=100.0,
        )
        celeb_compare_max_frames = _parse_int_param(
            os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_MAX_FRAMES"),
            default=int(celeb_verify_max_frames or 30),
            min_value=0,
            max_value=300,
        )
        celeb_compare_allow_override = (os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_ALLOW_OVERRIDE") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        celeb_compare_extra_names = [
            n.strip()
            for n in (os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_EXTRA_NAMES") or "").split(",")
            if n.strip()
        ][:25]

        celeb_compare_debug_seconds: set[int] = set()
        debug_raw = (os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_DEBUG_SECONDS") or "").strip()
        if debug_raw:
            for part in debug_raw.split(","):
                p = (part or "").strip()
                if not p:
                    continue
                try:
                    celeb_compare_debug_seconds.add(int(float(p)))
                except Exception:
                    continue

        celeb_compare_bbox_pad = _parse_float_param(
            os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_BBOX_PAD"),
            default=0.08,
            min_value=0.0,
            max_value=0.5,
        )

        celeb_compare_debug_apply = (os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_DEBUG_APPLY") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        celeb_compare_debug_min_similarity = _parse_float_param(
            os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_DEBUG_MIN_SIMILARITY"),
            default=0.0,
            min_value=0.0,
            max_value=100.0,
        )
        celeb_compare_debug_min_delta = _parse_float_param(
            os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_DEBUG_MIN_DELTA"),
            default=0.0,
            min_value=0.0,
            max_value=100.0,
        )

        verified_names_by_second: Dict[int, set[str]] = {}
        verified_best_name_by_second: Dict[int, str] = {}
        all_video_celeb_names: set[str] = set()
        urls_by_name: Dict[str, List[str]] = {}
        celeb_ts_ms_by_name: Dict[str, List[int]] = {}
        for it in celebs_raw:
            if not isinstance(it, dict):
                continue
            celeb_obj = it.get("Celebrity") if isinstance(it.get("Celebrity"), dict) else {}
            nm0 = (celeb_obj.get("Name") or "").strip() if isinstance(celeb_obj, dict) else ""
            if nm0:
                all_video_celeb_names.add(nm0)
                u0 = [u for u in (celeb_obj.get("Urls") or []) if isinstance(u, str) and u.strip()]
                if u0 and nm0 not in urls_by_name:
                    urls_by_name[nm0] = u0[:5]

        if celeb_verify_enable and celebs_raw and int(celeb_verify_max_frames or 0) > 0:

            verify_source: str | None = None
            try:
                verify_source = _presign_s3_get_object_url(bucket=analysis_bucket, key=analysis_key)
            except Exception:
                verify_source = None

            if verify_source:
                verify_temp_dir = Path(tempfile.mkdtemp(prefix=f"envid_metadata_celeb_verify_{job_id}_"))
                try:
                    def _extract_verify_frame_jpeg_bytes(*, ts_seconds: int) -> bytes | None:
                        try:
                            ts = max(0, int(ts_seconds))
                        except Exception:
                            ts = 0
                        out_path = verify_temp_dir / f"verify_{ts:06d}.jpg"
                        try:
                            if out_path.exists():
                                out_path.unlink()
                        except Exception:
                            pass
                        cmd = [
                            FFMPEG_PATH,
                            "-ss",
                            str(ts),
                            "-i",
                            str(verify_source),
                            "-vframes",
                            "1",
                            "-q:v",
                            "2",
                            str(out_path),
                            "-y",
                        ]
                        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        if res.returncode != 0 or not out_path.exists():
                            return None
                        try:
                            return out_path.read_bytes()
                        except Exception:
                            return None
                        finally:
                            try:
                                out_path.unlink()
                            except Exception:
                                pass

                    try:
                        candidate_seconds = sorted(
                            {
                                int(float(item.get("Timestamp") or 0.0) / 1000.0)
                                for item in celebs_raw
                                if isinstance(item, dict)
                            }
                        )
                    except Exception:
                        candidate_seconds = []

                    checked = 0
                    for sec in candidate_seconds:
                        if checked >= int(celeb_verify_max_frames):
                            break
                        fb = _extract_verify_frame_jpeg_bytes(ts_seconds=int(sec))
                        if not fb:
                            continue
                        try:
                            resp = rk.recognize_celebrities(Image={"Bytes": fb})
                        except Exception:
                            continue
                        names: set[str] = set()
                        best_name = ""
                        best_conf = -1.0
                        for face in (resp.get("CelebrityFaces") or []):
                            if not isinstance(face, dict):
                                continue
                            nm = (face.get("Name") or "").strip()
                            if not nm:
                                continue
                            try:
                                mconf = float(face.get("MatchConfidence") or 0.0)
                            except Exception:
                                mconf = 0.0
                            if mconf < float(celeb_verify_min_conf):
                                continue
                            face_obj = face.get("Face") if isinstance(face.get("Face"), dict) else {}
                            bbox_img = face_obj.get("BoundingBox") if isinstance(face_obj, dict) else None
                            if _bbox_looks_partial_or_too_small(
                                bbox_img if isinstance(bbox_img, dict) else None,
                                edge_margin=celeb_edge_margin,
                                min_width=celeb_min_bbox,
                                min_height=celeb_min_bbox,
                            ):
                                continue
                            names.add(nm)
                            if mconf > best_conf:
                                best_conf = mconf
                                best_name = nm

                        if names:
                            verified_names_by_second[int(sec)] = names
                            if best_name:
                                verified_best_name_by_second[int(sec)] = best_name
                        checked += 1
                finally:
                    try:
                        shutil.rmtree(verify_temp_dir, ignore_errors=True)
                    except Exception:
                        pass

        comparefaces_best_name_by_second: Dict[int, str] = {}
        comparefaces_best_similarity_by_second: Dict[int, float] = {}
        comparefaces_candidate_names: List[str] = []
        if celeb_compare_enable and celebs_raw and int(celeb_compare_max_frames or 0) > 0:
            if celeb_compare_debug_seconds:
                app.logger.warning(
                    "CompareFaces: starting (debug_seconds=%s; celebs_raw=%d)",
                    ",".join(str(x) for x in sorted(celeb_compare_debug_seconds)),
                    int(len(celebs_raw)),
                )
            compare_source: str | None = None
            try:
                compare_source = _presign_s3_get_object_url(bucket=analysis_bucket, key=analysis_key)
            except Exception:
                compare_source = None

            if compare_source:
                compare_temp_dir = Path(tempfile.mkdtemp(prefix=f"envid_metadata_celeb_compare_{job_id}_"))
                try:
                    def _extract_compare_frame_jpeg_bytes(*, ts_seconds: float) -> bytes | None:
                        try:
                            ts_f = max(0.0, float(ts_seconds))
                        except Exception:
                            ts_f = 0.0
                        ts_ms = int(round(ts_f * 1000.0))
                        out_path = compare_temp_dir / f"compare_{ts_ms:09d}.jpg"
                        try:
                            if out_path.exists():
                                out_path.unlink()
                        except Exception:
                            pass
                        cmd = [
                            FFMPEG_PATH,
                            "-ss",
                            f"{ts_f:.3f}",
                            "-i",
                            str(compare_source),
                            "-vframes",
                            "1",
                            "-q:v",
                            "2",
                            str(out_path),
                            "-y",
                        ]
                        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        if res.returncode != 0 or not out_path.exists():
                            return None
                        try:
                            return out_path.read_bytes()
                        except Exception:
                            return None
                        finally:
                            try:
                                out_path.unlink()
                            except Exception:
                                pass

                    try:
                        distinct_names = sorted(
                            {
                                (it.get("Celebrity") or {}).get("Name")
                                for it in celebs_raw
                                if isinstance(it, dict) and isinstance(it.get("Celebrity"), dict)
                            }
                        )
                    except Exception:
                        distinct_names = []
                    distinct_names = [(n or "").strip() for n in distinct_names if isinstance(n, str) and (n or "").strip()]
                    comparefaces_candidate_names = sorted(set(distinct_names + celeb_compare_extra_names))[:50]

                    portrait_meta = _celebrity_bios_for_names(comparefaces_candidate_names) if comparefaces_candidate_names else {}

                    portrait_bytes_by_name: Dict[str, bytes] = {}
                    for nm in comparefaces_candidate_names:
                        url = ((portrait_meta.get(nm) or {}).get("portrait_url") or "").strip()
                        if not url:
                            continue
                        pb = _http_get_bytes(url)
                        pb_norm = _normalize_image_bytes_for_rekognition(pb) if pb else None
                        if pb_norm:
                            portrait_bytes_by_name[nm] = pb_norm

                    sec_to_best_item: Dict[int, Dict[str, Any]] = {}
                    for it in celebs_raw:
                        if not isinstance(it, dict):
                            continue
                        try:
                            sec = int(float(it.get("Timestamp") or 0.0) / 1000.0)
                        except Exception:
                            sec = 0
                        debug_sec = int(sec) in celeb_compare_debug_seconds
                        celeb_obj = it.get("Celebrity") if isinstance(it.get("Celebrity"), dict) else {}
                        nm = (celeb_obj.get("Name") or "").strip() if isinstance(celeb_obj, dict) else ""
                        if not nm:
                            continue
                        try:
                            conf0 = float(it.get("MatchConfidence") or celeb_obj.get("Confidence") or 0.0)
                        except Exception:
                            conf0 = 0.0
                        face = celeb_obj.get("Face") if isinstance(celeb_obj.get("Face"), dict) else {}
                        bbox0 = face.get("BoundingBox") if isinstance(face, dict) else None
                        # For explicit debug seconds, allow evaluation even if the raw Rekognition
                        # hit fails our strict confidence/partial-face filters.
                        if (not debug_sec) and conf0 < float(celeb_min_conf):
                            continue
                        if (not debug_sec) and _bbox_looks_partial_or_too_small(
                            bbox0 if isinstance(bbox0, dict) else None,
                            edge_margin=celeb_edge_margin,
                            min_width=celeb_min_bbox,
                            min_height=celeb_min_bbox,
                        ):
                            continue
                        if not isinstance(bbox0, dict) or not bbox0:
                            continue

                        try:
                            area0 = float(bbox0.get("Width") or 0.0) * float(bbox0.get("Height") or 0.0)
                        except Exception:
                            area0 = 0.0

                        prev = sec_to_best_item.get(int(sec))
                        prev_conf = float(prev.get("conf") or 0.0) if isinstance(prev, dict) else -1.0
                        prev_area = float(prev.get("area") or 0.0) if isinstance(prev, dict) else -1.0
                        if prev is None or conf0 > prev_conf or (conf0 == prev_conf and area0 > prev_area):
                            sec_to_best_item[int(sec)] = {
                                "detected_name": nm,
                                "bbox": bbox0,
                                "conf": conf0,
                                "area": area0,
                                "ts_s": float(it.get("Timestamp") or 0.0) / 1000.0,
                            }

                    # If the user asked to debug a specific second, ensure we can evaluate that second
                    # even when Rekognition Video's millisecond timestamps fall just before/after the
                    # integer boundary.
                    #
                    # 1) Prefer: pick the best raw celeb item within a +/- window around the requested
                    #    second (based on the true float timestamp), and store it under the requested
                    #    debug second key.
                    # 2) Fallback: borrow the nearest bbox/name from an adjacent second BUT force the
                    #    frame extraction timestamp to the requested debug second.
                    if celeb_compare_debug_seconds:
                        try:
                            nearby_window_s = int(
                                os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_DEBUG_NEARBY_WINDOW_S", "2") or 2
                            )
                        except Exception:
                            nearby_window_s = 2

                        def _best_item_near_debug_second(ds: int) -> Dict[str, Any] | None:
                            # For debug seconds, prioritize *time proximity* to ds so we evaluate
                            # the intended moment, not the highest-confidence nearby hit.
                            best: Dict[str, Any] | None = None
                            best_dt = 1e9
                            best_conf = -1.0
                            best_area = -1.0
                            for it2 in celebs_raw:
                                if not isinstance(it2, dict):
                                    continue
                                celeb_obj2 = it2.get("Celebrity") if isinstance(it2.get("Celebrity"), dict) else {}
                                nm2 = (celeb_obj2.get("Name") or "").strip() if isinstance(celeb_obj2, dict) else ""
                                if not nm2:
                                    continue
                                try:
                                    ts_s2 = float(it2.get("Timestamp") or 0.0) / 1000.0
                                except Exception:
                                    ts_s2 = 0.0
                                dt = abs(float(ts_s2) - float(ds))
                                if dt > float(nearby_window_s):
                                    continue
                                face2 = celeb_obj2.get("Face") if isinstance(celeb_obj2.get("Face"), dict) else {}
                                bbox2 = face2.get("BoundingBox") if isinstance(face2, dict) else None
                                if not isinstance(bbox2, dict) or not bbox2:
                                    continue
                                try:
                                    conf2 = float(it2.get("MatchConfidence") or celeb_obj2.get("Confidence") or 0.0)
                                except Exception:
                                    conf2 = 0.0
                                try:
                                    area2 = float(bbox2.get("Width") or 0.0) * float(bbox2.get("Height") or 0.0)
                                except Exception:
                                    area2 = 0.0
                                if (
                                    best is None
                                    or dt < best_dt
                                    or (dt == best_dt and conf2 > best_conf)
                                    or (dt == best_dt and conf2 == best_conf and area2 > best_area)
                                ):
                                    best = {
                                        "detected_name": nm2,
                                        "bbox": bbox2,
                                        "conf": conf2,
                                        "area": area2,
                                        "ts_s": ts_s2,
                                        "_debug_from_ts_s": ts_s2,
                                    }
                                    best_dt = dt
                                    best_conf = conf2
                                    best_area = area2
                            return best

                        existing_secs_sorted = sorted(sec_to_best_item.keys())
                        for ds in sorted(celeb_compare_debug_seconds):
                            if ds in sec_to_best_item:
                                continue

                            cand = _best_item_near_debug_second(int(ds))
                            if cand:
                                # For debug seconds we want to evaluate the exact requested second,
                                # even when the nearest raw Rekognition hit is slightly earlier/later.
                                # Keep the raw timestamp for logging, but force frame extraction to ds.
                                cand["_debug_force_ts_s"] = float(ds)
                                cand["ts_s"] = float(ds)
                                sec_to_best_item[int(ds)] = cand
                                app.logger.warning(
                                    "CompareFaces debug sec=%d missing; using nearby raw ts_s=%.3f (detected=%s) and forcing frame_ts_s=%.3f",
                                    int(ds),
                                    float(cand.get("_debug_from_ts_s") or 0.0),
                                    str(cand.get("detected_name") or ""),
                                    float(ds),
                                )
                                continue

                            if existing_secs_sorted:
                                closest = min(existing_secs_sorted, key=lambda s: abs(int(s) - int(ds)))
                                if abs(int(closest) - int(ds)) <= nearby_window_s:
                                    borrowed = dict(sec_to_best_item.get(int(closest)) or {})
                                    borrowed["_debug_from_sec"] = int(closest)
                                    borrowed["_debug_force_ts_s"] = float(ds)
                                    borrowed["ts_s"] = float(ds)
                                    sec_to_best_item[int(ds)] = borrowed
                                    app.logger.warning(
                                        "CompareFaces debug sec=%d missing; borrowing bbox from sec=%d (detected=%s) and forcing frame_ts_s=%.3f",
                                        int(ds),
                                        int(closest),
                                        str((sec_to_best_item.get(int(closest)) or {}).get("detected_name") or ""),
                                        float(ds),
                                    )

                    if celeb_compare_debug_seconds:
                        for ds in sorted(celeb_compare_debug_seconds):
                            if ds in sec_to_best_item:
                                app.logger.warning(
                                    "CompareFaces debug seconds: sec=%d present (detected=%s)",
                                    int(ds),
                                    str((sec_to_best_item.get(ds) or {}).get("detected_name") or ""),
                                )
                            else:
                                app.logger.warning("CompareFaces debug seconds: sec=%d NOT present", int(ds))

                    frames_for_compare: Dict[int, bytes] = {}
                    seconds = sorted(sec_to_best_item.keys())[: int(celeb_compare_max_frames)]
                    for sec in seconds:
                        best_item = sec_to_best_item.get(int(sec)) or {}
                        try:
                            frame_ts_s = float(best_item.get("ts_s") or float(sec)) if isinstance(best_item, dict) else float(sec)
                        except Exception:
                            frame_ts_s = float(sec)
                        frame_key = int(round(frame_ts_s * 1000.0))
                        if int(sec) in celeb_compare_debug_seconds and isinstance(best_item, dict):
                            if best_item.get("_debug_from_sec") is not None:
                                app.logger.warning(
                                    "CompareFaces debug sec=%d using frame_ts_s=%.3f (borrowed bbox from sec=%s)",
                                    int(sec),
                                    float(frame_ts_s),
                                    str(best_item.get("_debug_from_sec")),
                                )
                            elif best_item.get("_debug_force_ts_s") is not None and best_item.get("_debug_from_ts_s") is not None:
                                app.logger.warning(
                                    "CompareFaces debug sec=%d using frame_ts_s=%.3f (forced; bbox from raw ts_s=%.3f)",
                                    int(sec),
                                    float(frame_ts_s),
                                    float(best_item.get("_debug_from_ts_s") or 0.0),
                                )
                            elif best_item.get("_debug_from_ts_s") is not None:
                                app.logger.warning(
                                    "CompareFaces debug sec=%d using frame_ts_s=%.3f (matched nearby raw ts_s=%.3f)",
                                    int(sec),
                                    float(frame_ts_s),
                                    float(best_item.get("_debug_from_ts_s") or 0.0),
                                )

                        if frame_key not in frames_for_compare:
                            fb = _extract_compare_frame_jpeg_bytes(ts_seconds=float(frame_ts_s))
                            if fb:
                                frames_for_compare[frame_key] = fb
                        frame_bytes = frames_for_compare.get(frame_key)
                        if not frame_bytes:
                            if int(sec) in celeb_compare_debug_seconds:
                                app.logger.warning("CompareFaces debug sec=%d skipped: no frame bytes", int(sec))
                            continue
                        # If we're forcing the debug frame timestamp (e.g., sec=10 uses frame at 10.000s
                        # but bbox from a nearby raw hit at 9.000s), re-detect faces on the *actual* frame
                        # so CompareFaces gets a meaningful target face crop.
                        bbox0 = best_item.get("bbox")
                        if int(sec) in celeb_compare_debug_seconds and isinstance(best_item, dict):
                            if best_item.get("_debug_force_ts_s") is not None:
                                df_bbox = _rekognition_detect_largest_face_bbox(rk, frame_bytes)
                                if isinstance(df_bbox, dict) and df_bbox:
                                    bbox0 = df_bbox
                                    best_item["_debug_bbox_from_detect_faces"] = True
                                    app.logger.warning(
                                        "CompareFaces debug sec=%d bbox_source=detect_faces",
                                        int(sec),
                                    )
                        if not isinstance(bbox0, dict) or not bbox0:
                            if int(sec) in celeb_compare_debug_seconds:
                                app.logger.warning("CompareFaces debug sec=%d skipped: no bbox", int(sec))
                            continue

                        bbox_for_crop = bbox0
                        try:
                            pad = float(celeb_compare_bbox_pad)
                        except Exception:
                            pad = 0.0
                        if pad > 0.0:
                            try:
                                l = float(bbox0.get("Left") or 0.0)
                                t = float(bbox0.get("Top") or 0.0)
                                w = float(bbox0.get("Width") or 0.0)
                                h = float(bbox0.get("Height") or 0.0)
                                l2 = max(0.0, l - w * pad)
                                t2 = max(0.0, t - h * pad)
                                w2 = min(1.0 - l2, w * (1.0 + 2.0 * pad))
                                h2 = min(1.0 - t2, h * (1.0 + 2.0 * pad))
                                bbox_for_crop = {"Left": l2, "Top": t2, "Width": w2, "Height": h2}
                            except Exception:
                                bbox_for_crop = bbox0

                        best_name = ""
                        best_sim = -1.0
                        second_name = ""
                        second_sim = -1.0

                        debug_this_sec = int(sec) in celeb_compare_debug_seconds

                        try:
                            compare_crop_min_px = int(
                                os.getenv("ENVID_METADATA_CELEB_COMPAREFACES_CROP_MIN_PX", "80") or 80
                            )
                        except Exception:
                            compare_crop_min_px = 80

                        target_crop = _pil_crop_bbox_to_jpeg_bytes(
                            frame_bytes, bbox_for_crop, min_size_px=int(compare_crop_min_px)
                        )
                        # Some Rekognition APIs are picky about small target images; if the crop is too
                        # small, fall back to comparing against the full frame.
                        target_bytes_for_compare = target_crop or frame_bytes
                        if debug_this_sec:
                            app.logger.warning(
                                "CompareFaces debug sec=%d target_image=%s",
                                int(sec),
                                "crop" if target_crop else "full_frame",
                            )

                        for nm in comparefaces_candidate_names:
                            pb = portrait_bytes_by_name.get(nm)
                            if not pb:
                                if debug_this_sec:
                                    app.logger.warning(
                                        "CompareFaces debug sec=%d candidate=%s skipped: no portrait bytes",
                                        int(sec),
                                        str(nm),
                                    )
                                continue
                            try:
                                resp = rk.compare_faces(
                                    SourceImage={"Bytes": pb},
                                    TargetImage={"Bytes": target_bytes_for_compare},
                                    SimilarityThreshold=0.0,
                                )
                            except Exception as exc:
                                if debug_this_sec:
                                    app.logger.warning(
                                        "CompareFaces debug sec=%d candidate=%s compare_faces error: %s: %s",
                                        int(sec),
                                        str(nm),
                                        exc.__class__.__name__,
                                        str(exc)[:200],
                                    )
                                continue
                            sims = []
                            for fm in (resp.get("FaceMatches") or []):
                                if not isinstance(fm, dict):
                                    continue
                                try:
                                    sims.append(float(fm.get("Similarity") or 0.0))
                                except Exception:
                                    pass
                            sim = max(sims) if sims else -1.0
                            if debug_this_sec:
                                try:
                                    match_count = int(len(resp.get("FaceMatches") or []))
                                except Exception:
                                    match_count = 0
                                app.logger.warning(
                                    "CompareFaces debug sec=%d candidate=%s sim=%.1f matches=%d",
                                    int(sec),
                                    str(nm),
                                    float(sim),
                                    match_count,
                                )
                            if sim > best_sim:
                                second_sim = best_sim
                                second_name = best_name
                                best_sim = sim
                                best_name = nm
                            elif sim > second_sim:
                                second_sim = sim
                                second_name = nm

                        if int(sec) in celeb_compare_debug_seconds:
                            detected_nm = (best_item.get("detected_name") or "").strip() if isinstance(best_item, dict) else ""
                            app.logger.warning(
                                "CompareFaces debug sec=%d detected=%s best=%s sim=%.1f second=%s sim2=%.1f (candidates=%d)",
                                int(sec),
                                detected_nm,
                                (best_name or ""),
                                float(best_sim),
                                (second_name or ""),
                                float(second_sim),
                                int(len(comparefaces_candidate_names)),
                            )

                        if best_name:
                            record = False
                            if best_sim >= float(celeb_compare_similarity):
                                record = True
                            elif (
                                debug_this_sec
                                and celeb_compare_debug_apply
                                and best_sim >= float(celeb_compare_debug_min_similarity)
                                and (best_sim - (second_sim if second_sim >= 0.0 else 0.0)) >= float(celeb_compare_debug_min_delta)
                            ):
                                record = True
                                app.logger.warning(
                                    "CompareFaces debug sec=%d APPLY best=%s sim=%.1f second=%s sim2=%.1f (debug_apply=true)",
                                    int(sec),
                                    (best_name or ""),
                                    float(best_sim),
                                    (second_name or ""),
                                    float(second_sim),
                                )

                            if record:
                                comparefaces_best_name_by_second[int(sec)] = best_name
                                comparefaces_best_similarity_by_second[int(sec)] = float(best_sim)
                finally:
                    try:
                        shutil.rmtree(compare_temp_dir, ignore_errors=True)
                    except Exception:
                        pass

            else:
                if celeb_compare_debug_seconds:
                    app.logger.warning("CompareFaces: skipped (no compare_source URL)")

        dropped_by_verify = 0
        checked_by_verify = 0
        overridden_by_verify = 0
        dropped_by_comparefaces = 0
        checked_by_comparefaces = 0
        overridden_by_comparefaces = 0

        for item in labels_raw:
            try:
                sec = int(float(item.get("Timestamp") or 0.0) / 1000.0)
            except Exception:
                sec = 0
            b = _bin_for_second(sec)
            lbl = item.get("Label") or {}
            name = (lbl.get("Name") or "").strip()
            if not name:
                continue
            try:
                conf = float(lbl.get("Confidence") or 0.0)
            except Exception:
                conf = 0.0
            bins[b]["labels"].append({"name": name, "confidence": conf})

        for item in moderation_raw:
            try:
                sec = int(float(item.get("Timestamp") or 0.0) / 1000.0)
            except Exception:
                sec = 0
            b = _bin_for_second(sec)
            ml = item.get("ModerationLabel") or {}
            name = (ml.get("Name") or "").strip()
            if not name:
                continue
            try:
                conf = float(ml.get("Confidence") or 0.0)
            except Exception:
                conf = 0.0
            bins[b]["moderation"].append({"name": name, "confidence": conf})

        for item in text_raw:
            try:
                sec = int(float(item.get("Timestamp") or 0.0) / 1000.0)
            except Exception:
                sec = 0
            b = _bin_for_second(sec)
            td = item.get("TextDetection") or {}
            if (td.get("Type") or "") != "LINE":
                continue
            text_val = (td.get("DetectedText") or "").strip()
            if not text_val:
                continue
            try:
                conf = float(td.get("Confidence") or 0.0)
            except Exception:
                conf = 0.0
            bins[b]["text"].append({"type": "LINE", "text": text_val, "confidence": conf})

        for item in celebs_raw:
            try:
                sec = int(float(item.get("Timestamp") or 0.0) / 1000.0)
            except Exception:
                sec = 0
            b = _bin_for_second(sec)
            celeb = item.get("Celebrity") or {}
            name = (celeb.get("Name") or "").strip()
            if not name:
                continue
            urls = [u for u in (celeb.get("Urls") or []) if isinstance(u, str) and u.strip()]
            try:
                conf = float(item.get("MatchConfidence") or celeb.get("Confidence") or 0.0)
            except Exception:
                conf = 0.0
            face = celeb.get("Face") if isinstance(celeb.get("Face"), dict) else {}
            bbox = face.get("BoundingBox") if isinstance(face, dict) else None

            if conf < float(celeb_min_conf):
                continue
            if _bbox_looks_partial_or_too_small(
                bbox if isinstance(bbox, dict) else None,
                edge_margin=celeb_edge_margin,
                min_width=celeb_min_bbox,
                min_height=celeb_min_bbox,
            ):
                continue

            if celeb_verify_enable and sec in verified_names_by_second:
                checked_by_verify += 1
                verified_names = verified_names_by_second.get(sec) or set()
                if name not in verified_names:
                    if celeb_verify_allow_override:
                        best_name = (verified_best_name_by_second.get(sec) or "").strip()
                        if best_name and best_name in verified_names and best_name != name:
                            if best_name in all_video_celeb_names:
                                name = best_name
                                urls = urls_by_name.get(best_name) or urls
                                overridden_by_verify += 1
                            else:
                                dropped_by_verify += 1
                                continue
                        else:
                            dropped_by_verify += 1
                            continue
                    else:
                        dropped_by_verify += 1
                        continue

            if celeb_compare_enable and sec in comparefaces_best_name_by_second:
                checked_by_comparefaces += 1
                best_name = (comparefaces_best_name_by_second.get(sec) or "").strip()
                if best_name and best_name != name:
                    sim0 = float(comparefaces_best_similarity_by_second.get(sec) or 0.0)
                    if int(sec) in celeb_compare_debug_seconds:
                        app.logger.warning(
                            "CompareFaces override sec=%d %s -> %s (sim=%.1f)",
                            int(sec),
                            str(name),
                            str(best_name),
                            float(sim0),
                        )
                    if celeb_compare_allow_override:
                        if (best_name not in all_video_celeb_names) and (best_name not in celeb_compare_extra_names):
                            dropped_by_comparefaces += 1
                            continue
                        name = best_name
                        urls = urls_by_name.get(best_name) or urls
                        overridden_by_comparefaces += 1
                    else:
                        dropped_by_comparefaces += 1
                        continue

            # Preserve the original Rekognition Video timestamp (milliseconds) for sub-second UIs.
            try:
                ts_ms = int(float(item.get("Timestamp") or 0.0))
            except Exception:
                ts_ms = int(sec) * 1000
            celeb_ts_ms_by_name.setdefault(name, []).append(max(0, int(ts_ms)))
            bins[b]["celebrities"].append({"name": name, "confidence": conf, "urls": urls})
            if isinstance(bbox, dict) and bbox and (name, b) not in celeb_bbox_by_name_and_bin:
                celeb_bbox_by_name_and_bin[(name, b)] = bbox

        if celeb_verify_enable and dropped_by_verify:
            app.logger.info(
                "Celebrity verify (image) dropped %d/%d Rekognition-Video hits (max_frames=%s, min_conf=%s)",
                dropped_by_verify,
                checked_by_verify,
                str(celeb_verify_max_frames),
                str(celeb_verify_min_conf),
            )
        if celeb_verify_enable and overridden_by_verify:
            app.logger.info(
                "Celebrity verify (image) overridden %d/%d Rekognition-Video hits (allow_override=%s)",
                overridden_by_verify,
                checked_by_verify,
                str(celeb_verify_allow_override),
            )
        if celeb_compare_enable:
            app.logger.warning(
                "Celebrity comparefaces checked=%d dropped=%d overridden=%d (candidates=%d, extra=%d, debug=%d, debug_apply=%s, debug_min_sim=%.1f, debug_min_delta=%.1f, pad=%.2f, similarity>=%.1f, allow_override=%s)",
                checked_by_comparefaces,
                dropped_by_comparefaces,
                overridden_by_comparefaces,
                int(len(comparefaces_candidate_names) if isinstance(comparefaces_candidate_names, list) else 0),
                int(len(celeb_compare_extra_names) if isinstance(celeb_compare_extra_names, list) else 0),
                int(len(celeb_compare_debug_seconds) if isinstance(celeb_compare_debug_seconds, set) else 0),
                str(celeb_compare_debug_apply),
                float(celeb_compare_debug_min_similarity),
                float(celeb_compare_debug_min_delta),
                float(celeb_compare_bbox_pad),
                float(celeb_compare_similarity),
                str(celeb_compare_allow_override),
            )

        for t, obj in bins.items():
            obj["labels"] = _dedupe_by_key(sorted(obj["labels"], key=lambda x: float(x.get("confidence") or 0.0), reverse=True), "name")[:12]
            obj["moderation"] = _dedupe_by_key(sorted(obj["moderation"], key=lambda x: float(x.get("confidence") or 0.0), reverse=True), "name")[:12]
            obj["text"] = _dedupe_by_key(sorted(obj["text"], key=lambda x: float(x.get("confidence") or 0.0), reverse=True), "text")[:12]
            obj["celebrities"] = _dedupe_by_key(sorted(obj["celebrities"], key=lambda x: float(x.get("confidence") or 0.0), reverse=True), "name")[:12]

        frames_metadata: List[Dict[str, Any]] = [bins[t] for t in sorted(bins.keys())]

        # Aggregate Rekognition Video results
        label_stats: dict[str, dict[str, float]] = {}
        for item in labels_raw:
            lbl = (item.get("Label") or {})
            name = (lbl.get("Name") or "").strip()
            if not name:
                continue
            conf = float(lbl.get("Confidence") or 0.0)
            stat = label_stats.setdefault(name, {"count": 0.0, "max": 0.0, "sum": 0.0})
            stat["count"] += 1.0
            stat["sum"] += conf
            stat["max"] = max(stat["max"], conf)
        labels_ranked = sorted(
            (
                (
                    name,
                    int(stats.get("count", 0.0) or 0),
                    float(stats.get("max", 0.0) or 0.0),
                    float(stats.get("sum", 0.0) or 0.0) / max(1.0, float(stats.get("count", 0.0) or 1.0)),
                )
                for name, stats in label_stats.items()
            ),
            key=lambda row: (row[1], row[2], row[0]),
            reverse=True,
        )
        labels_detailed = [
            {"name": name, "occurrences": count, "max_confidence": max_conf, "avg_confidence": avg_conf}
            for (name, count, max_conf, avg_conf) in labels_ranked[:50]
        ]

        celeb_stats: Dict[str, Dict[str, Any]] = {}
        for f in frames_metadata:
            ts = f.get("timestamp")
            for c in (f.get("celebrities") or []):
                name = (c.get("name") or "").strip()
                if not name:
                    continue
                try:
                    conf = float(c.get("confidence") or 0.0)
                except Exception:
                    conf = 0.0
                urls = [u for u in (c.get("urls") or []) if isinstance(u, str) and u.strip()]
                stat = celeb_stats.get(name) or {
                    "name": name,
                    "max_confidence": 0.0,
                    "occurrences": 0,
                    "first_seen_seconds": ts,
                    "last_seen_seconds": ts,
                    "urls": [],
                    "thumbnail": None,
                }
                stat["max_confidence"] = max(float(stat.get("max_confidence") or 0.0), conf)
                stat["occurrences"] = int(stat.get("occurrences") or 0) + 1
                try:
                    if ts is not None:
                        stat["first_seen_seconds"] = (
                            min(int(stat.get("first_seen_seconds") or ts), int(ts))
                            if stat.get("first_seen_seconds") is not None
                            else int(ts)
                        )
                        stat["last_seen_seconds"] = (
                            max(int(stat.get("last_seen_seconds") or ts), int(ts))
                            if stat.get("last_seen_seconds") is not None
                            else int(ts)
                        )
                except Exception:
                    pass
                if urls:
                    existing_urls = [u for u in (stat.get("urls") or []) if isinstance(u, str)]
                    merged = list(dict.fromkeys(existing_urls + urls))
                    stat["urls"] = merged[:5]
                celeb_stats[name] = stat

        celebs_ranked = sorted(
            list(celeb_stats.values()),
            key=lambda x: (float(x.get("max_confidence") or 0.0), str(x.get("name") or "")),
            reverse=True,
        )[:50]

        bios = _celebrity_bios_for_names([c.get("name") for c in celebs_ranked if c.get("name")])

        # Thumbnails (cloud-only): extract a JPEG frame via ffmpeg from a presigned S3 URL, then crop using Rekognition bbox.
        temp_dir = Path(tempfile.mkdtemp(prefix=f"envid_metadata_thumbs_{job_id}_"))
        frames_cache: Dict[int, bytes] = {}
        thumb_input_source: str | None = None
        try:
            try:
                thumb_input_source = _presign_s3_get_object_url(bucket=analysis_bucket, key=analysis_key)
            except Exception:
                thumb_input_source = None

            def _extract_single_frame_jpeg_bytes(*, ts_seconds: int) -> bytes | None:
                if not thumb_input_source:
                    return None
                try:
                    ts = max(0, int(ts_seconds))
                except Exception:
                    ts = 0
                out_path = temp_dir / f"thumb_{ts:06d}.jpg"
                try:
                    if out_path.exists():
                        out_path.unlink()
                except Exception:
                    pass

                cmd = [
                    FFMPEG_PATH,
                    "-ss",
                    str(ts),
                    "-i",
                    str(thumb_input_source),
                    "-vframes",
                    "1",
                    "-q:v",
                    "2",
                    str(out_path),
                    "-y",
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if res.returncode != 0 or not out_path.exists():
                    return None
                try:
                    return out_path.read_bytes()
                except Exception:
                    return None
                finally:
                    try:
                        out_path.unlink()
                    except Exception:
                        pass

            thumb_by_name: Dict[str, str] = {}
            celebs_detailed = []
            for c in celebs_ranked:
                nm = (c.get("name") or "").strip()
                if not nm:
                    continue
                bio_obj = bios.get(nm) or {}

                first_seen = c.get("first_seen_seconds")
                ts_for_crop = int(first_seen) if first_seen is not None else 0
                b = _bin_for_second(ts_for_crop)
                bbox = celeb_bbox_by_name_and_bin.get((nm, b))

                thumb_b64 = None
                if isinstance(bbox, dict) and bbox:
                    if b not in frames_cache:
                        fb = _extract_single_frame_jpeg_bytes(ts_seconds=b)
                        if fb:
                            frames_cache[b] = fb
                    if b in frames_cache:
                        thumb_b64 = _pil_crop_bbox_thumbnail_base64(frames_cache[b], bbox)

                if thumb_b64:
                    thumb_by_name[nm] = thumb_b64

                celebs_detailed.append(
                    {
                        "name": nm,
                        "max_confidence": float(c.get("max_confidence") or 0.0),
                        "occurrences": int(c.get("occurrences") or 0),
                        "first_seen_seconds": c.get("first_seen_seconds"),
                        "last_seen_seconds": c.get("last_seen_seconds"),
                        "urls": c.get("urls") or [],
                        "thumbnail": thumb_b64,
                        "bio": (bio_obj.get("bio") or "").strip(),
                        "bio_source": bio_obj.get("source") or None,
                        "portrait_url": (bio_obj.get("portrait_url") or "").strip() or None,
                        "portrait_source": (bio_obj.get("portrait_source") or "").strip() or None,
                        "portrait_license": (bio_obj.get("portrait_license") or "").strip() or None,
                        "portrait_license_url": (bio_obj.get("portrait_license_url") or "").strip() or None,
                        "portrait_attribution": (bio_obj.get("portrait_attribution") or "").strip() or None,
                    }
                )

            # Also attach thumbnails to the frame overlay payload (best-effort, by name).
            if thumb_by_name:
                for f in frames_metadata:
                    for c in (f.get("celebrities") or []):
                        nm = (c.get("name") or "").strip()
                        if nm and (not c.get("thumbnail")) and nm in thumb_by_name:
                            c["thumbnail"] = thumb_by_name[nm]

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Ensure thumbnails are time-mapped for UI + required JSON.
        celebs_detailed = _attach_celebrity_timestamps(frames=frames_metadata, celebrities_detailed=celebs_detailed)

        # Attach millisecond-level timestamps (if we captured them) for better overlay matching.
        if isinstance(celebs_detailed, list) and celebs_detailed:
            for c in celebs_detailed:
                if not isinstance(c, dict):
                    continue
                nm = (c.get("name") or "").strip()
                if not nm:
                    continue
                ts_ms = celeb_ts_ms_by_name.get(nm) or []
                try:
                    ts_ms_norm = sorted(list(dict.fromkeys([int(x) for x in ts_ms if x is not None])))
                except Exception:
                    ts_ms_norm = []
                if ts_ms_norm:
                    c["timestamps_ms"] = ts_ms_norm[:240]

        moderation_stats: dict[str, dict[str, float]] = {}
        for item in moderation_raw:
            lbl = (item.get("ModerationLabel") or {})
            name = (lbl.get("Name") or "").strip()
            if not name:
                continue
            conf = float(lbl.get("Confidence") or 0.0)
            stat = moderation_stats.setdefault(name, {"count": 0.0, "max": 0.0})
            stat["count"] += 1.0
            stat["max"] = max(stat["max"], conf)
        moderation_detailed = [
            {"name": name, "occurrences": int(stats.get("count") or 0), "max_confidence": float(stats.get("max") or 0.0)}
            for name, stats in sorted(moderation_stats.items(), key=lambda kv: (kv[1].get("count") or 0.0, kv[1].get("max") or 0.0, kv[0]), reverse=True)[:50]
        ]

        text_stats: dict[str, dict[str, float]] = {}
        for item in text_raw:
            det = (item.get("TextDetection") or {})
            value = (det.get("DetectedText") or "").strip()
            if not value:
                continue
            conf = float(det.get("Confidence") or 0.0)
            stat = text_stats.setdefault(value, {"count": 0.0, "max": 0.0})
            stat["count"] += 1.0
            stat["max"] = max(stat["max"], conf)
        text_detailed = [
            {"text": txt, "occurrences": int(stats.get("count") or 0), "max_confidence": float(stats.get("max") or 0.0)}
            for txt, stats in sorted(text_stats.items(), key=lambda kv: (kv[1].get("count") or 0.0, kv[1].get("max") or 0.0, kv[0]), reverse=True)[:100]
        ]
        text_detected = [t.get("text") for t in text_detailed if t.get("text")][:50]

        transcribe_result: dict[str, Any] = {}
        transcript = ""
        transcript_words: list[dict[str, Any]] = []
        transcript_segments: list[dict[str, Any]] = []
        language_code = None
        transcript_s3 = None
        languages_detected: list[dict[str, Any]] = []

        if enable_transcribe:
            transcribe_result = _transcribe_s3_media_rich(
                media_bucket=analysis_bucket,
                media_key=analysis_key,
                job_id=job_id,
            )
            transcript = (transcribe_result.get("text") or "")
            transcript_words = transcribe_result.get("words") or []
            transcript_segments = transcribe_result.get("segments") or []
            language_code = transcribe_result.get("language_code")
            transcript_s3 = transcribe_result.get("transcript_s3")

            transcript = _normalize_transcript_basic(transcript)

            transcript = _apply_transcript_patches(transcript, language_code=language_code)
            transcript = _apply_language_spelling_fixes(transcript, language_code=language_code)

            if transcript_segments:
                for seg in transcript_segments:
                    if isinstance(seg, dict) and seg.get("text"):
                        seg["text"] = _normalize_transcript_basic(str(seg.get("text") or ""))
                        seg["text"] = _apply_transcript_patches(str(seg.get("text") or ""), language_code=language_code)
                        seg["text"] = _apply_language_spelling_fixes(str(seg.get("text") or ""), language_code=language_code)
            languages_detected = _detect_dominant_languages_from_text(transcript)
        else:
            _job_update(job_id, progress=78, message="Transcribe: disabled")

        # Subtitles artifacts (SRT/VTT) for UI download.
        subtitles: dict[str, Any] = {}
        if transcript_segments:
            try:
                _job_update(job_id, progress=79, message="Writing subtitles")
                srt_text = _segments_to_srt(transcript_segments)
                vtt_text = _segments_to_vtt(transcript_segments)
                subtitles = {"language_code": language_code}

                bucket_out = _metadata_artifacts_s3_bucket()
                srt_key = _subtitles_s3_key(job_id, lang="orig", fmt="srt")
                vtt_key = _subtitles_s3_key(job_id, lang="orig", fmt="vtt")
                _s3_put_bytes(
                    bucket=bucket_out,
                    key=srt_key,
                    body=srt_text.encode("utf-8"),
                    content_type="application/x-subrip",
                )
                _s3_put_bytes(
                    bucket=bucket_out,
                    key=vtt_key,
                    body=vtt_text.encode("utf-8"),
                    content_type="text/vtt",
                )

                s3_block: dict[str, Any] = {
                    "bucket": bucket_out,
                    "srt_key": srt_key,
                    "vtt_key": vtt_key,
                    "srt_uri": f"s3://{bucket_out}/{srt_key}",
                    "vtt_uri": f"s3://{bucket_out}/{vtt_key}",
                }

                # English-translated subtitles (best-effort)
                if enable_translate:
                    en_segments = _translate_segments_to_english(transcript_segments, source_language_code=language_code)
                    if en_segments:
                        srt_en_text = _segments_to_srt(en_segments)
                        vtt_en_text = _segments_to_vtt(en_segments)
                        srt_en_key = _subtitles_s3_key(job_id, lang="en", fmt="srt")
                        vtt_en_key = _subtitles_s3_key(job_id, lang="en", fmt="vtt")
                        _s3_put_bytes(
                            bucket=bucket_out,
                            key=srt_en_key,
                            body=srt_en_text.encode("utf-8"),
                            content_type="application/x-subrip",
                        )
                        _s3_put_bytes(
                            bucket=bucket_out,
                            key=vtt_en_key,
                            body=vtt_en_text.encode("utf-8"),
                            content_type="text/vtt",
                        )
                        subtitles["en"] = {"language_code": "en"}
                        s3_block["en"] = {
                            "srt_key": srt_en_key,
                            "vtt_key": vtt_en_key,
                            "srt_uri": f"s3://{bucket_out}/{srt_en_key}",
                            "vtt_uri": f"s3://{bucket_out}/{vtt_en_key}",
                        }

                subtitles["s3"] = s3_block
            except Exception:
                subtitles = {}

        locations_text = _comprehend_locations_from_text(transcript, language_code=language_code)
        famous_locations = _build_famous_locations_payload(
            text_locations=locations_text if isinstance(locations_text, list) else [],
            transcript_segments=transcript_segments if isinstance(transcript_segments, list) else [],
            frames=frames_metadata if isinstance(frames_metadata, list) else [],
            geocode_cache=None,
        )
        locations = famous_locations.get("locations") if isinstance(famous_locations, dict) else (locations_text or [])
        synopses_by_age = _bedrock_synopsis_by_age_group(transcript or video_description or "", title=video_title)

        scenes_pack = _build_scene_by_scene_metadata(
            frames=frames_metadata,
            transcript_segments=transcript_segments,
            duration_seconds=duration_seconds,
            window_seconds=_parse_int_param(os.getenv("ENVID_METADATA_SCENE_WINDOW_SECONDS"), default=15, min_value=5, max_value=120),
        )

        translated_en = ""
        if enable_translate and transcript:
            translated_en = _translate_to_english(transcript, source_language_code=language_code)

        comprehend_insights: dict[str, Any] = {"entities": [], "key_phrases": [], "sentiment": None}
        if enable_comprehend and (translated_en or transcript):
            comprehend_insights = _comprehend_insights_from_text(translated_en or transcript)

        summary_text = ""
        summary_ssml = None
        if enable_summary and (translated_en or transcript):
            _job_update(job_id, progress=90, message="Bedrock: generating summary")
            summary_payload = _bedrock_summary(translated_en or transcript, title=video_title)
            summary_text = (summary_payload.get("summary") or "").strip()
            summary_ssml = summary_payload.get("ssml")
        else:
            _job_update(job_id, progress=90, message="Bedrock: summary disabled")

        polly_audio = None
        if enable_polly and summary_ssml:
            _job_update(job_id, progress=92, message="Polly: synthesizing summary")
            polly_audio = _polly_synthesize_to_s3(video_id=job_id, ssml=summary_ssml or "")
        else:
            _job_update(job_id, progress=92, message="Polly: disabled")

        metadata_text = ""
        metadata_parts = [video_title, video_description]
        if summary_text:
            metadata_parts.append(f"Summary: {summary_text}")
        if labels_detailed:
            metadata_parts.append("Visual elements: " + ", ".join([l["name"] for l in labels_detailed[:50] if l.get("name")]))
        if celebs_detailed:
            metadata_parts.append("Celebrities detected: " + ", ".join([c["name"] for c in celebs_detailed[:25] if c.get("name")]))
        if text_detected:
            metadata_parts.append("On-screen text: " + ", ".join(text_detected[:25]))
        if translated_en:
            metadata_parts.append("Transcript (EN): " + translated_en[:2000])
        metadata_text = " ".join([p for p in metadata_parts if (p or "").strip()])
        embedding: list[float] = []
        if enable_embedding:
            _job_update(job_id, progress=94, message="Generating embedding")
            embedding = _generate_embedding(metadata_text) if metadata_text else _generate_embedding(video_title)
        else:
            _job_update(job_id, progress=94, message="Embedding: disabled")

        _job_update(job_id, progress=97, message="Saving & indexing")
        video_entry = {
            "id": job_id,
            "title": video_title,
            "description": video_description,
            "original_filename": Path(key).name,
            "stored_filename": None,
            "file_path": "",
            "s3_video_key": key,
            "s3_video_uri": s3_uri,
            "s3_proxy_key": analysis_key if (analysis_bucket, analysis_key) != (bucket, key) else None,
            "s3_proxy_uri": s3_proxy_uri,
            "transcript": transcript,
            "transcript_translation_en": translated_en or "",
            "language_code": language_code,
            "languages_detected": languages_detected,
            "transcript_words": transcript_words,
            "transcript_segments": transcript_segments,
            "transcribe_transcript_s3": transcript_s3,
            "labels": [l["name"] for l in labels_detailed if l.get("name")][:50],
            "labels_detailed": labels_detailed,
            "text_detected": text_detected,
            "text_detailed": text_detailed,
            "emotions": [],
            "celebrities": [c["name"] for c in celebs_detailed if c.get("name")][:50],
            "celebrities_detailed": celebs_detailed,
            "moderation_labels": [m.get("name") for m in moderation_detailed if m.get("name")][:50],
            "moderation_detailed": moderation_detailed,
            "faces_summary": {},
            "comprehend_entities": comprehend_insights.get("entities") or [],
            "comprehend_key_phrases": comprehend_insights.get("key_phrases") or [],
            "comprehend_sentiment": comprehend_insights.get("sentiment"),
            "bedrock_summary": summary_text,
            "polly_summary_audio": polly_audio,
            "embedding": embedding,
            "thumbnail": None,
            "metadata_text": metadata_text,
            "frame_count": int(len(frames_metadata) or 0),
            "frames_analyzed": int(len(frames_metadata) or 0),
            "frame_interval_seconds": interval,
            "duration_seconds": duration_seconds,
            "technical_ffprobe": technical_ffprobe or None,
            "technical_mediainfo": technical_mediainfo or None,
            "technical_verification": technical_verification or None,
            "technical_rekognition_video_metadata": rekognition_video_metadata,
            "file_size_bytes": file_size_bytes,
            "frames": frames_metadata,
            "uploaded_at": datetime.utcnow().isoformat(),
            "cloud_only": True,
            "output_profile": profile,

            # Required-output extras
            "subtitles": subtitles,
            "locations": locations,
            "famous_locations": famous_locations,
            "locations_geocoded": (famous_locations.get("geocode_cache") if isinstance(famous_locations, dict) else None),
            "synopses_by_age_group": synopses_by_age,
            "scene_metadata": scenes_pack,
        }

        # Provide an overall video thumbnail (middle frame) for history cards (best-effort).
        try:
            mid_ts = int(float(duration_seconds or 0.0) / 2.0) if duration_seconds else 0
            temp_dir2 = Path(tempfile.mkdtemp(prefix=f"envid_metadata_video_thumb_{job_id}_"))
            try:
                thumb_input_source2 = _presign_s3_get_object_url(bucket=analysis_bucket, key=analysis_key)
                out_path = temp_dir2 / "video_thumb.jpg"
                cmd = [
                    FFMPEG_PATH,
                    "-ss",
                    str(max(0, int(mid_ts))),
                    "-i",
                    str(thumb_input_source2),
                    "-vframes",
                    "1",
                    "-q:v",
                    "2",
                    str(out_path),
                    "-y",
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if res.returncode == 0 and out_path.exists():
                    video_entry["thumbnail"] = base64.b64encode(out_path.read_bytes()).decode("utf-8")
            finally:
                shutil.rmtree(temp_dir2, ignore_errors=True)
        except Exception:
            pass

        categorized = _build_categorized_metadata_json(video_entry)
        video_entry["metadata_categories"] = categorized.get("categories")
        video_entry["metadata_combined"] = categorized.get("combined")
        _ensure_metadata_artifacts_on_s3(video_entry, save_index=False)

        existing_idx = next((i for i, v in enumerate(VIDEO_INDEX) if str(v.get("id")) == str(video_entry.get("id"))), None)
        if existing_idx is not None:
            VIDEO_INDEX[existing_idx] = video_entry
        else:
            VIDEO_INDEX.append(video_entry)
        _save_video_index()

        result = {
            "id": job_id,
            "title": video_title,
            "message": "S3 video indexed successfully (cloud-only)",
            "labels_count": len(video_entry.get("labels") or []),
            "frame_count": 0,
        }
        _job_update(job_id, status="completed", progress=100, message="Completed", result=result)
    except Exception as exc:
        _job_update(job_id, status="failed", progress=100, message="Failed", error=str(exc))


def _process_s3_video_job_cloud_only(
    *,
    job_id: str,
    s3_key: str,
    video_title: str,
    video_description: str,
    frame_interval_seconds: int,
) -> None:
    """Backward-compatible wrapper for rawvideo cloud-only processing."""
    normalized_key = _normalize_rawvideo_key(s3_key)
    _process_s3_object_job_cloud_only(
        job_id=job_id,
        s3_bucket=_rawvideo_bucket(),
        s3_key=normalized_key,
        video_title=video_title,
        video_description=video_description,
        frame_interval_seconds=frame_interval_seconds,
    )



def _upload_video_to_s3_or_raise(
    *,
    video_id: str,
    local_path: Path,
    original_filename: str,
    job_id: str | None = None,
) -> tuple[str, str]:
    bucket = _video_s3_bucket()
    key = _video_s3_key(video_id, original_filename)

    bucket_region = _detect_bucket_region(bucket)
    s3_region = bucket_region or DEFAULT_AWS_REGION

    use_accelerate = _env_truthy(os.getenv("ENVID_METADATA_S3_ACCELERATE"), default=False)
    s3_upload_client = (
        boto3.client(
            "s3",
            region_name=s3_region,
            config=BotoConfig(s3={"use_accelerate_endpoint": True}) if use_accelerate else None,
        )
        if use_accelerate
        else boto3.client("s3", region_name=s3_region)
    )

    # `upload_file` uses multipart automatically above the threshold.
    # These env vars let us force/tune multipart and concurrency for faster uploads.
    multipart_threshold_mb = _parse_int_param(
        os.getenv("ENVID_METADATA_S3_MULTIPART_THRESHOLD_MB"),
        default=8,
        min_value=1,
        max_value=2048,
    )
    multipart_chunk_mb = _parse_int_param(
        os.getenv("ENVID_METADATA_S3_MULTIPART_CHUNK_MB"),
        default=16,
        min_value=5,
        max_value=512,
    )
    max_concurrency = _parse_int_param(
        os.getenv("ENVID_METADATA_S3_MAX_CONCURRENCY"),
        default=8,
        min_value=1,
        max_value=32,
    )

    threshold_bytes = multipart_threshold_mb * 1024 * 1024
    chunk_bytes = multipart_chunk_mb * 1024 * 1024
    transfer_config = TransferConfig(
        multipart_threshold=multipart_threshold_mb * 1024 * 1024,
        multipart_chunksize=multipart_chunk_mb * 1024 * 1024,
        max_concurrency=max_concurrency,
        use_threads=True,
    )

    extra_args: dict[str, str] = {}
    content_type = mimetypes.guess_type(original_filename)[0]
    if content_type:
        extra_args["ContentType"] = content_type

    total_bytes: int | None = None
    try:
        total_bytes = int(local_path.stat().st_size)
    except Exception:
        total_bytes = None

    multipart_expected = bool(total_bytes and total_bytes >= threshold_bytes)

    uploaded_bytes = 0
    last_emit_t = 0.0
    last_emit_pct = -1
    start_t = time.monotonic()

    # Best-effort: log the first S3 request URL so we can confirm accelerate host/multipart calls.
    _logged_request_url = False

    def _log_first_request_url(**kwargs: Any) -> None:
        nonlocal _logged_request_url
        if _logged_request_url:
            return
        req = kwargs.get("request")
        url = getattr(req, "url", None)
        if not url:
            return
        _logged_request_url = True
        print(f"[envid-metadata] S3 first request URL: {url}", flush=True)
        try:
            app.logger.warning("S3 first request URL: %s", url)
        except Exception:
            pass

    try:
        s3_upload_client.meta.events.register("before-send.s3", _log_first_request_url)
        s3_upload_client.meta.events.register("request-created.s3", _log_first_request_url)
    except Exception:
        pass

    def _progress_callback(bytes_amount: int) -> None:
        nonlocal uploaded_bytes, last_emit_t, last_emit_pct
        uploaded_bytes += int(bytes_amount or 0)
        if not job_id:
            return

        now = time.monotonic()
        elapsed = max(0.001, now - start_t)
        mbps = (uploaded_bytes / (1024 * 1024)) / elapsed
        if total_bytes and total_bytes > 0:
            pct = int(min(100.0, (uploaded_bytes / float(total_bytes)) * 100.0))
            if pct == last_emit_pct and (now - last_emit_t) < 1.0:
                return
            if (now - last_emit_t) < 0.5 and pct < 100:
                return
            last_emit_t = now
            last_emit_pct = pct
            _job_update(
                job_id,
                progress=2,
                message=f"Uploading video to S3 ({pct}%, {mbps:.1f} MB/s)",
                s3_upload_bytes=uploaded_bytes,
                s3_upload_total_bytes=total_bytes,
                s3_upload_percent=pct,
                s3_upload_mbps=mbps,
            )
            _job_step_update(
                job_id,
                "upload_to_s3",
                status="running" if pct < 100 else "completed",
                percent=pct,
                message="Uploading" if pct < 100 else "Uploaded",
            )
        else:
            if (now - last_emit_t) < 1.0:
                return
            last_emit_t = now
            _job_update(
                job_id,
                progress=2,
                message=f"Uploading video to S3 ({uploaded_bytes} bytes, {mbps:.1f} MB/s)",
                s3_upload_bytes=uploaded_bytes,
                s3_upload_total_bytes=None,
                s3_upload_percent=None,
                s3_upload_mbps=mbps,
            )
            _job_step_update(job_id, "upload_to_s3", status="running", message="Uploading")

    upload_diag = (
        f"[envid-metadata] Uploading video to S3: local={local_path} bucket={bucket} key={key} "
        f"(region={s3_region} accelerate={use_accelerate} threshold={multipart_threshold_mb}MB "
        f"chunk={multipart_chunk_mb}MB concurrency={max_concurrency} multipart_expected={multipart_expected} "
        f"size_bytes={total_bytes} threshold_bytes={threshold_bytes} chunk_bytes={chunk_bytes})"
    )
    print(upload_diag, flush=True)
    try:
        app.logger.warning(upload_diag)
    except Exception:
        pass
    s3_upload_client.upload_file(
        str(local_path),
        bucket,
        key,
        ExtraArgs=extra_args or None,
        Config=transfer_config,
        Callback=_progress_callback if job_id else None,
    )
    print(f"[envid-metadata] Uploaded video to S3: s3://{bucket}/{key}", flush=True)
    try:
        app.logger.warning("Uploaded video to S3: s3://%s/%s", bucket, key)
    except Exception:
        pass
    return key, f"s3://{bucket}/{key}"

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
MEDIAINFO_PATH = (os.getenv("MEDIAINFO_PATH") or "mediainfo").strip() or "mediainfo"


def _mediainfo_s3_partial_enabled() -> bool:
    return _env_truthy(os.getenv("ENVID_METADATA_MEDIAINFO_S3_PARTIAL"), default=False)


def _mediainfo_s3_partial_initial_bytes() -> int:
    return _parse_int_param(
        os.getenv("ENVID_METADATA_MEDIAINFO_S3_PARTIAL_INITIAL_BYTES"),
        default=16 * 1024 * 1024,
        min_value=1024 * 1024,
        max_value=1024 * 1024 * 1024,
    )


def _mediainfo_s3_partial_max_bytes() -> int:
    return _parse_int_param(
        os.getenv("ENVID_METADATA_MEDIAINFO_S3_PARTIAL_MAX_BYTES"),
        default=128 * 1024 * 1024,
        min_value=1024 * 1024,
        max_value=5 * 1024 * 1024 * 1024,
    )


def _mediainfo_s3_partial_tail_bytes() -> int:
    return _parse_int_param(
        os.getenv("ENVID_METADATA_MEDIAINFO_S3_PARTIAL_TAIL_BYTES"),
        default=0,
        min_value=0,
        max_value=5 * 1024 * 1024 * 1024,
    )


def _probe_mediainfo_s3_partial(*, bucket: str, key: str) -> dict[str, Any]:
    """Best-effort mediainfo for S3 objects without downloading the full file.

    Uses S3 Range GET to download only the first N bytes (and may increase up to a cap).
    Note: some formats (e.g., MP4 with `moov` atom at the end) may not yield complete metadata
    from the header bytes alone.
    """

    if not shutil.which(MEDIAINFO_PATH):
        return {"available": False, "error": f"mediainfo not found on PATH (MEDIAINFO_PATH={MEDIAINFO_PATH})"}

    b = (bucket or "").strip()
    k = (key or "").strip().lstrip("/")
    if not b or not k:
        return {"available": False, "error": "Missing bucket/key"}

    try:
        bucket_region = _detect_bucket_region(b)
        s3 = boto3.client("s3", region_name=(bucket_region or DEFAULT_AWS_REGION))
        head = s3.head_object(Bucket=b, Key=k)
        total = int(head.get("ContentLength") or 0)
        if total <= 0:
            return {"available": False, "error": "S3 object size unknown/zero"}

        initial = min(_mediainfo_s3_partial_initial_bytes(), total)
        max_bytes = min(_mediainfo_s3_partial_max_bytes(), total)
        attempt = initial

        suffix = Path(k).suffix or ".bin"
        last_err: str | None = None
        best: dict[str, Any] | None = None

        while attempt <= max_bytes:
            with tempfile.TemporaryDirectory(prefix="envid_mediainfo_s3_") as td:
                local_path = Path(td) / f"partial{suffix}"
                rng = f"bytes=0-{attempt - 1}"
                try:
                    obj = s3.get_object(Bucket=b, Key=k, Range=rng)
                    body = obj.get("Body")
                    data = body.read() if body else b""
                    if not data:
                        return {"available": False, "error": "Empty S3 range response"}
                    local_path.write_bytes(data)
                except Exception as exc:
                    last_err = f"S3 Range GET failed: {exc}"
                    break

                mi = _probe_mediainfo_metadata(local_path)
                # Always mark as partial.
                if isinstance(mi, dict):
                    mi["partial"] = True
                    mi["partial_bytes_downloaded"] = int(attempt)
                    mi["s3_bucket"] = b
                    mi["s3_key"] = k

                # If mediainfo itself is unavailable/error, keep trying is pointless.
                if not isinstance(mi, dict) or mi.get("available") is False:
                    best = mi if isinstance(mi, dict) else None
                    last_err = (mi.get("error") if isinstance(mi, dict) else None) or "mediainfo failed"
                    break

                best = mi

                # Heuristic: consider metadata "good enough" if we got at least duration or codec+resolution.
                has_duration = mi.get("duration_seconds") is not None
                res = mi.get("resolution")
                has_res = isinstance(res, dict) and (res.get("width") or res.get("height"))
                has_vcodec = bool((mi.get("video_codec") or "").strip())

                if has_duration or (has_res and has_vcodec):
                    return mi

            # Increase: double up to cap.
            attempt = min(max_bytes, attempt * 2)
            if attempt == max_bytes:
                # Ensure loop runs one last time at max_bytes.
                if best is not None:
                    break

        # Optional: tail probe (helps when MP4/MOV moov atom is at the end).
        tail_bytes = _mediainfo_s3_partial_tail_bytes()
        if tail_bytes and tail_bytes > 0 and total > 0:
            try:
                tail_start = max(0, total - tail_bytes)
                tail_count = total - tail_start
                with tempfile.TemporaryDirectory(prefix="envid_mediainfo_s3_tail_") as td:
                    tail_path = Path(td) / f"tail{suffix}"
                    rng = f"bytes={tail_start}-{total - 1}"
                    obj = s3.get_object(Bucket=b, Key=k, Range=rng)
                    body = obj.get("Body")
                    data = body.read() if body else b""
                    if data:
                        tail_path.write_bytes(data)
                        tail_mi = _probe_mediainfo_metadata(tail_path)
                        if isinstance(tail_mi, dict):
                            tail_mi["partial"] = True
                            tail_mi["partial_tail"] = True
                            tail_mi["partial_bytes_downloaded"] = int(tail_count)
                            tail_mi["partial_tail_start"] = int(tail_start)

                        # Merge: prefer head, fill missing from tail.
                        if isinstance(best, dict) and isinstance(tail_mi, dict) and best.get("available") is not False and tail_mi.get("available") is not False:
                            merged = dict(best)
                            for field in [
                                "container_format",
                                "duration_seconds",
                                "bitrate_bps",
                                "video_codec",
                                "audio_codec",
                                "resolution",
                                "frame_rate",
                                "aspect_ratio",
                                "audio_channels",
                                "audio_sampling_rate_hz",
                                "audio_language",
                            ]:
                                if merged.get(field) in (None, "", {}):
                                    if tail_mi.get(field) not in (None, "", {}):
                                        merged[field] = tail_mi.get(field)
                            merged["tail_probe"] = {
                                "available": tail_mi.get("available"),
                                "partial_tail_start": tail_mi.get("partial_tail_start"),
                                "partial_bytes_downloaded": tail_mi.get("partial_bytes_downloaded"),
                                "duration_seconds": tail_mi.get("duration_seconds"),
                            }
                            best = merged
                        elif best is None and isinstance(tail_mi, dict):
                            best = tail_mi
            except Exception as exc:
                last_err = last_err or f"Tail probe failed: {exc}"

        if best is not None:
            # Return what we could extract, but mark it as potentially incomplete.
            best.setdefault("warning", "Partial S3 probe may be incomplete; consider full download for exact container metadata")
            return best
        return {"available": False, "error": last_err or "Partial S3 mediainfo probe failed"}
    except Exception as exc:
        return {"available": False, "error": f"Partial S3 mediainfo probe exception: {exc}"}


def _mediainfo_lambda_name_or_arn() -> str | None:
    raw = (os.getenv("ENVID_METADATA_MEDIAINFO_LAMBDA") or os.getenv("MEDIAINFO_LAMBDA") or "").strip()
    return raw or None


def _mediainfo_lambda_region() -> str:
    return (os.getenv("ENVID_METADATA_MEDIAINFO_LAMBDA_REGION") or "us-east-1").strip() or "us-east-1"


def _invoke_mediainfo_lambda(*, s3_bucket: str, s3_key: str) -> dict[str, Any]:
    """Invoke the MediaInfo Lambda to extract technical metadata (OS-independent).

    Expected lambda response shape:
      {"ok": true, "mediainfo": {...}}
    """
    name_or_arn = _mediainfo_lambda_name_or_arn()
    if not name_or_arn:
        return {"available": False, "error": "MediaInfo Lambda not configured (ENVID_METADATA_MEDIAINFO_LAMBDA)"}

    bucket = (s3_bucket or "").strip()
    key = (s3_key or "").strip().lstrip("/")
    if not bucket or not key:
        return {"available": False, "error": "Missing s3_bucket/s3_key for MediaInfo Lambda"}

    try:
        lam = boto3.client("lambda", region_name=_mediainfo_lambda_region())
        resp = lam.invoke(
            FunctionName=name_or_arn,
            InvocationType="RequestResponse",
            Payload=json.dumps({"s3_bucket": bucket, "s3_key": key}).encode("utf-8"),
        )

        payload_bytes = resp.get("Payload").read() if resp.get("Payload") else b""
        raw = payload_bytes.decode("utf-8", errors="replace") if payload_bytes else ""
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            return {"available": False, "error": "Invalid MediaInfo Lambda response"}
        if not data.get("ok"):
            return {"available": False, "error": data.get("error") or "MediaInfo Lambda error"}

        mi = data.get("mediainfo")
        if isinstance(mi, dict):
            # Normalize to our stored shape.
            mi.setdefault("available", True)
            return mi
        return {"available": False, "error": "MediaInfo Lambda missing 'mediainfo'"}
    except Exception as exc:
        return {"available": False, "error": f"MediaInfo Lambda invoke failed: {exc}"}

# Local storage directories
STORAGE_BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = STORAGE_BASE_DIR / "documents"
VIDEOS_DIR = STORAGE_BASE_DIR / "videos"
INDICES_DIR = STORAGE_BASE_DIR / "indices"

# Ensure directories exist
DOCUMENTS_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)
INDICES_DIR.mkdir(exist_ok=True)

LOCAL_FACE_GALLERY_FILE = INDICES_DIR / "local_face_gallery.json"

SUBTITLES_DIR = None


def _as_data_uri_jpeg(base64_value: str | None) -> str | None:
    if not base64_value:
        return None
    if base64_value.startswith("data:"):
        return base64_value
    return f"data:image/jpeg;base64,{base64_value}"


def _relative_video_path(stored_filename: str) -> str:
    """Return the relative storage path for a video."""
    return str(Path("videos") / stored_filename)


def _relative_document_path(stored_filename: str) -> str:
    """Return the relative storage path for a document."""
    return str(Path("documents") / stored_filename)


def _absolute_video_path(stored_filename: str) -> Path:
    """Return the absolute storage path for a video."""
    return VIDEOS_DIR / stored_filename


def _absolute_document_path(stored_filename: str) -> Path:
    """Return the absolute storage path for a document."""
    return DOCUMENTS_DIR / stored_filename


# Index file paths
VIDEO_INDEX_FILE = INDICES_DIR / "video_index.json"
DOCUMENT_INDEX_FILE = INDICES_DIR / "document_index.json"

VIDEO_INDEX = []
DOCUMENT_INDEX = []


def _save_video_index():
    """Save video index to JSON file."""
    try:
        serialized_index = []
        for entry in VIDEO_INDEX:
            entry_copy = entry.copy()
            stored_filename = entry_copy.get("stored_filename")
            if stored_filename:
                entry_copy["file_path"] = _relative_video_path(stored_filename)
            serialized_index.append(entry_copy)

        with open(VIDEO_INDEX_FILE, 'w') as f:
            json.dump(serialized_index, f, indent=2, default=str)
    except Exception as e:
        print(f"Error saving video index: {e}")


def _load_video_index():
    """Load video index from JSON file."""
    global VIDEO_INDEX
    if VIDEO_INDEX_FILE.exists():
        try:
            with open(VIDEO_INDEX_FILE, 'r') as f:
                raw_index = json.load(f)

            VIDEO_INDEX = []
            for entry in raw_index:
                entry_copy = entry.copy()
                stored_filename = entry_copy.get("stored_filename")
                if not stored_filename:
                    file_path_value = entry_copy.get("file_path", "")
                    stored_filename = Path(file_path_value).name if file_path_value else ""
                if stored_filename:
                    entry_copy["stored_filename"] = stored_filename
                    entry_copy["file_path"] = _relative_video_path(stored_filename)
                VIDEO_INDEX.append(entry_copy)
            print(f"Loaded {len(VIDEO_INDEX)} videos from index")
        except Exception as e:
            print(f"Error loading video index: {e}")
            VIDEO_INDEX = []


def _save_document_index():
    """Save document index to JSON file."""
    try:
        serialized_index = []
        for entry in DOCUMENT_INDEX:
            entry_copy = entry.copy()
            stored_filename = entry_copy.get("stored_filename")
            if stored_filename:
                entry_copy["file_path"] = _relative_document_path(stored_filename)
            serialized_index.append(entry_copy)

        with open(DOCUMENT_INDEX_FILE, 'w') as f:
            json.dump(serialized_index, f, indent=2, default=str)
    except Exception as e:
        print(f"Error saving document index: {e}")


def _load_document_index():
    """Load document index from JSON file."""
    global DOCUMENT_INDEX
    if DOCUMENT_INDEX_FILE.exists():
        try:
            with open(DOCUMENT_INDEX_FILE, 'r') as f:
                raw_index = json.load(f)

            DOCUMENT_INDEX = []
            for entry in raw_index:
                entry_copy = entry.copy()
                stored_filename = entry_copy.get("stored_filename")
                if not stored_filename:
                    file_path_value = entry_copy.get("file_path", "")
                    stored_filename = Path(file_path_value).name if file_path_value else ""
                if stored_filename:
                    entry_copy["stored_filename"] = stored_filename
                    entry_copy["file_path"] = _relative_document_path(stored_filename)
                DOCUMENT_INDEX.append(entry_copy)
            print(f"Loaded {len(DOCUMENT_INDEX)} documents from index")
        except Exception as e:
            print(f"Error loading document index: {e}")
            DOCUMENT_INDEX = []


# Load indices on startup
_load_video_index()
_load_document_index()


def _admin_allowed() -> bool:
    """Guardrails for admin-only endpoints used in local demos.

    Enable by setting ENVID_METADATA_ALLOW_ADMIN=true.
    Optionally set ENVID_METADATA_ADMIN_TOKEN and send it as header X-Admin-Token.
    """

    if not _env_truthy(os.getenv("ENVID_METADATA_ALLOW_ADMIN"), default=False):
        return False
    expected = (os.getenv("ENVID_METADATA_ADMIN_TOKEN") or "").strip()
    if not expected:
        return True
    try:
        provided = (request.headers.get("X-Admin-Token") or "").strip()
    except Exception:
        provided = ""
    return bool(provided) and provided == expected


def _s3_get_json_best_effort(*, bucket: str, key: str) -> dict[str, Any] | None:
    b = (bucket or "").strip()
    k = (key or "").strip().lstrip("/")
    if not b or not k:
        return None
    try:
        bucket_region = _detect_bucket_region(b)
        s3_region = bucket_region or DEFAULT_AWS_REGION
        client = _s3_client_for_transfer(region_name=s3_region)
        resp = client.get_object(Bucket=b, Key=k)
        body = resp.get("Body")
        raw = body.read() if body else b""
        txt = raw.decode("utf-8", errors="replace") if raw else ""
        data = json.loads(txt) if txt else None
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _extract_video_id_from_combined_key(*, prefix: str, key: str) -> str | None:
    """Extract <video_id> from <prefix>/<video_id>/combined.json."""
    p = (prefix or "").strip().strip("/")
    k = (key or "").strip().lstrip("/")
    if not p or not k:
        return None
    expected_suffix = "/combined.json"
    if not k.endswith(expected_suffix):
        return None
    if not k.startswith(p + "/"):
        return None
    rest = k[len(p) + 1 :]
    if not rest.endswith(expected_suffix):
        return None
    vid = rest[: -len(expected_suffix)].strip().strip("/")
    return vid or None


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({
        "status": "ok",
        "service": "envid-metadata",
        "region": DEFAULT_AWS_REGION
    })


@app.route("/videos/rebuild-index", methods=["POST"])
def rebuild_video_index_from_s3() -> Any:
    """Rebuild VIDEO_INDEX from S3 metadata artifacts.

    This is intended for local demos where the on-disk index file is wiped.
    It scans s3://<artifacts_bucket>/<artifacts_prefix>/*/combined.json and
    creates minimal index entries so the UI history/catalogue comes back.
    """

    if not _admin_allowed():
        return jsonify({"error": "Admin endpoint disabled"}), 403

    try:
        bucket = _metadata_artifacts_s3_bucket()
        prefix = _metadata_artifacts_s3_prefix().strip().strip("/")
        scan_prefix = prefix + "/"
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    max_keys_raw = (request.args.get("max_keys") or "1000").strip()
    try:
        max_keys = int(max_keys_raw)
    except Exception:
        max_keys = 1000
    max_keys = max(1, min(10000, max_keys))

    bucket_region = _detect_bucket_region(bucket)
    s3_region = bucket_region or DEFAULT_AWS_REGION
    client = _s3_client_for_transfer(region_name=s3_region)

    found_ids: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": scan_prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        contents = resp.get("Contents") or []
        for obj in contents:
            k = (obj.get("Key") or "").strip()
            vid = _extract_video_id_from_combined_key(prefix=prefix, key=k)
            if not vid:
                continue
            found_ids.append(vid)
            if len(found_ids) >= max_keys:
                break
        if len(found_ids) >= max_keys:
            break
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
        if not token:
            break

    # De-dupe while keeping stable ordering.
    seen: set[str] = set()
    unique_ids: list[str] = []
    for vid in found_ids:
        if vid in seen:
            continue
        seen.add(vid)
        unique_ids.append(vid)

    rebuilt: list[dict[str, Any]] = []
    for vid in unique_ids:
        combined_key = _metadata_combined_s3_key(vid)
        combined = _s3_get_json_best_effort(bucket=bucket, key=combined_key) or {}

        # Best-effort extraction of title/description from various possible schemas.
        title = (
            (combined.get("title") or "")
            or ((combined.get("video") or {}).get("title") if isinstance(combined.get("video"), dict) else "")
            or ((combined.get("metadata") or {}).get("title") if isinstance(combined.get("metadata"), dict) else "")
        )
        description = (
            (combined.get("description") or "")
            or ((combined.get("video") or {}).get("description") if isinstance(combined.get("video"), dict) else "")
            or ((combined.get("metadata") or {}).get("description") if isinstance(combined.get("metadata"), dict) else "")
        )

        if isinstance(title, str):
            title = title.strip()
        else:
            title = ""
        if isinstance(description, str):
            description = description.strip()
        else:
            description = ""

        rebuilt.append(
            {
                "id": str(vid),
                "title": title or str(vid),
                "description": description or "",
                "labels": [],
                "text_detected": [],
                "emotions": [],
                "celebrities": [],
                "moderation_labels": [],
                "transcript": "",
                "thumbnail": None,
                "uploaded_at": None,
                "stored_filename": "",
                "file_path": "",
                "metadata_s3": {
                    "bucket": bucket,
                    "prefix": prefix,
                    "combined": combined_key,
                    "categories": {},
                    "zip": _metadata_zip_s3_key(vid),
                },
            }
        )

    global VIDEO_INDEX
    VIDEO_INDEX = rebuilt
    _save_video_index()

    return jsonify(
        {
            "ok": True,
            "rebuilt": len(rebuilt),
            "bucket": bucket,
            "prefix": prefix,
        }
    ), 200


@app.route("/rekognition-check", methods=["GET"])
def rekognition_check() -> Any:
    """Validate AWS Rekognition connectivity and permissions.

    This does not prove celebrity detection will happen for a given video; it only verifies
    that the AWS call to RecognizeCelebrities can be executed successfully.
    """

    def _call(fn_name: str, fn) -> Dict[str, Any]:
        try:
            resp = fn()
            return {
                "ok": True,
                "fn": fn_name,
                "status": "success",
                "summary": {
                    "keys": list(resp.keys()) if isinstance(resp, dict) else None,
                },
            }
        except (BotoCoreError, ClientError) as exc:
            return {
                "ok": False,
                "fn": fn_name,
                "status": "error",
                "error": str(exc),
            }
        except Exception as exc:  # pragma: no cover
            return {
                "ok": False,
                "fn": fn_name,
                "status": "error",
                "error": str(exc),
            }

    # Generate a small, valid JPEG in-memory so Rekognition accepts Image bytes.
    img = Image.new("RGB", (64, 64), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    image_bytes = buf.getvalue()

    checks: List[Dict[str, Any]] = []
    checks.append(
        _call(
            "recognize_celebrities",
            lambda: rekognition.recognize_celebrities(Image={"Bytes": image_bytes}),
        )
    )

    # Optional: also validate a basic Rekognition call, useful for diagnosing IAM.
    checks.append(
        _call(
            "detect_labels",
            lambda: rekognition.detect_labels(Image={"Bytes": image_bytes}, MaxLabels=5),
        )
    )

    ok = all(c.get("ok") for c in checks)
    return jsonify(
        {
            "ok": ok,
            "service": "rekognition",
            "region": DEFAULT_AWS_REGION,
            "checks": checks,
            "note": "This validates AWS calls; celebrity detection still depends on the video content.",
        }
    ), (200 if ok else 500)


@app.route("/face-collection/create", methods=["POST"])
def face_collection_create() -> Any:
    payload = request.get_json(silent=True) or {}
    collection_id = _rekognition_collection_id_normalize(payload.get("collection_id"))
    if not collection_id:
        return jsonify({"error": "collection_id is required"}), 400

    try:
        resp = rekognition.create_collection(CollectionId=collection_id)
        return jsonify({"ok": True, "collection_id": collection_id, "status": "created", "response": resp}), 200
    except rekognition.exceptions.ResourceAlreadyExistsException:
        return jsonify({"ok": True, "collection_id": collection_id, "status": "exists"}), 200
    except (BotoCoreError, ClientError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/face-collection/<collection_id>/faces", methods=["GET"])
def face_collection_list_faces(collection_id: str) -> Any:
    collection_id = _rekognition_collection_id_normalize(collection_id)
    if not collection_id:
        return jsonify({"error": "collection_id is required"}), 400

    faces: List[Dict[str, Any]] = []
    next_token = None
    try:
        while True:
            kwargs: Dict[str, Any] = {"CollectionId": collection_id, "MaxResults": 100}
            if next_token:
                kwargs["NextToken"] = next_token
            resp = rekognition.list_faces(**kwargs)
            for f in resp.get("Faces", []) or []:
                faces.append(
                    {
                        "face_id": f.get("FaceId"),
                        "image_id": f.get("ImageId"),
                        "external_image_id": f.get("ExternalImageId"),
                        "confidence": f.get("Confidence"),
                    }
                )
            next_token = resp.get("NextToken")
            if not next_token:
                break
        return jsonify({"ok": True, "collection_id": collection_id, "faces": faces, "total": len(faces)}), 200
    except (BotoCoreError, ClientError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/face-collection/enroll", methods=["POST"])
def face_collection_enroll() -> Any:
    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400
    collection_id = _rekognition_collection_id_normalize(request.form.get("collection_id"))
    actor_name = _rekognition_external_image_id_normalize(request.form.get("actor_name"))
    if not collection_id:
        return jsonify({"error": "collection_id is required"}), 400
    if not actor_name:
        return jsonify({"error": "actor_name is required"}), 400

    image_bytes = request.files["image"].read()
    if not image_bytes:
        return jsonify({"error": "empty image"}), 400

    try:
        resp = rekognition.index_faces(
            CollectionId=collection_id,
            Image={"Bytes": image_bytes},
            ExternalImageId=actor_name,
            MaxFaces=1,
            QualityFilter="AUTO",
            DetectionAttributes=[],
        )
        face_records = resp.get("FaceRecords") or []
        return jsonify(
            {
                "ok": True,
                "collection_id": collection_id,
                "actor_name": actor_name,
                "faces_indexed": len(face_records),
                "face_records": face_records,
                "unindexed_faces": resp.get("UnindexedFaces") or [],
            }
        ), 200
    except rekognition.exceptions.ResourceNotFoundException:
        return jsonify({"ok": False, "error": f"collection not found: {collection_id}"}), 404
    except (BotoCoreError, ClientError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/local-faces", methods=["GET"])
def local_faces_list() -> Any:
    ok, reason = _insightface_available()
    gallery = _local_gallery_load()
    actors = gallery.get("actors") if isinstance(gallery, dict) else {}
    if not isinstance(actors, dict):
        actors = {}
    summary = [{"name": name, "samples": len(samples) if isinstance(samples, list) else 0} for name, samples in actors.items()]
    summary.sort(key=lambda x: (-(x.get("samples") or 0), x.get("name") or ""))
    return jsonify(
        {
            "ok": True,
            "engine": {"available": ok, "reason": reason},
            "actors": summary,
            "total_actors": len(summary),
        }
    ), 200


@app.route("/local-faces/enroll", methods=["POST"])
def local_faces_enroll() -> Any:
    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400
    actor_name = (request.form.get("actor_name") or "").strip()
    actor_name = re.sub(r"\s+", " ", actor_name)
    if not actor_name:
        return jsonify({"error": "actor_name is required"}), 400

    ok, reason = _insightface_available()
    if not ok:
        return jsonify(
            {
                "error": "Local face engine not available",
                "detail": reason,
                "hint": "Install optional dependencies (insightface, onnxruntime, numpy) to enable local matching.",
            }
        ), 501

    image_bytes = request.files["image"].read()
    if not image_bytes:
        return jsonify({"error": "empty image"}), 400

    try:
        analyzer = _get_insightface_analyzer()
        img = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))  # type: ignore
        faces = analyzer.get(img)
        if not faces:
            return jsonify({"error": "No face detected in image"}), 400

        # Choose the largest detected face
        def _area(face_obj) -> float:
            bbox = getattr(face_obj, "bbox", None)
            if bbox is None:
                return 0.0
            x1, y1, x2, y2 = [float(v) for v in list(bbox)]
            return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

        face_obj = max(faces, key=_area)
        emb = getattr(face_obj, "embedding", None)
        if emb is None:
            return jsonify({"error": "Face embedding unavailable"}), 500
        emb_vec = np.asarray(emb, dtype=np.float32)  # type: ignore

        gallery = _local_gallery_load()
        gallery.setdefault("actors", {})
        gallery["actors"].setdefault(actor_name, [])
        gallery["actors"][actor_name].append(
            {
                "embedding": emb_vec.tolist(),
                "added_at": datetime.utcnow().isoformat(),
                "filename": request.files["image"].filename,
            }
        )
        _local_gallery_save(gallery)
        return jsonify({"ok": True, "actor_name": actor_name, "samples": len(gallery["actors"][actor_name])}), 200
    except Exception as exc:
        app.logger.error("Local enroll failed: %s", exc)
        return jsonify({"error": f"Local enroll failed: {exc}"}), 500


def _extract_video_frames(video_path: Path, output_dir: Path, interval: int = 5) -> List[Path]:
    """Extract frames from video at specified interval (in seconds)."""
    frames = []
    
    # Get video duration
    probe_cmd = [
        FFPROBE_PATH, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    duration_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    duration = float(duration_result.stdout.strip())
    
    # Extract frames at intervals
    for timestamp in range(0, int(duration), interval):
        frame_path = output_dir / f"frame_{timestamp:04d}.jpg"
        extract_cmd = [
            FFMPEG_PATH, "-ss", str(timestamp),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(frame_path), "-y"
        ]
        result = subprocess.run(extract_cmd, capture_output=True)
        if result.returncode == 0:
            frames.append(frame_path)
    
    return frames


def _probe_video_duration_seconds(video_path: Path) -> float | None:
    probe_cmd = [
        FFPROBE_PATH,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        duration_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        raw = (duration_result.stdout or "").strip()
        if not raw:
            return None
        return float(raw)
    except Exception:
        return None


def _probe_technical_metadata(video_path: Path) -> dict[str, Any]:
    """Extract technical/container metadata using ffprobe.

    Best-effort. Returns empty dict on failure.
    """
    cmd = [
        FFPROBE_PATH,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        raw = (proc.stdout or "").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        fmt = data.get("format") or {}
        streams = data.get("streams") or []

        video_stream = next((s for s in streams if (s.get("codec_type") == "video")), {})
        audio_stream = next((s for s in streams if (s.get("codec_type") == "audio")), {})

        def _to_int(v: Any) -> int | None:
            try:
                if v is None or v == "":
                    return None
                return int(float(v))
            except Exception:
                return None

        def _frame_rate(rate: Any) -> float | None:
            try:
                if not rate or rate == "0/0":
                    return None
                if isinstance(rate, str) and "/" in rate:
                    a, b = rate.split("/", 1)
                    return float(a) / max(1.0, float(b))
                return float(rate)
            except Exception:
                return None

        width = _to_int(video_stream.get("width"))
        height = _to_int(video_stream.get("height"))
        aspect_ratio = (video_stream.get("display_aspect_ratio") or video_stream.get("sample_aspect_ratio") or "").strip() or None

        return {
            "container_format": (fmt.get("format_name") or "").strip() or None,
            "duration_seconds": float(fmt.get("duration")) if str(fmt.get("duration") or "").strip() else None,
            "bitrate_bps": _to_int(fmt.get("bit_rate")),
            "video_codec": (video_stream.get("codec_name") or "").strip() or None,
            "audio_codec": (audio_stream.get("codec_name") or "").strip() or None,
            "resolution": {"width": width, "height": height},
            "frame_rate": _frame_rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")),
            "aspect_ratio": aspect_ratio,
            "audio_channels": _to_int(audio_stream.get("channels")),
            "audio_channel_layout": (audio_stream.get("channel_layout") or "").strip() or None,
            # HDR/subtitle streams are not reliably present in all files; keep best-effort raw info.
            "raw": {"format": fmt, "video_stream": video_stream, "audio_stream": audio_stream},
        }
    except Exception:
        return {}


def _looks_rekognition_compatible(technical: dict[str, Any]) -> bool:
    """Return True when the media looks compatible with Rekognition Video.

    Rekognition Video is most reliable with MP4 (ISO BMFF) containers + H.264 (AVC)
    video and AAC audio. This is a best-effort heuristic based on ffprobe output.
    """

    if not isinstance(technical, dict):
        return False
    container = str(technical.get("container_format") or "").lower()
    video_codec = str(technical.get("video_codec") or "").lower()
    audio_codec = str(technical.get("audio_codec") or "").lower()

    container_ok = any(x in container for x in ("mp4", "mov", "isom"))
    video_ok = (video_codec == "h264")

    # Some files may be video-only. Rekognition doesn't strictly require audio,
    # but when present it should be AAC to avoid format issues.
    audio_ok = (not audio_codec) or (audio_codec == "aac")
    return bool(container_ok and video_ok and audio_ok)


def _transcode_to_rekognition_mp4(*, input_path: Path, output_path: Path, job_id: str) -> None:
    """Transcode to MP4 (H.264/AAC) for Rekognition compatibility."""

    input_tech = _probe_technical_metadata(input_path)
    has_audio = bool(input_tech.get("audio_codec"))

    cmd: list[str] = [
        FFMPEG_PATH,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    if has_audio:
        cmd += ["-i", str(input_path)]
    else:
        # Add silent audio when missing, to keep the output consistently AAC.
        cmd += [
            "-i",
            str(input_path),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-shortest",
        ]

    cmd += [
        "-map",
        "0:v:0",
        "-map",
        "0:a:0" if has_audio else "1:a:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-preset",
        str(os.getenv("ENVID_METADATA_FFMPEG_PRESET") or "veryfast"),
        "-crf",
        str(os.getenv("ENVID_METADATA_FFMPEG_CRF") or "20"),
        "-maxrate",
        str(os.getenv("ENVID_METADATA_FFMPEG_MAXRATE") or "6000k"),
        "-bufsize",
        str(os.getenv("ENVID_METADATA_FFMPEG_BUFSIZE") or "12000k"),
        "-c:a",
        "aac",
        "-b:a",
        str(os.getenv("ENVID_METADATA_FFMPEG_ABR") or "128k"),
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    _job_update(job_id, progress=2, message="Transcoding to Rekognition-compatible MP4")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg transcode failed: {err or 'unknown error'}")


def _ensure_rekognition_compatible_before_upload(
    *,
    video_path: Path,
    temp_dir: Path,
    job_id: str,
) -> Path:
    """Validate + transcode if needed, returning the path to upload."""

    if not _env_truthy(os.getenv("ENVID_METADATA_TRANSCODE_BEFORE_UPLOAD"), default=True):
        return video_path

    tech = _probe_technical_metadata(video_path)
    if _looks_rekognition_compatible(tech):
        return video_path

    output_path = temp_dir / "rekognition_proxy.mp4"
    _transcode_to_rekognition_mp4(input_path=video_path, output_path=output_path, job_id=job_id)
    # Sanity-check the output.
    out_tech = _probe_technical_metadata(output_path)
    if not _looks_rekognition_compatible(out_tech):
        raise RuntimeError(
            "Transcoded output still looks incompatible with Rekognition Video. "
            f"ffprobe: {json.dumps(out_tech, ensure_ascii=False)[:1200]}"
        )
    return output_path


def _probe_mediainfo_metadata(video_path: Path) -> dict[str, Any]:
    """Extract technical/container metadata using mediainfo (CLI) when available.

    This is best-effort and returns an empty dict if mediainfo is not installed or fails.
    """

    def _to_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            s = str(value).strip()
            if not s:
                return None
            s = s.replace(" ", "")
            return int(float(s))
        except Exception:
            return None

    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            s = str(value).strip()
            if not s:
                return None
            s = s.replace(" ", "")
            return float(s)
        except Exception:
            return None

    def _duration_seconds_from_track(track: dict[str, Any]) -> float | None:
        # MediaInfo often returns Duration in ms.
        dur_ms = _to_float(track.get("Duration"))
        if dur_ms and dur_ms > 1000:
            return float(dur_ms) / 1000.0
        # Sometimes it returns seconds.
        if dur_ms and 0 < dur_ms <= 1000:
            # If it was seconds but less than 1000, accept as seconds.
            return float(dur_ms)
        return None

    try:
        if not shutil.which(MEDIAINFO_PATH):
            return {"available": False, "error": f"mediainfo not found on PATH (MEDIAINFO_PATH={MEDIAINFO_PATH})"}
        if not video_path.exists():
            return {}

        cmd = [
            MEDIAINFO_PATH,
            "--Output=JSON",
            str(video_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
            return {"available": False, "error": err or "mediainfo failed"}

        raw = (proc.stdout or "").strip()
        if not raw:
            return {}

        payload = json.loads(raw)
        media = (payload.get("media") or {}) if isinstance(payload, dict) else {}
        tracks = media.get("track")
        if not isinstance(tracks, list):
            tracks = []

        general = next((t for t in tracks if str(t.get("@type") or "").lower() == "general"), {})
        video = next((t for t in tracks if str(t.get("@type") or "").lower() == "video"), {})
        audio = next((t for t in tracks if str(t.get("@type") or "").lower() == "audio"), {})

        width = _to_int(video.get("Width"))
        height = _to_int(video.get("Height"))
        frame_rate = _to_float(video.get("FrameRate"))

        return {
            "available": True,
            "container_format": (general.get("Format") or "").strip() or None,
            "duration_seconds": _duration_seconds_from_track(general) or _duration_seconds_from_track(video) or None,
            "bitrate_bps": _to_int(general.get("OverallBitRate")) or _to_int(video.get("BitRate")) or None,
            "file_size_bytes": _to_int(general.get("FileSize")),
            "video_codec": (video.get("Format") or video.get("CodecID") or "").strip() or None,
            "audio_codec": (audio.get("Format") or audio.get("CodecID") or "").strip() or None,
            "resolution": {"width": width, "height": height} if (width or height) else None,
            "frame_rate": frame_rate,
            "aspect_ratio": (video.get("DisplayAspectRatio") or "").strip() or None,
            "audio_channels": _to_int(audio.get("Channels")),
            "audio_sampling_rate_hz": _to_int(audio.get("SamplingRate")),
            "audio_language": (audio.get("Language") or "").strip() or None,
            "raw": {"general": general, "video": video, "audio": audio},
        }
    except Exception:
        return {"available": False, "error": "mediainfo probe exception"}


def _verify_technical_metadata(
    *,
    ffprobe: dict[str, Any] | None,
    mediainfo: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare ffprobe vs mediainfo for core fields.

    Returns best-effort diffs for debugging/verification.
    """

    def _num(v: Any) -> float | None:
        try:
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v).strip()
            if not s:
                return None
            return float(s)
        except Exception:
            return None

    def _get_res(meta: dict[str, Any] | None) -> tuple[int | None, int | None]:
        if not isinstance(meta, dict):
            return (None, None)
        r = meta.get("resolution")
        if not isinstance(r, dict):
            return (None, None)
        try:
            return (int(r.get("width")) if r.get("width") is not None else None, int(r.get("height")) if r.get("height") is not None else None)
        except Exception:
            return (None, None)

    ff = ffprobe or {}
    mi = mediainfo or {}

    ff_dur = _num(ff.get("duration_seconds"))
    mi_dur = _num(mi.get("duration_seconds"))
    ff_fps = _num(ff.get("frame_rate"))
    mi_fps = _num(mi.get("frame_rate"))
    ff_w, ff_h = _get_res(ff)
    mi_w, mi_h = _get_res(mi)

    duration_delta = None
    if ff_dur is not None and mi_dur is not None:
        duration_delta = abs(ff_dur - mi_dur)
    fps_delta = None
    if ff_fps is not None and mi_fps is not None:
        fps_delta = abs(ff_fps - mi_fps)

    return {
        "ffprobe_present": bool(ffprobe),
        "mediainfo_present": bool(mediainfo),
        "duration_seconds": {"ffprobe": ff_dur, "mediainfo": mi_dur, "abs_delta": duration_delta},
        "frame_rate": {"ffprobe": ff_fps, "mediainfo": mi_fps, "abs_delta": fps_delta},
        "resolution": {"ffprobe": {"width": ff_w, "height": ff_h}, "mediainfo": {"width": mi_w, "height": mi_h}},
        "video_codec": {"ffprobe": ff.get("video_codec"), "mediainfo": mi.get("video_codec")},
        "audio_codec": {"ffprobe": ff.get("audio_codec"), "mediainfo": mi.get("audio_codec")},
        "container_format": {"ffprobe": ff.get("container_format"), "mediainfo": mi.get("container_format")},
    }


def _build_categorized_metadata_json(video: dict[str, Any]) -> dict[str, Any]:
    """Build requested metadata categories as separate JSON objects.

    This is intentionally best-effort: fields not available from the current pipeline
    are returned as null/empty lists.
    """

    title = (video.get("title") or "").strip()
    description = (video.get("description") or "").strip()
    original_filename = (video.get("original_filename") or "").strip()
    s3_uri = (video.get("s3_video_uri") or "").strip() or None
    s3_key = (video.get("s3_video_key") or "").strip() or None

    duration_seconds = video.get("duration_seconds")
    language_code = video.get("language_code")
    languages_detected = video.get("languages_detected") or []

    labels_detailed = video.get("labels_detailed") or []
    moderation_detailed = video.get("moderation_detailed") or []
    text_detailed = video.get("text_detailed") or []
    transcript = (video.get("transcript") or "")
    transcript_segments = video.get("transcript_segments") or []
    celebrities_detailed = video.get("celebrities_detailed") or []
    faces_summary = video.get("faces_summary") or {}
    emotions = video.get("emotions") or []

    thumbnail = video.get("thumbnail")
    frames = video.get("frames") or []
    frame_interval_seconds = video.get("frame_interval_seconds")

    # Heuristic compliance flags
    mod_threshold = float(os.getenv("ENVID_METADATA_MODERATION_FLAG_THRESHOLD") or 80.0)
    has_moderation = any(float(m.get("max_confidence") or 0.0) >= mod_threshold for m in moderation_detailed)

    profile = (str(video.get("output_profile") or "") or _envid_metadata_output_profile()).strip().lower()
    if profile in {"required", "minimal"}:
        bio_cache = _bio_cache_load()

        # Best-effort: ensure celebrity bios meet the minimum length requirement.
        try:
            names_needing_bio: list[str] = []
            for c in celebrities_detailed if isinstance(celebrities_detailed, list) else []:
                if not isinstance(c, dict):
                    continue
                nm = (c.get("name") or "").strip()
                if not nm:
                    continue
                current = (c.get("bio_short") or c.get("bio") or "").strip()
                if _bio_is_too_short(current):
                    names_needing_bio.append(nm)
            names_needing_bio = sorted(list(dict.fromkeys(names_needing_bio)))
            refreshed_bios = _celebrity_bios_for_names(names_needing_bio) if names_needing_bio else {}
        except Exception:
            refreshed_bios = {}

        # Ensure celebrities include timeline segments derived from timestamps.
        celeb_gap_seconds = max(1, int(float(frame_interval_seconds or 1) * 2))
        celebrities_time_mapped: List[Dict[str, Any]] = []
        for c in celebrities_detailed if isinstance(celebrities_detailed, list) else []:
            if not isinstance(c, dict):
                continue
            nm = (c.get("name") or "").strip()

            # Apply refreshed bio (>=30 words) if available.
            if nm and isinstance(refreshed_bios, dict) and nm in refreshed_bios:
                bio_obj = refreshed_bios.get(nm)
                if isinstance(bio_obj, dict):
                    bio_val = (bio_obj.get("bio") or "").strip()
                    if not _bio_is_too_short(bio_val):
                        c = dict(c)
                        c["bio"] = bio_val
                        c["bio_source"] = bio_obj.get("source")
                    if "bio_short" in c and _bio_is_too_short((c.get("bio_short") or "").strip()):
                        c = dict(c)
                        c["bio_short"] = None
                    # Portraits: also apply if provided.
                    if not (c.get("portrait_url") or "").strip() and (bio_obj.get("portrait_url") or "").strip():
                        c = dict(c)
                        c["portrait_url"] = bio_obj.get("portrait_url")
                        c["portrait_source"] = bio_obj.get("portrait_source")
                        c["portrait_license"] = bio_obj.get("portrait_license")
                        c["portrait_license_url"] = bio_obj.get("portrait_license_url")
                        c["portrait_attribution"] = bio_obj.get("portrait_attribution")

            cached = bio_cache.get(nm) if nm and isinstance(bio_cache, dict) else None
            if isinstance(cached, dict):
                # Backfill portrait fields (and bio if missing) from local cache.
                if not (c.get("portrait_url") or "").strip() and (cached.get("portrait_url") or "").strip():
                    c = dict(c)
                    c["portrait_url"] = cached.get("portrait_url")
                    c["portrait_source"] = cached.get("portrait_source")
                    c["portrait_license"] = cached.get("portrait_license")
                    c["portrait_license_url"] = cached.get("portrait_license_url")
                    c["portrait_attribution"] = cached.get("portrait_attribution")
                # Only backfill cached bios that meet the minimum length.
                cached_bio = (cached.get("bio") or "").strip()
                if _bio_is_too_short((c.get("bio") or "").strip()) and not _bio_is_too_short(cached_bio):
                    c = dict(c)
                    c["bio"] = cached_bio
                    c["bio_source"] = cached.get("source")
            ts_list = c.get("timestamps_seconds") or []
            try:
                ts_norm = sorted(list(dict.fromkeys([int(t) for t in ts_list if t is not None])))
            except Exception:
                ts_norm = []

            ts_ms_list = c.get("timestamps_ms") or []
            try:
                ts_ms_norm = sorted(list(dict.fromkeys([int(t) for t in ts_ms_list if t is not None])))
            except Exception:
                ts_ms_norm = []

            c2 = dict(c)
            if ts_norm:
                c2["timestamps_seconds"] = ts_norm
                segs = _timestamps_to_segments(ts_norm, gap_seconds=celeb_gap_seconds)
                # The timestamps come from binning onto a fixed interval grid.
                # Extend the end slightly so a detection in bin `t` still matches
                # when the user pauses near the end of that bin.
                try:
                    extend_s = max(0, int(frame_interval_seconds or 0))
                except Exception:
                    extend_s = 0
                if extend_s > 0 and segs:
                    dur_s = None
                    try:
                        dur_s = int(float(duration_seconds)) if duration_seconds is not None else None
                    except Exception:
                        dur_s = None
                    segs2 = []
                    for s in segs:
                        a = s.get("start_seconds")
                        b = s.get("end_seconds")
                        try:
                            b2 = int(b) + int(extend_s)
                        except Exception:
                            b2 = b
                        if dur_s is not None:
                            try:
                                b2 = min(int(b2), int(dur_s))
                            except Exception:
                                pass
                        segs2.append({"start_seconds": a, "end_seconds": b2})
                    segs = segs2
                c2["segments"] = segs
                # If first/last are missing, fill them.
                if c2.get("first_seen_seconds") is None:
                    c2["first_seen_seconds"] = ts_norm[0]
                if c2.get("last_seen_seconds") is None:
                    c2["last_seen_seconds"] = ts_norm[-1]
            else:
                c2["segments"] = []

            # Optional: higher-resolution celebrity timeline for frame-based timecodes.
            # If not present, synthesize from second-level timestamps.
            if ts_ms_norm:
                c2["timestamps_ms"] = ts_ms_norm
            elif ts_norm:
                c2["timestamps_ms"] = [int(t) * 1000 for t in ts_norm]

            if isinstance(c2.get("timestamps_ms"), list) and c2.get("timestamps_ms"):
                try:
                    gap_ms = max(200, min(5000, int(celeb_gap_seconds) * 1000))
                except Exception:
                    gap_ms = 1000
                try:
                    extend_ms = max(0, int(frame_interval_seconds or 0) * 1000)
                except Exception:
                    extend_ms = 0
                c2["segments_ms"] = _timestamps_ms_to_segments(
                    [int(t) for t in (c2.get("timestamps_ms") or []) if t is not None],
                    gap_ms=gap_ms,
                    extend_end_ms=extend_ms,
                )
            else:
                c2["segments_ms"] = []

            celebrities_time_mapped.append(c2)

        ff = video.get("technical_ffprobe") if isinstance(video.get("technical_ffprobe"), dict) else {}
        mi = video.get("technical_mediainfo") if isinstance(video.get("technical_mediainfo"), dict) else {}
        rkvm = (
            video.get("technical_rekognition_video_metadata")
            if isinstance(video.get("technical_rekognition_video_metadata"), dict)
            else {}
        )

        # Best-effort fallbacks: cloud-only paths often have MediaInfo but not ffprobe.
        mi_container = mi.get("container_format")
        mi_duration = mi.get("duration_seconds")
        mi_resolution = mi.get("resolution")
        mi_frame_rate = mi.get("frame_rate")
        mi_bitrate = mi.get("bitrate_bps")
        mi_vcodec = mi.get("video_codec")
        mi_acodec = mi.get("audio_codec")

        rk_container = rkvm.get("Format") or rkvm.get("format")
        rk_duration = rkvm.get("DurationMillis")
        try:
            rk_duration_s = (float(rk_duration) / 1000.0) if rk_duration is not None else None
        except Exception:
            rk_duration_s = None
        rk_fps = rkvm.get("FrameRate") or rkvm.get("frame_rate")
        rk_w = rkvm.get("FrameWidth") or rkvm.get("frame_width")
        rk_h = rkvm.get("FrameHeight") or rkvm.get("frame_height")
        rk_resolution = None
        try:
            w = int(rk_w) if rk_w is not None else None
            h = int(rk_h) if rk_h is not None else None
            if w and h:
                rk_resolution = {"width": w, "height": h}
        except Exception:
            rk_resolution = None

        technical_required = {
            "title": title,
            "container_format": ff.get("container_format") or mi_container or rk_container,
            "duration_seconds": (
                ff.get("duration_seconds")
                if ff.get("duration_seconds") is not None
                else (mi_duration if mi_duration is not None else (rk_duration_s if rk_duration_s is not None else duration_seconds))
            ),
            "resolution": ff.get("resolution") or mi_resolution or rk_resolution,
            "frame_rate": ff.get("frame_rate") or mi_frame_rate or rk_fps,
            "file_size_bytes": video.get("file_size_bytes") or mi.get("file_size_bytes"),
            "bitrate_bps": ff.get("bitrate_bps") or mi_bitrate,
            "video_codec": ff.get("video_codec") or mi_vcodec,
            "audio_codec": ff.get("audio_codec") or mi_acodec,
        }

        detected_content = _build_detected_content_timelines(
            frames=frames,
            frame_interval_seconds=frame_interval_seconds,
        )

        detected_content_required = {
            "labels": detected_content.get("labels") if isinstance(detected_content, dict) else [],
            "moderation": detected_content.get("moderation_labels") if isinstance(detected_content, dict) else [],
            "on_screen_text": detected_content.get("on_screen_text") if isinstance(detected_content, dict) else [],
            "frame_interval_seconds": detected_content.get("frame_interval_seconds") if isinstance(detected_content, dict) else None,
        }

        famous_locations_existing = video.get("famous_locations") if isinstance(video.get("famous_locations"), dict) else None
        if isinstance(famous_locations_existing, dict) and famous_locations_existing.get("locations"):
            locations_required = {
                "locations": famous_locations_existing.get("locations") or [],
                "time_mapped": famous_locations_existing.get("time_mapped") or [],
                "from_transcript": famous_locations_existing.get("from_transcript") or [],
                "from_landmarks": famous_locations_existing.get("from_landmarks") or [],
            }
        else:
            raw_locations = video.get("locations") or []
            built = _build_famous_locations_payload(
                text_locations=raw_locations if isinstance(raw_locations, list) else [],
                transcript_segments=transcript_segments if isinstance(transcript_segments, list) else [],
                frames=frames if isinstance(frames, list) else [],
                geocode_cache=(video.get("locations_geocoded") if isinstance(video.get("locations_geocoded"), dict) else None),
            )
            locations_required = {
                "locations": built.get("locations") if isinstance(built, dict) else (raw_locations if isinstance(raw_locations, list) else []),
                "time_mapped": built.get("time_mapped") if isinstance(built, dict) else _time_map_locations_from_segments(
                    locations=raw_locations if isinstance(raw_locations, list) else [],
                    transcript_segments=transcript_segments if isinstance(transcript_segments, list) else [],
                ),
                "from_transcript": built.get("from_transcript") if isinstance(built, dict) else (raw_locations if isinstance(raw_locations, list) else []),
                "from_landmarks": built.get("from_landmarks") if isinstance(built, dict) else [],
            }

        scene_pack = video.get("scene_metadata") if isinstance(video.get("scene_metadata"), dict) else {}
        raw_scenes = scene_pack.get("scenes") if isinstance(scene_pack.get("scenes"), list) else []
        scenes_required = [
            {
                "scene_index": s.get("scene_index"),
                "start_seconds": s.get("start_seconds"),
                "end_seconds": s.get("end_seconds"),

                # Human-readable snippet (deterministic) + rich time-mapped context
                "summary_text": s.get("summary_text"),
                "celebrities": s.get("celebrities") if isinstance(s.get("celebrities"), list) else [],
                "labels": s.get("labels") if isinstance(s.get("labels"), list) else [],
                "moderation_labels": s.get("moderation_labels") if isinstance(s.get("moderation_labels"), list) else [],
                "on_screen_text": s.get("on_screen_text") if isinstance(s.get("on_screen_text"), list) else [],
                "transcript_segments": s.get("transcript_segments") if isinstance(s.get("transcript_segments"), list) else [],
            }
            for s in raw_scenes
            if isinstance(s, dict)
        ]

        categories_required: dict[str, Any] = {
            "celebrity_detection": {
                "celebrities": celebrities_time_mapped,
            },
            "celebrity_table": {
                "celebrities": celebrities_time_mapped,
            },
            "detected_content": detected_content_required,
            "subtitles": video.get("subtitles") or {},
            "famous_locations": locations_required,
            "technical_metadata": technical_required,
            "synopses_by_age_group": _normalize_synopses_by_age_group_for_output(
                video.get("synopses_by_age_group"),
                source_text=(video.get("transcript") or video.get("description") or ""),
                title=video.get("title"),
            ),
            "scene_by_scene_metadata": scenes_required,
            "key_scenes": scene_pack.get("key_scenes") if isinstance(scene_pack.get("key_scenes"), list) else [],
            "high_points": scene_pack.get("high_points") if isinstance(scene_pack.get("high_points"), list) else [],
        }

        combined_required = {
            "video_id": video.get("id"),
            "categories": categories_required,
        }

        return {"categories": categories_required, "combined": combined_required}

    core_descriptive = {
        "title": title,
        "original_title": None,
        "localized_title": None,
        "synopsis_short": description[:280] if description else None,
        "synopsis_long": description or None,
        "genre": [],
        "release_date": None,
        "runtime_seconds": duration_seconds,
        "production_company": None,
        "original_language": language_code,
        "dubbed_languages": [],
        "subtitle_languages": [d.get("language_code") for d in languages_detected if d.get("language_code")],
        "country_of_origin": None,
        "source": {"original_filename": original_filename, "s3_uri": s3_uri, "s3_key": s3_key},
    }

    creative_credits = {
        "directors": [],
        "producers": [],
        "writers": [],
        # Best-effort cast from celebrity recognition.
        "cast": [{"actor_name": c.get("name"), "max_confidence": c.get("max_confidence")} for c in celebrities_detailed if c.get("name")],
        "crew": {"cinematographer": None, "editor": None, "music_composer": None},
    }

    technical = {
        "video_codec": None,
        "audio_codec": None,
        "container_format": None,
        "resolution": None,
        "frame_rate": None,
        "bitrate_bps": None,
        "aspect_ratio": None,
        "hdr_type": None,
        "audio_channels": None,
        "subtitle_formats": [],
        "ffprobe": video.get("technical_ffprobe") or None,
        "mediainfo": video.get("technical_mediainfo") or None,
        "verification": video.get("technical_verification") or None,
    }
    # If we stored ffprobe, surface top-level fields too.
    if isinstance(video.get("technical_ffprobe"), dict):
        ff = video.get("technical_ffprobe") or {}
        technical.update(
            {
                "video_codec": ff.get("video_codec"),
                "audio_codec": ff.get("audio_codec"),
                "container_format": ff.get("container_format"),
                "resolution": ff.get("resolution"),
                "frame_rate": ff.get("frame_rate"),
                "bitrate_bps": ff.get("bitrate_bps"),
                "aspect_ratio": ff.get("aspect_ratio"),
                "audio_channels": ff.get("audio_channels"),
            }
        )

    # If ffprobe is missing (or incomplete), use mediainfo as fallback for top-level fields.
    if not technical.get("video_codec") and isinstance(video.get("technical_mediainfo"), dict):
        mi = video.get("technical_mediainfo") or {}
        technical.update(
            {
                "video_codec": technical.get("video_codec") or mi.get("video_codec"),
                "audio_codec": technical.get("audio_codec") or mi.get("audio_codec"),
                "container_format": technical.get("container_format") or mi.get("container_format"),
                "resolution": technical.get("resolution") or mi.get("resolution"),
                "frame_rate": technical.get("frame_rate") or mi.get("frame_rate"),
                "bitrate_bps": technical.get("bitrate_bps") or mi.get("bitrate_bps"),
                "aspect_ratio": technical.get("aspect_ratio") or mi.get("aspect_ratio"),
                "audio_channels": technical.get("audio_channels") or mi.get("audio_channels"),
            }
        )

    artwork_assets = {
        "poster_images": [],
        "thumbnails": [
            {"type": "base64_jpeg", "data": thumbnail} if thumbnail else None
        ],
        "trailers": [],
        "subtitle_files": [],
    }
    artwork_assets["thumbnails"] = [t for t in artwork_assets["thumbnails"] if t]

    structural_temporal = {
        "shot_boundaries": [],
        "scene_boundaries": [],
        "chapter_markers": [],
        "timestamps": [{"timestamp": f.get("timestamp")} for f in frames if f.get("timestamp") is not None],
        "keyframes": frames,
        "frame_interval_seconds": frame_interval_seconds,
    }

    content_rating_compliance = {
        "age_rating": None,
        "content_advisory_tags": [m.get("name") for m in moderation_detailed if m.get("name")],
        "regional_compliance_flags": {
            "allowed": None,
            "blocked": None,
            "age_restricted": None,
            "requires_edit": None,
            "ad_safe": None,
            "manual_review_required": bool(has_moderation),
        },
    }

    text_based_ai = {
        "on_screen_text": [{"text": t.get("text"), "occurrences": t.get("occurrences")} for t in text_detailed if t.get("text")],
        "transcript_translation_en": (video.get("transcript_translation_en") or "") if isinstance(video.get("transcript_translation_en"), str) else "",
        "entities": video.get("comprehend_entities") or [],
        "key_phrases": video.get("comprehend_key_phrases") or [],
        "sentiment": video.get("comprehend_sentiment"),
        "summary": (video.get("bedrock_summary") or "") if isinstance(video.get("bedrock_summary"), str) else "",
        "summary_audio_s3_uri": ((video.get("polly_summary_audio") or {}).get("s3_uri") if isinstance(video.get("polly_summary_audio"), dict) else None),
        "detected_logos": [],
        "watermarks": [],
    }

    speech_audio_intel = {
        "transcript": transcript,
        "speaker_labels": [],
        "language_detection": {
            "language_code": language_code,
            "languages_detected": languages_detected,
        },
        "profanity_detection": None,
        "audio_event_tags": [],
        "transcript_segments": transcript_segments,
    }

    visual_object_scene = {
        "objects_detected": [{"name": l.get("name"), "max_confidence": l.get("max_confidence"), "avg_confidence": l.get("avg_confidence")} for l in labels_detailed if l.get("name")],
        "scene_labels": [],
        "environment_labels": [],
    }

    facial_human = {
        "face_detection": faces_summary,
        "face_landmarks": [],
        "facial_attributes": {"emotions": emotions},
        "person_tracking": [],
        "crowd_detection": None,
        "celebrities": celebrities_detailed,
    }

    sentiment_emotion = {
        "scene_emotion": [],
        "facial_emotion_scores": emotions,
        "audio_emotion": None,
    }

    action_activity = {
        "human_actions": [],
        "interactions": [],
    }

    safety_moderation = {
        "explicit_content_scores": [],
        "violence_detection": [],
        "drug_use": [],
        "hate_symbol_detection": [],
        "moderation_labels": moderation_detailed,
        "flag_threshold": mod_threshold,
    }

    discoverability_reco = {
        "keywords": sorted(list({*(video.get("labels") or []), *(video.get("celebrities") or []), *(video.get("text_detected") or [])})),
        "mood": None,
        "audience_type": None,
        "similar_content_vectors": {"embedding": video.get("embedding")},
    }

    categories = {
        "core_descriptive_metadata": core_descriptive,
        "creative_credits_metadata": creative_credits,
        "technical_metadata": technical,
        "artwork_asset_metadata": artwork_assets,
        "structural_temporal_metadata": structural_temporal,
        "content_rating_compliance_metadata": content_rating_compliance,
        "text_based_ai_metadata": text_based_ai,
        "speech_audio_intelligence": speech_audio_intel,
        "visual_object_scene_metadata": visual_object_scene,
        "facial_human_metadata": facial_human,
        "sentiment_emotion_metadata": sentiment_emotion,
        "action_activity_metadata": action_activity,
        "safety_moderation_metadata": safety_moderation,
        "discoverability_recommendation_metadata": discoverability_reco,
    }

    combined = {
        "video_id": video.get("id"),
        "generated_at": datetime.utcnow().isoformat(),
        "categories": categories,
    }

    return {"categories": categories, "combined": combined}


def _frame_timestamp_seconds(frame_path: Path) -> int | None:
    # Expected: frame_0010.jpg -> 10
    match = re.search(r"frame_(\d+)", frame_path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _speaker_by_start_time(transcript_data: Dict[str, Any]) -> Dict[str, str]:
    results = ((transcript_data or {}).get("results") or {})
    speaker_labels = results.get("speaker_labels") or {}
    segments = speaker_labels.get("segments") or []
    mapping: Dict[str, str] = {}
    for seg in segments:
        for it in (seg.get("items") or []):
            start_time = it.get("start_time")
            speaker = it.get("speaker_label")
            if start_time and speaker:
                mapping[str(start_time)] = str(speaker)
    return mapping


def _parse_transcribe_words(
    transcript_data: Dict[str, Any],
    *,
    max_words: int = 5000,
    speaker_map: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    items = (((transcript_data or {}).get("results") or {}).get("items") or [])
    words: List[Dict[str, Any]] = []
    for item in items:
        itype = item.get("type")
        alt = (item.get("alternatives") or [{}])[0] or {}
        content = alt.get("content")
        if not content:
            continue

        if itype == "punctuation":
            # Attach punctuation to the previous word so segments read naturally.
            if words:
                words[-1]["word"] = f"{words[-1].get('word', '')}{content}"
            continue

        if itype != "pronunciation":
            continue

        raw_start = item.get("start_time")
        try:
            start_time = float(item.get("start_time"))
            end_time = float(item.get("end_time"))
        except (TypeError, ValueError):
            continue
        word: Dict[str, Any] = {
            "start": start_time,
            "end": end_time,
            "word": content,
        }
        if speaker_map and raw_start:
            speaker = speaker_map.get(str(raw_start))
            if speaker:
                word["speaker"] = speaker
        confidence = alt.get("confidence")
        try:
            if confidence is not None:
                word["confidence"] = float(confidence)
        except (TypeError, ValueError):
            pass
        words.append(word)
        if len(words) >= max_words:
            break
    return words


def _parse_float_param(
    raw: str | None,
    *,
    default: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    try:
        value = float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        value = float(default)
    if min_value is not None:
        value = max(float(min_value), value)
    if max_value is not None:
        value = min(float(max_value), value)
    return value


def _transcribe_segment_params_from_env() -> Dict[str, Any]:
    # Default is intentionally permissive to avoid dropping words in subtitles.
    # Tuning can be done via env vars without code changes.
    return {
        "window_seconds": _parse_float_param(os.getenv("TRANSCRIBE_SEGMENT_WINDOW_SECONDS"), default=8.0, min_value=2.0, max_value=30.0),
        "gap_seconds": _parse_float_param(os.getenv("TRANSCRIBE_SEGMENT_GAP_SECONDS"), default=1.25, min_value=0.25, max_value=10.0),
        "min_confidence": _parse_float_param(os.getenv("TRANSCRIBE_SEGMENT_MIN_WORD_CONFIDENCE"), default=0.0, min_value=0.0, max_value=1.0),
        "max_segments": _parse_int_param(os.getenv("TRANSCRIBE_SEGMENT_MAX_SEGMENTS"), default=250, min_value=50, max_value=2000),
    }


def _normalize_transcript_basic(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # Collapse whitespace and clean up common punctuation spacing.
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    return t.strip()


def _apply_transcript_patches(text: str, *, language_code: str | None) -> str:
    """Apply user-provided regex substitutions to transcripts/subtitles.

    This is an opt-in deterministic mechanism for fixing known recurring ASR mistakes
    (including inserting a missing word) without using an LLM.

    Configure via ONE of:
      - ENVID_METADATA_TRANSCRIPT_PATCHES_JSON: JSON list of {pattern,replacement,flags?}
      - ENVID_METADATA_TRANSCRIPT_PATCHES_FILE: path to a JSON file with the same shape
    """

    raw = (text or "").strip()
    if not raw:
        return ""

    patches_json = (os.getenv("ENVID_METADATA_TRANSCRIPT_PATCHES_JSON") or "").strip()
    patches_file = (os.getenv("ENVID_METADATA_TRANSCRIPT_PATCHES_FILE") or "").strip()
    if not patches_json and not patches_file:
        return raw

    data = None
    try:
        if patches_json:
            data = json.loads(patches_json)
        elif patches_file:
            p = Path(patches_file)
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return raw

    if not isinstance(data, list):
        return raw

    out = raw
    # Keep this bounded: prevents runaway regex/pattern counts.
    max_rules = _parse_int_param(os.getenv("ENVID_METADATA_TRANSCRIPT_PATCHES_MAX_RULES"), default=25, min_value=1, max_value=200)
    for rule in data[: int(max_rules)]:
        if not isinstance(rule, dict):
            continue
        pattern = (rule.get("pattern") or "").strip()
        repl = str(rule.get("replacement") or "")
        if not pattern:
            continue

        flags_raw = (rule.get("flags") or "").strip().lower()
        flags = 0
        if "i" in flags_raw:
            flags |= re.IGNORECASE
        if "m" in flags_raw:
            flags |= re.MULTILINE

        try:
            out = re.sub(pattern, repl, out, flags=flags)
        except Exception:
            continue

    # Keep the final string tidy.
    out = _normalize_transcript_basic(out)
    return out


def _apply_language_spelling_fixes(text: str, *, language_code: str | None) -> str:
    """Apply small language-specific spelling normalizations.

    This is intended to be conservative and deterministic.
    """

    out = (text or "").strip()
    if not out:
        return ""
    lang = (language_code or "").strip().lower()
    if lang.startswith("hi"):
        # AWS Transcribe custom vocabulary for hi-IN rejects nukta characters (e.g., 'ड़').
        # Normalize common vocab-friendly spellings back to preferred Hindi orthography.
        out = out.replace("जकडा", "जकड़ा")
    return out


def _transcript_segments_from_words(
    words: List[Dict[str, Any]],
    *,
    window_seconds: float = 8.0,
    gap_seconds: float = 1.25,
    min_confidence: float = 0.45,
    max_segments: int = 250,
) -> List[Dict[str, Any]]:
    if not words:
        return []

    segments: List[Dict[str, Any]] = []
    current: List[str] = []
    current_speaker = words[0].get("speaker")
    seg_start = float(words[0].get("start", 0.0) or 0.0)
    seg_end = float(words[0].get("end", seg_start) or seg_start)

    last_end = seg_end
    for w in words:
        w_start = float(w.get("start", seg_end) or seg_end)
        w_end = float(w.get("end", w_start) or w_start)
        token = (w.get("word") or "").strip()
        speaker = w.get("speaker")
        conf = w.get("confidence")
        try:
            conf_value = float(conf) if conf is not None else None
        except (TypeError, ValueError):
            conf_value = None

        # Split when speaker changes.
        if current and (speaker is not None) and (speaker != current_speaker):
            text = " ".join(current).strip()
            if text:
                seg_obj: Dict[str, Any] = {"start": seg_start, "end": seg_end, "text": text}
                if current_speaker is not None:
                    seg_obj["speaker"] = current_speaker
                segments.append(seg_obj)
            current = []
            seg_start = w_start
            seg_end = w_end
            current_speaker = speaker

        # If there's a large pause, close the current segment early.
        if current and (w_start - last_end) >= gap_seconds:
            text = " ".join(current).strip()
            if text:
                seg_obj = {"start": seg_start, "end": seg_end, "text": text}
                if current_speaker is not None:
                    seg_obj["speaker"] = current_speaker
                segments.append(seg_obj)
            current = []
            seg_start = w_start

        if token:
            if conf_value is None or conf_value >= min_confidence:
                current.append(token)
        seg_end = max(seg_end, w_end)
        last_end = w_end

        if (seg_end - seg_start) >= window_seconds:
            text = " ".join(current).strip()
            if text:
                seg_obj = {"start": seg_start, "end": seg_end, "text": text}
                if current_speaker is not None:
                    seg_obj["speaker"] = current_speaker
                segments.append(seg_obj)
            current = []
            seg_start = seg_end
            if len(segments) >= max_segments:
                break

    if current and len(segments) < max_segments:
        text = " ".join(current).strip()
        if text:
            seg_obj = {"start": seg_start, "end": seg_end, "text": text}
            if current_speaker is not None:
                seg_obj["speaker"] = current_speaker
            segments.append(seg_obj)

    # Merge tiny trailing segments (often a single garbled word).
    if len(segments) >= 2:
        last = segments[-1]
        prev = segments[-2]
        last_words = (last.get("text") or "").split()
        if len(last_words) <= 2 and (float(last.get("end") or 0.0) - float(last.get("start") or 0.0)) <= 3.0:
            merged = (prev.get("text") or "").rstrip()
            tail = (last.get("text") or "").strip()
            if tail:
                prev["text"] = (merged + " " + tail).strip()
                prev["end"] = last.get("end")
                segments.pop()

    return segments


def _analyze_frame_with_rekognition(
    frame_path: Path,
    *,
    collection_id: str | None = None,
    local_face_mode: bool = False,
    local_similarity_threshold: float = 0.35,
    enable_label_detection: bool = True,
    enable_text_detection: bool = True,
    enable_face_detection: bool = True,
    enable_moderation_detection: bool = True,
    enable_celebrity_detection: bool = True,
) -> Dict[str, Any]:
    """Analyze a single frame using Amazon Rekognition (plus optional face matching modes)."""
    with open(frame_path, "rb") as image_file:
        image_bytes = image_file.read()
    
    analysis = {
        "labels": [],
        "text": [],
        "faces": [],
        "celebrities": [],
        "moderation": [],
        "custom_faces": [],
        "local_faces": [],
        "celebrity_checked": False,
    }

    face_details: List[Dict[str, Any]] = []
    
    if enable_label_detection:
        try:
            # Detect labels (objects, scenes, activities)
            label_response = rekognition.detect_labels(
                Image={"Bytes": image_bytes},
                MaxLabels=20,
                MinConfidence=70,
            )
            analysis["labels"] = [
                {
                    "name": label.get("Name"),
                    "confidence": label.get("Confidence"),
                    "parents": [p.get("Name") for p in (label.get("Parents") or []) if isinstance(p, dict) and p.get("Name")],
                }
                for label in label_response.get("Labels", [])
            ]
        except Exception as e:
            app.logger.error(f"Label detection failed: {e}")
    
    if enable_text_detection:
        try:
            # Detect text in image
            text_response = rekognition.detect_text(Image={"Bytes": image_bytes})
            analysis["text"] = [
                {
                    "text": item.get("DetectedText"),
                    "type": item.get("Type"),
                    "confidence": item.get("Confidence"),
                }
                for item in text_response.get("TextDetections", [])
                if item.get("Type") in {"LINE", "WORD"} and item.get("DetectedText")
            ]
        except Exception as e:
            app.logger.error(f"Text detection failed: {e}")
    
    if enable_face_detection:
        try:
            # Detect faces and emotions
            face_response = rekognition.detect_faces(Image={"Bytes": image_bytes}, Attributes=["ALL"])
            face_details = face_response.get("FaceDetails", []) or []
            analysis["faces"] = [
                {
                    "gender": (face.get("Gender") or {}).get("Value"),
                    "gender_confidence": (face.get("Gender") or {}).get("Confidence"),
                    "age_range": face.get("AgeRange"),
                    "smile": (face.get("Smile") or {}).get("Value"),
                    "eyeglasses": (face.get("Eyeglasses") or {}).get("Value"),
                    "sunglasses": (face.get("Sunglasses") or {}).get("Value"),
                    "beard": (face.get("Beard") or {}).get("Value"),
                    "mustache": (face.get("Mustache") or {}).get("Value"),
                    "bounding_box": (
                        face.get("BoundingBox") if isinstance(face.get("BoundingBox"), dict) else None
                    ),
                    "emotions": [{"type": e["Type"], "confidence": e["Confidence"]} for e in face.get("Emotions", [])][
                        :3
                    ],
                }
                for face in face_details
            ]
        except Exception as e:
            app.logger.error(f"Face detection failed: {e}")
            face_details = []

    # Optional: AWS Rekognition Custom Face Collection matching
    collection_id_norm = _rekognition_collection_id_normalize(collection_id)
    if collection_id_norm and face_details:
        try:
            matches: List[Dict[str, Any]] = []
            for face in face_details:
                bbox = face.get("BoundingBox") if isinstance(face.get("BoundingBox"), dict) else None
                face_jpeg = _pil_crop_bbox_to_jpeg_bytes(image_bytes, bbox)
                if not face_jpeg:
                    continue
                try:
                    sr = rekognition.search_faces_by_image(
                        CollectionId=collection_id_norm,
                        Image={"Bytes": face_jpeg},
                        FaceMatchThreshold=80,
                        MaxFaces=1,
                    )
                except Exception as exc:
                    app.logger.warning("search_faces_by_image failed: %s", exc)
                    continue

                face_matches = sr.get("FaceMatches") or []
                if not face_matches:
                    continue
                best_match = face_matches[0]
                face_obj = best_match.get("Face") or {}
                actor_name = face_obj.get("ExternalImageId") or face_obj.get("ImageId") or "Actor"
                similarity = best_match.get("Similarity")
                matches.append(
                    {
                        "name": actor_name,
                        "similarity": similarity,
                        "face_id": face_obj.get("FaceId"),
                        "collection_id": collection_id_norm,
                        "bounding_box": bbox,
                        "thumbnail": _pil_crop_bbox_thumbnail_base64(image_bytes, bbox),
                    }
                )
            analysis["custom_faces"] = matches
        except Exception as e:
            app.logger.error("Custom collection matching failed: %s", e)

    # Optional: local/open-source face matching (InsightFace)
    if local_face_mode:
        ok, reason = _insightface_available()
        if not ok:
            # Keep response shape stable; put a hint in logs only.
            app.logger.warning("Local face mode requested but unavailable: %s", reason)
        else:
            try:
                analyzer = _get_insightface_analyzer()
                gallery = _local_gallery_load()
                if isinstance(gallery.get("actors"), dict) and len(gallery.get("actors")) > 0:
                    img = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))  # type: ignore
                    faces = analyzer.get(img)
                    results: List[Dict[str, Any]] = []
                    for f in faces:
                        emb = getattr(f, "embedding", None)
                        bbox = getattr(f, "bbox", None)
                        if emb is None or bbox is None:
                            continue
                        try:
                            emb_vec = np.asarray(emb, dtype=np.float32)  # type: ignore
                        except Exception:
                            continue
                        best = _local_match_embedding(emb_vec, gallery, threshold=float(local_similarity_threshold))
                        if not best:
                            continue

                        # Convert bbox pixels to fractional for UI consistency
                        try:
                            h, w = img.shape[0], img.shape[1]
                            x1, y1, x2, y2 = [float(v) for v in list(bbox)]
                            frac_bbox = {
                                "Left": max(0.0, min(1.0, x1 / max(1.0, w))),
                                "Top": max(0.0, min(1.0, y1 / max(1.0, h))),
                                "Width": max(0.0, min(1.0, (x2 - x1) / max(1.0, w))),
                                "Height": max(0.0, min(1.0, (y2 - y1) / max(1.0, h))),
                            }
                        except Exception:
                            frac_bbox = None

                        results.append(
                            {
                                "name": best.get("name"),
                                "similarity": best.get("similarity"),
                                "bounding_box": frac_bbox,
                                "thumbnail": _pil_crop_bbox_thumbnail_base64(image_bytes, frac_bbox),
                            }
                        )
                    analysis["local_faces"] = results
            except Exception as e:
                app.logger.error("Local face matching failed: %s", e)

    # Celebrity detection is relatively expensive; only run it when there's at least one face.
    if enable_celebrity_detection and face_details:
        try:
            min_conf = _parse_float_param(
                os.getenv("ENVID_METADATA_CELEB_MIN_CONFIDENCE"),
                default=95.0,
                min_value=0.0,
                max_value=100.0,
            )
            edge_margin = _parse_float_param(
                os.getenv("ENVID_METADATA_CELEB_EDGE_MARGIN"),
                default=0.02,
                min_value=0.0,
                max_value=0.2,
            )
            min_bbox = _parse_float_param(
                os.getenv("ENVID_METADATA_CELEB_MIN_BBOX"),
                default=0.06,
                min_value=0.0,
                max_value=0.5,
            )
            celeb_response = rekognition.recognize_celebrities(Image={"Bytes": image_bytes})
            analysis["celebrity_checked"] = True
            analysis["celebrities"] = [
                {
                    "name": celeb.get("Name"),
                    "confidence": celeb.get("MatchConfidence"),
                    "urls": (celeb.get("Urls") or [])[:3],
                    "bounding_box": ((celeb.get("Face") or {}).get("BoundingBox") if isinstance(celeb.get("Face"), dict) else None),
                    "thumbnail": _pil_crop_bbox_thumbnail_base64(
                        image_bytes,
                        (celeb.get("Face") or {}).get("BoundingBox") if isinstance(celeb.get("Face"), dict) else None,
                    ),
                }
                for celeb in celeb_response.get("CelebrityFaces", [])
                if celeb.get("Name")
                and float(celeb.get("MatchConfidence") or 0.0) >= float(min_conf)
                and not _bbox_looks_partial_or_too_small(
                    (celeb.get("Face") or {}).get("BoundingBox") if isinstance(celeb.get("Face"), dict) else None,
                    edge_margin=edge_margin,
                    min_width=min_bbox,
                    min_height=min_bbox,
                )
            ]
        except Exception as e:
            app.logger.error(f"Celebrity detection failed: {e}")

    if enable_moderation_detection:
        try:
            # Detect moderation labels (violence, gore, drugs, etc.)
            moderation_response = rekognition.detect_moderation_labels(Image={"Bytes": image_bytes}, MinConfidence=30)
            analysis["moderation"] = [
                {"name": label["Name"], "confidence": label["Confidence"]}
                for label in moderation_response.get("ModerationLabels", [])
            ]
        except Exception as e:
            app.logger.error(f"Moderation label detection failed: {e}")
    
    return analysis


def _recognize_celebrities_for_frame(frame_path: Path) -> List[Dict[str, Any]]:
    with open(frame_path, "rb") as image_file:
        image_bytes = image_file.read()

    min_conf = _parse_float_param(
        os.getenv("ENVID_METADATA_CELEB_MIN_CONFIDENCE"),
        default=95.0,
        min_value=0.0,
        max_value=100.0,
    )
    edge_margin = _parse_float_param(
        os.getenv("ENVID_METADATA_CELEB_EDGE_MARGIN"),
        default=0.02,
        min_value=0.0,
        max_value=0.2,
    )
    min_bbox = _parse_float_param(
        os.getenv("ENVID_METADATA_CELEB_MIN_BBOX"),
        default=0.06,
        min_value=0.0,
        max_value=0.5,
    )
    celeb_response = rekognition.recognize_celebrities(Image={"Bytes": image_bytes})
    return [
        {
            "name": celeb.get("Name"),
            "confidence": celeb.get("MatchConfidence"),
            "urls": (celeb.get("Urls") or [])[:3],
            "bounding_box": (
                (celeb.get("Face") or {}).get("BoundingBox") if isinstance(celeb.get("Face"), dict) else None
            ),
            "thumbnail": _pil_crop_bbox_thumbnail_base64(
                image_bytes,
                (celeb.get("Face") or {}).get("BoundingBox") if isinstance(celeb.get("Face"), dict) else None,
            ),
        }
        for celeb in celeb_response.get("CelebrityFaces", [])
        if celeb.get("Name")
        and float(celeb.get("MatchConfidence") or 0.0) >= float(min_conf)
        and not _bbox_looks_partial_or_too_small(
            (celeb.get("Face") or {}).get("BoundingBox") if isinstance(celeb.get("Face"), dict) else None,
            edge_margin=edge_margin,
            min_width=min_bbox,
            min_height=min_bbox,
        )
    ]


def _list_matches_terms(values: List[str], query_terms: List[str]) -> bool:
    """Return True if all query terms match as word-prefixes in the provided list of strings."""
    if not query_terms:
        return True
    if not values:
        return False
    haystack = " ".join(values)
    return _chunk_matches_terms(haystack, query_terms)


def _extract_audio_and_transcribe(video_path: Path, temp_dir: Path, job_id: str) -> str:
    """Extract audio and transcribe using AWS Transcribe."""
    # Extract audio
    audio_path = temp_dir / "audio.wav"
    extract_cmd = [
        FFMPEG_PATH, "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-af", "loudnorm",
        "-acodec", "pcm_s16le",
        str(audio_path), "-y"
    ]
    subprocess.run(extract_cmd, capture_output=True, check=True)
    
    # Resolve regions for S3 and Transcribe
    bucket_region = _detect_bucket_region(SEMANTIC_SEARCH_BUCKET)
    if not bucket_region:
        app.logger.warning(
            "Could not detect region for bucket %s; falling back to AWS_REGION %s",
            SEMANTIC_SEARCH_BUCKET,
            DEFAULT_AWS_REGION,
        )
    s3_region = bucket_region or DEFAULT_AWS_REGION
    use_accelerate = _env_truthy(os.getenv("ENVID_METADATA_S3_ACCELERATE"), default=False)
    s3_control_client = boto3.client("s3", region_name=s3_region)
    s3_upload_client = (
        boto3.client(
            "s3",
            region_name=s3_region,
            config=BotoConfig(s3={"use_accelerate_endpoint": True}),
        )
        if use_accelerate
        else s3_control_client
    )
    transcribe_region = _resolve_transcribe_region(bucket_region)
    transcribe_client = boto3.client("transcribe", region_name=transcribe_region)
    app.logger.info(
        "Semantic search using S3 region %s and Transcribe region %s",
        s3_region,
        transcribe_region,
    )

    # Upload to S3
    s3_key = f"transcribe/{job_id}/audio.wav"
    try:
        s3_control_client.head_bucket(Bucket=SEMANTIC_SEARCH_BUCKET)
    except ClientError:
        create_kwargs: Dict[str, Any] = {"Bucket": SEMANTIC_SEARCH_BUCKET}
        if s3_region != "us-east-1":
            create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": s3_region}
        s3_control_client.create_bucket(**create_kwargs)
        bucket_region = bucket_region or s3_region
    
    s3_upload_client.upload_file(str(audio_path), SEMANTIC_SEARCH_BUCKET, s3_key)
    s3_uri = f"s3://{SEMANTIC_SEARCH_BUCKET}/{s3_key}"
    
    # Start transcription job
    transcribe_job_name = f"semantic-search-{job_id}-{uuid.uuid4().hex[:8]}"[:200]
    lang_opts_raw = (os.getenv("TRANSCRIBE_LANGUAGE_OPTIONS") or "").strip()
    if not lang_opts_raw:
        # Default: primary language is Hindi.
        # If you have frequent Hinglish, set TRANSCRIBE_LANGUAGE_OPTIONS=hi-IN,en-IN (or add en-US).
        # NOTE: must be valid Transcribe language codes.
        lang_opts_raw = "hi-IN"
    language_options = [x.strip() for x in lang_opts_raw.split(",") if x.strip()]
    seen = set()
    language_options = [x for x in language_options if not (x in seen or seen.add(x))]
    if "hi-IN" in language_options:
        language_options = ["hi-IN"] + [x for x in language_options if x != "hi-IN"]
    language_options = language_options[:5]

    transcribe_kwargs: Dict[str, Any] = {
        "TranscriptionJobName": transcribe_job_name,
        "Media": {"MediaFileUri": s3_uri},
        "MediaFormat": "wav",
    }
    if len(language_options) == 1:
        transcribe_kwargs["LanguageCode"] = language_options[0]
    else:
        transcribe_kwargs["IdentifyLanguage"] = True
        transcribe_kwargs["LanguageOptions"] = language_options

    try:
        transcribe_client.start_transcription_job(**transcribe_kwargs)
    except Exception as exc:
        # If language options contain an unsupported code, Transcribe throws a BadRequest.
        # Retry with a small safe set.
        safe_options = ["hi-IN", "en-US", "en-IN"]
        if transcribe_kwargs.get("IdentifyLanguage") and transcribe_kwargs.get("LanguageOptions"):
            app.logger.warning("Transcribe start failed; retrying with safe language options: %s", exc)
            transcribe_kwargs["LanguageOptions"] = safe_options
            transcribe_client.start_transcription_job(**transcribe_kwargs)
        else:
            raise
    
    # Wait for completion
    import time
    max_wait = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        status = transcribe_client.get_transcription_job(TranscriptionJobName=transcribe_job_name)
        job_status = status["TranscriptionJob"]["TranscriptionJobStatus"]
        
        if job_status == "COMPLETED":
            transcript_uri = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
            import requests
            transcript_data = requests.get(transcript_uri).json()
            transcript_text = transcript_data.get("results", {}).get("transcripts", [{}])[0].get("transcript", "")
            
            # Cleanup
            transcribe_client.delete_transcription_job(TranscriptionJobName=transcribe_job_name)
            s3_client.delete_object(Bucket=SEMANTIC_SEARCH_BUCKET, Key=s3_key)
            
            return transcript_text
        elif job_status == "FAILED":
            break
        
        time.sleep(5)
    
    return ""


def _extract_audio_and_transcribe_rich(
    video_path: Path,
    temp_dir: Path,
    job_id: str,
    *,
    max_words: int = 5000,
) -> Dict[str, Any]:
    """Like _extract_audio_and_transcribe but also returns word timestamps for UI preview."""
    # We re-run the same Transcribe flow, but keep the JSON for timestamps.
    # NOTE: Kept separate to avoid breaking older callers.
    # Extract audio
    audio_path = temp_dir / "audio.wav"
    audio_filter = (os.getenv("TRANSCRIBE_AUDIO_FILTER") or "enhanced").strip() or "enhanced"
    if audio_filter.lower() in {"enhanced", "denoise", "denoise+norm"}:
        # Best-effort: common speech bandpass + light denoise + normalization.
        # If the ffmpeg build lacks a filter (e.g., afftdn), we fall back to loudnorm below.
        audio_filter = "highpass=f=80,lowpass=f=8000,afftdn,loudnorm"
    extract_cmd = [
        FFMPEG_PATH, "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-af", audio_filter,
        "-acodec", "pcm_s16le",
        str(audio_path), "-y"
    ]
    try:
        subprocess.run(extract_cmd, capture_output=True, check=True)
    except Exception as exc:
        # Fallback to a conservative filter chain to avoid failing the entire job.
        app.logger.warning("Audio extract/transcode failed with filter '%s' (%s); retrying with loudnorm", audio_filter, exc)
        extract_cmd_fallback = list(extract_cmd)
        try:
            af_idx = extract_cmd_fallback.index("-af")
            extract_cmd_fallback[af_idx + 1] = "loudnorm"
        except Exception:
            pass
        subprocess.run(extract_cmd_fallback, capture_output=True, check=True)

    bucket_region = _detect_bucket_region(SEMANTIC_SEARCH_BUCKET)
    s3_region = bucket_region or DEFAULT_AWS_REGION
    s3_client = boto3.client("s3", region_name=s3_region)
    transcribe_region = _resolve_transcribe_region(bucket_region)
    transcribe_client = boto3.client("transcribe", region_name=transcribe_region)

    s3_key = f"transcribe/{job_id}/audio.wav"
    try:
        s3_client.head_bucket(Bucket=SEMANTIC_SEARCH_BUCKET)
    except ClientError:
        create_kwargs: Dict[str, Any] = {"Bucket": SEMANTIC_SEARCH_BUCKET}
        if s3_region != "us-east-1":
            create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": s3_region}
        s3_client.create_bucket(**create_kwargs)

    s3_client.upload_file(str(audio_path), SEMANTIC_SEARCH_BUCKET, s3_key)
    s3_uri = f"s3://{SEMANTIC_SEARCH_BUCKET}/{s3_key}"

    transcribe_job_name = f"semantic-search-{job_id}-{uuid.uuid4().hex[:8]}"[:200]
    lang_opts_raw = (os.getenv("TRANSCRIBE_LANGUAGE_OPTIONS") or "").strip()
    if not lang_opts_raw:
        lang_opts_raw = "hi-IN"
    language_options = [x.strip() for x in lang_opts_raw.split(",") if x.strip()]
    seen: set[str] = set()
    language_options = [x for x in language_options if not (x in seen or seen.add(x))]
    if "hi-IN" in language_options:
        language_options = ["hi-IN"] + [x for x in language_options if x != "hi-IN"]
    language_options = language_options[:5]

    transcribe_kwargs: Dict[str, Any] = {
        "TranscriptionJobName": transcribe_job_name,
        "Media": {"MediaFileUri": s3_uri},
        "MediaFormat": "wav",
    }
    if len(language_options) == 1:
        transcribe_kwargs["LanguageCode"] = language_options[0]
    else:
        transcribe_kwargs["IdentifyLanguage"] = True
        transcribe_kwargs["LanguageOptions"] = language_options

    # Speaker diarization (best-effort). Works for many multi-speaker clips.
    try:
        max_speakers = int(os.getenv("TRANSCRIBE_MAX_SPEAKERS", "5"))
    except ValueError:
        max_speakers = 5
    transcribe_kwargs["Settings"] = {
        "ShowSpeakerLabels": True,
        "MaxSpeakerLabels": max(2, min(10, max_speakers)),
    }

    vocab_name = (os.getenv("TRANSCRIBE_VOCABULARY_NAME") or "").strip()
    vocab_filter_name = (os.getenv("TRANSCRIBE_VOCABULARY_FILTER_NAME") or "").strip()
    vocab_filter_method = (os.getenv("TRANSCRIBE_VOCABULARY_FILTER_METHOD") or "").strip()
    if vocab_name:
        transcribe_kwargs["Settings"]["VocabularyName"] = vocab_name
    if vocab_filter_name:
        transcribe_kwargs["Settings"]["VocabularyFilterName"] = vocab_filter_name
    if vocab_filter_method:
        transcribe_kwargs["Settings"]["VocabularyFilterMethod"] = vocab_filter_method

    show_alts = (os.getenv("TRANSCRIBE_SHOW_ALTERNATIVES") or "").strip().lower() in {"1", "true", "yes", "y"}
    if show_alts:
        transcribe_kwargs["Settings"]["ShowAlternatives"] = True
        max_alts = _parse_int_param(os.getenv("TRANSCRIBE_MAX_ALTERNATIVES"), default=2, min_value=1, max_value=10)
        transcribe_kwargs["Settings"]["MaxAlternatives"] = max_alts

    try:
        transcribe_client.start_transcription_job(**transcribe_kwargs)
    except Exception as exc:
        safe_options = ["en-US", "en-IN", "hi-IN"]
        if transcribe_kwargs.get("IdentifyLanguage") and transcribe_kwargs.get("LanguageOptions"):
            app.logger.warning("Transcribe start failed; retrying with safe language options: %s", exc)
            transcribe_kwargs["LanguageOptions"] = safe_options
            transcribe_client.start_transcription_job(**transcribe_kwargs)
        else:
            raise

    import time
    max_wait = _parse_int_param(os.getenv("ENVID_METADATA_TRANSCRIBE_MAX_WAIT_SECONDS"), default=900, min_value=60, max_value=7200)
    start_time = time.time()
    import requests

    last_emit_pct: int | None = None

    while time.time() - start_time < max_wait:
        status = transcribe_client.get_transcription_job(TranscriptionJobName=transcribe_job_name)
        job_status = status["TranscriptionJob"]["TranscriptionJobStatus"]

        # AWS Transcribe does not expose a true percentage; emit a best-effort "time-based" progress
        # so the UI can show forward motion.
        try:
            elapsed = max(0.0, float(time.time() - start_time))
            pct = int(min(99.0, (elapsed / float(max_wait)) * 100.0))
        except Exception:
            pct = 0
        if last_emit_pct is None or pct >= last_emit_pct + 2:
            last_emit_pct = pct
            _job_update(job_id, message=f"Transcribing audio ({pct}%)")
            _job_step_update(job_id, "transcribe", status="running", percent=pct, message="Transcribing")
        if job_status == "COMPLETED":
            transcript_uri = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
            detected_language_code = status["TranscriptionJob"].get("LanguageCode")
            transcript_data = requests.get(transcript_uri).json()
            transcript_text = transcript_data.get("results", {}).get("transcripts", [{}])[0].get("transcript", "")
            speaker_map = _speaker_by_start_time(transcript_data)
            words = _parse_transcribe_words(transcript_data, max_words=max_words, speaker_map=speaker_map)
            segments = _transcript_segments_from_words(words, **_transcribe_segment_params_from_env())

            transcribe_client.delete_transcription_job(TranscriptionJobName=transcribe_job_name)
            s3_client.delete_object(Bucket=SEMANTIC_SEARCH_BUCKET, Key=s3_key)

            return {
                "text": transcript_text,
                "words": words,
                "segments": segments,
                "language_code": detected_language_code or (language_options[0] if language_options else "en-US"),
            }

        if job_status == "FAILED":
            break
        time.sleep(5)

    fallback_lang = (language_options[0] if ("language_options" in locals() and language_options) else "en-US")
    return {"text": "", "words": [], "segments": [], "language_code": fallback_lang}


def _detect_dominant_languages_from_text(text: str) -> List[Dict[str, Any]]:
    t = (text or "").strip()
    if not t:
        return []
    # Comprehend's Text limit is small; keep a short prefix.
    snippet = t[:4500]
    try:
        resp = comprehend.detect_dominant_language(Text=snippet)
        langs = resp.get("Languages") or []
        parsed: List[Dict[str, Any]] = []
        for l in langs:
            code = (l.get("LanguageCode") or "").strip()
            try:
                score = float(l.get("Score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if not code:
                continue
            parsed.append({"language_code": code, "score": score})
        parsed.sort(key=lambda x: (x.get("score") or 0.0), reverse=True)
        return parsed[:5]
    except Exception as exc:
        app.logger.warning("Language detection failed: %s", exc)
        return []


def _extract_text_from_bedrock_response(response_body: Dict[str, Any]) -> str:
    if not response_body:
        return ""
    for key in ("generation", "output", "result", "message"):
        if key in response_body and isinstance(response_body[key], str):
            return response_body[key]
    if "outputs" in response_body and isinstance(response_body["outputs"], list):
        parts: list[str] = []
        for item in response_body["outputs"]:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    if "content" in response_body:
        content = response_body.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
    return ""


def _translate_to_english(text: str, *, source_language_code: str | None) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    src = (source_language_code or "").strip()
    if src and "-" in src:
        src = src.split("-", 1)[0]
    src = src.lower()
    if not src or src == "en":
        return raw

    # TranslateText has a request size limit; keep chunks small.
    max_chars = _parse_int_param(os.getenv("ENVID_METADATA_TRANSLATE_CHUNK_CHARS"), default=3500, min_value=500, max_value=9000)
    out_parts: list[str] = []
    i = 0
    while i < len(raw):
        chunk = raw[i : i + max_chars]
        i += max_chars
        try:
            resp = translate.translate_text(
                Text=chunk,
                SourceLanguageCode=src,
                TargetLanguageCode="en",
            )
            out_parts.append((resp.get("TranslatedText") or "").strip())
        except Exception as exc:
            app.logger.warning("Translate failed (src=%s): %s", src, exc)
            # Best-effort fallback: keep the original chunk.
            out_parts.append(chunk)

    return "\n".join([p for p in out_parts if p])


def _translate_segments_to_english(
    segments: List[Dict[str, Any]],
    *,
    source_language_code: str | None,
) -> List[Dict[str, Any]]:
    """Translate transcript segments to English while preserving timing.

    This is best-effort: if translation fails, falls back to the original segment text.
    """
    if not segments:
        return []

    src = (source_language_code or "").strip()
    if src and "-" in src:
        src = src.split("-", 1)[0]
    src = src.lower()
    if not src or src == "en":
        # Already English; just normalize.
        out: List[Dict[str, Any]] = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            out_seg = dict(seg)
            out_seg["text"] = _normalize_transcript_basic(str(out_seg.get("text") or ""))
            out.append(out_seg)
        return out

    out_segments: List[Dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        original_text = (seg.get("text") or "").strip()
        translated = _translate_to_english(original_text, source_language_code=source_language_code)
        out_seg = dict(seg)
        out_seg["text"] = _normalize_transcript_basic(translated or original_text)
        out_segments.append(out_seg)
    return out_segments


def _comprehend_insights_from_text(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {"entities": [], "key_phrases": [], "sentiment": None}

    snippet = raw[:4500]
    try:
        entities_resp = comprehend.detect_entities(Text=snippet, LanguageCode="en")
        phrases_resp = comprehend.detect_key_phrases(Text=snippet, LanguageCode="en")
        sentiment_resp = comprehend.detect_sentiment(Text=snippet, LanguageCode="en")

        entities = [
            {"text": e.get("Text"), "type": e.get("Type"), "score": e.get("Score")}
            for e in (entities_resp.get("Entities") or [])
            if e.get("Text")
        ]
        key_phrases = [
            {"text": p.get("Text"), "score": p.get("Score")}
            for p in (phrases_resp.get("KeyPhrases") or [])
            if p.get("Text")
        ]
        sentiment = {
            "label": sentiment_resp.get("Sentiment"),
            "scores": sentiment_resp.get("SentimentScore"),
        }
        return {
            "entities": entities[:50],
            "key_phrases": key_phrases[:50],
            "sentiment": sentiment,
        }
    except Exception as exc:
        app.logger.warning("Comprehend insights failed: %s", exc)
        return {"entities": [], "key_phrases": [], "sentiment": None}


def _bedrock_summary(text: str, *, title: str | None = None) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {"summary": "", "ssml": None}

    model_id = (os.getenv("ENVID_METADATA_SUMMARY_MODEL_ID") or "meta.llama3-70b-instruct-v1:0").strip()
    max_gen_len = _parse_int_param(os.getenv("ENVID_METADATA_SUMMARY_MAX_TOKENS"), default=240, min_value=64, max_value=1024)
    temperature = float(os.getenv("ENVID_METADATA_SUMMARY_TEMPERATURE") or 0.4)
    top_p = float(os.getenv("ENVID_METADATA_SUMMARY_TOP_P") or 0.9)

    prompt_title = (title or "").strip()
    prompt = (
        "You are a media metadata assistant. Write a concise summary (max 5 sentences) of the content. "
        "Avoid spoilers if possible.\n\n"
        + (f"Title: {prompt_title}\n\n" if prompt_title else "")
        + f"Transcript/context:\n{raw[:12000]}\n\nSummary:\n"
    )

    try:
        resp = bedrock_runtime.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "prompt": prompt,
                    "max_gen_len": max_gen_len,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            ),
        )
        payload = json.loads(resp["body"].read())
        summary = _extract_text_from_bedrock_response(payload).strip()
        if not summary:
            summary = (payload.get("generation") or "").strip()
        ssml = f"<speak><p>{summary}</p></speak>" if summary else None
        return {"summary": summary, "ssml": ssml}
    except Exception as exc:
        app.logger.warning("Bedrock summary failed: %s", exc)
        # Best-effort heuristic fallback
        summary = raw[:400].strip()
        if len(raw) > 400:
            summary = summary.rstrip() + "…"
        ssml = f"<speak><p>{summary}</p></speak>" if summary else None
        return {"summary": summary, "ssml": ssml}


def _polly_synthesize_to_s3(*, video_id: str, ssml: str) -> Dict[str, Any] | None:
    text = (ssml or "").strip()
    if not text:
        return None

    voice_id = (os.getenv("ENVID_METADATA_POLLY_VOICE_ID") or "Joanna").strip()
    bucket = _metadata_artifacts_s3_bucket()
    key = f"envid-metadata/polly/{video_id}/summary.mp3"
    try:
        resp = polly.synthesize_speech(
            Engine="neural",
            OutputFormat="mp3",
            TextType="ssml",
            VoiceId=voice_id,
            Text=text,
        )
        stream = resp.get("AudioStream")
        if not stream:
            return None
        audio_bytes = stream.read()
        if not audio_bytes:
            return None
        _s3_put_bytes(bucket=bucket, key=key, body=audio_bytes, content_type="audio/mpeg")
        return {"bucket": bucket, "key": key, "s3_uri": f"s3://{bucket}/{key}", "voice_id": voice_id}
    except Exception as exc:
        app.logger.warning("Polly synthesis failed: %s", exc)
        return None


def _transcribe_s3_media_rich(
    *,
    media_bucket: str,
    media_key: str,
    job_id: str,
    max_words: int = 5000,
) -> Dict[str, Any]:
    """Run AWS Transcribe directly on an S3 media object, then store the raw transcript JSON to S3."""

    bucket_region = _detect_bucket_region(media_bucket)
    transcribe_region = _resolve_transcribe_region(bucket_region)
    transcribe_client = boto3.client("transcribe", region_name=transcribe_region)

    media_uri = f"s3://{media_bucket}/{media_key}"
    ext = Path(media_key).suffix.lower().lstrip(".")
    media_format = ext if ext in {"mp4", "mp3", "wav", "m4a", "flac"} else "mp4"

    transcribe_job_name = f"envid-metadata-{job_id}-{uuid.uuid4().hex[:8]}"[:200]
    lang_opts_raw = (os.getenv("TRANSCRIBE_LANGUAGE_OPTIONS") or "").strip() or "hi-IN"
    language_options = [x.strip() for x in lang_opts_raw.split(",") if x.strip()]
    seen: set[str] = set()
    language_options = [x for x in language_options if not (x in seen or seen.add(x))][:5]

    transcribe_kwargs: Dict[str, Any] = {
        "TranscriptionJobName": transcribe_job_name,
        "Media": {"MediaFileUri": media_uri},
        "MediaFormat": media_format,
    }
    if len(language_options) == 1:
        transcribe_kwargs["LanguageCode"] = language_options[0]
    else:
        transcribe_kwargs["IdentifyLanguage"] = True
        transcribe_kwargs["LanguageOptions"] = language_options

    try:
        max_speakers = int(os.getenv("TRANSCRIBE_MAX_SPEAKERS", "5"))
    except ValueError:
        max_speakers = 5
    transcribe_kwargs["Settings"] = {
        "ShowSpeakerLabels": True,
        "MaxSpeakerLabels": max(2, min(10, max_speakers)),
    }

    vocab_name = (os.getenv("TRANSCRIBE_VOCABULARY_NAME") or "").strip()
    vocab_filter_name = (os.getenv("TRANSCRIBE_VOCABULARY_FILTER_NAME") or "").strip()
    vocab_filter_method = (os.getenv("TRANSCRIBE_VOCABULARY_FILTER_METHOD") or "").strip()
    if vocab_name:
        transcribe_kwargs["Settings"]["VocabularyName"] = vocab_name
    if vocab_filter_name:
        transcribe_kwargs["Settings"]["VocabularyFilterName"] = vocab_filter_name
    if vocab_filter_method:
        transcribe_kwargs["Settings"]["VocabularyFilterMethod"] = vocab_filter_method

    show_alts = (os.getenv("TRANSCRIBE_SHOW_ALTERNATIVES") or "").strip().lower() in {"1", "true", "yes", "y"}
    if show_alts:
        transcribe_kwargs["Settings"]["ShowAlternatives"] = True
        max_alts = _parse_int_param(os.getenv("TRANSCRIBE_MAX_ALTERNATIVES"), default=2, min_value=1, max_value=10)
        transcribe_kwargs["Settings"]["MaxAlternatives"] = max_alts

    _job_update(job_id, progress=72, message="Transcribe: starting")
    try:
        transcribe_client.start_transcription_job(**transcribe_kwargs)
    except Exception as exc:
        safe_options = ["hi-IN", "en-US", "en-IN"]
        if transcribe_kwargs.get("IdentifyLanguage") and transcribe_kwargs.get("LanguageOptions"):
            app.logger.warning("Transcribe start failed; retrying with safe language options: %s", exc)
            transcribe_kwargs["LanguageOptions"] = safe_options
            transcribe_client.start_transcription_job(**transcribe_kwargs)
        else:
            raise

    max_wait = _parse_int_param(os.getenv("ENVID_METADATA_TRANSCRIBE_MAX_WAIT_SECONDS"), default=900, min_value=60, max_value=7200)
    start_t = time.monotonic()
    import requests

    while time.monotonic() - start_t < max_wait:
        status = transcribe_client.get_transcription_job(TranscriptionJobName=transcribe_job_name)
        job = status.get("TranscriptionJob") or {}
        job_status = (job.get("TranscriptionJobStatus") or "").strip().upper()
        if job_status == "COMPLETED":
            transcript_uri = ((job.get("Transcript") or {}).get("TranscriptFileUri") or "").strip()
            detected_language_code = (job.get("LanguageCode") or "").strip() or None
            transcript_data = requests.get(transcript_uri, timeout=30).json() if transcript_uri else {}

            transcript_text = transcript_data.get("results", {}).get("transcripts", [{}])[0].get("transcript", "")
            speaker_map = _speaker_by_start_time(transcript_data)
            words = _parse_transcribe_words(transcript_data, max_words=max_words, speaker_map=speaker_map)
            segments = _transcript_segments_from_words(words, **_transcribe_segment_params_from_env())

            # Persist raw transcript JSON to S3
            out_bucket = _metadata_artifacts_s3_bucket()
            out_key = f"envid-metadata/transcribe/{job_id}/transcript.json"
            try:
                _s3_put_bytes(
                    bucket=out_bucket,
                    key=out_key,
                    body=json.dumps(transcript_data, ensure_ascii=False, indent=2).encode("utf-8"),
                    content_type="application/json",
                )
            except Exception as exc:
                app.logger.warning("Failed to persist transcript JSON to S3: %s", exc)

            try:
                transcribe_client.delete_transcription_job(TranscriptionJobName=transcribe_job_name)
            except Exception:
                pass

            return {
                "text": transcript_text,
                "words": words,
                "segments": segments,
                "language_code": detected_language_code or (language_options[0] if language_options else "en-US"),
                "transcript_s3": {"bucket": out_bucket, "key": out_key, "s3_uri": f"s3://{out_bucket}/{out_key}"},
            }

        if job_status == "FAILED":
            break
        time.sleep(5)

    try:
        transcribe_client.delete_transcription_job(TranscriptionJobName=transcribe_job_name)
    except Exception:
        pass
    return {"text": "", "words": [], "segments": [], "language_code": (language_options[0] if language_options else "en-US"), "transcript_s3": None}


def _top_n_by_score(items: List[Tuple[str, float]], *, limit: int) -> List[Dict[str, Any]]:
    ranked = sorted(items, key=lambda pair: (pair[1], pair[0]), reverse=True)
    return [{"name": name, "score": score} for name, score in ranked[: max(0, limit)]]


def _generate_embedding(text: str) -> List[float]:
    """Generate embedding using Amazon Bedrock Titan Embeddings."""
    try:
        response = bedrock_runtime.invoke_model(
            modelId="amazon.titan-embed-text-v1",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text})
        )
        
        response_body = json.loads(response["body"].read())
        embedding = response_body.get("embedding", [])
        return embedding
    except Exception as e:
        app.logger.error(f"Embedding generation failed: {e}")
        return []


def _compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Compute cosine similarity between two embeddings."""
    if not embedding1 or not embedding2:
        return 0.0
    
    import math
    
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    magnitude1 = math.sqrt(sum(a * a for a in embedding1))
    magnitude2 = math.sqrt(sum(b * b for b in embedding2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def _normalize_similarity(cosine_similarity: float) -> float:
    """Normalize cosine similarity (-1..1) to a 0..1 score."""
    try:
        value = (float(cosine_similarity) + 1.0) / 2.0
    except (TypeError, ValueError):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _chunk_text_for_indexing(text: str, *, max_chars: int = 800, overlap: int = 120, max_chunks: int = 400) -> List[str]:
    """Split extracted document text into reasonably sized chunks for embedding."""
    if not text:
        return []

    # Prefer paragraph boundaries when available.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[str] = []
    step = max(1, max_chars - max(0, overlap))

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
        else:
            start = 0
            while start < len(paragraph):
                piece = paragraph[start:start + max_chars].strip()
                if piece:
                    chunks.append(piece)
                start += step

        if len(chunks) >= max_chunks:
            break

    return chunks[:max_chunks]


def _extract_match_snippet(text: str, query_terms: List[str], *, window: int = 160) -> str | None:
    """Return a short snippet around the first literal match of any query term."""
    if not text or not query_terms:
        return None

    positions: List[int] = []
    for term in query_terms:
        term = (term or "").strip()
        if not term:
            continue
        # Match word prefixes (e.g., 'amit' should match 'amitabh').
        pattern = re.compile(rf"\b{re.escape(term)}\w*", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            positions.append(match.start())

    if not positions:
        return None

    match_pos = min(positions)
    start = max(0, match_pos - window)
    end = min(len(text), match_pos + window)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


def _chunk_matches_terms(text: str, query_terms: List[str]) -> bool:
    """Return True if all query terms appear as word prefixes in text (case-insensitive)."""
    if not query_terms:
        return True
    if not text:
        return False

    for term in query_terms:
        term = (term or "").strip()
        if not term:
            continue
        if not re.search(rf"\b{re.escape(term)}\w*", text, flags=re.IGNORECASE):
            return False
    return True


def _chunk_matches_any_terms(text: str, query_terms: List[str]) -> bool:
    """Return True if any query term appears as a word prefix in text (case-insensitive)."""
    if not query_terms:
        return True
    if not text:
        return False

    for term in query_terms:
        term = (term or "").strip()
        if not term:
            continue
        if re.search(rf"\b{re.escape(term)}\w*", text, flags=re.IGNORECASE):
            return True
    return False


def _list_matches_any_terms(values: List[str], query_terms: List[str]) -> bool:
    """Return True if any query term matches as a word-prefix in the provided list of strings."""
    if not query_terms:
        return True
    if not values:
        return False
    haystack = " ".join(values)
    return _chunk_matches_any_terms(haystack, query_terms)


def _video_intent_expansion_terms(raw_query_terms: List[str]) -> List[str]:
    """Return expanded terms for certain single-intent video queries (e.g., song/music/dance).

    This is used for keyword gating with OR-semantics (any-term), so semantic noise doesn't
    surface unrelated results when we have good lexical signals.
    """
    lowered = [(t or "").strip().lower() for t in (raw_query_terms or []) if (t or "").strip()]
    if not lowered:
        return []

    music_intent = {
        "song", "songs", "music", "dance", "dancing", "singer", "singers", "sing", "singing", "lyrics", "lyric",
    }

    # Only expand when the user's query is strongly a single intent word.
    if len(lowered) == 1 and lowered[0] in music_intent:
        # Use stems/prefixes where helpful (e.g., perform -> performer/performance).
        return [
            lowered[0],
            "music",
            "dance",
            "sing",
            "lyric",
            "melod",
            "rhythm",
            "beat",
            "perform",
            "stage",
            "show",
            "finalist",
            "judge",
        ]

    return []


def _extract_movie_titles_by_actor(full_text: str, query_terms: List[str], *, max_titles: int = 50) -> List[str]:
    """Extract movie titles from "Title — actors" style lists where actors match query terms.

    Designed for PDF-extracted content where line breaks can be messy.
    """
    if not full_text or not query_terms:
        return []

    # Normalize whitespace so wrapped PDF lines don't break patterns.
    normalized = re.sub(r"\s+", " ", full_text)

    # Common pattern: "12. Movie Title — Actor1, Actor2".
    # Use a lookahead for the next numbered item boundary.
    pattern = re.compile(
        r"\b\d{1,3}\.\s+([^—]{2,120}?)\s+—\s+(.+?)(?=\s+\d{1,3}\.\s+|$)",
        re.UNICODE,
    )

    # Build prefix matchers for each term (e.g., 'amit' matches 'Amitabh').
    term_patterns = [re.compile(rf"\b{re.escape(term)}\w*", re.IGNORECASE) for term in query_terms if term.strip()]
    if not term_patterns:
        return []

    titles: List[str] = []
    seen = set()

    for match in pattern.finditer(normalized):
        raw_title = (match.group(1) or "").strip()
        actors = (match.group(2) or "").strip()

        # Clean up title noise.
        title = re.sub(r"\s+", " ", raw_title)
        title = title.strip("-–— ")
        if not title:
            continue

        if not all(p.search(actors) for p in term_patterns):
            continue

        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        titles.append(title)

        if len(titles) >= max_titles:
            break

    return titles


def _parse_int_param(
    raw_value: Any,
    *,
    default: int,
    min_value: int,
    max_value: int,
) -> int:
    try:
        value = int(str(raw_value).strip())
    except Exception:
        return default
    return max(min_value, min(max_value, value))


def _parse_float_param(
    raw_value: Any,
    *,
    default: float,
    min_value: float,
    max_value: float,
) -> float:
    try:
        value = float(str(raw_value).strip())
    except Exception:
        return default
    return max(min_value, min(max_value, value))


def _process_stored_video(
    *,
    video_id: str,
    stored_video_path: Path,
    video_title: str,
    video_description: str,
    original_filename: str,
    frame_interval_seconds: int = 10,
    max_frames_to_analyze: int = 1000,
    face_recognition_mode: str | None = None,
    collection_id: str | None = None,
    local_similarity_threshold: float = 0.35,
) -> Dict[str, Any]:
    """Re-process a video already stored on disk and return updated fields."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"envid_metadata_reprocess_{video_id}_"))
    try:
        working_video_path = temp_dir / "video.mp4"
        shutil.copy2(stored_video_path, working_video_path)

        s3_video_key: str | None = None
        s3_video_uri: str | None = None
        upload_videos_to_s3 = _env_truthy(os.getenv("ENVID_METADATA_UPLOAD_VIDEOS_TO_S3"), default=True)
        if upload_videos_to_s3:
            s3_video_key, s3_video_uri = _upload_video_to_s3_or_raise(
                video_id=video_id,
                local_path=working_video_path,
                original_filename=original_filename,
            )

        duration_seconds = _probe_video_duration_seconds(working_video_path)

        preset_default = _features_preset_default_enabled()
        enable_labels = _feature_enabled("LABELS", default=preset_default)
        enable_text = _feature_enabled("TEXT", default=preset_default)
        enable_moderation = _feature_enabled("MODERATION", default=preset_default)
        enable_faces = _feature_enabled("FACES", default=preset_default)
        enable_celebrities = _feature_enabled("CELEBRITIES", default=preset_default)
        enable_transcribe = _feature_enabled("TRANSCRIBE", default=preset_default)
        enable_embedding = _feature_enabled("EMBEDDING", default=preset_default)

        # Dependencies: celebrity detection needs faces; face matching modes need faces.
        if enable_celebrities:
            enable_faces = True
        if face_recognition_mode in {"aws_collection", "local"}:
            enable_faces = True

        if not frame_interval_seconds or int(frame_interval_seconds) <= 0:
            frame_interval_seconds = _default_frame_interval_seconds(duration_seconds)

        # Extract frames
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()
        frame_interval_seconds = _parse_int_param(
            frame_interval_seconds,
            default=10,
            min_value=1,
            max_value=30,
        )
        frames = _extract_video_frames(working_video_path, frames_dir, interval=frame_interval_seconds)

        # Analyze frames with Rekognition
        all_emotions: List[str] = []

        label_stats: Dict[str, Dict[str, float]] = {}
        text_stats: Dict[str, int] = {}
        celeb_stats: Dict[str, float] = {}
        custom_face_stats: Dict[str, float] = {}
        local_face_stats: Dict[str, float] = {}
        moderation_stats: Dict[str, float] = {}
        gender_counts: Dict[str, int] = {}
        age_ranges: List[Tuple[int, int]] = []

        frames_metadata: List[Dict[str, Any]] = []

        max_frames_to_analyze = _parse_int_param(
            max_frames_to_analyze,
            default=1000,
            min_value=1,
            max_value=10000,
        )
        frames_to_process = frames[:max_frames_to_analyze]
        total = len(frames_to_process)
        if total <= 0:
            raise RuntimeError("No frames extracted")

        max_workers = _parse_int_param(
            os.getenv("ENVID_METADATA_FRAME_WORKERS"),
            default=4,
            min_value=1,
            max_value=16,
        )
        max_label_frames = _parse_int_param(
            os.getenv("ENVID_METADATA_MAX_LABEL_FRAMES"),
            default=6,
            min_value=0,
            max_value=200,
        )
        max_text_frames = _parse_int_param(
            os.getenv("ENVID_METADATA_MAX_TEXT_FRAMES"),
            default=4,
            min_value=0,
            max_value=200,
        )
        max_moderation_frames = _parse_int_param(
            os.getenv("ENVID_METADATA_MAX_MODERATION_FRAMES"),
            default=4,
            min_value=0,
            max_value=200,
        )
        max_face_frames = _parse_int_param(
            os.getenv("ENVID_METADATA_MAX_FACE_FRAMES"),
            default=0,
            min_value=0,
            max_value=200,
        )

        if not enable_labels:
            max_label_frames = 0
        if not enable_text:
            max_text_frames = 0
        if not enable_moderation:
            max_moderation_frames = 0

        label_indices = _pick_evenly_spaced_indices(total, max_label_frames) if max_label_frames > 0 else set()
        text_indices = _pick_evenly_spaced_indices(total, max_text_frames) if max_text_frames > 0 else set()
        moderation_indices = (
            _pick_evenly_spaced_indices(total, max_moderation_frames) if max_moderation_frames > 0 else set()
        )
        if enable_faces:
            face_indices = (
                set(range(total)) if max_face_frames == 0 else _pick_evenly_spaced_indices(total, max_face_frames)
            )
        else:
            face_indices = set()

        if face_recognition_mode == "aws_collection":
            face_indices = set(range(total))

        analyses: List[Dict[str, Any]] = [
            {
                "labels": [],
                "text": [],
                "faces": [],
                "celebrities": [],
                "moderation": [],
                "custom_faces": [],
                "local_faces": [],
                "celebrity_checked": False,
            }
            for _ in range(total)
        ]

        def _analyze_one(idx: int, frame_path: Path) -> Dict[str, Any]:
            return _analyze_frame_with_rekognition(
                frame_path,
                collection_id=collection_id if (face_recognition_mode == "aws_collection") else None,
                local_face_mode=bool(face_recognition_mode == "local"),
                local_similarity_threshold=local_similarity_threshold,
                enable_label_detection=idx in label_indices,
                enable_text_detection=idx in text_indices,
                enable_face_detection=idx in face_indices,
                enable_moderation_detection=idx in moderation_indices,
                enable_celebrity_detection=False,
            )

        with ThreadPoolExecutor(max_workers=min(max_workers, total)) as executor:
            future_to_idx = {
                executor.submit(_analyze_one, idx, frame_path): idx
                for idx, frame_path in enumerate(frames_to_process)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    analyses[idx] = future.result()
                except Exception as exc:
                    app.logger.error("Frame analysis failed (%s): %s", frames_to_process[idx], exc)

        max_celebrity_frames = _parse_int_param(
            os.getenv("ENVID_METADATA_MAX_CELEBRITY_FRAMES"),
            default=8,
            min_value=0,
            max_value=200,
        )
        if not enable_celebrities:
            max_celebrity_frames = 0
        if max_celebrity_frames > 0:
            candidate_indices = [idx for idx, a in enumerate(analyses) if (a.get("faces") or [])]
            if candidate_indices:
                pick = _pick_evenly_spaced_indices(
                    len(candidate_indices),
                    min(int(max_celebrity_frames), len(candidate_indices)),
                )
                chosen_indices = [candidate_indices[i] for i in sorted(pick)]
                with ThreadPoolExecutor(max_workers=min(max_workers, len(chosen_indices))) as executor:
                    future_to_idx = {
                        executor.submit(_recognize_celebrities_for_frame, frames_to_process[idx]): idx
                        for idx in chosen_indices
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            analyses[idx]["celebrities"] = future.result()
                            analyses[idx]["celebrity_checked"] = True
                        except Exception as exc:
                            app.logger.error("Celebrity detection failed (%s): %s", frames_to_process[idx], exc)

        for i, frame in enumerate(frames_to_process):
            analysis = analyses[i]
            timestamp = _frame_timestamp_seconds(frame)

            # Aggregate labels with confidence
            for label in analysis.get("labels", []):
                name = (label.get("name") or "").strip()
                if not name:
                    continue
                confidence = float(label.get("confidence") or 0.0)
                stat = label_stats.setdefault(name, {"count": 0.0, "max": 0.0, "sum": 0.0})
                stat["count"] += 1.0
                stat["sum"] += confidence
                stat["max"] = max(stat["max"], confidence)

            # Aggregate text (prefer LINE)
            for txt in analysis.get("text", []):
                value = (txt.get("text") or "").strip()
                if not value:
                    continue
                if txt.get("type") != "LINE":
                    continue
                text_stats[value] = text_stats.get(value, 0) + 1

            # Aggregate faces
            for face in analysis.get("faces", []):
                for e in (face.get("emotions") or []):
                    et = (e.get("type") or "").strip()
                    if et:
                        all_emotions.append(et)
                gender = (face.get("gender") or "").strip()
                if gender:
                    gender_counts[gender] = gender_counts.get(gender, 0) + 1
                age = face.get("age_range") or {}
                try:
                    low = int(age.get("Low"))
                    high = int(age.get("High"))
                    age_ranges.append((low, high))
                except Exception:
                    pass

            # Aggregate celebrities
            for c in (analysis.get("celebrities") or []):
                name = (c.get("name") or "").strip()
                if not name:
                    continue
                try:
                    conf = float(c.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    conf = 0.0
                celeb_stats[name] = max(celeb_stats.get(name, 0.0), conf)

            # Aggregate custom collection matches
            for c in (analysis.get("custom_faces") or []):
                name = (c.get("name") or "").strip()
                if not name:
                    continue
                try:
                    sim = float(c.get("similarity") or 0.0)
                except (TypeError, ValueError):
                    sim = 0.0
                custom_face_stats[name] = max(custom_face_stats.get(name, 0.0), sim)

            # Aggregate local matches
            for c in (analysis.get("local_faces") or []):
                name = (c.get("name") or "").strip()
                if not name:
                    continue
                try:
                    sim = float(c.get("similarity") or 0.0)
                except (TypeError, ValueError):
                    sim = 0.0
                local_face_stats[name] = max(local_face_stats.get(name, 0.0), sim)

            # Aggregate moderation
            for m in (analysis.get("moderation") or []):
                name = (m.get("name") or "").strip()
                if not name:
                    continue
                try:
                    conf = float(m.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    conf = 0.0
                moderation_stats[name] = max(moderation_stats.get(name, 0.0), conf)

            frames_metadata.append(
                {
                    "timestamp": timestamp,
                    "labels": (analysis.get("labels") or [])[:12],
                    "text": [t for t in (analysis.get("text") or []) if t.get("type") == "LINE"][:12],
                    "faces": (analysis.get("faces") or [])[:12],
                    "celebrities": (analysis.get("celebrities") or [])[:12],
                    "custom_faces": (analysis.get("custom_faces") or [])[:12],
                    "local_faces": (analysis.get("local_faces") or [])[:12],
                    "moderation": (analysis.get("moderation") or [])[:12],
                }
            )

        emotions_unique = sorted(list(set(all_emotions)))
        labels_ranked = sorted(
            (
                (
                    name,
                    int(stats.get("count", 0.0) or 0),
                    float(stats.get("max", 0.0) or 0.0),
                    float(stats.get("sum", 0.0) or 0.0) / max(1.0, float(stats.get("count", 0.0) or 1.0)),
                )
                for name, stats in label_stats.items()
            ),
            key=lambda row: (row[1], row[2], row[0]),
            reverse=True,
        )
        labels_detailed = [
            {
                "name": name,
                "occurrences": count,
                "max_confidence": max_conf,
                "avg_confidence": avg_conf,
            }
            for (name, count, max_conf, avg_conf) in labels_ranked[:50]
        ]
        text_detailed = [
            {"text": text, "occurrences": count}
            for text, count in sorted(text_stats.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:50]
        ]
        celebs_detailed = [
            {"name": name, "max_confidence": conf}
            for name, conf in sorted(celeb_stats.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:50]
        ]

        custom_faces_detailed = [
            {"name": name, "max_similarity": conf}
            for name, conf in sorted(custom_face_stats.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:50]
        ]
        local_faces_detailed = [
            {"name": name, "max_similarity": conf}
            for name, conf in sorted(local_face_stats.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:50]
        ]
        moderation_detailed = [
            {"name": name, "max_confidence": conf}
            for name, conf in sorted(moderation_stats.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:50]
        ]

        transcript = ""
        transcript_words: List[Dict[str, Any]] = []
        transcript_segments: List[Dict[str, Any]] = []
        transcript_language_code: str | None = None
        languages_detected: List[str] = []

        # Extract and transcribe audio (rich)
        if enable_transcribe:
            transcribe_rich = _extract_audio_and_transcribe_rich(working_video_path, temp_dir, video_id)
            transcript = transcribe_rich.get("text") or ""
            transcript_words = transcribe_rich.get("words") or []
            transcript_segments = transcribe_rich.get("segments") or []
            transcript_language_code = (transcribe_rich.get("language_code") or "").strip() or None
            languages_detected = _detect_dominant_languages_from_text(transcript)

        # Build metadata text for embedding
        metadata_parts = [video_title, video_description]
        if transcript:
            metadata_parts.append(f"Transcript: {transcript}")
        if labels_detailed:
            metadata_parts.append(
                "Visual elements: " + ", ".join([l["name"] for l in labels_detailed[:50] if l.get("name")])
            )
        if text_detailed:
            metadata_parts.append(
                "Text in video: " + ", ".join([t["text"] for t in text_detailed[:20] if t.get("text")])
            )
        if emotions_unique:
            metadata_parts.append(f"Emotions detected: {', '.join(emotions_unique)}")
        if celebs_detailed:
            metadata_parts.append(
                "Celebrities detected: " + ", ".join([c["name"] for c in celebs_detailed[:25] if c.get("name")])
            )
        if custom_faces_detailed:
            metadata_parts.append(
                "Cast detected (custom collection): "
                + ", ".join([c["name"] for c in custom_faces_detailed[:25] if c.get("name")])
            )
        if local_faces_detailed:
            metadata_parts.append(
                "Cast detected (local): " + ", ".join([c["name"] for c in local_faces_detailed[:25] if c.get("name")])
            )
        if moderation_detailed:
            metadata_parts.append(
                "Moderation labels: " + ", ".join([m["name"] for m in moderation_detailed[:30] if m.get("name")])
            )
        metadata_text = " ".join([part for part in metadata_parts if part])

        embedding: List[float] = []
        if enable_embedding:
            embedding = _generate_embedding(metadata_text)

        # Create thumbnail
        thumbnail_path = frames[len(frames) // 2] if frames else None
        thumbnail_base64 = None
        if thumbnail_path and thumbnail_path.exists():
            with open(thumbnail_path, "rb") as f:
                thumbnail_base64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "title": video_title,
            "description": video_description,
            "original_filename": original_filename,
            "transcript": transcript,
            "language_code": transcript_language_code,
            "languages_detected": languages_detected,
            "transcript_words": transcript_words,
            "transcript_segments": transcript_segments,
            "labels": [l["name"] for l in labels_detailed if l.get("name")][:50],
            "labels_detailed": labels_detailed,
            "text_detected": [t["text"] for t in text_detailed if t.get("text")][:20],
            "text_detailed": text_detailed,
            "emotions": emotions_unique,
            "celebrities": [c["name"] for c in celebs_detailed if c.get("name")][:50],
            "celebrities_detailed": celebs_detailed,
            "custom_faces": [c["name"] for c in custom_faces_detailed if c.get("name")][:50],
            "custom_faces_detailed": custom_faces_detailed,
            "custom_collection_id": _rekognition_collection_id_normalize(collection_id) if face_recognition_mode == "aws_collection" else None,
            "local_faces": [c["name"] for c in local_faces_detailed if c.get("name")][:50],
            "local_faces_detailed": local_faces_detailed,
            "face_recognition_mode": face_recognition_mode,
            "local_similarity_threshold": float(local_similarity_threshold) if face_recognition_mode == "local" else None,
            "moderation_labels": [m["name"] for m in moderation_detailed if m.get("name")][:50],
            "moderation_detailed": moderation_detailed,
            "faces_summary": {
                "count": sum(gender_counts.values()),
                "genders": gender_counts,
                "age_ranges": age_ranges[:200],
            },
            "embedding": embedding,
            "thumbnail": thumbnail_base64,
            "metadata_text": metadata_text,
            "frame_count": len(frames),
            "frames_analyzed": len(frames_metadata),
            "frame_interval_seconds": frame_interval_seconds,
            "duration_seconds": duration_seconds,
            "frames": frames_metadata,
            "s3_video_key": s3_video_key,
            "s3_video_uri": s3_video_uri,
            "reprocessed_at": datetime.utcnow().isoformat(),
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route("/upload-video", methods=["POST"])
def upload_video() -> Any:
    """Upload and index a video for semantic search."""
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

    # Frame interval is now auto-managed by the backend.
    # We intentionally ignore any client-provided value.
    frame_interval_seconds = 0
    max_frames_raw = (request.form.get("max_frames_to_analyze") or "").strip()
    max_frames_to_analyze = (
        _parse_int_param(max_frames_raw, default=1000, min_value=1, max_value=10000)
        if max_frames_raw
        else 1000
    )
    face_recognition_mode = (request.form.get("face_recognition_mode") or "").strip() or None
    collection_id = (request.form.get("collection_id") or "").strip() or None

    try:
        video_path = temp_dir / "video.mp4"
        video_file.save(str(video_path))
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify({"error": f"Failed to save upload: {str(exc)}"}), 500

    _job_init(job_id, title=video_title)
    _job_update(job_id, status="processing", progress=1, message="Upload received")

    def _worker() -> None:
        try:
            # Validate + transcode locally (if needed) before uploading to S3.
            upload_path = _ensure_rekognition_compatible_before_upload(
                video_path=video_path,
                temp_dir=temp_dir,
                job_id=job_id,
            )

            # Verify technical metadata for the file we will upload.
            duration_seconds_local = _probe_video_duration_seconds(upload_path)
            technical_ffprobe = _probe_technical_metadata(upload_path)
            technical_mediainfo = _probe_mediainfo_metadata(upload_path)
            technical_verification = _verify_technical_metadata(ffprobe=technical_ffprobe, mediainfo=technical_mediainfo)

            _job_update(job_id, progress=2, message="Uploading video to S3")

            # Force a .mp4 key for cloud-only Rekognition compatibility.
            original_filename_for_upload = str(Path(original_filename).with_suffix(".mp4").name)

            s3_key, s3_uri = _upload_video_to_s3_or_raise(
                video_id=job_id,
                local_path=upload_path,
                original_filename=original_filename_for_upload,
                job_id=job_id,
            )
            _job_update(job_id, s3_video_key=s3_key, s3_video_uri=s3_uri)

            # Rekognition analysis uses the S3 video source (Rekognition Video) for required/minimal.
            # Local frame extraction is used only for non-Rekognition needs (e.g., thumbnails) and legacy/full profile.
            _process_video_job(
                job_id=job_id,
                temp_dir=temp_dir,
                video_path=upload_path,
                video_title=video_title,
                video_description=video_description,
                original_filename=original_filename_for_upload,
                frame_interval_seconds=frame_interval_seconds,
                max_frames_to_analyze=max_frames_to_analyze,
                face_recognition_mode=face_recognition_mode,
                collection_id=collection_id,
                preexisting_s3_video_key=s3_key,
                preexisting_s3_video_uri=s3_uri,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    thread = threading.Thread(
        target=_worker,
        daemon=True,
    )
    thread.start()

    with JOBS_LOCK:
        job = JOBS.get(job_id)
    return jsonify({"job_id": job_id, "job": job}), 202


@app.route("/presign-upload-video", methods=["POST"])
def presign_upload_video() -> Any:
    """Create a presigned S3 PUT URL so the browser can upload directly to S3.

    This enables a "zero local bytes" architecture: the server never receives the MP4 bytes.
    """

    payload = request.get_json(silent=True) or {}
    original_filename = (payload.get("filename") or payload.get("original_filename") or "video.mp4").strip()
    if not original_filename:
        original_filename = "video.mp4"

    # Rekognition Video compatibility: we prefer an .mp4 suffix for the uploaded object.
    try:
        original_filename_for_upload = str(Path(original_filename).with_suffix(".mp4").name)
    except Exception:
        original_filename_for_upload = "video.mp4"

    job_id = str(uuid.uuid4())
    bucket = _video_s3_bucket()
    key = _video_s3_key(job_id, original_filename_for_upload)

    bucket_region = _detect_bucket_region(bucket)
    s3_region = bucket_region or DEFAULT_AWS_REGION

    client = _s3_client_for_transfer(region_name=s3_region)
    expires = _parse_int_param(
        os.getenv("ENVID_METADATA_S3_PRESIGN_SECONDS"),
        default=3600,
        min_value=60,
        max_value=86400,
    )

    # Note: we intentionally do NOT sign Content-Type here so browsers can set it freely.
    upload_url = client.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )

    return (
        jsonify(
            {
                "job_id": job_id,
                "bucket": bucket,
                "key": key,
                "s3_uri": f"s3://{bucket}/{key}",
                "method": "PUT",
                "upload_url": upload_url,
            }
        ),
        200,
    )


@app.route("/process-s3-video", methods=["POST"])
def process_s3_video() -> Any:
    """Process an existing S3 video under mediagenailab/rawvideo.

    This endpoint downloads the S3 video and runs frame extraction + per-frame analysis.
    """
    payload = request.get_json(silent=True) or {}
    s3_key = (payload.get("s3_key") or payload.get("s3_uri") or "").strip()
    if not s3_key:
        return jsonify({"error": "Missing s3_key (or s3_uri)"}), 400

    video_title = (payload.get("title") or Path(s3_key).name or "S3 Video").strip()
    video_description = (payload.get("description") or "").strip()

    # Frame interval is now auto-managed by the backend.
    # We intentionally ignore any client-provided value.
    frame_interval_seconds = 0
    max_frames_raw = str(payload.get("max_frames_to_analyze") or "").strip()
    max_frames_to_analyze = (
        _parse_int_param(max_frames_raw, default=1000, min_value=1, max_value=10000)
        if max_frames_raw
        else 1000
    )
    face_recognition_mode = (payload.get("face_recognition_mode") or "").strip() or None
    collection_id = (payload.get("collection_id") or "").strip() or None

    try:
        normalized_key = _normalize_rawvideo_key(s3_key)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    job_id = str(uuid.uuid4())
    _job_init(job_id, title=video_title)
    _job_update(job_id, status="processing", progress=1, message="S3 video queued", s3_video_key=normalized_key)

    thread = threading.Thread(
        target=_process_s3_video_job,
        kwargs={
            "job_id": job_id,
            "video_title": video_title,
            "video_description": video_description,
            "frame_interval_seconds": frame_interval_seconds,
            "s3_key": normalized_key,
            "max_frames_to_analyze": max_frames_to_analyze,
            "face_recognition_mode": face_recognition_mode,
            "collection_id": collection_id,
        },
        daemon=True,
    )
    thread.start()

    with JOBS_LOCK:
        job = JOBS.get(job_id)
    return jsonify({"job_id": job_id, "job": job}), 202


@app.route("/process-s3-video-cloud", methods=["POST"])
def process_s3_video_cloud() -> Any:
    """Process an existing allowed S3 video.

    In S3-only / Rekognition-Video mode this does NOT download the video.
    """
    payload = request.get_json(silent=True) or {}
    raw = (payload.get("s3_key") or payload.get("s3_uri") or "").strip()
    if not raw:
        return jsonify({"error": "Missing s3_key (or s3_uri)"}), 400

    try:
        bucket, key = _parse_allowed_s3_video_source(raw)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    video_title = (payload.get("title") or Path(key).name or "S3 Video").strip()
    video_description = (payload.get("description") or "").strip()

    # Frame interval is now auto-managed by the backend.
    # We intentionally ignore any client-provided value.
    frame_interval_seconds = 0

    max_frames_raw = str(payload.get("max_frames_to_analyze") or "").strip()
    max_frames_to_analyze = (
        _parse_int_param(max_frames_raw, default=1000, min_value=1, max_value=10000)
        if max_frames_raw
        else 1000
    )
    face_recognition_mode = (payload.get("face_recognition_mode") or "").strip() or None
    collection_id = (payload.get("collection_id") or "").strip() or None

    requested_job_id = (payload.get("job_id") or payload.get("id") or "").strip()
    job_id = requested_job_id if _looks_like_uuid(requested_job_id) else str(uuid.uuid4())
    _job_init(job_id, title=video_title)
    _job_update(job_id, status="processing", progress=1, message="S3 video queued", s3_video_key=key, s3_video_uri=f"s3://{bucket}/{key}")
    _job_step_update(job_id, "upload_to_s3", status="completed", percent=100, message="Using existing S3 source")

    analysis_source = (os.getenv("ENVID_METADATA_REKOGNITION_ANALYSIS_SOURCE") or "").strip().lower()
    prefer_video = _disable_per_frame_analysis() or analysis_source in {"s3_video", "video", "rekognition_video"}
    cloud_only = bool(prefer_video or (_output_profile() in {"required", "minimal"}))

    worker = _process_s3_object_job_cloud_only if cloud_only else _process_s3_object_job_download_and_analyze
    worker_kwargs: Dict[str, Any] = {
        "job_id": job_id,
        "s3_bucket": bucket,
        "s3_key": key,
        "video_title": video_title,
        "video_description": video_description,
        "frame_interval_seconds": frame_interval_seconds,
    }
    if not cloud_only:
        worker_kwargs.update(
            {
                "max_frames_to_analyze": max_frames_to_analyze,
                "face_recognition_mode": face_recognition_mode,
                "collection_id": collection_id,
            }
        )

    thread = threading.Thread(
        target=worker,
        kwargs=worker_kwargs,
        daemon=True,
    )
    thread.start()

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


@app.route("/search", methods=["POST"])
def search_videos() -> Any:
    """Search videos using semantic similarity."""
    payload = request.get_json(silent=True) or {}
    query = payload.get("query", "").strip()
    top_k = int(payload.get("top_k", 5))
    min_similarity = payload.get("min_similarity")
    if min_similarity is None:
        min_similarity = os.getenv("SEMANTIC_VIDEO_MIN_SIMILARITY", "0.5")
    try:
        min_similarity_value = float(min_similarity)
    except (TypeError, ValueError):
        return jsonify({"error": "min_similarity must be a number"}), 400
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    if not VIDEO_INDEX:
        return jsonify({"error": "No videos indexed yet. Please upload videos first."}), 400
    
    try:
        # Generate query embedding
        app.logger.info(f"Searching for: {query}")
        query_embedding = _generate_embedding(query)
        
        if not query_embedding:
            return jsonify({"error": "Failed to generate query embedding"}), 500
        
        raw_query_terms = [t for t in query.split() if t]
        raw_query_terms_count = len(raw_query_terms)
        query_terms = list(raw_query_terms)

        # Intent-based expansion for better relevance on short queries like "song".
        intent_terms = _video_intent_expansion_terms(raw_query_terms)

        # Small synonym expansion for common safety-related queries.
        lowered = {t.lower() for t in query_terms}
        if "blood" in lowered:
            # Rekognition moderation labels tend to use categories like "Violence" / "Graphic Violence Or Gore".
            query_terms = list(dict.fromkeys(query_terms + ["violence", "gore", "graphic"]))

        def moderation_match(v: Dict[str, Any]) -> bool:
            return _list_matches_terms([str(x) for x in (v.get("moderation_labels") or [])], query_terms)

        def label_match(v: Dict[str, Any]) -> bool:
            return _list_matches_terms([str(x) for x in (v.get("labels") or [])], query_terms)

        def text_match(v: Dict[str, Any]) -> bool:
            combined = " ".join([
                str(v.get("title") or ""),
                str(v.get("description") or ""),
                str(v.get("transcript") or ""),
                " ".join([str(x) for x in (v.get("text_detected") or [])]),
            ])
            return _chunk_matches_terms(combined, query_terms)

        def label_match_any(v: Dict[str, Any]) -> bool:
            if not intent_terms:
                return False
            return _list_matches_any_terms([str(x) for x in (v.get("labels") or [])], intent_terms)

        def text_match_any(v: Dict[str, Any]) -> bool:
            if not intent_terms:
                return False
            combined = " ".join([
                str(v.get("title") or ""),
                str(v.get("description") or ""),
                str(v.get("transcript") or ""),
                " ".join([str(x) for x in (v.get("text_detected") or [])]),
            ])
            return _chunk_matches_any_terms(combined, intent_terms)

        has_moderation_hits = any(moderation_match(v) for v in VIDEO_INDEX)
        has_label_hits = any(label_match(v) for v in VIDEO_INDEX)
        has_text_hits = any(text_match(v) for v in VIDEO_INDEX)

        # For certain intent queries (e.g., "song"), prefer OR-based gating on expanded terms.
        has_intent_label_hits = any(label_match_any(v) for v in VIDEO_INDEX) if intent_terms else False
        has_intent_text_hits = any(text_match_any(v) for v in VIDEO_INDEX) if intent_terms else False

        def include_video(v: Dict[str, Any]) -> bool:
            if has_intent_label_hits:
                return label_match_any(v)
            if has_intent_text_hits:
                return text_match_any(v)
            if has_moderation_hits:
                return moderation_match(v)
            if has_label_hits:
                return label_match(v)
            if has_text_hits:
                return text_match(v)
            return True

        # Compute similarities (with keyword-aware filtering first)
        all_results = []
        for video in VIDEO_INDEX:
            if not video.get("embedding"):
                continue
            if not include_video(video):
                continue

            cosine_similarity = _compute_similarity(query_embedding, video["embedding"])
            similarity = _normalize_similarity(cosine_similarity)

            labels = [str(x) for x in (video.get("labels") or [])]
            moderation_labels = [str(x) for x in (video.get("moderation_labels") or [])]
            if intent_terms:
                matched_labels = [l for l in labels if _chunk_matches_any_terms(l, intent_terms)]
            else:
                matched_labels = [l for l in labels if _chunk_matches_terms(l, query_terms)] if query_terms else labels[:10]
            matched_moderation = [m for m in moderation_labels if _chunk_matches_terms(m, query_terms)] if query_terms else moderation_labels[:10]

            all_results.append({
                "id": video["id"],
                "video_id": video["id"],
                "title": video["title"],
                "description": video["description"],
                "transcript_snippet": video["transcript"][:200] if video["transcript"] else "",
                "labels": labels[:10],
                "matched_labels": matched_labels[:10],
                "moderation_labels": moderation_labels[:10],
                "matched_moderation_labels": matched_moderation[:10],
                "emotions": video["emotions"],
                "thumbnail": _as_data_uri_jpeg(video.get("thumbnail")),
                "similarity_score": round(similarity, 4),
                "cosine_similarity": round(cosine_similarity, 4),
                "uploaded_at": video["uploaded_at"]
            })
        
        # Sort by similarity
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Filter results.
        # - Always apply min_similarity.
        # - If this search is semantic-only (no keyword hits), additionally keep only results
        #   within a small margin of the best score to avoid unrelated matches.
        semantic_only = not (has_intent_label_hits or has_intent_text_hits or has_moderation_hits or has_label_hits or has_text_hits)
        effective_min_similarity = min_similarity_value
        if semantic_only and all_results:
            # Single-word queries tend to be more specific; use a tighter margin by default.
            default_margin = "0.02" if raw_query_terms_count == 1 else "0.03"
            margin = float(os.getenv("SEMANTIC_VIDEO_TOP_SCORE_MARGIN", default_margin))
            top_score = all_results[0]["similarity_score"]
            effective_min_similarity = max(effective_min_similarity, top_score - margin)

        results = [r for r in all_results if r["similarity_score"] >= effective_min_similarity]
        
        # Log for debugging
        app.logger.info(f"Query: {query}")
        app.logger.info(f"Top scores: {[r['similarity_score'] for r in all_results[:5]]}")
        app.logger.info(
            f"Results after min_similarity={effective_min_similarity:.2f} filter: {len(results)}/{len(all_results)}"
        )
        
        return jsonify({
            "query": query,
            "results": results[:top_k],
            "total_videos": len(VIDEO_INDEX)
        }), 200
        
    except Exception as e:
        app.logger.error(f"Search failed: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Search failed: {str(e)}"}), 500


@app.route("/videos", methods=["GET"])
def list_videos() -> Any:
    """List all indexed videos."""
    videos = [
        {
            "id": v["id"],
            "title": v["title"],
            "description": v["description"],
            "labels_count": len(v["labels"]),
            "frame_count": int(v.get("frame_count", 0) or 0),
            "has_transcript": bool(v["transcript"]),
            "uploaded_at": v["uploaded_at"],
            "thumbnail": v["thumbnail"]
        }
        for v in VIDEO_INDEX
    ]
    
    return jsonify({
        "videos": videos,
        "total": len(videos)
    }), 200


@app.route("/video/<video_id>", methods=["GET"])
def get_video_details(video_id: str) -> Any:
    """Get details of a specific video."""
    video = next((v for v in VIDEO_INDEX if v["id"] == video_id), None)
    
    if not video:
        return jsonify({"error": "Video not found"}), 404

    include_frames_raw = (request.args.get("include_frames") or "").strip().lower()
    include_words_raw = (request.args.get("include_words") or "").strip().lower()
    include_frames = include_frames_raw in {"1", "true", "yes", "y", "on"}
    include_words = include_words_raw in {"1", "true", "yes", "y", "on"}

    # Opportunistic backfill for older index entries.
    if (video.get("languages_detected") is None or (isinstance(video.get("languages_detected"), list) and len(video.get("languages_detected") or []) == 0)) and (video.get("transcript") or "").strip():
        try:
            detected = _detect_dominant_languages_from_text(video.get("transcript") or "")
            if detected:
                video["languages_detected"] = detected
                # Prefer detected top language for language_code if missing.
                if not (video.get("language_code") or "").strip():
                    video["language_code"] = detected[0].get("language_code")
                _save_video_index()
        except Exception as exc:
            app.logger.warning("Language backfill failed for %s: %s", video_id, exc)

    response: Dict[str, Any] = {
        "id": video["id"],
        "title": video["title"],
        "description": video["description"],
        "thumbnail": video.get("thumbnail"),
        "uploaded_at": video.get("uploaded_at"),

        # Core metadata (backwards compatible keys)
        "transcript": video.get("transcript") or "",
        "language_code": video.get("language_code"),
        "languages_detected": video.get("languages_detected") or [],
        "labels": video.get("labels") or [],
        "text_detected": video.get("text_detected") or [],
        "emotions": video.get("emotions") or [],
        "celebrities": video.get("celebrities") or [],

        # Optional face recognition modes
        "custom_faces": video.get("custom_faces") or [],
        "custom_faces_detailed": video.get("custom_faces_detailed") or [],
        "custom_collection_id": video.get("custom_collection_id"),
        "local_faces": video.get("local_faces") or [],
        "local_faces_detailed": video.get("local_faces_detailed") or [],

        # Enriched aggregates
        "duration_seconds": video.get("duration_seconds"),
        "frame_interval_seconds": video.get("frame_interval_seconds"),
        "frame_count": int(video.get("frame_count", 0) or 0),
        "frames_analyzed": int(video.get("frames_analyzed", 0) or 0),

        "labels_detailed": video.get("labels_detailed") or [],
        "text_detailed": video.get("text_detailed") or [],
        "celebrities_detailed": video.get("celebrities_detailed") or [],
        "moderation_labels": video.get("moderation_labels") or [],
        "moderation_detailed": video.get("moderation_detailed") or [],
        "faces_summary": video.get("faces_summary") or {},

        # Timed transcript
        "transcript_segments": video.get("transcript_segments") or [],
        "transcript_words_count": int(len(video.get("transcript_words") or [])),

        "reprocessed_at": video.get("reprocessed_at"),
    }

    if include_frames:
        response["frames"] = video.get("frames") or []
    if include_words:
        response["transcript_words"] = video.get("transcript_words") or []

    # Backfill portraits in response payload (do not require a reprocess).
    try:
        bio_cache = _bio_cache_load()

        def _backfill_portrait_fields(obj: dict[str, Any]) -> dict[str, Any]:
            if not isinstance(obj, dict):
                return obj
            nm = (obj.get("name") or "").strip()
            cached = bio_cache.get(nm) if nm and isinstance(bio_cache, dict) else None
            if not isinstance(cached, dict):
                return obj
            if (obj.get("portrait_url") or "").strip():
                return obj
            url = (cached.get("portrait_url") or "").strip()
            if not url:
                return obj
            out = dict(obj)
            out["portrait_url"] = url
            out["portrait_source"] = (cached.get("portrait_source") or "").strip() or None
            out["portrait_license"] = (cached.get("portrait_license") or "").strip() or None
            out["portrait_license_url"] = (cached.get("portrait_license_url") or "").strip() or None
            out["portrait_attribution"] = (cached.get("portrait_attribution") or "").strip() or None
            return out

        # celebrities_detailed
        if isinstance(response.get("celebrities_detailed"), list):
            response["celebrities_detailed"] = [
                _backfill_portrait_fields(c) if isinstance(c, dict) else c for c in (response.get("celebrities_detailed") or [])
            ]

        # legacy celebrities list
        if isinstance(response.get("celebrities"), list):
            response["celebrities"] = [
                _backfill_portrait_fields(c) if isinstance(c, dict) else c for c in (response.get("celebrities") or [])
            ]

        # frames[*].celebrities[*]
        if include_frames:
            frames = response.get("frames")
            if isinstance(frames, list):
                new_frames: list[dict[str, Any]] = []
                for f in frames:
                    if not isinstance(f, dict):
                        continue
                    f2 = dict(f)
                    if isinstance(f2.get("celebrities"), list):
                        f2["celebrities"] = [
                            _backfill_portrait_fields(c) if isinstance(c, dict) else c for c in (f2.get("celebrities") or [])
                        ]
                    new_frames.append(f2)
                response["frames"] = new_frames
    except Exception as exc:
        app.logger.warning("Portrait backfill failed for %s: %s", video_id, exc)

    # Ensure celebrity bios meet minimum length (>= 30 words), regenerating via cache/Wikipedia if needed.
    try:
        _refresh_celebrity_bios_in_payload(response, include_frames=include_frames)
    except Exception as exc:
        app.logger.warning("Bio refresh failed for %s: %s", video_id, exc)

    return jsonify(response), 200


@app.route("/video/<video_id>/metadata-json", methods=["GET"])
def get_video_metadata_json(video_id: str) -> Any:
    """Return categorized metadata JSON.

    Query params:
    - category: one of the category keys, or "combined" (default).
    """
    video = next((v for v in VIDEO_INDEX if v.get("id") == video_id), None)
    if not video:
        return jsonify({"error": "Video not found"}), 404

    category = (request.args.get("category") or "combined").strip()

    desired_profile = _envid_metadata_output_profile()
    stored_profile = (video.get("output_profile") or "").strip().lower() if isinstance(video.get("output_profile"), str) else ""

    def _looks_full_shape() -> bool:
        cats = video.get("metadata_categories")
        return isinstance(cats, dict) and any(k.endswith("_metadata") for k in cats.keys())

    # Build/rebuild on demand if missing or profile changed.
    if (
        not isinstance(video.get("metadata_categories"), dict)
        or not isinstance(video.get("metadata_combined"), dict)
        or (stored_profile != desired_profile)
        or (desired_profile in {"required", "minimal"} and _looks_full_shape())
    ):
        try:
            video["output_profile"] = desired_profile
            categorized = _build_categorized_metadata_json(video)
            video["metadata_categories"] = categorized.get("categories")
            video["metadata_combined"] = categorized.get("combined")
            _save_video_index()
        except Exception:
            pass

    # Best-effort: persist artifacts to S3 (combined + categories + zip).
    try:
        _ensure_metadata_artifacts_on_s3(video)
    except Exception:
        pass

    # Refresh short/missing celebrity bios even when serving cached metadata.
    try:
        changed = False
        if isinstance(video.get("metadata_combined"), dict):
            changed = _refresh_celebrity_bios_in_payload(video["metadata_combined"], include_frames=True) or changed
        if isinstance(video.get("metadata_categories"), dict):
            changed = _refresh_celebrity_bios_in_payload(video["metadata_categories"], include_frames=True) or changed
        if changed:
            _save_video_index()
    except Exception:
        pass

    # Ensure synopses include senior_citizen and minimum word counts, even for cached metadata.
    try:
        src_text = str(video.get("transcript") or video.get("description") or "").strip()
        title = str(video.get("title") or "").strip() or None
        changed = False
        if isinstance(video.get("metadata_categories"), dict):
            cats = video["metadata_categories"]
            old_syn = cats.get("synopses_by_age_group")
            new_syn = _normalize_synopses_by_age_group_for_output(old_syn, source_text=src_text, title=title)
            if isinstance(new_syn, dict) and new_syn and new_syn != old_syn:
                cats["synopses_by_age_group"] = new_syn
                changed = True

        if isinstance(video.get("metadata_combined"), dict):
            comb = video["metadata_combined"]
            comb_cats = comb.get("categories") if isinstance(comb.get("categories"), dict) else None
            if isinstance(comb_cats, dict):
                old_syn = comb_cats.get("synopses_by_age_group")
                new_syn = _normalize_synopses_by_age_group_for_output(old_syn, source_text=src_text, title=title)
                if isinstance(new_syn, dict) and new_syn and new_syn != old_syn:
                    comb_cats["synopses_by_age_group"] = new_syn
                    changed = True

        if changed:
            _save_video_index()
    except Exception:
        pass

    # Ensure famous_locations has merged time_mapped + optional geocodes, even for cached metadata.
    try:
        changed = False
        transcript_segments = video.get("transcript_segments") if isinstance(video.get("transcript_segments"), list) else []
        frames = video.get("frames") if isinstance(video.get("frames"), list) else []
        raw_locations = video.get("locations") if isinstance(video.get("locations"), list) else []
        geo_cache = video.get("locations_geocoded") if isinstance(video.get("locations_geocoded"), dict) else None

        built = _build_famous_locations_payload(
            text_locations=raw_locations,
            transcript_segments=transcript_segments,
            frames=frames,
            geocode_cache=geo_cache,
        )
        if isinstance(built, dict) and built.get("locations"):
            # Persist caches on the video entry.
            old_fl = video.get("famous_locations") if isinstance(video.get("famous_locations"), dict) else None
            if old_fl != built:
                video["famous_locations"] = built
                changed = True
            new_geo = built.get("geocode_cache")
            if isinstance(new_geo, dict) and new_geo and new_geo != geo_cache:
                video["locations_geocoded"] = new_geo
                changed = True

            # Update cached metadata payloads.
            if isinstance(video.get("metadata_categories"), dict):
                cats = video["metadata_categories"]
                old = cats.get("famous_locations")
                cats["famous_locations"] = {
                    "locations": built.get("locations") or [],
                    "time_mapped": built.get("time_mapped") or [],
                    "from_transcript": built.get("from_transcript") or [],
                    "from_landmarks": built.get("from_landmarks") or [],
                }
                if old != cats.get("famous_locations"):
                    changed = True

            if isinstance(video.get("metadata_combined"), dict):
                comb = video["metadata_combined"]
                comb_cats = comb.get("categories") if isinstance(comb.get("categories"), dict) else None
                if isinstance(comb_cats, dict):
                    old = comb_cats.get("famous_locations")
                    comb_cats["famous_locations"] = {
                        "locations": built.get("locations") or [],
                        "time_mapped": built.get("time_mapped") or [],
                        "from_transcript": built.get("from_transcript") or [],
                        "from_landmarks": built.get("from_landmarks") or [],
                    }
                    if old != comb_cats.get("famous_locations"):
                        changed = True

        if changed:
            _save_video_index()
    except Exception:
        pass

    download = _env_truthy(request.args.get("download"), default=False)

    if category.lower() == "combined":
        if download:
            try:
                _ensure_metadata_artifacts_on_s3(video)
            except Exception:
                pass
            bucket = _metadata_artifacts_s3_bucket()
            key = _metadata_combined_s3_key(video_id)
            if not _s3_object_exists(bucket=bucket, key=key):
                return jsonify({"error": "Combined JSON not found in S3"}), 404
            url = _s3_presigned_download_url(
                bucket=bucket,
                key=key,
                download_name=f"{video_id}__combined.json",
                content_type="application/json",
            )
            return redirect(url, code=302)
        return jsonify(video.get("metadata_combined") or {}), 200

    categories = video.get("metadata_categories") or {}
    if category not in categories:
        return jsonify({
            "error": "Unknown category",
            "available_categories": sorted(list(categories.keys())),
        }), 400
    if download:
        try:
            _ensure_metadata_artifacts_on_s3(video)
        except Exception:
            pass
        bucket = _metadata_artifacts_s3_bucket()
        key = _metadata_category_s3_key(video_id, category)
        if not _s3_object_exists(bucket=bucket, key=key):
            return jsonify({"error": "Category JSON not found in S3"}), 404
        url = _s3_presigned_download_url(
            bucket=bucket,
            key=key,
            download_name=f"{video_id}__{category}.json",
            content_type="application/json",
        )
        return redirect(url, code=302)
    return jsonify(categories.get(category) or {}), 200


@app.route("/video/<video_id>/metadata-json.zip", methods=["GET"])
def download_video_metadata_json_zip(video_id: str) -> Any:
    """Download all category JSON files + combined JSON as a ZIP."""
    video = next((v for v in VIDEO_INDEX if v.get("id") == video_id), None)
    if not video:
        return jsonify({"error": "Video not found"}), 404

    desired_profile = _envid_metadata_output_profile()
    stored_profile = (video.get("output_profile") or "").strip().lower() if isinstance(video.get("output_profile"), str) else ""
    cats_existing = video.get("metadata_categories")
    looks_full_shape = isinstance(cats_existing, dict) and any(k.endswith("_metadata") for k in cats_existing.keys())

    # Build/rebuild on demand if missing or profile changed.
    if (
        not isinstance(video.get("metadata_categories"), dict)
        or not isinstance(video.get("metadata_combined"), dict)
        or (stored_profile != desired_profile)
        or (desired_profile in {"required", "minimal"} and looks_full_shape)
    ):
        categorized = _build_categorized_metadata_json(video)
        video["output_profile"] = desired_profile
        video["metadata_categories"] = categorized.get("categories")
        video["metadata_combined"] = categorized.get("combined")
        _save_video_index()

    # Persist artifacts to S3 (combined + categories + zip), then redirect to cloud download.
    try:
        _ensure_metadata_artifacts_on_s3(video)
    except Exception:
        pass

    # Refresh short/missing celebrity bios even when serving cached metadata.
    try:
        changed = False
        if isinstance(video.get("metadata_combined"), dict):
            changed = _refresh_celebrity_bios_in_payload(video["metadata_combined"], include_frames=True) or changed
        if isinstance(video.get("metadata_categories"), dict):
            changed = _refresh_celebrity_bios_in_payload(video["metadata_categories"], include_frames=True) or changed
        if changed:
            _save_video_index()
    except Exception:
        pass

    safe_base = str(video.get("title") or video.get("id") or "metadata")
    safe_base = re.sub(r"[^a-zA-Z0-9\-_]+", "_", safe_base).strip("_")[:80] or "metadata"
    download_name = f"{safe_base}__metadata_json.zip"

    bucket = _metadata_artifacts_s3_bucket()
    key = _metadata_zip_s3_key(video_id)
    if not _s3_object_exists(bucket=bucket, key=key):
        return jsonify({"error": "Metadata ZIP not found in S3"}), 404
    url = _s3_presigned_download_url(
        bucket=bucket,
        key=key,
        download_name=download_name,
        content_type="application/zip",
    )
    return redirect(url, code=302)


@app.route("/video/<video_id>/subtitles.srt", methods=["GET"])
def download_subtitles_srt(video_id: str) -> Any:
    bucket = _metadata_artifacts_s3_bucket()
    key = _subtitles_s3_key(video_id, lang="orig", fmt="srt")
    if not _s3_object_exists(bucket=bucket, key=key):
        return jsonify({"error": "Subtitles not found in S3"}), 404
    url = _s3_presigned_download_url(
        bucket=bucket,
        key=key,
        download_name=f"{video_id}.srt",
        content_type="application/x-subrip",
    )
    return redirect(url, code=302)


@app.route("/video/<video_id>/subtitles.vtt", methods=["GET"])
def download_subtitles_vtt(video_id: str) -> Any:
    bucket = _metadata_artifacts_s3_bucket()
    key = _subtitles_s3_key(video_id, lang="orig", fmt="vtt")
    if not _s3_object_exists(bucket=bucket, key=key):
        return jsonify({"error": "Subtitles not found in S3"}), 404
    url = _s3_presigned_download_url(
        bucket=bucket,
        key=key,
        download_name=f"{video_id}.vtt",
        content_type="text/vtt",
    )
    return redirect(url, code=302)


@app.route("/video/<video_id>/subtitles.en.srt", methods=["GET"])
def download_subtitles_en_srt(video_id: str) -> Any:
    bucket = _metadata_artifacts_s3_bucket()
    key = _subtitles_s3_key(video_id, lang="en", fmt="srt")
    if not _s3_object_exists(bucket=bucket, key=key):
        return jsonify({"error": "English subtitles not found in S3"}), 404
    url = _s3_presigned_download_url(
        bucket=bucket,
        key=key,
        download_name=f"{video_id}.en.srt",
        content_type="application/x-subrip",
    )
    return redirect(url, code=302)


@app.route("/video/<video_id>/subtitles.en.vtt", methods=["GET"])
def download_subtitles_en_vtt(video_id: str) -> Any:
    bucket = _metadata_artifacts_s3_bucket()
    key = _subtitles_s3_key(video_id, lang="en", fmt="vtt")
    if not _s3_object_exists(bucket=bucket, key=key):
        return jsonify({"error": "English subtitles not found in S3"}), 404
    url = _s3_presigned_download_url(
        bucket=bucket,
        key=key,
        download_name=f"{video_id}.en.vtt",
        content_type="text/vtt",
    )
    return redirect(url, code=302)


@app.route("/video-file/<video_id>", methods=["GET"])
def get_video_file(video_id: str) -> Any:
    """Stream the original uploaded video file for playback in the UI."""
    video = next((v for v in VIDEO_INDEX if v.get("id") == video_id), None)
    if not video:
        return jsonify({"error": "Video not found"}), 404

    # Cloud-only playback: always prefer a presigned S3 URL when available.
    raw_source = (video.get("s3_video_uri") or "").strip() or (video.get("s3_video_key") or "").strip()
    if raw_source:
        try:
            bucket, key = _parse_allowed_s3_video_source(raw_source)
            url = _presign_s3_get_object_url(bucket=bucket, key=key)
            return redirect(url, code=302)
        except Exception as exc:
            return jsonify({"error": f"Failed to create S3 playback URL: {exc}"}), 500

    stored_filename = video.get("stored_filename")
    if not stored_filename:
        file_path_value = (video.get("file_path") or "").strip()
        stored_filename = Path(file_path_value).name if file_path_value else ""
    if not stored_filename:
        return jsonify({"error": "Video is missing stored_filename"}), 400

    video_path = _absolute_video_path(stored_filename)
    if not video_path.exists():
        legacy_path_value = (video.get("file_path") or "").strip()
        legacy_path = (STORAGE_BASE_DIR / legacy_path_value) if legacy_path_value else None
        if legacy_path and legacy_path.exists():
            video_path = legacy_path
        else:
            return jsonify({"error": "Video file not found on disk"}), 404

    mimetype = mimetypes.guess_type(str(video_path))[0] or "application/octet-stream"
    return send_file(str(video_path), mimetype=mimetype, conditional=True)


@app.route("/video/<video_id>", methods=["DELETE"])
def delete_video(video_id: str) -> Any:
    """Delete a video and its associated files."""
    global VIDEO_INDEX
    
    # Find the video
    video = next((v for v in VIDEO_INDEX if v["id"] == video_id), None)
    
    if not video:
        return jsonify({"error": "Video not found"}), 404
    
    try:
        deleted_local_files: list[str] = []
        deleted_s3_objects = 0
        s3_warnings: list[str] = []

        # Delete the video file from storage
        stored_filename = video.get("stored_filename")
        video_path = _absolute_video_path(stored_filename) if stored_filename else None

        if video_path and video_path.exists():
            video_path.unlink()
            deleted_local_files.append(str(video_path))
        else:
            legacy_path_value = video.get("file_path")
            if legacy_path_value:
                legacy_path = Path(legacy_path_value)
                if legacy_path.exists():
                    legacy_path.unlink()

                    deleted_local_files.append(str(legacy_path))

        # Delete local subtitle artifacts
        try:
            if SUBTITLES_DIR is not None:
                for ext in (".srt", ".vtt"):
                    p = Path(SUBTITLES_DIR) / f"{video_id}{ext}"
                    if p.exists():
                        p.unlink()
                        deleted_local_files.append(str(p))
        except Exception:
            pass

        # Delete S3 metadata artifacts (combined/categories/zip/subtitles) under artifacts prefix
        try:
            bucket = _metadata_artifacts_s3_bucket()
            base = _metadata_artifacts_s3_prefix()
            deleted_s3_objects += _s3_delete_prefix_best_effort(bucket=bucket, prefix=f"{base}/{video_id}")
        except Exception:
            s3_warnings.append("Failed to delete S3 metadata artifacts (best-effort)")

        # Delete the S3 source video object as well, but only when it is under an allowed managed prefix.
        # - envid-metadata uploads: ENVID_METADATA_VIDEO_S3_PREFIX in video bucket
        # - rawvideo sources: ENVID_METADATA_RAWVIDEO_S3_PREFIX in rawvideo bucket
        try:
            video_bucket = _video_s3_bucket()
            video_prefix = (os.getenv("ENVID_METADATA_VIDEO_S3_PREFIX") or "envid-metadata/videos").strip().strip("/")
            raw_bucket = _rawvideo_bucket()
            raw_prefix = _rawvideo_prefix().strip().strip("/")
            raw_source = (video.get("s3_video_uri") or "").strip() or (video.get("s3_video_key") or "").strip()
            if raw_source:
                b, k = _parse_allowed_s3_video_source(raw_source)
                if b == video_bucket and k.startswith(video_prefix + "/"):
                    if _s3_delete_object_best_effort(bucket=b, key=k):
                        deleted_s3_objects += 1
                    else:
                        s3_warnings.append(f"Failed to delete managed S3 video object: s3://{b}/{k}")
                if b == raw_bucket and k.startswith(raw_prefix + "/"):
                    if _s3_delete_object_best_effort(bucket=b, key=k):
                        deleted_s3_objects += 1
                    else:
                        s3_warnings.append(f"Failed to delete managed S3 rawvideo object: s3://{b}/{k}")

            # Also delete MediaConvert proxy artifacts (best-effort).
            # Proxies are written under ENVID_METADATA_MEDIACONVERT_PROXY_PREFIX/<video_id>/...
            # in either the input bucket or ENVID_METADATA_MEDIACONVERT_OUTPUT_BUCKET.
            proxy_prefix = _mediaconvert_proxy_prefix().strip().strip("/")
            if proxy_prefix:
                out_bucket_override = (os.getenv("ENVID_METADATA_MEDIACONVERT_OUTPUT_BUCKET") or "").strip()
                candidate_buckets = {video_bucket, raw_bucket}
                if out_bucket_override:
                    candidate_buckets.add(out_bucket_override)
                for b2 in sorted(candidate_buckets):
                    deleted_s3_objects += _s3_delete_prefix_best_effort(bucket=b2, prefix=f"{proxy_prefix}/{video_id}")

            # Backward-compat: if a specific proxy key was recorded, attempt direct deletion too.
            proxy_key = (video.get("s3_proxy_key") or "").strip()
            proxy_uri = (video.get("s3_proxy_uri") or "").strip()
            if proxy_uri.startswith("s3://"):
                try:
                    parts = proxy_uri.replace("s3://", "", 1).split("/", 1)
                    b3 = (parts[0] or "").strip()
                    k3 = (parts[1] if len(parts) > 1 else "").strip().lstrip("/")
                    if b3 and k3:
                        if _s3_delete_object_best_effort(bucket=b3, key=k3):
                            deleted_s3_objects += 1
                        else:
                            s3_warnings.append(f"Failed to delete S3 proxy object: {proxy_uri}")
                except Exception:
                    s3_warnings.append("Failed to parse/delete s3_proxy_uri")
            elif proxy_key:
                # If we only have a key, try deletion from likely buckets.
                out_bucket_override = (os.getenv("ENVID_METADATA_MEDIACONVERT_OUTPUT_BUCKET") or "").strip()
                candidate_buckets = {video_bucket, raw_bucket}
                if out_bucket_override:
                    candidate_buckets.add(out_bucket_override)
                for b3 in sorted(candidate_buckets):
                    if _s3_delete_object_best_effort(bucket=b3, key=proxy_key):
                        deleted_s3_objects += 1
        except Exception:
            s3_warnings.append("Failed to delete S3 video/proxy objects (best-effort)")
        
        # Remove from index
        VIDEO_INDEX = [v for v in VIDEO_INDEX if v["id"] != video_id]
        
        # Save updated index
        _save_video_index()
        
        print(f"Video {video_id} deleted successfully")

        return jsonify({
            "status": "success",
            "message": f"Video '{video['title']}' deleted successfully",
            "video_id": video_id,
            "deleted_local_files": deleted_local_files,
            "deleted_s3_objects": deleted_s3_objects,
            "s3_warnings": s3_warnings,
        }), 200
        
    except Exception as e:
        print(f"Error deleting video {video_id}: {str(e)}")
        return jsonify({"error": f"Failed to delete video: {str(e)}"}), 500


@app.route("/reprocess-video/<video_id>", methods=["POST"])
def reprocess_video(video_id: str) -> Any:
    """Re-process an already-indexed video.

    Prefers Rekognition Video (S3 source) when configured.
    """
    video = next((v for v in VIDEO_INDEX if v.get("id") == video_id), None)
    if not video:
        return jsonify({"error": "Video not found"}), 404

    # Source must exist in S3 for cloud-only reprocess.
    raw_source = (video.get("s3_video_uri") or "").strip() or (video.get("s3_video_key") or "").strip()
    if not raw_source:
        return jsonify({"error": "Video is missing s3_video_uri/s3_video_key; cannot reprocess in cloud-only mode"}), 400

    try:
        bucket, key = _parse_allowed_s3_video_source(raw_source)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    frame_interval_seconds = _parse_int_param(
        request.args.get("frame_interval_seconds"),
        default=int(video.get("frame_interval_seconds") or 10),
        min_value=1,
        max_value=30,
    )

    max_frames_to_analyze = _parse_int_param(
        request.args.get("max_frames_to_analyze"),
        default=1000,
        min_value=1,
        max_value=10000,
    )
    face_recognition_mode = (request.args.get("face_recognition_mode") or "").strip() or None
    collection_id = (request.args.get("collection_id") or "").strip() or None

    video_title = (video.get("title") or Path(key).name or video_id).strip()
    video_description = (video.get("description") or "").strip()

    _job_init(video_id, title=video_title)
    _job_update(video_id, status="processing", progress=1, message="Reprocess queued", s3_video_key=key, s3_video_uri=f"s3://{bucket}/{key}")
    _job_step_update(video_id, "upload_to_s3", status="completed", percent=100, message="Using existing S3 source")

    analysis_source = (os.getenv("ENVID_METADATA_REKOGNITION_ANALYSIS_SOURCE") or "").strip().lower()
    prefer_video = _disable_per_frame_analysis() or analysis_source in {"s3_video", "video", "rekognition_video"}

    def _worker_local_prefer_video() -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix=f"envid_metadata_reprocess_{video_id}_"))
        try:
            stored_filename = str(video.get("stored_filename") or "").strip()
            stored_video_path = (VIDEOS_DIR / stored_filename)
            if not stored_filename or not stored_video_path.exists():
                # S3-only entries: reprocess without downloading.
                _process_s3_object_job_cloud_only(
                    job_id=video_id,
                    s3_bucket=bucket,
                    s3_key=key,
                    video_title=video_title,
                    video_description=video_description,
                    frame_interval_seconds=frame_interval_seconds,
                    technical_ffprobe=video.get("technical_ffprobe") if isinstance(video.get("technical_ffprobe"), dict) else None,
                    duration_seconds=video.get("duration_seconds"),
                )
                return

            working_video_path = temp_dir / "video.mp4"
            shutil.copy2(stored_video_path, working_video_path)

            _process_video_job(
                job_id=video_id,
                temp_dir=temp_dir,
                video_path=working_video_path,
                video_title=video_title,
                video_description=video_description,
                original_filename=str(video.get("original_filename") or stored_filename or f"{video_id}.mp4"),
                frame_interval_seconds=frame_interval_seconds,
                max_frames_to_analyze=max_frames_to_analyze,
                face_recognition_mode=face_recognition_mode,
                collection_id=collection_id,
                preexisting_s3_video_key=key,
                preexisting_s3_video_uri=f"s3://{bucket}/{key}",
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    thread = threading.Thread(
        target=_worker_local_prefer_video if prefer_video else _process_s3_object_job_download_and_analyze,
        kwargs={}
        if prefer_video
        else {
            "job_id": video_id,
            "s3_bucket": bucket,
            "s3_key": key,
            "video_title": video_title,
            "video_description": video_description,
            "frame_interval_seconds": frame_interval_seconds,
            "max_frames_to_analyze": max_frames_to_analyze,
            "face_recognition_mode": face_recognition_mode,
            "collection_id": collection_id,
        },
        daemon=True,
    )
    thread.start()

    with JOBS_LOCK:
        job = JOBS.get(video_id)
    return jsonify({"job_id": video_id, "job": job}), 202


@app.route("/video/<video_id>/reprocess", methods=["POST"])
def reprocess_video_legacy_alias(video_id: str) -> Any:
    """Backward-compatible alias for older clients."""
    return reprocess_video(video_id)


@app.route("/reprocess-videos", methods=["POST"])
def reprocess_videos() -> Any:
    """Re-process indexed videos (best-effort).

    Videos without an S3 source are skipped.
    """
    payload = request.get_json(silent=True) or {}
    requested_ids = payload.get("video_ids")
    if requested_ids is not None and not isinstance(requested_ids, list):
        return jsonify({"error": "video_ids must be a list when provided"}), 400

    targets = VIDEO_INDEX
    if isinstance(requested_ids, list):
        requested_set = {str(v) for v in requested_ids}
        targets = [v for v in VIDEO_INDEX if str(v.get("id")) in requested_set]

    frame_interval_seconds = _parse_int_param(
        request.args.get("frame_interval_seconds"),
        default=10,
        min_value=1,
        max_value=30,
    )

    max_frames_to_analyze = _parse_int_param(
        request.args.get("max_frames_to_analyze"),
        default=1000,
        min_value=1,
        max_value=10000,
    )
    face_recognition_mode = (request.args.get("face_recognition_mode") or "").strip() or None
    collection_id = (request.args.get("collection_id") or "").strip() or None

    results: List[Dict[str, Any]] = []
    queued_count = 0
    for video in targets:
        vid = str(video.get("id"))

        raw_source = (video.get("s3_video_uri") or "").strip() or (video.get("s3_video_key") or "").strip()
        if not raw_source:
            results.append({"id": vid, "status": "skipped", "reason": "missing s3 source"})
            continue

        try:
            bucket, key = _parse_allowed_s3_video_source(raw_source)
        except Exception as exc:
            results.append({"id": vid, "status": "skipped", "reason": str(exc)})
            continue

        video_title = (video.get("title") or Path(key).name or vid).strip()
        video_description = (video.get("description") or "").strip()

        try:
            _job_init(vid, title=video_title)
            _job_update(vid, status="processing", progress=1, message="Reprocess queued", s3_video_key=key, s3_video_uri=f"s3://{bucket}/{key}")

            analysis_source = (os.getenv("ENVID_METADATA_REKOGNITION_ANALYSIS_SOURCE") or "").strip().lower()
            prefer_video = _disable_per_frame_analysis() or analysis_source in {"s3_video", "video", "rekognition_video"}

            def _worker_local_prefer_video(
                video_obj: Dict[str, Any] = video,
                job_id: str = vid,
                s3_bucket: str = bucket,
                s3_key: str = key,
                title: str = video_title,
                description: str = video_description,
            ) -> None:
                temp_dir = Path(tempfile.mkdtemp(prefix=f"envid_metadata_reprocess_{job_id}_"))
                try:
                    stored_filename = str(video_obj.get("stored_filename") or "").strip()
                    stored_video_path = (VIDEOS_DIR / stored_filename)
                    if not stored_filename or not stored_video_path.exists():
                        _process_s3_object_job_cloud_only(
                            job_id=job_id,
                            s3_bucket=s3_bucket,
                            s3_key=s3_key,
                            video_title=title,
                            video_description=description,
                            frame_interval_seconds=frame_interval_seconds,
                            technical_ffprobe=video_obj.get("technical_ffprobe") if isinstance(video_obj.get("technical_ffprobe"), dict) else None,
                            duration_seconds=video_obj.get("duration_seconds"),
                        )
                        return

                    working_video_path = temp_dir / "video.mp4"
                    shutil.copy2(stored_video_path, working_video_path)

                    _process_video_job(
                        job_id=job_id,
                        temp_dir=temp_dir,
                        video_path=working_video_path,
                        video_title=title,
                        video_description=description,
                        original_filename=str(video_obj.get("original_filename") or stored_filename or f"{job_id}.mp4"),
                        frame_interval_seconds=frame_interval_seconds,
                        max_frames_to_analyze=max_frames_to_analyze,
                        face_recognition_mode=face_recognition_mode,
                        collection_id=collection_id,
                        preexisting_s3_video_key=s3_key,
                        preexisting_s3_video_uri=f"s3://{s3_bucket}/{s3_key}",
                    )
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)

            thread = threading.Thread(
                target=_worker_local_prefer_video if prefer_video else _process_s3_object_job_download_and_analyze,
                kwargs={}
                if prefer_video
                else {
                    "job_id": vid,
                    "s3_bucket": bucket,
                    "s3_key": key,
                    "video_title": video_title,
                    "video_description": video_description,
                    "frame_interval_seconds": frame_interval_seconds,
                    "max_frames_to_analyze": max_frames_to_analyze,
                    "face_recognition_mode": face_recognition_mode,
                    "collection_id": collection_id,
                },
                daemon=True,
            )
            thread.start()
            queued_count += 1
            results.append({"id": vid, "status": "queued", "job_id": vid, "s3_uri": f"s3://{bucket}/{key}"})
        except Exception as exc:
            app.logger.error("Bulk reprocess failed to queue for video %s: %s", vid, exc)
            results.append({"id": vid, "status": "error", "error": str(exc)})

    return jsonify({"message": "Reprocess queued", "total": len(results), "queued": queued_count, "results": results}), 202


def _extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract text from different file formats."""
    file_extension = Path(filename).suffix.lower()
    
    try:
        if file_extension == '.pdf':
            # Extract text from PDF
            pdf_reader = PdfReader(io.BytesIO(file_content))
            text_parts = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return '\n\n'.join(text_parts)
        
        elif file_extension in ['.docx', '.doc']:
            # Extract text from DOCX
            doc = DocxDocument(io.BytesIO(file_content))
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            return '\n\n'.join(text_parts)
        
        elif file_extension == '.txt':
            # Plain text file
            return file_content.decode('utf-8', errors='ignore')
        
        else:
            # Try decoding as text for unknown file types
            return file_content.decode('utf-8', errors='ignore')
    
    except Exception as e:
        app.logger.error(f"Error extracting text from {filename}: {str(e)}")
        # Fallback: try to decode as text
        try:
            return file_content.decode('utf-8', errors='ignore')
        except:
            return ""


@app.route("/upload-document", methods=["POST"])
def upload_document() -> Any:
    return jsonify({"error": "Document extraction is disabled"}), 410


@app.route("/documents", methods=["GET"])
def list_documents() -> Any:
    """List all indexed documents."""
    documents = [
        {
            "id": d["id"],
            "title": d["title"],
            "chunks_count": d["chunks_count"],
            "uploaded_at": d["uploaded_at"]
        }
        for d in DOCUMENT_INDEX
    ]
    
    return jsonify({
        "documents": documents,
        "total": len(documents)
    }), 200


@app.route("/documents/<document_id>", methods=["DELETE"])
def delete_document(document_id: str) -> Any:
    """Delete a document and all its chunks."""
    global DOCUMENT_INDEX
    
    # Find document index
    doc_idx = None
    for idx, doc in enumerate(DOCUMENT_INDEX):
        if doc["id"] == document_id:
            doc_idx = idx
            break
    
    if doc_idx is None:
        return jsonify({"error": "Document not found"}), 404
    
    # Remove document from index
    deleted_doc = DOCUMENT_INDEX.pop(doc_idx)
    
    # Delete stored document file
    stored_filename = deleted_doc.get("stored_filename")
    doc_path = _absolute_document_path(stored_filename) if stored_filename else None
    if doc_path and doc_path.exists():
        doc_path.unlink()
        app.logger.info(f"Deleted document file: {doc_path}")
    else:
        legacy_path_value = deleted_doc.get("file_path")
        if legacy_path_value:
            legacy_path = Path(legacy_path_value)
            if legacy_path.exists():
                legacy_path.unlink()
                app.logger.info(f"Deleted legacy document file: {legacy_path}")
    
    app.logger.info(f"Deleted document: {deleted_doc['title']} (ID: {document_id})")
    
    return jsonify({
        "message": "Document deleted successfully",
        "deleted_document": {
            "id": deleted_doc["id"],
            "title": deleted_doc["title"]
        }
    }), 200


@app.route("/search-text", methods=["POST"])
def search_text() -> Any:
    """Search documents using text/semantic similarity."""
    payload = request.get_json(silent=True) or {}
    query = payload.get("query", "").strip()
    top_k = int(payload.get("top_k", 5))
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    if not DOCUMENT_INDEX:
        return jsonify({"error": "No documents indexed yet"}), 400
    
    try:
        query_terms = [t for t in query.split() if t]

        # If the document looks like a Movie — Actors list, return titles-only matches.
        # This gives users the expected UX for queries like "amit".
        movie_results: List[Dict[str, Any]] = []
        for doc in DOCUMENT_INDEX:
            full_text = doc.get("full_text") or ""
            titles = _extract_movie_titles_by_actor(full_text, query_terms)
            if not titles:
                continue

            # Return a compact list of titles; keep response shape stable with "text".
            movie_results.append({
                "document_id": doc.get("id"),
                "document_title": doc.get("title"),
                "text": "\n".join(titles[: min(len(titles), 25)]),
                "matches_count": len(titles),
                "similarity_score": 1.0,
                "cosine_similarity": 1.0,
            })

        if movie_results:
            movie_results.sort(key=lambda r: (r.get("matches_count", 0), r.get("document_title") or ""), reverse=True)
            return jsonify({
                "query": query,
                "results": movie_results[:top_k]
            }), 200

        # Otherwise, fall back to semantic snippet search.
        query_embedding = _generate_embedding(query)

        if not query_embedding:
            return jsonify({"error": "Failed to generate query embedding"}), 500
        
        # Search across all chunks in all documents.
        # Enforce prefix-term containment (see _chunk_matches_terms) and return only the best hit per document.
        best_by_document: Dict[str, Dict[str, Any]] = {}
        for doc in DOCUMENT_INDEX:
            doc_id = doc.get("id")
            if not doc_id:
                continue

            best_hit: Dict[str, Any] | None = None
            for chunk in doc.get("chunks", []):
                chunk_text = (chunk or {}).get("text", "")
                if query_terms and not _chunk_matches_terms(chunk_text, query_terms):
                    continue

                cosine_similarity = _compute_similarity(query_embedding, (chunk or {}).get("embedding", []))
                similarity = _normalize_similarity(cosine_similarity)
                snippet = _extract_match_snippet(chunk_text, query_terms) or chunk_text[:400]

                candidate = {
                    "document_id": doc_id,
                    "document_title": doc.get("title") or doc_id,
                    "text": snippet,
                    "similarity_score": round(similarity, 4),
                    "cosine_similarity": round(cosine_similarity, 4),
                }

                if best_hit is None or candidate["similarity_score"] > best_hit["similarity_score"]:
                    best_hit = candidate

            if best_hit is not None:
                best_by_document[doc_id] = best_hit

        results = list(best_by_document.values())
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return jsonify({
            "query": query,
            "results": results[:top_k]
        }), 200
        
    except Exception as e:
        app.logger.error(f"Text search failed: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Search failed: {str(e)}"}), 500


@app.route("/ask-question", methods=["POST"])
def ask_question() -> Any:
    """Ask a question about a document using Meta Llama via Bedrock."""
    payload = request.get_json(silent=True) or {}
    document_id = payload.get("document_id", "").strip()
    question = payload.get("question", "").strip()
    
    if not document_id or not question:
        return jsonify({"error": "Both document_id and question are required"}), 400
    
    # Find the document
    doc = next((d for d in DOCUMENT_INDEX if d["id"] == document_id), None)
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    
    try:
        # Generate query embedding
        query_embedding = _generate_embedding(question)
        
        # Find most relevant chunks
        chunk_scores = []
        for chunk in doc["chunks"]:
            similarity = _compute_similarity(query_embedding, chunk["embedding"])
            chunk_scores.append((chunk["text"], similarity))
        
        # Sort and get top 3 chunks
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        relevant_chunks = [chunk[0] for chunk in chunk_scores[:3]]
        
        # Build prompt for Llama
        context = "\n\n".join(relevant_chunks)
        prompt = f"""Based on the following context from the document, please answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        # Call Bedrock with Meta Llama
        response = bedrock_runtime.invoke_model(
            modelId="meta.llama3-70b-instruct-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "prompt": prompt,
                "max_gen_len": 512,
                "temperature": 0.7,
                "top_p": 0.9
            })
        )
        
        response_body = json.loads(response["body"].read())
        answer = response_body.get("generation", "").strip()
        
        return jsonify({
            "question": question,
            "answer": answer,
            "context": relevant_chunks,
            "document_title": doc["title"]
        }), 200
        
    except Exception as e:
        app.logger.error(f"Q&A failed: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Question answering failed: {str(e)}"}), 500


@app.route("/askme", methods=["POST"])
def askme_chatbot():
    """
    AskMe chatbot endpoint - answers questions about MediaGenAI use cases and documentation.
    This endpoint reads from documentation files and uses Bedrock Llama to provide answers.
    """
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Load documentation content
        docs_dir = Path(__file__).parent.parent / "Documentation"
        readme_file = Path(__file__).parent.parent / "SERVICES_README.md"
        
        context_docs = []
        
        # Read SERVICES_README.md
        if readme_file.exists():
            try:
                with open(readme_file, 'r', encoding='utf-8') as f:
                    context_docs.append({
                        "filename": "SERVICES_README.md",
                        "content": f.read()
                    })
            except Exception as e:
                print(f"Error reading SERVICES_README.md: {e}")
        
        # Read documentation files
        if docs_dir.exists():
            for doc_file in docs_dir.glob("*.md"):
                try:
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        context_docs.append({
                            "filename": doc_file.name,
                            "content": f.read()
                        })
                except Exception as e:
                    print(f"Error reading {doc_file}: {e}")
        
        if not context_docs:
            return jsonify({
                "error": "No documentation available"
            }), 500
        
        # Build context from documentation
        context_text = "\n\n---\n\n".join([
            f"Document: {doc['filename']}\n\n{doc['content'][:3000]}"  # Limit each doc to 3000 chars
            for doc in context_docs[:5]  # Use top 5 documents
        ])
        
        # Create prompt for Llama
        prompt = f"""You are AskMe, a helpful chatbot assistant for MediaGenAI platform. You help users understand the various AI-powered media generation use cases available in the platform.

Documentation Context:
{context_text}

User Question: {question}

Provide a helpful, accurate, and concise answer based on the documentation above. If the question is about a specific use case or feature, explain what it does and how it works. If you're not sure about something, say so.

Answer:"""
        
        # Call Bedrock Llama
        response = bedrock_runtime.invoke_model(
            modelId="meta.llama3-70b-instruct-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "prompt": prompt,
                "max_gen_len": 512,
                "temperature": 0.7,
                "top_p": 0.9
            })
        )
        
        response_body = json.loads(response["body"].read())
        answer = response_body.get("generation", "").strip()
        
        return jsonify({
            "question": question,
            "answer": answer,
            "sources": [doc["filename"] for doc in context_docs[:5]]
        }), 200
        
    except Exception as e:
        app.logger.error(f"AskMe chatbot failed: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Chatbot request failed: {str(e)}"}), 500


if __name__ == "__main__":
    _load_video_index()
    _load_document_index()
    port = int(os.getenv("ENVID_METADATA_PORT", "5014"))
    debug_raw = (os.getenv("DEBUG") or "false").strip().lower()
    reloader_raw = (os.getenv("RELOADER") or "false").strip().lower()
    debug = debug_raw in {"1", "true", "yes", "y", "on"}
    use_reloader = reloader_raw in {"1", "true", "yes", "y", "on"}
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=use_reloader)
