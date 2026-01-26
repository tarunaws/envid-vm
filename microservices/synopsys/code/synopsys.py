from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)


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


def _normalize_transcript_basic(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    return t.strip()


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    st = max(float(a_start or 0.0), float(b_start or 0.0))
    en = min(float(a_end or 0.0), float(b_end or 0.0))
    return max(0.0, en - st)


def _openrouter_headers() -> tuple[Dict[str, str] | None, dict[str, Any]]:
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        return None, {"available": False}

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
    return headers, {"available": True}


def _openrouter_chat_completion(payload: dict[str, Any], timeout_s: float) -> str | None:
    headers, meta = _openrouter_headers()
    if not headers:
        return None

    base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip().rstrip("/")
    resp = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=float(timeout_s))
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenRouter failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    return content or None


@app.get("/health")
def health() -> Any:
    return jsonify({"ok": True}), 200


@app.post("/normalize_transcript")
def normalize_transcript() -> Any:
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text") or "").strip()
    language_code = str(payload.get("language_code") or "").strip() or None

    headers, meta = _openrouter_headers()
    if not headers:
        return jsonify({"text": None, "meta": meta}), 200

    raw = _normalize_transcript_basic(text)
    if not raw:
        return jsonify({"text": None, "meta": {"available": True, "applied": False}}), 200

    model = (os.getenv("OPENROUTER_TRANSCRIPT_MODEL") or os.getenv("OPENROUTER_MODEL") or "meta-llama/llama-3.3-70b-instruct:free").strip()
    timeout_s = _safe_float(os.getenv("OPENROUTER_TIMEOUT_SECONDS"), 15.0)
    lang = (language_code or "").strip() or "unknown"

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
        f"Language hint: {lang}\n\n"
        f"Transcript:\n{raw[:12000]}\n"
    )

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Fix punctuation/casing/spacing only. Output plain text."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }

    try:
        content = _openrouter_chat_completion(data, timeout_s)
        if content:
            return jsonify({"text": content, "meta": {"available": True, "applied": True, "provider": "openrouter", "model": model}}), 200
        return jsonify({"text": None, "meta": {"available": True, "applied": False, "provider": "openrouter", "model": model}}), 200
    except Exception as exc:
        return jsonify({"text": None, "meta": {"available": True, "applied": False, "provider": "openrouter", "model": model, "error": str(exc)[:240]}}), 200


@app.post("/synopses")
def generate_synopses() -> Any:
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text") or "").strip()
    language_code = str(payload.get("language_code") or "").strip() or None

    headers, meta = _openrouter_headers()
    if not headers:
        return jsonify({"synopses": None, "meta": meta}), 200

    raw = _normalize_transcript_basic(text)
    if not raw:
        return jsonify({"synopses": None, "meta": {"available": True, "applied": False}}), 200

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

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 900,
    }

    try:
        content = _openrouter_chat_completion(data, timeout_s) or ""
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        payload_json = json.loads(m.group(0) if m else content)
        if isinstance(payload_json, dict):
            return jsonify({"synopses": payload_json, "meta": {"available": True, "applied": True, "provider": "openrouter", "model": model}}), 200
        return jsonify({"synopses": None, "meta": {"available": True, "applied": False, "provider": "openrouter", "model": model}}), 200
    except Exception as exc:
        return jsonify({"synopses": None, "meta": {"available": True, "applied": False, "provider": "openrouter", "model": model, "error": str(exc)[:240]}}), 200


@app.post("/scene_summaries")
def scene_summaries() -> Any:
    payload = request.get_json(silent=True) or {}
    scenes = payload.get("scenes") or []
    transcript_segments = payload.get("transcript_segments") or []
    labels_src = payload.get("labels")
    language_code = str(payload.get("language_code") or "").strip() or None

    headers, meta = _openrouter_headers()
    if not headers:
        return jsonify({"summaries": None, "meta": meta}), 200

    if not isinstance(scenes, list) or not scenes:
        return jsonify({"summaries": None, "meta": {"available": True, "applied": False}}), 200

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
        return jsonify({"summaries": None, "meta": {"available": True, "applied": False, "provider": "openrouter", "model": model}}), 200

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
                return jsonify({"summaries": out, "meta": {"available": True, "applied": True, "provider": "openrouter", "model": model}}), 200
        return jsonify({"summaries": None, "meta": {"available": True, "applied": False, "provider": "openrouter", "model": model}}), 200
    except Exception as exc:
        return jsonify({"summaries": None, "meta": {"available": True, "applied": False, "provider": "openrouter", "model": model, "error": str(exc)[:240]}}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5099")))
