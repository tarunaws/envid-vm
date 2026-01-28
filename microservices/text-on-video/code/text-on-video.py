from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from flask import Flask, jsonify, request
from PIL import Image, ImageEnhance, ImageOps  # type: ignore
import pytesseract  # type: ignore

app = Flask(__name__)

def _frame_extractor_url() -> str:
    return "http://frame-extractor:5093"


def _scene_detect_url() -> str:
    return "http://scene-detect:5094"


def _backend_url() -> str:
    return (os.getenv("ENVID_BACKEND_URL") or "http://backend:5016").rstrip("/")


def _ensure_tmp_path(raw: str) -> Path:
    p = Path(raw).resolve()
    allowed = ("/tmp/", "/mnt/gcs/")
    if not str(p).startswith(allowed):
        raise RuntimeError("video_path must be under /tmp or /mnt/gcs")
    if not p.exists() or not p.is_file():
        raise RuntimeError("video_path not found")
    return p


def _resolve_video_path(job_id: str, kind: str) -> str:
    if not job_id:
        raise RuntimeError("Missing job_id")
    url = f"{_backend_url()}/jobs/{job_id}/outputs/{kind}"
    resp = requests.get(url, timeout=10)
    if resp.status_code >= 400:
        raise RuntimeError(f"Failed to resolve video_path ({resp.status_code}): {resp.text}")
    payload = resp.json()
    path = str(payload.get("path") or "").strip()
    if not path:
        raise RuntimeError("Resolved video_path is empty")
    return path


def _parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_int(value: Any, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def _env_truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_text(raw: str) -> str:
    cleaned = re.sub(r"\s+", " ", raw or "").strip()
    cleaned = re.sub(r"[\x00-\x1f]+", " ", cleaned).strip()
    return cleaned


def _is_plausible_text(text: str) -> bool:
    if not text:
        return False
    min_len = _parse_int(os.getenv("ENVID_METADATA_OCR_MIN_TEXT_LEN"), 3, min_value=1, max_value=20)
    min_letters = _parse_int(os.getenv("ENVID_METADATA_OCR_MIN_LETTER_COUNT"), 2, min_value=0, max_value=50)
    min_alpha_ratio = _parse_float(os.getenv("ENVID_METADATA_OCR_MIN_ALPHA_RATIO"), 0.5)
    if len(text) < min_len:
        return False
    if re.search(r"(.)\1{4,}", text):
        return False
    letters = sum(1 for ch in text if ch.isalpha())
    alnum = sum(1 for ch in text if ch.isalnum())
    if letters < min_letters:
        return False
    if alnum > 0 and (alnum / len(text)) < min_alpha_ratio:
        return False
    return True


def _preprocess_image(img: Image.Image) -> Image.Image:
    upscale = _parse_float(os.getenv("ENVID_METADATA_OCR_UPSCALE"), 1.5)
    if upscale < 1.0:
        upscale = 1.0
    contrast = _parse_float(os.getenv("ENVID_METADATA_OCR_CONTRAST"), 1.4)
    sharpen = _parse_float(os.getenv("ENVID_METADATA_OCR_SHARPEN"), 1.1)
    threshold = _parse_int(os.getenv("ENVID_METADATA_OCR_THRESHOLD"), 0, min_value=0, max_value=255)

    if upscale > 1.0:
        w, h = img.size
        img = img.resize((int(w * upscale), int(h * upscale)), resample=Image.LANCZOS)

    img = ImageOps.exif_transpose(img)
    img = ImageOps.grayscale(img)

    if contrast > 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if sharpen > 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpen)

    if threshold > 0:
        img = img.point(lambda p: 255 if p >= threshold else 0)
    return img


def _candidate_regions(img: Image.Image) -> List[Tuple[str, Image.Image]]:
    w, h = img.size
    regions: List[Tuple[str, Image.Image]] = [("full", img)]
    if _env_truthy(os.getenv("ENVID_METADATA_OCR_ENABLE_LOWER_THIRD"), default=True):
        y0 = int(h * 0.55)
        if y0 < h - 10:
            regions.append(("lower_third", img.crop((0, y0, w, h))))
    if _env_truthy(os.getenv("ENVID_METADATA_OCR_ENABLE_TOP_THIRD"), default=False):
        y1 = int(h * 0.35)
        if y1 > 10:
            regions.append(("top_third", img.crop((0, 0, w, y1))))
    if _env_truthy(os.getenv("ENVID_METADATA_OCR_ENABLE_MIDDLE_BAND"), default=True):
        band_ratio = _parse_float(os.getenv("ENVID_METADATA_OCR_MIDDLE_BAND_HEIGHT"), 0.4)
        if band_ratio <= 0:
            band_ratio = 0.4
        if band_ratio > 1.0:
            band_ratio = 1.0
        band_h = int(h * band_ratio)
        y_mid = int((h - band_h) / 2)
        if band_h > 10 and y_mid >= 0:
            regions.append(("middle_band", img.crop((0, y_mid, w, y_mid + band_h))))
    return regions


def _run_tesseract(img: Image.Image) -> Tuple[str, float]:
    lang = (os.getenv("ENVID_METADATA_OCR_TESSERACT_LANG") or "eng+hin").strip() or "eng"
    psm = _parse_int(os.getenv("ENVID_METADATA_OCR_TESSERACT_PSM"), 6, min_value=3, max_value=13)
    oem = _parse_int(os.getenv("ENVID_METADATA_OCR_TESSERACT_OEM"), 1, min_value=0, max_value=3)
    config = f"--oem {oem} --psm {psm}"

    data = pytesseract.image_to_data(img, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    words: List[str] = []
    confs: List[float] = []
    n = len(data.get("text") or [])
    for i in range(n):
        txt = str((data.get("text") or [""])[i] or "").strip()
        if not txt:
            continue
        try:
            conf = float((data.get("conf") or [0])[i])
        except Exception:
            conf = 0.0
        if conf <= 0:
            continue
        words.append(txt)
        confs.append(max(0.0, min(100.0, conf)) / 100.0)

    text = " ".join(words).strip()
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return text, mean_conf


def _fetch_scene_timestamps(*, video_path: Path, max_frames: int) -> list[float]:
    url = f"{_scene_detect_url()}/pyscenedetect/scenes"
    threshold = _parse_float(os.getenv("ENVID_METADATA_OCR_SCENE_THRESHOLD"), 27.0)
    max_seconds = _parse_int(os.getenv("ENVID_METADATA_OCR_SCENE_MAX_SECONDS"), 900, min_value=30, max_value=7200)

    with video_path.open("rb") as handle:
        files = {"video": (video_path.name, handle, "application/octet-stream")}
        data = {"threshold": str(threshold), "max_seconds": str(max_seconds)}
        resp = requests.post(url, files=files, data=data, timeout=max(30, max_seconds + 10))
    if resp.status_code >= 400:
        return []
    payload = resp.json()
    scenes = payload.get("scenes") if isinstance(payload, dict) else None
    if not isinstance(scenes, list):
        return []

    timestamps: list[float] = []
    for s in scenes:
        if not isinstance(s, dict):
            continue
        try:
            st = float(s.get("start"))
            en = float(s.get("end"))
        except Exception:
            continue
        if en <= st:
            continue
        timestamps.append((st + en) / 2.0)

    if max_frames > 0 and len(timestamps) > max_frames:
        step = len(timestamps) / max_frames
        timestamps = [timestamps[int(i * step)] for i in range(max_frames)]
    return timestamps


@app.get("/health")
def health() -> Any:
    engine = (os.getenv("ENVID_METADATA_OCR_ENGINE") or "tesseract").strip().lower()
    return (
        jsonify(
            {
                "ok": True,
                "ocr_engine": engine,
            }
        ),
        200,
    )


@app.post("/ocr")
def ocr() -> Any:
    payload = request.get_json(silent=True) or {}
    video_path_raw = str(payload.get("video_path") or "").strip()

    interval = 0.0
    max_frames = 200000
    job_id = str(payload.get("job_id") or "").strip()
    output_kind = str(payload.get("output_kind") or "processed_local").strip() or "processed_local"

    try:
        if not video_path_raw:
            video_path_raw = _resolve_video_path(job_id, output_kind)
        _ensure_tmp_path(video_path_raw)
        extractor_url = _frame_extractor_url()
        use_scene_sampling = False
        timestamps: list[float] = []
        if use_scene_sampling:
            try:
                timestamps = _fetch_scene_timestamps(video_path=_ensure_tmp_path(video_path_raw), max_frames=max_frames)
            except Exception:
                timestamps = []

        payload = {
            "video_path": video_path_raw,
            "interval_seconds": interval,
            "max_frames": max_frames,
            "job_id": job_id,
        }
        if timestamps:
            payload["timestamps"] = timestamps

        resp = requests.post(
            f"{extractor_url}/extract",
            json=payload,
            timeout=120,
        )
        if resp.status_code >= 400:
            return jsonify({"error": f"frame extractor failed ({resp.status_code}): {resp.text}"}), 502
        data = resp.json()
        frames = data.get("frames") if isinstance(data, dict) else None
        if not isinstance(frames, list):
            return jsonify({"error": "Invalid frame extractor response"}), 502

        min_conf = _parse_float(os.getenv("ENVID_METADATA_OCR_MIN_CONF"), 0.55)
        min_token_len = _parse_int(os.getenv("ENVID_METADATA_OCR_MIN_TOKEN_LEN"), 2, min_value=1, max_value=10)
        dedupe_window = _parse_float(os.getenv("ENVID_METADATA_OCR_DEDUPE_WINDOW_SECONDS"), interval * 1.5)
        last_text = ""
        last_time = -1e9

        engine = (os.getenv("ENVID_METADATA_OCR_ENGINE") or "tesseract").strip().lower()
        if engine == "auto":
            engine = "tesseract"

        results: List[Dict[str, Any]] = []
        for f in frames:
            if not isinstance(f, dict):
                continue
            path = f.get("path")
            if not isinstance(path, str) or not path:
                continue
            try:
                img = Image.open(path)
                best_text = ""
                best_conf = 0.0

                for _, region in _candidate_regions(img):
                    processed = _preprocess_image(region)
                    text, conf = _run_tesseract(processed)
                    text = _normalize_text(text)
                    if not text:
                        continue
                    if conf > best_conf:
                        best_text = text
                        best_conf = conf

                if not best_text:
                    continue

                if len(best_text) < min_token_len:
                    continue
                if best_conf < min_conf:
                    continue
                if not _is_plausible_text(best_text):
                    continue

                t = float(f.get("time") or 0.0)
                if best_text == last_text and (t - last_time) <= dedupe_window:
                    continue

                results.append({"time": t, "text": best_text, "confidence": best_conf})
                last_text = best_text
                last_time = t
            except Exception:
                continue

        out_dir = Path("/mnt/gcs/envid/work") / (job_id or "job") / "ocr"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"ocr_{job_id or 'job'}.json"
        out_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

        return jsonify({"text": results, "output_path": str(out_path), "frames_dir": data.get("frames_dir")}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5083")))
