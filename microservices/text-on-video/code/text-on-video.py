from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2  # type: ignore
import requests
from flask import Flask, jsonify, request
from PIL import Image, ImageEnhance, ImageOps  # type: ignore
import pytesseract  # type: ignore

app = Flask(__name__)

def _frame_extractor_url() -> str:
    return "http://text-on-video:5083"


def _moderation_service_url() -> str:
    return "http://moderation:5081"


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


def _sample_video_frames(*, video_path: Path, interval_seconds: float, max_frames: int) -> List[Tuple[float, Any]]:
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


def _ffprobe_time_base(*, video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=time_base",
        "-of",
        "default=nw=1:nk=1",
        str(video_path),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    raw = (proc.stdout or "").strip()
    if not raw or "/" not in raw:
        return 1.0 / 1000.0
    num, den = raw.split("/", 1)
    try:
        num_f = float(num)
        den_f = float(den)
        if den_f <= 0:
            return 1.0 / 1000.0
        return num_f / den_f
    except Exception:
        return 1.0 / 1000.0


def _sample_video_frames_ffmpeg(
    *, video_path: Path, interval_seconds: float, max_frames: int, out_dir: Path
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if float(interval_seconds) <= 0:
        out_pattern = out_dir / "frame_%010d.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vsync",
            "vfr",
            "-frame_pts",
            "1",
            "-frames:v",
            str(int(max_frames)),
            str(out_pattern),
        ]
        subprocess.run(cmd, check=True)
        frames = sorted(out_dir.glob("frame_*.jpg"))
        if not frames:
            raise RuntimeError("ffmpeg produced no frames")
        time_base = _ffprobe_time_base(video_path=video_path)
        manifest: list[dict[str, Any]] = []
        for frame_path in frames:
            name = frame_path.stem
            pts_raw = name.replace("frame_", "")
            try:
                pts_val = float(pts_raw)
            except Exception:
                pts_val = 0.0
            manifest.append({"time": float(pts_val) * time_base, "path": str(frame_path)})
        return manifest

    out_pattern = out_dir / "frame_%05d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{max(0.01, float(interval_seconds))}",
        "-frames:v",
        str(int(max_frames)),
        str(out_pattern),
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("ffmpeg produced no frames")
    manifest: list[dict[str, Any]] = []
    for idx, frame_path in enumerate(frames):
        manifest.append({"time": float(idx) * float(interval_seconds), "path": str(frame_path)})
    return manifest


def _sample_video_frames_ffmpeg_scene(
    *, video_path: Path, max_frames: int, out_dir: Path, scene_threshold: float
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = out_dir / "frame_%010d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select='gt(scene,{scene_threshold})'",
        "-vsync",
        "vfr",
        "-frame_pts",
        "1",
        "-frames:v",
        str(int(max_frames)),
        str(out_pattern),
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("ffmpeg scene detect produced no frames")
    time_base = _ffprobe_time_base(video_path=video_path)
    manifest: list[dict[str, Any]] = []
    for frame_path in frames:
        name = frame_path.stem
        pts_raw = name.replace("frame_", "")
        try:
            pts_val = float(pts_raw)
        except Exception:
            pts_val = 0.0
        manifest.append({"time": float(pts_val) * time_base, "path": str(frame_path)})
    return manifest


def _extract_frames_at_timestamps_ffmpeg(
    *, video_path: Path, timestamps: list[float], out_dir: Path
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for idx, t in enumerate(timestamps):
        try:
            t_val = float(t)
        except Exception:
            continue
        if t_val < 0:
            continue
        out_path = out_dir / f"frame_{idx:05d}.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{t_val}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        if out_path.exists():
            manifest.append({"time": float(t_val), "path": str(out_path)})
    return manifest


@app.post("/extract")
def extract() -> Any:
    payload = request.get_json(silent=True) or {}
    video_path_raw = str(payload.get("video_path") or "").strip()
    output_kind = str(payload.get("output_kind") or "processed_local").strip() or "processed_local"

    interval = float(payload.get("interval_seconds") or 0)
    max_frames = int(payload.get("max_frames") or 200000)
    job_id = str(payload.get("job_id") or "").strip() or uuid.uuid4().hex

    try:
        if not video_path_raw:
            video_path_raw = _resolve_video_path(job_id, output_kind)
        video_path = _ensure_tmp_path(video_path_raw)
        out_dir = Path("/mnt/gcs/envid/work") / job_id / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_timestamps = payload.get("timestamps")
        timestamps: list[float] = []
        if isinstance(raw_timestamps, list):
            for t in raw_timestamps:
                try:
                    timestamps.append(float(t))
                except Exception:
                    continue

        use_scene = False
        scene_threshold = 0.3

        manifest: list[dict[str, Any]] = []
        if timestamps:
            manifest = _extract_frames_at_timestamps_ffmpeg(
                video_path=video_path,
                timestamps=timestamps[: max_frames if max_frames > 0 else None],
                out_dir=out_dir,
            )

        if (not manifest) and use_scene:
            try:
                manifest = _sample_video_frames_ffmpeg_scene(
                    video_path=video_path,
                    max_frames=max_frames,
                    out_dir=out_dir,
                    scene_threshold=scene_threshold,
                )
            except Exception:
                manifest = []

        if not manifest:
            frames = _sample_video_frames(video_path=video_path, interval_seconds=interval, max_frames=max_frames)
            if frames:
                for idx, (t, bgr) in enumerate(frames):
                    frame_path = out_dir / f"frame_{idx:05d}.jpg"
                    ok = cv2.imwrite(str(frame_path), bgr)
                    if not ok:
                        continue
                    manifest.append({"time": float(t), "path": str(frame_path)})
            else:
                manifest = _sample_video_frames_ffmpeg(
                    video_path=video_path,
                    interval_seconds=interval,
                    max_frames=max_frames,
                    out_dir=out_dir,
                )

        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

        return jsonify({"frames": manifest, "manifest_path": str(manifest_path), "frames_dir": str(out_dir)}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/moderate/path")
def moderate_path() -> Any:
    payload = request.get_json(silent=True) or {}
    video_path_raw = str(payload.get("video_path") or "").strip()
    output_kind = str(payload.get("output_kind") or "processed_local").strip() or "processed_local"

    interval = float(payload.get("interval_seconds") or 2)
    max_frames = int(payload.get("max_frames") or 120)
    job_id = str(payload.get("job_id") or "").strip() or uuid.uuid4().hex

    try:
        if not video_path_raw:
            video_path_raw = _resolve_video_path(job_id, output_kind)
        video_path = _ensure_tmp_path(video_path_raw)
        out_dir = Path("/mnt/gcs/envid/work") / job_id / "moderation" / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_timestamps = payload.get("timestamps")
        timestamps: list[float] = []
        if isinstance(raw_timestamps, list):
            for t in raw_timestamps:
                try:
                    timestamps.append(float(t))
                except Exception:
                    continue

        force_interval = bool(payload.get("force_interval"))
        use_scene = _env_truthy(os.getenv("ENVID_METADATA_MODERATION_SCENE_SAMPLING"), default=True)
        scene_threshold = float(os.getenv("ENVID_METADATA_MODERATION_SCENE_THRESHOLD") or 0.3)

        manifest: list[dict[str, Any]] = []
        if timestamps:
            manifest = _extract_frames_at_timestamps_ffmpeg(
                video_path=video_path,
                timestamps=timestamps[: max_frames if max_frames > 0 else None],
                out_dir=out_dir,
            )

        if (not manifest) and use_scene and not force_interval:
            try:
                manifest = _sample_video_frames_ffmpeg_scene(
                    video_path=video_path,
                    max_frames=max_frames,
                    out_dir=out_dir,
                    scene_threshold=scene_threshold,
                )
            except Exception:
                manifest = []

        if not manifest:
            frames = _sample_video_frames(video_path=video_path, interval_seconds=interval, max_frames=max_frames)
            if frames:
                for idx, (t, bgr) in enumerate(frames):
                    frame_path = out_dir / f"frame_{idx:05d}.jpg"
                    ok = cv2.imwrite(str(frame_path), bgr)
                    if not ok:
                        continue
                    manifest.append({"time": float(t), "path": str(frame_path)})
            else:
                manifest = _sample_video_frames_ffmpeg(
                    video_path=video_path,
                    interval_seconds=interval,
                    max_frames=max_frames,
                    out_dir=out_dir,
                )

        payload_frames = []
        for f in manifest:
            path = f.get("path")
            if not isinstance(path, str) or not path:
                continue
            with open(path, "rb") as handle:
                payload_frames.append(
                    {
                        "time": float(f.get("time") or 0.0),
                        "image_b64": base64.b64encode(handle.read()).decode("ascii"),
                        "image_mime": "image/jpeg",
                    }
                )

        url = f"{_moderation_service_url()}/moderate/frames"
        resp2 = requests.post(url, json={"frames": payload_frames}, timeout=120)
        if resp2.status_code >= 400:
            return jsonify({"error": f"moderation service failed ({resp2.status_code}): {resp2.text}"}), 502
        out = resp2.json()

        out_dir_meta = Path("/mnt/gcs/envid/work") / (job_id or "job") / "moderation"
        out_dir_meta.mkdir(parents=True, exist_ok=True)
        out_path = out_dir_meta / f"moderation_{job_id or 'job'}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        out["output_path"] = str(out_path)
        out["frames_dir"] = str(out_dir)
        return jsonify(out), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/health")
def health() -> Any:
    engine = (os.getenv("ENVID_METADATA_OCR_ENGINE") or "tesseract").strip().lower()
    return (
        jsonify(
            {
                "ok": True,
                "ocr_engine": engine,
                "frame_extractor": True,
            }
        ),
        200,
    )


@app.post("/ocr")
def ocr() -> Any:
    payload = request.get_json(silent=True) or {}
    video_path_raw = str(payload.get("video_path") or "").strip()

    interval = _parse_float(payload.get("interval_seconds"), _parse_float(os.getenv("ENVID_METADATA_OCR_INTERVAL_SECONDS"), 2.0))
    max_frames = _parse_int(
        payload.get("max_frames"),
        _parse_int(os.getenv("ENVID_METADATA_OCR_MAX_FRAMES"), 3000, min_value=1, max_value=50000),
        min_value=1,
        max_value=50000,
    )
    job_id = str(payload.get("job_id") or "").strip()
    output_kind = str(payload.get("output_kind") or "processed_local").strip() or "processed_local"

    try:
        if not video_path_raw:
            video_path_raw = _resolve_video_path(job_id, output_kind)
        _ensure_tmp_path(video_path_raw)
        extractor_url = _frame_extractor_url()
        use_scene_sampling = False
        timestamps: list[float] = []
        raw_ts = payload.get("timestamps") if isinstance(payload, dict) else None
        if isinstance(raw_ts, list) and raw_ts:
            for t in raw_ts[:100000]:
                try:
                    timestamps.append(float(t))
                except Exception:
                    continue
        elif use_scene_sampling:
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
            timeout=900,
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
