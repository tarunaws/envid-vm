from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, List, Tuple

import cv2  # type: ignore
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

# NudeNet 3.4.x uses ONNXRuntime; this service runs in a separate Python runtime
# from the main Envid Multimodal backend.
try:
    from nudenet import NudeDetector  # type: ignore
except Exception as exc:  # pragma: no cover
    NudeDetector = None  # type: ignore
    _NUDENET_IMPORT_ERROR = str(exc)
else:
    _NUDENET_IMPORT_ERROR = None

try:
    from PIL import Image, ImageEnhance, ImageOps  # type: ignore
except Exception as exc:  # pragma: no cover
    Image = None  # type: ignore
    _PIL_IMPORT_ERROR = str(exc)
else:
    _PIL_IMPORT_ERROR = None

try:
    import onnxruntime as ort  # type: ignore
except Exception as exc:  # pragma: no cover
    ort = None  # type: ignore
    _ORT_IMPORT_ERROR = str(exc)
else:
    _ORT_IMPORT_ERROR = None

app = Flask(__name__)
CORS(app)

_CLASSIFIER: Any = None
_CLASSIFIER_PROVIDERS: list[str] | None = None


def _preprocess_image(img: Image.Image) -> Image.Image:
    upscale = 1.2
    contrast = 1.3
    sharpen = 1.1
    try:
        upscale = float(os.getenv("ENVID_METADATA_MODERATION_UPSCALE") or upscale)
    except Exception:
        upscale = 1.2
    try:
        contrast = float(os.getenv("ENVID_METADATA_MODERATION_CONTRAST") or contrast)
    except Exception:
        contrast = 1.3
    try:
        sharpen = float(os.getenv("ENVID_METADATA_MODERATION_SHARPEN") or sharpen)
    except Exception:
        sharpen = 1.1

    if upscale < 1.0:
        upscale = 1.0
    if contrast < 0.5:
        contrast = 0.5
    if sharpen < 0.5:
        sharpen = 0.5

    img = ImageOps.exif_transpose(img)
    if upscale > 1.0:
        w, h = img.size
        img = img.resize((int(w * upscale), int(h * upscale)), resample=Image.LANCZOS)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if sharpen != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpen)
    return img


def _get_detector() -> Any:
    global _CLASSIFIER
    global _CLASSIFIER_PROVIDERS
    if NudeDetector is None:
        raise RuntimeError(f"nudenet import failed: {_NUDENET_IMPORT_ERROR}")
    if _CLASSIFIER is None:
        providers: list[str] = []
        if ort is not None:
            available = [p for p in ort.get_available_providers() if isinstance(p, str)]
        else:
            available = []

        env_providers = (os.getenv("ENVID_METADATA_MODERATION_PROVIDERS") or "").strip()
        if env_providers:
            providers = [p.strip() for p in env_providers.split(",") if p.strip()]
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        try:
            _CLASSIFIER = NudeDetector(providers=providers)
            _CLASSIFIER_PROVIDERS = providers
        except TypeError:
            _CLASSIFIER = NudeDetector()
            _CLASSIFIER_PROVIDERS = None
    return _CLASSIFIER


def _likelihood_from_score(score: float) -> str:
    if score >= 0.90:
        return "VERY_LIKELY"
    if score >= 0.70:
        return "LIKELY"
    if score >= 0.40:
        return "POSSIBLE"
    if score >= 0.20:
        return "UNLIKELY"
    return "VERY_UNLIKELY"


def _severity_from_score(score: float) -> str:
    # Neutral, model-agnostic scale (no provider-specific buckets).
    if score >= 0.90:
        return "critical"
    if score >= 0.70:
        return "high"
    if score >= 0.40:
        return "medium"
    if score >= 0.20:
        return "low"
    return "minimal"


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


def _env_truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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


def _detect_explicit_frames(*, image_paths: list[Path], times: list[float]) -> list[dict[str, Any]]:
    if Image is None:
        raise RuntimeError(f"pillow import failed: {_PIL_IMPORT_ERROR}")

    detector = _get_detector()

    explicit_labels = {
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_EXPOSED",
        "BUTTOCKS_EXPOSED",
    }
    reason_min_score = 0.25
    try:
        reason_min_score = float(os.getenv("ENVID_METADATA_MODERATION_REASON_MIN_SCORE") or 0.25)
    except Exception:
        reason_min_score = 0.25

    detections_by_frame: list[list[dict[str, Any]]] = []
    try:
        detections_by_frame = detector.detect_batch([str(p) for p in image_paths], batch_size=4)
    except Exception:
        detections_by_frame = []
        for p in image_paths:
            try:
                det = detector.detect(str(p))
                detections_by_frame.append(det if isinstance(det, list) else [])
            except Exception:
                detections_by_frame.append([])

    explicit_frames: list[dict[str, Any]] = []
    for dets, t in zip(detections_by_frame, times):
        unsafe = 0.0
        reasons: list[dict[str, Any]] = []
        if isinstance(dets, list):
            for d in dets:
                if not isinstance(d, dict):
                    continue
                label = d.get("class")
                score = d.get("score")
                if label in explicit_labels:
                    try:
                        score_val = float(score or 0.0)
                    except Exception:
                        score_val = 0.0
                    unsafe = max(unsafe, score_val)
                    if score_val >= reason_min_score:
                        reasons.append({"label": str(label), "score": score_val})
        safe = max(0.0, 1.0 - unsafe)
        reasons.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        top_reason = reasons[0] if reasons else None
        explicit_frames.append(
            {
                "time": float(t),
                "severity": _severity_from_score(unsafe),
                "unsafe": unsafe,
                "safe": safe,
                **({"reasons": reasons} if reasons else {}),
                **({"top_label": top_reason.get("label"), "top_score": top_reason.get("score")} if top_reason else {}),
            }
        )

    return explicit_frames


@app.get("/health")
def health() -> Any:
    return (
        jsonify(
            {
                "status": "ok",
                "service": "moderation",
                "has_nudenet": NudeDetector is not None,
                "nudenet_import_error": _NUDENET_IMPORT_ERROR,
                "has_pillow": Image is not None,
                "pillow_import_error": _PIL_IMPORT_ERROR,
                "onnxruntime_import_error": _ORT_IMPORT_ERROR,
                "onnxruntime_available_providers": (
                    [p for p in ort.get_available_providers() if isinstance(p, str)]
                    if ort is not None
                    else []
                ),
                "onnxruntime_selected_providers": _CLASSIFIER_PROVIDERS,
            }
        ),
        200,
    )


@app.post("/moderate/frames")
def moderate_frames() -> Any:
    """Accept frames as base64 JPEG/PNG and return explicit_frames.

    Request JSON:
      {
        "frames": [ {"time": 12.3, "image_b64": "...", "image_mime": "image/jpeg"}, ... ]
      }

    Response JSON:
      {
                "explicit_frames": [ {"time": 12.3, "severity": "high", "unsafe": 0.81, "safe": 0.19}, ... ]
      }
    """

    payload = request.get_json(silent=True) or {}
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        return jsonify({"error": "Missing frames[]"}), 400

    if Image is None:
        return jsonify({"error": f"pillow import failed: {_PIL_IMPORT_ERROR}"}), 500

    tmp_dir = Path(tempfile.mkdtemp(prefix="nudenet_frames_"))
    try:
        image_paths: list[Path] = []
        times: list[float] = []

        max_input_frames = 200000
        for idx, fr in enumerate(frames[:max_input_frames]):
            if not isinstance(fr, dict):
                continue
            try:
                t = float(fr.get("time") or 0.0)
            except Exception:
                t = 0.0

            b64 = fr.get("image_b64")
            if not isinstance(b64, str) or not b64.strip():
                continue

            mime = (fr.get("image_mime") or "image/jpeg").strip().lower()
            ext = ".jpg" if "jpeg" in mime or "jpg" in mime else ".png"

            try:
                raw = base64.b64decode(b64)
                img = Image.open(io.BytesIO(raw))
                img = img.convert("RGB")
                img = _preprocess_image(img)
            except Exception:
                continue

            p = tmp_dir / f"frame_{idx:05d}{ext}"
            try:
                img.save(p, format="JPEG" if ext == ".jpg" else "PNG")
            except Exception:
                continue

            image_paths.append(p)
            times.append(t)

        if not image_paths:
            return jsonify({"explicit_frames": []}), 200

        explicit_frames = _detect_explicit_frames(image_paths=image_paths, times=times)
        return jsonify({"explicit_frames": explicit_frames}), 200
    finally:
        try:
            for child in tmp_dir.iterdir():
                try:
                    child.unlink()
                except Exception:
                    pass
            tmp_dir.rmdir()
        except Exception:
            pass


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

        image_paths: list[Path] = []
        times: list[float] = []
        for item in manifest:
            path = item.get("path")
            if not isinstance(path, str) or not path:
                continue
            try:
                t_val = float(item.get("time") or 0.0)
            except Exception:
                t_val = 0.0
            p = Path(path)
            if p.exists():
                image_paths.append(p)
                times.append(t_val)

        explicit_frames = _detect_explicit_frames(image_paths=image_paths, times=times)

        out_dir_meta = Path("/mnt/gcs/envid/work") / (job_id or "job") / "moderation"
        out_dir_meta.mkdir(parents=True, exist_ok=True)
        out_path = out_dir_meta / f"moderation_{job_id or 'job'}.json"
        out_path.write_text(json.dumps({"explicit_frames": explicit_frames}, ensure_ascii=False), encoding="utf-8")

        return jsonify({"explicit_frames": explicit_frames, "output_path": str(out_path), "frames_dir": str(out_dir)}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
