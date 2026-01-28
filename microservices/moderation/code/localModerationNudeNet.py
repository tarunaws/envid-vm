from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path
from typing import Any

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

    detector = _get_detector()

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

        explicit_frames: list[dict[str, Any]] = []
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
