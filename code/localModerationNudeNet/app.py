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
    from PIL import Image  # type: ignore
except Exception as exc:  # pragma: no cover
    Image = None  # type: ignore
    _PIL_IMPORT_ERROR = str(exc)
else:
    _PIL_IMPORT_ERROR = None

app = Flask(__name__)
CORS(app)

_CLASSIFIER: Any = None


def _get_detector() -> Any:
    global _CLASSIFIER
    if NudeDetector is None:
        raise RuntimeError(f"nudenet import failed: {_NUDENET_IMPORT_ERROR}")
    if _CLASSIFIER is None:
        _CLASSIFIER = NudeDetector()
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
                "service": "local-moderation-nudenet",
                "has_nudenet": NudeDetector is not None,
                "nudenet_import_error": _NUDENET_IMPORT_ERROR,
                "has_pillow": Image is not None,
                "pillow_import_error": _PIL_IMPORT_ERROR,
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

        for idx, fr in enumerate(frames[:2000]):
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
            if isinstance(dets, list):
                for d in dets:
                    if not isinstance(d, dict):
                        continue
                    label = d.get("class")
                    score = d.get("score")
                    if label in explicit_labels:
                        try:
                            unsafe = max(unsafe, float(score or 0.0))
                        except Exception:
                            pass
            safe = max(0.0, 1.0 - unsafe)
            explicit_frames.append(
                {
                    "time": float(t),
                    "severity": _severity_from_score(unsafe),
                    "unsafe": unsafe,
                    "safe": safe,
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
