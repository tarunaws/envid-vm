from __future__ import annotations

import base64
import io
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    import numpy as np  # type: ignore
except Exception as exc:  # pragma: no cover
    np = None  # type: ignore
    _NUMPY_IMPORT_ERROR = str(exc)
else:
    _NUMPY_IMPORT_ERROR = None

try:
    from PIL import Image  # type: ignore
except Exception as exc:  # pragma: no cover
    Image = None  # type: ignore
    _PIL_IMPORT_ERROR = str(exc)
else:
    _PIL_IMPORT_ERROR = None

try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception as exc:  # pragma: no cover
    PaddleOCR = None  # type: ignore
    _PADDLEOCR_IMPORT_ERROR = str(exc)
else:
    _PADDLEOCR_IMPORT_ERROR = None

try:
    import paddle  # type: ignore
except Exception as exc:  # pragma: no cover
    paddle = None  # type: ignore
    _PADDLE_IMPORT_ERROR = str(exc)
else:
    _PADDLE_IMPORT_ERROR = None

app = Flask(__name__)
CORS(app)

PORT = int(os.getenv("PORT") or "5084")
DEFAULT_LANG = (os.getenv("PADDLEOCR_LANG") or "en").strip() or "en"

_OCR_BY_LANG: Dict[str, Any] = {}

_MODEL_LOADED = False
_MODEL_LOAD_ERROR: str | None = None


@dataclass
class _FrameStats:
    frames_seen: int = 0
    frames_decoded: int = 0
    ocr_calls: int = 0
    ocr_exceptions: int = 0
    decode_exceptions: int = 0
    parsed_lines: int = 0
    kept_lines: int = 0


def _is_truthy_env(name: str, default: str = "0") -> bool:
    v = (os.getenv(name) or default).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _normalize_paddleocr_result(res: Any) -> List[Any]:
    """Return a flat list of OCR line items.

    PaddleOCR may return either:
    - [ [box, (text, score)], ... ]
    - [ [ [box, (text, score)], ... ] ]  (outer list per-image)
    """
    if not isinstance(res, list) or not res:
        return []

    first = res[0]
    if isinstance(first, list) and first:
        first0 = first[0]
        # Nested: first0 is a detection like [box, (text, score)] (len==2)
        if isinstance(first0, (list, tuple)) and len(first0) == 2:
            return list(first)

    return list(res)


def _get_ocr(lang: str) -> Any:
    if PaddleOCR is None:
        raise RuntimeError(f"paddleocr import failed: {_PADDLEOCR_IMPORT_ERROR}")
    if paddle is None:
        raise RuntimeError(f"paddle import failed: {_PADDLE_IMPORT_ERROR}")

    key = (lang or DEFAULT_LANG).strip() or DEFAULT_LANG
    if key not in _OCR_BY_LANG:
        # Note: init is expensive; keep it cached.
        _OCR_BY_LANG[key] = PaddleOCR(use_angle_cls=True, lang=key)
    return _OCR_BY_LANG[key]


def _warm_load_default_model() -> None:
    global _MODEL_LOADED, _MODEL_LOAD_ERROR
    try:
        _get_ocr(DEFAULT_LANG)
        _MODEL_LOADED = True
        _MODEL_LOAD_ERROR = None
    except Exception as exc:  # pragma: no cover
        _MODEL_LOADED = False
        _MODEL_LOAD_ERROR = str(exc)


def _aggregate_text_segments(hits: List[Tuple[str, float, float, float]], max_items: int = 250) -> List[Dict[str, Any]]:
    by_text: Dict[str, List[Dict[str, Any]]] = {}
    for txt, st, en, conf in hits:
        t = (txt or "").strip()
        if not t:
            continue
        by_text.setdefault(t, []).append({"start": float(st), "end": float(en), "confidence": float(conf)})

    out: List[Dict[str, Any]] = []
    for txt, segs in by_text.items():
        segs.sort(key=lambda x: (float(x.get("start") or 0.0), float(x.get("end") or 0.0)))
        out.append({"text": txt, "segments": segs})

    out.sort(key=lambda x: (len(x.get("segments") or []), len(str(x.get("text") or ""))), reverse=True)
    return out[:max_items]


def _decode_image_b64_to_bgr(image_b64: str) -> Any:
    if Image is None:
        raise RuntimeError(f"pillow import failed: {_PIL_IMPORT_ERROR}")
    if np is None:
        raise RuntimeError(f"numpy import failed: {_NUMPY_IMPORT_ERROR}")

    s = image_b64.strip()
    # Allow data URIs like: data:image/jpeg;base64,....
    if "," in s and s.lower().startswith("data:"):
        s = s.split(",", 1)[1].strip()

    raw = base64.b64decode(s)
    img = Image.open(io.BytesIO(raw))
    img.load()
    img = img.convert("RGB")
    rgb = np.array(img)  # shape (h,w,3) RGB
    # Convert RGB -> BGR
    bgr = rgb[:, :, ::-1]
    return bgr


@app.get("/health")
def health() -> Any:
    return (
        jsonify(
            {
                "status": "ok" if (PaddleOCR is not None and paddle is not None) else "degraded",
                "service": "local-ocr-paddle",
                "has_paddleocr": PaddleOCR is not None,
                "paddleocr_import_error": _PADDLEOCR_IMPORT_ERROR,
                "has_paddle": paddle is not None,
                "paddle_import_error": _PADDLE_IMPORT_ERROR,
                "has_pillow": Image is not None,
                "pillow_import_error": _PIL_IMPORT_ERROR,
                "has_numpy": np is not None,
                "numpy_import_error": _NUMPY_IMPORT_ERROR,
                "default_lang": DEFAULT_LANG,
                "loaded_langs": sorted(_OCR_BY_LANG.keys()),
                "model_loaded": _MODEL_LOADED,
                "model_error": _MODEL_LOAD_ERROR,
            }
        ),
        200,
    )


@app.post("/ocr/frames")
def ocr_frames() -> Any:
    payload = request.get_json(silent=True) or {}
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        return jsonify({"error": "Missing frames[]"}), 400

    try:
        frame_len = float(payload.get("frame_len") or payload.get("interval_seconds") or 1.0)
    except Exception:
        frame_len = 1.0
    frame_len = max(0.1, min(30.0, frame_len))

    lang = (payload.get("lang") or DEFAULT_LANG)
    try:
        ocr = _get_ocr(str(lang))
    except Exception as exc:
        return jsonify({"error": str(exc)[:240]}), 500

    hits: List[Tuple[str, float, float, float]] = []
    stats = _FrameStats()
    include_debug = _is_truthy_env("DEBUG_OCR") or bool(payload.get("debug"))

    # Decode + run OCR per frame
    for fr in frames[:2000]:
        stats.frames_seen += 1
        if not isinstance(fr, dict):
            continue
        try:
            t = float(fr.get("time") or 0.0)
        except Exception:
            t = 0.0

        b64 = fr.get("image_b64")
        if not isinstance(b64, str) or not b64.strip():
            continue

        try:
            bgr = _decode_image_b64_to_bgr(b64)
            stats.frames_decoded += 1
        except Exception:
            stats.decode_exceptions += 1
            continue

        try:
            stats.ocr_calls += 1
            res = ocr.ocr(bgr, cls=True)
        except Exception:
            stats.ocr_exceptions += 1
            continue

        lines = _normalize_paddleocr_result(res)
        stats.parsed_lines += len(lines)

        for line in lines:
            try:
                txt = str(line[1][0] or "").strip()
                conf = float(line[1][1] or 0.0)
            except Exception:
                continue
            if not txt:
                continue
            hits.append((txt, float(t), float(t + frame_len), conf))
            stats.kept_lines += 1

    resp: Dict[str, Any] = {"text": _aggregate_text_segments(hits)}
    if include_debug:
        resp["debug"] = {
            **stats.__dict__,
            "lang": str(lang),
            "frame_len": float(frame_len),
            "frames_received": len(frames),
        }

    return jsonify(resp), 200


if __name__ == "__main__":
    # Warm-load in background so the first OCR request doesn't pay model download/init cost.
    threading.Thread(target=_warm_load_default_model, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)
