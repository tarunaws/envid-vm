from __future__ import annotations

import base64
import os
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


SCORE_THRESH = _env_float("LABEL_SERVICE_SCORE_THRESH", 0.5)


def _d2_score_thresh() -> float:
    return _env_float("DETECTRON2_SCORE_THRESH", SCORE_THRESH)


def _mmdet_score_thresh() -> float:
    return _env_float("MMDET_SCORE_THRESH", SCORE_THRESH)


_COCO_THING_CLASSES: List[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


# Avoid importing heavy libs at import-time.

def _try_import_detectron2() -> Tuple[bool, str]:
    try:
        import detectron2  # noqa: F401

        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _try_import_mmdet() -> Tuple[bool, str]:
    try:
        import mmdet  # noqa: F401

        # MMDetection can import even when MMCV ops aren't available.
        # Validate ops availability by importing a representative op.
        try:
            from mmcv.ops import nms  # type: ignore  # noqa: F401
        except Exception as exc:
            return False, f"mmcv ops missing/unavailable: {exc}"

        return True, "ok"
    except Exception as exc:
        return False, str(exc)


@lru_cache(maxsize=4)
def _detectron2_predictor():
    from detectron2.config import get_cfg  # type: ignore
    from detectron2.engine import DefaultPredictor  # type: ignore
    from detectron2.model_zoo import get_checkpoint_url, get_config_file  # type: ignore

    cfg = get_cfg()
    # Default COCO detector; override via env if desired.
    model_cfg = (os.getenv("DETECTRON2_MODEL_ZOO_CONFIG") or "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml").strip()
    cfg.merge_from_file(get_config_file(model_cfg))
    cfg.MODEL.WEIGHTS = os.getenv("DETECTRON2_WEIGHTS") or get_checkpoint_url(model_cfg)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = _d2_score_thresh()
    cfg.MODEL.DEVICE = os.getenv("DETECTRON2_DEVICE") or "cpu"
    return DefaultPredictor(cfg)


@lru_cache(maxsize=2)
def _detectron2_class_names() -> List[str]:
    from detectron2.data import MetadataCatalog  # type: ignore

    dataset = (os.getenv("DETECTRON2_DATASET") or "coco_2017_val").strip()
    try:
        meta = MetadataCatalog.get(dataset)
        classes = list(getattr(meta, "thing_classes", []) or [])
        classes = [str(c) for c in classes if str(c).strip()]
        if classes:
            return classes
    except Exception:
        pass

    # Fallback: COCO thing classes (matches Detectron2 model zoo COCO detectors).
    return list(_COCO_THING_CLASSES)


@lru_cache(maxsize=4)
def _mmdet_inferencer():
    """Return an MMDetection inferencer.

    Supports two modes:
    - Explicit config+checkpoint via MMDET_CONFIG/MMDET_CHECKPOINT.
    - Out-of-the-box default model via MMDET_MODEL (or a safe default).
    """

    from mmdet.apis import DetInferencer  # type: ignore

    cfg_path = (os.getenv("MMDET_CONFIG") or "").strip()
    ckpt_path = (os.getenv("MMDET_CHECKPOINT") or "").strip()
    device = (os.getenv("MMDET_DEVICE") or "cpu").strip()

    if cfg_path:
        weights = ckpt_path or None
        return DetInferencer(model=cfg_path, weights=weights, device=device)

    model_name = (os.getenv("MMDET_MODEL") or "rtmdet_tiny_8xb32-300e_coco").strip()
    return DetInferencer(model=model_name, device=device)


def _mmdet_class_names(inferencer) -> List[str]:
    meta = None
    try:
        meta = getattr(getattr(inferencer, "model", None), "dataset_meta", None)
    except Exception:
        meta = None
    classes = []
    if isinstance(meta, dict):
        classes = meta.get("classes") or meta.get("CLASSES") or []
    return [str(c) for c in (classes or [])]


def _mmdet_run_dict(inferencer, rgb: np.ndarray) -> Dict[str, Any]:
    """Run inferencer and return a dict output.

    MMDetection has had minor API variations across versions; try a few call patterns.
    """

    call_attempts = [
        ([], {"return_vis": False, "return_datasamples": False}),
        ([], {"return_vis": False}),
        ([], {"return_datasamples": False}),
        ([], {}),
    ]
    for args, kwargs in call_attempts:
        try:
            out = inferencer([rgb], *args, **kwargs)
            if isinstance(out, dict):
                return out
        except TypeError:
            continue

    # Fallback: try without wrapping in a list.
    for args, kwargs in call_attempts:
        try:
            out = inferencer(rgb, *args, **kwargs)
            if isinstance(out, dict):
                return out
        except TypeError:
            continue

    raise RuntimeError("MMDetection inferencer returned an unexpected output")


def _aggregate_label_hits(hits: List[Tuple[str, float, float, float]], max_items: int = 250) -> List[Dict[str, Any]]:
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for lbl, st, en, conf in hits:
        k = (lbl or "").strip()
        if not k:
            continue
        by_label.setdefault(k, []).append({"start": float(st), "end": float(en), "confidence": float(conf)})

    out: List[Dict[str, Any]] = []
    for lbl, segs in by_label.items():
        segs.sort(key=lambda x: (float(x.get("start") or 0.0), float(x.get("end") or 0.0)))
        out.append({"label": lbl, "segments": segs})

    out.sort(key=lambda x: (len(x.get("segments") or []), len(str(x.get("label") or ""))), reverse=True)
    return out[:max_items]


@app.get("/health")
def health():
    has_d2, d2_err = _try_import_detectron2()
    has_mm, mm_err = _try_import_mmdet()
    ready = bool(has_d2 or has_mm)
    return jsonify(
        {
            "ok": True,
            "ready": ready,
            "has_detectron2": has_d2,
            "has_mmdet": has_mm,
            "detectron2_error": None if has_d2 else d2_err[:200],
            "mmdet_error": None if has_mm else mm_err[:200],
            "score_thresh": SCORE_THRESH,
        }
    )


@app.post("/detect/frames")
def detect_frames():
    body = request.get_json(silent=True) or {}
    model = str(body.get("model") or "").strip().lower()
    frames = body.get("frames")
    frame_len = float(body.get("frame_len") or 1.0)
    debug = bool(body.get("debug"))

    if model not in {"detectron2", "mmdetection"}:
        return jsonify({"error": "model must be detectron2 or mmdetection"}), 400
    if not isinstance(frames, list) or not frames:
        return jsonify({"error": "frames must be a non-empty list"}), 400

    # Decode frames
    decoded: List[Tuple[float, np.ndarray]] = []
    for f in frames:
        if not isinstance(f, dict):
            continue
        t = float(f.get("time") or 0.0)
        b64 = f.get("image_b64")
        if not isinstance(b64, str) or not b64:
            continue
        try:
            import io

            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw))
            img = img.convert("RGB")
            arr = np.asarray(img)
            decoded.append((t, arr))
        except Exception:
            continue

    if not decoded:
        return jsonify({"error": "no decodable frames"}), 400

    hits: List[Tuple[str, float, float, float]] = []
    debug_info: Dict[str, Any] = {}

    if model == "detectron2":
        predictor = _detectron2_predictor()
        classes = _detectron2_class_names()
        d2_thresh = _d2_score_thresh()
        if debug:
            debug_info["detectron2"] = {
                "score_thresh": d2_thresh,
                "classes_count": len(classes),
                "model_cfg": (os.getenv("DETECTRON2_MODEL_ZOO_CONFIG") or "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml").strip(),
            }

        per_frame: List[Dict[str, Any]] = []
        for t, rgb in decoded:
            # Detectron2's DefaultPredictor expects BGR images (OpenCV convention).
            # Our decoded frames are RGB (PIL). Convert to BGR to avoid degraded detections.
            if isinstance(rgb, np.ndarray) and rgb.ndim == 3 and rgb.shape[-1] == 3:
                bgr = rgb[:, :, ::-1].copy()
            else:
                bgr = rgb

            outputs = predictor(bgr)
            inst = outputs.get("instances")
            if inst is None:
                continue
            try:
                # Ensure CPU tensors before extracting.
                inst = inst.to("cpu")
                pred_classes = inst.pred_classes.tolist()
                scores = inst.scores.tolist()
            except Exception:
                continue

            if debug:
                # Capture top scores to understand if we're just under the threshold.
                top = sorted(
                    (
                        {
                            "cls": int(cls_idx),
                            "label": (classes[int(cls_idx)] if 0 <= int(cls_idx) < len(classes) else str(int(cls_idx))),
                            "score": float(score),
                        }
                        for cls_idx, score in zip(pred_classes, scores)
                    ),
                    key=lambda x: x["score"],
                    reverse=True,
                )
                per_frame.append({"time": float(t), "n": len(top), "top": top[:8]})

            for cls_idx, score in zip(pred_classes, scores):
                # Predictor already applies cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, but keep a consistent
                # explicit filter here in case config changes.
                if float(score) < d2_thresh:
                    continue
                label = classes[int(cls_idx)] if int(cls_idx) < len(classes) else str(int(cls_idx))
                hits.append((label, float(t), float(t + frame_len), float(score)))

        if debug:
            debug_info["detectron2"]["frames"] = per_frame

    if model == "mmdetection":
        try:
            inferencer = _mmdet_inferencer()
            class_names = _mmdet_class_names(inferencer)
            mm_thresh = _mmdet_score_thresh()
            for t, rgb in decoded:
                out = _mmdet_run_dict(inferencer, rgb)
                preds = out.get("predictions")
                if not isinstance(preds, list) or not preds:
                    continue
                pred0 = preds[0]
                if not isinstance(pred0, dict):
                    continue
                labels = pred0.get("labels") or []
                scores = pred0.get("scores") or []
                if not isinstance(labels, list) or not isinstance(scores, list):
                    continue
                for cls_idx, score in zip(labels, scores):
                    try:
                        score_f = float(score)
                    except Exception:
                        continue
                    if score_f < mm_thresh:
                        continue
                    try:
                        idx = int(cls_idx)
                    except Exception:
                        idx = -1
                    label = class_names[idx] if 0 <= idx < len(class_names) else str(cls_idx)
                    hits.append((label, float(t), float(t + frame_len), float(score_f)))
        except Exception as exc:
            # Make the most common failure mode actionable.
            msg = str(exc)
            if "mmcv" in msg and ("_ext" in msg or "mmcv ops" in msg or "mmcv.ops" in msg):
                msg = "MMDetection requires MMCV ops (mmcv.ops). Install a full MMCV build with ops. Original error: " + msg
            return jsonify({"error": msg}), 500

    labels = _aggregate_label_hits(hits)
    resp: Dict[str, Any] = {"labels": labels}
    if debug:
        resp["debug"] = debug_info
    return jsonify(resp)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT") or 5083))
