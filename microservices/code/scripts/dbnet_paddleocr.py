#!/usr/bin/env python3
"""DBNet (MMOCR) + PaddleOCR helper.

This is intentionally standalone and best-effort: it is meant for local experiments
and for validating the OCR stack outside the Flask pipeline.

Notes:
- Requires heavy deps (mmocr + paddleocr + opencv-python + numpy). These may not
  be available for your Python version/platform.
- For video input, this script uses ffmpeg to sample frames at a fixed interval.

Outputs:
- JSONL: one record per detected text line/region.
- Optional visualization images.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class OcrItem:
    frame: str
    timestamp_ms: int | None
    text: str
    score_det: float
    score_rec: float
    polygon: list[list[float]]

    def to_json(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "timestamp_ms": self.timestamp_ms,
            "text": self.text,
            "score_det": self.score_det,
            "score_rec": self.score_rec,
            "polygon": self.polygon,
        }


def _require_imports() -> tuple[Any, Any, Any, Any]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Missing deps: opencv-python and numpy are required: {exc}")
    try:
        from mmocr.apis import TextDetInferencer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing deps: mmocr is required for DBNet detection. "
            f"If this fails on your Python version, you may need a different Python env. Details: {exc}"
        )
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Missing deps: paddleocr is required for recognition: {exc}")
    return cv2, np, TextDetInferencer, PaddleOCR


def _ffmpeg_path() -> str:
    return os.getenv("FFMPEG_PATH") or "ffmpeg"


def _run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n--- stderr ---\n"
            + (res.stderr.decode("utf-8", errors="replace") if res.stderr else "")
        )


def _iter_images(images_dir: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _extract_video_frames(
    *,
    video_path: Path,
    out_dir: Path,
    interval_seconds: int,
    max_frames: int,
) -> list[tuple[Path, int]]:
    """Extract frames and return (frame_path, timestamp_ms)."""
    interval_seconds = max(1, int(interval_seconds))
    max_frames = max(1, int(max_frames))

    # Use fps filter: 1 frame every N seconds => fps=1/N
    # This does not guarantee perfect timestamp alignment; we approximate by index.
    out_pat = out_dir / "frame_%06d.jpg"
    cmd = [
        _ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval_seconds}",
        "-q:v",
        "2",
        str(out_pat),
        "-y",
    ]
    _run(cmd)

    frames = sorted(out_dir.glob("frame_*.jpg"))
    frames = frames[:max_frames]
    out: list[tuple[Path, int]] = []
    for idx, fp in enumerate(frames):
        ts_ms = idx * interval_seconds * 1000
        out.append((fp, ts_ms))
    return out


def _ensure_quad(np: Any, cv2: Any, poly: Any) -> list[list[float]] | None:
    try:
        pts = np.array(poly, dtype=np.float32)
    except Exception:
        return None
    if pts.ndim != 2 or pts.shape[0] < 4:
        return None
    if pts.shape[0] != 4:
        try:
            rect = cv2.minAreaRect(pts.astype(np.float32))
            pts = cv2.boxPoints(rect)
        except Exception:
            return None
    return pts.astype(float).tolist()


def _order_quad(np: Any, pts: Any) -> Any:
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmin(d))]
    bl = pts[int(np.argmax(d))]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _warp_crop(cv2: Any, np: Any, img_bgr: Any, quad: list[list[float]], pad: int = 2) -> Any:
    pts = np.array(quad, dtype=np.float32)
    pts = _order_quad(np, pts)
    w = int(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])))
    w = max(2, min(w, img_bgr.shape[1]))
    h = max(2, min(h, img_bgr.shape[0]))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(img_bgr, M, (w, h))
    if pad > 0:
        crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return crop


def run_ocr_on_image(
    *,
    image_path: Path,
    timestamp_ms: int | None,
    det: Any,
    rec: Any,
    cv2: Any,
    np: Any,
    max_polys: int,
) -> list[OcrItem]:
    b = image_path.read_bytes()
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    det_out = det(img, return_vis=False)
    preds = det_out.get("predictions") if isinstance(det_out, dict) else None
    if not isinstance(preds, list) or not preds:
        return []
    pred0 = preds[0] if isinstance(preds[0], dict) else {}
    polys = pred0.get("polygons") if isinstance(pred0, dict) else None
    scores = pred0.get("scores") if isinstance(pred0, dict) else None
    if not isinstance(polys, list):
        return []

    items: list[OcrItem] = []
    for j, poly in enumerate(polys[: max(1, int(max_polys))]):
        quad = _ensure_quad(np, cv2, poly)
        if not quad:
            continue
        det_score = 0.0
        if isinstance(scores, list) and j < len(scores):
            try:
                det_score = float(scores[j] or 0.0)
            except Exception:
                det_score = 0.0

        crop = _warp_crop(cv2, np, img, quad, pad=2)
        rec_out = rec.ocr(crop, cls=True, det=False)
        if not rec_out or not isinstance(rec_out, list) or not rec_out[0]:
            continue
        try:
            text = str(rec_out[0][0][0] or "").strip()
            rec_score = float(rec_out[0][0][1] or 0.0)
        except Exception:
            continue
        if not text:
            continue

        items.append(
            OcrItem(
                frame=image_path.name,
                timestamp_ms=timestamp_ms,
                text=text,
                score_det=float(det_score),
                score_rec=float(rec_score),
                polygon=quad,
            )
        )

    return items


def _draw_poly(cv2: Any, np: Any, img: Any, poly: list[list[float]], color: tuple[int, int, int]) -> None:
    try:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color, 2)
    except Exception:
        return


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Run DBNet (MMOCR) + PaddleOCR on images or video")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images-dir", type=str, help="Directory containing images")
    src.add_argument("--video", type=str, help="Path to video file")

    p.add_argument("--out-jsonl", type=str, required=True, help="Output JSONL path")
    p.add_argument("--viz-dir", type=str, default=None, help="Optional output directory for visualization frames")

    p.add_argument("--interval-seconds", type=int, default=10, help="Video: sample interval seconds")
    p.add_argument("--max-frames", type=int, default=24, help="Video: max frames to process")
    p.add_argument("--max-polys", type=int, default=200, help="Max polygons per frame to consider")

    p.add_argument("--device", type=str, default="cpu", help="MMOCR device (cpu/cuda)")
    p.add_argument("--lang", type=str, default="en", help="PaddleOCR language")

    args = p.parse_args(argv)

    cv2, np, TextDetInferencer, PaddleOCR = _require_imports()

    det = TextDetInferencer(model="dbnet", device=args.device)
    rec = PaddleOCR(det=False, rec=True, use_angle_cls=True, lang=args.lang)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    viz_dir = Path(args.viz_dir) if args.viz_dir else None
    if viz_dir:
        viz_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir_ctx: tempfile.TemporaryDirectory[str] | None = None
    frames: list[tuple[Path, int | None]] = []

    try:
        if args.images_dir:
            img_dir = Path(args.images_dir)
            if not img_dir.exists() or not img_dir.is_dir():
                raise RuntimeError("--images-dir must be a directory")
            frames = [(p, None) for p in _iter_images(img_dir)]
        else:
            video_path = Path(args.video)
            if not video_path.exists() or not video_path.is_file():
                raise RuntimeError("--video must be a file")
            tmp_dir_ctx = tempfile.TemporaryDirectory(prefix="dbnet_paddleocr_frames_")
            out_dir = Path(tmp_dir_ctx.name)
            extracted = _extract_video_frames(
                video_path=video_path,
                out_dir=out_dir,
                interval_seconds=args.interval_seconds,
                max_frames=args.max_frames,
            )
            frames = [(fp, ts_ms) for fp, ts_ms in extracted]

        with out_path.open("w", encoding="utf-8") as f:
            for idx, (frame_path, ts_ms) in enumerate(frames):
                items = run_ocr_on_image(
                    image_path=frame_path,
                    timestamp_ms=ts_ms,
                    det=det,
                    rec=rec,
                    cv2=cv2,
                    np=np,
                    max_polys=args.max_polys,
                )

                for it in items:
                    f.write(json.dumps(it.to_json(), ensure_ascii=False) + "\n")

                if viz_dir:
                    try:
                        b = frame_path.read_bytes()
                        arr = np.frombuffer(b, dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            for it in items:
                                _draw_poly(cv2, np, img, it.polygon, (0, 255, 0))
                                cv2.putText(
                                    img,
                                    it.text[:48],
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 0),
                                    2,
                                )
                            out_img = viz_dir / f"{idx:06d}_{frame_path.stem}.jpg"
                            cv2.imwrite(str(out_img), img)
                    except Exception:
                        pass

        return 0
    finally:
        if tmp_dir_ctx is not None:
            try:
                tmp_dir_ctx.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
