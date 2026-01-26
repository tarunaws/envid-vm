from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, List, Tuple

import cv2  # type: ignore
from flask import Flask, jsonify, request

app = Flask(__name__)


def _ensure_tmp_path(raw: str) -> Path:
    p = Path(raw).resolve()
    if not str(p).startswith("/tmp/"):
        raise RuntimeError("video_path must be under /tmp")
    if not p.exists() or not p.is_file():
        raise RuntimeError("video_path not found")
    return p


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


@app.get("/health")
def health() -> Any:
    return jsonify({"ok": True}), 200


@app.post("/extract")
def extract() -> Any:
    payload = request.get_json(silent=True) or {}
    video_path_raw = str(payload.get("video_path") or "").strip()
    if not video_path_raw:
        return jsonify({"error": "Missing video_path"}), 400

    interval = float(payload.get("interval_seconds") or 2)
    max_frames = int(payload.get("max_frames") or 120)
    job_id = str(payload.get("job_id") or "").strip() or uuid.uuid4().hex

    try:
        video_path = _ensure_tmp_path(video_path_raw)
        out_dir = Path("/tmp") / f"frames_{job_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        frames = _sample_video_frames(video_path=video_path, interval_seconds=interval, max_frames=max_frames)
        manifest: list[dict[str, Any]] = []
        for idx, (t, bgr) in enumerate(frames):
            frame_path = out_dir / f"frame_{idx:05d}.jpg"
            ok = cv2.imwrite(str(frame_path), bgr)
            if not ok:
                continue
            manifest.append({"time": float(t), "path": str(frame_path)})

        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

        return jsonify({"frames": manifest, "manifest_path": str(manifest_path), "frames_dir": str(out_dir)}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500



import base64
import requests

def _moderation_service_url() -> str:
    return (os.getenv("ENVID_MODERATION_SERVICE_URL") or "http://moderation:5081").strip().rstrip("/")

@app.post("/moderate/path")
def moderate_path() -> Any:
    payload = request.get_json(silent=True) or {}
    video_path_raw = str(payload.get("video_path") or "").strip()
    if not video_path_raw:
        return jsonify({"error": "Missing video_path"}), 400

    interval = float(payload.get("interval_seconds") or 2)
    max_frames = int(payload.get("max_frames") or 120)
    job_id = str(payload.get("job_id") or "").strip() or uuid.uuid4().hex

    try:
        video_path = _ensure_tmp_path(video_path_raw)
        out_dir = Path("/tmp") / f"frames_{job_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        frames = _sample_video_frames(video_path=video_path, interval_seconds=interval, max_frames=max_frames)
        manifest: list[dict[str, Any]] = []
        for idx, (t, bgr) in enumerate(frames):
            frame_path = out_dir / f"frame_{idx:05d}.jpg"
            ok = cv2.imwrite(str(frame_path), bgr)
            if not ok:
                continue
            manifest.append({"time": float(t), "path": str(frame_path)})

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

        out_path = Path("/tmp") / f"moderation_{job_id or 'job'}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        out["output_path"] = str(out_path)
        out["frames_dir"] = str(out_dir)
        return jsonify(out), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5093")))
