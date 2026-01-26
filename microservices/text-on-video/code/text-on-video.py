from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests
from flask import Flask, jsonify, request
from PIL import Image  # type: ignore
import pytesseract  # type: ignore

app = Flask(__name__)


def _frame_extractor_url() -> str:
    return (os.getenv("ENVID_FRAME_EXTRACTOR_URL") or "http://frame-extractor:5093").strip().rstrip("/")


def _ensure_tmp_path(raw: str) -> Path:
    p = Path(raw).resolve()
    if not str(p).startswith("/tmp/"):
        raise RuntimeError("video_path must be under /tmp")
    if not p.exists() or not p.is_file():
        raise RuntimeError("video_path not found")
    return p


@app.get("/health")
def health() -> Any:
    return jsonify({"ok": True}), 200


@app.post("/ocr")
def ocr() -> Any:
    payload = request.get_json(silent=True) or {}
    video_path_raw = str(payload.get("video_path") or "").strip()
    if not video_path_raw:
        return jsonify({"error": "Missing video_path"}), 400

    interval = float(payload.get("interval_seconds") or 2)
    max_frames = int(payload.get("max_frames") or 120)
    job_id = str(payload.get("job_id") or "").strip()

    try:
        _ensure_tmp_path(video_path_raw)
        extractor_url = _frame_extractor_url()
        resp = requests.post(
            f"{extractor_url}/extract",
            json={
                "video_path": video_path_raw,
                "interval_seconds": interval,
                "max_frames": max_frames,
                "job_id": job_id,
            },
            timeout=120,
        )
        if resp.status_code >= 400:
            return jsonify({"error": f"frame extractor failed ({resp.status_code}): {resp.text}"}), 502
        data = resp.json()
        frames = data.get("frames") if isinstance(data, dict) else None
        if not isinstance(frames, list):
            return jsonify({"error": "Invalid frame extractor response"}), 502

        results: List[Dict[str, Any]] = []
        for f in frames:
            if not isinstance(f, dict):
                continue
            path = f.get("path")
            if not isinstance(path, str) or not path:
                continue
            try:
                img = Image.open(path)
                text = pytesseract.image_to_string(img) or ""
                text = text.strip()
                if not text:
                    continue
                results.append({"time": float(f.get("time") or 0.0), "text": text, "confidence": 0.0})
            except Exception:
                continue

        out_path = Path("/tmp") / f"ocr_{job_id or 'job'}.json"
        out_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

        return jsonify({"text": results, "output_path": str(out_path), "frames_dir": data.get("frames_dir")}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5083")))
