from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, request

app = Flask(__name__)


def _parse_hhmmss_to_seconds(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return 0.0
    m = re.match(r"^(?P<h>\d+):(?P<m>\d{2}):(?P<sec>\d{2})(?:\.(?P<ms>\d{1,6}))?$", s)
    if not m:
        return 0.0
    h = int(m.group("h"))
    mi = int(m.group("m"))
    sec = int(m.group("sec"))
    ms_raw = m.group("ms")
    frac = 0.0 if not ms_raw else float(int(ms_raw)) / (10.0 ** len(ms_raw))
    return float(h * 3600 + mi * 60 + sec) + frac


@app.get("/health")
def health() -> Any:
    return jsonify({"ok": True}), 200


@app.post("/pyscenedetect/scenes")
def pyscenedetect_scenes() -> Any:
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    video_file = request.files["video"]
    if not video_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    threshold = float(request.form.get("threshold") or os.getenv("ENVID_METADATA_SCENEDETECT_THRESHOLD") or 27.0)
    max_seconds = int(request.form.get("max_seconds") or 900)

    temp_dir = Path(tempfile.mkdtemp(prefix="scenedetect_"))
    try:
        video_path = temp_dir / (Path(video_file.filename).name or "video.mp4")
        video_file.save(str(video_path))

        out_dir = temp_dir / "scenedetect"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "scenedetect",
            "-i",
            str(video_path),
            "-o",
            str(out_dir),
            "detect-content",
            "--threshold",
            str(threshold),
            "list-scenes",
            "-q",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=max_seconds)
        if res.returncode != 0:
            err = (res.stderr or b"").decode("utf-8", errors="ignore")
            return jsonify({"error": f"scenedetect failed (code {res.returncode}): {err[:240]}"}), 500

        csv_files = sorted(out_dir.glob("*.csv"))
        if not csv_files:
            csv_files = sorted(out_dir.glob("*-Scenes.csv"))

        if csv_files:
            text = csv_files[0].read_text(encoding="utf-8", errors="ignore")
        else:
            text = ((res.stdout or b"") + b"\n" + (res.stderr or b"")).decode("utf-8", errors="ignore")
            if not text.strip():
                return jsonify({"scenes": []}), 200

        scenes: List[Dict[str, Any]] = []
        for line in text.splitlines():
            if not line.strip() or line.lower().startswith("scene") or "timecode" in line.lower():
                continue
            parts = [p.strip().strip('"') for p in line.split(",")]
            times = []
            for p in parts:
                if re.match(r"^\d+:\d{2}:\d{2}(?:\.\d{1,6})?$", p):
                    times.append(p)
            if len(times) >= 2:
                st = _parse_hhmmss_to_seconds(times[0])
                en = _parse_hhmmss_to_seconds(times[1])
            else:
                floats: list[float] = []
                for p in parts:
                    if re.match(r"^-?\d+\.\d+$", p):
                        try:
                            floats.append(float(p))
                        except Exception:
                            pass
                st = floats[0] if len(floats) >= 1 else 0.0
                en = floats[1] if len(floats) >= 2 else 0.0
            if en > st:
                scenes.append({"index": len(scenes), "start": st, "end": en})

        return jsonify({"scenes": scenes}), 200
    finally:
        try:
            for child in temp_dir.rglob("*"):
                try:
                    child.unlink()
                except Exception:
                    pass
            temp_dir.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5094")))
