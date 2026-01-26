from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from functools import lru_cache

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)


def _run_ffmpeg(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError((res.stderr or res.stdout or "ffmpeg failed").strip())


def _run_ffprobe(video_path: Path) -> dict[str, Any]:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(video_path)],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError((res.stderr or res.stdout or "ffprobe failed").strip())
    return json.loads(res.stdout or "{}")


def _parse_blackdetect_output(text: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for m in re.finditer(r"black_start:(?P<st>[0-9.]+)\s+black_end:(?P<en>[0-9.]+)\s+black_duration:(?P<dur>[0-9.]+)", text):
        try:
            st = float(m.group("st"))
            en = float(m.group("en"))
            dur = float(m.group("dur"))
        except Exception:
            continue
        if dur <= 0:
            continue
        segments.append({"start": st, "end": en, "duration": dur})
    segments.sort(key=lambda x: (float(x.get("start") or 0.0), float(x.get("end") or 0.0)))
    return segments


@lru_cache(maxsize=1)
def _ffmpeg_supports_nvenc() -> bool:
    try:
        res = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
        out = (res.stdout or "") + (res.stderr or "")
        return "h264_nvenc" in out or "hevc_nvenc" in out
    except Exception:
        return False


def _use_gpu_transcode() -> bool:
    raw = (os.getenv("ENVID_FFMPEG_USE_GPU") or "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"ok": True}), 200


@app.route("/normalize", methods=["POST"])
def normalize() -> Any:
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    video_file = request.files["video"]
    if not video_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    bitrate_k = (request.form.get("video_bitrate_k") or "1500").strip()
    audio_bitrate_k = (request.form.get("audio_bitrate_k") or "128").strip()

    suffix = Path(video_file.filename).suffix or ".mp4"
    input_path: Path | None = None
    output_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            video_file.save(tmp.name)
            input_path = Path(tmp.name)

        output_path = Path(tempfile.mkstemp(suffix=suffix)[1])

        use_gpu = _use_gpu_transcode()
        if use_gpu and not _ffmpeg_supports_nvenc():
            raise RuntimeError("GPU transcode required but NVENC is not available")
        video_codec = "h264_nvenc" if use_gpu else "libx264"
        preset = (os.getenv("ENVID_FFMPEG_GPU_PRESET") or "fast").strip() if use_gpu else "veryfast"

        cmd = [
            "ffmpeg",
            "-y",
        ]
        if use_gpu:
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        cmd += [
            "-i",
            str(input_path),
            "-c:v",
            video_codec,
            "-b:v",
            f"{bitrate_k}k",
            "-maxrate",
            f"{bitrate_k}k",
            "-bufsize",
            f"{int(bitrate_k) * 2}k" if bitrate_k.isdigit() else "3000k",
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            f"{audio_bitrate_k}k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]

        _run_ffmpeg(cmd)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("ffmpeg produced empty output")

        return send_file(
            str(output_path),
            mimetype="video/mp4",
            as_attachment=True,
            download_name=f"normalized{suffix}",
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if input_path and input_path.exists():
            try:
                input_path.unlink()
            except Exception:
                pass
        if output_path and output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass


@app.route("/probe", methods=["POST"])
def probe() -> Any:
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    video_file = request.files["video"]
    if not video_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    suffix = Path(video_file.filename).suffix or ".mp4"
    input_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            video_file.save(tmp.name)
            input_path = Path(tmp.name)

        data = _run_ffprobe(input_path)
        return jsonify({"available": True, "raw": data}), 200
    except Exception as exc:
        return jsonify({"available": False, "error": str(exc)}), 500
    finally:
        if input_path and input_path.exists():
            try:
                input_path.unlink()
            except Exception:
                pass


@app.route("/blackdetect", methods=["POST"])
def blackdetect() -> Any:
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    video_file = request.files["video"]
    if not video_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    min_black_seconds = float(request.form.get("min_black_seconds") or 0.7)
    picture_black_threshold = float(request.form.get("picture_black_threshold") or 0.98)
    pixel_black_threshold = float(request.form.get("pixel_black_threshold") or 0.10)
    max_seconds = int(float(request.form.get("max_seconds") or 900))

    suffix = Path(video_file.filename).suffix or ".mp4"
    input_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            video_file.save(tmp.name)
            input_path = Path(tmp.name)

        vf = f"blackdetect=d={min_black_seconds}:pic_th={picture_black_threshold}:pix_th={pixel_black_threshold}"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(input_path),
            "-vf",
            vf,
            "-an",
            "-f",
            "null",
            "-",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=max_seconds)
        text = (res.stderr or b"").decode("utf-8", errors="ignore")
        if res.returncode not in (0, 1):
            raise RuntimeError(f"ffmpeg blackdetect failed (code {res.returncode})")

        segments = _parse_blackdetect_output(text)
        return jsonify({"segments": segments}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if input_path and input_path.exists():
            try:
                input_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5091")))
