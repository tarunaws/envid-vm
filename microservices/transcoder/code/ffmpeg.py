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


def _ensure_allowed_path(raw: str) -> Path:
    p = Path(raw).resolve()
    allowed = ("/tmp/", "/mnt/gcs/")
    if not str(p).startswith(allowed):
        raise RuntimeError("path must be under /tmp or /mnt/gcs")
    return p


def _run_ffprobe(video_path: Path) -> dict[str, Any]:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(video_path)],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError((res.stderr or res.stdout or "ffprobe failed").strip())
    return json.loads(res.stdout or "{}")


def _env_truthy(value: str | None, default: bool = False) -> bool:
    raw = (value or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _primary_video_codec(probe: dict[str, Any]) -> str:
    streams = probe.get("streams") if isinstance(probe, dict) else None
    if not isinstance(streams, list):
        return ""
    for st in streams:
        if not isinstance(st, dict):
            continue
        if str(st.get("codec_type") or "").lower() == "video":
            return str(st.get("codec_name") or "").lower().strip()
    return ""


def _gpu_codec_supported(codec: str) -> bool:
    if not codec:
        return True
    supported = {
        "h264",
        "hevc",
        "h265",
        "av1",
        "vp9",
        "mpeg2video",
    }
    return codec in supported


def _extract_audio(
    *,
    video_path: Path,
    out_path: Path,
    sample_rate: int,
    channels: int,
    fmt: str,
) -> None:
    if fmt == "flac":
        codec = "flac"
        suffix = ".flac"
    else:
        codec = "pcm_s16le"
        suffix = ".wav"
    if out_path.suffix != suffix:
        out_path = out_path.with_suffix(suffix)
    filters: list[str] = [
        "highpass=f=80",
        "lowpass=f=8000",
        "anlmdn",
        "afftdn",
    ]
    atempo_raw = (os.getenv("ENVID_WHISPER_AUDIO_ATEMPO") or "0.9").strip()
    if atempo_raw:
        try:
            atempo = float(atempo_raw)
        except ValueError:
            atempo = 1.0
        if atempo > 0 and atempo != 1.0:
            filters.append(f"atempo={atempo}")
    filters.append("loudnorm=I=-16:LRA=11:TP=-1.5")
    filter_chain = ",".join(filters)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-af",
        filter_chain,
        "-ac",
        str(max(1, int(channels))),
        "-ar",
        "16000",
        "-acodec",
        codec,
        str(out_path),
    ]
    _run_ffmpeg(cmd)


def _extract_frame(
    *,
    video_path: Path,
    out_path: Path,
    timestamp: float,
    scale: int,
    quality: int,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(0.0, float(timestamp)):.3f}",
        "-i",
        str(video_path),
        "-vframes",
        "1",
        "-vf",
        f"scale={max(32, int(scale))}:-1",
        "-q:v",
        str(max(2, min(int(quality), 31))),
        str(out_path),
    ]
    _run_ffmpeg(cmd)


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
    use_gpu = _env_truthy(os.getenv("ENVID_FFMPEG_USE_GPU"), default=True)
    if not use_gpu:
        return False
    if not _ffmpeg_supports_nvenc():
        return False
    if not _gpu_device_available():
        return False
    return True


def _gpu_device_available() -> bool:
    if Path("/proc/driver/nvidia/gpus").exists():
        try:
            if any(Path("/proc/driver/nvidia/gpus").iterdir()):
                return True
        except Exception:
            pass
    if Path("/dev/nvidia0").exists():
        try:
            res = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0 and (res.stdout or "").strip():
                return True
        except Exception:
            pass
    return False


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

        probe = _run_ffprobe(input_path)
        input_codec = _primary_video_codec(probe)

        use_gpu = _use_gpu_transcode()
        gpu_codec_ok = _gpu_codec_supported(input_codec)
        app.logger.info(
            "normalize: codec=%s gpu_available=%s nvenc=%s gpu_codec_ok=%s",
            input_codec or "unknown",
            _gpu_device_available(),
            _ffmpeg_supports_nvenc(),
            gpu_codec_ok,
        )
        if use_gpu and not gpu_codec_ok:
            app.logger.warning(
                "GPU decode may be unsupported for codec %s; attempting GPU encode anyway",
                input_codec or "unknown",
            )
        app.logger.info("normalize: selected %s", "GPU (NVENC)" if use_gpu else "CPU")

        def _build_cmd(gpu: bool) -> list[str]:
            video_codec = "h264_nvenc" if gpu else "libx264"
            preset = (os.getenv("ENVID_FFMPEG_GPU_PRESET") or "fast").strip() if gpu else "veryfast"
            cmd = ["ffmpeg", "-y"]
            if gpu:
                cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
            cmd += ["-i", str(input_path)]
            cmd += [
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
            ]
            if not gpu:
                cmd += ["-pix_fmt", "yuv420p"]
            cmd += [
                "-c:a",
                "aac",
                "-b:a",
                f"{audio_bitrate_k}k",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            return cmd

        try:
            if use_gpu:
                app.logger.info("normalize: using GPU transcode (NVENC)")
                last_exc: Exception | None = None
                for attempt in range(1, 4):
                    try:
                        app.logger.info("normalize: attempting GPU transcode (attempt %s)", attempt)
                        _run_ffmpeg(_build_cmd(True))
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        app.logger.warning("GPU transcode attempt %s failed: %s", attempt, exc)
                if last_exc is not None:
                    app.logger.warning("GPU transcode failed after retries; falling back to CPU")
                    _run_ffmpeg(_build_cmd(False))
            else:
                app.logger.info("normalize: using CPU transcode")
                _run_ffmpeg(_build_cmd(False))
        except Exception as exc:
            raise exc
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


@app.route("/extract-audio", methods=["POST"])
def extract_audio() -> Any:
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    video_file = request.files["video"]
    if not video_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    strict_audio = (os.getenv("ENVID_WHISPER_AUDIO_STRICT") or "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if strict_audio:
        fmt = "wav"
        sample_rate = 16000
        channels = 1
    else:
        fmt = (request.form.get("format") or "flac").strip().lower()
        sample_rate = int(float(request.form.get("sample_rate") or 16000))
        channels = int(float(request.form.get("channels") or 1))

    suffix = Path(video_file.filename).suffix or ".mp4"
    input_path: Path | None = None
    output_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            video_file.save(tmp.name)
            input_path = Path(tmp.name)

        out_suffix = ".flac" if fmt == "flac" else ".wav"
        output_path = Path(tempfile.mkstemp(suffix=out_suffix)[1])
        _extract_audio(video_path=input_path, out_path=output_path, sample_rate=sample_rate, channels=channels, fmt=fmt)

        return send_file(
            str(output_path),
            mimetype="audio/flac" if fmt == "flac" else "audio/wav",
            as_attachment=True,
            download_name=f"audio{out_suffix}",
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


@app.route("/segment", methods=["POST"])
def segment() -> Any:
    payload = request.get_json(silent=True) or {}
    video_path_raw = str(payload.get("video_path") or "").strip()
    start_seconds = float(payload.get("start_seconds") or 0.0)
    duration_seconds = float(payload.get("duration_seconds") or 0.0)
    output_path_raw = str(payload.get("output_path") or "").strip()

    if not video_path_raw:
        return jsonify({"error": "video_path is required"}), 400
    if duration_seconds <= 0:
        return jsonify({"error": "duration_seconds must be > 0"}), 400

    try:
        input_path = _ensure_allowed_path(video_path_raw)
        if not input_path.exists():
            return jsonify({"error": "input video not found"}), 404

        if output_path_raw:
            output_path = _ensure_allowed_path(output_path_raw)
        else:
            output_path = Path(tempfile.mkstemp(suffix=input_path.suffix or ".mp4")[1])

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd_copy = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{max(0.0, start_seconds):.3f}",
            "-t",
            f"{max(0.0, duration_seconds):.3f}",
            "-i",
            str(input_path),
            "-c",
            "copy",
            "-reset_timestamps",
            "1",
            str(output_path),
        ]
        try:
            _run_ffmpeg(cmd_copy)
        except Exception:
            use_gpu = _use_gpu_transcode()
            cmd_gpu = [
                "ffmpeg",
                "-y",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-ss",
                f"{max(0.0, start_seconds):.3f}",
                "-t",
                f"{max(0.0, duration_seconds):.3f}",
                "-i",
                str(input_path),
                "-c:v",
                "h264_nvenc",
                "-preset",
                (os.getenv("ENVID_FFMPEG_GPU_PRESET") or "fast").strip(),
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            cmd_cpu = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{max(0.0, start_seconds):.3f}",
                "-t",
                f"{max(0.0, duration_seconds):.3f}",
                "-i",
                str(input_path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            if use_gpu:
                try:
                    _run_ffmpeg(cmd_gpu)
                except Exception:
                    _run_ffmpeg(cmd_cpu)
            else:
                _run_ffmpeg(cmd_cpu)

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("ffmpeg produced empty segment")
        return jsonify({"ok": True, "output_path": str(output_path)}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/extract-frame", methods=["POST"])
def extract_frame() -> Any:
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    video_file = request.files["video"]
    if not video_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    timestamp = float(request.form.get("timestamp") or 0.0)
    scale = int(float(request.form.get("scale") or 224))
    quality = int(float(request.form.get("quality") or 3))

    suffix = Path(video_file.filename).suffix or ".mp4"
    input_path: Path | None = None
    output_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            video_file.save(tmp.name)
            input_path = Path(tmp.name)

        output_path = Path(tempfile.mkstemp(suffix=".jpg")[1])
        _extract_frame(
            video_path=input_path,
            out_path=output_path,
            timestamp=timestamp,
            scale=scale,
            quality=quality,
        )

        return send_file(
            str(output_path),
            mimetype="image/jpeg",
            as_attachment=True,
            download_name="frame.jpg",
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
