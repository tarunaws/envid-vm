from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, List, Tuple

import cv2  # type: ignore
from flask import Flask, jsonify, request

app = Flask(__name__)


def _ensure_tmp_path(raw: str) -> Path:
    p = Path(raw).resolve()
    allowed = ("/tmp/", "/mnt/gcs/")
    if not str(p).startswith(allowed):
        raise RuntimeError("video_path must be under /tmp or /mnt/gcs")
    if not p.exists() or not p.is_file():
        raise RuntimeError("video_path not found")
    return p


def _backend_url() -> str:
    return (os.getenv("ENVID_BACKEND_URL") or "http://backend:5016").rstrip("/")


def _env_truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_video_path(job_id: str, kind: str) -> str:
    if not job_id:
        raise RuntimeError("Missing job_id")
    url = f"{_backend_url()}/jobs/{job_id}/outputs/{kind}"
    resp = requests.get(url, timeout=10)
    if resp.status_code >= 400:
        raise RuntimeError(f"Failed to resolve video_path ({resp.status_code}): {resp.text}")
    payload = resp.json()
    path = str(payload.get("path") or "").strip()
    if not path:
        raise RuntimeError("Resolved video_path is empty")
    return path


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


def _sample_video_frames_ffmpeg(*, video_path: Path, interval_seconds: float, max_frames: int, out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if float(interval_seconds) <= 0:
        out_pattern = out_dir / "frame_%010d.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vsync",
            "vfr",
            "-frame_pts",
            "1",
            "-frames:v",
            str(int(max_frames)),
            str(out_pattern),
        ]
        subprocess.run(cmd, check=True)
        frames = sorted(out_dir.glob("frame_*.jpg"))
        if not frames:
            raise RuntimeError("ffmpeg produced no frames")
        time_base = _ffprobe_time_base(video_path=video_path)
        manifest: list[dict[str, Any]] = []
        for frame_path in frames:
            name = frame_path.stem
            pts_raw = name.replace("frame_", "")
            try:
                pts_val = float(pts_raw)
            except Exception:
                pts_val = 0.0
            manifest.append({"time": float(pts_val) * time_base, "path": str(frame_path)})
        return manifest

    out_pattern = out_dir / "frame_%05d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{max(0.01, float(interval_seconds))}",
        "-frames:v",
        str(int(max_frames)),
        str(out_pattern),
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("ffmpeg produced no frames")
    manifest: list[dict[str, Any]] = []
    for idx, frame_path in enumerate(frames):
        manifest.append({"time": float(idx) * float(interval_seconds), "path": str(frame_path)})
    return manifest


def _ffprobe_time_base(*, video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=time_base",
        "-of",
        "default=nw=1:nk=1",
        str(video_path),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    raw = (proc.stdout or "").strip()
    if not raw or "/" not in raw:
        return 1.0 / 1000.0
    num, den = raw.split("/", 1)
    try:
        num_f = float(num)
        den_f = float(den)
        if den_f <= 0:
            return 1.0 / 1000.0
        return num_f / den_f
    except Exception:
        return 1.0 / 1000.0


def _sample_video_frames_ffmpeg_scene(
    *, video_path: Path, max_frames: int, out_dir: Path, scene_threshold: float
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = out_dir / "frame_%010d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select='gt(scene,{scene_threshold})'",
        "-vsync",
        "vfr",
        "-frame_pts",
        "1",
        "-frames:v",
        str(int(max_frames)),
        str(out_pattern),
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("ffmpeg scene detect produced no frames")
    time_base = _ffprobe_time_base(video_path=video_path)
    manifest: list[dict[str, Any]] = []
    for frame_path in frames:
        name = frame_path.stem
        pts_raw = name.replace("frame_", "")
        try:
            pts_val = float(pts_raw)
        except Exception:
            pts_val = 0.0
        manifest.append({"time": float(pts_val) * time_base, "path": str(frame_path)})
    return manifest


def _extract_frames_at_timestamps_ffmpeg(
    *, video_path: Path, timestamps: list[float], out_dir: Path
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for idx, t in enumerate(timestamps):
        try:
            t_val = float(t)
        except Exception:
            continue
        if t_val < 0:
            continue
        out_path = out_dir / f"frame_{idx:05d}.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{t_val}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        if out_path.exists():
            manifest.append({"time": float(t_val), "path": str(out_path)})
    return manifest


@app.get("/health")
def health() -> Any:
    return jsonify({"ok": True}), 200


@app.post("/extract")
def extract() -> Any:
    payload = request.get_json(silent=True) or {}
    video_path_raw = str(payload.get("video_path") or "").strip()
    output_kind = str(payload.get("output_kind") or "processed_local").strip() or "processed_local"

    interval = 0.0
    max_frames = 200000
    job_id = str(payload.get("job_id") or "").strip() or uuid.uuid4().hex

    try:
        if not video_path_raw:
            video_path_raw = _resolve_video_path(job_id, output_kind)
        video_path = _ensure_tmp_path(video_path_raw)
        out_dir = Path("/mnt/gcs/envid/work") / job_id / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_timestamps = payload.get("timestamps")
        timestamps: list[float] = []
        if isinstance(raw_timestamps, list):
            for t in raw_timestamps:
                try:
                    timestamps.append(float(t))
                except Exception:
                    continue

        use_scene = False
        scene_threshold = 0.3

        manifest: list[dict[str, Any]] = []
        if timestamps:
            manifest = _extract_frames_at_timestamps_ffmpeg(
                video_path=video_path,
                timestamps=timestamps[: max_frames if max_frames > 0 else None],
                out_dir=out_dir,
            )

        if (not manifest) and use_scene:
            try:
                manifest = _sample_video_frames_ffmpeg_scene(
                    video_path=video_path,
                    max_frames=max_frames,
                    out_dir=out_dir,
                    scene_threshold=scene_threshold,
                )
            except Exception:
                manifest = []

        if not manifest:
            frames = _sample_video_frames(video_path=video_path, interval_seconds=interval, max_frames=max_frames)
            if frames:
                for idx, (t, bgr) in enumerate(frames):
                    frame_path = out_dir / f"frame_{idx:05d}.jpg"
                    ok = cv2.imwrite(str(frame_path), bgr)
                    if not ok:
                        continue
                    manifest.append({"time": float(t), "path": str(frame_path)})
            else:
                manifest = _sample_video_frames_ffmpeg(
                    video_path=video_path,
                    interval_seconds=interval,
                    max_frames=max_frames,
                    out_dir=out_dir,
                )

        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

        return jsonify({"frames": manifest, "manifest_path": str(manifest_path), "frames_dir": str(out_dir)}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500



import base64
import requests

def _moderation_service_url() -> str:
    return "http://moderation:5081"

@app.post("/moderate/path")
def moderate_path() -> Any:
    payload = request.get_json(silent=True) or {}
    video_path_raw = str(payload.get("video_path") or "").strip()
    output_kind = str(payload.get("output_kind") or "processed_local").strip() or "processed_local"

    interval = float(payload.get("interval_seconds") or 2)
    max_frames = int(payload.get("max_frames") or 120)
    job_id = str(payload.get("job_id") or "").strip() or uuid.uuid4().hex

    try:
        if not video_path_raw:
            video_path_raw = _resolve_video_path(job_id, output_kind)
        video_path = _ensure_tmp_path(video_path_raw)
        out_dir = Path("/mnt/gcs/envid/work") / job_id / "moderation" / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_timestamps = payload.get("timestamps")
        timestamps: list[float] = []
        if isinstance(raw_timestamps, list):
            for t in raw_timestamps:
                try:
                    timestamps.append(float(t))
                except Exception:
                    continue

        force_interval = bool(payload.get("force_interval"))
        use_scene = _env_truthy(os.getenv("ENVID_METADATA_MODERATION_SCENE_SAMPLING"), default=True)
        scene_threshold = float(os.getenv("ENVID_METADATA_MODERATION_SCENE_THRESHOLD") or 0.3)

        manifest: list[dict[str, Any]] = []
        if timestamps:
            manifest = _extract_frames_at_timestamps_ffmpeg(
                video_path=video_path,
                timestamps=timestamps[: max_frames if max_frames > 0 else None],
                out_dir=out_dir,
            )

        if (not manifest) and use_scene and not force_interval:
            try:
                manifest = _sample_video_frames_ffmpeg_scene(
                    video_path=video_path,
                    max_frames=max_frames,
                    out_dir=out_dir,
                    scene_threshold=scene_threshold,
                )
            except Exception:
                manifest = []

        if not manifest:
            frames = _sample_video_frames(video_path=video_path, interval_seconds=interval, max_frames=max_frames)
            if frames:
                for idx, (t, bgr) in enumerate(frames):
                    frame_path = out_dir / f"frame_{idx:05d}.jpg"
                    ok = cv2.imwrite(str(frame_path), bgr)
                    if not ok:
                        continue
                    manifest.append({"time": float(t), "path": str(frame_path)})
            else:
                manifest = _sample_video_frames_ffmpeg(
                    video_path=video_path,
                    interval_seconds=interval,
                    max_frames=max_frames,
                    out_dir=out_dir,
                )

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

        out_dir_meta = Path("/mnt/gcs/envid/work") / (job_id or "job") / "moderation"
        out_dir_meta.mkdir(parents=True, exist_ok=True)
        out_path = out_dir_meta / f"moderation_{job_id or 'job'}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        out["output_path"] = str(out_path)
        out["frames_dir"] = str(out_dir)
        return jsonify(out), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5093")))
