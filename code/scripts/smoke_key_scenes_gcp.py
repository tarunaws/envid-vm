"""Smoke-test: run key scene + high point detection on a GCS raw video.

Usage:
  code/.venv/bin/python code/scripts/smoke_key_scenes_gcp.py

Environment:
  ENVID_MULTIMODAL_URL   (default: http://localhost:5016)
  ENVID_GCS_OBJECT       (optional: override selected object)
  ENVID_KEY_SCENE_TOP_K  (optional, int)

This script:
- Fetches /gcs/rawvideo/list to pick an object (unless ENVID_GCS_OBJECT is set)
- POSTs /process-gcs-video-cloud with key scene detection enabled
- Polls /jobs/<id> until completed
- Fetches /video/<id>/metadata-json and prints key_scenes/high_points
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests


_VIDEO_EXTS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".mpg",
    ".mpeg",
    ".m4v",
    ".ts",
}


def _pick_gcs_object(base_url: str) -> str:
    override = (os.getenv("ENVID_GCS_OBJECT") or "").strip()
    if override:
        return override

    r = requests.get(f"{base_url}/gcs/rawvideo/list", timeout=30)
    r.raise_for_status()
    data = r.json()
    objects = data.get("objects") or []
    if not objects:
        raise SystemExit("No objects returned from /gcs/rawvideo/list")

    def _to_name(item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("name") or item.get("key") or "").strip()
        return str(item or "").strip()

    candidates = [_to_name(o) for o in objects]
    candidates = [c for c in candidates if c]

    def _looks_like_video(path: str) -> bool:
        if not path or path.endswith("/"):
            return False
        base = path.rsplit("/", 1)[-1]
        if base in {".keep", "_SUCCESS"}:
            return False
        lower = base.lower()
        return any(lower.endswith(ext) for ext in _VIDEO_EXTS)

    for c in candidates:
        if _looks_like_video(c):
            return c

    for c in candidates:
        if not c.endswith("/") and c.rsplit("/", 1)[-1] not in {".keep", "_SUCCESS"}:
            return c

    raise SystemExit(
        "Could not find a usable video object from /gcs/rawvideo/list. "
        "Set ENVID_GCS_OBJECT to a specific video key (e.g. rawVideo/.../file.mp4)."
    )


def main() -> int:
    base = (os.getenv("ENVID_MULTIMODAL_URL") or "http://localhost:5016").rstrip("/")
    gcs_object = _pick_gcs_object(base)
    key_scene_model = (os.getenv("ENVID_KEY_SCENE_MODEL") or "gcp_video_intelligence").strip() or "gcp_video_intelligence"

    print("BASE:", base)
    print("Using gcs_object:", gcs_object)
    print("Key scene model:", key_scene_model)

    payload: dict[str, Any] = {
        "gcs_object": gcs_object,
        "video_title": f"KeyScene smoke: {gcs_object.split('/')[-1]}",
        "task_selection": {
            "enable_key_scene_detection": True,
            "key_scene_detection_model": key_scene_model,
            "enable_text": True,
            "text_model": "auto",
            "enable_moderation": True,
            "moderation_model": "gcp_video_intelligence",
            "enable_transcribe": False,
            "enable_scene_by_scene_metadata": False,
            "enable_opening_closing_credit_detection": False,
            "enable_celebrity_detection": False,
            "enable_celebrity_bio_image": False,
            "enable_famous_location_detection": False,
            "enable_label_detection": False,
        },
    }

    r = requests.post(f"{base}/process-gcs-video-cloud", json=payload, timeout=30)
    if not r.ok:
        raise SystemExit(f"POST /process-gcs-video-cloud failed: HTTP {r.status_code}: {r.text[:1500]}")
    job_id = (r.json() or {}).get("job_id")
    if not job_id:
        raise SystemExit(f"No job_id in response: {r.text[:500]}")

    print("Job:", job_id)

    # Poll job
    last_line = ""
    for _ in range(240):  # ~20 min
        j = requests.get(f"{base}/jobs/{job_id}", timeout=30).json()
        status = str(j.get("status") or "").lower()
        progress = j.get("progress")
        message = j.get("message")
        line = f"{status} {progress}% {message}"
        if line != last_line:
            print("-", line)
            last_line = line

        if status in {"completed", "failed"}:
            break
        time.sleep(5)

    if status != "completed":
        raise SystemExit(f"Job did not complete successfully: {status}")

    meta = requests.get(f"{base}/video/{job_id}/metadata-json", timeout=30).json()
    cats = meta.get("categories") or {}
    key_scenes = cats.get("key_scenes") or []
    high_points = cats.get("high_points") or []

    print("\nResults")
    print("key_scenes:", len(key_scenes))
    print("high_points:", len(high_points))

    if key_scenes:
        print("\nfirst key_scene:")
        print(json.dumps(key_scenes[0], indent=2)[:1200])
    if high_points:
        print("\ntop high_point:")
        print(json.dumps(high_points[0], indent=2)[:1200])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
