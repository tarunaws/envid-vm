#!/usr/bin/env python3
"""Smoke test: run Multimodal pipeline with moderation only.

- Picks first usable GCS rawVideo/* object
- Submits /process-gcs-video-cloud with task_selection enabling only Moderation
- Polls /jobs/<id> until completion
- Fetches /video/<id>/metadata-json?category=combined and prints moderation summary

Env:
- ENVID_METADATA_BASE_URL (default http://localhost:5016)
- ENVID_MODERATION_MODEL (default nudenet)
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any

BASE = os.getenv("ENVID_METADATA_BASE_URL", "http://localhost:5016").rstrip("/")
MODERATION_MODEL = (os.getenv("ENVID_MODERATION_MODEL") or "nudenet").strip().lower() or "nudenet"
LOCAL_SERVICE_URL = (
    (os.getenv("ENVID_METADATA_LOCAL_MODERATION_NSFWJS_URL") or "").strip()
    if MODERATION_MODEL == "nsfwjs"
    else (os.getenv("ENVID_METADATA_LOCAL_MODERATION_URL") or "").strip()
)


def _get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post_json(url: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _is_video(name: str) -> bool:
    n = name.lower()
    return n.endswith((".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mxf"))


def main() -> int:
    health = _get_json(f"{BASE}/health", timeout=15)
    print("health:", {k: health.get(k) for k in ("status", "service", "gcs_bucket")})

    listing = _get_json(f"{BASE}/gcs/rawvideo/list?max_results=300", timeout=60)
    objs = listing.get("objects") or []

    cand = None
    for o in objs:
        name = (o.get("name") or "").strip()
        if not name or name.endswith("/"):
            continue
        if name.endswith("/.keep") or name.endswith(".keep"):
            continue
        if _is_video(name):
            cand = name
            break

    if not cand:
        print(f"No usable rawVideo object found. objects_count={len(objs)}")
        return 2

    print("picked:", cand)

    task_selection = {
        "enable_moderation": True,
        "moderation_model": MODERATION_MODEL,
        "enable_text": False,
        "enable_label_detection": False,
        "enable_transcribe": False,
        "enable_famous_location_detection": False,
        "enable_scene_by_scene_metadata": False,
        "enable_key_scene_detection": False,
        "enable_high_point": False,
        "enable_opening_closing_credit_detection": False,
        "enable_celebrity_detection": False,
        "enable_celebrity_bio_image": False,
    }

    if LOCAL_SERVICE_URL:
        task_selection["local_moderation_url"] = LOCAL_SERVICE_URL

    resp = _post_json(
        f"{BASE}/process-gcs-video-cloud",
        {
            "gcs_object": cand,
            "title": f"Moderation Smoke ({MODERATION_MODEL})",
            "description": "Smoke test moderation only",
            "task_selection": task_selection,
        },
        timeout=60,
    )

    job_id = resp.get("job_id") or resp.get("id") or (resp.get("job") or {}).get("id")
    if not job_id:
        print("Unexpected submit response keys:", sorted(resp.keys()))
        return 3

    print("job_id:", job_id)

    start = time.time()
    deadline = start + 30 * 60
    last_line = None
    final_job: dict[str, Any] | None = None

    while True:
        job = _get_json(f"{BASE}/jobs/{job_id}", timeout=30)
        final_job = job
        status = (job.get("status") or "").strip()
        step = next((s for s in (job.get("steps") or []) if (s or {}).get("id") == "moderation"), {})
        line = (
            status,
            job.get("progress"),
            job.get("message"),
            step.get("status"),
            step.get("message"),
        )
        if line != last_line:
            print("poll:", line)
            last_line = line

        if status in {"completed", "failed"}:
            break
        if time.time() >= deadline:
            print("Timed out waiting for job")
            return 4
        time.sleep(5)

    if not final_job:
        print("No job result")
        return 5

    if (final_job.get("status") or "").strip() != "completed":
        print("job_failed:", final_job.get("error") or final_job.get("message"))
        return 6

    meta = _get_json(f"{BASE}/video/{job_id}/metadata-json?category=combined", timeout=60)
    combined = meta.get("combined") or meta.get("metadata") or {}
    vi = combined.get("video_intelligence") or {}
    mod = vi.get("moderation") or {}
    frames = mod.get("explicit_frames") or []

    print("final_status: completed")
    print("moderation_model:", MODERATION_MODEL)
    print("explicit_frames:", len(frames))
    print("first5_frames:", frames[:5])

    print("elapsed_sec:", int(time.time() - start))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
