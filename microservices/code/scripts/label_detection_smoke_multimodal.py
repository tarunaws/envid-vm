#!/usr/bin/env python3
"""Smoke test for Multimodal (5016) label detection engines.

- Picks a usable rawVideo/* object (or accepts --gcs-object)
- Submits /process-gcs-video-cloud with only labels enabled
- Polls /jobs/<id>
- Fetches /video/<id> and prints label counts

Use:
  cd code
  ./.venv/bin/python scripts/label_detection_smoke_multimodal.py --model detectron2
  ./.venv/bin/python scripts/label_detection_smoke_multimodal.py --model mmdetection
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from typing import Any


def _get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post_json(url: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _pick_first_video_object(base_url: str) -> str:
    listing = _get_json(f"{base_url}/gcs/rawvideo/list?max_results=300", timeout=60)
    objs = listing.get("objects") or []

    def is_video(name: str) -> bool:
        n = name.lower()
        return n.endswith((".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mxf"))

    for o in objs:
        name = (o.get("name") or "").strip()
        if not name or name.endswith("/"):
            continue
        if name.endswith("/.keep") or name.endswith(".keep"):
            continue
        if is_video(name):
            return name

    raise RuntimeError(f"No usable video object found under rawVideo/. objects_count={len(objs)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Label detection smoke test for Multimodal backend")
    ap.add_argument("--base-url", default=os.getenv("ENVID_METADATA_BASE_URL", "http://localhost:5016").rstrip("/"))
    ap.add_argument("--gcs-object", default="", help="rawVideo/... object name (optional)")
    ap.add_argument(
        "--model",
        required=True,
        choices=["google_video_intelligence", "yolo", "detectron2", "mmdetection"],
        help="Label detection engine to request",
    )
    ap.add_argument("--timeout-seconds", type=int, default=420)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")

    health = _get_json(f"{base}/health", timeout=15)
    if (health.get("status") or "").lower() != "ok":
        raise RuntimeError(f"Backend not healthy: {health}")

    gcs_object = args.gcs_object.strip() or _pick_first_video_object(base)
    print("picked:", gcs_object)

    model_map = {
        "google_video_intelligence": "google_video_intelligence",
        "yolo": "yolo",
        "detectron2": "detectron2",
        "mmdetection": "mmdetection",
    }

    payload = {
        "gcs_object": gcs_object,
        "title": f"label-smoke-{args.model}",
        "task_selection": {
            "enable_labels": True,
            "label_detection_model": model_map[args.model],
            "enable_famous_locations": False,
            "enable_moderation": False,
            "enable_celebrities": False,
            "enable_rekognition_shots": False,
            "enable_rekognition_technical_cues": False,
            "enable_key_scene_detection": False,
            "enable_high_point": False,
            "enable_scene_by_scene_metadata": False,
            "enable_opening_closing_credit_detection": False,
            "enable_transcribe": False,
            "enable_text": False,
        },
    }

    resp = _post_json(f"{base}/process-gcs-video-cloud", payload, timeout=60)
    job_id = resp.get("job_id") or resp.get("id") or (resp.get("job") or {}).get("id")
    if not job_id:
        raise RuntimeError(f"No job_id in response: {resp}")

    print("job_id:", job_id)

    start = time.time()
    last_line = None
    while True:
        job = _get_json(f"{base}/jobs/{job_id}", timeout=30)
        line = f"status={job.get('status')} progress={job.get('progress')} message={job.get('message')}"
        if line != last_line:
            print(line)
            last_line = line

        status = (job.get("status") or "").lower()
        if status in {"completed", "failed", "error"}:
            break

        if time.time() - start > args.timeout_seconds:
            raise TimeoutError(f"Timed out after {args.timeout_seconds}s")

        time.sleep(3)

    if (job.get("status") or "").lower() != "completed":
        raise RuntimeError(f"Job failed: {job.get('error') or job.get('message')}")

    video = _get_json(f"{base}/video/{job_id}", timeout=60)

    vi = video.get("video_intelligence") or {}
    labels = vi.get("labels") or []
    print("labels_count:", len(labels) if isinstance(labels, list) else 0)

    if isinstance(labels, list) and labels:
        print("labels_first5:", [l.get("label") for l in labels[:5] if isinstance(l, dict)])
    else:
        raise RuntimeError("No labels returned (video_intelligence.labels is empty)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
