#!/usr/bin/env python3
"""End-to-end smoke test for the Multimodal Envid Metadata backend (port 5016).

- Picks the first usable video under GCS rawVideo/
- Submits /process-gcs-video-cloud
- Polls /jobs/<id>
- Verifies step IDs
- Fetches /video/<id>/metadata-json?category=combined

This script is safe to run repeatedly.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any

BASE = os.getenv("ENVID_METADATA_BASE_URL", "http://localhost:5016").rstrip("/")

EXPECTED_STEP_IDS = [
    "upload_to_cloud_storage",
    "technical_metadata",
    "transcode_normalize",
    "celebrity_detection",
    "celebrity_bio_image",
    "label_detection",
    "famous_location_detection",
    "moderation",
    "text_on_screen",
    "key_scene_detection",
    "high_point",
    "scene_by_scene_metadata",
    "opening_closing_credit_detection",
    "transcribe",
    "synopsis_generation",
    "save_as_json",
]


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


def main() -> int:
    health = _get_json(f"{BASE}/health", timeout=15)
    print("health:", {k: health.get(k) for k in ("status", "service", "gcs_bucket")})

    listing = _get_json(f"{BASE}/gcs/rawvideo/list?max_results=300", timeout=60)
    objs = listing.get("objects") or []

    def is_video(name: str) -> bool:
        n = name.lower()
        return n.endswith((".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mxf"))

    cand = None
    for o in objs:
        name = (o.get("name") or "").strip()
        if not name or name.endswith("/"):
            continue
        if name.endswith("/.keep") or name.endswith(".keep"):
            continue
        if is_video(name):
            cand = name
            break

    if not cand:
        print(f"No usable rawVideo object found. objects_count={len(objs)}")
        return 2

    print("picked:", cand)

    resp = _post_json(
        f"{BASE}/process-gcs-video-cloud",
        {
            "gcs_object": cand,
            "title": "Multimodal E2E Smoke",
            "description": "Auto-run to validate step IDs",
        },
        timeout=30,
    )

    job_id = resp.get("job_id") or resp.get("id") or (resp.get("job") or {}).get("id")
    if not job_id:
        print("Unexpected submit response keys:", sorted(resp.keys()))
        return 3

    print("job_id:", job_id)

    start = time.time()
    deadline = start + 25 * 60
    last_status = None
    while True:
        job = _get_json(f"{BASE}/jobs/{job_id}", timeout=20)
        status = (job.get("status") or "").strip()
        if status != last_status:
            print("status:", status, "progress:", job.get("progress"), "msg:", job.get("message"))
            last_status = status
        if status in {"completed", "failed"}:
            break
        if time.time() >= deadline:
            print("Timed out waiting for job")
            return 4
        time.sleep(5)

    steps = job.get("steps") or []
    step_ids = [s.get("id") for s in steps if isinstance(s, dict)]
    missing = [x for x in EXPECTED_STEP_IDS if x not in step_ids]
    extra = [x for x in step_ids if x not in EXPECTED_STEP_IDS]

    print("steps.count:", len(steps))
    print("steps.missing:", missing)
    print("steps.extra:", extra)

    meta = _get_json(f"{BASE}/video/{job_id}/metadata-json?category=combined", timeout=30)
    combined = meta.get("combined") or meta.get("metadata") or {}
    print("combined.keys.sample:", sorted(list(combined.keys()))[:25])

    print("elapsed_sec:", int(time.time() - start))
    return 0 if not missing else 5


if __name__ == "__main__":
    raise SystemExit(main())
