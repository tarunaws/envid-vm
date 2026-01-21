#!/usr/bin/env python3
"""Trigger an envid-metadata reprocess and optionally wait for completion.

Usage:
    python3 code/scripts/reprocess_envid_metadata.py <video_id> [--base-url http://localhost:5016] [--frame-interval 1] [--max-frames 1000] [--wait]

Notes:
- Uses the legacy /video/<id>/reprocess endpoint.
- Polls /jobs/<id> when --wait is provided.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request


def _http_json(url: str, *, method: str = "GET", body: dict | None = None) -> dict:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _pick_active_step(job: dict) -> dict | None:
    steps = job.get("steps")
    if not isinstance(steps, list) or not steps:
        return None

    job_status = (job.get("status") or "").lower()
    if job_status in {"completed", "failed"}:
        finished = [
            s
            for s in steps
            if isinstance(s, dict) and (s.get("status") or "").lower() in {"completed", "failed", "skipped"}
        ]
        return finished[-1] if finished else None

    # Prefer the explicitly running step.
    for s in steps:
        if isinstance(s, dict) and (s.get("status") or "").lower() == "running":
            return s

    # Otherwise pick the first not-yet-completed step (useful when status flips quickly).
    for s in steps:
        if not isinstance(s, dict):
            continue
        st = (s.get("status") or "").lower()
        if st in {"not_started", "running"}:
            return s

    # Otherwise return the last finished step.
    finished = [
        s
        for s in steps
        if isinstance(s, dict) and (s.get("status") or "").lower() in {"completed", "failed", "skipped"}
    ]
    return finished[-1] if finished else None


def _format_step(s: dict | None) -> str:
    if not isinstance(s, dict):
        return "(unknown)"
    label = (s.get("label") or s.get("id") or "(unknown)").strip()
    status = (s.get("status") or "").strip()
    percent = s.get("percent")
    msg = (s.get("message") or "").strip()
    bits = [label]
    if status:
        bits.append(status)
    if percent is not None:
        bits.append(f"{percent}%")
    if msg:
        bits.append(msg)
    return " | ".join(bits)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("video_id")
    p.add_argument("--base-url", default="http://localhost:5016")
    p.add_argument("--frame-interval", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=1000)
    p.add_argument("--wait", action="store_true")
    p.add_argument("--poll-only", action="store_true", help="Do not POST; only poll /jobs/<id>.")
    p.add_argument("--poll-seconds", type=float, default=2.0)
    p.add_argument(
        "--show-steps",
        action="store_true",
        help="Print all step statuses when they change (not just the active step).",
    )
    args = p.parse_args()

    base = args.base_url.rstrip("/")
    qs = urllib.parse.urlencode(
        {
            "frame_interval_seconds": str(max(1, min(30, int(args.frame_interval)))),
            "max_frames_to_analyze": str(max(1, min(10000, int(args.max_frames)))),
        }
    )
    url = f"{base}/video/{args.video_id}/reprocess?{qs}"

    job_id = args.video_id
    if not args.poll_only:
        print(f"POST {url}")
        resp = _http_json(url, method="POST", body={})
        job_id = resp.get("job_id") or args.video_id
        print(f"queued job_id={job_id}")

    if not args.wait:
        return 0

    job_url = f"{base}/jobs/{job_id}"
    last_line = None
    last_steps_snapshot = None
    while True:
        job = _http_json(job_url)
        status = (job.get("status") or "").lower()
        progress = job.get("progress")
        message = job.get("message")

        active_step = _pick_active_step(job)
        active_line = _format_step(active_step)

        line = f"status={status} progress={progress} active_step={active_line} message={message}"
        if line != last_line:
            print(line)
            last_line = line

        if args.show_steps:
            steps = job.get("steps")
            snapshot = None
            if isinstance(steps, list):
                snapshot = [
                    {
                        "id": (s.get("id") if isinstance(s, dict) else None),
                        "status": (s.get("status") if isinstance(s, dict) else None),
                        "percent": (s.get("percent") if isinstance(s, dict) else None),
                        "message": (s.get("message") if isinstance(s, dict) else None),
                    }
                    for s in steps
                ]
            if snapshot != last_steps_snapshot:
                last_steps_snapshot = snapshot
                if isinstance(steps, list):
                    for s in steps:
                        if isinstance(s, dict):
                            print("  -", _format_step(s))
        if status in {"completed", "failed"}:
            return 0 if status == "completed" else 1
        time.sleep(max(0.5, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
