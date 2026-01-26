#!/usr/bin/env python3

import argparse
import json
import sys
import time
import urllib.request


def _json_request(url: str, payload: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _json_get(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Multimodal (5016) local Whisper smoke test")
    ap.add_argument(
        "--base-url",
        default="http://localhost:5016",
        help="Multimodal backend base URL (default: http://localhost:5016)",
    )
    ap.add_argument(
        "--gcs-object",
        required=True,
        help="GCS object name under rawVideo/..., e.g. rawVideo/<id>/file.mp4",
    )
    ap.add_argument("--timeout-seconds", type=int, default=240, help="Max poll time")
    args = ap.parse_args()

    payload = {
        "gcs_object": args.gcs_object,
        "title": "whisper-smoke",
        "task_selection": {
            "enable_labels": False,
            "enable_text": False,
            "enable_moderation": False,
            "enable_celebrities": False,
            "enable_transcribe": True,
            "transcribe_model": "openai_whisper",
            "enable_rekognition_shots": False,
            "enable_rekognition_technical_cues": False,
        },
    }

    start = time.time()
    resp = _json_request(f"{args.base_url}/process-gcs-video-cloud", payload)
    job_id = resp.get("job_id")
    if not job_id:
        raise RuntimeError(f"No job_id in response: {resp}")

    print(f"job_id {job_id}")

    last_line = None
    while True:
        job = _json_get(f"{args.base_url}/jobs/{job_id}")
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
        raise RuntimeError(f"Job did not complete successfully: {job.get('status')}: {job.get('error') or job.get('message')}")

    video = _json_get(f"{args.base_url}/video/{job_id}")
    transcript = (video.get("transcript") or "").strip()

    print(f"transcript_chars {len(transcript)}")
    print("preview", transcript[:300].replace("\n", " "))

    if not transcript:
        raise RuntimeError("Transcript is empty")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
