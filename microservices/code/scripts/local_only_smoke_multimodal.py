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


def _steps_by_id(job: dict) -> dict:
    out = {}
    for s in (job.get("steps") or []):
        if isinstance(s, dict) and s.get("id"):
            out[str(s.get("id"))] = s
    return out


def _iter_messages(job: dict) -> list[str]:
    msgs: list[str] = []
    for k in ("message", "error"):
        v = job.get(k)
        if isinstance(v, str) and v.strip():
            msgs.append(v.strip())
    for s in (job.get("steps") or []):
        if not isinstance(s, dict):
            continue
        sid = s.get("id")
        sm = s.get("message")
        if isinstance(sm, str) and sm.strip():
            if isinstance(sid, str) and sid.strip():
                msgs.append(f"step[{sid}] {sm.strip()}")
            else:
                msgs.append(sm.strip())
    return msgs


def _looks_like_s3_proxy_activity(msg: str) -> bool:
    m = (msg or "").strip().lower()
    if not m:
        return False

    # Don't fail on explicit "skipped/disabled" messages.
    if any(w in m for w in ("skip", "skipped", "disabled", "not enabled", "not required", "no-op")):
        return False

    # Strong signals of S3/proxy activity.
    strong_phrases = [
        "uploading video to s3",
        "uploading proxy to s3",
        "upload video to s3",
        "upload proxy to s3",
        "s3_proxy",
        "s3 proxy",
    ]
    if any(p in m for p in strong_phrases):
        return True

    # General heuristic: mention of upload + s3, or proxy + s3.
    if ("upload" in m and "s3" in m) or ("proxy" in m and "s3" in m):
        return True

    # Also treat raw s3 URI references as suspicious if they appear alongside verbs.
    if "s3://" in m and any(v in m for v in ("upload", "put", "copy", "transcode")):
        return True

    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Multimodal (5016) local-only smoke test: PaddleOCR + Whisper")
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
    ap.add_argument("--timeout-seconds", type=int, default=420, help="Max poll time")
    args = ap.parse_args()

    payload = {
        "gcs_object": args.gcs_object,
        "title": "local-only-smoke",
        "task_selection": {
            "enable_labels": False,
            "enable_moderation": False,
            "enable_celebrities": False,
            "enable_rekognition_shots": False,
            "enable_rekognition_technical_cues": False,
            "enable_transcribe": True,
            "transcribe_model": "openai_whisper",
            "enable_text": True,
            "text_model": "paddleocr",
        },
    }

    start = time.time()
    resp = _json_request(f"{args.base_url}/process-gcs-video-cloud", payload)
    job_id = resp.get("job_id")
    if not job_id:
        raise RuntimeError(f"No job_id in response: {resp}")

    print(f"job_id {job_id}")

    last_line = None
    job = {}
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

    steps = _steps_by_id(job)

    # Expectations for local-only mode:
    # - Transcribe should be completed
    # - Text step should be completed (even though it reuses step id rekognition_text)
    # - There should be no Rekognition async job ids present
    want_steps = ["transcribe", "rekognition_text"]
    for sid in want_steps:
        s = steps.get(sid) or {}
        print(f"step {sid}: {s.get('status')} - {s.get('message')}")

    rk_jobs = job.get("rekognition_job_ids")
    if rk_jobs:
        raise RuntimeError(f"Unexpected rekognition_job_ids present in local-only job: {rk_jobs}")

    # Enforce: local-only runs should not do S3/proxy work.
    suspicious = [m for m in _iter_messages(job) if _looks_like_s3_proxy_activity(m)]
    if suspicious:
        raise RuntimeError(
            "Local-only job appears to have attempted S3/proxy activity. "
            "First suspicious message: "
            + suspicious[0]
        )

    video = _json_get(f"{args.base_url}/video/{job_id}")

    transcript = (video.get("transcript") or "").strip()
    print(f"transcript_chars {len(transcript)}")

    # text outputs vary; check that we have *some* text detections in any common field.
    text_detailed = video.get("text_detailed")
    text_raw = video.get("text_raw")
    print(f"text_detailed_count {len(text_detailed) if isinstance(text_detailed, list) else 0}")
    print(f"text_raw_count {len(text_raw) if isinstance(text_raw, list) else 0}")

    if not transcript:
        raise RuntimeError("Transcript is empty")

    if not (isinstance(text_detailed, list) and text_detailed) and not (isinstance(text_raw, list) and text_raw):
        raise RuntimeError("No text detections found (text_detailed/text_raw both empty)")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
