#!/usr/bin/env python3

"""Legacy Rekognition proxy-retry script (deprecated).

AWS Rekognition/S3 support has been removed from the demo stack.

This script is kept for convenience as a generic job starter/poller for the
multimodal backend (default :5016), using GCS (`/process-gcs-video-cloud`).

Preferred scripts:
- local-only: code/scripts/local_only_smoke_multimodal.py
- whisper-only: code/scripts/whisper_smoke_multimodal.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v or default


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _looks_like_job_not_found(body: str) -> bool:
    try:
        payload = json.loads(body)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    msg = str(payload.get("error") or "").strip().lower()
    return msg in {"job not found", "not found"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "[DEPRECATED] Rekognition proxy retry smoke. Now acts as a multimodal (5016) job starter/poller via GCS. "
            "Creates a job (or polls an existing job) and prints progress updates."
        )
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="Metadata service base URL (overrides ENVID_METADATA_URL; default: http://localhost:5016)",
    )
    p.add_argument(
        "--gcs-object",
        default=None,
        help=(
            "GCS object name under rawVideo/... to process (overrides ENVID_TEST_GCS_OBJECT), e.g. rawVideo/<id>/file.mp4"
        ),
    )
    p.add_argument(
        "--job-id",
        default=None,
        help="Existing job id to poll (overrides ENVID_JOB_ID). If omitted, creates a new job.",
    )
    p.add_argument(
        "--poll-seconds",
        type=float,
        default=None,
        help="Polling interval seconds (overrides ENVID_POLL_SECONDS; default: 4)",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Max runtime seconds (overrides ENVID_MAX_SECONDS; default: 1800)",
    )
    p.add_argument(
        "--restart-on-job-not-found",
        action="store_true",
        help=(
            "If an existing job-id returns 404 (backend restart / in-memory job store), "
            "start a fresh job and continue polling. Also enabled by ENVID_RESTART_ON_JOB_NOT_FOUND=1."
        ),
    )
    return p.parse_args()


def _step_map(job: dict) -> dict[str, dict]:
    steps = job.get("steps")
    if not isinstance(steps, list):
        return {}
    out: dict[str, dict] = {}
    for s in steps:
        if isinstance(s, dict) and isinstance(s.get("id"), str):
            out[s["id"]] = s
    return out


def _step_summary(step: dict | None) -> tuple[str, int | None, str]:
    if not isinstance(step, dict):
        return ("missing", None, "")
    status = str(step.get("status") or "").strip().lower() or "unknown"
    percent = step.get("percent")
    try:
        percent_int = int(percent) if percent is not None else None
    except Exception:
        percent_int = None
    message = str(step.get("message") or "").strip()
    return (status, percent_int, message)


def main() -> int:
    args = _parse_args()

    base_url = (
        (args.base_url or "").strip() or _env("ENVID_METADATA_URL", "http://localhost:5016")
    ).rstrip("/")
    gcs_object = ((args.gcs_object or "").strip() or _env("ENVID_TEST_GCS_OBJECT", "").strip()).strip()
    existing_job_id = ((args.job_id or "").strip() or _env("ENVID_JOB_ID", "").strip()).strip()
    poll_seconds = float(args.poll_seconds) if args.poll_seconds is not None else float(_env("ENVID_POLL_SECONDS", "4"))
    max_seconds = float(args.max_seconds) if args.max_seconds is not None else float(_env("ENVID_MAX_SECONDS", "1800"))

    restart_on_not_found_raw = _env("ENVID_RESTART_ON_JOB_NOT_FOUND", "0").strip().lower()
    restart_on_not_found_env = restart_on_not_found_raw in {"1", "true", "yes", "y", "on"}
    restart_on_not_found = bool(args.restart_on_job_not_found) or restart_on_not_found_env

    created_in_this_run = not bool(existing_job_id)

    if existing_job_id:
        job_id = existing_job_id
        print(f"job_id={job_id} (existing)")
    else:
        if not gcs_object:
            raise SystemExit(
                "Missing --gcs-object (or ENVID_TEST_GCS_OBJECT). "
                "Example: ENVID_TEST_GCS_OBJECT='rawVideo/<id>/file.mp4'"
            )

        start_url = f"{base_url}/process-gcs-video-cloud"
        payload = {
            "gcs_object": gcs_object,
            "title": "smoke-test",
            "description": "e2e multimodal run (local replacements)",
            "task_selection": {
                "enable_labels": False,
                "enable_moderation": False,
                "enable_celebrities": False,
                "enable_text": True,
                "text_model": "paddleocr",
                "enable_transcribe": True,
                "transcribe_model": "local_whisper_cpp",
                "enable_rekognition_shots": True,
                "enable_rekognition_technical_cues": True,
            },
        }

        try:
            start = _post_json(start_url, payload)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise SystemExit(f"POST failed: {e.code} {e.reason}: {body[:1000]}")

        job_id = (start.get("job_id") or "").strip()
        if not job_id:
            raise SystemExit(f"No job_id in response: {start}")
        print(f"job_id={job_id}")

    job_url = f"{base_url}/jobs/{job_id}"
    deadline = time.time() + max_seconds
    last_summary = None

    watched_steps = [
        "rekognition_text",
        "rekognition_shots",
        "rekognition_technical_cues",
        "transcribe",
    ]
    t0 = time.time()
    last_step_summaries: dict[str, tuple[str, int | None, str]] = {}
    first_running_at: dict[str, float] = {}
    first_completed_at: dict[str, float] = {}
    first_seen_at: dict[str, float] = {}

    while time.time() < deadline:
        try:
            job = _get_json(job_url)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 404 and _looks_like_job_not_found(body):
                if created_in_this_run:
                    print(f"fatal=1 reason=job_disappeared status=404 url={job_url}")
                    print(body[:1000])
                    return 2

                if restart_on_not_found:
                    if not gcs_object:
                        print(f"fatal=1 reason=job_not_found_and_no_gcs_object status=404 url={job_url}")
                        return 2
                    print(f"warn=1 reason=job_not_found_will_restart status=404 url={job_url}")
                    start_url = f"{base_url}/process-gcs-video-cloud"
                    payload = {
                        "gcs_object": gcs_object,
                        "title": "smoke-test",
                        "description": "e2e multimodal run (restart)",
                    }
                    try:
                        start = _post_json(start_url, payload)
                    except urllib.error.HTTPError as err:
                        start_body = err.read().decode("utf-8", errors="replace")
                        print(f"fatal=1 reason=restart_post_failed status={err.code} url={start_url}")
                        print(start_body[:1000])
                        return 2

                    new_job_id = (start.get("job_id") or "").strip()
                    if not new_job_id:
                        print("fatal=1 reason=restart_missing_job_id")
                        print(str(start)[:1000])
                        return 2

                    job_id = new_job_id
                    job_url = f"{base_url}/jobs/{job_id}"
                    created_in_this_run = True
                    last_summary = None
                    print(f"job_id={job_id} (restarted)")
                    time.sleep(min(max(poll_seconds, 0.1), 5.0))
                    continue

                print(f"fatal=1 reason=job_not_found status=404 url={job_url}")
                print(
                    "Hint: job status is stored in-memory by the backend; if the service restarted, it will forget old job IDs. "
                    "Re-run without ENVID_JOB_ID to start a fresh job, or set ENVID_RESTART_ON_JOB_NOT_FOUND=1."
                )
                return 2

            print(f"fatal=1 reason=http_error status={e.code} url={job_url}")
            print(body[:1000])
            return 2
        except urllib.error.URLError as e:
            print(f"warn=1 reason=connection_error url={job_url} error={e}")
            time.sleep(poll_seconds)
            continue
        except json.JSONDecodeError as e:
            print(f"warn=1 reason=invalid_json url={job_url} error={e}")
            time.sleep(poll_seconds)
            continue

        status = (job.get("status") or "").strip().lower()
        progress = job.get("progress")
        message = (job.get("message") or "").strip()

        summary = (status, progress, message[:140])
        if summary != last_summary:
            last_summary = summary
            print(f"status={status} progress={progress} message={message[:140]}")

        smap = _step_map(job)
        for sid in watched_steps:
            step = smap.get(sid)
            s_status, s_percent, s_message = _step_summary(step)

            now = time.time()
            if sid not in first_seen_at and s_status != "missing":
                first_seen_at[sid] = now
            if sid not in first_running_at and s_status == "running":
                first_running_at[sid] = now
            if sid not in first_completed_at and s_status in {"completed", "skipped"}:
                first_completed_at[sid] = now

            prev = last_step_summaries.get(sid)
            cur = (s_status, s_percent, s_message[:140])
            if prev != cur:
                last_step_summaries[sid] = cur
                dt = now - t0
                percent_str = "" if s_percent is None else f" percent={s_percent}"
                msg_str = "" if not s_message else f" message={s_message[:140]}"
                print(f"t+{dt:.1f}s step={sid} status={s_status}{percent_str}{msg_str}")

        if status in {"completed", "failed"}:
            if job.get("error"):
                print(f"error={str(job.get('error'))[:800]}")
            if first_running_at:
                print("--- step_timings")
                for sid in watched_steps:
                    if sid not in first_seen_at:
                        continue
                    seen_dt = first_seen_at[sid] - t0
                    run_dt = (first_running_at.get(sid) - t0) if sid in first_running_at else None
                    done_dt = (first_completed_at.get(sid) - t0) if sid in first_completed_at else None
                    run_str = "" if run_dt is None else f" running_at=+{run_dt:.1f}s"
                    done_str = "" if done_dt is None else f" done_at=+{done_dt:.1f}s"
                    print(f"step={sid} seen_at=+{seen_dt:.1f}s{run_str}{done_str}")
            return 0 if status == "completed" else 2

        time.sleep(poll_seconds)

    print("timed_out=1")
    if last_step_summaries:
        print("--- last_step_states")
        for sid in watched_steps:
            st = last_step_summaries.get(sid)
            if not st:
                continue
            s_status, s_percent, s_msg = st
            percent_str = "" if s_percent is None else f" percent={s_percent}"
            msg_str = "" if not s_msg else f" message={s_msg[:140]}"
            print(f"step={sid} status={s_status}{percent_str}{msg_str}")
    print("hint=rerun_with_higher_max_seconds or poll_existing_job_id")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
