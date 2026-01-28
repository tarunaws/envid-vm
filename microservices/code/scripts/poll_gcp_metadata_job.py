#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from urllib import error, request


def _get_json(url: str, *, timeout: int = 30):
    with request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def _head(url: str, *, timeout: int = 30):
    req = request.Request(url, method="HEAD")
    return request.urlopen(req, timeout=timeout)


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def _get_redirect_location(url: str, *, timeout: int = 30):
    opener = request.build_opener(_NoRedirect)
    req = request.Request(url, method="GET")
    try:
        resp = opener.open(req, timeout=timeout)
        return getattr(resp, "status", None), resp.headers.get("Location")
    except error.HTTPError as e:
        # With no-redirect handler, 30x becomes an exception; headers still contain Location.
        return e.code, e.headers.get("Location")


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll Envid metadata job until completion.")
    parser.add_argument("--base", default="http://localhost:5016")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--job-id-file", default=".last_gcs_job_id")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=int, default=20 * 60)
    args = parser.parse_args()

    job_id = (args.job_id or "").strip()
    if not job_id:
        job_id = Path(args.job_id_file).read_text(encoding="utf-8").strip()

    print("Polling job_id", job_id)
    deadline = time.time() + args.max_seconds
    i = 0
    while True:
        i += 1
        job = _get_json(f"{args.base}/jobs/{job_id}")
        status = (job.get("status") or "").strip()
        progress = job.get("progress")
        msg = (job.get("message") or "").strip()
        print(f"[{i}] status={status} progress={progress} msg={msg}")

        if status == "completed":
            break
        if status == "failed":
            print(json.dumps(job, indent=2))
            return 2
        if time.time() >= deadline:
            print("Timed out")
            print(json.dumps(job, indent=2))
            return 3
        time.sleep(args.interval)

    meta = _get_json(f"{args.base}/video/{job_id}/metadata-json")
    # The GCP service typically returns a categorized JSON payload under `categories`.
    categories = meta.get("categories") or {}

    # Backward-compatible summary for any flat fields (if present).
    if any(k in meta for k in ("id", "title", "gcs_video_uri", "duration_seconds")):
        print("meta_id", meta.get("id"))
        print("meta_title", meta.get("title"))
        print("has_gcs_video_uri", bool(meta.get("gcs_video_uri")))
        print("duration_seconds", meta.get("duration_seconds"))

    print("category_keys", sorted(categories.keys()))
    subtitles = categories.get("subtitles") or {}
    print("subtitles_language_code", subtitles.get("language_code"))
    print("subtitles_keys", sorted(subtitles.keys()))

    detected = categories.get("detected_content") or {}
    print(
        "detected_counts",
        {
            "labels": len(detected.get("labels") or []),
            "on_screen_text": len(detected.get("on_screen_text") or []),
            "moderation": len(detected.get("moderation") or []),
        },
    )
    synopsis = categories.get("synopsis") or {}
    print("synopsis_keys", sorted(synopsis.keys()))

    try:
        try:
            r = _head(f"{args.base}/video-file/{job_id}")
            print("video_file_status", getattr(r, "status", None))
            print("video_file_location", r.headers.get("Location"))
        except Exception:
            status, location = _get_redirect_location(f"{args.base}/video-file/{job_id}")
            print("video_file_status", status)
            print("video_file_location", location)
    except Exception as exc:
        print("video_file_head_error", str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
