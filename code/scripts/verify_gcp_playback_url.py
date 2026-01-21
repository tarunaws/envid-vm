#!/usr/bin/env python3

from __future__ import annotations

import argparse
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify /video-file redirect and signed cloud URL usability")
    parser.add_argument("--base", default="http://localhost:5016", help="Backend base URL")
    parser.add_argument("video_id")
    args = parser.parse_args()

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, hdrs, newurl):  # type: ignore[override]
            return None

    opener = urllib.request.build_opener(NoRedirect)

    req = urllib.request.Request(f"{args.base}/video-file/{args.video_id}", method="HEAD")
    try:
        resp = opener.open(req, timeout=20)
        status = getattr(resp, "status", None)
        loc = resp.headers.get("Location")
    except urllib.error.HTTPError as e:
        status = e.code
        loc = e.headers.get("Location")

    print("backend_status", status)
    print("location", loc)

    if not loc:
        print("ERROR: missing Location header")
        return 2

    # Validate the signed URL is usable without downloading the whole file.
    range_req = urllib.request.Request(loc)
    range_req.add_header("Range", "bytes=0-1")

    try:
        r = urllib.request.urlopen(range_req, timeout=20)
        print("gcs_status", getattr(r, "status", None))
        print("gcs_content_range", r.headers.get("Content-Range"))
        return 0
    except urllib.error.HTTPError as e:
        print("gcs_http_error", e.code)
        print("gcs_error_head", (e.read(300) or b"")[:300])
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
