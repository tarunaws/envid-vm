import json
import urllib.error
import urllib.request

BASE = "http://localhost:5016"


def fetch(url: str):
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            body = r.read(300)
            return r.status, r.headers, body
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read(300)


def main():
    urls = [
        f"{BASE}/health",
        f"{BASE}/gcs/rawvideo/list",
        f"{BASE}/face-collection/create",  # should be absent in GCP-only surface
        f"{BASE}/videos/rebuild-index",  # should be absent in GCP-only surface
        f"{BASE}/list-s3-videos",  # should be absent in GCP-only surface
        f"{BASE}/rekognition-check",  # should be absent in GCP-only surface
    ]

    for url in urls:
        status, headers, body = fetch(url)
        print(url, "->", status)
        loc = headers.get("Location")
        if loc:
            print("  Location:", loc)
        ct = headers.get("Content-Type")
        if ct:
            print("  Content-Type:", ct)
        print("  body_sample:", body[:120])

    # POST endpoints that must be absent
    post_urls = [
        f"{BASE}/presign-upload-video",
        f"{BASE}/process-s3-video",
        f"{BASE}/process-s3-video-cloud",
    ]
    for url in post_urls:
        req = urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                body = r.read(300)
                print(url, "->", r.status)
                print("  body_sample:", body[:120])
        except urllib.error.HTTPError as e:
            body = e.read(300)
            print(url, "->", e.code)
            print("  body_sample:", body[:120])


if __name__ == "__main__":
    main()
