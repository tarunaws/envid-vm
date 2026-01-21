import json
import urllib.error
import urllib.request

BASE = "http://localhost:5016"


def _fetch_json(url: str, timeout: int = 30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _print_result(url: str):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            print(url, "->", r.status)
            print("  content-type:", r.headers.get("Content-Type"))
            print("  location:", r.headers.get("Location"))
            print("  disposition:", r.headers.get("Content-Disposition"))
            print("  sample:", r.read(40))
    except urllib.error.HTTPError as e:
        print(url, "->", e.code)
        print("  content-type:", e.headers.get("Content-Type"))
        print("  location:", e.headers.get("Location"))
        print("  disposition:", e.headers.get("Content-Disposition"))
        print("  body:", e.read(200))


def main():
    data = _fetch_json(f"{BASE}/videos")
    videos = data.get("videos") if isinstance(data, dict) else data
    if not isinstance(videos, list) or not videos:
        raise SystemExit(f"Expected a non-empty list from {BASE}/videos")

    vid = videos[0].get("id")
    if not vid:
        raise SystemExit("No video id found")

    print("video_id", vid)

    urls = [
        f"{BASE}/video/{vid}/metadata-json?category=combined&download=true",
        f"{BASE}/video/{vid}/metadata-json?category=synopses_by_age_group&download=true",
        f"{BASE}/video/{vid}/metadata-json.zip",
    ]
    for url in urls:
        _print_result(url)


if __name__ == "__main__":
    main()
