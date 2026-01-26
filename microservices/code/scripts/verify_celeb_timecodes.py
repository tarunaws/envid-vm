#!/usr/bin/env python3
"""Verify whether specific timecodes fall within celebrity segments.

This is a small local helper for debugging Rekognition/metadata alignment.

Usage:
    python3 code/scripts/verify_celeb_timecodes.py <video_id> [--base-url http://localhost:5016]

Exit code:
  0 if all requested timecodes are covered, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def _segments_for(celebrity: dict) -> list[tuple[float, float, str]]:
    segments: list[tuple[float, float, str]] = []

    for key in ("segments_ms", "segments"):
        for seg in (celebrity.get(key) or []):
            if not isinstance(seg, (list, tuple)) or len(seg) < 2:
                continue
            start, end = seg[0], seg[1]
            if key == "segments_ms":
                start = float(start) / 1000.0
                end = float(end) / 1000.0
            else:
                start = float(start)
                end = float(end)
            segments.append((start, end, key))

    segments.sort(key=lambda x: (x[0], x[1], x[2]))
    return segments


def _covers(segments: list[tuple[float, float, str]], t_seconds: float) -> tuple[bool, tuple[float, float, str] | None]:
    for start, end, key in segments:
        if start <= t_seconds <= end:
            return True, (start, end, key)
    return False, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    parser.add_argument("--base-url", default="http://localhost:5016")
    args = parser.parse_args()

    url = f"{args.base_url.rstrip('/')}/video/{args.video_id}/metadata-json"
    vid = _fetch_json(url)
    celebs = (((vid.get("categories") or {}).get("celebrity_table") or {}).get("celebrities") or [])

    requested: dict[str, list[int]] = {
        "Kunal Khemu": [8, 16, 23, 27],
        "Tanuj Virwani": [19, 21],
        "Vijay Raaz": [12],
    }

    by_name = {c.get("name"): c for c in celebs if c.get("name")}

    print(f"Video: {args.video_id}")
    print(f"URL:   {url}")
    print(f"Celebs in metadata: {len(celebs)}")

    all_ok = True
    for name, times in requested.items():
        c = by_name.get(name)
        if not c:
            print(f"\n{name}: NOT PRESENT in metadata")
            all_ok = False
            continue

        segments = _segments_for(c)
        print(f"\n{name}: segments={len(segments)}")
        for start, end, key in segments[:12]:
            print(f"  - {start:.3f}..{end:.3f} ({key})")

        for t in times:
            ok, hit = _covers(segments, float(t))
            if ok and hit:
                start, end, key = hit
                print(f"  t={t:>2}s: COVERED by {start:.3f}..{end:.3f} ({key})")
            else:
                print(f"  t={t:>2}s: NOT covered")
                all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
