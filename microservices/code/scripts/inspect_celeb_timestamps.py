#!/usr/bin/env python3
"""Inspect celebrity timestamp fields in metadata-json."""

from __future__ import annotations

import argparse
import json
import urllib.request


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id")
    parser.add_argument("--base-url", default="http://localhost:5016")
    args = parser.parse_args()

    url = f"{args.base_url.rstrip('/')}/video/{args.video_id}/metadata-json"
    obj = _fetch_json(url)

    celebs = (((obj.get("categories") or {}).get("celebrity_table") or {}).get("celebrities") or [])
    print(f"Video: {args.video_id}")
    print(f"Celebs: {len(celebs)}")

    for c in celebs:
        if not isinstance(c, dict):
            continue
        name = (c.get("name") or "").strip() or "(unknown)"
        ts_ms = c.get("timestamps_ms") or []
        ts_s = c.get("timestamps_seconds") or []
        segs_ms = c.get("segments_ms") or []
        segs_s = c.get("segments") or []

        print("\n" + name)
        print(f"  timestamps_ms:      {len(ts_ms)}  sample={list(ts_ms)[:12]}")
        print(f"  timestamps_seconds: {len(ts_s)}  sample={list(ts_s)[:12]}")
        print(f"  segments_ms:        {len(segs_ms)}  sample={list(segs_ms)[:3]}")
        print(f"  segments:           {len(segs_s)}  sample={list(segs_s)[:3]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
