#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import urllib.request


def _len(value) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize key sections of /metadata-json")
    parser.add_argument("--base", default="http://localhost:5016")
    parser.add_argument("video_id")
    args = parser.parse_args()

    with urllib.request.urlopen(f"{args.base}/video/{args.video_id}/metadata-json", timeout=30) as r:
        meta = json.load(r)

    cat = meta.get("categories") or {}
    detected = cat.get("detected_content") or {}

    print("category_keys", sorted(cat.keys()))
    print("detected_content.labels", _len(detected.get("labels")))
    print("detected_content.on_screen_text", _len(detected.get("on_screen_text")))
    print("detected_content.moderation", _len(detected.get("moderation")))

    celeb_table = cat.get("celebrity_table") or {}
    print("celebrity_table.celebrities", _len(celeb_table.get("celebrities")))

    synopsis = (cat.get("synopsis") or {})
    print("synopsis.keys", sorted(synopsis.keys()))

    print("high_points", _len(cat.get("high_points")))
    print("key_scenes", _len(cat.get("key_scenes")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
