#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="List videos from envid metadata backend")
    parser.add_argument("--base", default="http://localhost:5016")
    args = parser.parse_args()

    with urllib.request.urlopen(f"{args.base}/videos", timeout=20) as r:
        vids = json.load(r)

    if isinstance(vids, dict):
        vids_list = vids.get("videos") or vids.get("items") or vids.get("data") or []
    else:
        vids_list = vids

    if not isinstance(vids_list, list):
        print("unexpected_shape", type(vids).__name__, "wrapper_keys", list(vids.keys()) if isinstance(vids, dict) else None)
        return 2

    print("count", len(vids_list))
    for v in vids_list[:20]:
        print("-", v.get("id") or v.get("video_id"), "|", v.get("title"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
