#!/usr/bin/env python3

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


def _load_video_ids(index_path: Path) -> list[str]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("videos"), list):
        videos = data.get("videos") or []
    elif isinstance(data, list):
        videos = data
    else:
        videos = []

    ids: list[str] = []
    for v in videos:
        if not isinstance(v, dict):
            continue
        vid = (v.get("id") or "").strip()
        if vid:
            ids.append(vid)
    # De-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for vid in ids:
        if vid in seen:
            continue
        seen.add(vid)
        out.append(vid)
    return out


def _http_get_json(url: str, timeout_seconds: float) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Force-regenerate synopses for every Envid Metadata video in the local index by calling the running backend."
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:5016",
        help="Base URL for the envid-metadata service (default: http://localhost:5016)",
    )
    parser.add_argument(
        "--index-path",
        default=None,
        help="Path to envidMetadata/indices/video_index.json (default: auto-detect from repo)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process first N videos (0 = all)")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Delay between requests to avoid hammering the backend (default: 0.2)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="Per-request timeout in seconds (default: 180)",
    )

    args = parser.parse_args()

    if args.index_path:
        index_path = Path(args.index_path).expanduser().resolve()
    else:
        # scripts/ -> code/ -> repo_root/
        repo_root = Path(__file__).resolve().parents[2]
        index_path = repo_root / "code" / "envidMetadata" / "indices" / "video_index.json"

    if not index_path.exists():
        print(f"ERROR: index not found: {index_path}", file=sys.stderr)
        return 2

    ids = _load_video_ids(index_path)
    if args.limit and args.limit > 0:
        ids = ids[: args.limit]

    if not ids:
        print("No videos found in index.")
        return 0

    total = len(ids)
    ok = 0
    failed = 0

    for i, vid in enumerate(ids, start=1):
        qs = urllib.parse.urlencode({"category": "combined", "force_synopses_regen": "1"})
        url = f"{args.base_url.rstrip('/')}/video/{vid}/metadata-json?{qs}"

        try:
            data = _http_get_json(url, timeout_seconds=float(args.timeout_seconds))
            syn = ((data.get("categories") or {}).get("synopses_by_age_group") or {})
            groups = sorted(list(syn.keys()))
            if groups:
                ok += 1
                print(f"[{i}/{total}] OK {vid} groups={groups}")
            else:
                failed += 1
                print(f"[{i}/{total}] FAIL {vid} (no synopses returned)")
        except Exception as exc:
            failed += 1
            print(f"[{i}/{total}] FAIL {vid} ({exc})")

        if args.sleep_seconds and args.sleep_seconds > 0 and i < total:
            time.sleep(float(args.sleep_seconds))

    print(f"Done. ok={ok} failed={failed} total={total}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
