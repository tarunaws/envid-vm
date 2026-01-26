"""Check the Envid metadata rawVideo listing endpoint.

Usage:
  code/.venv/bin/python code/scripts/check_gcp_rawvideo_list_endpoint.py

This avoids curl (not installed in some environments) and avoids shell quoting issues.
"""

from __future__ import annotations

import json
import urllib.request


def main() -> None:
    url = "http://localhost:5016/gcs/rawvideo/list?max_results=500"
    with urllib.request.urlopen(url, timeout=10) as resp:
        payload = json.load(resp)

    objs = payload.get("objects") or []
    print("bucket", payload.get("bucket"))
    print("prefix", payload.get("prefix"))
    print("count", len(objs))
    if objs:
        print("first", objs[0])


if __name__ == "__main__":
    main()
