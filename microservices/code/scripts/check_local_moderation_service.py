#!/usr/bin/env python3
"""Quick check for the local moderation NudeNet service."""

from __future__ import annotations

import json
import os
import urllib.request

BASE = (os.getenv("ENVID_METADATA_LOCAL_MODERATION_URL") or "http://localhost:5081").rstrip("/")


def main() -> int:
    with urllib.request.urlopen(f"{BASE}/health", timeout=10) as r:
        data = json.loads(r.read().decode("utf-8"))
    print("health:", data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
