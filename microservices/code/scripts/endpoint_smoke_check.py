"""Small HTTP status smoke-check without curl.

Usage:
    code/.venv/bin/python code/scripts/endpoint_smoke_check.py

Prints HTTP status codes for a handful of endpoints on the Envid metadata service.
"""

from __future__ import annotations

import urllib.error
import urllib.request


def _status(url: str, method: str = "GET") -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "endpoint-smoke-check"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return str(resp.getcode())
    except urllib.error.HTTPError as exc:
        return str(exc.code)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR:{exc.__class__.__name__}"


def main() -> None:
    print("BEGIN endpoint_smoke_check", flush=True)
    base = "http://localhost:5016"
    checks = [
        ("/health", f"{base}/health", "GET"),
        ("/gcs/rawvideo/list", f"{base}/gcs/rawvideo/list", "GET"),
        ("/local-faces", f"{base}/local-faces", "GET"),
        ("/local-faces/enroll (POST)", f"{base}/local-faces/enroll", "POST"),
    ]

    for label, url, method in checks:
        print(f"{label}: {_status(url, method=method)}", flush=True)


if __name__ == "__main__":
    main()
