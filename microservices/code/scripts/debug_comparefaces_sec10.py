#!/usr/bin/env python3

"""Deprecated.

This helper depended on AWS Rekognition + presigned S3 URLs to debug face comparisons.
AWS support has been removed from this repo.

If you want to debug face matching locally/GCP-side, we should add a new script that:
- extracts a frame via ffmpeg from a local file or a signed GCS URL
- runs a local face detector/embedding model (e.g., insightface)
"""

import sys


def main() -> int:
    sys.stderr.write(
        "Deprecated: debug_comparefaces_sec10.py required AWS Rekognition/S3, which are removed.\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
