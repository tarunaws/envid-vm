"""Check whether a GCS object exists and print close matches.

Usage:
  code/.venv/bin/python code/scripts/gcs_check_object.py <bucket> <object>

Example:
  code/.venv/bin/python code/scripts/gcs_check_object.py envid-metadata-tarun rawVideo/abhayPromo.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

from google.cloud import storage


def main(argv: list[str]) -> int:
    # Load local env file if present (keeps this runnable outside the repo scripts).
    if load_dotenv is not None:
        repo_root = Path(__file__).resolve().parents[2]
        for rel in (".env.multimodal.local", ".env.local"):
            env_path = repo_root / "code" / rel
            if env_path.exists():
                load_dotenv(env_path, override=False)

    if len(argv) != 3:
        print("Usage: gcs_check_object.py <bucket> <object>")
        return 2

    bucket_name = argv[1]
    object_name = argv[2]

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    exists = blob.exists(client)
    print(f"EXISTS {exists} gs://{bucket_name}/{object_name}")

    if exists:
        try:
            blob.reload(client)
            print(f"SIZE {blob.size}")
            print(f"CONTENT_TYPE {blob.content_type}")
        except Exception:
            pass
        return 0

    prefix = "rawVideo/"
    needle = object_name.split("/", 1)[-1].lower()
    needle_base = needle.rsplit(".", 1)[0]

    matches: list[str] = []
    sample: list[str] = []

    for bl in client.list_blobs(bucket_name, prefix=prefix, max_results=500):
        name = bl.name
        if len(sample) < 30:
            sample.append(name)
        low = name.lower()
        if needle_base and needle_base in low:
            matches.append(name)

    print(f"MATCHES_CONTAINING '{needle_base}': {len(matches)}")
    for name in matches[:50]:
        print(" -", name)

    print("SAMPLE_RAWVIDEO")
    for name in sample:
        print(" -", name)

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
