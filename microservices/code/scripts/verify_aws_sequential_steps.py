#!/usr/bin/env python3
"""DEPRECATED: AWS-only step sequencing verifier.

AWS S3/Rekognition support has been removed from this repo.

Use these instead:
- code/scripts/local_only_smoke_multimodal.py (multimodal :5016, GCS input)
- code/scripts/whisper_smoke_multimodal.py (whisper-only :5016)
"""

def main() -> int:
    print(
        "This script is deprecated because AWS S3/Rekognition support was removed.\n"
        "Run one of:\n"
        "  cd code && ./.venv/bin/python scripts/local_only_smoke_multimodal.py\n"
        "  cd code && ./.venv/bin/python scripts/whisper_smoke_multimodal.py\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
