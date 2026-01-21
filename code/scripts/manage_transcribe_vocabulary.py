#!/usr/bin/env python3

"""Deprecated.

This script managed AWS Transcribe custom vocabularies. AWS Transcribe has been
removed from this repo in favor of local transcription (whisper.cpp).
"""

import sys


def main() -> int:
    sys.stderr.write(
        "Deprecated: AWS Transcribe is not used in this repo anymore (local whisper.cpp is used instead).\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
