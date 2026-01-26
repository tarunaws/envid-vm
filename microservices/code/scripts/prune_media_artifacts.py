#!/usr/bin/env python3
"""Purge stale upload/audio/output artifacts to reclaim disk space."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from shared.artifact_cleanup import purge_stale_artifacts
from shared.logging_utils import configure_json_logging

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = [
    ROOT / "aiSubtitle" / "uploads",
    ROOT / "aiSubtitle" / "audio",
    ROOT / "aiSubtitle" / "outputs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean old aiSubtitle artifacts")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=DEFAULT_PATHS,
        help="One or more directories to clean (defaults to aiSubtitle storage)",
    )
    parser.add_argument(
        "--retention-hours",
        type=int,
        default=int(os.environ.get("ARTIFACT_RETENTION_HOURS", "72")),
        help="Delete artifacts older than this many hours (default: 72)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = configure_json_logging("artifact-cleanup")
    summary = purge_stale_artifacts(
        args.paths,
        retention_hours=args.retention_hours,
        logger=logger,
    )
    logger.info(
        "Cleanup complete", extra={"deleted": summary["deleted"], "freed_bytes": summary["freed_bytes"]}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
