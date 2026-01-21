"""Utility helpers to purge stale upload artifacts across services."""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Iterable, Mapping

LOGGER = logging.getLogger(__name__)


def _derive_path(path_or_str: str | Path) -> Path:
    path = Path(path_or_str).expanduser().resolve()
    return path


def _path_size_bytes(target: Path) -> int:
    if not target.exists():
        return 0
    if target.is_file():
        return target.stat().st_size
    total = 0
    for child in target.rglob('*'):
        try:
            if child.is_file():
                total += child.stat().st_size
        except FileNotFoundError:
            continue
    return total


def purge_stale_artifacts(
    paths: Iterable[str | Path],
    *,
    retention_hours: int,
    logger: logging.Logger | None = None,
) -> Mapping[str, int]:
    """Delete files and folders older than ``retention_hours`` inside ``paths``.

    Returns a summary mapping with ``deleted`` and ``freed_bytes`` counts.
    """
    log = logger or LOGGER
    cutoff = time.time() - (retention_hours * 3600)
    deleted = 0
    freed = 0

    for base in paths:
        base_path = _derive_path(base)
        if not base_path.exists():
            log.debug("Skipping cleanup for missing path %s", base_path)
            continue
        try:
            entries = list(base_path.iterdir())
        except OSError as exc:
            log.warning("Unable to iterate %s: %s", base_path, exc)
            continue
        for entry in entries:
            try:
                mtime = entry.stat().st_mtime
            except FileNotFoundError:
                continue
            if mtime >= cutoff:
                continue
            reclaimed = _path_size_bytes(entry)
            try:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=False)
                else:
                    entry.unlink()
                deleted += 1
                freed += reclaimed
                log.info(
                    "Cleaned %s (%.2f MB)",
                    entry,
                    reclaimed / (1024 * 1024) if reclaimed else 0,
                )
            except OSError as exc:
                log.warning("Failed to remove %s: %s", entry, exc)

    return {"deleted": deleted, "freed_bytes": freed}
