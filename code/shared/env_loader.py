"""Helpers to load layered .env files for local development."""
from __future__ import annotations

from typing import Optional

import os

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    find_dotenv = None  # type: ignore[assignment]
    load_dotenv = None  # type: ignore[assignment]


def _load_file(filename: str, *, override: bool) -> bool:
    if not load_dotenv or not find_dotenv:
        return False
    path = find_dotenv(filename=filename, raise_error_if_not_found=False, usecwd=True)
    if not path:
        return False
    return bool(load_dotenv(path, override=override))


def load_environment() -> bool:
    """Load shared `.env` first, then override with `.env.local` if present."""
    if not load_dotenv or not find_dotenv:
        return False
    loaded = False
    loaded = _load_file(".env", override=False) or loaded
    loaded = _load_file(".env.local", override=True) or loaded

    # Canonical S3 bucket: allow a single variable (`S3_BUCKET`) to drive all
    # service-specific bucket env vars while still allowing per-service overrides.
    canonical_bucket = (os.getenv("S3_BUCKET") or "").strip()
    if canonical_bucket:
        for bucket_key in (
            # Common legacy variables used throughout services
            "AWS_S3_BUCKET",
            "MEDIA_S3_BUCKET",
            "VIDEO_GEN_S3_BUCKET",
            "MEDIA_SUPPLY_CHAIN_UPLOAD_BUCKET",
            "PERSONALIZED_TRAILER_S3_BUCKET",
            # Service-specific overrides (only set if missing)
            "SEMANTIC_SEARCH_BUCKET",
            "CONTENT_MODERATION_BUCKET",
            "SCENE_SUMMARY_S3_BUCKET",
            "TRANSCRIBE_BUCKET",
            "MEDIA_SUPPLY_CHAIN_BUCKET",
            "AI_BASED_TRAILER_S3_BUCKET",
        ):
            os.environ.setdefault(bucket_key, canonical_bucket)
    else:
        # Ensure `S3_BUCKET` is still available when legacy vars are set.
        for legacy_bucket_key in (
            "MEDIA_S3_BUCKET",
            "AWS_S3_BUCKET",
            "VIDEO_GEN_S3_BUCKET",
        ):
            legacy_bucket = (os.getenv(legacy_bucket_key) or "").strip()
            if legacy_bucket:
                os.environ.setdefault("S3_BUCKET", legacy_bucket)
                break

    return loaded
