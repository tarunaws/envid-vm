from __future__ import annotations

import io
import json
import os
import re
import zipfile
from typing import Any, Dict

from flask import Flask, jsonify, request

try:
    from google.cloud import storage as gcs_storage  # type: ignore
except Exception:  # pragma: no cover
    gcs_storage = None  # type: ignore

app = Flask(__name__)


def _normalize_artifact_lang(lang: str | None) -> str:
    lang_norm = (lang or "").strip().lower()
    if not lang_norm or lang_norm in {"original", "orig"}:
        return "orig"
    return lang_norm


def _collect_artifact_languages(
    *,
    payload_languages: list[str] | None = None,
    combined_by_language: dict[str, Any] | None = None,
    subtitles_payload: dict[str, Any] | None = None,
    include_orig: bool = True,
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def _add(lang: str | None) -> None:
        lang_norm = _normalize_artifact_lang(lang)
        if not lang_norm or lang_norm in seen:
            return
        seen.add(lang_norm)
        out.append(lang_norm)

    if include_orig:
        _add("orig")

    if payload_languages:
        for lang in payload_languages:
            _add(str(lang or "").strip())

    if combined_by_language:
        for lang in combined_by_language.keys():
            _add(str(lang or "").strip())

    if subtitles_payload:
        for key in subtitles_payload.keys():
            lang_part = str(key or "").split(".")[0].strip()
            if lang_part:
                _add(lang_part)

    return out


def _build_metadata_zip_bytes(
    *,
    job_id: str,
    categories: dict[str, Any],
    combined_by_language: dict[str, dict[str, Any]],
    subtitles_payload: dict[str, dict[str, Any]],
    languages: list[str],
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for lang in languages:
            lang_norm = _normalize_artifact_lang(lang)
            combined = combined_by_language.get(lang_norm) or combined_by_language.get("orig") or {}

            z.writestr(
                f"metadata/{lang_norm}/metadata/combined.json",
                json.dumps(combined, indent=2, ensure_ascii=False),
            )

            for name, cat_payload in (categories or {}).items():
                safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name or "category").strip()) or "category"
                z.writestr(
                    f"metadata/{lang_norm}/metadata/categories/{safe}.json",
                    json.dumps(cat_payload, indent=2, ensure_ascii=False),
                )

            for fmt in ("srt", "vtt"):
                key = f"{lang_norm}.{fmt}"
                subtitle = subtitles_payload.get(key) if isinstance(subtitles_payload, dict) else None
                if not isinstance(subtitle, dict):
                    continue
                content = subtitle.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                z.writestr(
                    f"metadata/{lang_norm}/subtitles/subtitles.{fmt}",
                    content,
                )
    buf.seek(0)
    return buf.read()


def _gcs_bucket_name() -> str:
    bucket = (
        os.getenv("ENVID_METADATA_GCS_BUCKET")
        or os.getenv("GCP_GCS_BUCKET")
        or os.getenv("GCS_BUCKET")
        or ""
    ).strip()
    if not bucket:
        raise ValueError("Missing GCS bucket. Set ENVID_METADATA_GCS_BUCKET or GCP_GCS_BUCKET.")
    return bucket


def _gcs_artifacts_bucket(fallback: str) -> str:
    return (
        (os.getenv("ENVID_METADATA_GCS_ARTIFACTS_BUCKET")
        or os.getenv("ENVID_METADATA_GCP_ARTIFACTS_BUCKET")
        or fallback)
        .strip()
        or fallback
    )


def _gcs_artifacts_prefix() -> str:
    return (
        os.getenv("ENVID_METADATA_GCS_ARTIFACTS_PREFIX")
        or os.getenv("ENVID_METADATA_GCP_ARTIFACTS_PREFIX")
        or "envid-metadata/artifacts"
    ).strip().strip("/")


def _gcs_client() -> Any:
    if gcs_storage is None:
        raise RuntimeError("google-cloud-storage is not installed")
    return gcs_storage.Client()


@app.get("/health")
def health() -> Any:
    return jsonify({"ok": True}), 200


@app.post("/upload_artifacts")
def upload_artifacts() -> Any:
    data = request.get_json(silent=True) or {}
    job_id = str(data.get("job_id") or "").strip()
    payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
    subtitles = data.get("subtitles") if isinstance(data.get("subtitles"), dict) else {}
    if not job_id:
        return jsonify({"error": "Missing job_id"}), 400

    artifacts_bucket = _gcs_artifacts_bucket(_gcs_bucket_name())
    artifacts_prefix = _gcs_artifacts_prefix()
    base = f"{artifacts_prefix}/{job_id}/metadata".strip("/")

    client = _gcs_client()
    bkt = client.bucket(artifacts_bucket)

    out: dict[str, Any] = {
        "bucket": artifacts_bucket,
        "base_prefix": base,
        "combined": None,
        "categories": {},
        "zip": None,
        "subtitles": {},
        "languages": [],
    }

    cats = payload.get("categories") if isinstance(payload.get("categories"), dict) else {}
    combined = payload.get("combined") if isinstance(payload.get("combined"), dict) else {}
    combined_by_language = (
        payload.get("combined_by_language") if isinstance(payload.get("combined_by_language"), dict) else None
    )
    payload_languages = payload.get("languages") if isinstance(payload.get("languages"), list) else None
    include_orig = True
    if isinstance(payload.get("include_orig"), bool):
        include_orig = bool(payload.get("include_orig"))
    languages = _collect_artifact_languages(
        payload_languages=payload_languages,
        combined_by_language=combined_by_language,
        subtitles_payload=subtitles,
        include_orig=include_orig,
    )
    if not combined_by_language:
        combined_by_language = {"orig": combined}
    out["languages"] = languages

    for lang in languages:
        lang_norm = _normalize_artifact_lang(lang)
        combined_lang = combined_by_language.get(lang_norm) or combined_by_language.get("orig") or {}
        combined_obj = f"{base}/{lang_norm}/metadata/combined.json"
        bkt.blob(combined_obj).upload_from_string(
            json.dumps(combined_lang, indent=2, ensure_ascii=False),
            content_type="application/json; charset=utf-8",
        )
        if lang_norm == "orig":
            out["combined"] = {"object": combined_obj, "uri": f"gs://{artifacts_bucket}/{combined_obj}"}

        for name, cat_payload in (cats or {}).items():
            safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name or "category").strip()) or "category"
            obj = f"{base}/{lang_norm}/metadata/categories/{safe}.json"
            bkt.blob(obj).upload_from_string(
                json.dumps(cat_payload, indent=2, ensure_ascii=False),
                content_type="application/json; charset=utf-8",
            )
            if lang_norm == "orig":
                out["categories"][str(name)] = {"object": obj, "uri": f"gs://{artifacts_bucket}/{obj}"}

    for key, subtitle in subtitles.items():
        if not isinstance(subtitle, dict):
            continue
        content = subtitle.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        content_type = str(subtitle.get("content_type") or "text/plain").strip() or "text/plain"
        parts = str(key or "").split(".")
        if len(parts) < 2:
            continue
        lang_norm = _normalize_artifact_lang(parts[0])
        fmt = parts[1].strip().lower()
        if fmt not in {"srt", "vtt"}:
            continue
        obj = f"{base}/{lang_norm}/subtitles/subtitles.{fmt}"
        bkt.blob(obj).upload_from_string(content, content_type=content_type)
        out["subtitles"][f"{lang_norm}.{fmt}"] = {"object": obj, "uri": f"gs://{artifacts_bucket}/{obj}"}

    zip_bytes = _build_metadata_zip_bytes(
        job_id=job_id,
        categories=cats or {},
        combined_by_language=combined_by_language or {"orig": combined},
        subtitles_payload=subtitles,
        languages=languages,
    )
    zip_obj = f"{base}/metadata_json.zip"
    bkt.blob(zip_obj).upload_from_string(zip_bytes, content_type="application/zip")
    out["zip"] = {"object": zip_obj, "uri": f"gs://{artifacts_bucket}/{zip_obj}"}

    return jsonify({"artifacts": out}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5096")))
