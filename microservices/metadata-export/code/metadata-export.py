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
    base = f"{artifacts_prefix}/{job_id}".strip("/")

    client = _gcs_client()
    bkt = client.bucket(artifacts_bucket)

    out: dict[str, Any] = {
        "bucket": artifacts_bucket,
        "base_prefix": base,
        "combined": None,
        "categories": {},
        "zip": None,
        "subtitles": {},
    }

    combined_obj = f"{base}/combined.json"
    bkt.blob(combined_obj).upload_from_string(
        json.dumps(payload.get("combined") or {}, indent=2, ensure_ascii=False),
        content_type="application/json; charset=utf-8",
    )
    out["combined"] = {"object": combined_obj, "uri": f"gs://{artifacts_bucket}/{combined_obj}"}

    cats = payload.get("categories") if isinstance(payload.get("categories"), dict) else {}
    for name, cat_payload in (cats or {}).items():
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name or "category").strip()) or "category"
        obj = f"{base}/categories/{safe}.json"
        bkt.blob(obj).upload_from_string(
            json.dumps(cat_payload, indent=2, ensure_ascii=False),
            content_type="application/json; charset=utf-8",
        )
        out["categories"][str(name)] = {"object": obj, "uri": f"gs://{artifacts_bucket}/{obj}"}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{job_id}.metadata.json", json.dumps(payload, indent=2, ensure_ascii=False))
    buf.seek(0)
    zip_obj = f"{base}/metadata_json.zip"
    bkt.blob(zip_obj).upload_from_file(buf, rewind=True, content_type="application/zip")
    out["zip"] = {"object": zip_obj, "uri": f"gs://{artifacts_bucket}/{zip_obj}"}

    for key, subtitle in subtitles.items():
        if not isinstance(subtitle, dict):
            continue
        content = subtitle.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        content_type = str(subtitle.get("content_type") or "text/plain").strip() or "text/plain"
        obj = f"{base}/subtitles/{key}"
        bkt.blob(obj).upload_from_string(content, content_type=content_type)
        out["subtitles"][str(key)] = {"object": obj, "uri": f"gs://{artifacts_bucket}/{obj}"}

    return jsonify({"artifacts": out}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5096")))
