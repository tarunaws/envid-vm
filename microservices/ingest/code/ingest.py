from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

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
        raise ValueError(
            "Missing GCS bucket. Set ENVID_METADATA_GCS_BUCKET (or GCP_GCS_BUCKET)."
        )
    return bucket


def _gcs_rawvideo_prefix() -> str:
    prefix = (os.getenv("GCP_GCS_RAWVIDEO_PREFIX") or "rawVideo/").strip()
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix


def _gcs_client() -> Any:
    if gcs_storage is None:
        raise RuntimeError("google-cloud-storage is not installed")
    return gcs_storage.Client()


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"ok": True}), 200


@app.route("/upload-video", methods=["POST"])
def upload_video() -> Any:
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    video_file = request.files["video"]
    if not video_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    job_id = (request.form.get("job_id") or "").strip() or str(uuid.uuid4())
    original_filename = video_file.filename

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(original_filename).suffix or ".mp4") as tmp:
            video_file.save(tmp.name)
            tmp_path = Path(tmp.name)

        bucket = _gcs_bucket_name()
        prefix = _gcs_rawvideo_prefix()
        safe_name = Path(original_filename).name or "video.mp4"
        obj = f"{prefix}{job_id}/{safe_name}"
        gcs_uri = f"gs://{bucket}/{obj}"

        client = _gcs_client()
        client.bucket(bucket).blob(obj).upload_from_filename(str(tmp_path))

        return jsonify({"job_id": job_id, "gcs_bucket": bucket, "gcs_object": obj, "gcs_uri": gcs_uri}), 201
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5090")))
