from __future__ import annotations

import base64
import io
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    return (os.getenv(name) or default).strip() or default


def _b64_to_pil(img_b64: str):
    from PIL import Image

    raw = base64.b64decode(img_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


@dataclass
class _ClipState:
    model: Any
    preprocess: Any
    device: str
    model_name: str
    pretrained: str


def _get_clip(*, model_name: str | None = None, pretrained: str | None = None, device: str | None = None) -> _ClipState:
    # Zero caching: always load a fresh model instance per call.
    import torch  # type: ignore
    import open_clip  # type: ignore

    req_model_name = (model_name or _env_str("CLIP_MODEL", "ViT-B-32")).strip() or "ViT-B-32"
    req_pretrained = (pretrained or _env_str("CLIP_PRETRAINED", "laion2b_s34b_b79k")).strip() or "laion2b_s34b_b79k"

    req_device = (device or "").strip().lower() or ("cuda" if torch.cuda.is_available() else "cpu")
    if req_device == "cuda" and not torch.cuda.is_available():
        req_device = "cpu"
    if req_device not in {"cpu", "cuda"}:
        req_device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(req_model_name, pretrained=req_pretrained)
    model.eval()
    model.to(req_device)

    return _ClipState(model=model, preprocess=preprocess, device=req_device, model_name=req_model_name, pretrained=req_pretrained)


@app.get("/health")
def health():
    ok = True
    details: dict[str, Any] = {}

    # CLIP: optional, but should load.
    try:
        st = _get_clip()
        details["clip"] = {"ok": True, "device": st.device, "model": st.model_name, "pretrained": st.pretrained}
    except Exception as exc:
        ok = False
        details["clip"] = {"ok": False, "error": str(exc)[:200]}

    # TransNetV2 weights presence is optional; endpoint will error if missing.
    model_dir = (os.getenv("TRANSNETV2_MODEL_DIR") or "").strip()
    details["transnetv2"] = {"model_dir": model_dir or None}

    return jsonify({"status": "ok" if ok else "degraded", "details": details}), 200


@app.post("/clip/cluster")
def clip_cluster():
    payload = request.get_json(force=True, silent=False) or {}
    images_b64 = payload.get("images_b64") or []
    k = int(payload.get("k") or 10)
    seed = int(payload.get("seed") or 0)

    # Optional per-request overrides to ensure independence across model variants.
    clip_model = payload.get("clip_model")
    clip_pretrained = payload.get("clip_pretrained")
    clip_device = payload.get("clip_device")

    if not isinstance(images_b64, list) or not images_b64:
        return jsonify({"error": "images_b64 must be a non-empty list"}), 400

    st = _get_clip(
        model_name=str(clip_model).strip() if isinstance(clip_model, str) and clip_model.strip() else None,
        pretrained=str(clip_pretrained).strip() if isinstance(clip_pretrained, str) and clip_pretrained.strip() else None,
        device=str(clip_device).strip() if isinstance(clip_device, str) and clip_device.strip() else None,
    )

    import torch  # type: ignore
    from sklearn.cluster import KMeans  # type: ignore

    pil_images = []
    for b64 in images_b64:
        if not isinstance(b64, str) or not b64.strip():
            pil_images.append(None)
            continue
        try:
            pil_images.append(_b64_to_pil(b64))
        except Exception:
            pil_images.append(None)

    # Replace missing images with a blank so we still return stable length.
    from PIL import Image

    blank = Image.new("RGB", (224, 224), color=(0, 0, 0))
    tensors = []
    for im in pil_images:
        im = im or blank
        tensors.append(st.preprocess(im))

    batch = torch.stack(tensors).to(st.device)
    with torch.no_grad():
        emb = st.model.encode_image(batch)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    emb_np = emb.detach().cpu().numpy().astype(np.float32)

    n = emb_np.shape[0]
    k = max(1, min(int(k), n))
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(emb_np)

    return jsonify({"cluster_ids": labels.tolist(), "k": k}), 200


def _ffprobe_fps_and_duration(path: str) -> tuple[float | None, float | None]:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,duration",
                "-of",
                "json",
                path,
            ],
            stderr=subprocess.STDOUT,
        )
        import json

        j = json.loads(out.decode("utf-8", errors="ignore") or "{}")
        streams = j.get("streams") or []
        if not streams:
            return (None, None)
        s0 = streams[0] or {}
        afr = str(s0.get("avg_frame_rate") or "").strip()
        dur = s0.get("duration")
        fps = None
        if "/" in afr:
            num, den = afr.split("/", 1)
            try:
                fps = float(num) / float(den)
            except Exception:
                fps = None
        elif afr:
            try:
                fps = float(afr)
            except Exception:
                fps = None

        duration = None
        try:
            duration = float(dur)
        except Exception:
            duration = None

        return (fps, duration)
    except Exception:
        return (None, None)


@app.post("/transnetv2/scenes")
def transnet_scenes():
    file_bytes: bytes | None = None
    filename = "video.mp4"

    if "video" in request.files:
        f = request.files["video"]
        filename = f.filename or filename
        file_bytes = f.read()
    else:
        # Support raw uploads from simple HTTP clients.
        # - Content-Type: application/octet-stream
        # - Optional: X-Filename header
        raw = request.get_data(cache=False)  # type: ignore[no-untyped-call]
        if raw:
            file_bytes = raw
            filename = (request.headers.get("X-Filename") or filename).strip() or filename

    if not file_bytes:
        return jsonify({"error": "provide multipart field 'video' or raw request body"}), 400

    model_dir = (os.getenv("TRANSNETV2_MODEL_DIR") or "").strip() or None

    # Heavy import only when needed
    try:
        from transnetv2 import TransNetV2  # type: ignore
    except Exception as exc:
        return jsonify({"error": f"transnetv2 import failed: {str(exc)[:200]}"}), 500

    suffix = os.path.splitext(filename)[1] or ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(file_bytes)

    try:
        fps, duration = _ffprobe_fps_and_duration(tmp_path)
        model = TransNetV2(model_dir)
        video_frames, single_pred, all_pred = model.predict_video(tmp_path)
        scenes = model.predictions_to_scenes(single_pred)

        # scenes are [start_frame, end_frame] inclusive indices
        fps_eff = float(fps) if fps and fps > 0 else None
        if fps_eff is None:
            # fall back to approximate fps from duration
            fps_eff = (len(video_frames) / float(duration)) if duration and duration > 0 else 25.0

        results = []
        for idx, (st_f, en_f) in enumerate(np.asarray(scenes).tolist()):
            st_s = float(st_f) / fps_eff
            en_s = float(en_f) / fps_eff
            if en_s < st_s:
                en_s = st_s
            results.append({"index": idx, "start": st_s, "end": en_s})

        return jsonify(
            {
                "source": "transnetv2",
                "fps": fps_eff,
                "duration_seconds": duration,
                "n_frames": int(len(video_frames)),
                "scenes": results,
            }
        ), 200
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
