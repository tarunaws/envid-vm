from __future__ import annotations

import base64
import io
import os
import subprocess
import tempfile
from dataclasses import dataclass
from math import exp
from pathlib import Path
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    if a_end <= a_start or b_end <= b_start:
        return 0.0
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, end - start)


def _count_overlapping_segments(items: Any, start_s: float, end_s: float) -> int:
    if not isinstance(items, list) or not items:
        return 0
    cnt = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        segs = it.get("segments")
        if not isinstance(segs, list):
            continue
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            st = seg.get("start_seconds")
            en = seg.get("end_seconds")
            if st is None or en is None:
                st = seg.get("start")
                en = seg.get("end")
            ss = _safe_float(st, 0.0)
            se = _safe_float(en, ss)
            if se <= ss:
                continue
            if _overlap_seconds(ss, se, start_s, end_s) > 0:
                cnt += 1
    return cnt


def _extract_keyframe_jpg(*, video_path: Path, at_seconds: float, out_path: Path, max_seconds: int = 60) -> bool:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, float(at_seconds or 0.0)):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        "scale=224:-1",
        "-q:v",
        "3",
        "-y",
        str(out_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=max_seconds)
    return res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


def _ahash64_from_jpg(jpg_path: Path) -> str | None:
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        img = Image.open(str(jpg_path)).convert("L").resize((8, 8))
        pixels = list(img.getdata())
        if not pixels:
            return None
        avg = sum(pixels) / float(len(pixels))
        bits = ["1" if p >= avg else "0" for p in pixels]
        return "".join(bits) if len(bits) == 64 else None
    except Exception:
        return None


def _hamming01(a: str | None, b: str | None) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0
    diff = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            diff += 1
    return float(diff) / float(len(a))


def _clip_cluster_images(images_b64: list[str], k: int, *, seed: int = 0) -> list[int] | None:
    if not images_b64:
        return None
    st = _get_clip()
    import torch  # type: ignore
    from sklearn.cluster import KMeans  # type: ignore
    from PIL import Image

    pil_images = []
    for b64 in images_b64:
        if not isinstance(b64, str) or not b64.strip():
            pil_images.append(None)
            continue
        try:
            pil_images.append(_b64_to_pil(b64))
        except Exception:
            pil_images.append(None)

    blank = Image.new("RGB", (224, 224), color=(0, 0, 0))
    tensors = []
    for im in pil_images:
        im = im or blank
        tensors.append(st.preprocess(im))

    batch = torch.stack(tensors).to(st.device)
    with torch.no_grad():
        emb = st.model.encode_image(batch)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    emb_np = emb.detach().cpu().numpy().astype("float32")
    n = emb_np.shape[0]
    k = max(1, min(int(k), n))
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(emb_np)
    return labels.tolist()


def _select_key_scenes_eventful(
    *,
    scenes: list[dict[str, Any]],
    scenes_source: str,
    video_intelligence: dict[str, Any],
    transcript_segments: list[dict[str, Any]],
    local_path: Path,
    temp_dir: Path,
    top_k: int = 10,
    use_clip_cluster: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not scenes:
        return ([], [])
    top_k = max(1, min(int(top_k or 10), 50))

    labels = video_intelligence.get("labels") if isinstance(video_intelligence, dict) else None
    text = video_intelligence.get("text") if isinstance(video_intelligence, dict) else None
    moderation = video_intelligence.get("moderation") if isinstance(video_intelligence, dict) else None
    explicit_frames = moderation.get("explicit_frames") if isinstance(moderation, dict) else []

    def _transcript_chars_in_window(st: float, en: float) -> int:
        if not isinstance(transcript_segments, list) or not transcript_segments:
            return 0
        total = 0
        for seg in transcript_segments:
            if not isinstance(seg, dict):
                continue
            ss = _safe_float(seg.get("start"), 0.0)
            se = _safe_float(seg.get("end"), ss)
            if se <= ss:
                continue
            if _overlap_seconds(ss, se, st, en) <= 0:
                continue
            txt = seg.get("text")
            if isinstance(txt, str) and txt.strip():
                total += len(txt.strip())
        return total

    def _norm(x: float, scale: float) -> float:
        x = max(0.0, float(x))
        s = max(1e-6, float(scale))
        return 1.0 - exp(-x / s)

    scored: list[dict[str, Any]] = []
    prev_emb: str | None = None
    keyframes_dir = temp_dir / "keyframes"

    for i, sc in enumerate(scenes):
        st = _safe_float(sc.get("start"), 0.0)
        en = _safe_float(sc.get("end"), st)
        if en <= st:
            continue
        dur = max(0.01, en - st)

        label_hits = _count_overlapping_segments(labels, st, en)
        text_hits = _count_overlapping_segments(text, st, en)

        explicit_hits = 0
        if isinstance(explicit_frames, list) and explicit_frames:
            for f in explicit_frames:
                if not isinstance(f, dict):
                    continue
                t = _safe_float(f.get("time"), 0.0)
                if st <= t <= en:
                    explicit_hits += 1

        transcript_chars = float(_transcript_chars_in_window(st, en))
        transcript_density = transcript_chars / float(dur)

        mid = st + 0.5 * dur
        jpg = keyframes_dir / f"sc_{i:04d}.jpg"
        emb = None
        if _extract_keyframe_jpg(video_path=local_path, at_seconds=mid, out_path=jpg):
            emb = _ahash64_from_jpg(jpg)

        change = 0.0
        if emb and prev_emb:
            change = _hamming01(emb, prev_emb)
        prev_emb = emb or prev_emb

        score = 0.0
        score += 2.5 * _norm(label_hits, 8.0)
        score += 3.0 * _norm(text_hits, 4.0)
        score += 4.0 * _norm(explicit_hits, 6.0)
        score += 2.0 * _norm(transcript_density, 120.0)
        score += 2.0 * _norm(change, 0.35)
        score += 0.5 * _norm(dur, 6.0)

        reasons: list[str] = []
        if explicit_hits:
            reasons.append("explicit")
        if text_hits:
            reasons.append("text")
        if label_hits:
            reasons.append("labels")
        if transcript_chars > 0:
            reasons.append("dialogue")
        if change >= 0.25:
            reasons.append("scene_change")
        if not reasons:
            reasons = ["activity"]

        scored.append({"index": i, "start": st, "end": en, "score": float(score), "reasons": reasons, "emb": emb})

    if not scored:
        return ([], [])

    scored.sort(key=lambda s: s["score"], reverse=True)
    picked: list[dict[str, Any]] = []
    min_gap_seconds = float(os.getenv("ENVID_METADATA_KEY_SCENE_MIN_GAP_SECONDS") or 8)

    cluster_by_scene_index: dict[int, int] = {}
    if use_clip_cluster:
        pool_n = min(len(scored), max(30, top_k * 6))
        pool = scored[:pool_n]
        images_b64: list[str] = []
        for s in pool:
            try:
                jpg_path = (temp_dir / "keyframes") / f"sc_{int(s['index']):04d}.jpg"
                if jpg_path.exists():
                    images_b64.append(base64.b64encode(jpg_path.read_bytes()).decode("ascii"))
                else:
                    images_b64.append("")
            except Exception:
                images_b64.append("")

        k_clusters = max(2, min(int(top_k), len(images_b64)))
        cluster_ids = _clip_cluster_images(images_b64, k_clusters)
        if cluster_ids:
            for s, cid in zip(pool, cluster_ids):
                cluster_by_scene_index[int(s["index"])] = int(cid)

    def _ok_gap(cand: dict[str, Any]) -> bool:
        if min_gap_seconds <= 0:
            return True
        for p in picked:
            if _overlap_seconds(cand["start"], cand["end"], p["start"] - min_gap_seconds, p["end"] + min_gap_seconds) > 0:
                return False
        return True

    if use_clip_cluster and cluster_by_scene_index:
        used_clusters: set[int] = set()
        remaining = list(scored)
        next_remaining: list[dict[str, Any]] = []
        for cand in remaining:
            if len(picked) >= top_k:
                next_remaining.append(cand)
                continue
            if not _ok_gap(cand):
                next_remaining.append(cand)
                continue
            cid = cluster_by_scene_index.get(int(cand["index"]), -1)
            if cid >= 0 and cid in used_clusters:
                next_remaining.append(cand)
                continue
            picked.append(cand)
            if cid >= 0:
                used_clusters.add(cid)

        for cand in next_remaining:
            if len(picked) >= top_k:
                break
            if not _ok_gap(cand):
                continue
            picked.append(cand)

        if len(picked) < top_k:
            for cand in scored:
                if len(picked) >= top_k:
                    break
                if cand in picked:
                    continue
                picked.append(cand)
    else:
        lambda_score = float(os.getenv("ENVID_METADATA_KEY_SCENE_MMR_LAMBDA") or 0.75)
        lambda_score = min(0.95, max(0.05, lambda_score))

        while scored and len(picked) < top_k:
            best_idx = None
            best_value = -1e9
            for idx, cand in enumerate(scored):
                if not _ok_gap(cand):
                    continue
                max_sim = 0.0
                if cand.get("emb") and picked:
                    for p in picked:
                        if not p.get("emb"):
                            continue
                        sim = 1.0 - _hamming01(cand.get("emb"), p.get("emb"))
                        if sim > max_sim:
                            max_sim = sim
                mmr = lambda_score * cand["score"] - (1.0 - lambda_score) * max_sim
                if mmr > best_value:
                    best_value = mmr
                    best_idx = idx

            if best_idx is None:
                picked.append(scored.pop(0))
            else:
                picked.append(scored.pop(best_idx))

    picked.sort(key=lambda s: (s["start"], s["end"]))
    key_scenes = [
        {
            "scene_index": int(s["index"]),
            "start_seconds": float(s["start"]),
            "end_seconds": float(s["end"]),
            "score": float(s["score"]),
            "reasons": s["reasons"],
            "source": scenes_source,
            **({"cluster_id": int(cluster_by_scene_index.get(int(s["index"]), -1))} if cluster_by_scene_index else {}),
        }
        for s in picked
    ]

    top_scored = sorted(picked, key=lambda s: s["score"], reverse=True)
    high_points = [
        {
            "scene_index": int(s["index"]),
            "start_seconds": float(s["start"]),
            "end_seconds": float(s["end"]),
            "score": float(s["score"]),
            "reason": ", ".join(s["reasons"]) if s.get("reasons") else "activity",
            "source": scenes_source,
            **({"cluster_id": int(cluster_by_scene_index.get(int(s["index"]), -1))} if cluster_by_scene_index else {}),
        }
        for s in top_scored[:3]
    ]
    return (key_scenes, high_points)


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


@app.post("/keyscene/select")
def keyscene_select():
    payload = request.get_json(force=True, silent=True) or {}
    scenes = payload.get("scenes") or []
    scenes_source = str(payload.get("scenes_source") or "unknown")
    video_intelligence = payload.get("video_intelligence") or {}
    transcript_segments = payload.get("transcript_segments") or []
    video_path = payload.get("video_path")
    top_k = int(payload.get("top_k") or 10)
    use_clip_cluster = bool(payload.get("use_clip_cluster"))

    if not isinstance(scenes, list) or not scenes:
        return jsonify({"key_scenes": [], "high_points": []}), 200
    if not isinstance(video_path, str) or not video_path.strip():
        return jsonify({"error": "video_path is required"}), 400

    local_path = Path(video_path)
    if not local_path.exists():
        return jsonify({"error": f"video_path not found: {video_path}"}), 400

    with tempfile.TemporaryDirectory(prefix="keyscene_") as tmpdir:
        temp_dir = Path(tmpdir)
        try:
            key_scenes, high_points = _select_key_scenes_eventful(
                scenes=scenes,
                scenes_source=scenes_source,
                video_intelligence=video_intelligence if isinstance(video_intelligence, dict) else {},
                transcript_segments=transcript_segments if isinstance(transcript_segments, list) else [],
                local_path=local_path,
                temp_dir=temp_dir,
                top_k=top_k,
                use_clip_cluster=use_clip_cluster,
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return jsonify({"key_scenes": key_scenes, "high_points": high_points}), 200
