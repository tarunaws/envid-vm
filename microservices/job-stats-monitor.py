#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_env(env_path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not env_path.exists():
        return env
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            env[key] = value
    return env


def _parse_size_to_bytes(value: str) -> int:
    if not value:
        return 0
    raw = value.strip()
    match = re.match(r"^([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)$", raw)
    if not match:
        return 0
    num = float(match.group(1))
    unit = match.group(2).lower()
    scale = 1
    if unit in {"b", "bytes"}:
        scale = 1
    elif unit in {"kb", "kib"}:
        scale = 1024
    elif unit in {"mb", "mib"}:
        scale = 1024 ** 2
    elif unit in {"gb", "gib"}:
        scale = 1024 ** 3
    elif unit in {"tb", "tib"}:
        scale = 1024 ** 4
    return int(num * scale)


def _run_cmd(cmd: List[str]) -> Tuple[int, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = (result.stdout or "").strip()
        return result.returncode, output
    except Exception:
        return 1, ""


def _docker_stats(containers: List[str]) -> Dict[str, Dict[str, Any]]:
    if not containers:
        return {}
    cmd = [
        "docker",
        "stats",
        "--no-stream",
        "--format",
        "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}",
    ] + containers
    code, output = _run_cmd(cmd)
    if code != 0 or not output:
        return {}
    stats: Dict[str, Dict[str, Any]] = {}
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        name, cpu_pct, mem_usage, mem_pct = (p.strip() for p in parts)
        mem_used_str, mem_limit_str = "", ""
        if " / " in mem_usage:
            mem_used_str, mem_limit_str = (seg.strip() for seg in mem_usage.split(" / ", 1))
        stats[name] = {
            "cpu_pct": cpu_pct,
            "mem_usage": mem_usage,
            "mem_pct": mem_pct,
            "mem_used_bytes": _parse_size_to_bytes(mem_used_str),
            "mem_limit_bytes": _parse_size_to_bytes(mem_limit_str),
        }
    return stats


def _existing_containers(wanted: List[str]) -> List[str]:
    code, output = _run_cmd(["docker", "ps", "--format", "{{.Names}}"])
    if code != 0 or not output:
        return [c for c in wanted if c]
    running = {line.strip() for line in output.splitlines() if line.strip()}
    return [c for c in wanted if c in running]


def _gpu_stats() -> List[Dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    code, output = _run_cmd(cmd)
    if code != 0 or not output:
        return []
    gpus: List[Dict[str, Any]] = []
    for line in output.splitlines():
        parts = [seg.strip() for seg in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "used_mb": int(float(parts[1])),
                    "total_mb": int(float(parts[2])),
                }
            )
        except Exception:
            continue
    return gpus


def _fetch_json(url: str) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _step_map(job: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    steps = job.get("steps") or []
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict) and step.get("id"):
                out[str(step.get("id"))] = step
    return out


def _current_step(job: Dict[str, Any]) -> str:
    step = str(job.get("current_step") or "").strip()
    if step:
        return step
    steps = job.get("steps") or []
    if isinstance(steps, list):
        for s in steps:
            if isinstance(s, dict) and str(s.get("status") or "") == "running":
                return str(s.get("id") or "running")
    return str(job.get("status") or "unknown")


def _guess_model(step_id: str, step_message: str | None, env: Dict[str, str]) -> str | None:
    msg = (step_message or "").strip()
    if msg:
        matches = re.findall(r"\(([^()]+)\)", msg)
        if matches:
            candidate = matches[-1].strip()
            if candidate:
                return candidate
    if step_id == "translate_output":
        en_indic = env.get("INDIC_TRANS_MODEL_EN_INDIC", "ai4bharat/indictrans2-en-indic-1B")
        indic_en = env.get("INDIC_TRANS_MODEL_INDIC_EN", "ai4bharat/indictrans2-indic-en-1B")
        indic_indic = env.get("INDIC_TRANS_MODEL_INDIC_INDIC", "ai4bharat/indictrans2-indic-indic-1B")
        return f"hybrid (indictrans2: {en_indic}, {indic_en}, {indic_indic}; libretranslate)"
    if step_id in {"synopsis_generation", "keyword_extraction", "introduction", "scene_by_scene_metadata"}:
        return env.get("ENVID_GENAI_MODEL") or env.get("ENVID_LLM_MODEL") or "unknown"
    if step_id == "key_scene_detection":
        return "scene-detect"
    return None


def _resolve_storage_base(env: Dict[str, str]) -> Path:
    base = (env.get("ENVID_STORAGE_BASE_DIR") or env.get("ENVID_GCS_MOUNT_PATH") or "").strip()
    if base and Path(base).exists():
        return Path(base)
    if base == "/tmp/envid-metadata-local" and Path("/home/tarun-envid/envid-local").exists():
        return Path("/home/tarun-envid/envid-local")
    fallback = Path("/home/tarun-envid/envid-local")
    return fallback if fallback.exists() else Path(base or "/tmp")


def _write_json_via_docker(container_path: str, payload: Dict[str, Any]) -> None:
    dir_path = str(Path(container_path).parent)
    cmd = [
        "docker",
        "exec",
        "-i",
        "backend",
        "sh",
        "-lc",
        f"mkdir -p '{dir_path}' && cat > '{container_path}'",
    ]
    try:
        subprocess.run(cmd, input=json.dumps(payload, indent=2, ensure_ascii=False), text=True, check=False)
    except Exception:
        pass


def _write_json(path: Path, payload: Dict[str, Any], *, container_path: str | None = None) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except PermissionError:
        if container_path:
            _write_json_via_docker(container_path, payload)


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: job-stats-monitor.py <job_id> [backend_url]", file=sys.stderr)
        return 2

    job_id = sys.argv[1]
    backend_url = sys.argv[2] if len(sys.argv) > 2 else os.getenv("BACKEND_URL", "http://localhost/backend")
    poll_seconds = int(os.getenv("JOB_STATS_POLL_SECONDS", "5"))

    env = _load_env(Path(__file__).resolve().parent / ".env")
    storage_base = _resolve_storage_base(env)
    job_root = storage_base / "artifacts-local" / job_id
    stats_path = job_root / "job_stats.json"
    container_storage_base = (env.get("ENVID_STORAGE_BASE_DIR") or "/tmp/envid-metadata-local").strip()
    stats_path_container = f"{container_storage_base.rstrip('/')}/artifacts-local/{job_id}/job_stats.json"

    containers = [
        "translate-indian",
        "translate-international",
        "audio-transcription",
        "genai",
        "backend",
        "text-on-video",
        "moderation",
        "scene-detect",
        "keyscene",
        "frame-extractor",
        "transcoder",
    ]
    containers = _existing_containers(containers)

    job_stats: Dict[str, Any] = {
        "job_id": job_id,
        "backend_url": backend_url,
        "started_at": _now_iso(),
        "completed_at": None,
        "status": "running",
        "steps": {},
    }

    while True:
        try:
            job = _fetch_json(f"{backend_url.rstrip('/')}/jobs/{job_id}")
        except Exception:
            job = {}

        step_id = _current_step(job)
        step_map = _step_map(job)
        step_info = step_map.get(step_id, {}) if isinstance(step_map, dict) else {}
        step_message = step_info.get("message") if isinstance(step_info, dict) else None
        step_status = step_info.get("status") if isinstance(step_info, dict) else None

        model = _guess_model(step_id, step_message, env)

        resources = {
            "timestamp": _now_iso(),
            "containers": _docker_stats(containers),
            "gpu": _gpu_stats(),
        }

        steps = job_stats.setdefault("steps", {})
        step_entry = steps.setdefault(
            step_id,
            {
                "id": step_id,
                "label": step_info.get("label") or step_id,
                "status": step_status or "unknown",
                "message": step_message,
                "model": model,
                "started_at": step_info.get("started_at"),
                "completed_at": step_info.get("completed_at"),
                "samples": [],
                "max": {"containers": {}, "gpu": {}},
            },
        )

        step_entry["status"] = step_status or step_entry.get("status")
        step_entry["message"] = step_message or step_entry.get("message")
        step_entry["model"] = model or step_entry.get("model")
        step_entry["started_at"] = step_info.get("started_at") or step_entry.get("started_at")
        step_entry["completed_at"] = step_info.get("completed_at") or step_entry.get("completed_at")
        step_entry["samples"].append(resources)

        max_containers: Dict[str, Any] = step_entry.get("max", {}).get("containers", {})
        for name, data in resources.get("containers", {}).items():
            prev = max_containers.get(name, {})
            used = int(data.get("mem_used_bytes") or 0)
            if used > int(prev.get("mem_used_bytes") or 0):
                max_containers[name] = data
        max_gpu: Dict[str, Any] = step_entry.get("max", {}).get("gpu", {})
        for gpu in resources.get("gpu", []):
            idx = str(gpu.get("index"))
            prev = max_gpu.get(idx, {})
            used = int(gpu.get("used_mb") or 0)
            if used > int(prev.get("used_mb") or 0):
                max_gpu[idx] = gpu

        step_entry["max"]["containers"] = max_containers
        step_entry["max"]["gpu"] = max_gpu

        job_status = str(job.get("status") or job_stats.get("status") or "running")
        job_stats["status"] = job_status
        job_stats["current_step"] = step_id
        if job.get("progress") is not None:
            job_stats["progress"] = job.get("progress")
        if job.get("message"):
            job_stats["message"] = job.get("message")

        _write_json(stats_path, job_stats, container_path=stats_path_container)

        if job_status in {"completed", "failed"}:
            job_stats["completed_at"] = _now_iso()
            _write_json(stats_path, job_stats, container_path=stats_path_container)
            break

        time.sleep(max(1, poll_seconds))

    print(json.dumps({"job_id": job_id, "stats_path": str(stats_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
