from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from whisperx_adapter import transcribe
from writers import write_json, write_srt, write_txt, write_vtt

LOGGER = logging.getLogger("whisperx-batch")

MEDIA_EXTS = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".mkv", ".flac"}


def _iter_media_files(input_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in MEDIA_EXTS:
            files.append(path)
    return files


def _preprocess_file(path: Path, out_dir: Path) -> Path:
    out_path = out_dir / f"{path.stem}.wav"
    cmd = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(out_path),
        "-y",
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        err = (res.stderr or b"").decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed for {path}: {err[:240]}")
    return out_path


def run_batch(input_dir: Path, output_dir: Path, *, concurrency: int = 2) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    input_dir = input_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = _iter_media_files(input_dir)
    if not files:
        return {"count": 0, "processed": 0, "errors": []}

    preprocess_dir = Path(tempfile.mkdtemp(prefix="whisperx_pre_"))
    processed = 0
    errors: list[str] = []

    LOGGER.info("Preprocessing %d files with concurrency=%d", len(files), concurrency)
    preprocessed: dict[Path, Path] = {}

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(_preprocess_file, path, preprocess_dir): path for path in files}
        iterable = as_completed(futures)
        if tqdm is not None:
            iterable = tqdm(iterable, total=len(futures), desc="preprocess")
        for future in iterable:
            src = futures[future]
            try:
                preprocessed[src] = future.result()
                LOGGER.info("Preprocessed %s", src.name)
            except Exception as exc:
                errors.append(f"{src}: {exc}")
                LOGGER.error("Preprocess failed %s: %s", src.name, exc)

    items = list(preprocessed.items())
    if tqdm is not None:
        items = tqdm(items, total=len(items), desc="transcribe")
    for src, wav in items:
        try:
            result = transcribe(
                str(wav),
                device="cuda",
                compute_type="float16",
                model_size=os.getenv("ENVID_WHISPERX_MODEL") or "large-v2",
                batch_size=int(os.getenv("ENVID_WHISPERX_BATCH_SIZE") or 32),
                vad=True,
                chunk_seconds=int(os.getenv("ENVID_WHISPERX_CHUNK_SECONDS") or 3600),
            )
            if isinstance(result, list):
                result = result[0] if result else {}
            base = output_dir / src.stem
            write_json(result, base.with_suffix(".json"))
            write_srt(result, base.with_suffix(".srt"))
            write_vtt(result, base.with_suffix(".vtt"))
            write_txt(result, base.with_suffix(".txt"))
            processed += 1
            LOGGER.info("Completed %s", src.name)
        except Exception as exc:
            errors.append(f"{src}: {exc}")
            LOGGER.error("Transcribe failed %s: %s", src.name, exc)

    shutil.rmtree(preprocess_dir, ignore_errors=True)
    return {"count": len(files), "processed": processed, "errors": errors}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WhisperX batch directory processing")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()

    summary = run_batch(Path(args.input_dir), Path(args.output_dir), concurrency=args.concurrency)
    if summary.get("errors"):
        raise SystemExit(2)
