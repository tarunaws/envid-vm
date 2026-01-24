from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _format_timestamp(seconds: float, *, decimal: str = ".") -> str:
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000.0))
    hours = millis // 3_600_000
    millis -= hours * 3_600_000
    minutes = millis // 60_000
    millis -= minutes * 60_000
    secs = millis // 1000
    millis -= secs * 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{millis:03d}"


def _iter_segments(result: dict[str, Any]) -> Iterable[dict[str, Any]]:
    return result.get("segments", []) or []


def write_json(result: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def write_txt(result: dict[str, Any], path: str | Path) -> None:
    text_parts: list[str] = []
    for seg in _iter_segments(result):
        text = str(seg.get("text") or "").strip()
        if text:
            text_parts.append(text)
    Path(path).write_text("\n".join(text_parts).strip() + "\n", encoding="utf-8")


def write_srt(result: dict[str, Any], path: str | Path) -> None:
    lines: list[str] = []
    index = 1
    for seg in _iter_segments(result):
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        start = _format_timestamp(float(seg.get("start") or 0.0), decimal=",")
        end = _format_timestamp(float(seg.get("end") or 0.0), decimal=",")
        lines.extend([str(index), f"{start} --> {end}", text, ""])
        index += 1
    Path(path).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_vtt(result: dict[str, Any], path: str | Path) -> None:
    lines: list[str] = ["WEBVTT", ""]
    for seg in _iter_segments(result):
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        start = _format_timestamp(float(seg.get("start") or 0.0), decimal=".")
        end = _format_timestamp(float(seg.get("end") or 0.0), decimal=".")
        lines.extend([f"{start} --> {end}", text, ""])
    Path(path).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
