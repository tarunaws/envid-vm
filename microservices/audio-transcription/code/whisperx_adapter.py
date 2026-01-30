from __future__ import annotations

from typing import Any, Iterable

from openai_whisper_adapter import transcribe as _openai_transcribe


def transcribe(*args: Any, **kwargs: Any) -> dict[str, Any]:
    if "input_path" in kwargs:
        return _openai_transcribe(**kwargs)
    if not args:
        raise ValueError("input_path is required")
    return _openai_transcribe(input_path=str(args[0]), **kwargs)


def transcribe_stream(*args: Any, **kwargs: Any) -> Iterable[dict[str, Any]]:
    raise RuntimeError("Streaming transcription is not supported for OpenAI Whisper")
