from __future__ import annotations


def run_batch(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("Batch transcription is not supported for OpenAI Whisper")
