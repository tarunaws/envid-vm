from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

from whisperx_adapter import transcribe
from writers import write_json, write_srt, write_txt, write_vtt


def _expand_inputs(inputs: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for item in inputs:
        if any(ch in item for ch in ["*", "?", "["]):
            expanded.extend([str(p) for p in Path().glob(item)])
        else:
            expanded.append(item)
    return expanded


def _output_base(output_dir: Path, input_item: str, index: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    name = Path(input_item).name
    if not name:
        name = f"input_{index}"
    stem = Path(name).stem
    return output_dir / stem


def main() -> int:
    parser = argparse.ArgumentParser(description="WhisperX transcription CLI")
    parser.add_argument("--input", action="append", required=True, help="Input file/URL (repeatable)")
    parser.add_argument("--output-dir", default="outputs", help="Directory to write outputs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--language", default=None, help="Language code (auto-detect if omitted)")
    parser.add_argument("--diarize", action="store_true", help="Enable diarization")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token (or HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--model-size", default="large-v2", help="Whisper model size")
    parser.add_argument("--compute-type", default=None, help="Compute type (float16/float32/int8)")
    parser.add_argument("--min-speech-dur", type=float, default=None, help="Min speech duration (seconds) for VAD")
    parser.add_argument("--vad", action="store_true", help="Enable VAD filtering")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    inputs = _expand_inputs(args.input)
    output_dir = Path(args.output_dir)

    for idx, item in enumerate(inputs, start=1):
        result = transcribe(
            item,
            batch_size=args.batch_size,
            language=args.language,
            diarize=args.diarize,
            hf_token=args.hf_token,
            model_size=args.model_size,
            compute_type=args.compute_type,
            min_speech_dur=args.min_speech_dur,
            vad=args.vad,
        )
        base = _output_base(output_dir, item, idx)
        write_json(result, base.with_suffix(".json"))
        write_srt(result, base.with_suffix(".srt"))
        write_vtt(result, base.with_suffix(".vtt"))
        write_txt(result, base.with_suffix(".txt"))

    logging.info("Done. Wrote outputs to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
