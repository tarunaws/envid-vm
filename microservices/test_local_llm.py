#!/usr/bin/env python3
import argparse
import os
import sys
import json
import requests


VERIFY_SYSTEM_PROMPT = (
    "You are a professional linguistic editor specializing in ASR (Automatic Speech Recognition) "
    "post-processing. Your goal is to clean and validate Whisper transcriptions for high-quality "
    "subtitle generation and summarization.\n\n"
    "Your specific tasks:\n"
    "Deduplication: Identify and remove 'repetition loops' (e.g., when Whisper repeats the same "
    "sentence or phrase multiple times due to background noise).\n"
    "Hallucination Cleaning: Remove nonsensical phrases often generated during silent periods "
    "(e.g., 'Thank you for watching,' or 'Please subscribe' when it doesn't fit the context).\n"
    "Grammar & Punctuation: Correct obvious transcription errors, fix sentence casing, and add proper "
    "punctuation to ensure readability.\n"
    "Logical Flow: If a word sounds phonetically similar but makes no sense in context, use the "
    "surrounding narrative to correct it.\n"
    "Preservation: Do not summarize at this stage. Keep the speaker's original meaning and vocabulary "
    "intact. Provide only the cleaned text."
)

SYNOPSIS_SYSTEM_PROMPT = (
    "You are an elite Hollywood Story Architect and Marketing Specialist. Your goal is to transform provided plot points, "
    "dialogue, or transcripts into professional movie synopses for any genre.\n\n"
    "Your writing must:\n"
    "- Identify and enhance the 'Core Conflict' and 'Protagonist Stakes'.\n"
    "- Adapt the tone (dark, uplifting, witty, or epic) to match the source material.\n"
    "- Use active, evocative language.\n"
    "- Avoid spoilers unless instructed otherwise.\n"
    "- Strictly adhere to word count constraints while maintaining a punchy, cinematic flow."
)


def build_verify_prompt(text: str, language: str) -> str:
    return (
        "Provide only the cleaned transcript text.\n"
        "Do not summarize. Do not add new content.\n"
        "Keep the same language and original meaning.\n\n"
        f"Language hint: {language}\n\n"
        f"Transcript:\n{text[:12000]}\n"
    )


def build_synopsis_prompt(text: str, language: str) -> str:
    return (
        "Return JSON only with keys \"short\" and \"long\".\n"
        "Short Synopsis (max 50 words): A high-concept logline/teaser that hooks immediately.\n"
        "Long Synopsis (100–150 words): A detailed narrative summary outlining setting, rising tension, and ultimate stakes.\n"
        "Use only information explicitly stated in the transcript; do NOT add new details.\n"
        "Paraphrase; do NOT copy phrases longer than 6 words from the transcript.\n"
        "Respond ONLY in the same language as the transcript. Do NOT translate.\n"
        "Preserve proper nouns.\n"
        f"Language hint: {language}\n\n"
        f"Transcript:\n{text[:12000]}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct local LLM test (OpenAI-compatible)")
    parser.add_argument("--mode", choices=["verify", "synopsis", "raw"], default="verify")
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--model", default=os.getenv("ENVID_LLM_MODEL", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"))
    parser.add_argument("--base-url", default=os.getenv("ENVID_LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/"))
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("prompt", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    raw_prompt = " ".join(args.prompt).strip()
    if not raw_prompt:
        raw_prompt = (
            "ऐसा लग रहा है ना, जैसे शरीर जंजीरों से जगड़ा हुआ है मुझे तो मार दिया, सोनब का क्या करोगे? "
            "तुमारी आत्मा सदियों से पेड़े था, मैं तुम्हें साहता देना चाहता हूँ"
        )

    if args.mode == "verify":
        system_prompt = VERIFY_SYSTEM_PROMPT
        user_prompt = build_verify_prompt(raw_prompt, args.lang)
    elif args.mode == "synopsis":
        system_prompt = SYNOPSIS_SYSTEM_PROMPT
        user_prompt = build_synopsis_prompt(raw_prompt, args.lang)
    else:
        system_prompt = "You are a helpful assistant."
        user_prompt = raw_prompt

    url = f"{args.base_url}/chat/completions"
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
    except Exception as exc:
        print(f"Request failed: {exc}")
        return 1

    data = resp.json()
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
