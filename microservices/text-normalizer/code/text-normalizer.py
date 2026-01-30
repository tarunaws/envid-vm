from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
from flask import Flask, jsonify, request

try:
    from wordfreq import top_n_list, zipf_frequency  # type: ignore
except Exception:  # pragma: no cover
    top_n_list = None
    zipf_frequency = None

try:
    from rapidfuzz import process as rapid_process, fuzz as rapid_fuzz  # type: ignore
except Exception:  # pragma: no cover
    rapid_process = None
    rapid_fuzz = None

app = Flask(__name__)

_LANG_DICTIONARY_WORDS: dict[str, list[str]] = {}
_LANG_DICTIONARY_SET: dict[str, set[str]] = {}
_LANG_CONFUSION_MAP: dict[str, dict[str, str]] = {}


def _normalize_language_code(language_code: str | None) -> str:
    if not language_code:
        return ""
    code = str(language_code).strip().lower()
    if code in {"hindi", "hin"}:
        return "hi"
    if code in {"urdu", "urd"}:
        return "ur"
    if code.startswith("zh"):
        return "zh"
    if code.startswith("ja"):
        return "ja"
    if code.startswith("ko"):
        return "ko"
    for sep in ("-", "_"):
        if sep in code:
            return code.split(sep, 1)[0]
    return code


def _is_hindi_language(language_code: str | None) -> bool:
    return _normalize_language_code(language_code) == "hi"


def _dictionary_languages_enabled() -> set[str]:
    return {
        "hi",
        "bn",
        "ta",
        "te",
        "ml",
        "mr",
        "gu",
        "kn",
        "pa",
        "ur",
        "ne",
        "si",
        "th",
        "id",
        "ms",
        "vi",
        "zh",
        "ja",
        "ko",
        "ar",
    }


def _language_word_pattern(lang: str) -> str:
    if lang == "hi":
        return r"[\u0900-\u097F]+"
    if lang == "ar":
        return r"[\u0600-\u06FF]+"
    if lang in {"zh", "ja", "ko"}:
        return r"[\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]+"
    return r"[^\W\d_]+"


def _load_language_dictionary(lang: str) -> tuple[list[str], set[str]]:
    if lang in _LANG_DICTIONARY_WORDS and lang in _LANG_DICTIONARY_SET:
        return _LANG_DICTIONARY_WORDS[lang], _LANG_DICTIONARY_SET[lang]

    words: list[str] = []
    raw_paths = (os.getenv("ENVID_DICTIONARY_PATHS_JSON") or "").strip()
    path = ""
    if raw_paths:
        try:
            parsed = json.loads(raw_paths)
            if isinstance(parsed, dict):
                path = str(parsed.get(lang) or "").strip()
        except Exception:
            path = ""

    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if not w or w.startswith("#"):
                        continue
                    if lang == "hi" and not re.search(r"[\u0900-\u097F]", w):
                        continue
                    words.append(w)
        except Exception:
            words = []
    elif top_n_list is not None:
        max_words = 5000
        try:
            words = [w for w in top_n_list(lang, max_words) if w]
        except Exception:
            words = []

    _LANG_DICTIONARY_WORDS[lang] = words
    _LANG_DICTIONARY_SET[lang] = set(words)
    return words, set(words)


def _load_language_confusions(lang: str) -> dict[str, str]:
    if lang in _LANG_CONFUSION_MAP:
        return _LANG_CONFUSION_MAP[lang]
    conf: dict[str, str] = {"सा": "ऐसा"} if lang == "hi" else {}
    _LANG_CONFUSION_MAP[lang] = conf
    return conf


def _dictionary_correct_text(*, text: str, language_code: str | None) -> str:
    if not text:
        return ""
    lang = _normalize_language_code(language_code)
    if not lang or lang not in _dictionary_languages_enabled():
        return text
    if zipf_frequency is None and rapid_process is None:
        return text

    words, wordset = _load_language_dictionary(lang)
    confusions = _load_language_confusions(lang)

    min_zipf = 2.5
    fuzz_cutoff = 92
    confusions_strict = False

    def _score(w: str) -> float:
        if zipf_frequency is None:
            return 0.0
        try:
            return float(zipf_frequency(w, lang))
        except Exception:
            return 0.0

    def _fix_word(w: str) -> str:
        if not w:
            return w
        if w in confusions:
            cand = confusions[w]
            if cand:
                if not confusions_strict:
                    return cand
                if _score(cand) >= (_score(w) + 0.4):
                    return cand

        if words and rapid_process is not None and rapid_fuzz is not None:
            if w in wordset:
                return w
            try:
                match = rapid_process.extractOne(w, words, scorer=rapid_fuzz.QRatio, score_cutoff=fuzz_cutoff)
            except Exception:
                match = None
            if match:
                cand = str(match[0])
                if cand and _score(cand) >= max(min_zipf, _score(w)):
                    return cand
        return w

    pattern = _language_word_pattern(lang)

    def _repl(m: re.Match) -> str:
        return _fix_word(m.group(0))

    return re.sub(pattern, _repl, text)


def _normalize_transcript_basic(text: str) -> str:
    raw = (text or "").strip()
    raw = re.sub(r"\s+", " ", raw)
    return raw


def _languagetool_remote_check(*, text: str, language: str) -> dict[str, Any] | None:
    url = "http://translate:8010"
    if url.endswith("/"):
        url = url[:-1]
    if not url.endswith("/v2/check") and not url.endswith("/check"):
        url = f"{url}/v2/check"

    lang = (language or "auto").strip() or "auto"
    timeout_s = float(os.getenv("ENVID_GRAMMAR_CORRECTION_TIMEOUT_SECONDS") or 10.0)
    try:
        resp = requests.post(url, data={"text": text, "language": lang}, timeout=timeout_s)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _apply_languagetool_matches(text: str, matches: list[dict[str, Any]]) -> str:
    if not matches:
        return text
    out = text
    ordered = sorted(matches, key=lambda m: int(m.get("offset", 0) or 0), reverse=True)
    for m in ordered:
        try:
            offset = int(m.get("offset", 0) or 0)
            length = int(m.get("length", 0) or 0)
            repls = m.get("replacements") or []
            replacement = ""
            if repls:
                rep0 = repls[0]
                if isinstance(rep0, dict):
                    replacement = str(rep0.get("value") or "")
                else:
                    replacement = str(rep0 or "")
            if length <= 0:
                continue
            out = out[:offset] + replacement + out[offset + length :]
        except Exception:
            continue
    return out


def _grammar_correct_text(*, text: str, language: str | None) -> tuple[str, bool]:
    if not text:
        return "", False
    lang = (language or "auto").strip() or "auto"
    data = _languagetool_remote_check(text=text, language=lang)
    if not isinstance(data, dict):
        return text, False
    matches = data.get("matches")
    if not isinstance(matches, list):
        return text, False
    corrected = _apply_languagetool_matches(text, matches)
    return corrected, corrected != text


def _enhance_transcript_punctuation(text: str, language_code: str | None) -> str:
    out = _normalize_transcript_basic(text)
    if not out:
        return ""
    tail = out[-1]
    if tail not in ".!?।":
        out = f"{out}{'।' if _is_hindi_language(language_code) else '.'}"
    return out


def _gemini_api_key() -> str | None:
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    return api_key or None


def _gemini_base_url() -> str:
    return (os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta").strip().rstrip("/")


def _gemini_model() -> str:
    model = (
        os.getenv("GEMINI_MODEL")
        or os.getenv("GEMINI_DEFAULT_MODEL")
        or "gemini-2.0-flash"
    ).strip()
    if model.startswith("models/"):
        return model
    return f"models/{model}"


def _gemini_generate_text(*, prompt: str, system: str | None, timeout_s: float, max_tokens: int) -> str | None:
    api_key = _gemini_api_key()
    if not api_key:
        return None
    base_url = _gemini_base_url()
    model = _gemini_model()
    url = f"{base_url}/{model}:generateContent?key={api_key}"
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": int(max_tokens),
            "responseMimeType": "application/json",
        },
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
        if resp.status_code >= 400:
            return None
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return None
        parts = ((candidates[0].get("content") or {}).get("parts") or [])
        texts = [str(p.get("text") or "").strip() for p in parts if isinstance(p, dict)]
        content = "\n".join([t for t in texts if t]).strip()
        return content or None
    except Exception:
        return None


def _gemini_normalize_transcript(text: str, language_code: str | None) -> tuple[str | None, bool]:
    if not _gemini_api_key():
        return None, False

    raw = _normalize_transcript_basic(text)
    if not raw:
        return None, False

    timeout_s = float(os.getenv("GEMINI_TIMEOUT_SECONDS") or os.getenv("ENVID_LLM_TIMEOUT_SECONDS") or 15.0)
    lang = (language_code or "").strip() or "unknown"
    prompt = (
        "You are a transcript normalizer.\n"
        "Task: improve readability ONLY by fixing punctuation, casing, spacing, and obvious sentence boundaries.\n"
        "Hard rules:\n"
        "- Do NOT add new words or remove words.\n"
        "- Do NOT guess missing words.\n"
        "- Do NOT rewrite, paraphrase, or summarize.\n"
        "- Keep the language as-is.\n"
        "- For Hindi, prefer the danda (।) for sentence endings.\n"
        "- Output plain text only (no markdown, no quotes).\n\n"
        f"Language hint: {lang}\n\n"
        f"Transcript:\n{raw[:12000]}\n"
    )

    content = _gemini_generate_text(
        prompt=prompt,
        system="Fix punctuation/casing/spacing only. Output plain text.",
        timeout_s=timeout_s,
        max_tokens=1200,
    )
    return (content if content else None), bool(content)


def _verify_prompt(text: str, language_code: str | None) -> str:
    lang = (language_code or "").strip() or "unknown"
    return (
        "Provide only the cleaned transcript text.\n"
        "Do not summarize. Do not add new content.\n"
        "Keep the same language and original meaning.\n\n"
        f"Language hint: {lang}\n\n"
        f"Transcript:\n{text[:12000]}\n"
    )


def _gemini_verify_transcript(text: str, language_code: str | None) -> dict[str, Any] | None:
    if not _gemini_api_key():
        return None
    raw = _normalize_transcript_basic(text)
    if not raw:
        return None
    timeout_s = float(os.getenv("GEMINI_TIMEOUT_SECONDS") or os.getenv("ENVID_LLM_TIMEOUT_SECONDS") or 15.0)
    prompt = _verify_prompt(raw, language_code)
    max_tokens = int(float(
        os.getenv("ENVID_GEMINI_MAX_OUTPUT_TOKENS")
        or os.getenv("ENVID_LLM_MAX_OUTPUT_TOKENS")
        or 2048
    ))
    max_tokens = max(256, min(8192, max_tokens))
    content = _gemini_generate_text(
        prompt=prompt,
        system=(
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
        ),
        timeout_s=timeout_s,
        max_tokens=max_tokens,
    )
    if not content:
        return None
    return {
        "text": content.strip(),
        "provider": "gemini",
        "model": _gemini_model(),
    }


def _local_llm_base_url() -> str:
    return (os.getenv("ENVID_LLM_BASE_URL") or "http://localhost:8000/v1").strip().rstrip("/")


def _local_llm_headers() -> dict[str, str]:
    api_key = (os.getenv("ENVID_LLM_API_KEY") or "").strip()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _local_llm_model() -> str:
    return (
        os.getenv("ENVID_LLM_TRANSCRIPT_MODEL")
        or os.getenv("ENVID_TEXT_NORMALIZE_MODEL")
        or os.getenv("ENVID_LLM_MODEL")
        or "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ).strip()


def _local_llm_normalize_transcript(text: str, language_code: str | None) -> tuple[str | None, bool]:
    raw = _normalize_transcript_basic(text)
    if not raw:
        return None, False

    base_url = _local_llm_base_url()
    if not base_url:
        return None, False

    model = _local_llm_model()
    timeout_s = float(os.getenv("ENVID_LLM_TIMEOUT_SECONDS") or 15.0)

    lang = (language_code or "").strip() or "unknown"
    prompt = (
        "You are a transcript normalizer.\n"
        "Task: improve readability ONLY by fixing punctuation, casing, spacing, and obvious sentence boundaries.\n"
        "Hard rules:\n"
        "- Do NOT add new words or remove words.\n"
        "- Do NOT guess missing words.\n"
        "- Do NOT rewrite, paraphrase, or summarize.\n"
        "- Keep the language as-is.\n"
        "- For Hindi, prefer the danda (।) for sentence endings.\n"
        "- Output plain text only (no markdown, no quotes).\n\n"
        f"Language hint: {lang}\n\n"
        f"Transcript:\n{raw[:12000]}\n"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Fix punctuation/casing/spacing only. Output plain text."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }

    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=_local_llm_headers(),
            json=payload,
            timeout=timeout_s,
        )
        if resp.status_code >= 400:
            return None, False
        data = resp.json()
        content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        return (content if content else None), bool(content)
    except Exception:
        return None, False


def _local_llm_verify_transcript(text: str, language_code: str | None) -> dict[str, Any] | None:
    raw = _normalize_transcript_basic(text)
    if not raw:
        return None
    base_url = _local_llm_base_url()
    if not base_url:
        return None
    model = _local_llm_model()
    timeout_s = float(os.getenv("ENVID_LLM_TIMEOUT_SECONDS") or 15.0)
    prompt = _verify_prompt(raw, language_code)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
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
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 600,
    }
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=_local_llm_headers(),
            json=payload,
            timeout=timeout_s,
        )
        if resp.status_code >= 400:
            return None
        data = resp.json()
        content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        if content:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    if isinstance(parsed.get("text"), str):
                        content = parsed["text"].strip()
                    elif isinstance(parsed.get("parameters"), dict) and isinstance(parsed["parameters"].get("text"), str):
                        content = parsed["parameters"]["text"].strip()
            except Exception:
                pass
        if not content:
            return None
        return {
            "text": content.strip(),
            "provider": "local",
            "model": model,
        }
    except Exception:
        return None


@app.get("/health")
def health() -> Any:
    return jsonify({"ok": True}), 200


@app.post("/normalize/segment")
def normalize_segment() -> Any:
    payload = request.get_json(force=True, silent=True) or {}
    text = str(payload.get("text") or "")
    language_code = payload.get("language_code")
    grammar_enabled = bool(payload.get("grammar_enabled"))
    dictionary_enabled = bool(payload.get("dictionary_enabled"))
    punctuation_enabled = bool(payload.get("punctuation_enabled"))
    nlp_mode = str(payload.get("nlp_mode") or "").strip().lower()

    out = _normalize_transcript_basic(text)
    meta = {
        "nlp_applied": False,
        "grammar_applied": False,
        "dictionary_applied": False,
        "hindi_applied": False,
        "punctuation_applied": False,
    }

    # LLM normalization disabled; keep normalization to non-LLM steps only.

    if grammar_enabled:
        corrected, applied = _grammar_correct_text(text=out, language=language_code)
        if applied and corrected:
            out = corrected
            meta["grammar_applied"] = True

    if dictionary_enabled:
        corrected = _dictionary_correct_text(text=out, language_code=language_code)
        if corrected and corrected != out:
            out = corrected
            meta["dictionary_applied"] = True
            if _is_hindi_language(language_code):
                meta["hindi_applied"] = True

    if punctuation_enabled:
        corrected = _enhance_transcript_punctuation(out, language_code)
        if corrected and corrected != out:
            out = corrected
            meta["punctuation_applied"] = True

    return jsonify({"text": out, "meta": meta}), 200


@app.post("/verify/segment")
def verify_segment() -> Any:
    payload = request.get_json(force=True, silent=True) or {}
    text = str(payload.get("text") or "")
    language_code = payload.get("language_code")
    nlp_mode = str(payload.get("nlp_mode") or "").strip().lower()

    result: dict[str, Any] | None = None
    if nlp_mode in {"gemini", "gemini_primary", "local_llama", "local"}:
        result = _gemini_verify_transcript(text, language_code)
        if not result:
            result = _local_llm_verify_transcript(text, language_code)

    if not isinstance(result, dict):
        return jsonify({"text": None, "meta": {"error": "unavailable"}}), 200

    text_out = str(result.get("text") or "").strip()
    meta = {
        "provider": result.get("provider"),
        "model": result.get("model"),
    }
    return jsonify({"text": text_out, "meta": meta}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5098")))
