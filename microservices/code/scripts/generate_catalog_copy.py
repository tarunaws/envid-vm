#!/usr/bin/env python3
"""Generate Meta Llama copy for each catalog product.

This script loops through the Interactive & Shoppable catalog and prompts a
local OpenAI-compatible LLM for richer product storytelling, then stores the
structured response under a configurable field (default: `aiCopy`).

Environment:
    ENVID_LLM_BASE_URL       -> optional (default: http://localhost:8000/v1)
    ENVID_LLM_API_KEY        -> optional
    CATALOG_COPY_MODEL       -> optional (defaults to ENVID_LLM_MODEL)
    ENVID_LLM_MODEL          -> optional (default: meta-llama/llama-3.3-70b-instruct)
    CATALOG_COPY_TEMPERATURE -> optional float for creativity
    CATALOG_COPY_TOP_P       -> optional float for nucleus sampling
    CATALOG_COPY_MAX_TOKENS  -> optional int

Usage examples:
    python scripts/generate_catalog_copy.py
    python scripts/generate_catalog_copy.py --field aiCopy --force
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import requests

DEFAULT_MODEL_ID = (
    os.getenv("CATALOG_COPY_MODEL")
    or os.getenv("ENVID_LLM_MODEL")
    or "meta-llama/llama-3.3-70b-instruct"
)
DEFAULT_TEMPERATURE = float(os.getenv("CATALOG_COPY_TEMPERATURE", "0.45"))
DEFAULT_TOP_P = float(os.getenv("CATALOG_COPY_TOP_P", "0.9"))
DEFAULT_MAX_TOKENS = int(os.getenv("CATALOG_COPY_MAX_TOKENS", "600"))

PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are a GenAI brand copy strategist. For each product, craft fresh content that helps shoppers
    understand why it matters inside a shoppable video experience. Keep the tone premium yet inviting
    and avoid repeating placeholder text.

    Return strict JSON with these fields:
        {{
            "tagline": "short hero line (max 12 words)",
            "elevatorPitch": "two-sentence micro story (max 70 words)",
            "bulletPoints": ["benefit", "material or tech detail", "use case"],
            "socialCaption": "One-sentence caption ending with 1-2 tasteful emojis",
            "keywords": ["comma-free", "searchable", "phrases"]
        }}

    Only output JSON. No markdown, no commentary, no extra keys.

    Product context:
    - Name: {name}
    - Label: {label}
    - CTA: {cta}
    - Price: {price}
    - Description: {description}
    - Scene timing (ms): start={start_ms}, heroMoment={timestamp_ms}, end={end_ms}
    """
).strip()


def _extract_text(response_body: Dict[str, Any]) -> str:
    """Best-effort helper to unwrap chat-completions responses."""

    def _from_content(value: Any) -> str:
        if isinstance(value, list):
            return "".join(_from_content(item) for item in value)
        if isinstance(value, dict):
            candidates: List[str] = []
            if value.get("text"):
                candidates.append(str(value["text"]))
            if value.get("content"):
                candidates.append(_from_content(value["content"]))
            if value.get("result"):
                candidates.append(str(value["result"]))
            return "".join(candidates)
        return str(value)

    if not response_body:
        return ""

    for key in ("output", "generation", "result", "message"):
        if key in response_body and isinstance(response_body[key], str):
            return response_body[key]

    if "outputs" in response_body:
        return "".join(_from_content(item) for item in response_body["outputs"])

    if "content" in response_body:
        return _from_content(response_body["content"])

    if "messages" in response_body:
        for message in response_body["messages"]:
            if isinstance(message, dict) and message.get("role") == "assistant":
                return _from_content(message.get("content"))

    return ""


def _coerce_json(payload: str) -> Dict[str, Any]:
    candidate = payload.strip()
    if not candidate:
        return {}
    # Some providers occasionally wrap JSON in code fences. Strip them.
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate.strip("`")
    if candidate.startswith("json\n"):
        candidate = candidate.split("\n", 1)[1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Attempt to find the first/last braces as a fallback.
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = candidate[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return {}
        return {}


def _build_prompt(entry: Dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        name=entry.get("name", "Unnamed Product"),
        label=entry.get("label", ""),
        cta=entry.get("ctaText", ""),
        price=entry.get("price", ""),
        description=entry.get("description", ""),
        start_ms=entry.get("startMs", 0),
        timestamp_ms=entry.get("timestampMs", entry.get("startMs", 0)),
        end_ms=entry.get("endMs", 0),
    )


def _build_fallback_prompt(entry: Dict[str, Any]) -> str:
    return textwrap.dedent(
        """
        Return JSON with keys "tagline", "elevatorPitch", "bulletPoints" (3 items),
        "socialCaption", and "keywords" (4 items) for the product below. Only output JSON.

        Product name: {name}
        Description: {description}
        Label: {label}
        CTA: {cta}
        """
    ).strip().format(
        name=entry.get("name", "Unnamed Product"),
        description=entry.get("description", ""),
        label=entry.get("label", ""),
        cta=entry.get("ctaText", ""),
    )


def _invoke_llm_chat(*, prompt: str, temperature: float, top_p: float, max_tokens: int) -> Dict[str, Any]:
    api_key = (os.getenv("ENVID_LLM_API_KEY") or "").strip()
    base_url = (os.getenv("ENVID_LLM_BASE_URL") or "http://localhost:8000/v1").strip().rstrip("/")
    model = DEFAULT_MODEL_ID

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return ONLY strict JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }

    resp = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(payload), timeout=30)
    resp.raise_for_status()
    return resp.json()


def enrich_catalog(
    catalog_path: Path,
    *,
    output_path: Path,
    field: str,
    force: bool,
    temperature: float,
    top_p: float,
    max_tokens: int,
    connect_timeout: int,
    read_timeout: int,
) -> int:
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    data = json.loads(catalog_path.read_text())
    if not isinstance(data, list):
        raise ValueError("Catalog JSON must be a list")

    # Keep existing timeout args for CLI compatibility.
    _ = (connect_timeout, read_timeout)

    updated = 0
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if field in entry and not force:
            continue

        structured: Dict[str, Any] | None = None
        for prompt_builder in (_build_prompt, _build_fallback_prompt):
            prompt = prompt_builder(entry)
            try:
                response_payload = _invoke_llm_chat(
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                print(f"[WARN] LLM request failed for {entry.get('id')}: {exc}", file=sys.stderr)
                response_payload = None

            assistant_text = _extract_text(response_payload or {})
            structured = _coerce_json(assistant_text)
            if structured:
                break

        if structured:
            entry[field] = structured
            updated += 1
            print(f"[INFO] Updated {entry.get('id')} -> {field}")
        else:
            print(
                f"[WARN] Unable to parse Llama response for {entry.get('id')}. Raw snippet: {assistant_text[:120]}",
                file=sys.stderr,
            )

    output_path.write_text(json.dumps(data, indent=2))
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Meta Llama copy for catalog products.")
    parser.add_argument(
        "--catalog",
        default="interactiveShoppable/backend/metadata/catalog.json",
        type=Path,
        help="Input catalog JSON path",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Optional output path (defaults to in-place overwrite)",
    )
    parser.add_argument(
        "--field",
        default="aiCopy",
        help="Catalog field to store the structured copy",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate copy even if the field already exists",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--connect-timeout", type=int, default=5)
    parser.add_argument("--read-timeout", type=int, default=20)

    args = parser.parse_args()
    output_path = args.output or args.catalog

    updated = enrich_catalog(
        args.catalog,
        output_path=output_path,
        field=args.field,
        force=args.force,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
    )

    print(f"Completed. {updated} products refreshed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
