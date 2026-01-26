#!/usr/bin/env python3
"""Batch-generate product imagery for the Interactive & Shoppable catalog.

This script reuses the existing Image Creation service (code/imageCreation) that
generates images using whatever model/provider the service is configured for.

For every catalog entry we submit a text prompt to the image service and update
`metadata/catalog.json` with the returned URL so product names and thumbnails stay
in sync. Run while both services are up:

    ./start-all.sh
    python scripts/update_catalog_images.py \
        --catalog interactiveShoppable/backend/metadata/catalog.json \
        --service-url http://localhost:5002/send_prompt

Use `--force` if you want to overwrite URLs that already point to generated
assets. Without `--force`, existing custom URLs are preserved.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from textwrap import dedent

import requests

DEFAULT_SERVICE_URL = "http://localhost:5002/send_prompt"
DEFAULT_DELAY = 1.5  # seconds between requests to avoid throttling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        default="interactiveShoppable/backend/metadata/catalog.json",
        help="Path to the catalog JSON file to update",
    )
    parser.add_argument(
        "--service-url",
        default=DEFAULT_SERVICE_URL,
        help="Image creation endpoint to call (defaults to the local Image Creation service)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate image URL even if the catalog already has a non-placeholder image",
    )
    parser.add_argument(
        "--prompt-field",
        choices=["name", "description", "combined"],
        default="combined",
        help="Which catalog field to use when building prompts (default: combined name + description)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Seconds to wait between image-generation requests",
    )
    return parser.parse_args()


def build_prompt(item: dict, mode: str) -> str:
    if mode == "name":
        return item.get("name", "Product")
    if mode == "description":
        return item.get("description") or item.get("name", "Product")

    description = item.get("description") or "hero product shot"
    label = item.get("label", "Product")
    style = (
        "Ultra-realistic product photography, cinematic lighting, 4K render, "
        "clean studio backdrop, physically accurate materials, sharp focus, depth of field"
    )
    return dedent(f"""
        Hyper-detailed marketing photo of {item['name']} ({label}).
        {description}
        {style}.
    """).strip()


def should_skip(item: dict, force: bool) -> bool:
    if force:
        return False
    image_url = (item.get("image") or "").strip()
    if not image_url:
        return False
    if "placeholder" in image_url:
        return False
    return True


def request_image(prompt: str, service_url: str) -> str:
    response = requests.post(service_url, json={"prompt": prompt}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    image_url = (payload.get("image_url") or "").strip()
    if not image_url:
        raise RuntimeError(f"Image service returned no image_url for prompt: {prompt[:80]}…")
    return image_url


def main() -> int:
    args = parse_args()
    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print(f"Catalog file not found: {catalog_path}", file=sys.stderr)
        return 1

    items = json.loads(catalog_path.read_text())
    updated = 0

    for idx, item in enumerate(items, start=1):
        if should_skip(item, args.force):
            continue
        prompt = build_prompt(item, args.prompt_field)
        try:
            image_url = request_image(prompt, args.service_url)
        except Exception as exc:  # noqa: BLE001
            print(f"[{idx}/{len(items)}] Failed to generate image for {item['id']}: {exc}", file=sys.stderr)
            continue

        item['image'] = image_url
        updated += 1
        print(f"[{idx}/{len(items)}] Updated {item['id']} -> {image_url}")
        time.sleep(max(args.delay, 0))

    if updated:
        catalog_path.write_text(json.dumps(items, indent=2))
        print(f"Saved catalog with {updated} refreshed image URLs -> {catalog_path}")
    else:
        print("No catalog entries were updated. Use --force to overwrite existing URLs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
