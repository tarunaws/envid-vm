#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _repo_code_dir() -> Path:
    # .../code/scripts/<this_file> -> .../code
    return Path(__file__).resolve().parents[1]


def _default_envid_metadata_dir() -> Path:
    return _repo_code_dir() / "envidMetadata"


def _load_json(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw) if raw else None


def _save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _stored_filename(entry: dict[str, Any]) -> str:
    stored = str(entry.get("stored_filename") or "").strip()
    if stored:
        return stored
    file_path_value = str(entry.get("file_path") or "").strip()
    if file_path_value:
        try:
            return Path(file_path_value).name
        except Exception:
            return ""
    return ""


def _has_cloud_source(entry: dict[str, Any]) -> bool:
    # Cloud source fields have changed over time. Keep this permissive so the
    # cleanup tool works with older indices.
    s3_uri = str(entry.get("s3_video_uri") or "").strip()
    s3_key = str(entry.get("s3_video_key") or "").strip()
    gcs_uri = str(entry.get("gcs_video_uri") or "").strip()
    gcs_object = str(entry.get("gcs_object") or "").strip()
    source_uri = str(entry.get("source_uri") or "").strip()
    return bool(s3_uri or s3_key or gcs_uri or gcs_object or source_uri)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Delete locally persisted envidMetadata videos (code/envidMetadata/videos/) "
            "for entries that already have a cloud source, and update the local video index."
        )
    )
    parser.add_argument(
        "--envid-metadata-dir",
        default=str(_default_envid_metadata_dir()),
        help="Path to code/envidMetadata (default: auto-detected)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files and write index changes (default: dry run)",
    )
    parser.add_argument(
        "--delete-unreferenced",
        action="store_true",
        help=(
            "Also delete files in videos/ that are not referenced by the index. "
            "(Still requires --apply.)"
        ),
    )

    args = parser.parse_args()

    envid_dir = Path(args.envid_metadata_dir).resolve()
    videos_dir = envid_dir / "videos"
    index_file = envid_dir / "indices" / "video_index.json"

    if not videos_dir.exists():
        print(f"videos dir not found: {videos_dir}")
        return 2

    index: list[dict[str, Any]] = []
    if index_file.exists():
        data = _load_json(index_file)
        if isinstance(data, list):
            index = data
        else:
            print(f"Index file is not a list: {index_file}")
            return 2
    else:
        print(f"Index file not found (continuing): {index_file}")

    referenced_filenames: set[str] = set()
    for entry in index:
        fn = _stored_filename(entry)
        if fn:
            referenced_filenames.add(fn)

    planned_deletes: list[Path] = []
    skipped_no_cloud: list[Path] = []
    updated_entries = 0

    for entry in index:
        fn = _stored_filename(entry)
        if not fn:
            continue

        local_path = videos_dir / fn
        if not local_path.exists():
            continue

        if not _has_cloud_source(entry):
            skipped_no_cloud.append(local_path)
            continue

        planned_deletes.append(local_path)
        if args.apply:
            # Make the entry cloud-only.
            entry["stored_filename"] = ""
            entry["file_path"] = ""
            updated_entries += 1

    unreferenced: list[Path] = []
    if args.delete_unreferenced:
        for p in sorted(videos_dir.glob("*")):
            if not p.is_file():
                continue
            if p.name in referenced_filenames:
                continue
            unreferenced.append(p)

    print("=== envidMetadata local video cleanup ===")
    print(f"envid_metadata_dir: {envid_dir}")
    print(f"videos_dir: {videos_dir}")
    print(f"index_file: {index_file}")
    print(f"mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print()

    if skipped_no_cloud:
        print("SKIP (no cloud source; keeping local copy):")
        for p in skipped_no_cloud:
            print(f"  - {p}")
        print()

    if planned_deletes:
        print("DELETE (has cloud source):")
        for p in planned_deletes:
            print(f"  - {p}")
        print()
    else:
        print("No indexed local videos eligible for deletion.")
        print()

    if unreferenced:
        print("UNREFERENCED files in videos/:")
        for p in unreferenced:
            print(f"  - {p}")
        print()

    if not args.apply:
        print("Dry run only. Re-run with --apply to delete and update the index.")
        return 0

    # Apply deletions
    deleted = 0
    for p in planned_deletes:
        try:
            p.unlink()
            deleted += 1
        except Exception as exc:
            print(f"Failed to delete {p}: {exc}")

    deleted_unreferenced = 0
    if args.delete_unreferenced:
        for p in unreferenced:
            try:
                p.unlink()
                deleted_unreferenced += 1
            except Exception as exc:
                print(f"Failed to delete unreferenced {p}: {exc}")

    # Persist index updates
    if index_file.exists():
        try:
            _save_json(index_file, index)
        except Exception as exc:
            print(f"Failed to write index file {index_file}: {exc}")
            return 2

    print("=== done ===")
    print(f"deleted_indexed_files: {deleted}")
    print(f"deleted_unreferenced_files: {deleted_unreferenced}")
    print(f"updated_index_entries: {updated_entries}")

    # Safety note
    if skipped_no_cloud:
        print("NOTE: Some files were kept because the index entry had no cloud source.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
