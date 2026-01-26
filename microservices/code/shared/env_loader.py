from __future__ import annotations

from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


def _repo_root() -> Path:
    # env_loader.py -> shared -> code -> repo root
    return Path(__file__).resolve().parents[2]


def load_environment(*, project_root: Path | None = None, extra_files: Iterable[Path] | None = None) -> list[Path]:
    """Load environment files in repo-defined order.

    Order:
      1) .env
      2) .env.local
    3) backend/.env.multimodal.local
    4) backend/.env.multimodal.secrets.local
    """
    root = project_root or _repo_root()
    candidates = [
        root / ".env",
        root / ".env.local",
        root / "backend/.env.multimodal.local",
        root / "backend/.env.multimodal.secrets.local",
    ]
    if extra_files:
        candidates.extend(extra_files)

    loaded: list[Path] = []
    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=True)
            loaded.append(path)
    return loaded
