"""Shared data paths for local runs and deployed CineMatch services."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """Return the configured data directory, defaulting to the repository's data/."""
    configured_root = os.environ.get("CINEMATCH_DATA_ROOT")
    if configured_root:
        return Path(configured_root).expanduser()
    return PROJECT_ROOT / "data"


def data_path(*parts: str) -> Path:
    """Build a path under the active CineMatch data directory."""
    return data_root().joinpath(*parts)
