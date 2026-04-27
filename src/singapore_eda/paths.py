"""Resolve paths to `data/reference` whether running from repo root or installed src layout."""

from __future__ import annotations

from pathlib import Path

_PACKAGE = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE.parent.parent


def reference_dir() -> Path:
    """Prefer `./data/reference` (cwd), else repo `data/reference` next to this package."""
    cwd = Path.cwd() / "data" / "reference"
    if cwd.exists():
        return cwd
    return _REPO_ROOT / "data" / "reference"
