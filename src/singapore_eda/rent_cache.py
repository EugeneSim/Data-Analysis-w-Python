"""File age helpers for near-live rent refresh in apps."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path

_DEFAULT_TTL_H = 24.0


def _mtime(path: Path) -> float:
    return path.stat().st_mtime


def rent_csv_is_fresh(
    path: Path | str,
    *,
    ttl_hours: float | None = None,
) -> bool:
    """
    Return True if file exists and is newer than ttl_hours.
    If ttl is None, read SINGAPORE_EDA_RENT_TTL_HOURS (default 24) or never expire if unset?
    we use default 24 if env not set
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return False
    h = ttl_hours
    if h is None:
        h = float(os.environ.get("SINGAPORE_EDA_RENT_TTL_HOURS", str(_DEFAULT_TTL_H)))
    age_sec = time.time() - _mtime(p)
    return age_sec < h * 3600.0


def age_hours(path: Path | str) -> float | None:
    """Hours since mtime, or None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    return (time.time() - _mtime(p)) / 3600.0


def iso_mtime(path: Path | str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    ts = datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)
    return ts.isoformat()
