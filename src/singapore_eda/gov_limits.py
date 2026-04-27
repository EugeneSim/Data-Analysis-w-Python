"""
data.gov.sg API rate targets (2025+ official table).

**Limits apply per 10 second window** and differ by product:
- *Datastore search* (CKAN `datastore_search`): 4 (no key) / 8 (dev) / 20 (prod) calls per 10s.
- *Dataset / file* downloads: 2 / 4 / 10 per 10s (stricter).

We derive **minimum wall-clock spacing** between calls of the same class as
``(10.0 / calls_per_10s) * (1 + headroom)`` so a steady stream of requests
stays under the table for single-threaded use. Use ``DATA_GOV_SG_API_KEY`` in
production and set ``SINGAPORE_EDA_API_TIER=prod`` for the prod row.

**Reference:** `API_RATE_DOCS` below. Limits may change; override intervals via env
if the portal publishes an update.
"""

from __future__ import annotations

import os
from typing import Any, Literal

API_RATE_DOCS = "https://guide.data.gov.sg/developer-guide/api-overview/api-rate-limits"

# Official max calls per 10s by tier (row names: none / dev / prod).
DATASTORE_CALLS_PER_10S: dict[Literal["none", "dev", "prod"], int] = {
    "none": 4,
    "dev": 8,
    "prod": 20,
}
# "Dataset downloads" in the same doc (geo assets, file endpoints use this)
FILE_CALLS_PER_10S: dict[Literal["none", "dev", "prod"], int] = {
    "none": 2,
    "dev": 4,
    "prod": 10,
}

DEFAULT_RATE_HEADROOM = 0.12  # 12% — burst padding inside the 10s window


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def rate_headroom() -> float:
    return _env_float("SINGAPORE_EDA_RATE_HEADROOM", DEFAULT_RATE_HEADROOM)


def resolve_api_tier() -> Literal["none", "dev", "prod"]:
    t = os.environ.get("SINGAPORE_EDA_API_TIER", "").strip().lower()
    if t in ("none", "dev", "prod"):
        return t  # type: ignore[return-value]
    if os.environ.get("DATA_GOV_SG_API_KEY", "").strip():
        return "dev"
    return "none"


def _interval_from_limit(calls_per_10s: int, headroom: float) -> float:
    if calls_per_10s < 1:
        calls_per_10s = 1
    return (10.0 / float(calls_per_10s)) * (1.0 + headroom)


def min_interval_datastore_sec() -> float:
    """
    Pacing for CKAN ``datastore_search`` (and same API family).

    For tier ``none`` (no API key), we treat the official limit as one call
    stricter (3/10s instead of 4/10s) to reduce 429s on busy networks.
    """
    ex = os.environ.get("SINGAPORE_EDA_MIN_INTERVAL_DATASTORE_SEC", "").strip()
    if ex:
        return max(0.0, float(ex))
    ex_legacy = os.environ.get("SINGAPORE_EDA_MIN_INTERVAL_SEC", "").strip()
    if ex_legacy:
        return max(0.0, float(ex_legacy))
    tier = resolve_api_tier()
    n = DATASTORE_CALLS_PER_10S[tier]
    if tier == "none":
        n = max(1, n - 1)
    return _interval_from_limit(n, rate_headroom())


def min_interval_file_sec() -> float:
    """
    Pacing for file / dataset **download** style endpoints
    (e.g. data.gov.sg poll-download / large file GETs).
    Tighter server limits — use a separate, slower pace than datastore search.
    """
    ex = os.environ.get("SINGAPORE_EDA_MIN_INTERVAL_FILE_SEC", "").strip()
    if ex:
        return max(0.0, float(ex))
    ex_legacy = os.environ.get("SINGAPORE_EDA_MIN_INTERVAL_SEC", "").strip()
    if ex_legacy:
        return max(0.0, float(ex_legacy))
    tier = resolve_api_tier()
    n = FILE_CALLS_PER_10S[tier]
    return _interval_from_limit(n, rate_headroom())


def pace_config_public() -> dict[str, Any]:
    """Safe dict for admin UI / health (no secrets)."""
    tier = resolve_api_tier()
    has_key = bool(os.environ.get("DATA_GOV_SG_API_KEY", "").strip())
    return {
        "api_rate_limit_docs": API_RATE_DOCS,
        "resolved_tier": tier,
        "has_data_gov_api_key": has_key,
        "datastore_calls_per_10s_official": DATASTORE_CALLS_PER_10S[tier],
        "file_download_calls_per_10s_official": FILE_CALLS_PER_10S[tier],
        "min_interval_datastore_sec_effective": round(min_interval_datastore_sec(), 4),
        "min_interval_file_sec_effective": round(min_interval_file_sec(), 4),
        "rate_headroom": rate_headroom(),
    }


def skip_download_fresh_hours_default() -> float | None:
    """Optional: skip re-downloading raw CSV if file exists and is newer than this (hours)."""
    raw = os.environ.get("SINGAPORE_EDA_SKIP_DOWNLOAD_IF_FRESH_HOURS", "").strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def skip_geo_fresh_days_default() -> float | None:
    """Optional: skip re-fetching planning GeoJSON (poll-download) if file is this fresh (days)."""
    raw = os.environ.get("SINGAPORE_EDA_GEO_SKIP_IF_FRESH_DAYS", "").strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None
