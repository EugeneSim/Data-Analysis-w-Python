"""Ops: health, HTTP client metrics, cache, security settings. Token-gated."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
if (_ROOT / "src").is_dir() and str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from singapore_eda.gov_http import (  # noqa: E402
    clear_http_cache,
    get_gov_client,
    get_metrics,
    verify_admin_token,
)
from singapore_eda.gov_limits import pace_config_public  # noqa: E402
from singapore_eda.health import run_health  # noqa: E402

st.set_page_config(page_title="Ops & Admin", layout="wide")
st.title("Ops, health & open-data client")

if "ops_unlocked" not in st.session_state:
    st.session_state.ops_unlocked = False


def _admin_token() -> str:
    try:
        t = st.secrets.get("SINGAPORE_EDA_ADMIN_TOKEN", "")
        if t:
            return str(t)
    except (AttributeError, FileNotFoundError, OSError, RuntimeError):
        pass
    return os.environ.get("SINGAPORE_EDA_ADMIN_TOKEN", "").strip()


expected = _admin_token()
if not expected:
    st.info(
        "Set **SINGAPORE_EDA_ADMIN_TOKEN** in the environment, or add "
        "`SINGAPORE_EDA_ADMIN_TOKEN` to **Streamlit Cloud secrets** / `.streamlit/secrets.toml` "
        "to unlock cache controls and full metrics. Health summary below is always visible."
    )
else:
    if not st.session_state.ops_unlocked:
        p = st.text_input("Admin token", type="password", key="ops_pw")
        if st.button("Unlock"):
            if verify_admin_token(p, expected):
                st.session_state.ops_unlocked = True
                st.rerun()
        elif p:
            st.caption("Click Unlock to verify.")
    else:
        st.caption("Admin session (this browser session).")
        if st.button("Lock again"):
            st.session_state.ops_unlocked = False
            st.rerun()

st.subheader("Health (imports, disk, optional CKAN probe)")
h = run_health()
st.metric("Status", h.status)
for c in h.checks:
    st.write(f"**{c.name}** — {'ok' if c.ok else 'fail'}: {c.detail}")
if h.status in ("degraded", "fail"):
    st.caption("Set SINGAPORE_EDA_HEALTH_SKIP_HTTP=1 in CI to skip the CKAN check.")

st.subheader("HTTP client (data.gov.sg / geo) — in-process")
st.caption("Pacing follows the official per-10s tables in `gov_limits` (tier + headroom).")
st.json(pace_config_public())
m = get_metrics()
st.json(m.as_dict())
c = get_gov_client()
info = c.cache_info()
if info.get("enabled") and info.get("path"):
    st.caption(
        f"Cache directory: `{info['path']}` "
        f"(~{info.get('approx_bytes', 0) // 1024} KiB)"
    )

st.markdown(
    """
| Environment variable | Purpose |
|---------------------|---------|
| `DATA_GOV_SG_API_KEY` | Optional; higher per-10s quotas (use with `SINGAPORE_EDA_API_TIER`) |
| `SINGAPORE_EDA_API_TIER` | `none` / `dev` / `prod` (tiers; default: no key→none, key→dev) |
| `SINGAPORE_EDA_RATE_HEADROOM` | Extra spacing on derived min intervals (default 0.12) |
| `SINGAPORE_EDA_MIN_INTERVAL_DATASTORE_SEC` | Override CKAN `datastore_search` spacing only |
| `SINGAPORE_EDA_MIN_INTERVAL_FILE_SEC` | Override file / poll-download spacing only |
| `SINGAPORE_EDA_MIN_INTERVAL_SEC` | Override **both** paces (legacy single knob) |
| `SINGAPORE_EDA_SKIP_DOWNLOAD_IF_FRESH_HOURS` | Skip CSV download if file younger than H hours |
| `SINGAPORE_EDA_CHECK_NEW_DATA` | Re-download on CKAN `total` growth (with skip-fresh) |
| `SINGAPORE_EDA_TOWN_COVERAGE` | OLS town coverage fraction (`0.8` default; `1` keeps all) |
| `SINGAPORE_EDA_GEO_SKIP_IF_FRESH_DAYS` | Skip GeoJSON if file younger than D days |
| `SINGAPORE_EDA_HTTP_CACHE_ENABLED` | `0` to disable on-disk response cache |
| `SINGAPORE_EDA_HTTP_CACHE_DIR` | Where JSON GET responses are stored |
| `SINGAPORE_EDA_HTTP_CACHE_TTL_SEC` | TTL for cached JSON (default 3600) |
| `SINGAPORE_EDA_DATASTORE_PAGE_CACHE` | `1` to cache each CKAN page (large on full imports) |
| `SINGAPORE_EDA_HEALTH_SKIP_HTTP` | `1` to skip online probe (e.g. CI) |
"""
)

if st.session_state.get("ops_unlocked") and expected:
    st.subheader("Cache actions")
    if st.button("Clear on-disk JSON cache (non-destructive to CSV data)"):
        n = clear_http_cache()
        st.success(f"Removed {n} cache file(s).")

st.divider()
st.caption(
    "Production: run the optional `singapore-eda-health` ASGI app for `/health`, `/ready`, "
    "`/metrics` (see README). This page is not a replacement for your cluster’s monitoring."
)
