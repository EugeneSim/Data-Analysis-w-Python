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
st.markdown(
    """
<style>
.block-container {padding-top: 1.0rem; padding-bottom: 1.5rem; max-width: 1240px;}
h1, h2, h3 {letter-spacing: -0.02em;}
.ux-hero {
  border-radius: 16px;
  border: 1px solid rgba(37, 99, 235, 0.22);
  background:
      radial-gradient(
          circle at 15% 20%,
          rgba(59, 130, 246, 0.20),
          rgba(255, 255, 255, 0.98) 40%
      ),
      linear-gradient(120deg, rgba(37, 99, 235, 0.08), rgba(15, 23, 42, 0.03));
  padding: 1.05rem 1.15rem;
  margin: 0.2rem 0 0.9rem 0;
}
.ux-hero .eyebrow {
  font-size: 0.74rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
  color: #1e3a8a;
  margin-bottom: 0.32rem;
}
.ux-hero h3 {
  margin: 0 0 0.3rem 0;
  color: #0f172a;
  font-size: 1.18rem;
}
.ux-hero p {
  margin: 0;
  color: #334155;
  line-height: 1.35rem;
}
.ux-divider {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  margin: 1.0rem 0 0.65rem 0;
  color: #0f172a;
}
.ux-divider .icon {
  width: 1.7rem;
  height: 1.7rem;
  border-radius: 999px;
  background: rgba(37, 99, 235, 0.12);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.95rem;
}
.ux-divider .title {
  font-size: 1.02rem;
  font-weight: 680;
}
.ux-divider-line {
  flex: 1;
  min-width: 40px;
  border-top: 1px solid rgba(148, 163, 184, 0.45);
}
.ux-metric {
  border-radius: 12px;
  border: 1px solid rgba(148, 163, 184, 0.3);
  background: linear-gradient(165deg, rgba(248, 250, 252, 1), rgba(241, 245, 249, 0.85));
  padding: 0.75rem 0.85rem;
  min-height: 96px;
}
.ux-metric .top {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  color: #334155;
  font-size: 0.85rem;
  font-weight: 620;
  margin-bottom: 0.35rem;
}
.ux-metric .val {
  color: #0f172a;
  font-size: 1.28rem;
  font-weight: 760;
  line-height: 1.25;
}
</style>
""",
    unsafe_allow_html=True,
)


def _hero_card(title: str, subtitle: str, eyebrow: str = "Operations Console") -> None:
    st.markdown(
        (
            "<div class='ux-hero'>"
            f"<div class='eyebrow'>{eyebrow}</div>"
            f"<h3>{title}</h3>"
            f"<p>{subtitle}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _section_divider(title: str, icon: str = "◉") -> None:
    st.markdown(
        (
            "<div class='ux-divider'>"
            f"<span class='icon'>{icon}</span>"
            f"<span class='title'>{title}</span>"
            "<span class='ux-divider-line'></span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, icon: str) -> None:
    st.markdown(
        (
            "<div class='ux-metric'>"
            f"<div class='top'><span>{icon}</span><span>{label}</span></div>"
            f"<div class='val'>{value}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


_hero_card(
    "Operations and Reliability Console",
    (
        "Monitor health checks, API pacing, cache behavior, and admin controls "
        "in a single operational view."
    ),
    eyebrow="Open Data Ops",
)

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

_section_divider("Health (imports, disk, optional CKAN probe)", icon="🩺")
h = run_health()
_metric_card("Status", h.status.upper(), "✅")
for c in h.checks:
    st.write(f"**{c.name}** — {'ok' if c.ok else 'fail'}: {c.detail}")
if h.status in ("degraded", "fail"):
    st.caption("Set SINGAPORE_EDA_HEALTH_SKIP_HTTP=1 in CI to skip the CKAN check.")

_section_divider("HTTP client (data.gov.sg / geo) — in-process", icon="🌐")
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

_section_divider("Environment configuration reference", icon="⚙️")
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
    _section_divider("Cache actions", icon="🧹")
    if st.button("Clear on-disk JSON cache (non-destructive to CSV data)"):
        n = clear_http_cache()
        st.success(f"Removed {n} cache file(s).")

st.divider()
st.caption(
    "Production: run the optional `singapore-eda-health` ASGI app for `/health`, `/ready`, "
    "`/metrics` (see README). This page is not a replacement for your cluster’s monitoring."
)
