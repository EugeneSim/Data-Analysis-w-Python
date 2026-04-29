"""Streamlit dashboard: HDB resale EDA (maps, clusters, forecast, graph, yields)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

if Path(__file__).resolve().parent.joinpath("src").exists():
    _src = Path(__file__).resolve().parent / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from singapore_eda.clustering import cluster_interpretation, cluster_kmeans
from singapore_eda.constants import (
    DEFAULT_RAW_CSV,
    DEFAULT_RENT_CSV,
    HDB_CITATION_URL,
    HDB_DATASTORE_SEARCH,
    HDB_MEDIAN_RENT_CITATION_URL,
    HDB_MEDIAN_RENT_RESOURCE_ID,
    PLANNING_AREA_GEOJSON_POLL_URL,
)
from singapore_eda.display_table import (
    block_table_for_display,
    cluster_medians_for_display,
    correlation_edges_for_display,
    forecast_rmse_for_display,
    gross_yield_table_for_display,
    storey_table_for_display,
)
from singapore_eda.eip import eip_match_stats
from singapore_eda.features import add_bto_reference_features, model_design_subset
from singapore_eda.feedback import append_feedback
from singapore_eda.forecaster_config import load_forecaster_config
from singapore_eda.forecaster_v1 import predict_with_explain
from singapore_eda.forecasting import backtest_rmse, monthly_median_price
from singapore_eda.gov_http import get_gov_client
from singapore_eda.graph_analytics import (
    correlation_community_table,
    correlation_edges_dataframe,
    correlation_graph,
    graph_summary,
    spatial_adjacency_graph,
    town_price_pivot,
    unweighted_spatial_edge_table,
)
from singapore_eda.housing_finance import (
    GrantSelection,
    HouseholdProfile,
    HousingFinanceScenario,
    HousingType,
    LoanType,
    RateSegment,
    load_policy_defaults,
    run_housing_finance,
)
from singapore_eda.housing_finance.calculators import make_fixed_then_sora_segments
from singapore_eda.housing_finance.formatters import (
    cashflow_table,
    government_return_table,
    itemized_cost_table,
    profit_breakdown_table,
)
from singapore_eda.insights import build_insights
from singapore_eda.mapviz import (
    block_street_table,
    folium_choropleth_by_name,
    geo_median_value_dict,
)
from singapore_eda.pipeline import load_enriched
from singapore_eda.rent_cache import age_hours, rent_csv_is_fresh
from singapore_eda.rent_ingest import download_median_rent
from singapore_eda.rental_yields import gross_yield_table
from singapore_eda.stats import (
    numeric_correlation,
    ols_log_price,
    ols_log_price_with_storey,
    ttest_resale_by_group,
)
from singapore_eda.storey import median_price_by_storey_stratum
from singapore_eda.viz import (
    fig_correlation_heatmap,
    fig_flat_type_box,
    fig_median_price_by_town,
    fig_price_over_time,
    fig_town_median_lines_with_forecast,
)

_ROOT = Path(__file__).resolve().parent
_DEFAULT_FIXTURE = _ROOT / "tests" / "fixtures" / "hdb_sample.csv"
_ALT_BIG_CSV = _ROOT / "data" / "data" / "sales_data.csv"
_RENT_CAND = _ROOT / str(DEFAULT_RENT_CSV)
_FIX_RENT = _ROOT / "tests" / "fixtures" / "median_rent_sample.csv"
_DEFAULT_RENT = str(_RENT_CAND) if _RENT_CAND.is_file() else str(_FIX_RENT)
_PLANNING_GEO = _ROOT / "data" / "reference" / "planning_areas.geojson"
_TINY_GEO = _ROOT / "tests" / "fixtures" / "planning_areas_tiny.geojson"
_REQUIRED_RESALE_COLS = {"month", "resale_price"}
_FORECASTER_MODEL_PATH = _ROOT / "models" / "forecaster_v1" / "model.joblib"
_FORECASTER_META_PATH = _ROOT / "models" / "forecaster_v1" / "metadata.json"
_FORECASTER_CFG_PATH = _ROOT / "configs" / "forecaster_v1.yaml"
_HOUSING_FINANCE_CFG_PATH = _ROOT / "configs" / "housing_finance_v1.yaml"
_BTO_REFERENCE_PATH = _ROOT / "data" / "reference" / "hdb_bto_reference.csv"


def _is_truthy(v: str | None) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _setting(name: str) -> str | None:
    # Prefer real environment variables, then Streamlit secrets for Cloud deploys.
    raw = os.environ.get(name)
    if raw is not None:
        return raw
    try:
        sec = st.secrets.get(name)
    except (AttributeError, FileNotFoundError, OSError, RuntimeError):
        sec = None
    if sec is None:
        return None
    return str(sec)


def _on_streamlit_cloud() -> bool:
    # Streamlit Cloud exposes this in hosted apps.
    return _is_truthy(os.environ.get("STREAMLIT_SHARING_MODE"))


def _default_bool_env(name: str, cloud_default: bool, local_default: bool) -> bool:
    raw = _setting(name)
    if raw is not None:
        return _is_truthy(raw)
    return cloud_default if _on_streamlit_cloud() else local_default


def _default_geo() -> str:
    if _PLANNING_GEO.exists():
        return str(_PLANNING_GEO)
    if _TINY_GEO.exists():
        return str(_TINY_GEO)
    return ""


def _bootstrap_resale_if_missing(default_path: str) -> str:
    def _has_required_cols(p: Path) -> bool:
        try:
            cols = pd.read_csv(p, nrows=0).columns
        except (OSError, ValueError, pd.errors.ParserError):
            return False
        norm = {str(c).strip().lower().replace(" ", "_") for c in cols}
        return _REQUIRED_RESALE_COLS.issubset(norm)

    p = Path(default_path)
    if p.exists() and _has_required_cols(p):
        return default_path
    if p.exists() and not _has_required_cols(p):
        st.sidebar.warning(
            f"CSV at `{p}` does not look like HDB resale data "
            f"(needs columns: {sorted(_REQUIRED_RESALE_COLS)})."
        )
    if _ALT_BIG_CSV.exists():
        if _has_required_cols(_ALT_BIG_CSV):
            st.sidebar.info(f"Using bundled resale CSV: `{_ALT_BIG_CSV}`")
            return str(_ALT_BIG_CSV)
        st.sidebar.warning(
            f"Bundled CSV `{_ALT_BIG_CSV}` is not HDB-format; skipping."
        )

    auto_fetch = _default_bool_env(
        "SINGAPORE_EDA_AUTO_DOWNLOAD_ON_MISSING",
        cloud_default=True,
        local_default=False,
    )
    if auto_fetch:
        max_rows_raw = _setting("SINGAPORE_EDA_BOOTSTRAP_MAX_ROWS") or "20000"
        try:
            max_rows = max(1000, int(str(max_rows_raw).strip()))
        except ValueError:
            max_rows = 20000
        try:
            from singapore_eda.download_data import download_hdb_resale

            p.parent.mkdir(parents=True, exist_ok=True)
            n = download_hdb_resale(
                p,
                max_rows=max_rows,
                latest_first=True,
                skip_if_fresh_hours=24.0,
            )
            st.sidebar.success(f"Fetched {n:,} resale rows from data.gov.sg.")
            if p.exists() and _has_required_cols(p):
                return str(p)
            if p.exists():
                st.sidebar.warning(
                    f"Downloaded file `{p}` is missing required HDB columns; falling back."
                )
        except (OSError, ValueError, RuntimeError) as ex:
            st.sidebar.warning(f"Auto-download failed; using fixture fallback. ({ex})")

    if _DEFAULT_FIXTURE.exists():
        st.sidebar.info(
            "Using fallback fixture (small sample). "
            "Set `SINGAPORE_EDA_AUTO_DOWNLOAD_ON_MISSING=1` to auto-fetch data."
        )
        return str(_DEFAULT_FIXTURE)
    return default_path


def _bootstrap_rent_if_missing(default_path: str) -> str:
    p = Path(default_path)
    if p.exists():
        return default_path

    auto_fetch = _default_bool_env(
        "SINGAPORE_EDA_AUTO_DOWNLOAD_RENT_ON_MISSING",
        cloud_default=True,
        local_default=False,
    )
    if auto_fetch:
        rid = (_setting("HDB_MEDIAN_RENT_RESOURCE_ID") or HDB_MEDIAN_RENT_RESOURCE_ID).strip()
        max_rows_raw = _setting("SINGAPORE_EDA_RENT_BOOTSTRAP_MAX_ROWS") or "200000"
        try:
            max_rows = max(1000, int(str(max_rows_raw).strip()))
        except ValueError:
            max_rows = 200000
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            n = download_median_rent(
                p,
                rid,
                max_rows=max_rows,
                skip_if_fresh_hours=24.0,
            )
            st.sidebar.success(f"Fetched {n:,} rent rows from data.gov.sg.")
            if p.exists():
                return str(p)
        except (OSError, ValueError, RuntimeError) as ex:
            st.sidebar.warning(f"Rent auto-download failed; using fixture fallback. ({ex})")

    if _FIX_RENT.exists():
        return str(_FIX_RENT)
    return default_path


def _bootstrap_geo_if_tiny_or_missing(default_path: str) -> str:
    p = Path(default_path) if default_path else Path("")
    is_tiny = p.exists() and p.name == _TINY_GEO.name
    if p.exists() and not is_tiny:
        return str(p)

    auto_fetch = _default_bool_env(
        "SINGAPORE_EDA_AUTO_FETCH_GEO_ON_TINY",
        cloud_default=True,
        local_default=False,
    )
    if auto_fetch:
        try:
            body = get_gov_client().get_json(
                PLANNING_AREA_GEOJSON_POLL_URL,
                timeout=90,
                use_cache=False,
                use_file_pace=True,
            )
            if body.get("code") != 0:
                raise RuntimeError(str(body.get("errMsg") or body.get("errorMsg") or "poll failed"))
            blob_url = (body.get("data") or {}).get("url")
            if not blob_url:
                raise RuntimeError("poll-download response missing URL")
            r = requests.get(str(blob_url), timeout=180)
            r.raise_for_status()
            _PLANNING_GEO.parent.mkdir(parents=True, exist_ok=True)
            _PLANNING_GEO.write_bytes(r.content)
            st.sidebar.success("Fetched full planning-area GeoJSON.")
            return str(_PLANNING_GEO)
        except (OSError, ValueError, RuntimeError, requests.RequestException) as ex:
            st.sidebar.warning(f"Geo auto-fetch failed; keeping current GeoJSON. ({ex})")

    return default_path


def _ols_coef_table(model: Any, top_n: int = 12) -> pd.DataFrame:
    ci = model.conf_int()
    rows: list[dict[str, object]] = []
    for term, beta in model.params.items():
        if term == "Intercept":
            continue
        lo = float(ci.loc[term, 0]) if term in ci.index else float("nan")
        hi = float(ci.loc[term, 1]) if term in ci.index else float("nan")
        pct = (pow(2.718281828459045, float(beta)) - 1.0) * 100.0
        rows.append(
            {
                "term": str(term),
                "coef_log": float(beta),
                "approx_price_change_pct": pct,
                "p_value": float(model.pvalues.get(term, float("nan"))),
                "ci95_log_low": lo,
                "ci95_log_high": hi,
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # Keep continuous drivers first, then strongest remaining terms by magnitude.
    priority = ["floor_area_sqm", "remaining_lease_years", "C(storey_band)"]
    pri_mask = out["term"].map(lambda t: any(k in str(t) for k in priority))
    pri = out[pri_mask].copy()
    rest = out[~out.index.isin(pri.index)].copy()
    rest = rest.sort_values(by="approx_price_change_pct", key=lambda s: s.abs(), ascending=False)
    out2 = pd.concat([pri, rest.head(max(0, top_n - len(pri)))], ignore_index=True)
    out2 = out2.sort_values(by="p_value", na_position="last")
    return out2


def _render_ols_readable(model: Any, title: str, add_storey_note: bool = False) -> None:
    st.subheader(title)
    c0, c1, c2, c3 = st.columns(4)
    c0.metric("Rows used", f"{int(model.nobs):,}")
    c1.metric("R-squared", f"{float(model.rsquared):.3f}")
    c2.metric("Adj. R-squared", f"{float(model.rsquared_adj):.3f}")
    c3.metric("F-test p-value", f"{float(model.f_pvalue):.2e}")
    st.markdown(
        """
**What this means to you**
- **What OLS solves here:** it estimates each factor's relationship with resale price while
  **holding the other included factors fixed** (e.g. area, lease, town group, and optionally
  storey band).
- **Higher is better?** not universally. A **higher predicted price** can be good for
  sellers/owners, but usually worse for buyers seeking affordability.
- **How to read signs:** a **positive** coefficient means higher expected price; **negative**
  means lower expected price (within this dataset and controls).
- **Decision use:** use this as a **structured explanation tool**, not a valuation engine or
  investment signal.
"""
    )
    if add_storey_note:
        st.caption(
            "Storey coefficients are relative to a hidden reference band. They are descriptive, "
            "not proof of causal floor premiums."
        )
    coef_tbl = _ols_coef_table(model)
    if not coef_tbl.empty:
        st.markdown("**Most relevant coefficients (human-readable)**")
        st.dataframe(coef_tbl, width="stretch", height=340)
    with st.expander("Raw statistical output (advanced)", expanded=False):
        st.text(model.summary().as_text()[:4000])


PROJECT_AND_METHOD = """
### What problem does this app address?
Exploratory analysis of **HDB resale transactions**: how median prices differ by **town / planning
area**, **block**, **storey band**, and **time**; a simple view of **gross rental yield** when
median rent is joined; **k-means segments** to group similar homes; a **trend + short forecast** per
town; and a **co-movement** graph between town price series. It does **not** provide valuation or
investment advice.

### What data do you use?
- **Resale (required):** HDB *Resale flat prices* open data (CSV) — at least `month`, `town`,
  `resale_price`, and ideally `flat_type`, `block`, `street_name`, `storey_range`, `floor_area_sqm`.
- **Optional rent (for yields):** HDB
  [Median rent by town and flat type (quarterly)](https://data.gov.sg/datasets/d_23000a00c52996c55106084ed0339566/view)
  via `hdb-rent-download` (see `HDB_MEDIAN_RENT_RESOURCE_ID`). CSV fields: `quarter`, `town`,
  `flat_type`, `median_rent` (whole‑flat medians; flat-type codes are normalised to match resale
  e.g. `3-RM` → `3 ROOM`).
- **Optional map:** planning-area **GeoJSON** (e.g. from `python scripts/fetch_reference_geo.py`).

HDB’s published **median rent is for the whole unit** for that `flat_type` in that place/period, not
a bedroom. Very old resales matched to a recent rent file will look “wrong” in yield — you must
align **period and cohort**; the app joins on **quarter + town + flat_type**.

### How is data stored in this project?
- **In this repo / your machine:** raw CSVs under `data/raw/`, small reference tables under
  `data/reference/`, optional downloaded GeoJSON there.
- **In the app:** the selected resale CSV and rent CSV are **read in memory** (Pandas) — there is
  no separate database unless you add one. For automation, a pipeline can write **Parquet**
  (see `scripts/run_analysis.py`).

### What models or methods are used?
- **Descriptive & plots:** medians, groupby, time aggregation.
- **Regression (OLS, Overview tab):** *Ordinary Least Squares* on **log(resale price)**. We model
  **log** price (not levels) so a common multiplicative story applies: the same *percentage* change
  in e.g. area is easier to compare across towns. Fixed effects are *dummy variables for* **town
  group** (see “town coverage” below) plus **floor area (m²)** and **remaining lease (years, from
  HDB’s text)**. **R²** is *in‑sample* explanatory power; it is **not** a forecast accuracy score
  and does not validate trading decisions.
- **OLS + storey bands:** the same OLS, plus **categorical storey bands** (low / mid / high) from
  HDB’s *storey range* (a band, not the exact floor). Coefficients are **relative to a reference
  category**; use them as stylised *within file* structure, not as causal “floor premia” without
  controls for block and time.
- **Two-sample t-test (Overview):** *Welch’s t-test* (unequal variances) on **raw** resale prices
  for two *town* groups. It asks whether the two town samples have **different mean** prices, under
  normal-ish data; it does **not** control for size, age, or lease, so a difference can reflect
  *composition* of flats, not a “town effect” in the causal sense.
- **Clustering:** k-means (standardized features) for **segmentation labels**; labels are
  interpreted against sample medians in the Clusters tab.
- **Time series:** Holt–Winters **ETS** on **monthly median** by town; history must be long enough
  (~24+ months) for a meaningful seasonal fit.
- **Graphs — correlation network:** edges where Pearson **|ρ|** between **monthly median**
  **price series** (by town) exceeds your sidebar threshold. **Not geography** (layout is a force
  algorithm).
- **Graphs — planning-area touch (neighbour) graph:** **polygons that share a boundary** in your
  GeoJSON. Install optional **`pip install -e ".[geo]"` (geopandas)**; we build edges with a spatial
  join (`touches`) and fall back to a slower geometry loop if needed.

### Towns in the sample (“~80% rule”)
For regression, rare towns are collapsed into a single **OTHER** category so the model is stable.
We keep the **smallest** set of towns that together make up a target share of *rows* (default
**80%** of transactions). The rest are **OTHER**; they are still in charts and the raw data.
Override with env **`SINGAPORE_EDA_TOWN_COVERAGE`** (e.g. `1.0` to keep every town as its own
group — many dummy levels) or in code `top_towns=…` with `town_coverage=None`.

### Remaining lease
The CSV gives **lease text** and we compute `remaining_lease_years` as a *decimal* (e.g. 92.4167 ≈
92 years and 5 months in our parser). The app also shows **`remaining_lease_label`**: the **HDB
string** when present, else a compact **Y/M/D** style so “92.4167” is not the only readout.

### Caching and refreshing downloads
- **Resale:** after each `hdb-download` run, a **sidecar** `*.meta.json` (next to the CSV) records
  CKAN **`api_total`**. If you set **`SINGAPORE_EDA_SKIP_DOWNLOAD_IF_FRESH_HOURS=…`** and
  **`SINGAPORE_EDA_CHECK_NEW_DATA=1`**, a **fresh** local file is still re-fetched when the
  *live* API **total** exceeds the **stored** total (indicating new records on the portal).
- **Comprehensive data:** the default Jan‑2017+ tranche is one resource; *older* tranches and merges
  are available via `hdb-download --list-tranches` and `hdb-resale-merge` (see help when schemas
  align).
- **geo:** for planning-area map and neighbour graph, `python scripts/fetch_reference_geo.py` and
  `pip install -e ".[geo]"`.

### How would you deploy, and how do users interact?
- **This Streamlit app:** `streamlit run streamlit_app.py` — users explore in the **browser**;
  no model is “served” as an API; it’s an interactive dashboard.
- **Static report:** `quarto render` under `quarto/` and host on **GitHub Pages** (read-only, no
  Python in-browser).
- **A production option** (not in this bundle): precompute features and predictions to a
  **database or API** (e.g. FastAPI + Postgres) and build a read-only or authenticated UI — only if
  you need scale, access control, or scheduled scoring.

---
**Limits:** the open HDB file has **no building coordinates**; we cannot draw true “pins per block”
on a basemap. The **Map** uses planning-area outlines; **Block** is a table (town / street / block
medians) from the same rows.
"""


def _town_coverage_env() -> float | None:
    raw = str(os.environ.get("SINGAPORE_EDA_TOWN_COVERAGE", "")).strip()
    if not raw:
        return 0.8
    if raw.lower() in ("all", "1", "full", "1.0"):
        return 1.0
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.8


@st.cache_data(show_spinner=True)
def _load(path: str) -> pd.DataFrame:
    return load_enriched(
        path,
        top_towns=15,
        town_coverage=_town_coverage_env(),
    )


def _fig_network(g: nx.Graph) -> go.Figure:
    if g.number_of_nodes() == 0:
        return go.Figure()
    pos = nx.spring_layout(g, seed=42)
    edge_x: list[float] = []
    edge_y: list[float] = []
    for u, v in g.edges():
        edge_x.extend([pos[u][0], pos[v][0], None])
        edge_y.extend([pos[u][1], pos[v][1], None])
    fig = go.Figure()
    line_kw = dict(width=1, color="#888")
    fig.add_trace(
        go.Scatter(
            x=edge_x, y=edge_y, mode="lines", line=line_kw, hoverinfo="none"
        )
    )
    nx_ = [pos[n][0] for n in g.nodes()]
    ny = [pos[n][1] for n in g.nodes()]
    fig.add_trace(
        go.Scatter(
            x=nx_,
            y=ny,
            mode="markers+text",
            text=list(g.nodes()),
            textposition="top center",
            marker=dict(size=12, color="#1f77b4"),
        )
    )
    t = "Network layout (for shape only — not a geographic map)"
    fig.update_layout(title=t, showlegend=False)
    return fig


def _fig_housing_cost_mix(cost_df: pd.DataFrame) -> go.Figure:
    if cost_df.empty:
        return go.Figure()
    top = cost_df[cost_df["item"] != "Total cost of ownership"].copy()
    fig = go.Figure(
        data=[
            go.Pie(
                labels=top["item"],
                values=top["amount_sgd"],
                hole=0.45,
            )
        ]
    )
    fig.update_layout(title="Cost mix (excluding grand total)")
    return fig


def _fig_cashflow_trend(cft: pd.DataFrame) -> go.Figure:
    if cft.empty:
        return go.Figure()
    view = cft.head(240).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=view["month"],
            y=view["instalment"],
            mode="lines",
            name="Instalment",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view["month"],
            y=view["cash_used"],
            mode="lines",
            name="Cash used",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view["month"],
            y=view["cpf_used"],
            mode="lines",
            name="CPF used",
        )
    )
    fig.update_layout(
        title="Monthly loan cashflow trend",
        xaxis_title="Month",
        yaxis_title="SGD",
    )
    return fig


def _fig_profit_waterfall(pf: pd.DataFrame) -> go.Figure:
    if pf.empty:
        return go.Figure()
    vals = {
        str(r["item"]): float(r["amount_sgd"])
        for _, r in pf.iterrows()
    }
    gross = vals.get("Gross sale proceeds", 0.0)
    fee = vals.get("Sale agent fee", 0.0)
    loan = vals.get("Loan redemption", 0.0)
    user_contrib = vals.get("User contributions", 0.0)
    net = vals.get("Net proceeds after obligations", 0.0)
    profit = vals.get("Estimated profit", 0.0)
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "total", "relative", "total"],
            x=[
                "Gross sale",
                "Agent fee",
                "Loan redemption",
                "Net proceeds",
                "User contributions",
                "Estimated profit",
            ],
            y=[gross, -fee, -loan, net, -user_contrib, profit],
        )
    )
    fig.update_layout(title="Profit waterfall")
    return fig


def _fig_repricing_savings(rs: Any) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Gross interest savings", "Switch costs", "Net savings"],
            y=[rs.gross_interest_savings, -rs.total_switch_cost, rs.net_savings],
        )
    )
    fig.update_layout(title="Repricing/refinancing savings bridge", yaxis_title="SGD")
    return fig


def _fig_compare_cashflow(cft_a: pd.DataFrame, cft_b: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not cft_a.empty:
        a = cft_a.head(240)
        fig.add_trace(
            go.Scatter(
                x=a["month"],
                y=a["net_cash_outflow"],
                mode="lines",
                name="Scenario A net outflow",
            )
        )
    if not cft_b.empty:
        b = cft_b.head(240)
        fig.add_trace(
            go.Scatter(
                x=b["month"],
                y=b["net_cash_outflow"],
                mode="lines",
                name="Scenario B net outflow",
            )
        )
    fig.update_layout(
        title="Scenario comparison: monthly net cash outflow",
        xaxis_title="Month",
        yaxis_title="SGD",
    )
    return fig


def _fig_compare_summary(result_a: Any, result_b: Any) -> go.Figure:
    fig = go.Figure()
    categories = ["Estimated profit", "Net proceeds", "Total ownership cost"]
    vals_a = [
        float(result_a.profit.estimated_profit),
        float(result_a.profit.net_proceeds_after_obligations),
        float(result_a.costs.total_cost_of_ownership),
    ]
    vals_b = [
        float(result_b.profit.estimated_profit),
        float(result_b.profit.net_proceeds_after_obligations),
        float(result_b.costs.total_cost_of_ownership),
    ]
    fig.add_trace(go.Bar(name="Scenario A", x=categories, y=vals_a))
    fig.add_trace(go.Bar(name="Scenario B", x=categories, y=vals_b))
    fig.update_layout(barmode="group", title="Scenario A vs B: key outcome comparison")
    return fig


def _residential_bsd(amount: float) -> float:
    tiers = (
        (180_000.0, 0.01),
        (180_000.0, 0.02),
        (640_000.0, 0.03),
        (500_000.0, 0.04),
        (1_500_000.0, 0.05),
        (float("inf"), 0.06),
    )
    remaining = max(0.0, amount)
    duty = 0.0
    for cap, rate in tiers:
        taxable = min(remaining, cap)
        if taxable <= 0:
            break
        duty += taxable * rate
        remaining -= taxable
    return float(np.floor(max(1.0, duty))) if amount > 0 else 0.0


def _absd_rate_pct(profile: HouseholdProfile, property_count_after_buy: int) -> float:
    pcount = max(1, int(property_count_after_buy))
    if profile == HouseholdProfile.SG_SG or profile == HouseholdProfile.SINGLE_CITIZEN:
        return 0.0 if pcount <= 1 else (20.0 if pcount == 2 else 30.0)
    if profile == HouseholdProfile.SG_PR:
        return 5.0 if pcount <= 1 else (30.0 if pcount == 2 else 35.0)
    if profile == HouseholdProfile.PR_PR:
        return 5.0 if pcount <= 1 else (30.0 if pcount == 2 else 35.0)
    return 60.0


def _ehg_amount(average_income: float, household: HouseholdProfile) -> float:
    income = max(0.0, average_income)
    if household == HouseholdProfile.SINGLE_CITIZEN:
        if income > 4500:
            return 0.0
        steps = int(income // 250.0)
        return max(0.0, 60_000.0 - (steps * 2_500.0))
    if household in (HouseholdProfile.SG_SG, HouseholdProfile.SG_PR, HouseholdProfile.PR_PR):
        if income > 9000:
            return 0.0
        steps = int(income // 500.0)
        return max(0.0, 120_000.0 - (steps * 5_000.0))
    return 0.0


def _fig_debt_vs_rental_subsidy(
    cft: pd.DataFrame,
    *,
    initial_loan: float,
    rental_tax_rate_pct: float,
) -> go.Figure:
    df = cft.copy()
    tax_mult = max(0.0, min(1.0, 1.0 - (rental_tax_rate_pct / 100.0)))
    df["rental_net"] = df["rental_inflow"] * tax_mult
    df["cum_rental_net"] = df["rental_net"].cumsum()
    df["cum_cash_outflow"] = df["net_cash_outflow"].clip(lower=0.0).cumsum()
    df["cum_principal"] = df["principal"].cumsum()
    df["remaining_balance"] = (
        initial_loan - df["cum_principal"]
    ).clip(lower=0.0)
    df["debt_free_progress_pct"] = (
        (df["cum_principal"] / max(1.0, initial_loan)) * 100.0
    ).clip(lower=0.0, upper=100.0)
    df["rental_subsidy_pct"] = (
        df["cum_rental_net"] / df["cum_cash_outflow"].replace(0.0, np.nan) * 100.0
    ).fillna(0.0).clip(lower=0.0, upper=300.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["month"],
            y=df["remaining_balance"],
            mode="lines",
            name="Remaining debt (SGD)",
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["month"],
            y=df["cum_rental_net"],
            mode="lines",
            name="Cumulative net rental (SGD)",
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["month"],
            y=df["debt_free_progress_pct"],
            mode="lines",
            name="Debt-free progress (%)",
            yaxis="y2",
            line=dict(width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["month"],
            y=df["rental_subsidy_pct"],
            mode="lines",
            name="Rental subsidy of cash outflow (%)",
            yaxis="y2",
            line=dict(width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="Debt-free journey vs rental subsidy (after rental tax)",
        xaxis_title="Month",
        yaxis_title="SGD",
        yaxis2=dict(title="Progress %", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    return fig


def _inject_minimal_ui() -> None:
    st.markdown(
        """
<style>
.block-container {padding-top: 1.0rem; padding-bottom: 1.5rem; max-width: 1240px;}
h1, h2, h3 {letter-spacing: -0.02em;}
.ux-h2 {font-size: 1.25rem; font-weight: 650; margin: 0.2rem 0 0.35rem 0; color: var(--text-color);}
.ux-divider {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  margin: 1.0rem 0 0.65rem 0;
  color: var(--text-color);
}
.ux-divider .icon {
  width: 1.7rem;
  height: 1.7rem;
  border-radius: 999px;
  background: rgba(59, 130, 246, 0.18);
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
.ux-note {
  color: var(--text-color);
  border-left: 3px solid rgba(96, 165, 250, 0.6);
  background: rgba(30, 58, 138, 0.18);
  border-radius: 8px;
  padding: 0.55rem 0.75rem;
  margin: 0.3rem 0 0.75rem 0;
}
[data-testid="stMetricValue"] {font-size: 1.1rem;}
[data-testid="stDataFrame"] {border-radius: 8px;}
.ux-metric {
  border-radius: 12px;
  border: 1px solid rgba(148, 163, 184, 0.35);
  background: linear-gradient(165deg, rgba(30, 41, 59, 0.18), rgba(71, 85, 105, 0.10));
  padding: 0.75rem 0.85rem;
  min-height: 96px;
}
.ux-metric .top {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  color: var(--text-color);
  font-size: 0.85rem;
  font-weight: 620;
  margin-bottom: 0.35rem;
}
.ux-metric .val {
  color: var(--text-color);
  font-size: 1.28rem;
  font-weight: 760;
  line-height: 1.25;
}
.ux-banner {
  border-radius: 14px;
  border: 1px solid rgba(96, 165, 250, 0.45);
  background: linear-gradient(90deg, rgba(30, 64, 175, 0.25), rgba(2, 132, 199, 0.12));
  padding: 0.9rem 1rem;
  margin-bottom: 0.7rem;
}
.ux-hero {
  border-radius: 16px;
  border: 1px solid rgba(96, 165, 250, 0.45);
  background: radial-gradient(
              circle at 12% 18%, rgba(59, 130, 246, 0.24), rgba(15, 23, 42, 0.18) 45%
            ),
              linear-gradient(120deg, rgba(30, 64, 175, 0.20), rgba(15, 23, 42, 0.08));
  padding: 1.05rem 1.15rem;
  margin: 0.2rem 0 0.9rem 0;
}
.ux-hero .eyebrow {
  font-size: 0.74rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
  color: rgba(147, 197, 253, 0.96);
  margin-bottom: 0.32rem;
}
.ux-hero h3 {
  margin: 0 0 0.3rem 0;
  color: var(--text-color);
  font-size: 1.18rem;
}
.ux-hero p {
  margin: 0;
  color: var(--text-color);
  line-height: 1.35rem;
}
.ux-banner h3 {
  margin: 0 0 0.25rem 0;
  font-size: 1.1rem;
}
.ux-banner p {
  margin: 0;
  color: var(--text-color);
}
</style>
""",
        unsafe_allow_html=True,
    )


def _h2(text: str) -> None:
    st.markdown(f"<div class='ux-h2'>{text}</div>", unsafe_allow_html=True)


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


def _hero_card(title: str, subtitle: str, eyebrow: str = "Dashboard") -> None:
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


def _why(shows: str, solves: str) -> None:
    st.markdown(
        f"<div class='ux-note'><b>What this shows:</b> {shows}<br>"
        f"<b>Why this helps users:</b> {solves}</div>",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _load_bto_reference() -> pd.DataFrame:
    cols = ["project_name", "town", "flat_type", "flat_model", "lease_commence_date"]
    if _BTO_REFERENCE_PATH.exists():
        ref = pd.read_csv(_BTO_REFERENCE_PATH)
        missing = [c for c in cols if c not in ref.columns]
        if missing:
            raise ValueError(f"BTO reference missing columns: {missing}")
        return ref[cols].dropna(subset=["town", "flat_type"]).copy()

    query = quote_plus("hdb bto launch project lease commence")
    search_url = f"https://data.gov.sg/api/action/package_search?q={query}&rows=5"
    try:
        body = get_gov_client().get_json(search_url, timeout=60, use_cache=True)
    except (RuntimeError, ValueError, requests.RequestException):
        return pd.DataFrame(columns=cols)
    results = ((body or {}).get("result") or {}).get("results") or []
    for item in results:
        resources = item.get("resources") or []
        for r in resources:
            rid = str(r.get("id") or "").strip()
            if not rid:
                continue
            try:
                raw = get_gov_client().get_json(
                    HDB_DATASTORE_SEARCH,
                    params={"resource_id": rid, "limit": 10000},
                    timeout=120,
                    use_cache=True,
                )
            except (RuntimeError, ValueError, requests.RequestException):
                continue
            records = ((raw or {}).get("result") or {}).get("records") or []
            if not records:
                continue
            candidate = pd.DataFrame(records)
            alias_map = {
                "project_name": ["project_name", "project", "projecttitle", "name"],
                "town": ["town", "estate"],
                "flat_type": ["flat_type", "flat", "room_type"],
                "flat_model": ["flat_model", "model"],
                "lease_commence_date": ["lease_commence_date", "lease_commence_year", "lease_year"],
            }
            out = pd.DataFrame()
            for target, aliases in alias_map.items():
                hit = next((a for a in aliases if a in candidate.columns), None)
                if hit:
                    out[target] = candidate[hit]
                elif target == "flat_model":
                    out[target] = "MODEL A"
                else:
                    out[target] = np.nan
            out["lease_commence_date"] = pd.to_numeric(out["lease_commence_date"], errors="coerce")
            out = out.dropna(subset=["town", "flat_type", "lease_commence_date"])
            if len(out) >= 20:
                out = out[cols].copy()
                out["project_name"] = out["project_name"].fillna("BTO Project")
                return out
    return pd.DataFrame(columns=cols)


def _valid_storey_ranges(
    frame: pd.DataFrame,
    town: str,
    flat_type: str,
    flat_model: str,
) -> list[str]:
    if "storey_range" not in frame.columns:
        return []
    sub = frame.copy()
    mask = (
        (sub["town"].astype(str) == str(town))
        & (sub["flat_type"].astype(str) == str(flat_type))
        & (sub["flat_model"].astype(str) == str(flat_model))
    )
    picks = sorted(sub.loc[mask, "storey_range"].dropna().astype(str).unique().tolist())
    if picks:
        return picks
    return sorted(sub["storey_range"].dropna().astype(str).unique().tolist())


@st.cache_data(show_spinner=False, ttl=60 * 30)
def _load_forecaster_metadata(path: str) -> dict[str, Any]:
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


@st.cache_data(show_spinner=False, ttl=60 * 30)
def _future_connectivity_note_cached() -> str:
    try:
        fmrt = pd.read_csv(_ROOT / "data" / "reference" / "future_mrt_stations.csv")
    except (OSError, ValueError):
        return "Future MRT data unavailable in local reference files."
    if fmrt.empty:
        return "Future MRT data unavailable in local reference files."
    yrs = (
        pd.to_numeric(fmrt.get("earliest_opening_year"), errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    yrs = sorted(yrs)
    if not yrs:
        return (
            f"Government MRT reference loaded ({len(fmrt):,} station-name entries), "
            "but this source does not publish opening-year fields."
        )
    stations = fmrt["station_name"].dropna().astype(str).head(6).tolist()
    return (
        f"Future-connectivity reference loaded (opening years: {min(yrs)}-{max(yrs)}). "
        f"Sample planned stations: {', '.join(stations)}."
    )


@st.cache_data(show_spinner=False, ttl=60 * 30)
def _official_cbd_travel_summary_cached() -> str | None:
    p = _ROOT / "data" / "reference" / "mrt_travel_time_to_cbd.csv"
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p)
    except (OSError, ValueError):
        return None
    if d.empty or "to_cbd_median_travel_min" not in d.columns:
        return None
    vals = pd.to_numeric(d["to_cbd_median_travel_min"], errors="coerce").dropna()
    if vals.empty:
        return None
    best_idx = vals.idxmin()
    best_station = str(d.loc[best_idx, "station_name"]) if "station_name" in d.columns else "N/A"
    return (
        "Official OneMap MRT benchmark loaded: "
        f"{len(vals):,} stations, network median-to-CBD={float(vals.median()):.1f} min, "
        f"fastest station={best_station} ({float(vals.min()):.1f} min)."
    )


def _render_forecaster_v1(df: pd.DataFrame) -> None:
    _hero_card(
        "Forecaster Input Studio",
        (
            "Use guided controls to generate a decision-support estimate with uncertainty, "
            "comparables, and reliability context."
        ),
        eyebrow="Prediction Workspace",
    )
    st.markdown(
        "Prediction output is a decision-support estimate from historical data; "
        "it is not financial, valuation, or legal advice."
    )
    has_model = _FORECASTER_MODEL_PATH.exists() and _FORECASTER_META_PATH.exists()
    if not has_model:
        st.info(
            "Forecaster model not found. Build artifacts first: "
            "`python scripts/build_forecaster_v1.py --input data/raw/hdb_resale_2017_onwards.csv`"
        )
        return
    if df.empty:
        st.warning("Loaded dataframe is empty.")
        return

    def _calc_remaining_lease_years(txn_dt: pd.Timestamp, commence_year: float) -> float:
        years_elapsed = max(0.0, float(txn_dt.year) - float(commence_year))
        # HDB leases are typically 99 years; use this as a conservative default.
        return max(1.0, 99.0 - years_elapsed)

    def _recent_comparable_median(
        frame: pd.DataFrame,
        *,
        town_name: str,
        flat_type_name: str,
        area_sqm: float,
        txn_dt: pd.Timestamp,
    ) -> float | None:
        need = {"month", "town", "flat_type", "floor_area_sqm", "resale_price"}
        if not need.issubset(frame.columns):
            return None
        sub = frame.copy()
        sub["month"] = pd.to_datetime(sub["month"], errors="coerce")
        start = txn_dt - pd.DateOffset(months=24)
        mask = (
            (sub["town"].astype(str) == str(town_name))
            & (sub["flat_type"].astype(str) == str(flat_type_name))
            & sub["month"].between(start, txn_dt)
            & (sub["floor_area_sqm"].astype(float).between(area_sqm - 10.0, area_sqm + 10.0))
        )
        comp = sub.loc[mask, "resale_price"].dropna()
        if len(comp) < 5:
            return None
        return float(comp.median())

    def _nearest_comparables(
        frame: pd.DataFrame,
        *,
        town_name: str,
        flat_type_name: str,
        area_sqm: float,
        remaining_lease: float,
        txn_dt: pd.Timestamp,
        top_k: int = 15,
    ) -> pd.DataFrame:
        need = {
            "month",
            "town",
            "flat_type",
            "floor_area_sqm",
            "remaining_lease_years",
            "resale_price",
        }
        if not need.issubset(frame.columns):
            return pd.DataFrame()
        sub = frame.copy()
        sub["month"] = pd.to_datetime(sub["month"], errors="coerce")
        sub = sub.dropna(
            subset=["month", "floor_area_sqm", "remaining_lease_years", "resale_price"]
        ).copy()
        start = txn_dt - pd.DateOffset(months=36)
        sub = sub[
            (sub["month"].between(start, txn_dt))
            & (sub["town"].astype(str) == str(town_name))
            & (sub["flat_type"].astype(str) == str(flat_type_name))
        ].copy()
        if sub.empty:
            return sub
        d_area = (sub["floor_area_sqm"].astype(float) - float(area_sqm)).abs() / 10.0
        d_lease = (sub["remaining_lease_years"].astype(float) - float(remaining_lease)).abs() / 5.0
        d_time = ((txn_dt - sub["month"]).dt.days.astype(float)).abs() / 180.0
        sub["similarity_distance"] = d_area + d_lease + d_time
        cols = [
            "month",
            "town",
            "flat_type",
            "floor_area_sqm",
            "remaining_lease_years",
            "resale_price",
            "similarity_distance",
        ]
        return sub.sort_values("similarity_distance").head(top_k)[cols]

    def _confidence_score(
        *,
        comparables_count: int,
        warnings_count: int,
        interval_width_pct: float,
        segment_rmse: float | None,
        predicted: float,
    ) -> tuple[int, str]:
        score = 100.0
        if comparables_count < 5:
            score -= 25.0
        elif comparables_count < 10:
            score -= 12.0
        score -= min(25.0, float(warnings_count) * 8.0)
        score -= min(20.0, interval_width_pct / 2.0)
        if segment_rmse is not None and predicted > 0:
            score -= min(20.0, (segment_rmse / predicted) * 80.0)
        score_i = int(max(0, min(100, round(score))))
        if score_i >= 80:
            bucket = "High"
        elif score_i >= 60:
            bucket = "Moderate"
        else:
            bucket = "Low"
        return score_i, bucket

    def _fmt_seg_table(seg_df: pd.DataFrame, key_col: str) -> pd.DataFrame:
        if seg_df.empty:
            return seg_df
        out = seg_df.copy()
        if "n" in out.columns:
            out["sample_size"] = out["n"].map(lambda v: f"{int(v):,}")
        if "mae" in out.columns:
            out["avg_error_abs_sgd"] = out["mae"].map(lambda v: f"${float(v):,.2f}")
        if "rmse" in out.columns:
            out["typical_error_sgd"] = out["rmse"].map(lambda v: f"${float(v):,.2f}")
        if "mape" in out.columns:
            out["avg_error_pct"] = out["mape"].map(lambda v: f"{float(v) * 100.0:.2f}%")
        keep_cols = [
            key_col,
            "sample_size",
            "avg_error_abs_sgd",
            "typical_error_sgd",
            "avg_error_pct",
        ]
        keep = [c for c in keep_cols if c in out.columns]
        return out[keep]

    def _location_premium_context(
        frame: pd.DataFrame,
        *,
        town_name: str,
        txn_dt: pd.Timestamp,
    ) -> dict[str, float | str | None]:
        sub = frame.copy()
        if "month" in sub.columns:
            sub["month"] = pd.to_datetime(sub["month"], errors="coerce")
            sub = sub[sub["month"] <= txn_dt]
        sub = sub.dropna(subset=["resale_price"])
        if sub.empty:
            return {}
        global_med = float(sub["resale_price"].median())
        town_med = float(
            sub.loc[sub["town"].astype(str) == str(town_name), "resale_price"].median()
        )
        premium_pct = ((town_med - global_med) / max(global_med, 1.0)) * 100.0
        row_any = sub.loc[sub["town"].astype(str) == str(town_name)].head(1)
        planning_area = (
            str(row_any["planning_area"].iloc[0])
            if "planning_area" in row_any.columns and not row_any.empty
            else "Unknown"
        )
        region = (
            str(row_any["region_ocr"].iloc[0])
            if "region_ocr" in row_any.columns and not row_any.empty
            else "Unknown"
        )
        mrt_count = (
            float(row_any["mrt_station_count"].iloc[0])
            if "mrt_station_count" in row_any.columns and not row_any.empty
            else np.nan
        )
        mrt_km = (
            float(row_any["nearest_mrt_km_proxy"].iloc[0])
            if "nearest_mrt_km_proxy" in row_any.columns and not row_any.empty
            else np.nan
        )
        # Rough transit proxy for user interpretation only.
        cbd_eta = np.nan
        if pd.notna(mrt_km):
            cbd_eta = float(mrt_km) * 12.0 + (20.0 if str(region).upper() == "CCR" else 30.0)
        return {
            "global_median": global_med,
            "town_median": town_med,
            "town_premium_pct": premium_pct,
            "planning_area": planning_area,
            "region_ocr": region,
            "mrt_station_count": mrt_count if pd.notna(mrt_count) else None,
            "nearest_mrt_km_proxy": mrt_km if pd.notna(mrt_km) else None,
            "cbd_eta_min_proxy": cbd_eta if pd.notna(cbd_eta) else None,
        }

    sample = df.dropna(subset=["month"]).iloc[-1]
    bto_ref = _load_bto_reference()
    mode_col, mode_note = st.columns([1, 2])
    listing_mode = mode_col.selectbox(
        "Listing profile",
        ["Resale", "BTO"],
        help="BTO mode pre-fills names and lease fields using BTO reference rows when available.",
    )
    mode_note.caption(
        "BTO mode keeps the same model engine but assists with "
        "project/town/type naming and lease anchor."
    )

    c1, c2, c3 = st.columns(3)
    town_options = sorted(df["town"].dropna().astype(str).unique().tolist())
    town = c1.selectbox("Town", town_options)
    subset_town = df[df["town"].astype(str) == str(town)].copy()

    if listing_mode == "BTO" and not bto_ref.empty:
        bto_town = bto_ref[bto_ref["town"].astype(str) == str(town)].copy()
        project_choices = sorted(bto_town["project_name"].astype(str).unique().tolist())
        project_name = c1.selectbox(
            "BTO project",
            (
                project_choices
                if project_choices
                else sorted(bto_ref["project_name"].astype(str).unique())
            ),
        )
        bto_pick = bto_ref[bto_ref["project_name"].astype(str) == str(project_name)].copy()
        if not bto_pick.empty:
            if "flat_type" in bto_pick.columns:
                bto_ft = sorted(bto_pick["flat_type"].dropna().astype(str).unique().tolist())
            else:
                bto_ft = []
            if "flat_model" in bto_pick.columns:
                bto_fm = sorted(bto_pick["flat_model"].dropna().astype(str).unique().tolist())
            else:
                bto_fm = []
        else:
            bto_ft, bto_fm = [], []
    else:
        project_name = None
        bto_ft, bto_fm = [], []

    flat_type_options = (
        bto_ft
        if listing_mode == "BTO" and bto_ft
        else sorted(subset_town["flat_type"].dropna().astype(str).unique().tolist())
    )
    flat_type = c2.selectbox("Flat type", flat_type_options)
    subset_type = subset_town[subset_town["flat_type"].astype(str) == str(flat_type)].copy()

    model_options = (
        bto_fm
        if listing_mode == "BTO" and bto_fm
        else sorted(subset_type["flat_model"].dropna().astype(str).unique().tolist())
    )
    flat_model = c3.selectbox("Flat model", model_options)
    subset_model = subset_type[subset_type["flat_model"].astype(str) == str(flat_model)].copy()

    c4, c5, c6 = st.columns(3)
    valid_storeys = _valid_storey_ranges(df, town, flat_type, flat_model)
    default_storey = str(
        sample.get("storey_range", valid_storeys[0] if valid_storeys else "04 TO 06")
    )
    storey_range = c4.selectbox(
        "Storey range",
        valid_storeys if valid_storeys else [default_storey],
        index=(valid_storeys.index(default_storey) if default_storey in valid_storeys else 0),
        help="Dropdown-only to prevent invalid text (e.g. free-form 'ground floor').",
    )
    area_unit = c5.selectbox("Floor area unit", ["sqm", "sqft"], index=0)
    if not subset_model.empty and "floor_area_sqm" in subset_model.columns:
        area_min = float(max(20.0, np.floor(subset_model["floor_area_sqm"].quantile(0.01))))
        area_max = float(min(300.0, np.ceil(subset_model["floor_area_sqm"].quantile(0.99))))
        area_default = float(subset_model["floor_area_sqm"].median())
    else:
        area_min, area_max, area_default = 20.0, 300.0, float(sample.get("floor_area_sqm", 90.0))
    if area_max <= area_min:
        area_max = area_min + 1.0
    raw_default_area = area_default if area_unit == "sqm" else area_default * 10.7639104167
    min_area_input = area_min if area_unit == "sqm" else area_min * 10.7639104167
    max_area_input = area_max if area_unit == "sqm" else area_max * 10.7639104167
    floor_area_input = c5.slider(
        "Floor area",
        min_value=float(round(min_area_input, 1)),
        max_value=float(round(max_area_input, 1)),
        value=float(round(raw_default_area, 1)),
        step=0.1,
        help="Range auto-adjusts based on selected town + flat type + flat model.",
    )
    floor_area_sqm = (
        float(floor_area_input)
        if area_unit == "sqm"
        else float(floor_area_input) / 10.7639104167
    )
    month_seed = pd.Timestamp.now().date()
    c7, c8 = st.columns(2)
    month_val = c8.date_input("Transaction month anchor", value=month_seed)
    txn_ts = pd.Timestamp(month_val)
    max_commence = int(txn_ts.year)
    lease_seed = min(int(sample.get("lease_commence_date", 1998)), max_commence)
    if listing_mode == "BTO" and not bto_ref.empty and project_name is not None:
        bto_lease = pd.to_numeric(
            bto_ref.loc[
                bto_ref["project_name"].astype(str) == str(project_name), "lease_commence_date"
            ],
            errors="coerce",
        ).dropna()
        if not bto_lease.empty:
            lease_seed = int(min(max_commence, max(1960, int(round(float(bto_lease.median()))))))
    lease_commence_date = c6.number_input(
        "Lease commence year",
        min_value=1960,
        max_value=max_commence,
        value=int(lease_seed),
    )
    if int(lease_commence_date) > max_commence:
        st.warning(
            "Lease commence year cannot be later than transaction year; "
            "adjusted automatically."
        )
        lease_commence_date = max_commence

    auto_remaining_lease = c7.checkbox(
        "Auto-calculate remaining lease (99-year default)",
        value=True,
    )
    computed_remaining = _calc_remaining_lease_years(txn_ts, float(lease_commence_date))
    if auto_remaining_lease:
        remaining_lease_years = c7.number_input(
            "Remaining lease years",
            min_value=1.0,
            max_value=99.0,
            value=float(computed_remaining),
            disabled=True,
        )
    else:
        remaining_lease_years = c7.number_input(
            "Remaining lease years",
            min_value=1.0,
            max_value=99.0,
            value=float(
                min(max(1.0, sample.get("remaining_lease_years", computed_remaining)), 99.0)
            ),
        )

    if st.button("Predict resale price", type="primary"):
        if floor_area_sqm < area_min or floor_area_sqm > area_max:
            st.error(
                f"Selected floor area is outside expected range for {flat_type} / {flat_model}: "
                f"{area_min:.1f}-{area_max:.1f} sqm."
            )
            return
        payload = {
            "month": str(month_val),
            "town": town,
            "flat_type": flat_type,
            "flat_model": flat_model,
            "storey_range": storey_range,
            "floor_area_sqm": float(floor_area_sqm),
            "lease_commence_date": float(lease_commence_date),
            "remaining_lease_years": float(remaining_lease_years),
        }
        if listing_mode == "BTO":
            payload["listing_mode"] = "BTO"
            if project_name:
                payload["project_name"] = str(project_name)
        try:
            result = predict_with_explain(
                payload,
                model_path=_FORECASTER_MODEL_PATH,
                metadata_path=_FORECASTER_META_PATH,
            )
        except (RuntimeError, KeyError, ValueError) as ex:
            st.error(f"Forecaster inference failed: {ex}")
            return

        raw_pred = float(result["prediction"])
        comp_med = _recent_comparable_median(
            df,
            town_name=town,
            flat_type_name=flat_type,
            area_sqm=float(floor_area_sqm),
            txn_dt=txn_ts,
        )
        anchor_weight = st.slider(
            "Market anchor blend (toward recent comparable median)",
            min_value=0.0,
            max_value=0.7,
            value=0.35,
            step=0.05,
            help="Use a small blend to stabilize noisy model predictions.",
        )
        final_pred = raw_pred
        if comp_med is not None:
            final_pred = (1.0 - anchor_weight) * raw_pred + anchor_weight * comp_med
        st.metric("Predicted resale price (final)", f"${final_pred:,.0f}")
        st.caption(f"Raw model prediction: ${raw_pred:,.0f}")
        if comp_med is not None:
            st.caption(
                "Recent comparable median (24m, same town/type, +/-10sqm): "
                f"${comp_med:,.0f}"
            )
        else:
            st.caption("Recent comparable median unavailable (not enough comparable rows).")
        p10 = result["prediction_interval"]["p10"]
        p90 = result["prediction_interval"]["p90"]
        st.caption(f"Prediction interval (p10-p90): ${p10:,.0f} to ${p90:,.0f}")
        interval_width_pct = ((p90 - p10) / max(final_pred, 1.0)) * 100.0
        st.caption(f"Interval width: {interval_width_pct:.1f}% of predicted price")
        st.caption(
            "Model version: "
            f"{result.get('model_version', 'forecaster_v1')} "
            f"({result.get('model_family', 'xgboost')})"
        )
        st.caption(
            "Lease time-decay coefficient (historical fit): "
            f"{float(result.get('lease_decay_lambda', 0.0)):.4f}"
        )
        st.caption(
            "Interpretation: this is an empirical decay factor, not a strict "
            "rule that leasehold value goes to zero."
        )
        st.caption(
            "Forecast mode tip: pick a future transaction anchor date first, "
            "then set lease commence year up to that forecast year."
        )
        if result.get("warnings"):
            for w in result["warnings"]:
                st.warning(w)
        comparables_df = _nearest_comparables(
            df,
            town_name=town,
            flat_type_name=flat_type,
            area_sqm=float(floor_area_sqm),
            remaining_lease=float(remaining_lease_years),
            txn_dt=txn_ts,
            top_k=15,
        )
        metadata = _load_forecaster_metadata(str(_FORECASTER_META_PATH))
        seg_rmse = None
        try:
            tms = (
                metadata.get("metrics", {})
                .get("segments", {})
                .get("town_metrics", [])
            )
            hit = next((r for r in tms if str(r.get("town")) == str(town)), None)
            seg_rmse = float(hit.get("rmse")) if hit else None
        except (TypeError, ValueError):
            seg_rmse = None
        conf_score, conf_bucket = _confidence_score(
            comparables_count=int(len(comparables_df)),
            warnings_count=int(len(result.get("warnings", []))),
            interval_width_pct=float(interval_width_pct),
            segment_rmse=seg_rmse,
            predicted=float(final_pred),
        )
        c_conf, c_cmp, c_rmse = st.columns(3)
        with c_conf:
            _metric_card("Reliability score", f"{conf_score}/100 ({conf_bucket})", "🛡️")
        with c_cmp:
            _metric_card("Comparable rows", f"{int(len(comparables_df))}", "📚")
        with c_rmse:
            _metric_card(
                "Town RMSE (metadata)",
                f"${seg_rmse:,.0f}" if seg_rmse else "N/A",
                "📏",
            )
        if conf_score < 60:
            st.error("Low reliability for this profile. Treat estimate as directional only.")
        elif conf_score < 80:
            st.warning("Moderate reliability. Cross-check with comparables and interval.")
        else:
            st.success("High reliability relative to current model coverage.")
        with st.expander("Segment error slices (from training metadata)", expanded=False):
            seg = metadata.get("metrics", {}).get("segments", {})
            tdf = pd.DataFrame(seg.get("town_metrics", []))
            fdf = pd.DataFrame(seg.get("flat_type_metrics", []))
            if not tdf.empty:
                st.markdown("**Town-level error slices**")
                st.caption(
                    "How to read: larger sample_size is more statistically reliable. "
                    "Lower avg/typical error is better."
                )
                tdf_fmt = _fmt_seg_table(tdf.sort_values("rmse", ascending=False), "town")
                st.dataframe(tdf_fmt, width="stretch", height=220)
            if not fdf.empty:
                st.markdown("**Flat-type error slices**")
                st.caption(
                    "Like FPS benchmarks: compare rows directly. For errors, lower is better; "
                    "for sample_size, higher is better."
                )
                fdf_fmt = _fmt_seg_table(fdf.sort_values("rmse", ascending=False), "flat_type")
                st.dataframe(fdf_fmt, width="stretch", height=180)
        _section_divider("Top contributors (SHAP, log-price scale)", icon="🎯")
        st.caption(
            "Interpretation: positive SHAP pushes price up, negative SHAP pushes price down. "
            "Magnitude means influence strength; it is not a 'good vs bad' score."
        )
        contrib_df = pd.DataFrame(result["top_contributors"])
        if not contrib_df.empty:
            contrib_df["effect"] = contrib_df["shap_log_price"].map(
                lambda v: "Upward pressure" if float(v) >= 0 else "Downward pressure"
            )
        st.dataframe(contrib_df, width="stretch")
        st.caption(
            "A higher price is not universally better: sellers may prefer it, "
            "buyers usually do not."
        )

        _section_divider("Graph analytics context", icon="📉")
        st.caption(
            "Town trend view with short forecast plus where the final predicted price sits "
            "relative to recent town medians."
        )
        st.plotly_chart(
            fig_town_median_lines_with_forecast(df, [town], horizon=6),
            width="stretch",
        )
        town_slice = (
            df.loc[df["town"].astype(str) == str(town), "resale_price"]
            .dropna()
            .astype(float)
        )
        if len(town_slice) >= 10:
            pct = (town_slice <= final_pred).mean() * 100.0
            st.caption(f"Predicted price percentile within {town} sample: {pct:.1f}th percentile.")
        else:
            st.caption("Not enough town rows for robust percentile context.")

        _section_divider("Location, connectivity, and policy premium context", icon="🚇")
        loc = _location_premium_context(df, town_name=town, txn_dt=txn_ts)
        if loc:
            l1, l2, l3 = st.columns(3)
            l1.metric(
                "Town median premium vs whole dataset",
                f"{float(loc.get('town_premium_pct', 0.0)):+.2f}%",
            )
            l2.metric("Planning area", str(loc.get("planning_area", "Unknown")))
            l3.metric("Region profile", str(loc.get("region_ocr", "Unknown")))
            st.caption(
                f"Town median=${float(loc.get('town_median', 0.0)):,.2f}, "
                f"Dataset median=${float(loc.get('global_median', 0.0)):,.2f}."
            )
            if loc.get("nearest_mrt_km_proxy") is not None:
                st.caption(
                    "MRT access proxy: nearest station distance "
                    f"{float(loc['nearest_mrt_km_proxy']):.2f} km; "
                    f"station-count proxy={float(loc.get('mrt_station_count') or 0):.0f}."
                )
            if loc.get("cbd_eta_min_proxy") is not None:
                st.caption(
                    "Estimated transit-to-CBD proxy (Raffles Place/City Hall corridor): "
                    f"{float(loc['cbd_eta_min_proxy']):.1f} mins (heuristic)."
                )
        st.caption(_future_connectivity_note_cached())
        official_tt = _official_cbd_travel_summary_cached()
        if official_tt:
            st.caption(official_tt)

        with st.expander("Policy + lifestyle overlays (scenario adjustment)", expanded=False):
            st.caption(
                "These overlays are optional user assumptions (not model-learned coefficients). "
                "Use them to scenario-test school/shopping/CBD preference premiums."
            )
            near_popular_school = st.checkbox(
                "Within 1km of high-demand primary school zone", value=False
            )
            near_major_mall = st.checkbox(
                "Close to major shopping hub (e.g., NEX/JEM/PLQ)",
                value=False,
            )
            c5a, c5b = st.columns(2)
            school_premium_pct = c5a.slider("School-zone premium (%)", 0.0, 15.0, 4.0, 0.5)
            mall_premium_pct = c5b.slider("Shopping-convenience premium (%)", 0.0, 10.0, 2.0, 0.5)
            cbd_target = st.slider("CBD access target threshold (minutes)", 10, 45, 20, 1)
            cdb_access_bonus = st.slider("CBD-fast-access premium (%)", 0.0, 10.0, 2.0, 0.5)
            scenario_mult = 1.0
            if near_popular_school:
                scenario_mult += school_premium_pct / 100.0
            if near_major_mall:
                scenario_mult += mall_premium_pct / 100.0
            cbd_eta = (
                float(loc.get("cbd_eta_min_proxy"))
                if loc and loc.get("cbd_eta_min_proxy") is not None
                else None
            )
            if cbd_eta is not None and cbd_eta <= float(cbd_target):
                scenario_mult += cdb_access_bonus / 100.0
            scenario_price = float(final_pred) * scenario_mult
            st.metric("Scenario-adjusted price (overlay)", f"${scenario_price:,.2f}")
            st.caption(
                "MOE distance-priority concept reminder (general): homes nearer to schools "
                "may face stronger demand pressure; exact balloting phases require "
                "official MOE updates."
            )

        with st.expander("BTO feature diagnostics (inputs used by model)", expanded=False):
            diag_row = pd.DataFrame(
                [{"month": pd.to_datetime(str(month_val), errors="coerce"), "town": str(town)}]
            )
            diag_row["year"] = diag_row["month"].dt.year
            diag = add_bto_reference_features(diag_row)
            bto_cols = [
                "bto_launch_count_town_3y",
                "bto_avg_price_range_mid_town_3y",
                "bto_under_construction_units_town",
                "bto_completed_units_town",
            ]
            if all(c in diag.columns for c in bto_cols):
                bto_tbl = pd.DataFrame(
                    {
                        "feature": bto_cols,
                        "value_used": [float(diag.loc[0, c]) for c in bto_cols],
                    }
                )
                st.dataframe(bto_tbl, width="stretch", height=170)
                st.caption(
                    "These town/year context features are computed from local BTO reference "
                    "files under `data/reference` and passed to the forecaster input vector."
                )
            else:
                st.info(
                    "BTO diagnostics unavailable. Refresh with "
                    "`hdb-bto-download -o data/reference`."
                )

        _section_divider("Comparable transactions (most similar)", icon="🧮")
        if comparables_df.empty:
            st.info("No close comparables found for the selected profile and time window.")
        else:
            q1 = float(comparables_df["resale_price"].quantile(0.25))
            q3 = float(comparables_df["resale_price"].quantile(0.75))
            med_cmp = float(comparables_df["resale_price"].median())
            st.caption(
                f"Comparable median=${med_cmp:,.0f}, IQR=${q1:,.0f}-${q3:,.0f}. "
                "Lower distance means more similar."
            )
            st.dataframe(comparables_df, width="stretch", height=260)

        _section_divider("Lease age vs median resale (historical)", icon="⌛")
        st.caption(
            "This chart helps sanity-check how resale levels move with lease age "
            "in the loaded dataset. It is descriptive, not a causal curve."
        )
        if "lease_age_years" not in df.columns and "remaining_lease_years" in df.columns:
            lease_df = df.copy()
            lease_df["lease_age_years"] = (99.0 - lease_df["remaining_lease_years"]).clip(
                lower=0.0, upper=99.0
            )
        else:
            lease_df = df.copy()
        if {"lease_age_years", "resale_price"}.issubset(lease_df.columns):
            cur = lease_df[["lease_age_years", "resale_price"]].dropna().copy()
            if len(cur) >= 30:
                cur["lease_age_bin"] = (cur["lease_age_years"] // 5) * 5
                med = (
                    cur.groupby("lease_age_bin", as_index=False)["resale_price"]
                    .median()
                    .rename(columns={"resale_price": "median_resale_price"})
                )
                fig_decay = go.Figure()
                fig_decay.add_trace(
                    go.Scatter(
                        x=med["lease_age_bin"],
                        y=med["median_resale_price"],
                        mode="lines+markers",
                        name="Median resale by lease-age bin",
                    )
                )
                fig_decay.update_layout(
                    title="Historical lease-age profile (5-year bins)",
                    xaxis_title="Lease age (years, binned)",
                    yaxis_title="Median resale price (SGD)",
                )
                st.plotly_chart(fig_decay, width="stretch")
                st.caption(
                    f"Fitted lease decay coefficient in model: "
                    f"{float(result.get('lease_decay_lambda', 0.0)):.4f}"
                )
            else:
                st.caption("Not enough rows for stable lease-age chart.")
        else:
            st.caption("Lease-age fields unavailable in current dataframe.")
        st.markdown("### Feedback loop")
        st.caption(
            "Submit feedback on this prediction to improve future model iterations. "
            "Feedback is written to local CSV configured in forecaster_v1.yaml."
        )
        with st.form("forecaster_feedback_form", clear_on_submit=True):
            rating = st.slider("Prediction quality rating", 1, 5, 3)
            actual = st.number_input(
                "Actual transaction price (optional, if known)",
                min_value=0.0,
                value=0.0,
                step=1000.0,
            )
            comment = st.text_area("Comment (optional)")
            submitted = st.form_submit_button("Submit feedback")
            if submitted:
                try:
                    cfg = load_forecaster_config(_FORECASTER_CFG_PATH)
                    out_path = append_feedback(
                        store_path=cfg.feedback_store_path,
                        model_version=result.get("model_version", "forecaster_v1"),
                        model_family=result.get("model_family", "unknown"),
                        predicted_price=float(result["prediction"]),
                        user_rating=int(rating),
                        user_comment=comment,
                        input_payload=payload,
                        actual_price=float(actual) if actual > 0 else None,
                    )
                    st.success(f"Feedback saved to `{out_path}`")
                except (OSError, ValueError, RuntimeError) as ex:
                    st.error(f"Failed to save feedback: {ex}")


def _render_housing_economics_details() -> None:
    _hero_card(
        "HDB Calculator",
        (
            "Plan full ownership economics including grants, levy, cashflow, "
            "and estimated exit outcome."
        ),
        eyebrow="BTO / Resale / EC / Private Estimator",
    )
    if not _HOUSING_FINANCE_CFG_PATH.exists():
        st.warning(f"Missing policy config: `{_HOUSING_FINANCE_CFG_PATH}`")
        return
    defaults = load_policy_defaults(_HOUSING_FINANCE_CFG_PATH)
    st.caption(
        f"Policy defaults effective `{defaults.effective_date}`; "
        f"last updated `{defaults.updated_on}`. "
        f"{defaults.disclaimer}"
    )
    c1, c2, c3 = st.columns(3)
    housing_type = HousingType(
        c1.selectbox(
            "Housing type",
            [t.value for t in HousingType],
            index=0,
        )
    )
    household_profile = HouseholdProfile(
        c2.selectbox(
            "Couple / household status",
            [p.value for p in HouseholdProfile],
            index=0,
            help="Examples: sg_sg, sg_pr, pr_pr.",
        )
    )
    loan_type = LoanType(
        c3.selectbox(
            "Loan option",
            [t.value for t in LoanType],
            index=0,
            help="HDB fixed track or bank rate tracks.",
        )
    )
    p1, p2, p3 = st.columns(3)
    purchase_price = p1.number_input(
        "Purchase price (SGD)",
        min_value=1.0,
        value=550000.0,
        step=1000.0,
    )
    expected_sale_price = p2.number_input(
        "Expected sale price (SGD)",
        min_value=1.0,
        value=750000.0,
        step=1000.0,
    )
    years_to_sell = p3.number_input("Years to estimated sale", min_value=0.1, value=8.0, step=0.5)
    hpol1, hpol2, hpol3 = st.columns(3)
    housing_policy_tier = hpol1.selectbox(
        "Flat policy tier",
        ("standard", "plus", "prime"),
        index=0,
        help="Standard usually has 5-year MOP, Plus/Prime usually 10-year MOP.",
    )
    default_mop = 5.0 if housing_policy_tier == "standard" else 10.0
    default_subsidy_recovery_pct = (
        0.0 if housing_policy_tier == "standard" else (6.0 if housing_policy_tier == "plus" else 9.0)
    )
    t1, t2, t3 = st.columns(3)
    mop_years = t1.number_input("MOP years", min_value=0.0, value=default_mop, step=0.5)
    wait_years_to_keys = t2.number_input("Wait to keys (years)", min_value=0.0, value=4.0, step=0.5)
    loan_tenure_years = int(
        t3.number_input("Loan tenure (years)", min_value=1.0, max_value=35.0, value=25.0, step=1.0)
    )
    f1, f2, f3 = st.columns(3)
    downpayment_pct = f1.slider("Downpayment %", min_value=0.0, max_value=0.8, value=0.2, step=0.01)
    annual_interest_rate_pct = f2.number_input(
        "Annual interest rate (%)",
        min_value=0.0,
        value=(
            defaults.hdb_loan_rate_pct
            if loan_type == LoanType.HDB
            else defaults.bank_fixed_rate_pct
        ),
        step=0.05,
    )
    cpf_oa_monthly_available = f3.number_input(
        "CPF OA available monthly (SGD)",
        min_value=0.0,
        value=1200.0,
        step=50.0,
    )
    bank_segments: tuple[RateSegment, ...] = tuple()
    if loan_type in (LoanType.BANK_FIXED, LoanType.BANK_FLOATING):
        st.markdown("#### Bank loan structure")
        st.caption(
            "Cross-check note: MAS defines/publishes compounded SORA benchmarks, and major "
            "Singapore banks offer fixed and SORA-pegged mortgage packages."
        )
        b1, b2, b3 = st.columns(3)
        use_fixed_then_sora = b1.checkbox(
            "Use 2-year fixed then SORA",
            value=(loan_type == LoanType.BANK_FIXED),
            help=(
                "Common structure in Singapore bank packages: lock fixed period, "
                "then SORA + spread."
            ),
        )
        fixed_period_months = int(
            b1.number_input(
                "Fixed period (months)",
                min_value=0.0,
                max_value=float(loan_tenure_years * 12),
                value=float(defaults.bank_fixed_period_months),
                step=1.0,
            )
        )
        sora_rate_pct = b2.number_input(
            "SORA base rate (%)",
            min_value=0.0,
            value=defaults.bank_floating_base_rate_pct,
            step=0.05,
        )
        sora_spread_pct = b3.number_input(
            "SORA spread (%)",
            min_value=0.0,
            value=defaults.bank_sora_spread_pct,
            step=0.05,
        )
        if use_fixed_then_sora:
            bank_segments = make_fixed_then_sora_segments(
                tenure_years=loan_tenure_years,
                fixed_period_months=fixed_period_months,
                fixed_rate_pct=float(annual_interest_rate_pct),
                sora_rate_pct=float(sora_rate_pct),
                sora_spread_pct=float(sora_spread_pct),
            )
            if bank_segments:
                st.caption(
                    "Active rate segments: "
                    + " -> ".join(
                        f"{seg.months}m @ {seg.annual_rate_pct:.2f}%"
                        for seg in bank_segments
                    )
                )
    cst1, cst2, cst3, cst4 = st.columns(4)
    cov_amount = cst1.number_input(
        "COV / premium cash (SGD)",
        min_value=0.0,
        value=0.0,
        step=1000.0,
    )
    renovation_cost = cst2.number_input(
        "Renovation cost (SGD)",
        min_value=0.0,
        value=45000.0,
        step=1000.0,
    )
    monthly_other_costs = cst3.number_input(
        "Monthly non-loan costs (SGD)",
        min_value=0.0,
        value=defaults.maintenance_monthly_default,
        step=10.0,
    )
    monthly_rental_income = cst4.number_input(
        "Monthly rental inflow (SGD)",
        min_value=0.0,
        value=0.0,
        step=100.0,
        help="Set >0 only when legally allowed by your occupancy scenario.",
    )
    rental_tax_rate_pct = cst4.number_input(
        "Effective rental tax/expense rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        help="Used only for the debt-free vs rental subsidy graph.",
    )
    if housing_policy_tier in {"plus", "prime"}:
        st.caption(
            "Policy note: Plus/Prime flats usually have stricter rental rules (no whole-unit "
            "rental after MOP). Set rental inflow conservatively."
        )
    fee1, fee2, fee3, fee4 = st.columns(4)
    legal_fees = fee1.number_input(
        "Legal fees (SGD)",
        min_value=0.0,
        value=defaults.legal_fees_default,
        step=100.0,
    )
    valuation_fees = fee2.number_input(
        "Valuation/admin fees (SGD)",
        min_value=0.0,
        value=defaults.valuation_fees_default,
        step=50.0,
    )
    bsd_auto = _residential_bsd(float(purchase_price))
    buyer_stamp_duty = fee3.number_input(
        "Buyer's Stamp Duty (SGD)",
        min_value=0.0,
        value=bsd_auto,
        step=100.0,
        help="IRAS residential BSD tier table is used for the default value.",
    )
    prop_count = int(
        fee4.number_input(
            "Residential property count after purchase",
            min_value=1.0,
            max_value=6.0,
            value=1.0,
            step=1.0,
        )
    )
    absd_rate_pct = _absd_rate_pct(household_profile, prop_count)
    additional_buyer_stamp_duty = fee4.number_input(
        "Additional Buyer's Stamp Duty (SGD)",
        min_value=0.0,
        value=float(purchase_price) * absd_rate_pct / 100.0,
        step=100.0,
        help=f"Default uses profile/property-count ABSD estimate at {absd_rate_pct:.1f}%.",
    )
    gov1, gov2 = st.columns(2)
    resale_levy_amount = gov1.number_input(
        "Levy estimate (SGD)",
        min_value=0.0,
        value=defaults.resale_levy_by_housing_type.get(housing_type.value, 0.0),
        step=1000.0,
    )
    subsidy_recovery_pct = gov1.number_input(
        "Subsidy recovery on first resale (%)",
        min_value=0.0,
        max_value=25.0,
        value=default_subsidy_recovery_pct,
        step=0.5,
        help="Standard=0 by default. Plus/Prime defaults are editable assumptions.",
    )
    gov_return_extra_amount = gov2.number_input(
        "Other return to government (SGD, incl subsidy recovery)",
        min_value=0.0,
        value=float(expected_sale_price) * subsidy_recovery_pct / 100.0,
        step=1000.0,
    )
    st.markdown("#### Repricing / refinancing behavior")
    r1, r2, r3, r4 = st.columns(4)
    enable_repricing = r1.checkbox(
        "Enable repricing/refinancing simulation",
        value=False,
        help=(
            "Cross-verified note: banks often impose lock-in prepayment charges and "
            "repricing/refinancing processing costs."
        ),
    )
    repricing_month = int(
        r2.number_input(
            "Switch month",
            min_value=1.0,
            max_value=float(loan_tenure_years * 12),
            value=25.0,
            step=1.0,
            disabled=not enable_repricing,
        )
    )
    repricing_target_rate_pct = r3.number_input(
        "New package effective rate (%)",
        min_value=0.0,
        value=max(0.1, defaults.bank_floating_base_rate_pct + defaults.bank_sora_spread_pct),
        step=0.05,
        disabled=not enable_repricing,
    )
    lock_in_months = int(
        r4.number_input(
            "Current package lock-in (months)",
            min_value=0.0,
            max_value=float(loan_tenure_years * 12),
            value=float(defaults.bank_fixed_period_months),
            step=1.0,
            disabled=not enable_repricing,
        )
    )
    rf1, rf2, rf3, rf4 = st.columns(4)
    repricing_admin_fee = rf1.number_input(
        "Repricing admin fee (SGD)",
        min_value=0.0,
        value=500.0,
        step=50.0,
        disabled=not enable_repricing,
        help=(
            "OCBC publicly states a one-time processing fee of S$500 unless "
            "fee-free switch applies."
        ),
    )
    refinancing_legal_fee = rf2.number_input(
        "Refinancing legal fee (SGD)",
        min_value=0.0,
        value=2000.0,
        step=100.0,
        disabled=not enable_repricing,
    )
    refinancing_valuation_fee = rf3.number_input(
        "Refinancing valuation fee (SGD)",
        min_value=0.0,
        value=300.0,
        step=50.0,
        disabled=not enable_repricing,
    )
    early_repayment_penalty_pct = rf4.number_input(
        "Early prepayment penalty (%)",
        min_value=0.0,
        max_value=10.0,
        value=1.5,
        step=0.1,
        disabled=not enable_repricing,
        help="Model as % of outstanding loan when switching before lock-in expiry.",
    )
    clawback_fee = st.number_input(
        "Subsidy clawback fee (SGD)",
        min_value=0.0,
        value=0.0,
        step=100.0,
        disabled=not enable_repricing,
        help="Some refinancing subsidies may be clawed back if redeemed early.",
    )
    st.markdown("#### Grants taken checklist")
    g1, g2, g3 = st.columns(3)
    avg_monthly_income = g1.number_input(
        "Average gross monthly household income (SGD)",
        min_value=0.0,
        value=7000.0,
        step=100.0,
    )
    ehg_amt = _ehg_amount(float(avg_monthly_income), household_profile)
    include_family_grant = g2.checkbox(
        "Include CPF Family / Singles Grant (resale)",
        value=True,
    )
    include_phg = g3.checkbox(
        "Include Proximity Housing Grant (resale)",
        value=False,
    )
    live_with_family = g3.checkbox("Live with parent/child (PHG higher tier)", value=False)
    include_citizen_topup = g3.checkbox("Include Citizen Top-Up Grant", value=False)

    grant_rows: list[GrantSelection] = []
    if ehg_amt > 0:
        grant_rows.append(GrantSelection(name="EHG (income-tiered)", amount=ehg_amt, selected=True))
    if include_family_grant and housing_type == HousingType.RESALE:
        if household_profile == HouseholdProfile.SG_SG:
            fg = 80_000.0 if purchase_price <= 750_000 else 50_000.0
            grant_rows.append(GrantSelection(name="Family Grant", amount=fg, selected=True))
        elif household_profile == HouseholdProfile.SG_PR:
            fg = 70_000.0 if purchase_price <= 750_000 else 40_000.0
            grant_rows.append(
                GrantSelection(name="Family Grant (Citizen-PR)", amount=fg, selected=True)
            )
        elif household_profile == HouseholdProfile.SINGLE_CITIZEN:
            sg = 40_000.0 if purchase_price <= 750_000 else 25_000.0
            grant_rows.append(GrantSelection(name="Singles Grant", amount=sg, selected=True))
    if include_phg and housing_type == HousingType.RESALE:
        if household_profile == HouseholdProfile.SINGLE_CITIZEN:
            grant_rows.append(
                GrantSelection(
                    name="Proximity Housing Grant (Singles)",
                    amount=15_000.0 if live_with_family else 10_000.0,
                    selected=True,
                )
            )
        else:
            grant_rows.append(
                GrantSelection(
                    name="Proximity Housing Grant",
                    amount=30_000.0 if live_with_family else 20_000.0,
                    selected=True,
                )
            )
    if include_citizen_topup and household_profile == HouseholdProfile.SG_PR:
        grant_rows.append(GrantSelection(name="Citizen Top-Up Grant (estimate)", amount=10_000.0))
    if not grant_rows:
        st.caption("No grants selected for this scenario.")
    else:
        st.caption("Grant assumptions shown below are editable policy estimates.")
        gc = st.columns(min(3, len(grant_rows)))
        for i, g in enumerate(grant_rows):
            col = gc[i % len(gc)]
            col.checkbox(
                f"{g.name} (${g.amount:,.0f})",
                value=g.selected,
                key=f"grant_{i}_{g.name}",
                disabled=True,
            )

    run_clicked = st.button("Compute housing economics", type="primary")
    if not run_clicked:
        return
    scenario = HousingFinanceScenario(
        housing_type=housing_type,
        household_profile=household_profile,
        purchase_price=float(purchase_price),
        expected_sale_price=float(expected_sale_price),
        years_to_sell=float(years_to_sell),
        mop_years=float(mop_years),
        wait_years_to_keys=float(wait_years_to_keys),
        cov_amount=float(cov_amount),
        renovation_cost=float(renovation_cost),
        monthly_other_costs=float(monthly_other_costs),
        monthly_rental_income=float(monthly_rental_income),
        use_hdb_loan=(loan_type == LoanType.HDB),
        loan_type=loan_type,
        loan_tenure_years=loan_tenure_years,
        downpayment_pct=float(downpayment_pct),
        annual_interest_rate_pct=float(annual_interest_rate_pct),
        bank_rate_segments=bank_segments,
        cpf_oa_monthly_available=float(cpf_oa_monthly_available),
        grants=tuple(grant_rows),
        resale_levy_amount=float(resale_levy_amount),
        gov_return_extra_amount=float(gov_return_extra_amount),
        legal_fees=float(legal_fees),
        valuation_fees=float(valuation_fees),
        buyer_stamp_duty=float(buyer_stamp_duty),
        additional_buyer_stamp_duty=float(additional_buyer_stamp_duty),
        enable_repricing=bool(enable_repricing),
        repricing_month=int(repricing_month),
        repricing_target_rate_pct=float(repricing_target_rate_pct),
        repricing_admin_fee=float(repricing_admin_fee),
        refinancing_legal_fee=float(refinancing_legal_fee),
        refinancing_valuation_fee=float(refinancing_valuation_fee),
        lock_in_months=int(lock_in_months),
        early_repayment_penalty_pct=float(early_repayment_penalty_pct) / 100.0,
        clawback_fee=float(clawback_fee),
    )
    result = run_housing_finance(scenario, defaults)
    st.markdown("### Scenario comparison (A vs B)")
    enable_compare = st.checkbox(
        "Enable Scenario B overlay",
        value=False,
        help="Runs a second scenario using modified assumptions for direct comparison.",
    )
    result_b = None
    if enable_compare:
        cb1, cb2, cb3, cb4 = st.columns(4)
        b_rate_shift = cb1.number_input(
            "Scenario B: rate shift (pp)",
            min_value=-5.0,
            max_value=5.0,
            value=0.5,
            step=0.05,
        )
        b_sale_shift_pct = cb2.number_input(
            "Scenario B: sale price shift (%)",
            min_value=-50.0,
            max_value=100.0,
            value=5.0,
            step=0.5,
        )
        b_reno_delta = cb3.number_input(
            "Scenario B: renovation delta (SGD)",
            min_value=-200000.0,
            max_value=500000.0,
            value=0.0,
            step=1000.0,
        )
        b_switch_month = int(
            cb4.number_input(
                "Scenario B: repricing month override",
                min_value=1.0,
                max_value=float(loan_tenure_years * 12),
                value=float(repricing_month),
                step=1.0,
                disabled=not enable_repricing,
            )
        )
        sale_b = float(expected_sale_price) * (1.0 + float(b_sale_shift_pct) / 100.0)
        rate_b = max(0.0, float(annual_interest_rate_pct) + float(b_rate_shift))
        scenario_b = HousingFinanceScenario(
            housing_type=housing_type,
            household_profile=household_profile,
            purchase_price=float(purchase_price),
            expected_sale_price=sale_b,
            years_to_sell=float(years_to_sell),
            mop_years=float(mop_years),
            wait_years_to_keys=float(wait_years_to_keys),
            cov_amount=float(cov_amount),
            renovation_cost=max(0.0, float(renovation_cost) + float(b_reno_delta)),
            monthly_other_costs=float(monthly_other_costs),
            monthly_rental_income=float(monthly_rental_income),
            use_hdb_loan=(loan_type == LoanType.HDB),
            loan_type=loan_type,
            loan_tenure_years=loan_tenure_years,
            downpayment_pct=float(downpayment_pct),
            annual_interest_rate_pct=rate_b,
            bank_rate_segments=bank_segments,
            cpf_oa_monthly_available=float(cpf_oa_monthly_available),
            grants=tuple(grant_rows),
            resale_levy_amount=float(resale_levy_amount),
            gov_return_extra_amount=float(gov_return_extra_amount),
            legal_fees=float(legal_fees),
            valuation_fees=float(valuation_fees),
            buyer_stamp_duty=float(buyer_stamp_duty),
            additional_buyer_stamp_duty=float(additional_buyer_stamp_duty),
            enable_repricing=bool(enable_repricing),
            repricing_month=int(b_switch_month),
            repricing_target_rate_pct=float(repricing_target_rate_pct),
            repricing_admin_fee=float(repricing_admin_fee),
            refinancing_legal_fee=float(refinancing_legal_fee),
            refinancing_valuation_fee=float(refinancing_valuation_fee),
            lock_in_months=int(lock_in_months),
            early_repayment_penalty_pct=float(early_repayment_penalty_pct) / 100.0,
            clawback_fee=float(clawback_fee),
        )
        result_b = run_housing_finance(scenario_b, defaults)
    _section_divider("Timeline & Eligibility", icon="📅")
    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    tcol1.metric("Years to MOP", f"{result.timeline.years_to_mop:.1f}")
    tcol2.metric("Earliest sale year", f"{result.timeline.earliest_sale_year_from_purchase:.1f}")
    tcol3.metric("Your sale year", f"{result.timeline.years_until_estimated_sale:.1f}")
    tcol4.metric("MOP satisfied", "Yes" if result.timeline.mop_satisfied_by_sale else "No")
    if result.eligibility.messages:
        for msg in result.eligibility.messages:
            st.warning(msg)
    _section_divider("Return to Government", icon="🏛️")
    gov_tbl = government_return_table(result)
    st.dataframe(gov_tbl, width="stretch")
    st.plotly_chart(_fig_housing_cost_mix(gov_tbl), width="stretch")
    _section_divider("All Costs Itemized", icon="🧾")
    cost_tbl = itemized_cost_table(result)
    st.dataframe(cost_tbl, width="stretch")
    st.plotly_chart(_fig_housing_cost_mix(cost_tbl), width="stretch")
    _section_divider("Profit & Proceeds Breakdown", icon="💰")
    pcol1, pcol2, pcol3 = st.columns(3)
    pcol1.metric("Estimated profit", f"${result.profit.estimated_profit:,.0f}")
    pcol2.metric("Net proceeds", f"${result.profit.net_proceeds_after_obligations:,.0f}")
    pcol3.metric("Annualized profit rate", f"{result.profit.annualized_profit_rate_pct:.2f}%")
    profit_tbl = profit_breakdown_table(result)
    st.dataframe(profit_tbl, width="stretch")
    st.plotly_chart(_fig_profit_waterfall(profit_tbl), width="stretch")
    _section_divider("Instalment Cash Flow (Inflow/Outflow)", icon="📊")
    cft = cashflow_table(result, max_rows=360)
    st.dataframe(cft, width="stretch", height=360)
    if not cft.empty:
        st.plotly_chart(_fig_cashflow_trend(cft), width="stretch")
        cpf_topup = (cft["instalment"] - cft["cpf_used"]).clip(lower=0.0).sum()
        st.caption(
            "Estimated cash top-up to CPF-funded instalments over shown "
            f"schedule: ${cpf_topup:,.2f}."
        )
        _section_divider("Debt-Free Progress vs Rental Subsidy", icon="🧭")
        initial_loan = float(purchase_price) * (1.0 - float(downpayment_pct))
        st.plotly_chart(
            _fig_debt_vs_rental_subsidy(
                cft,
                initial_loan=initial_loan,
                rental_tax_rate_pct=float(rental_tax_rate_pct),
            ),
            width="stretch",
        )
    st.caption(
        "Loan interest-rate track used: "
        f"{scenario.loan_type.value}; base={scenario.annual_interest_rate_pct:.2f}% "
        f"with {len(result.scenario.bank_rate_segments)} segment(s)."
    )
    if result_b is not None:
        _section_divider("Scenario A vs B Visual Comparison", icon="🆚")
        st.plotly_chart(
            _fig_compare_summary(result, result_b),
            width="stretch",
        )
        cft_a = cashflow_table(result, max_rows=360)
        cft_b = cashflow_table(result_b, max_rows=360)
        st.plotly_chart(_fig_compare_cashflow(cft_a, cft_b), width="stretch")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric(
            "Profit delta (B - A)",
            f"${(result_b.profit.estimated_profit - result.profit.estimated_profit):,.0f}",
        )
        ownership_delta = (
            result_b.costs.total_cost_of_ownership - result.costs.total_cost_of_ownership
        )
        cc2.metric(
            "Ownership cost delta (B - A)",
            f"${ownership_delta:,.0f}",
        )
        if result.repricing is not None and result_b.repricing is not None:
            cc3.metric(
                "Net savings delta (B - A)",
                f"${(result_b.repricing.net_savings - result.repricing.net_savings):,.0f}",
            )
    if result.repricing is not None:
        _section_divider("Repricing / Refinancing Savings", icon="🔁")
        rs = result.repricing
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Gross interest savings", f"${rs.gross_interest_savings:,.0f}")
        rc2.metric("Total switch costs", f"${rs.total_switch_cost:,.0f}")
        rc3.metric("Net savings", f"${rs.net_savings:,.0f}")
        st.plotly_chart(_fig_repricing_savings(rs), width="stretch")
        if rs.net_savings > 0 and rs.gross_interest_savings > 0:
            breakeven_ratio = min(1.0, rs.total_switch_cost / rs.gross_interest_savings)
            breakeven_month = max(1, int(round(rs.month * breakeven_ratio)))
            st.caption(f"Estimated breakeven month after switch: ~month {breakeven_month}.")
        st.caption(
            "Switch assumptions include admin/legal/valuation/clawback fees and "
            "lock-in early-prepayment penalty where applicable."
        )


def main() -> None:
    st.set_page_config(page_title="Singapore HDB EDA", layout="wide")
    _inject_minimal_ui()
    st.title("Singapore HDB resale — interactive analysis")
    _hero_card(
        "Market Intelligence Workspace",
        (
            "Explore resale dynamics, compare towns, and review model-backed pricing "
            "context in one visual layer."
        ),
        eyebrow="HDB Analytics Suite",
    )
    st.caption(f"Resale: [data.gov.sg]({HDB_CITATION_URL}) (Open Data Licence)")
    st.caption(
        "Optional: `pip install -e \".[geo]\"` for geopandas "
        "(planning-area neighbour graph). "
        "Town dummies: ~80% of rows (env SINGAPORE_EDA_TOWN_COVERAGE, e.g. 1.0 = all towns)."
    )

    default = os.environ.get("SINGAPORE_EDA_CSV", str(DEFAULT_RAW_CSV))
    default = _bootstrap_resale_if_missing(default)
    rent_default = _bootstrap_rent_if_missing(str(_DEFAULT_RENT))
    geo_default = _bootstrap_geo_if_tiny_or_missing(_default_geo())
    use_path = st.sidebar.text_input("Resale CSV path", value=default)
    rent_path = st.sidebar.text_input(
        "Median rent CSV (for yields)",
        value=rent_default,
    )
    geo_path = st.sidebar.text_input("Planning-area GeoJSON (map)", value=geo_default)
    corr_thr = st.sidebar.slider("Graph min |correlation|", 0.0, 0.99, 0.2, 0.05)

    try:
        df = _load(use_path)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except KeyError as e:
        st.error(
            "Selected CSV is missing required columns for HDB processing "
            f"(missing: {e}). Please choose an HDB resale CSV."
        )
        st.stop()
    except ValueError as e:
        st.error(f"Could not parse selected CSV: {e}")
        st.stop()

    (
        tab_pm,
        tab_o,
        tab_map,
        tab_storey,
        tab_yield,
        tab_cl,
        tab_fc,
        tab_g,
        tab_pol,
        tab_pred,
        tab_housing,
    ) = st.tabs(
        [
            "Project & method",
            "Overview",
            "Map & blocks",
            "Storey",
            "Rental yield",
            "Clusters",
            "Forecast",
            "Graphs",
            "Policy / culture",
            "Forecaster V1",
            "HDB calculator",
        ]
    )

    with tab_pm:
        st.markdown(PROJECT_AND_METHOD)

    with tab_o:
        _section_divider("Executive Snapshot", icon="📌")
        _why(
            "Core file stats, baseline model fit, and network coverage at a glance.",
            "Users can quickly decide whether the loaded dataset is broad enough and "
            "statistically usable before drilling into details.",
        )
        model_df = model_design_subset(df)
        model = None
        model_s = None
        if len(model_df) >= 10:
            try:
                model = ols_log_price(model_df)
            except Exception as ex:  # noqa: BLE001
                st.warning(f"OLS: {ex}")
            try:
                model_s = ols_log_price_with_storey(model_df)
            except Exception as ex:  # noqa: BLE001
                st.warning(f"OLS+storey: {ex}")
        gsum = None
        try:
            rawm = df.dropna(subset=["month", "town", "resale_price"]).copy()
            rawm["month"] = pd.to_datetime(rawm["month"])
            p0 = town_price_pivot(rawm, min_months=1)
            g0 = correlation_graph(p0, min_corr=corr_thr)
            gsum = graph_summary(g0)
        except Exception:  # noqa: BLE001
            gsum = None
        ins = build_insights(
            df,
            model=model_s or model,
            graph=gsum,
            eip=eip_match_stats(df),
        )
        c0, c1, c2, c3, c4, c5 = st.columns(6)
        with c0:
            _metric_card("Rows", f"{int(ins.get('n_transactions', 0) or 0):,}", "🧾")
        with c1:
            _metric_card(
                "Median resale",
                f"${ins['median_resale']:,.2f}" if "median_resale" in ins else "—",
                "💵",
            )
        with c2:
            _metric_card(
                "Towns in file",
                f"{int(ins['n_towns']):,}" if "n_towns" in ins else "—",
                "🏘️",
            )
        with c3:
            _metric_card(
                "Town groups in OLS",
                f"{int(ins.get('towns_in_model', 0) or 0):,}"
                if "towns_in_model" in ins
                else "—",
                "🧠",
            )
        with c4:
            _metric_card(
                "OLS R-squared",
                f"{float(ins['ols_r2']):,.3f}" if ins.get("ols_r2") is not None else "—",
                "📈",
            )
        with c5:
            _metric_card(
                "Graph towns",
                f"{int(ins['graph']['n_nodes']):,}"
                if ins.get("graph") and ins["graph"].get("n_nodes")
                else "—",
                "🕸️",
            )
        with st.expander("Structured summary (JSON)", expanded=False):
            st.write(ins)
        _section_divider("Descriptive Visuals", icon="🖼️")
        _why(
            "Price levels, time trend, distribution by flat type, and numeric correlation matrix.",
            "Users can validate market shape (level, trend, spread, co-movement) before "
            "interpreting model outputs.",
        )
        towns = sorted(df["town"].dropna().unique().tolist()) if "town" in df.columns else []
        pick = st.multiselect("Filter towns (overview)", towns, default=[])
        sub = df if not pick else df[df["town"].isin(pick)]
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_median_price_by_town(sub, top=20), width="stretch")
        with c2:
            st.plotly_chart(fig_price_over_time(sub, resample="M"), width="stretch")
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(fig_flat_type_box(df), width="stretch")
        with c4:
            _cols = (
                "resale_price",
                "floor_area_sqm",
                "remaining_lease_years",
                "log_resale_price",
            )
            nc = [c for c in _cols if c in df.columns]
            cmat = numeric_correlation(df, nc)
            st.plotly_chart(fig_correlation_heatmap(cmat), width="stretch")
        if model is not None:
            _section_divider("OLS Model (Readable)", icon="📐")
            with st.expander("What this OLS is (read this first)", expanded=False):
                st.markdown(
                    """
- **Outcome:** `log_resale_price` (natural log of resale price) so coefficients on continuous
  variables are *elasticity‑like* in spirit (small effects ≈ % change in price for a 1‑unit
  change in the predictor, only approximately).
- **Predictors:** `floor_area_sqm`, `remaining_lease_years` (from parsed HDB lease text), and
  **categorical town_group** (major towns that cover a target share of rows; others = `OTHER`).
- **Interpretation:** p‑values and CIs are for this **sample**; they do *not* prove causality
  (e.g. town dummies confound *location* with *flat mix* in that town).
- **R²:** descriptive fit. High R² does not mean the model is right for policy or out‑of‑sample
  prediction.
"""
                )
            _render_ols_readable(model, "OLS (log price ~ area + lease + town groups)")
        if model_s is not None and model_s is not model:
            _section_divider("OLS + Storey Band", icon="🏢")
            with st.expander("What this adds (storey)", expanded=False):
                st.markdown(
                    """
We add **dummy variables for storey band** (low / mid / high) derived from the *midpoint* of
HDB’s **range** (e.g. “04 TO 06”), not the exact level. Coefficients are *relative* to a reference
band. This is a **heuristic control**; it does not equal a causal “floor premium” because block,
time, and flat type are not fully modelled.
"""
                )
            st.caption("Same as base OLS with `C(storey_band)`; see the `storey` module for bands.")
            _render_ols_readable(
                model_s,
                "OLS + storey band (HDB `storey_range` as categories)",
                add_storey_note=True,
            )
        if len(towns) >= 2:
            _section_divider("Two-Town Statistical Difference Test", icon="⚖️")
            _why(
                "Welch t-test compares average resale price between two selected towns.",
                "Users can quickly check whether observed town-level mean differences are "
                "likely signal or sampling noise (without claiming causality).",
            )
            t_pick = st.multiselect(
                "Pick two towns to compare (mean of raw $; not size‑adjusted):",
                towns,
                default=towns[:2],
                help=(
                    "Tests difference in *average* price between two samples; "
                    "does not control for area or lease."
                ),
            )
            if len(t_pick) == 2:
                with st.expander("What the t-test is", expanded=False):
                    st.markdown(
                        """
- **Test:** *Welch’s* two‑sample t‑test on **resale price** in levels (S$), not on log or per‑sqm.
- **H₀:** the two town groups have the **same** mean price.
- **Caveat:** if one town has larger flats on average, “higher mean” does not mean “more expensive
  *per m²*”. Use the OLS for *all else equal* (within this data’s controls), not the t‑test.
"""
                    )
                tt = ttest_resale_by_group(df, "town", t_pick[0], t_pick[1])
                if "error" in tt:
                    st.caption(str(tt.get("error", tt)))
                else:
                    ga, gb = tt.get("group_a"), tt.get("group_b")
                    ts, pv = tt.get("t_stat", float("nan")), tt.get("p_value", float("nan"))
                    na, nb = tt.get("n_a"), tt.get("n_b")
                    ma, mb = tt.get("mean_a"), tt.get("mean_b")
                    st.markdown(
                        f"**{ga}** vs **{gb}** — t={ts:,.4f}, p={pv:,.6g} (n={na:,}, {nb:,}).\n\n"
                        f"Mean **{ga}:** ${ma:,.2f} · Mean **{gb}:** ${mb:,.2f}."
                    )
            else:
                st.caption("Select **exactly two** towns in the list above to run the test.")

    with tab_map:
        _section_divider("Planning-Area Choropleth", icon="🗺️")
        st.markdown(
            """
**Planning map:** each polygon is a **planning area** (from your GeoJSON). **Colours** = median
resale in your file joined via `planning_area` (from `data/reference/town_to_planning_area.csv`)
or `town` if needed. Unmatched areas stay grey. **This is not block-level** — HDB’s open resale CSV
does not include block coordinates. **Block detail** is in the table below.
"""
        )
        if geo_path and Path(geo_path).exists() and str(geo_path).endswith("tiny.geojson"):
            st.warning(
                "You are using the **2-polygon test fixture** (only two areas visible). For all "
                "planning areas, run: `python scripts/fetch_reference_geo.py` and point the field "
                f"to `{_PLANNING_GEO.name}` in `data/reference/`."
            )
        if not (geo_path and Path(geo_path).exists()):
            st.info("Set a GeoJSON path, or run `python scripts/fetch_reference_geo.py`.")
        else:
            vals = geo_median_value_dict(df)
            m = folium_choropleth_by_name(
                Path(geo_path),
                vals,
                name_prop="name",
                show_unmatched=True,
            )
            st_folium(m, width=1200, height=500)
        _section_divider("Block and Street Medians", icon="🏷️")
        _why(
            "Transaction-level median resale and count by town/street/block combinations.",
            "Users can inspect micro-level concentration and compare streets/blocks in the "
            "loaded file, despite no official block coordinates in source data.",
        )
        blk = block_street_table(df, top_n=200)
        if not blk.empty:
            st.caption(
                "Group = HDB town + street + block. Median and count in your loaded file only."
            )
            st.dataframe(
                block_table_for_display(blk), width="stretch", height=400
            )
        else:
            st.info("Need `town`, `street_name`, `block`, and `resale_price` in the CSV.")

    with tab_storey:
        _section_divider("Storey and Size Stratification", icon="🏬")
        st.markdown(
            """
- **`storey_range`** in open data is a **band** (e.g. “04 TO 06”), not an exact floor; we order
  bands into **low / mid / high** for OLS in the Overview tab.
- **Area** bins are **floor area** quantiles *within this file* (m² and **sqft** for Singapore
  listings). Bins are **not** the same as BTO marketing banding — they’re sample-dependent.
- **`floor_area_sqft`** in the enriched frame: **1 m² ≈ 10.764 sqft** (conversion); marketing copy
  may round differently.
"""
        )
        mps = median_price_by_storey_stratum(df)
        _why(
            "Median resale split by storey band and floor-area strata (m² + sqft).",
            "Users can compare vertical premium patterns while controlling for broad size buckets.",
        )
        st.dataframe(
            storey_table_for_display(mps) if not mps.empty else mps,
            width="stretch",
            height=400,
        )

    with tab_yield:
        _section_divider("Gross Rental Yield", icon="💹")
        st.markdown(
            f"""
**Gross yield (%)** = `12 × median_rent / median_resale × 100%` (annualised rent as a % of
stratum median resale) for joined **quarter × town × flat_type** cells. HDB’s median rent is
for the **whole** flat of that `flat_type` in that town/period (not a single bedroom). Use the
[official quarterly median-rent file]({HDB_MEDIAN_RENT_CITATION_URL})
(apr 2005–present; re-download to refresh) so resale rows match **contemporaneous** rent
quarters, not 2017-only data.

**Source:** [data.gov.sg — Median rent by town and flat type]({HDB_MEDIAN_RENT_CITATION_URL}).
"""
        )
        if rent_path and Path(rent_path).exists():
            ah = age_hours(rent_path)
            fresh = rent_csv_is_fresh(rent_path)
            st.caption(
                f"Rent file age: {ah:.1f} h" if ah is not None else "No rent file"
            )
            st.caption(
                "Within TTL" if fresh else "Older than TTL — consider `hdb-rent-download`"
            )
            try:
                ytab = gross_yield_table(df, Path(rent_path))
                ytab = ytab.sort_values("quarter", ascending=False)
                if ytab.empty:
                    st.warning(
                        "No matched rent/resale rows after join on quarter, town, and flat type. "
                        "Use the official full rent extract or check path/time overlap."
                    )
                else:
                    _why(
                        "Annualised gross yield (%) for quarter × town × flat type matched rows.",
                        "Users can benchmark income-return levels across segments and time with "
                        "consistent rent/resale cohort alignment.",
                    )
                    st.dataframe(
                        gross_yield_table_for_display(ytab),
                        width="stretch",
                        height=400,
                    )
            except Exception as ex:  # noqa: BLE001
                st.warning(str(ex))
        else:
            st.info(
                f"Download the official median-rent extract to e.g. `{str(DEFAULT_RENT_CSV)}`: "
                f"`hdb-rent-download -o {str(DEFAULT_RENT_CSV)}`. "
                f"Override with env `HDB_MEDIAN_RENT_RESOURCE_ID` if the portal id changes. "
                f"Dataset: {HDB_MEDIAN_RENT_CITATION_URL}"
            )

    with tab_cl:
        _section_divider("Clustering Segments", icon="🧩")
        _why(
            "K-means groups records into behaviorally similar segments by price, area, and lease.",
            "Users get compact segment labels for communication and portfolio-style comparison.",
        )
        _f = ("resale_price", "floor_area_sqm", "remaining_lease_years")
        feats = [c for c in _f if c in df.columns]
        if len(feats) >= 2:
            lab, _km, _ = cluster_kmeans(df, feats, n_clusters=4, random_state=0)
            med, explain = cluster_interpretation(df, lab, feats)
            st.markdown(
                "**How to read this:** k-means splits rows into 4 **segments**; each caption "
                "compares to **this file** (not the whole of Singapore)."
            )
            for k in sorted(explain):
                st.caption(explain[k])
            if not med.empty:
                st.dataframe(cluster_medians_for_display(med), width="stretch")
        else:
            st.info("Need numeric features for clustering.")

    with tab_fc:
        _section_divider("Town Forecast and Backtest", icon="🔭")
        _why(
            "Per-town monthly median history with ETS forecast and backtest RMSE.",
            "Users can compare directional outlook and model reliability across selected towns.",
        )
        towns_f = sorted(df["town"].dropna().unique().tolist()) if "town" in df.columns else []
        st.markdown(
            "Pick one or more **towns**; each line shows its history and 6m **ETS** (dotted)."
        )
        n_def = min(3, max(1, len(towns_f)))
        sel = st.multiselect(
            "Towns to compare (required — one colour per town; no single “all” mix)",
            towns_f,
            default=towns_f[:n_def] if len(towns_f) >= 1 else [],
        )
        if not sel:
            st.info("Select at least one town.")
        else:
            st.plotly_chart(
                fig_town_median_lines_with_forecast(
                    df, [t for t in sel if t in towns_f], horizon=6
                ),
                width="stretch",
            )
            st.caption(
                "ETS needs ~24+ months per town; no dotted line means the series is too short."
            )
            r_rows = []
            for t in sel:
                s = monthly_median_price(df, town=t)
                if len(s) < 1:
                    continue
                r_rows.append(
                    {
                        "town": t,
                        "months": len(s),
                        "backtest_rmse": backtest_rmse(
                            s, test_months=min(6, max(1, len(s) // 4))
                        ),
                    }
                )
            if r_rows:
                st.dataframe(forecast_rmse_for_display(pd.DataFrame(r_rows)), width="stretch")

    with tab_g:
        _section_divider("Correlation Network", icon="🕸️")
        _why(
            "Graph of towns linked when monthly median prices co-move above |rho| threshold.",
            "Users can identify synchronised market clusters and potentially diversified "
            "town sets.",
        )
        rawm = df.dropna(subset=["month", "town", "resale_price"]).copy()
        rawm["month"] = pd.to_datetime(rawm["month"])
        p = town_price_pivot(rawm, min_months=1)
        g = correlation_graph(p, min_corr=corr_thr)
        gsum = graph_summary(g)
        st.subheader("Town price correlation (monthly medians)")
        st.markdown(
            f"**Summary:** {gsum.get('n_nodes', 0)} towns with enough months; **"
            f"{gsum.get('n_edges', 0)}** edges where |ρ| ≥ {corr_thr}. "
            f"**{gsum.get('n_communities', 0)}** modularity-based groups. "
            "The layout below is a spring embedding — not geography."
        )
        if g.number_of_nodes():
            ctab = correlation_community_table(g)
            st.dataframe(ctab, width="stretch", height=200)
        edge_df = correlation_edges_dataframe(g)
        if not edge_df.empty:
            _section_divider("Top Correlated Town Pairs", icon="🔗")
            st.dataframe(
                correlation_edges_for_display(edge_df.head(50)),
                width="stretch",
                height=300,
            )
        st.plotly_chart(_fig_network(g), width="stretch")
        if geo_path and Path(geo_path).exists():
            _section_divider("Planning-Area Touch Graph", icon="🧭")
            st.caption(
                "Edges where planning polygons **share a boundary** (GeoJSON, spatial `touches`). "
                "Install: `pip install -e \".[geo]\"` (geopandas + engines)."
            )
            _why(
                "Neighbour graph from polygon boundary contact (adjacent planning areas).",
                "Users can reason about local spillover zones, contiguous clusters, and "
                "cross-area planning relationships beyond price correlation.",
            )
            sg, note = spatial_adjacency_graph(Path(geo_path), name_prop="name")
            st.caption(f"Build status: **{note}**")
            if sg.number_of_nodes() and note == "ok":
                st.success(
                    f"{sg.number_of_nodes():,} areas, {sg.number_of_edges():,} undirected border "
                    "pairs (spatial join / touches)."
                )
                st.dataframe(
                    unweighted_spatial_edge_table(sg).head(100),
                    width="stretch",
                    height=300,
                )
            if sg.number_of_nodes() and sg.number_of_edges() and note == "ok":
                ctab2 = correlation_community_table(sg)
                if "n_towns" in ctab2.columns:
                    ctab2["# areas in group"] = ctab2["n_towns"].map(
                        lambda v: f"{int(v):,}" if pd.notna(v) else ""
                    )
                    ctab2 = ctab2.drop(columns=["n_towns"], errors="ignore")
                st.dataframe(ctab2, width="stretch", height=150)
                st.plotly_chart(_fig_network(sg), width="stretch")
            if note != "ok" and "geopandas" in note.lower():
                st.info(
                    f"{note} Then: `python scripts/fetch_reference_geo.py` and select the "
                    f"GeoJSON in the sidebar."
                )
            elif note != "ok" and "geopandas" not in note.lower():
                st.warning(note)

    with tab_pol:
        _section_divider("Policy and Cultural Features", icon="🏛️")
        _why(
            "Coverage of EIP-matched rows and policy-adjacent enrichments "
            "(maturity, placeholders).",
            "Users understand where policy context exists versus where analysis "
            "remains descriptive.",
        )
        e = eip_match_stats(df)
        c1, c2 = st.columns(2)
        c1.metric("EIP-matched rows", f"{e.get('eip_matched_rows', 0):,}")
        c2.metric("Share of rows", f"{e.get('eip_row_rate', 0) * 100:.1f}%")
        st.markdown(
            "**EIP:** block-level stub join when `data/reference/eip_block_stub.csv` matches. "
            "**Sun:** set `ENABLE_SUN_PROXY=1` for placeholder columns. "
            "**Numerology:** digit flags are descriptive only."
        )
        if "eip_status_note" in df.columns:
            eipdf = df[["town", "block", "eip_status_note"]].drop_duplicates()
            st.dataframe(eipdf, width="stretch")
        if "maturity" in df.columns:
            st.dataframe(df[["town", "maturity"]].drop_duplicates(), width="stretch")

    with tab_pred:
        _section_divider("HDB Forecaster V1", icon="🔮")
        _why(
            "Point estimate, uncertainty range, and top model drivers for a "
            "candidate flat profile.",
            "Users get transparent, interview-ready explainability instead of a black-box number.",
        )
        st.markdown(
            """
### Structured Prediction Workspace
Use the guided controls, then run prediction. Inputs are validated with
historical combinations to reduce invalid scenarios.
"""
        )
        _render_forecaster_v1(df)
    with tab_housing:
        _section_divider("BTO / Housing Detail Planner", icon="🏠")
        _why(
            (
                "End-to-end ownership economics including MOP, grants, levy, "
                "loans, itemized costs, and profit waterfall."
            ),
            (
                "Users can compare affordability and potential outcomes with "
                "transparent assumptions and monthly cashflow."
            ),
        )
        _render_housing_economics_details()

    st.download_button(
        "Download enriched sample (5000 rows)",
        df.head(5000).to_csv(index=False).encode("utf-8"),
        file_name="hdb_enriched.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
