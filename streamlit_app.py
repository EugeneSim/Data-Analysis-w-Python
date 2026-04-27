"""Streamlit dashboard: HDB resale EDA (maps, clusters, forecast, graph, yields)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
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
    HDB_MEDIAN_RENT_CITATION_URL,
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
from singapore_eda.features import model_design_subset
from singapore_eda.forecasting import backtest_rmse, monthly_median_price
from singapore_eda.graph_analytics import (
    correlation_community_table,
    correlation_edges_dataframe,
    correlation_graph,
    graph_summary,
    spatial_adjacency_graph,
    town_price_pivot,
    unweighted_spatial_edge_table,
)
from singapore_eda.insights import build_insights
from singapore_eda.mapviz import (
    block_street_table,
    folium_choropleth_by_name,
    geo_median_value_dict,
)
from singapore_eda.pipeline import load_enriched
from singapore_eda.rent_cache import age_hours, rent_csv_is_fresh
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
_RENT_CAND = _ROOT / str(DEFAULT_RENT_CSV)
_FIX_RENT = _ROOT / "tests" / "fixtures" / "median_rent_sample.csv"
_DEFAULT_RENT = str(_RENT_CAND) if _RENT_CAND.is_file() else str(_FIX_RENT)
_PLANNING_GEO = _ROOT / "data" / "reference" / "planning_areas.geojson"
_TINY_GEO = _ROOT / "tests" / "fixtures" / "planning_areas_tiny.geojson"


def _is_truthy(v: str | None) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _on_streamlit_cloud() -> bool:
    # Streamlit Cloud exposes this in hosted apps.
    return _is_truthy(os.environ.get("STREAMLIT_SHARING_MODE"))


def _default_bool_env(name: str, cloud_default: bool, local_default: bool) -> bool:
    raw = os.environ.get(name)
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
    p = Path(default_path)
    if p.exists():
        return default_path

    auto_fetch = _default_bool_env(
        "SINGAPORE_EDA_AUTO_DOWNLOAD_ON_MISSING",
        cloud_default=True,
        local_default=False,
    )
    if auto_fetch:
        max_rows_raw = os.environ.get("SINGAPORE_EDA_BOOTSTRAP_MAX_ROWS", "20000")
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
            if p.exists():
                return str(p)
        except (OSError, ValueError, RuntimeError) as ex:
            st.sidebar.warning(f"Auto-download failed; using fixture fallback. ({ex})")

    if _DEFAULT_FIXTURE.exists():
        st.sidebar.info(
            "Using fallback fixture (small sample). "
            "Set `SINGAPORE_EDA_AUTO_DOWNLOAD_ON_MISSING=1` to auto-fetch data."
        )
        return str(_DEFAULT_FIXTURE)
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
    pri = out[out["term"].str.contains("|".join(priority), regex=True)].copy()
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


def _inject_minimal_ui() -> None:
    st.markdown(
        """
<style>
.block-container {padding-top: 1.25rem; padding-bottom: 1.5rem; max-width: 1200px;}
h1, h2, h3 {letter-spacing: -0.02em;}
.ux-h2 {font-size: 1.25rem; font-weight: 650; margin: 0.2rem 0 0.35rem 0;}
.ux-note {
  color: rgba(49, 51, 63, 0.78);
  border-left: 2px solid rgba(49, 51, 63, 0.20);
  padding-left: 0.65rem;
  margin: 0.2rem 0 0.75rem 0;
}
[data-testid="stMetricValue"] {font-size: 1.1rem;}
[data-testid="stDataFrame"] {border-radius: 8px;}
</style>
""",
        unsafe_allow_html=True,
    )


def _h2(text: str) -> None:
    st.markdown(f"<div class='ux-h2'>{text}</div>", unsafe_allow_html=True)


def _why(shows: str, solves: str) -> None:
    st.markdown(
        f"<div class='ux-note'><b>What this shows:</b> {shows}<br>"
        f"<b>Why this helps users:</b> {solves}</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Singapore HDB EDA", layout="wide")
    _inject_minimal_ui()
    st.title("Singapore HDB resale — interactive analysis")
    st.caption(f"Resale: [data.gov.sg]({HDB_CITATION_URL}) (Open Data Licence)")
    st.caption(
        "Optional: `pip install -e \".[geo]\"` for geopandas "
        "(planning-area neighbour graph). "
        "Town dummies: ~80% of rows (env SINGAPORE_EDA_TOWN_COVERAGE, e.g. 1.0 = all towns)."
    )

    default = os.environ.get("SINGAPORE_EDA_CSV", str(DEFAULT_RAW_CSV))
    default = _bootstrap_resale_if_missing(default)
    use_path = st.sidebar.text_input("Resale CSV path", value=default)
    rent_path = st.sidebar.text_input(
        "Median rent CSV (for yields)",
        value=str(_DEFAULT_RENT),
    )
    geo_path = st.sidebar.text_input("Planning-area GeoJSON (map)", value=_default_geo())
    corr_thr = st.sidebar.slider("Graph min |correlation|", 0.0, 0.99, 0.2, 0.05)

    try:
        df = _load(use_path)
    except FileNotFoundError as e:
        st.error(str(e))
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
        ]
    )

    with tab_pm:
        st.markdown(PROJECT_AND_METHOD)

    with tab_o:
        _h2("Executive Snapshot")
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
        c0.metric("Rows", f"{int(ins.get('n_transactions', 0) or 0):,}")
        c1.metric(
            "Median resale",
            f"${ins['median_resale']:,.2f}" if "median_resale" in ins else "—",
        )
        c2.metric(
            "Towns in file (raw CSV)",
            f"{int(ins['n_towns']):,}" if "n_towns" in ins else "—",
        )
        c3.metric(
            "Named town groups in OLS",
            f"{int(ins.get('towns_in_model', 0) or 0):,}"
            if "towns_in_model" in ins
            else "—",
        )
        c4.metric(
            "OLS R²",
            f"{float(ins['ols_r2']):,.3f}" if ins.get("ols_r2") is not None else "—",
        )
        c5.metric(
            "Graph towns",
            f"{int(ins['graph']['n_nodes']):,}"
            if ins.get("graph") and ins["graph"].get("n_nodes")
            else "—",
        )
        with st.expander("Structured summary (JSON)", expanded=False):
            st.write(ins)
        _h2("Descriptive Visuals")
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
            _h2("OLS Model (Readable)")
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
            _h2("OLS + Storey Band")
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
            _h2("Two-Town Statistical Difference Test")
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
        _h2("Planning-Area Choropleth")
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
        _h2("Block and Street Medians")
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
        _h2("Storey and Size Stratification")
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
        _h2("Gross Rental Yield")
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
        _h2("Clustering Segments")
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
        _h2("Town Forecast and Backtest")
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
        _h2("Correlation Network")
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
            _h2("Top Correlated Town Pairs")
            st.dataframe(
                correlation_edges_for_display(edge_df.head(50)),
                width="stretch",
                height=300,
            )
        st.plotly_chart(_fig_network(g), width="stretch")
        if geo_path and Path(geo_path).exists():
            _h2("Planning-Area Touch Graph")
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
        _h2("Policy and Cultural Features")
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

    st.download_button(
        "Download enriched sample (5000 rows)",
        df.head(5000).to_csv(index=False).encode("utf-8"),
        file_name="hdb_enriched.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
