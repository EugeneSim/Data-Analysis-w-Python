"""Format tables for human-readable display (thousands, currency, %, lease)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def format_currency(value: float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or not math.isfinite(value)):
        return ""
    return f"{float(value):,.2f}"


def format_percent(value: float | None) -> str:
    """Format a 0–100 percent value (e.g. 5.25 from yield*100) with 2 decimals."""
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or not math.isfinite(value)):
        return ""
    return f"{float(value):,.2f}%"


def format_ratio_as_percent(ratio: float | None) -> str:
    """`ratio` in [0,1] e.g. 0.0525 -> 5.25%."""
    if ratio is None:
        return ""
    if isinstance(ratio, float) and (math.isnan(ratio) or not math.isfinite(ratio)):
        return ""
    return f"{float(ratio) * 100.0:,.2f}%"


def remaining_lease_text_from_years(y: float) -> str:
    """
    Turn decimal years (e.g. 92.4167) into '92y 5m 0d' style, consistent with
    parsing `years + months/12` in clean.py.
    """
    if y is None or not isinstance(y, (int, float, np.floating)):
        return ""
    yf = float(y)
    if not math.isfinite(yf) or np.isnan(yf):
        return ""
    y = yf
    y_i = int(y)
    m_float = (y - y_i) * 12.0
    m_i = int(m_float)
    # carry if rounding pushes months to 12
    if m_i >= 12:
        y_i += 1
        m_i = 0
    d_part = m_float - m_i
    d_i = int(max(0, min(29, round(d_part * 30.4375))))  # ~mean month length
    return f"{y_i}y {m_i}m {d_i}d"


def gross_yield_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Readable strings for Streamlit: gross yield as %, medians with commas."""
    if df.empty:
        return df
    out = df.copy()
    if "gross_yield_pct" in out.columns:
        out["gross yield (%)"] = out["gross_yield_pct"].map(format_percent)
    elif "gross_yield" in out.columns:
        out["gross yield (%)"] = (out["gross_yield"] * 100.0).map(format_percent)
    if "median_resale" in out.columns:
        out["median resale ($)"] = out["median_resale"].map(format_currency)
    if "median_rent" in out.columns:
        out["median rent ($/m)"] = out["median_rent"].map(format_currency)
    if "n_resale" in out.columns:
        out["n resale"] = out["n_resale"].map(lambda v: f"{int(v):,}" if pd.notna(v) else "")
    _drop = (
        "gross_yield",
        "gross_yield_pct",
        "median_resale",
        "median_rent",
        "n_resale",
    )
    out = out.drop(
        columns=[c for c in _drop if c in out.columns],
        errors="ignore",
    )
    front = [c for c in ("quarter", "town", "flat_type") if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def correlation_edges_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "abs_correlation" not in df.columns:
        return df
    out = df.copy()
    if "correlation" in out.columns:
        out["ρ"] = out["correlation"].map(
            lambda v: f"{float(v):,.4f}" if pd.notna(v) and np.isfinite(v) else ""
        )
        out = out.drop(columns=["correlation"], errors="ignore")
    out["|ρ|"] = out["abs_correlation"].map(
        lambda v: f"{float(v):,.4f}" if pd.notna(v) and np.isfinite(v) else ""
    )
    return out.drop(columns=["abs_correlation"], errors="ignore")


def storey_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "median_resale" in out.columns:
        out["median resale ($)"] = out["median_resale"].map(format_currency)
        out = out.drop(columns=["median_resale"], errors="ignore")
    return out


def block_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "median_resale" in out.columns:
        out["median resale ($)"] = out["median_resale"].map(format_currency)
        out = out.drop(columns=["median_resale"], errors="ignore")
    if "n_trans" in out.columns:
        out["n trans"] = out["n_trans"].map(lambda v: f"{int(v):,}" if pd.notna(v) else "")
        out = out.drop(columns=["n_trans"], errors="ignore")
    return out


def cluster_medians_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if c == "cluster_id":
            continue
        if out[c].dtype.kind in "fiu":
            out[c] = out[c].map(
                lambda v: f"{float(v):,.2f}" if pd.notna(v) and np.isfinite(float(v)) else ""
            )
    return out


def forecast_rmse_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "backtest_rmse" not in df.columns:
        return df
    out = df.copy()

    def _fmt_rmse(v: object) -> str:
        if not pd.notna(v) or not np.isfinite(float(v)):
            return ""
        return f"{float(v):,.2f}"

    out["backtest RMSE"] = out["backtest_rmse"].map(_fmt_rmse)
    if "town" in out.columns:
        return out.drop(columns=["backtest_rmse"], errors="ignore")
    return out
