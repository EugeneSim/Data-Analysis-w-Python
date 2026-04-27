"""Storey / floor-band analysis (HDB `storey_range` is a band, not exact floor)."""

from __future__ import annotations

import pandas as pd

from singapore_eda.constants import SQM_TO_SQFT


def add_storey_band(df: pd.DataFrame) -> pd.DataFrame:
    """Ordered categories: low / mid / high from storey_mid when available."""
    out = df.copy()
    if "storey_mid" not in out.columns:
        return out
    sm = out["storey_mid"]
    out["storey_band"] = pd.cut(
        sm,
        bins=[-1, 3, 9, 100],
        labels=["low_1_3", "mid_4_9", "high_10plus"],
    ).astype(str)
    out.loc[sm.isna(), "storey_band"] = pd.NA
    return out


def median_price_by_storey_stratum(
    df: pd.DataFrame,
    *,
    area_bins: int = 5,
) -> pd.DataFrame:
    """Stratify by floor-area quantiles within sample; median price by storey_band."""
    need = {"resale_price", "floor_area_sqm", "storey_band"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()
    d = df.dropna(subset=list(need)).copy()
    if d.empty:
        return pd.DataFrame()
    try:
        q = min(area_bins, len(d))
        d["area_stratum"] = pd.qcut(d["floor_area_sqm"], q=q, duplicates="drop")
    except ValueError:
        d["area_stratum"] = "all"
    g = d.groupby(["area_stratum", "storey_band"], observed=True).agg(
        median_price=("resale_price", "median")
    )
    out = g.reset_index()
    out["area_stratum (m²)"] = out["area_stratum"].map(_format_area_stratum)
    out["area_stratum (sqft)"] = out["area_stratum"].map(_format_area_stratum_sqft)
    out = out.drop(columns=["area_stratum"], errors="ignore")
    if "median_price" in out.columns:
        out = out.rename(columns={"median_price": "median_resale"})
    return out


def _format_area_stratum(x) -> str:
    """Turn qcut / Interval / mixed types into a readable m² range for tables."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if hasattr(x, "left") and hasattr(x, "right"):
        a, b = float(x.left), float(x.right)
        return f"{a:.0f}–{b:.0f} m²"
    s = str(x)
    if s == "all":
        return "all (single bin)"
    return s


def _format_area_stratum_sqft(x) -> str:
    """Same bins as m², expressed in square feet (common in Singapore marketing)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if hasattr(x, "left") and hasattr(x, "right"):
        a = float(x.left) * SQM_TO_SQFT
        b = float(x.right) * SQM_TO_SQFT
        return f"{a:.0f}–{b:.0f} sqft"
    s = str(x)
    if s == "all":
        return "all (single bin)"
    return s
