"""Feature engineering for modeling and EDA."""

from __future__ import annotations

import numpy as np
import pandas as pd

from singapore_eda.constants import SQM_TO_SQFT


def add_log_price_and_psm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "resale_price" in out.columns:
        out["log_resale_price"] = np.log(out["resale_price"].clip(lower=1))
    if "floor_area_sqm" in out.columns and "resale_price" in out.columns:
        area = out["floor_area_sqm"].replace(0, np.nan)
        out["price_per_sqm"] = out["resale_price"] / area
    if "floor_area_sqm" in out.columns:
        out["floor_area_sqft"] = out["floor_area_sqm"] * SQM_TO_SQFT
    if "month" in out.columns:
        out["year"] = out["month"].dt.year
        out["quarter"] = out["month"].dt.to_period("Q").astype(str)
    return out


def _towns_reaching_row_coverage(ser: pd.Series, coverage: float) -> list:
    """
    E.g. 0.8 → smallest set of most frequent `town` values that cover ≥80% of rows
    (for dummy coding without 26 separate levels when data is long-tailed).
    """
    n = int(ser.notna().sum())
    if n < 1:
        return []
    need = min(1.0, max(0.0, float(coverage)))
    vc = ser.value_counts()
    acc = 0
    out: list = []
    for t, c in vc.items():
        acc += c
        out.append(t)
        if acc / n >= need:
            break
    return out


def top_n_town_other(
    df: pd.DataFrame, *, n: int = 15, town_coverage: float | None = None
) -> pd.DataFrame:
    """
    Collapse rare towns into 'OTHER' for stable dummy coding.

    If ``town_coverage`` is not None, keep the fewest towns that cover that fraction
    of rows (e.g. 0.8 = ~80% of transactions by town mass). If None, use top ``n``
    by count.
    """
    out = df.copy()
    if "town" not in out.columns:
        return out
    if town_coverage is not None:
        top = _towns_reaching_row_coverage(out["town"], town_coverage)
    else:
        top = out["town"].value_counts().head(n).index.tolist()
    if not top:
        out["town_group"] = "OTHER"
        return out
    out["town_group"] = out["town"].where(out["town"].isin(top), "OTHER")
    return out


def add_features(
    df: pd.DataFrame,
    *,
    top_towns: int = 15,
    town_coverage: float | None = 0.8,
) -> pd.DataFrame:
    """
    Add log price, psm, time splits, and town groups.

    ``town_coverage`` (default 0.8) keeps the smallest set of towns that cover that
    share of **rows**; rare towns are ``OTHER``. Set ``town_coverage`` to 1.0 to
    use every town (many levels) or set ``top_towns`` and ``town_coverage=None`` to
    use a fixed count instead.
    """
    out = add_log_price_and_psm(df)
    if town_coverage is not None:
        out = top_n_town_other(out, town_coverage=town_coverage)
    else:
        out = top_n_town_other(out, n=top_towns)
    return out


def model_design_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for OLS: non-null price, area, remaining lease, town_group."""
    _cols = ("resale_price", "floor_area_sqm", "remaining_lease_years", "town_group")
    need = [c for c in _cols if c in df.columns]
    out = df.dropna(subset=need) if need else df.copy()
    if "resale_price" in out.columns:
        out = out[out["resale_price"] > 0]
    if "floor_area_sqm" in out.columns:
        out = out[out["floor_area_sqm"] > 0]
    return out.reset_index(drop=True)
