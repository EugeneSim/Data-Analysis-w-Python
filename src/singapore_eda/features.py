"""Feature engineering for modeling and EDA."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from singapore_eda.constants import (
    DEFAULT_BTO_COMPLETION_STATUS_CSV,
    DEFAULT_BTO_PRICE_RANGE_CSV,
    SQM_TO_SQFT,
)


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


def _normalize_town_key(ser: pd.Series) -> pd.Series:
    return ser.astype(str).str.strip().str.upper()


def _bto_price_path() -> Path:
    return Path(
        os.environ.get("SINGAPORE_EDA_BTO_PRICE_RANGE_CSV", str(DEFAULT_BTO_PRICE_RANGE_CSV))
    )


def _bto_completion_path() -> Path:
    return Path(
        os.environ.get(
            "SINGAPORE_EDA_BTO_COMPLETION_STATUS_CSV",
            str(DEFAULT_BTO_COMPLETION_STATUS_CSV),
        )
    )


def _price_mid(min_ser: pd.Series, max_ser: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(min_ser, errors="coerce").fillna(0.0)
        + pd.to_numeric(max_ser, errors="coerce").fillna(0.0)
    ) / 2.0


def add_bto_reference_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add BTO historical/future supply context keyed by town and transaction year.

    Output columns:
    - bto_launch_count_town_3y
    - bto_avg_price_range_mid_town_3y
    - bto_under_construction_units_town
    - bto_completed_units_town
    """
    out = df.copy()
    for col in (
        "bto_launch_count_town_3y",
        "bto_avg_price_range_mid_town_3y",
        "bto_under_construction_units_town",
        "bto_completed_units_town",
    ):
        out[col] = 0.0
    if "town" not in out.columns:
        return out
    if "year" not in out.columns:
        if "month" in out.columns:
            out["year"] = pd.to_datetime(out["month"], errors="coerce").dt.year
        else:
            return out
    out["_town_key"] = _normalize_town_key(out["town"])
    out["_tx_year"] = pd.to_numeric(out["year"], errors="coerce")
    valid_mask = out["_tx_year"].notna()
    if not valid_mask.any():
        out = out.drop(columns=["_town_key", "_tx_year"], errors="ignore")
        return out

    bto_price_path = _bto_price_path()
    if bto_price_path.exists():
        price = pd.read_csv(bto_price_path)
        price.columns = [str(c).strip().lower().replace(" ", "_") for c in price.columns]
        need = {"town", "financial_year", "min_selling_price", "max_selling_price"}
        if need.issubset(price.columns):
            price = price.copy()
            price["_town_key"] = _normalize_town_key(price["town"])
            price["_year"] = pd.to_numeric(price["financial_year"], errors="coerce")
            price["_mid"] = _price_mid(price["min_selling_price"], price["max_selling_price"])
            price = price.dropna(subset=["_year"])
            agg = (
                price.groupby(["_town_key", "_year"], as_index=False)
                .agg(
                    launches=("financial_year", "count"),
                    mean_mid_price=("_mid", "mean"),
                )
                .reset_index(drop=True)
            )
            for yr in sorted(out.loc[valid_mask, "_tx_year"].astype(int).unique().tolist()):
                yr_mask = out["_tx_year"].astype("Int64") == int(yr)
                w = agg[(agg["_year"] >= (yr - 2)) & (agg["_year"] <= yr)]
                if w.empty:
                    continue
                by_town = w.groupby("_town_key", as_index=False).agg(
                    launch_count_3y=("launches", "sum"),
                    mid_price_3y=("mean_mid_price", "mean"),
                )
                c_map = by_town.set_index("_town_key")["launch_count_3y"].to_dict()
                p_map = by_town.set_index("_town_key")["mid_price_3y"].to_dict()
                out.loc[yr_mask, "bto_launch_count_town_3y"] = (
                    out.loc[yr_mask, "_town_key"].map(c_map).fillna(0.0).astype(float)
                )
                out.loc[yr_mask, "bto_avg_price_range_mid_town_3y"] = (
                    out.loc[yr_mask, "_town_key"].map(p_map).fillna(0.0).astype(float)
                )

    bto_comp_path = _bto_completion_path()
    if bto_comp_path.exists():
        comp = pd.read_csv(bto_comp_path)
        comp.columns = [str(c).strip().lower().replace(" ", "_") for c in comp.columns]
        need = {"town_or_estate", "financial_year", "status", "no_of_units"}
        if need.issubset(comp.columns):
            comp = comp.copy()
            comp["_town_key"] = _normalize_town_key(comp["town_or_estate"])
            comp["_year"] = pd.to_numeric(comp["financial_year"], errors="coerce")
            comp["_units"] = pd.to_numeric(comp["no_of_units"], errors="coerce").fillna(0.0)
            comp["_status"] = comp["status"].astype(str).str.strip().str.upper()
            comp = comp.dropna(subset=["_year"])
            for yr in sorted(out.loc[valid_mask, "_tx_year"].astype(int).unique().tolist()):
                yr_mask = out["_tx_year"].astype("Int64") == int(yr)
                hist = comp[comp["_year"] <= yr]
                if hist.empty:
                    continue
                uc = (
                    hist[hist["_status"].str.contains("UNDER", na=False)]
                    .groupby("_town_key")["_units"]
                    .sum()
                    .to_dict()
                )
                done = (
                    hist[hist["_status"].str.contains("COMPLETED", na=False)]
                    .groupby("_town_key")["_units"]
                    .sum()
                    .to_dict()
                )
                out.loc[yr_mask, "bto_under_construction_units_town"] = (
                    out.loc[yr_mask, "_town_key"].map(uc).fillna(0.0).astype(float)
                )
                out.loc[yr_mask, "bto_completed_units_town"] = (
                    out.loc[yr_mask, "_town_key"].map(done).fillna(0.0).astype(float)
                )

    out = out.drop(columns=["_town_key", "_tx_year"], errors="ignore")
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
    out = add_bto_reference_features(out)
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
