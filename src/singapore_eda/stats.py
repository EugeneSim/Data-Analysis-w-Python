"""Hypothesis tests, correlation, and OLS for HDB EDA."""

from __future__ import annotations

from typing import Any

import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf


def numeric_correlation(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    if cols is None:
        num = df.select_dtypes(include=["number", "bool"]).columns
        cols = [c for c in num if c in df.columns]
    sub = df[cols].dropna(how="all")
    return sub.corr(numeric_only=True)


def ttest_resale_by_group(
    df: pd.DataFrame, group_col: str, a: str, b: str, value: str = "resale_price"
) -> dict:
    g1 = df.loc[df[group_col] == a, value].dropna()
    g2 = df.loc[df[group_col] == b, value].dropna()
    if len(g1) < 2 or len(g2) < 2:
        return {"error": "insufficient data for two-sample t-test"}
    t_stat, p_value = st.ttest_ind(g1, g2, equal_var=False)
    return {
        "test": "Welch t-test",
        "group_a": a,
        "group_b": b,
        "n_a": int(len(g1)),
        "n_b": int(len(g2)),
        "mean_a": float(g1.mean()),
        "mean_b": float(g2.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def ols_log_price(df: pd.DataFrame) -> Any:
    """
    OLS: log_resale_price ~ floor_area_sqm + remaining_lease_years + C(town_group)
    Requires: log_resale_price, floor_area_sqm, remaining_lease_years, town_group
    """
    need = ["log_resale_price", "floor_area_sqm", "remaining_lease_years", "town_group"]
    d = df.dropna(subset=need)
    model = smf.ols(
        "log_resale_price ~ floor_area_sqm + remaining_lease_years + C(town_group)",
        data=d,
    ).fit()
    return model


def ols_log_price_with_storey(df: pd.DataFrame) -> Any:
    """
    OLS with C(storey_band) when enough non-null bands exist; else base OLS.
    Requires storey_band, log_resale_price, floor_area_sqm, remaining_lease_years, town_group.
    """
    need = ["log_resale_price", "floor_area_sqm", "remaining_lease_years", "town_group"]
    if "storey_band" not in df.columns or df["storey_band"].notna().sum() < 20:
        return ols_log_price(df)
    d = df.dropna(subset=need + ["storey_band"])
    if len(d) < 30:
        return ols_log_price(df)
    model = smf.ols(
        "log_resale_price ~ floor_area_sqm + remaining_lease_years + "
        "C(town_group) + C(storey_band)",
        data=d,
    ).fit()
    return model
