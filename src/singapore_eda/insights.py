"""Structured summary of EDA and model outputs for reports and the app."""

from __future__ import annotations

import pandas as pd


def build_insights(
    df: pd.DataFrame,
    model=None,
    *,
    graph: dict | None = None,
    eip: dict | None = None,
    extra: dict | None = None,
) -> dict:
    out: dict = {
        "n_transactions": int(len(df)),
    }
    if "resale_price" in df.columns:
        out["median_resale"] = float(df["resale_price"].median())
        out["mean_resale"] = float(df["resale_price"].mean())
    if "month" in df.columns and len(df) > 0:
        out["date_range"] = {
            "start": str(df["month"].min().date()),
            "end": str(df["month"].max().date()),
        }
    if "town" in df.columns:
        out["n_towns"] = int(df["town"].nunique())
        out["top_town_median"] = (
            df.groupby("town", observed=True)["resale_price"].median().idxmax()
            if "resale_price" in df.columns
            else None
        )
    if "town_group" in df.columns:
        tg = df["town_group"].dropna().astype(str).unique().tolist()
        out["towns_in_model"] = int(len([t for t in tg if t.upper() != "OTHER"]))
    if model is not None and hasattr(model, "rsquared"):
        out["ols_r2"] = float(model.rsquared)
        out["ols_adj_r2"] = float(model.rsquared_adj)
    if graph is not None:
        out["graph"] = graph
    if eip is not None:
        out["eip"] = eip
    if extra is not None:
        out["extra"] = extra
    return out
