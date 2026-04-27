"""Gross rental yield from aligned median rent vs median resale (stratum-level)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_quarter_key(q: object) -> str:
    """
    Align CKAN quarter strings with pandas period labels (e.g. ``2017Q1``).

    Accepts ``2017Q1``, ``2017-Q1``, ``2017 Q1``, ``2017q1``.
    """
    if q is None or (isinstance(q, float) and np.isnan(q)):
        return ""
    s = str(q).strip().upper()
    s = s.replace("-", "")
    s = re.sub(r"[\s_–—]+", "", s)
    return s


def normalize_hdb_flat_type(raw: object) -> str:
    """
    Map HDB open-data variants to one key for joining resale ↔ median-rent tables.

    Resale CSVs often use ``3 ROOM``; data.gov.sg median rent often uses ``3-RM`` / ``3-RM`` style.
    """
    if raw is None:
        return ""
    if isinstance(raw, float) and np.isnan(raw):
        return ""
    t = str(raw).strip().upper()
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"\s+", " ", t)
    m = re.match(r"^(\d)\s*-?\s*RM$", t)
    if m:
        return f"{m.group(1)} ROOM"
    m = re.match(r"^(\d)RM$", t)
    if m:
        return f"{m.group(1)} ROOM"
    m = re.match(r"^(\d)\s*-\s*ROOM$", t)
    if m:
        return f"{m.group(1)} ROOM"
    m = re.match(r"^(\d)\s*ROOM$", t)
    if m:
        return f"{m.group(1)} ROOM"
    if t == "EXEC":
        return "EXECUTIVE"
    if t in ("MULTI GENERATION", "MULTI-GENERATION"):
        return "MULTI-GENERATION"
    if t in ("MG", "3GEN"):
        return "MULTI-GENERATION"
    return t


def resale_to_quarter_median(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate resale to quarter x town x flat_type median price and count."""
    d = df.dropna(subset=["month", "town", "flat_type", "resale_price"]).copy()
    d["town"] = d["town"].astype(str).str.strip().str.upper()
    d["flat_type"] = d["flat_type"].map(normalize_hdb_flat_type)
    d["quarter"] = d["month"].dt.to_period("Q").astype(str)
    d["quarter"] = d["quarter"].map(normalize_quarter_key)
    g = (
        d.groupby(["quarter", "town", "flat_type"], observed=True)
        .agg(median_resale=("resale_price", "median"), n_resale=("resale_price", "size"))
        .reset_index()
    )
    return g


def _coerce_median_rent_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        x = s.astype(str).str.strip().str.upper()
        x = x.replace(
            {
                "NA": np.nan,
                "N/A": np.nan,
                "-": np.nan,
                "": np.nan,
                "NS": np.nan,
            }
        )
        return pd.to_numeric(x, errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def load_median_rent_csv(path: Path) -> pd.DataFrame:
    """Expect columns including quarter, town, flat_type, median_rent (flexible names)."""
    r = pd.read_csv(path)
    r.columns = [c.strip().lower().replace(" ", "_") for c in r.columns]
    if "quarter" not in r.columns and "quarter_" in "".join(r.columns):
        for c in r.columns:
            if c.startswith("quarter"):
                r = r.rename(columns={c: "quarter"})
                break
    for c in ("town", "flat_type"):
        if c in r.columns:
            r[c] = r[c].astype(str).str.strip().str.upper()
    if "flat_type" in r.columns:
        r["flat_type"] = r["flat_type"].map(normalize_hdb_flat_type)
    if "quarter" in r.columns:
        r["quarter"] = r["quarter"].map(normalize_quarter_key)
    if "median_rent" in r.columns:
        r["median_rent"] = _coerce_median_rent_series(r["median_rent"])
    r = r.dropna(subset=["median_rent"], how="any")
    if "median_rent" in r.columns:
        r = r[r["median_rent"] > 0]
    return r


def gross_yield_table(
    resale_df: pd.DataFrame,
    rent_path: Path,
) -> pd.DataFrame:
    """Join stratum medians; gross_yield = 12 * median_rent / median_resale."""
    r = load_median_rent_csv(rent_path)
    if "quarter" not in r.columns and "month" in r.columns:
        r["month"] = pd.to_datetime(r["month"], errors="coerce")
        r["quarter"] = r["month"].dt.to_period("Q").astype(str)
    r["quarter"] = r["quarter"].map(normalize_quarter_key)
    pr = resale_to_quarter_median(resale_df)
    key = ["quarter", "town", "flat_type"]
    for k in key:
        if k not in r.columns:
            raise ValueError(f"Rent file missing {k}")
    if "median_rent" not in r.columns:
        raise ValueError("Rent file needs median_rent column")
    rent = r.groupby(key, observed=True)["median_rent"].median().reset_index()
    m = pr.merge(rent, on=key, how="inner")
    m["gross_yield"] = (12 * m["median_rent"]) / m["median_resale"].replace(0, np.nan)
    m["gross_yield_pct"] = m["gross_yield"] * 100.0
    return m
