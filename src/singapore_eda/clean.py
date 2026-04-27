"""Clean and type raw HDB columns."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from singapore_eda.constants import DEFAULT_PROCESSED_PARQUET

_LEASE_RE = re.compile(r"(?P<years>\d+)\s*years?(?:\s*(?P<months>\d+)\s*months?)?", re.IGNORECASE)
_STOREY_RE = re.compile(r"(\d+)\s*TO\s*(\d+)", re.IGNORECASE)


def _parse_remaining_lease_to_years(lease: str | float) -> float:
    if pd.isna(lease) or (isinstance(lease, str) and not str(lease).strip()):
        return float("nan")
    s = str(lease).strip()
    m = _LEASE_RE.search(s)
    if not m:
        return float("nan")
    years = float(m.group("years"))
    months = float(m.group("months") or 0)
    return years + months / 12.0


def _remaining_lease_text_from_years(y: float) -> str:
    """
    Human-readable from decimal years: 92.4167 -> '92y 5m 0d' (matches month fraction).
    """
    if y is None:
        return ""
    if isinstance(y, (float, np.floating)) and (not math.isfinite(float(y)) or np.isnan(y)):
        return ""
    y = float(y)
    y_i = int(y)
    m_float = (y - y_i) * 12.0
    m_i = int(m_float)
    if m_i >= 12:
        y_i += 1
        m_i = 0
    d_part = m_float - m_i
    d_i = int(max(0, min(29, round(d_part * 30.4375))))
    return f"{y_i}y {m_i}m {d_i}d"


def _parse_storey_mid(storey: str | float) -> float:
    if pd.isna(storey) or (isinstance(storey, str) and not str(storey).strip()):
        return float("nan")
    s = str(storey).strip()
    m = _STOREY_RE.search(s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (a + b) / 2.0
    m2 = re.search(r"(\d+)", s)
    if m2:
        return float(m2.group(1))
    return float("nan")


def _coerce_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")


def clean_hdb(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy: dates, numerics, dedup, sorted."""
    out = df.copy()
    if "month" in out.columns:
        out["month"] = pd.to_datetime(out["month"], errors="coerce")
    if "floor_area_sqm" in out.columns:
        out["floor_area_sqm"] = _coerce_int_series(out["floor_area_sqm"])
    if "lease_commence_date" in out.columns:
        out["lease_commence_date"] = _coerce_int_series(out["lease_commence_date"])
    if "resale_price" in out.columns:
        out["resale_price"] = pd.to_numeric(out["resale_price"], errors="coerce")

    if "remaining_lease" in out.columns:
        out["remaining_lease_years"] = out["remaining_lease"].map(_parse_remaining_lease_to_years)
        _raw = out["remaining_lease"].map(
            lambda x: str(x).strip() if pd.notna(x) and str(x).strip() else None
        )
        _from_y = out["remaining_lease_years"].map(_remaining_lease_text_from_years)
        out["remaining_lease_label"] = _raw.where(_raw.notna(), _from_y)
    elif "remaining_lease_years" in out.columns:
        out["remaining_lease_label"] = out["remaining_lease_years"].map(
            _remaining_lease_text_from_years
        )
    if "storey_range" in out.columns:
        out["storey_mid"] = out["storey_range"].map(_parse_storey_mid)

    for c in ("town", "flat_type", "flat_model"):
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip().str.upper()

    out = out.dropna(subset=["resale_price", "month"], how="any")
    out = out.drop_duplicates()
    out = out.sort_values("month").reset_index(drop=True)
    return out


def clean_and_save(df: pd.DataFrame, out_path: str | Path | None = None) -> pd.DataFrame:
    cleaned = clean_hdb(df)
    out = Path(out_path) if out_path is not None else DEFAULT_PROCESSED_PARQUET
    out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(out, index=False, engine="pyarrow")
    return cleaned
