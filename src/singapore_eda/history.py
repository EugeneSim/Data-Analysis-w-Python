"""Merge multiple HDB resale tranches into one long panel with a period label."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_NON_ALNUM = re.compile(r"[^a-z0-9]+", re.IGNORECASE)


def _snake(s: str) -> str:
    s = s.strip()
    s = _NON_ALNUM.sub("_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower()


def standardize_hdb_schema(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_snake(c) for c in out.columns]
    out["data_period"] = period_label
    if "month" in out.columns:
        out["month"] = pd.to_datetime(out["month"], errors="coerce")
    return out


def concat_hdb_tranches(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def canonicalize_resale_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename snake_case columns to align with the Jan-2017+ canonical schema when needed."""
    out = df.copy()
    out.columns = [_snake(c) for c in out.columns]
    alias_to_canon: dict[str, str] = {
        "resale__price": "resale_price",
        "resale__price_": "resale_price",
    }
    renames: dict[str, str] = {}
    for c in list(out.columns):
        if c in alias_to_canon and alias_to_canon[c] not in out.columns:
            renames[c] = alias_to_canon[c]
    if renames:
        out = out.rename(columns=renames)
    return out


def merge_resale_tranches(
    tranches: list[tuple[Path, str]],
    out_path: Path | str | None = None,
    *,
    year_min: int | None = None,
    year_max: int | None = None,
    as_parquet: bool = False,
) -> pd.DataFrame:
    """
    Read each CSV, apply canonical + standardized schema, concat, optional year window.

    Tranches: list of (path, data_period label). Output path optional.
    """
    dfs: list[pd.DataFrame] = []
    for p, plabel in tranches:
        path = Path(p)
        raw = pd.read_csv(path, low_memory=False)
        raw = canonicalize_resale_columns(raw)
        st = standardize_hdb_schema(raw, plabel)
        dfs.append(st)
    out = concat_hdb_tranches(dfs)
    if "month" in out.columns and (year_min is not None or year_max is not None):
        has_m = out["month"].notna()
        yy = out["month"].dt.year
        keep = has_m
        if year_min is not None:
            keep &= yy >= year_min
        if year_max is not None:
            keep &= yy <= year_max
        out = out.loc[keep]
    if out_path is not None:
        op = Path(out_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        if as_parquet or str(op).lower().endswith(".parquet"):
            out.to_parquet(op, index=False, engine="pyarrow")
        else:
            out.to_csv(op, index=False)
    return out
