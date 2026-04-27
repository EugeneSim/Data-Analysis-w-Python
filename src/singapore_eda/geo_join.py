"""Join HDB rows to planning area, maturity, and connectivity reference tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from singapore_eda.paths import reference_dir


def load_town_to_planning_area(path: Path | None = None) -> pd.DataFrame:
    p = path or (reference_dir() / "town_to_planning_area.csv")
    df = pd.read_csv(p)
    df["town"] = df["town"].astype(str).str.strip().str.upper()
    return df


def load_town_maturity(path: Path | None = None) -> pd.DataFrame:
    p = path or (reference_dir() / "town_maturity.csv")
    df = pd.read_csv(p)
    df["town"] = df["town"].astype(str).str.strip().str.upper()
    return df.drop_duplicates(subset=["town"], keep="first")


def load_mrt_access(path: Path | None = None) -> pd.DataFrame:
    p = path or (reference_dir() / "mrt_access_by_town.csv")
    if not p.exists():
        return pd.DataFrame(columns=["town", "mrt_station_count", "nearest_mrt_km_proxy"])
    df = pd.read_csv(p)
    df["town"] = df["town"].astype(str).str.strip().str.upper()
    return df.drop_duplicates(subset=["town"], keep="first")


def enrich_with_reference(
    df: pd.DataFrame,
    *,
    planning_path: Path | None = None,
    maturity_path: Path | None = None,
    mrt_path: Path | None = None,
) -> pd.DataFrame:
    """Add planning_area, region_ocr, maturity, mrt columns when town is present."""
    out = df.copy()
    if "town" not in out.columns:
        return out
    out["town"] = out["town"].astype(str).str.strip().str.upper()

    pa = load_town_to_planning_area(planning_path)
    out = out.merge(pa, on="town", how="left")

    mat = load_town_maturity(maturity_path)
    cols = mat[["town", "maturity", "notes"]]
    out = out.merge(cols, on="town", how="left", suffixes=("", "_mat"))

    mrt = load_mrt_access(mrt_path)
    if not mrt.empty:
        out = out.merge(mrt, on="town", how="left")

    return out
