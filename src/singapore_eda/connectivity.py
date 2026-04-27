"""Future / current MRT reference tables (static CSV in data/reference)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from singapore_eda.paths import reference_dir


def load_future_mrt(path: Path | None = None) -> pd.DataFrame:
    p = path or (reference_dir() / "future_mrt_stations.csv")
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def town_connectivity_enriched(df: pd.DataFrame) -> pd.DataFrame:
    """No-op: MRT columns come from `geo_join.load_mrt_access` merge."""
    return df
