from __future__ import annotations

from pathlib import Path

import pandas as pd

from singapore_eda.geo_join import enrich_with_reference

_FIX = Path(__file__).parent / "fixtures" / "hdb_sample.csv"


def test_enrich_adds_planning_area() -> None:
    raw = pd.read_csv(_FIX)
    d = enrich_with_reference(raw.assign(town=raw["town"].str.upper()))
    assert "planning_area" in d.columns
    assert d["planning_area"].notna().any()
