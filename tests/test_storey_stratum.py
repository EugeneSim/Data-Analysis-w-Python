from __future__ import annotations

from pathlib import Path

import pandas as pd

from singapore_eda.clean import clean_hdb
from singapore_eda.features import add_features
from singapore_eda.storey import add_storey_band, median_price_by_storey_stratum


def test_median_by_storey_has_readable_area_stratum() -> None:
    raw = pd.read_csv(Path(__file__).parent / "fixtures" / "hdb_sample.csv")
    c = add_storey_band(add_features(clean_hdb(raw), town_coverage=None, top_towns=20))
    out = median_price_by_storey_stratum(c)
    assert not out.empty
    assert "area_stratum (m²)" in out.columns or "storey_band" in out.columns
    assert "m²" in str(out.to_string())
