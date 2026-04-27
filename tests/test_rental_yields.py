from __future__ import annotations

from pathlib import Path

import pandas as pd

from singapore_eda.clean import clean_hdb
from singapore_eda.rental_yields import (
    gross_yield_table,
    normalize_hdb_flat_type,
    normalize_quarter_key,
)

_FIX = Path(__file__).parent / "fixtures" / "hdb_sample.csv"
_RENT = Path(__file__).parent / "fixtures" / "median_rent_sample.csv"


def test_gross_yield_joins() -> None:
    raw = pd.read_csv(_FIX)
    c = clean_hdb(raw)
    y = gross_yield_table(c, _RENT)
    assert not y.empty
    assert "gross_yield" in y.columns
    assert "gross_yield_pct" in y.columns
    assert y["gross_yield"].notna().any()
    assert (y["gross_yield_pct"] / 100.0 - y["gross_yield"]).abs().max() < 1e-6


def test_normalizers_align_rm_and_room() -> None:
    assert normalize_hdb_flat_type("3-RM") == normalize_hdb_flat_type("3 ROOM")
    assert normalize_hdb_flat_type("4-ROOM") == "4 ROOM"
    assert normalize_quarter_key("2017-Q1") == normalize_quarter_key("2017Q1")


def test_gross_yield_joins_rm_rent_to_room_resale(tmp_path) -> None:
    """CKAN-style 4-RM should match resale 4 ROOM after normalisation."""
    raw = pd.read_csv(_FIX)
    c = clean_hdb(raw)
    p = tmp_path / "rent.csv"
    p.write_text(
        "quarter,town,flat_type,median_rent\n2017Q1,ANG MO KIO,3-RM,2000\n",
        encoding="utf-8",
    )
    y = gross_yield_table(c, p)
    assert not y.empty
    assert (y["gross_yield"] > 0).any()
