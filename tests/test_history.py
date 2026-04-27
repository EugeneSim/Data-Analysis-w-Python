from __future__ import annotations

import pandas as pd

from singapore_eda.history import (
    concat_hdb_tranches,
    merge_resale_tranches,
    standardize_hdb_schema,
)


def test_standardize_adds_period() -> None:
    raw = pd.DataFrame({"Month": ["2017-01-01"], "Resale_Price": [1]})
    s = standardize_hdb_schema(raw, "p1")
    assert "data_period" in s.columns
    assert s["resale_price"].iloc[0] == 1
    assert s["data_period"].iloc[0] == "p1"


def test_concat() -> None:
    a = standardize_hdb_schema(pd.DataFrame({"month": ["2017-01-01"]}), "a")
    b = standardize_hdb_schema(pd.DataFrame({"month": ["2018-01-01"]}), "b")
    c = concat_hdb_tranches([a, b])
    assert len(c) == 2


def test_merge_tranches(tmp_path) -> None:
    hdr = (
        "month,town,flat_type,block,street_name,storey_range,floor_area_sqm,"
        "flat_model,lease_commence_date,remaining_lease,resale_price\n"
    )
    (tmp_path / "t1.csv").write_text(
        hdr + "2015-01,ANG MO KIO,2 ROOM,1,ST,01 TO 03,40,X,1980,50 years,100000\n"
    )
    (tmp_path / "t2.csv").write_text(
        hdr + "2017-01,TAMPINES,3 ROOM,2,ST,01 TO 03,60,Y,1990,60 years,200000\n"
    )
    out = tmp_path / "merged.csv"
    m = merge_resale_tranches(
        [(tmp_path / "t1.csv", "p2015"), (tmp_path / "t2.csv", "p2017")],
        out,
        year_min=2015,
    )
    assert len(m) == 2
    assert "data_period" in m.columns
    assert out.exists()
