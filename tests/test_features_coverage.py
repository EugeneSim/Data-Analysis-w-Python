from __future__ import annotations

import pandas as pd

from singapore_eda.features import (
    _towns_reaching_row_coverage,
    add_bto_reference_features,
    add_features,
    top_n_town_other,
)


def test_towns_80pc_covers_enough_rows() -> None:
    s = pd.Series(list("A") * 8 + list("B") * 1 + list("C") * 1)  # 10 rows, A=80%
    top = _towns_reaching_row_coverage(s, 0.8)
    assert top == ["A"]


def test_add_features_uses_coverage() -> None:
    df = pd.DataFrame(
        {
            "town": ["X"] * 9 + ["Y"] * 1,
            "resale_price": [400_000.0] * 10,
            "month": pd.to_datetime(pd.date_range("2020-01-01", periods=10, freq="MS")),
        }
    )
    out = top_n_town_other(df, town_coverage=0.8)
    assert (out["town_group"] == "X").sum() == 9
    assert (out["town_group"] == "OTHER").sum() == 1


def test_town_coverage_1_uses_all_towns() -> None:
    df = pd.DataFrame(
        {
            "town": ["A", "B", "A"],
            "resale_price": [1.0, 1.0, 1.0],
            "month": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
        }
    )
    out = add_features(df, town_coverage=1.0)
    assert (out["town_group"] == out["town"]).all()


def test_add_bto_reference_features_joins_from_reference_files(
    tmp_path, monkeypatch
) -> None:
    price = tmp_path / "bto_price.csv"
    comp = tmp_path / "bto_comp.csv"
    pd.DataFrame(
        {
            "financial_year": [2020, 2021, 2022],
            "town": ["ANG MO KIO", "ANG MO KIO", "BEDOK"],
            "room_type": ["4-RM", "4-RM", "4-RM"],
            "min_selling_price": [300000, 320000, 400000],
            "max_selling_price": [420000, 450000, 520000],
        }
    ).to_csv(price, index=False)
    pd.DataFrame(
        {
            "financial_year": [2020, 2021, 2022],
            "town_or_estate": ["ANG MO KIO", "ANG MO KIO", "BEDOK"],
            "status": ["Under Construction", "Completed", "Under Construction"],
            "no_of_units": [300, 100, 500],
            "hdb_or_dbss": ["HDB", "HDB", "HDB"],
        }
    ).to_csv(comp, index=False)
    monkeypatch.setenv("SINGAPORE_EDA_BTO_PRICE_RANGE_CSV", str(price))
    monkeypatch.setenv("SINGAPORE_EDA_BTO_COMPLETION_STATUS_CSV", str(comp))
    frame = pd.DataFrame(
        {
            "month": pd.to_datetime(["2021-06-01", "2022-06-01"]),
            "year": [2021, 2022],
            "town": ["ANG MO KIO", "BEDOK"],
        }
    )
    out = add_bto_reference_features(frame)
    assert "bto_launch_count_town_3y" in out.columns
    assert float(out.loc[0, "bto_launch_count_town_3y"]) >= 2.0
    assert float(out.loc[0, "bto_under_construction_units_town"]) >= 300.0
    assert float(out.loc[1, "bto_under_construction_units_town"]) >= 500.0
