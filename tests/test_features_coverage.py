from __future__ import annotations

import pandas as pd

from singapore_eda.features import _towns_reaching_row_coverage, add_features, top_n_town_other


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
