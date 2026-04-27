from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from singapore_eda.clean import clean_hdb
from singapore_eda.features import add_features, model_design_subset
from singapore_eda.pipeline import load_enriched
from singapore_eda.stats import (
    numeric_correlation,
    ols_log_price,
    ols_log_price_with_storey,
    ttest_resale_by_group,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "hdb_sample.csv"


def test_ttest_two_towns() -> None:
    raw = pd.read_csv(_FIXTURE)
    c = add_features(clean_hdb(raw), top_towns=5, town_coverage=None)
    r = ttest_resale_by_group(c, "town", "TAMPINES", "PUNGGOL", value="resale_price")
    assert "p_value" in r
    assert r["n_a"] >= 1
    assert r["n_b"] >= 1


def test_ols_runs() -> None:
    raw = pd.read_csv(_FIXTURE)
    c = add_features(clean_hdb(raw), top_towns=3, town_coverage=None)
    m = model_design_subset(c)
    if len(m) < 10:
        pytest.skip("not enough rows for OLS in fixture")
    fit = ols_log_price(m)
    assert fit.rsquared >= 0.0
    assert len(fit.params) > 0


def test_ols_with_storey_runs() -> None:
    df = load_enriched(_FIXTURE, top_towns=5, town_coverage=None)
    if len(df) < 20:
        pytest.skip("not enough rows for storey OLS in fixture")
    fit = ols_log_price_with_storey(df)
    assert fit.rsquared >= 0.0
    assert len(fit.params) > 0


def test_correlation() -> None:
    raw = pd.read_csv(_FIXTURE)
    c = add_features(clean_hdb(raw), town_coverage=None, top_towns=15)
    _want = (
        "resale_price",
        "floor_area_sqm",
        "remaining_lease_years",
        "log_resale_price",
    )
    cols = [x for x in _want if x in c.columns]
    corr = numeric_correlation(c, cols)
    assert not corr.empty
    assert abs(corr.loc["resale_price", "floor_area_sqm"]) <= 1.0
