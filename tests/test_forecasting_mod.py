from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from singapore_eda.clean import clean_hdb
from singapore_eda.features import add_features
from singapore_eda.forecasting import backtest_rmse, forecast_ets, monthly_median_price

_FIX = Path(__file__).parent / "fixtures" / "hdb_sample.csv"


def test_monthly_series() -> None:
    raw = pd.read_csv(_FIX)
    c = add_features(clean_hdb(raw), town_coverage=None, top_towns=15)
    s = monthly_median_price(c, town="TAMPINES")
    assert len(s) >= 1


def test_ets_on_long_monthly_series() -> None:
    idx = pd.date_range("2018-01-01", periods=30, freq="MS")
    s = pd.Series(np.linspace(100.0, 200.0, 30), index=idx)
    fc = forecast_ets(s, horizon=3)
    assert len(fc) == 3
    assert isinstance(fc.index, pd.DatetimeIndex)
    rm = backtest_rmse(s, test_months=6)
    assert rm == rm and rm < 1e6  # finite
