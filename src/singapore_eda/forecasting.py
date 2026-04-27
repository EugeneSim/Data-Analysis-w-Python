"""Simple ETS forecast on monthly median price series with rolling backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def monthly_median_price(
    df: pd.DataFrame,
    town: str | None = None,
) -> pd.Series:
    d = df.copy()
    if town is not None and "town" in d.columns:
        d = d[d["town"].astype(str).str.upper() == town.upper()]
    d = d.dropna(subset=["month", "resale_price"])
    s = d.groupby(d["month"].dt.to_period("M").dt.to_timestamp())["resale_price"].median()
    return s.sort_index()


def forecast_ets(
    series: pd.Series,
    horizon: int = 6,
) -> pd.Series:
    if len(series) < 24:
        return pd.Series(dtype=float)
    try:
        # Pass ndarray to avoid statsmodels "date index has no frequency" warnings
        y = series.astype(float).to_numpy()
        model = ExponentialSmoothing(
            y,
            seasonal_periods=12,
            trend="add",
            seasonal="add",
        ).fit()
        fcast = np.asarray(model.forecast(horizon), dtype=float)
        if isinstance(series.index, pd.DatetimeIndex) and len(series) > 0:
            last = series.index.max()
            idx = pd.date_range(last, periods=horizon + 1, freq="MS")[1:]
            return pd.Series(fcast, index=idx)
        return pd.Series(fcast)
    except Exception:
        return pd.Series(dtype=float)


def backtest_rmse(
    series: pd.Series,
    *,
    test_months: int = 6,
) -> float:
    if len(series) <= test_months + 12:
        return float("nan")
    train = series.iloc[:-test_months]
    test = series.iloc[-test_months:]
    try:
        model = ExponentialSmoothing(
            train.astype(float).to_numpy(),
            seasonal_periods=12,
            trend="add",
            seasonal="add",
        ).fit()
        pred = model.forecast(test_months)
        pv = np.asarray(pred, dtype=float)
        return float(np.sqrt(((test.values - pv) ** 2).mean()))
    except Exception:
        return float("nan")
