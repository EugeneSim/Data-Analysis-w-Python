from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from singapore_eda.feature_store import validate_schema
from singapore_eda.forecaster_v1 import (
    FEATURE_COLUMNS,
    TrainConfig,
    _add_leakage_safe_market_features,
    _row_to_frame,
    predict_with_explain,
    time_split,
)


class DummyModel:
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.array([np.log1p(500000.0)])


def _sample_train_df(n: int = 36) -> pd.DataFrame:
    months = pd.date_range("2021-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "month": months,
            "town": ["ANG MO KIO"] * n,
            "flat_type": ["4 ROOM"] * n,
            "flat_model": ["MODEL A"] * n,
            "storey_range": ["04 TO 06"] * n,
            "floor_area_sqm": np.linspace(70, 100, n),
            "lease_commence_date": [1998] * n,
            "remaining_lease_years": np.linspace(90, 60, n),
            "resale_price": np.linspace(320000, 620000, n),
            "year": [int(m.year) for m in months],
            "month_num": [int(m.month) for m in months],
            "source_path": ["data/raw/sample.csv"] * n,
            "source_sha256": ["abc"] * n,
            "ingested_at_utc": ["2026-01-01T00:00:00+00:00"] * n,
        }
    )


def test_validate_schema_ok() -> None:
    df = _sample_train_df()
    validate_schema(df)


def test_time_split_monotonic_and_nonempty() -> None:
    df = pd.concat([_sample_train_df(30), _sample_train_df(30)], ignore_index=True)
    df = df.sort_values("month").reset_index(drop=True)
    train, val, test = time_split(df, TrainConfig(train_frac=0.6, val_frac=0.2))
    assert not train.empty and not val.empty and not test.empty
    assert train["month"].max() <= val["month"].min()
    assert val["month"].max() <= test["month"].min()
    train_m = set(train["month"].dt.to_period("M"))
    val_m = set(val["month"].dt.to_period("M"))
    test_m = set(test["month"].dt.to_period("M"))
    assert train_m.isdisjoint(val_m)
    assert train_m.isdisjoint(test_m)
    assert val_m.isdisjoint(test_m)


def test_predict_with_explain_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_blob = {
        "model": DummyModel(),
        "feature_columns": list(FEATURE_COLUMNS),
        "raw_features": list(FEATURE_COLUMNS),
        "numeric_bounds": {
            "floor_area_sqm": {"min": 30.0, "max": 180.0},
            "remaining_lease_years": {"min": 40.0, "max": 99.0},
            "year": {"min": 2019.0, "max": 2026.0},
            "month_num": {"min": 1.0, "max": 12.0},
        },
        "residual_interval": {"q10": -20000.0, "q90": 30000.0},
    }
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"

    import joblib

    joblib.dump(model_blob, model_path)
    metadata_path.write_text(json.dumps({"version": "forecaster_v1"}), encoding="utf-8")

    class _FakeExplainer:
        def __init__(self, model: object) -> None:
            self.model = model

        def shap_values(self, x: pd.DataFrame) -> np.ndarray:
            return np.ones((1, x.shape[1]), dtype=float) * 0.01

    class _FakeShap:
        TreeExplainer = _FakeExplainer

    monkeypatch.setitem(__import__("sys").modules, "shap", _FakeShap())
    out = predict_with_explain(
        {
            "month": "2026-01-01",
            "town": "ANG MO KIO",
            "flat_type": "4 ROOM",
            "flat_model": "MODEL A",
            "storey_range": "04 TO 06",
            "floor_area_sqm": 200.0,
            "lease_commence_date": 1998,
            "remaining_lease_years": 60.0,
        },
        model_path=model_path,
        metadata_path=metadata_path,
    )
    assert out["prediction"] > 0
    assert out["prediction_interval"]["p90"] >= out["prediction_interval"]["p10"]
    assert isinstance(out["top_contributors"], list) and out["top_contributors"]
    assert out["warnings"]


def test_leakage_safe_lag_features_use_prior_rows_only() -> None:
    df = _sample_train_df(8)
    out = _add_leakage_safe_market_features(df)
    assert {"lag_town_median_price", "lag_town_txn_count"}.issubset(out.columns)
    assert float(out.iloc[0]["lag_town_txn_count"]) == 0.0
    assert float(out.iloc[1]["lag_town_txn_count"]) == 1.0
    # first row has no history; should fallback to finite global median
    assert np.isfinite(float(out.iloc[0]["lag_town_median_price"]))


def test_row_to_frame_requires_month() -> None:
    with pytest.raises(ValueError):
        _row_to_frame({"town": "ANG MO KIO"}, list(FEATURE_COLUMNS))
