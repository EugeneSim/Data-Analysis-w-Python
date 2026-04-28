from __future__ import annotations

from pathlib import Path

import pandas as pd

from singapore_eda.feedback import append_feedback, materialize_feedback_views
from singapore_eda.forecaster_config import load_forecaster_config


def test_load_forecaster_config_defaults() -> None:
    cfg = load_forecaster_config(Path("configs/forecaster_v1.yaml"))
    assert cfg.random_state == 42
    assert "xgboost" in cfg.candidate_models
    assert "lightgbm" in cfg.candidate_models
    assert cfg.feedback_store_path.name == "forecaster_feedback.csv"
    assert cfg.rolling_min_windows >= 1
    assert cfg.feedback_validated_path.name == "forecaster_feedback_validated.csv"


def test_append_feedback_writes_csv(tmp_path: Path) -> None:
    out = append_feedback(
        store_path=tmp_path / "feedback.csv",
        model_version="forecaster_v1",
        model_family="xgboost",
        predicted_price=500000,
        user_rating=4,
        user_comment="Looks reasonable.",
        input_payload={"town": "ANG MO KIO"},
        actual_price=510000,
    )
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) == 1
    assert str(df.loc[0, "model_family"]) == "xgboost"
    assert "feedback_id" in df.columns


def test_materialize_feedback_views_generates_outputs(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    validated = tmp_path / "validated.csv"
    retraining = tmp_path / "retraining.csv"
    append_feedback(
        store_path=raw,
        model_version="forecaster_v1",
        model_family="xgboost",
        predicted_price=500000,
        user_rating=5,
        user_comment="Call me at 98765432",
        input_payload={"town": "ANG MO KIO"},
        actual_price=510000,
    )
    stats = materialize_feedback_views(
        raw_path=raw,
        validated_path=validated,
        retraining_path=retraining,
        retention_days=365,
        min_comment_redact_digits=6,
    )
    assert stats["validated_rows"] == 1
    vdf = pd.read_csv(validated)
    rdf = pd.read_csv(retraining)
    assert len(vdf) == 1
    assert len(rdf) == 1
    assert "[REDACTED_NUMERIC]" in str(vdf.loc[0, "user_comment"])
