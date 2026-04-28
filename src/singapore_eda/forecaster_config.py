"""Configuration management for forecaster v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ForecasterConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15
    random_state: int = 42
    model_dir: Path = Path("models/forecaster_v1")
    candidate_models: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm"])
    xgboost_params: dict[str, Any] = field(default_factory=dict)
    lightgbm_params: dict[str, Any] = field(default_factory=dict)
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "singapore-eda-forecaster-v1"
    deepchecks_enabled: bool = False
    deepchecks_output_dir: Path = Path("reports/deepchecks/forecaster_v1")
    use_lag_features: bool = False
    interval_nominal_coverage: float = 0.80
    rolling_min_windows: int = 3
    coverage_tolerance: float = 0.10
    max_mean_interval_width: float = 220000.0
    feedback_store_path: Path = Path("data/feedback/forecaster_feedback.csv")
    feedback_validated_path: Path = Path("data/feedback/forecaster_feedback_validated.csv")
    feedback_retraining_path: Path = Path("data/feedback/forecaster_feedback_retraining.csv")
    feedback_retention_days: int = 365
    feedback_min_comment_redact_digits: int = 6


def load_forecaster_config(path: Path | str) -> ForecasterConfig:
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    t = raw.get("train", {})
    m = raw.get("models", {})
    tr = raw.get("tracking", {})
    v = raw.get("validation", {})
    fb = raw.get("feedback", {})
    return ForecasterConfig(
        train_frac=float(t.get("train_frac", 0.70)),
        val_frac=float(t.get("val_frac", 0.15)),
        random_state=int(t.get("random_state", 42)),
        model_dir=Path(t.get("model_dir", "models/forecaster_v1")),
        candidate_models=list(t.get("candidate_models", ["xgboost", "lightgbm"])),
        xgboost_params=dict(m.get("xgboost", {})),
        lightgbm_params=dict(m.get("lightgbm", {})),
        mlflow_enabled=bool(tr.get("mlflow_enabled", False)),
        mlflow_tracking_uri=(
            str(tr.get("mlflow_tracking_uri", "sqlite:///mlflow.db")).strip() or "sqlite:///mlflow.db"
        ),
        mlflow_experiment=str(tr.get("mlflow_experiment", "singapore-eda-forecaster-v1")),
        deepchecks_enabled=bool(v.get("deepchecks_enabled", False)),
        deepchecks_output_dir=Path(
            v.get("deepchecks_output_dir", "reports/deepchecks/forecaster_v1")
        ),
        use_lag_features=bool(t.get("use_lag_features", False)),
        interval_nominal_coverage=float(v.get("interval_nominal_coverage", 0.80)),
        rolling_min_windows=int(v.get("rolling_min_windows", 3)),
        coverage_tolerance=float(v.get("coverage_tolerance", 0.10)),
        max_mean_interval_width=float(v.get("max_mean_interval_width", 220000.0)),
        feedback_store_path=Path(fb.get("store_path", "data/feedback/forecaster_feedback.csv")),
        feedback_validated_path=Path(
            fb.get("validated_path", "data/feedback/forecaster_feedback_validated.csv")
        ),
        feedback_retraining_path=Path(
            fb.get("retraining_path", "data/feedback/forecaster_feedback_retraining.csv")
        ),
        feedback_retention_days=int(fb.get("retention_days", 365)),
        feedback_min_comment_redact_digits=int(fb.get("min_comment_redact_digits", 6)),
    )
