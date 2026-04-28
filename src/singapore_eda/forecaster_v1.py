"""V1 HDB forecaster: contract, training, evaluation, explainability, and inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from singapore_eda.features import add_bto_reference_features
from singapore_eda.pipeline import load_enriched

REQUIRED_COLUMNS: tuple[str, ...] = (
    "month",
    "town",
    "flat_type",
    "flat_model",
    "storey_range",
    "floor_area_sqm",
    "lease_commence_date",
    "remaining_lease_years",
    "resale_price",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "town",
    "flat_type",
    "flat_model",
    "storey_range",
    "floor_area_sqm",
    "lease_commence_date",
    "remaining_lease_years",
    "lease_age_years",
    "lease_decay_factor",
    "year",
    "month_num",
    "bto_launch_count_town_3y",
    "bto_avg_price_range_mid_town_3y",
    "bto_under_construction_units_town",
    "bto_completed_units_town",
)

OPTIONAL_LOCATION_COLUMNS: tuple[str, ...] = (
    "planning_area",
    "region_ocr",
    "maturity",
    "mrt_station_count",
    "nearest_mrt_km_proxy",
)

OPTIONAL_BTO_COLUMNS: tuple[str, ...] = (
    "bto_launch_count_town_3y",
    "bto_avg_price_range_mid_town_3y",
    "bto_under_construction_units_town",
    "bto_completed_units_town",
)

LAG_MARKET_COLUMNS: tuple[str, ...] = (
    "lag_town_median_price",
    "lag_town_flat_type_median_price",
    "lag_town_txn_count",
    "lag_town_flat_type_txn_count",
)


@dataclass(frozen=True)
class TrainConfig:
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


@dataclass(frozen=True)
class TrainArtifacts:
    model_path: Path
    metadata_path: Path
    config_path: Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_training_frame(csv_path: str | Path) -> pd.DataFrame:
    """Create the strict V1 training dataframe from enriched pipeline output."""
    src = Path(csv_path)
    df = load_enriched(src)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for forecaster v1: {missing}")

    optional_cols = [
        c for c in OPTIONAL_LOCATION_COLUMNS + OPTIONAL_BTO_COLUMNS if c in df.columns
    ]
    out = df[list(REQUIRED_COLUMNS) + optional_cols].copy()
    out["month"] = pd.to_datetime(out["month"], errors="coerce")
    out["year"] = out["month"].dt.year
    out["month_num"] = out["month"].dt.month
    out = out.dropna(subset=["month", "resale_price", "floor_area_sqm", "remaining_lease_years"])
    out = out[out["resale_price"] > 0]
    out = out[out["floor_area_sqm"] > 0]
    # Singapore HDB flats are typically 99-year leasehold; keep this bounded for stability.
    out["remaining_lease_years"] = out["remaining_lease_years"].clip(lower=1.0, upper=99.0)
    out["lease_age_years"] = (99.0 - out["remaining_lease_years"]).clip(lower=0.0, upper=99.0)
    out = out.sort_values("month").reset_index(drop=True)
    out["source_path"] = str(src.resolve())
    out["source_sha256"] = _sha256_file(src)
    out["ingested_at_utc"] = datetime.now(UTC).isoformat()
    out = _add_leakage_safe_market_features(out)
    return out


def _add_leakage_safe_market_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("month").reset_index(drop=True).copy()
    town_group = out.groupby("town", dropna=False)["resale_price"]
    seg_group = out.groupby(["town", "flat_type"], dropna=False)["resale_price"]
    out["lag_town_median_price"] = town_group.transform(lambda s: s.shift(1).expanding().median())
    out["lag_town_flat_type_median_price"] = seg_group.transform(
        lambda s: s.shift(1).expanding().median()
    )
    out["lag_town_txn_count"] = town_group.cumcount()
    out["lag_town_flat_type_txn_count"] = seg_group.cumcount()
    global_med = float(out["resale_price"].median()) if len(out) else 0.0
    out["lag_town_median_price"] = out["lag_town_median_price"].fillna(global_med)
    out["lag_town_flat_type_median_price"] = out["lag_town_flat_type_median_price"].fillna(
        global_med
    )
    return out


def _one_hot_features(df: pd.DataFrame, cols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    x = df.copy()
    for col in cols:
        if col not in x.columns:
            x[col] = np.nan
    x = x[list(cols)].copy()
    cat_cols = [
        "town",
        "flat_type",
        "flat_model",
        "storey_range",
        "planning_area",
        "region_ocr",
        "maturity",
    ]
    for col in cat_cols:
        if col in x.columns:
            x[col] = x[col].astype(str).str.upper()
    x = pd.get_dummies(
        x,
        columns=[c for c in cat_cols if c in x.columns],
        dummy_na=True,
        dtype=float,
    )
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.astype(float).fillna(0.0)
    return x


def time_split(
    df: pd.DataFrame, cfg: TrainConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Cannot split empty dataframe.")
    ordered = df.sort_values("month").reset_index(drop=True)
    month_keys = (
        ordered["month"].dt.to_period("M").dt.to_timestamp().dropna().sort_values().unique().tolist()
    )
    n_months = len(month_keys)
    if n_months < 3:
        raise ValueError("Need at least 3 unique months for month-bucket split.")

    n_train_m = max(1, int(n_months * cfg.train_frac))
    n_val_m = max(1, int(n_months * cfg.val_frac))
    n_train_m = min(n_train_m, n_months - 2)
    n_val_m = min(n_val_m, n_months - n_train_m - 1)
    train_months = set(month_keys[:n_train_m])
    val_months = set(month_keys[n_train_m : n_train_m + n_val_m])
    test_months = set(month_keys[n_train_m + n_val_m :])
    month_bucket = ordered["month"].dt.to_period("M").dt.to_timestamp()
    train = ordered.loc[month_bucket.isin(train_months)].copy()
    val = ordered.loc[month_bucket.isin(val_months)].copy()
    test = ordered.loc[month_bucket.isin(test_months)].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("Time split produced empty subset; provide more rows.")
    return train, val, test


def _segment_metrics(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("town", "flat_type"):
        if key not in df.columns:
            continue
        rows: list[dict[str, Any]] = []
        for name, grp in df.groupby(key):
            idx = grp.index.to_numpy()
            yt = y_true[idx]
            yp = y_pred[idx]
            if len(yt) < 3:
                continue
            rows.append(
                {
                    key: str(name),
                    "n": int(len(yt)),
                    "mae": float(mean_absolute_error(yt, yp)),
                    "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                    "mape": float(np.mean(np.abs((yt - yp) / np.clip(yt, 1.0, None)))),
                }
            )
        out[f"{key}_metrics"] = rows
    return out


def _fit_lease_decay_lambda(df: pd.DataFrame) -> float:
    """
    Estimate an empirical lease decay coefficient from historical data.
    We model log(price) ~= a - lambda * lease_age_years and use lambda>=0.
    """
    need = ["resale_price", "lease_age_years"]
    if not set(need).issubset(df.columns):
        return 0.01
    d = df[need].dropna().copy()
    if len(d) < 200:
        return 0.01
    x = d["lease_age_years"].to_numpy(dtype=float)
    y = np.log1p(d["resale_price"].to_numpy(dtype=float))
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 0:
        return 0.01
    slope = float(np.dot(x_centered, y - y.mean()) / denom)
    # slope expected negative; convert to positive decay intensity.
    lam = max(0.0, -slope)
    # Guardrail against unstable extremes.
    return float(min(max(lam, 0.001), 0.08))


def _apply_lease_decay_features(df: pd.DataFrame, decay_lambda: float) -> pd.DataFrame:
    out = df.copy()
    if "lease_age_years" not in out.columns:
        if "remaining_lease_years" in out.columns:
            out["lease_age_years"] = (99.0 - out["remaining_lease_years"]).clip(
                lower=0.0, upper=99.0
            )
        else:
            out["lease_age_years"] = 0.0
    out["lease_decay_factor"] = np.exp(-float(decay_lambda) * out["lease_age_years"].astype(float))
    return out


def _write_model_card(*, metadata: dict[str, Any], output_path: Path) -> None:
    bench = metadata.get("model_benchmark", {})
    metrics = metadata.get("metrics", {}).get("test", {})
    split = metadata.get("split", {})
    lines = [
        "# Forecaster V1 Model Card",
        "",
        "## Summary",
        f"- Version: `{metadata.get('version', 'forecaster_v1')}`",
        f"- Created: `{metadata.get('created_at_utc', '')}`",
        f"- Selected model family: `{metadata.get('selected_model_family', '')}`",
        (
            "- Lease decay lambda: "
            f"`{metadata.get('train_config', {}).get('lease_decay_lambda', 0.0):.6f}`"
        ),
        "",
        "## Data Split",
        f"- Train rows: `{split.get('train_rows', 0)}`",
        f"- Validation rows: `{split.get('val_rows', 0)}`",
        f"- Test rows: `{split.get('test_rows', 0)}`",
        "",
        "## Test Metrics (Selected Model)",
        f"- MAE: `{metrics.get('mae', 0.0):.2f}`",
        f"- RMSE: `{metrics.get('rmse', 0.0):.2f}`",
        f"- MAPE: `{metrics.get('mape', 0.0):.4f}`",
        "",
        "## Candidate Benchmark",
    ]
    for name, vals in bench.items():
        lines.extend(
            [
                f"- `{name}`: "
                f"val_mae={vals.get('val_mae', 0.0):.2f}, "
                f"val_rmse={vals.get('val_rmse', 0.0):.2f}, "
                f"test_mae={vals.get('test_mae', 0.0):.2f}, "
                f"test_rmse={vals.get('test_rmse', 0.0):.2f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Intended Use",
            "- Decision support for Singapore HDB resale price estimation.",
            "- Not a legal valuation or financial advice artifact.",
        ]
    )
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def train_forecaster_v1(df: pd.DataFrame, cfg: TrainConfig = TrainConfig()) -> TrainArtifacts:
    """Train LightGBM and XGBoost, select by validation metrics, and persist artifacts."""
    try:
        from xgboost import XGBRegressor
    except ImportError as ex:
        raise RuntimeError("xgboost is required for forecaster v1. Install project deps.") from ex
    try:
        from lightgbm import LGBMRegressor
    except ImportError as ex:
        raise RuntimeError("lightgbm is required for forecaster v1. Install project deps.") from ex
    try:
        import joblib
    except ImportError as ex:
        raise RuntimeError("joblib is required for model artifact persistence.") from ex

    train_df, val_df, test_df = time_split(df, cfg)
    active_features = list(FEATURE_COLUMNS)
    active_features.extend(c for c in OPTIONAL_LOCATION_COLUMNS if c in train_df.columns)
    if cfg.use_lag_features:
        active_features.extend(c for c in LAG_MARKET_COLUMNS if c in train_df.columns)
    decay_lambda = _fit_lease_decay_lambda(train_df)
    train_df = _apply_lease_decay_features(train_df, decay_lambda)
    val_df = _apply_lease_decay_features(val_df, decay_lambda)
    test_df = _apply_lease_decay_features(test_df, decay_lambda)
    x_train = _one_hot_features(train_df, active_features)
    x_val = _one_hot_features(val_df, active_features).reindex(
        columns=x_train.columns, fill_value=0.0
    )
    x_test = _one_hot_features(test_df, active_features).reindex(
        columns=x_train.columns, fill_value=0.0
    )

    y_train = train_df["resale_price"].to_numpy(dtype=float)
    y_val = val_df["resale_price"].to_numpy(dtype=float)
    y_test = test_df["resale_price"].to_numpy(dtype=float)

    if cfg.deepchecks_enabled:
        _run_deepchecks_validation(df, cfg.deepchecks_output_dir)

    xgb_kw = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "random_state": cfg.random_state,
    }
    xgb_kw.update(cfg.xgboost_params)
    lgbm_kw = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": -1,
        "num_leaves": 63,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": cfg.random_state,
        "verbose": -1,
    }
    lgbm_kw.update(cfg.lightgbm_params)
    candidates: dict[str, Any] = {}
    if "xgboost" in cfg.candidate_models:
        candidates["xgboost"] = XGBRegressor(**xgb_kw)
    if "lightgbm" in cfg.candidate_models:
        candidates["lightgbm"] = LGBMRegressor(**lgbm_kw)
    if not candidates:
        raise ValueError("No candidate models configured; set candidate_models in config.")
    candidate_metrics: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}
    for name, mdl in candidates.items():
        mdl.fit(x_train, np.log1p(y_train))
        p_val = np.expm1(mdl.predict(x_val))
        p_test = np.expm1(mdl.predict(x_test))
        candidate_metrics[name] = {
            "val_mae": float(mean_absolute_error(y_val, p_val)),
            "val_rmse": float(np.sqrt(mean_squared_error(y_val, p_val))),
            "test_mae": float(mean_absolute_error(y_test, p_test)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, p_test))),
        }
        fitted[name] = mdl

    best_name = min(
        candidate_metrics,
        key=lambda n: (candidate_metrics[n]["val_rmse"], candidate_metrics[n]["val_mae"]),
    )
    model = fitted[best_name]
    pred_train = np.expm1(model.predict(x_train))
    pred_val = np.expm1(model.predict(x_val))
    pred_test = np.expm1(model.predict(x_test))
    residuals = y_val - pred_val
    nominal_cov = float(min(max(cfg.interval_nominal_coverage, 0.5), 0.99))
    abs_q = float(np.quantile(np.abs(residuals), nominal_cov))
    low_q = -abs_q
    high_q = abs_q

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.model_dir / "model.joblib"
    metadata_path = cfg.model_dir / "metadata.json"
    config_path = cfg.model_dir / "train_config.json"
    model_card_path = cfg.model_dir / "model_card.md"
    joblib.dump(
        {
            "model": model,
            "model_family": best_name,
            "lease_decay_lambda": decay_lambda,
            "feature_columns": list(x_train.columns),
            "raw_features": active_features,
            "default_feature_values": {
                c: float(train_df[c].median()) if c in train_df.columns else 0.0
                for c in (
                    "floor_area_sqm",
                    "lease_commence_date",
                    "remaining_lease_years",
                    "lease_age_years",
                    "lease_decay_factor",
                    "year",
                    "month_num",
                    "mrt_station_count",
                    "nearest_mrt_km_proxy",
                    "bto_launch_count_town_3y",
                    "bto_avg_price_range_mid_town_3y",
                    "bto_under_construction_units_town",
                    "bto_completed_units_town",
                    "lag_town_median_price",
                    "lag_town_flat_type_median_price",
                    "lag_town_txn_count",
                    "lag_town_flat_type_txn_count",
                )
            },
            "numeric_bounds": {
                col: {
                    "min": float(train_df[col].min()),
                    "max": float(train_df[col].max()),
                }
                for col in (
                    "floor_area_sqm",
                    "lease_commence_date",
                    "remaining_lease_years",
                    "year",
                    "month_num",
                )
                if col in train_df.columns
            },
            "residual_interval": {"q10": float(low_q), "q90": float(high_q)},
            "calibration": {
                "method": "split_conformal_absolute_residual",
                "nominal_coverage": nominal_cov,
                "absolute_residual_quantile": abs_q,
                "validation_rows": int(len(val_df)),
            },
        },
        model_path,
    )

    test_seg = _segment_metrics(test_df.reset_index(drop=True), y_test, pred_test)
    metadata = {
        "version": "forecaster_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selected_model_family": best_name,
        "model_benchmark": candidate_metrics,
        "train_config": {
            "train_frac": cfg.train_frac,
            "val_frac": cfg.val_frac,
            "random_state": cfg.random_state,
            "model_dir": str(cfg.model_dir),
            "candidate_models": cfg.candidate_models,
            "xgboost_params": cfg.xgboost_params,
            "lightgbm_params": cfg.lightgbm_params,
            "lease_decay_lambda": decay_lambda,
            "use_lag_features": cfg.use_lag_features,
            "interval_nominal_coverage": nominal_cov,
        },
        "target": "resale_price",
        "features": active_features,
        "split": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_end_month": str(train_df["month"].max().date()),
            "val_end_month": str(val_df["month"].max().date()),
            "test_end_month": str(test_df["month"].max().date()),
        },
        "calibration": {
            "method": "split_conformal_absolute_residual",
            "nominal_coverage": nominal_cov,
            "absolute_residual_quantile": abs_q,
            "validation_interval_coverage": float(np.mean(np.abs(residuals) <= abs_q)),
            "validation_interval_mean_width": float(2.0 * abs_q),
        },
        "metrics": {
            "train": {
                "mae": float(mean_absolute_error(y_train, pred_train)),
                "rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
            },
            "val": {
                "mae": float(mean_absolute_error(y_val, pred_val)),
                "rmse": float(np.sqrt(mean_squared_error(y_val, pred_val))),
            },
            "test": {
                "mae": float(mean_absolute_error(y_test, pred_test)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
                "mape": float(np.mean(np.abs((y_test - pred_test) / np.clip(y_test, 1.0, None)))),
            },
            "segments": test_seg,
        },
        "provenance": {
            "source_path": (
                str(df["source_path"].iloc[0]) if "source_path" in df.columns else ""
            ),
            "source_sha256": (
                str(df["source_sha256"].iloc[0]) if "source_sha256" in df.columns else ""
            ),
            "ingested_at_utc": (
                str(df["ingested_at_utc"].iloc[0]) if "ingested_at_utc" in df.columns else ""
            ),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "train_frac": cfg.train_frac,
                "val_frac": cfg.val_frac,
                "random_state": cfg.random_state,
                "model_dir": str(cfg.model_dir),
                "candidate_models": cfg.candidate_models,
                "xgboost_params": cfg.xgboost_params,
                "lightgbm_params": cfg.lightgbm_params,
                "mlflow_enabled": cfg.mlflow_enabled,
                "mlflow_tracking_uri": cfg.mlflow_tracking_uri,
                "mlflow_experiment": cfg.mlflow_experiment,
                "deepchecks_enabled": cfg.deepchecks_enabled,
                "deepchecks_output_dir": str(cfg.deepchecks_output_dir),
                "use_lag_features": cfg.use_lag_features,
                "interval_nominal_coverage": nominal_cov,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_model_card(metadata=metadata, output_path=model_card_path)
    _log_mlflow_run(
        cfg,
        best_name=best_name,
        candidate_metrics=candidate_metrics,
        selected_metrics=metadata["metrics"]["test"],
        model_path=model_path,
        metadata_path=metadata_path,
        config_path=config_path,
        model_card_path=model_card_path,
    )
    return TrainArtifacts(
        model_path=model_path,
        metadata_path=metadata_path,
        config_path=config_path,
    )


def _run_deepchecks_validation(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_note = output_dir / "deepchecks_unavailable.txt"
    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import data_integrity
    except Exception as ex:  # noqa: BLE001
        failure_note.write_text(
            "Deepchecks validation could not run in this environment.\n"
            f"Reason: {ex}\n",
            encoding="utf-8",
        )
        return
    label = "resale_price" if "resale_price" in df.columns else None
    ds = Dataset(df, label=label, cat_features=["town", "flat_type", "flat_model", "storey_range"])
    suite = data_integrity()
    try:
        result = suite.run(ds)
        result.save_as_html(str(output_dir / "data_integrity.html"))
    except Exception as ex:  # noqa: BLE001
        failure_note.write_text(
            "Deepchecks suite execution failed.\n"
            f"Reason: {ex}\n",
            encoding="utf-8",
        )


def run_deepchecks_validation(df: pd.DataFrame, output_dir: Path) -> None:
    """Public wrapper for Deepchecks validation."""
    _run_deepchecks_validation(df, output_dir)


def _log_mlflow_run(
    cfg: TrainConfig,
    *,
    best_name: str,
    candidate_metrics: dict[str, dict[str, float]],
    selected_metrics: dict[str, Any],
    model_path: Path,
    metadata_path: Path,
    config_path: Path,
    model_card_path: Path | None = None,
) -> None:
    if not cfg.mlflow_enabled:
        return
    try:
        import mlflow
    except ImportError as ex:
        raise RuntimeError("mlflow is enabled but not installed.") from ex

    if cfg.mlflow_tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)
    with mlflow.start_run(run_name="forecaster_v1_train"):
        mlflow.log_param("selected_model_family", best_name)
        mlflow.log_param("train_frac", cfg.train_frac)
        mlflow.log_param("val_frac", cfg.val_frac)
        mlflow.log_param("random_state", cfg.random_state)
        mlflow.log_param("candidate_models", ",".join(cfg.candidate_models))
        for family, vals in candidate_metrics.items():
            for k, v in vals.items():
                mlflow.log_metric(f"{family}_{k}", float(v))
        for k, v in selected_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"selected_{k}", float(v))
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metadata_path))
        mlflow.log_artifact(str(config_path))
        if model_card_path and model_card_path.exists():
            mlflow.log_artifact(str(model_card_path))


def _row_to_frame(row: dict[str, Any], raw_features: list[str]) -> pd.DataFrame:
    out: dict[str, Any] = {k: row.get(k) for k in raw_features}
    month = pd.to_datetime(row.get("month"), errors="coerce")
    if pd.isna(month):
        raise ValueError("Field 'month' is required and must be parseable as a date.")
    out["year"] = int(month.year)
    out["month_num"] = int(month.month)
    frame = pd.DataFrame([out])
    frame = add_bto_reference_features(frame)
    return frame


def predict_with_explain(
    row: dict[str, Any],
    *,
    model_path: Path = Path("models/forecaster_v1/model.joblib"),
    metadata_path: Path = Path("models/forecaster_v1/metadata.json"),
) -> dict[str, Any]:
    """Predict with local SHAP explanation and interval."""
    try:
        import joblib
        import shap
    except ImportError as ex:
        raise RuntimeError("joblib and shap are required for prediction explainability.") from ex

    blob = joblib.load(model_path)
    model = blob["model"]
    model_family = str(blob.get("model_family", "xgboost"))
    decay_lambda = float(blob.get("lease_decay_lambda", 0.01))
    raw_features: list[str] = blob["raw_features"]
    train_cols: list[str] = blob["feature_columns"]
    default_feature_values: dict[str, float] = blob.get("default_feature_values", {})
    bounds: dict[str, dict[str, float]] = blob["numeric_bounds"]
    residual_interval: dict[str, float] = blob["residual_interval"]
    calibration: dict[str, Any] = blob.get("calibration", {})

    row_df = _row_to_frame(row, raw_features)
    row_df = _apply_lease_decay_features(row_df, decay_lambda)
    for k, v in default_feature_values.items():
        if k not in row_df.columns or pd.isna(row_df[k].iloc[0]):
            row_df[k] = float(v)
    x = _one_hot_features(row_df, raw_features).reindex(columns=train_cols, fill_value=0.0)
    y_pred = float(np.expm1(model.predict(x)[0]))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    shap_row = np.asarray(shap_values)[0]
    contribs = (
        pd.Series(shap_row, index=x.columns)
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(8)
    )
    top_contrib = [{"feature": str(k), "shap_log_price": float(v)} for k, v in contribs.items()]

    out_of_range: list[str] = []
    lag_missing: list[str] = []
    for lag_col in LAG_MARKET_COLUMNS:
        if lag_col in raw_features and lag_col not in row:
            lag_missing.append(lag_col)
    for key, lim in bounds.items():
        v = row_df[key].iloc[0] if key in row_df.columns else None
        if v is None or pd.isna(v):
            continue
        if float(v) < lim["min"] or float(v) > lim["max"]:
            out_of_range.append(key)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if calibration.get("method") == "split_conformal_absolute_residual":
        width = float(calibration.get("absolute_residual_quantile", 0.0))
        low = max(0.0, y_pred - width)
        high = max(low, y_pred + width)
    else:
        low = max(0.0, y_pred + float(residual_interval["q10"]))
        high = max(low, y_pred + float(residual_interval["q90"]))
    return {
        "prediction": y_pred,
        "prediction_interval": {"p10": low, "p90": high},
        "top_contributors": top_contrib,
        "model_version": metadata.get("version", "forecaster_v1"),
        "model_family": metadata.get("selected_model_family", model_family),
        "lease_decay_lambda": decay_lambda,
        "warnings": (
            (
                [f"Input outside training range for: {', '.join(out_of_range)}"]
                if out_of_range
                else []
            )
            + (
                [
                    "Lag market features missing in request; falling back to training medians "
                    f"for {', '.join(lag_missing)}."
                ]
                if lag_missing
                else []
            )
        ),
    }
