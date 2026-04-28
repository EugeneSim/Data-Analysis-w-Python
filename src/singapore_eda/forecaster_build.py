"""CLI entrypoint for building forecaster v1 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from singapore_eda.constants import DEFAULT_RAW_CSV
from singapore_eda.feature_store import write_curated_to_postgres
from singapore_eda.forecaster_config import load_forecaster_config
from singapore_eda.forecaster_v1 import TrainConfig, build_training_frame, train_forecaster_v1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_RAW_CSV, help="Input resale CSV path")
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/forecaster_v1"),
        help="Output model artifact directory",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/forecaster_v1.yaml"),
        help="YAML config for hyperparameters, tracking, and validation options",
    )
    p.add_argument(
        "--postgres-dsn",
        type=str,
        default="",
        help="Optional Postgres DSN. If set, curated training frame is written to Postgres.",
    )
    args = p.parse_args()

    frame = build_training_frame(args.input)
    if args.postgres_dsn:
        n = write_curated_to_postgres(frame, dsn=args.postgres_dsn)
        print(f"Wrote {n:,} curated rows to Postgres.")

    cfg_file = load_forecaster_config(args.config)
    cfg = TrainConfig(
        train_frac=cfg_file.train_frac,
        val_frac=cfg_file.val_frac,
        random_state=cfg_file.random_state,
        model_dir=args.model_dir or cfg_file.model_dir,
        candidate_models=cfg_file.candidate_models,
        xgboost_params=cfg_file.xgboost_params,
        lightgbm_params=cfg_file.lightgbm_params,
        mlflow_enabled=cfg_file.mlflow_enabled,
        mlflow_tracking_uri=cfg_file.mlflow_tracking_uri,
        mlflow_experiment=cfg_file.mlflow_experiment,
        deepchecks_enabled=cfg_file.deepchecks_enabled,
        deepchecks_output_dir=cfg_file.deepchecks_output_dir,
        use_lag_features=cfg_file.use_lag_features,
        interval_nominal_coverage=cfg_file.interval_nominal_coverage,
    )
    artifacts = train_forecaster_v1(frame, cfg=cfg)
    print(
        json.dumps(
            {
                "model_path": str(artifacts.model_path),
                "metadata_path": str(artifacts.metadata_path),
                "config_path": str(artifacts.config_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
