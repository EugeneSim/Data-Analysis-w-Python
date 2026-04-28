#!/usr/bin/env python3
"""Run Deepchecks data integrity suite for forecaster input frame."""

from __future__ import annotations

from pathlib import Path

from singapore_eda.forecaster_config import load_forecaster_config
from singapore_eda.forecaster_v1 import build_training_frame, run_deepchecks_validation


def main() -> None:
    cfg = load_forecaster_config(Path("configs/forecaster_v1.yaml"))
    df = build_training_frame("data/raw/hdb_resale_2017_onwards.csv")
    run_deepchecks_validation(df, cfg.deepchecks_output_dir)
    print(f"Deepchecks report generated in: {cfg.deepchecks_output_dir}")


if __name__ == "__main__":
    main()
