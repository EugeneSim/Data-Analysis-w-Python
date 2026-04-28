#!/usr/bin/env python3
"""Prepare validated/retraining feedback datasets from raw feedback log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from singapore_eda.feedback import materialize_feedback_views
from singapore_eda.forecaster_config import load_forecaster_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/forecaster_v1.yaml"),
        help="Forecaster YAML config path.",
    )
    p.add_argument(
        "--raw-path",
        type=Path,
        default=None,
        help="Optional raw feedback CSV override.",
    )
    p.add_argument(
        "--validated-path",
        type=Path,
        default=None,
        help="Optional validated feedback CSV override.",
    )
    p.add_argument(
        "--retraining-path",
        type=Path,
        default=None,
        help="Optional retraining-eligible feedback CSV override.",
    )
    p.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help="Optional retention window in days.",
    )
    args = p.parse_args()

    cfg = load_forecaster_config(args.config)
    stats = materialize_feedback_views(
        raw_path=args.raw_path or cfg.feedback_store_path,
        validated_path=args.validated_path or cfg.feedback_validated_path,
        retraining_path=args.retraining_path or cfg.feedback_retraining_path,
        retention_days=args.retention_days or cfg.feedback_retention_days,
        min_comment_redact_digits=cfg.feedback_min_comment_redact_digits,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
