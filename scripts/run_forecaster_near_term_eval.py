#!/usr/bin/env python3
"""Near-term forecaster evaluation: rolling backtests, calibration, and ablation."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from singapore_eda.constants import DEFAULT_RAW_CSV
from singapore_eda.forecaster_config import load_forecaster_config
from singapore_eda.forecaster_v1 import (
    OPTIONAL_LOCATION_COLUMNS,
    TrainConfig,
    build_training_frame,
    predict_with_explain,
    train_forecaster_v1,
)


def _row_payload(row: pd.Series) -> dict[str, object]:
    return {
        "month": str(pd.to_datetime(row["month"]).date()),
        "town": str(row["town"]),
        "flat_type": str(row["flat_type"]),
        "flat_model": str(row["flat_model"]),
        "storey_range": str(row["storey_range"]),
        "floor_area_sqm": float(row["floor_area_sqm"]),
        "lease_commence_date": float(row["lease_commence_date"]),
        "remaining_lease_years": float(row["remaining_lease_years"]),
    }


def _rolling_windows(
    months: list[pd.Timestamp],
    *,
    min_train_months: int = 24,
    horizon_months: int = 3,
    step: int = 3,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    end = len(months) - horizon_months
    for i in range(min_train_months, end, step):
        out.append((months[i - 1], months[i + horizon_months - 1]))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_RAW_CSV)
    p.add_argument("--output-dir", type=Path, default=Path("reports/forecaster_v1"))
    p.add_argument("--config", type=Path, default=Path("configs/forecaster_v1.yaml"))
    p.add_argument(
        "--enforce-gate",
        action="store_true",
        help="Exit non-zero when rolling-window minimum or promotion gate fails.",
    )
    args = p.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_forecaster_config(args.config)
    nominal = float(min(max(cfg.interval_nominal_coverage, 0.5), 0.99))
    coverage_tol = float(max(0.01, min(0.49, getattr(cfg, "coverage_tolerance", 0.10))))
    min_windows = int(max(1, getattr(cfg, "rolling_min_windows", 3)))
    max_mean_width = float(getattr(cfg, "max_mean_interval_width", 220000.0))

    frame = build_training_frame(args.input).sort_values("month").reset_index(drop=True)
    month_keys = sorted(frame["month"].dt.to_period("M").dt.to_timestamp().unique().tolist())
    horizon_months = 3
    min_train_months = max(12, min(24, max(12, len(month_keys) - horizon_months - 1)))
    windows = _rolling_windows(
        month_keys,
        min_train_months=min_train_months,
        horizon_months=horizon_months,
        step=3,
    )
    if not windows and len(month_keys) > horizon_months + 6:
        cutoff = len(month_keys) - horizon_months
        windows = [(month_keys[cutoff - 1], month_keys[-1])]

    rolling_rows: list[dict[str, object]] = []
    rolling_segment_rows: list[dict[str, object]] = []
    all_hits: list[bool] = []
    all_widths: list[float] = []
    for i, (train_end, test_end) in enumerate(windows, start=1):
        train_df = frame.loc[frame["month"] <= train_end].copy()
        test_df = frame.loc[(frame["month"] > train_end) & (frame["month"] <= test_end)].copy()
        if len(train_df) < 100 or len(test_df) < 10:
            continue
        with tempfile.TemporaryDirectory(prefix=f"forecaster_bt_{i}_") as td:
            cfg = TrainConfig(model_dir=Path(td), mlflow_enabled=False, deepchecks_enabled=False)
            artifacts = train_forecaster_v1(train_df, cfg=cfg)
            preds: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for _, row in test_df.iterrows():
                r = predict_with_explain(
                    _row_payload(row),
                    model_path=artifacts.model_path,
                    metadata_path=artifacts.metadata_path,
                )
                preds.append(float(r["prediction"]))
                lows.append(float(r["prediction_interval"]["p10"]))
                highs.append(float(r["prediction_interval"]["p90"]))
            y = test_df["resale_price"].to_numpy(dtype=float)
            p_arr = np.asarray(preds, dtype=float)
            rmse = float(np.sqrt(np.mean((y - p_arr) ** 2)))
            mae = float(np.mean(np.abs(y - p_arr)))
            hit = (y >= np.asarray(lows)) & (y <= np.asarray(highs))
            coverage = float(hit.mean()) if len(hit) else float("nan")
            width = (np.asarray(highs, dtype=float) - np.asarray(lows, dtype=float)).astype(float)
            mean_width = float(np.mean(width)) if len(width) else float("nan")
            all_hits.extend(hit.tolist())
            all_widths.extend(width.tolist())
            rolling_rows.append(
                {
                    "window": i,
                    "train_end": str(train_end.date()),
                    "test_end": str(test_end.date()),
                    "rows": int(len(test_df)),
                    "rmse": rmse,
                    "mae": mae,
                    "interval_coverage_p10_p90": coverage,
                    "mean_interval_width": mean_width,
                }
            )
            seg = test_df[["town"]].copy()
            seg["hit"] = hit.astype(int)
            for town, grp in seg.groupby("town"):
                if len(grp) < 10:
                    continue
                rolling_segment_rows.append(
                    {
                        "window": i,
                        "town": str(town),
                        "rows": int(len(grp)),
                        "coverage": float(grp["hit"].mean()),
                    }
                )

    min_windows_pass = len(rolling_rows) >= min_windows

    # Exogenous/location ablation: compare without and with location/accessibility columns.
    with tempfile.TemporaryDirectory(prefix="forecaster_ablation_base_") as td_base:
        base_cfg = TrainConfig(
            model_dir=Path(td_base),
            mlflow_enabled=False,
            deepchecks_enabled=False,
        )
        base_df = frame.drop(columns=[c for c in OPTIONAL_LOCATION_COLUMNS if c in frame.columns])
        base_art = train_forecaster_v1(base_df, cfg=base_cfg)
        base_meta = json.loads(base_art.metadata_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="forecaster_ablation_full_") as td_full:
        full_cfg = TrainConfig(
            model_dir=Path(td_full),
            mlflow_enabled=False,
            deepchecks_enabled=False,
        )
        full_art = train_forecaster_v1(frame, cfg=full_cfg)
        full_meta = json.loads(full_art.metadata_path.read_text(encoding="utf-8"))

    overall_cov = float(np.mean(all_hits)) if all_hits else float("nan")
    mean_width_all = float(np.mean(all_widths)) if all_widths else float("nan")
    coverage_pass = bool(abs(overall_cov - nominal) <= coverage_tol)
    width_pass = bool(mean_width_all <= max_mean_width)

    out_json = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "rolling_origin": {
            "windows": rolling_rows,
            "mean_rmse": (
                float(np.mean([r["rmse"] for r in rolling_rows])) if rolling_rows else float("nan")
            ),
            "mean_mae": (
                float(np.mean([r["mae"] for r in rolling_rows])) if rolling_rows else float("nan")
            ),
            "overall_interval_coverage_p10_p90": overall_cov,
            "mean_interval_width": mean_width_all,
            "segment_coverage_by_town": rolling_segment_rows,
        },
        "ablation": {
            "without_location_features": base_meta.get("metrics", {}).get("test", {}),
            "with_location_features": full_meta.get("metrics", {}).get("test", {}),
        },
        "promotion_gate": {
            "nominal_coverage": nominal,
            "coverage_tolerance": coverage_tol,
            "min_windows": min_windows,
            "max_mean_interval_width": max_mean_width,
            "min_windows_pass": min_windows_pass,
            "coverage_pass": coverage_pass,
            "width_pass": width_pass,
        },
    }
    out_json["promotion_gate"]["pass"] = bool(
        out_json["promotion_gate"]["min_windows_pass"]
        and out_json["promotion_gate"]["coverage_pass"]
        and out_json["promotion_gate"]["width_pass"]
    )
    json_path = out_dir / "near_term_eval.json"
    json_path.write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    md_lines = [
        "# Near-term Evaluation Report",
        "",
        f"- Generated: `{out_json['created_at_utc']}`",
        f"- Rolling windows evaluated: `{len(rolling_rows)}`",
        f"- Mean rolling RMSE: `{out_json['rolling_origin']['mean_rmse']:.2f}`",
        f"- Mean rolling MAE: `{out_json['rolling_origin']['mean_mae']:.2f}`",
        (
            "- Interval coverage (p10-p90): "
            f"`{out_json['rolling_origin']['overall_interval_coverage_p10_p90']:.3f}`"
        ),
        (
            "- Mean interval width: "
            f"`{out_json['rolling_origin']['mean_interval_width']:.2f}`"
        ),
        "",
        "## Exogenous/Location Ablation (test split)",
        (
            "- Without location features RMSE: "
            f"`{out_json['ablation']['without_location_features'].get('rmse', float('nan')):.2f}`"
        ),
        (
            "- With location features RMSE: "
            f"`{out_json['ablation']['with_location_features'].get('rmse', float('nan')):.2f}`"
        ),
        "",
        "Artifacts:",
        f"- JSON: `{json_path}`",
        (
            "- Promotion gate: "
            f"`{'PASS' if out_json['promotion_gate']['pass'] else 'FAIL'}` "
            f"(coverage_pass={out_json['promotion_gate']['coverage_pass']}, "
            f"width_pass={out_json['promotion_gate']['width_pass']})"
        ),
    ]
    (out_dir / "near_term_eval.md").write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")
    print(f"Wrote near-term evaluation artifacts to {out_dir}")
    if args.enforce_gate and not out_json["promotion_gate"]["pass"]:
        raise SystemExit(
            "Near-term evaluation promotion gate failed: "
            f"{json.dumps(out_json['promotion_gate'])}"
        )


if __name__ == "__main__":
    main()

