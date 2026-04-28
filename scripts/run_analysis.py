#!/usr/bin/env python3
"""End-to-end: ingest -> clean -> features -> OLS and insights (prints summary)."""

from __future__ import annotations

import json
from pathlib import Path

from singapore_eda.bto_ingest import download_bto_data
from singapore_eda.clean import clean_hdb
from singapore_eda.constants import DEFAULT_PROCESSED_PARQUET, DEFAULT_RAW_CSV
from singapore_eda.features import add_features, model_design_subset
from singapore_eda.ingest import read_hdb_csv
from singapore_eda.insights import build_insights
from singapore_eda.stats import ols_log_price


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_RAW_CSV, help="Raw CSV")
    p.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PROCESSED_PARQUET,
        help="Write cleaned parquet here",
    )
    p.add_argument(
        "--refresh-bto",
        action="store_true",
        help="Download latest BTO historical + future-facing reference datasets first.",
    )
    args = p.parse_args()

    if args.refresh_bto:
        download_bto_data(out_dir=Path("data/reference"))

    raw, _meta = read_hdb_csv(args.input)
    clean = clean_hdb(raw)
    clean.to_parquet(args.parquet, index=False, engine="pyarrow")
    feat = add_features(clean)
    model_df = model_design_subset(feat)
    model = ols_log_price(model_df)
    insight = build_insights(feat, model=model)
    insight["n_model_rows"] = int(len(model_df))
    print(json.dumps({k: v for k, v in insight.items()}, indent=2, default=str))
    print("\n--- OLS summary (tail) ---\n", model.summary().as_text()[-2000:])


if __name__ == "__main__":
    main()
