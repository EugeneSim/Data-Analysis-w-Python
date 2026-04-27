"""CLI: merge several resale CSV tranches (standardize + optional year window)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from singapore_eda.history import merge_resale_tranches


def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge HDB resale CSVs with schema standardization (see history module)."
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("data/processed/hdb_merged_resale.csv"),
        help="Output path (.csv or .parquet).",
    )
    p.add_argument(
        "--parquet",
        action="store_true",
        help="Write Parquet (also implied if --out ends with .parquet).",
    )
    p.add_argument("--year-min", type=int, default=None, dest="year_min")
    p.add_argument("--year-max", type=int, default=None, dest="year_max")
    p.add_argument(
        "csvs",
        nargs="+",
        type=Path,
        help="Resale CSV files in chronological tranche order",
    )
    p.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="data_period label per file (default: tranche_0, tranche_1, ...)",
    )
    args = p.parse_args()
    n = len(args.csvs)
    if args.labels is None or len(args.labels) == 0:
        labels = [f"tranche_{i}" for i in range(n)]
    elif len(args.labels) != n:
        print("Error: need one --labels entry per input CSV", file=sys.stderr)
        raise SystemExit(1)
    else:
        labels = list(args.labels)
    tranches = list(zip(args.csvs, labels, strict=True))
    m = merge_resale_tranches(
        tranches,
        args.out,
        year_min=args.year_min,
        year_max=args.year_max,
        as_parquet=args.parquet,
    )
    print(f"Wrote {len(m):,} rows to {Path(args.out).resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
