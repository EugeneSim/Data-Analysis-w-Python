"""Download and materialize HDB BTO historical and future-facing reference datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from singapore_eda.constants import (
    DEFAULT_BTO_COMPLETION_STATUS_CSV,
    DEFAULT_BTO_PRICE_RANGE_CSV,
    DEFAULT_HDB_PROPERTY_INFO_CSV,
    HDB_BTO_DATASETS,
    PAGE_SIZE,
)
from singapore_eda.datastore import download_paginated_resource


def _default_output(name: str) -> Path:
    if name == "bto_price_range":
        return DEFAULT_BTO_PRICE_RANGE_CSV
    if name == "bto_completion_status":
        return DEFAULT_BTO_COMPLETION_STATUS_CSV
    if name == "hdb_property_information":
        return DEFAULT_HDB_PROPERTY_INFO_CSV
    return Path("data/reference") / f"{name}.csv"


def download_bto_data(
    *,
    out_dir: Path | str = Path("data/reference"),
    max_rows: int | None = None,
    skip_if_fresh_hours: float | None = None,
) -> list[dict[str, Any]]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for ds in HDB_BTO_DATASETS:
        out_path = out_root / _default_output(str(ds["name"])).name
        # HDB property dataset can return 413 on large page-size requests.
        page_size = 2000 if str(ds["name"]) == "hdb_property_information" else PAGE_SIZE
        n = download_paginated_resource(
            str(ds["resource_id"]),
            out_path,
            max_rows=max_rows,
            page_size=page_size,
            skip_if_fresh_hours=skip_if_fresh_hours,
        )
        rows.append(
            {
                "name": str(ds["name"]),
                "kind": str(ds["kind"]),
                "resource_id": str(ds["resource_id"]),
                "citation_url": str(ds["citation_url"]),
                "path": str(out_path),
                "rows": int(n),
            }
        )
    _write_bto_reference_snapshot(out_root)
    return rows


def _write_bto_reference_snapshot(out_root: Path) -> None:
    # Build a lightweight, UI-friendly BTO seed table from historical launch pricing data.
    src = out_root / DEFAULT_BTO_PRICE_RANGE_CSV.name
    if not src.exists():
        return
    bto = pd.read_csv(src)
    bto.columns = [str(c).strip().lower().replace(" ", "_") for c in bto.columns]
    needed = {"financial_year", "town", "room_type"}
    if not needed.issubset(bto.columns):
        return
    out = bto[list(needed)].copy()
    out["project_name"] = out["financial_year"].astype(str).map(lambda y: f"BTO {y}")
    out["flat_type"] = out["room_type"].astype(str).str.upper()
    out["flat_model"] = "MODEL A"
    out["lease_commence_date"] = (
        pd.to_numeric(out["financial_year"], errors="coerce").fillna(0).astype(int)
    )
    out = out.rename(columns={"town": "town"})
    out = out[
        ["project_name", "town", "flat_type", "flat_model", "lease_commence_date"]
    ].drop_duplicates()
    out = out[out["lease_commence_date"] > 0]
    out.to_csv(out_root / "hdb_bto_reference.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download HDB BTO historical + future-facing datasets from data.gov.sg."
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("data/reference"),
        help="Directory to write BTO CSV outputs",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after this many rows per dataset (for development/testing)",
    )
    p.add_argument(
        "--skip-if-fresh-hours",
        type=float,
        default=None,
        metavar="H",
        help="If existing file is younger than H hours, skip API call",
    )
    args = p.parse_args()
    summary = download_bto_data(
        out_dir=args.output_dir,
        max_rows=args.max_rows,
        skip_if_fresh_hours=args.skip_if_fresh_hours,
    )
    print(json.dumps(summary, indent=2))
    print(
        "BTO reference snapshot: "
        f"{(Path(args.output_dir) / 'hdb_bto_reference.csv').resolve()}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
