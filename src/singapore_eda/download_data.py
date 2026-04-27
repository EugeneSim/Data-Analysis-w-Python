"""Download HDB resale data from data.gov.sg paginated API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from singapore_eda.constants import (
    DEFAULT_RAW_CSV,
    HDB_CITATION_URL,
    HDB_RESALE_RESOURCE_ID,
    HDB_RESALE_TRANCHES,
    PAGE_SIZE,
)
from singapore_eda.datastore import download_paginated_resource


def tranche_by_label(label: str) -> dict:
    for t in HDB_RESALE_TRANCHES:
        if t.get("label") == label:
            return t
    raise KeyError(
        f"Unknown tranche label {label!r}. Keys: {[t.get('label') for t in HDB_RESALE_TRANCHES]}"
    )


def download_hdb_resale(
    out_path: Path,
    *,
    resource_id: str | None = None,
    max_rows: int | None = None,
    page_size: int = PAGE_SIZE,
    skip_if_fresh_hours: float | None = None,
    latest_first: bool = False,
) -> int:
    """Fetch all (or up to max_rows) records and write a CSV. Returns row count written."""
    rid = resource_id or HDB_RESALE_RESOURCE_ID
    qp: dict[str, str] | None = None
    if latest_first:
        # For capped sample pulls, return newest records first.
        qp = {"sort": "month desc"}
    return download_paginated_resource(
        rid,
        Path(out_path),
        max_rows=max_rows,
        page_size=page_size,
        skip_if_fresh_hours=skip_if_fresh_hours,
        query_params=qp,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download HDB resale flat prices from data.gov.sg into CSV."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_RAW_CSV,
        help="Output CSV path",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after this many rows (for testing; full dataset is large)",
    )
    parser.add_argument(
        "--latest-first",
        action="store_true",
        help=(
            "When used with --max-rows, sample newest records first (`sort=month desc`) "
            "instead of oldest rows from offset 0."
        ),
    )
    parser.add_argument(
        "--resource-id",
        type=str,
        default=None,
        help="data.gov.sg datastore resource id (overrides default Jan-2017+ tranche)",
    )
    parser.add_argument(
        "--tranche",
        type=str,
        default=None,
        help="Tranche label from HDB_RESALE_TRANCHES (see --list-tranches)",
    )
    parser.add_argument(
        "--list-tranches",
        action="store_true",
        help="Print known resale tranche labels and resource ids, then exit",
    )
    parser.add_argument(
        "--skip-if-fresh-hours",
        type=float,
        default=None,
        metavar="H",
        help=(
            "If the output file exists and is newer than H hours, skip the API and keep "
            "the file. Same as env SINGAPORE_EDA_SKIP_DOWNLOAD_IF_FRESH_HOURS. "
            "Use this to avoid unnecessary data.gov.sg calls in cron jobs."
        ),
    )
    parser.add_argument(
        "--check-new-data",
        action="store_true",
        help=(
            "With --skip-if-fresh-hours: if sidecar .meta.json shows a lower api_total than "
            "the live CKAN resource, re-download anyway. Sets SINGAPORE_EDA_CHECK_NEW_DATA=1."
        ),
    )
    args = parser.parse_args()
    if args.list_tranches:
        for t in HDB_RESALE_TRANCHES:
            print(
                json.dumps(
                    {k: v for k, v in t.items() if not str(k).startswith("_")},
                )
            )
        return
    rid = args.resource_id
    if args.tranche:
        rid = tranche_by_label(args.tranche)["id"]
    if args.check_new_data:
        os.environ["SINGAPORE_EDA_CHECK_NEW_DATA"] = "1"
    n = download_hdb_resale(
        args.output,
        resource_id=rid,
        max_rows=args.max_rows,
        skip_if_fresh_hours=args.skip_if_fresh_hours,
        latest_first=args.latest_first,
    )
    print(f"Wrote {n} rows to {args.output.resolve()}", file=sys.stderr)
    print(f"Source: {HDB_CITATION_URL}", file=sys.stderr)


if __name__ == "__main__":
    main()
