"""Download or load median HDB rent for yield estimates."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from singapore_eda.constants import (
    DEFAULT_RENT_CSV,
    HDB_MEDIAN_RENT_CITATION_URL,
)
from singapore_eda.datastore import download_paginated_resource


def download_median_rent(
    out_path: Path,
    resource_id: str,
    *,
    max_rows: int | None = None,
    skip_if_fresh_hours: float | None = None,
) -> int:
    """Paginated CKAN download; returns row count."""
    return download_paginated_resource(
        resource_id,
        Path(out_path),
        max_rows=max_rows,
        skip_if_fresh_hours=skip_if_fresh_hours,
    )


def main() -> None:
    from singapore_eda import constants

    rid = os.environ.get("HDB_MEDIAN_RENT_RESOURCE_ID", "").strip() or getattr(
        constants, "HDB_MEDIAN_RENT_RESOURCE_ID", ""
    )
    if not rid:
        print(
            "Set HDB_MEDIAN_RENT_RESOURCE_ID to a data.gov.sg resource id.",
            file=sys.stderr,
        )
        sys.exit(1)
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_RENT_CSV)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument(
        "--skip-if-fresh-hours",
        type=float,
        default=None,
        metavar="H",
        help="Skip download if output exists and is newer than H hours (no API calls).",
    )
    args = p.parse_args()
    n = download_median_rent(
        args.output,
        rid,
        max_rows=args.max_rows,
        skip_if_fresh_hours=args.skip_if_fresh_hours,
    )
    print(f"Wrote {n} rows to {args.output}", file=sys.stderr)
    print(f"Source: {HDB_MEDIAN_RENT_CITATION_URL}", file=sys.stderr)


if __name__ == "__main__":
    main()
