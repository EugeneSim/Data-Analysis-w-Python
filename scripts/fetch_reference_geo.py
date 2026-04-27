#!/usr/bin/env python3
"""Download planning-area GeoJSON via data.gov.sg poll-download.

Run from repo root with the package installed (``pip install -e .``).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

from singapore_eda.constants import PLANNING_AREA_GEOJSON_POLL_URL
from singapore_eda.gov_http import get_gov_client
from singapore_eda.gov_limits import skip_geo_fresh_days_default

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch planning-area GeoJSON via open data.gov.sg poll-download."
    )
    parser.add_argument(
        "--skip-if-fresh-days",
        type=float,
        default=None,
        metavar="D",
        help=(
            "If the output file exists and is newer than D days, skip all network calls. "
            "Same as env SINGAPORE_EDA_GEO_SKIP_IF_FRESH_DAYS."
        ),
    )
    args = parser.parse_args()

    out = ROOT / "data" / "reference" / "planning_areas.geojson"
    out.parent.mkdir(parents=True, exist_ok=True)

    skip_d: float | None = args.skip_if_fresh_days
    if skip_d is None:
        skip_d = skip_geo_fresh_days_default()
    if (
        skip_d is not None
        and skip_d > 0
        and out.is_file()
        and out.stat().st_size > 0
    ):
        age_d = (time.time() - out.stat().st_mtime) / 86400.0
        if age_d < skip_d:
            print(
                f"Skip fetch: {out} is {age_d:.2f}d old (< {skip_d}d) — no API calls.",
                file=sys.stderr,
            )
            return

    try:
        # Official: Master Plan 2025 Planning Area Boundary (No Sea)
        # https://data.gov.sg/datasets/d_2cc750190544007400b2cfd5d7f53209/view
        # Poll + pre-signed URL: use *file* tier pacing (dataset-style limits on open API).
        body = get_gov_client().get_json(
            PLANNING_AREA_GEOJSON_POLL_URL,
            timeout=90,
            use_cache=False,
            use_file_pace=True,
        )
        if body.get("code") != 0:
            err = (body.get("errMsg") or body.get("errorMsg") or "") or "unknown error"
            raise RuntimeError(f"poll-download error: {err}")
        blob_url = (body.get("data") or {}).get("url")
        if not blob_url:
            raise RuntimeError("poll-download response missing data.url")
        r = requests.get(str(blob_url), timeout=180)
        r.raise_for_status()
    except (OSError, ValueError, requests.RequestException, RuntimeError) as e:
        print(
            f"Download failed: {e}\n"
            f"  API: {PLANNING_AREA_GEOJSON_POLL_URL}\n"
            "  Dataset: Master Plan 2025 Planning Area Boundary (No Sea) — see "
            "https://data.gov.sg/datasets/d_2cc750190544007400b2cfd5d7f53209/view\n"
            "  (Legacy geo.data.gov.sg MP14 static links are no longer available.)",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    out.write_bytes(r.content)
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
