#!/usr/bin/env python3
"""Fetch MRT reference files from official Singapore government open data APIs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]

# Official datasets on data.gov.sg / URA collections.
DATASET_MRT_SYMBOL = "d_649357d5cb04ddbef9166dfcf1fa8d21"
DATASET_MP03_MRT_NAME = "d_dbc192abee39f51efecc0adbe9f1a75d"


def _poll_download_url(dataset_id: str) -> str:
    url = f"https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}/poll-download"
    body = requests.get(url, timeout=60).json()
    code = int(body.get("code", 1))
    if code == 24:
        # Respect documented public rate limits.
        time.sleep(11)
        body = requests.get(url, timeout=60).json()
        code = int(body.get("code", 1))
    if code != 0:
        err = body.get("errorMsg") or body.get("errMsg") or "unknown error"
        raise RuntimeError(f"poll-download failed for {dataset_id}: {err}")
    blob = (body.get("data") or {}).get("url")
    if not blob:
        raise RuntimeError(f"poll-download missing data.url for {dataset_id}")
    return str(blob)


def _download_json(dataset_id: str) -> dict:
    blob = _poll_download_url(dataset_id)
    resp = requests.get(blob, timeout=180)
    resp.raise_for_status()
    return resp.json()


def _write_geojson(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _extract_station_names(mp03_geojson: dict) -> list[str]:
    out: set[str] = set()
    for feat in mp03_geojson.get("features", []):
        props = feat.get("properties", {}) or {}
        txt = str(props.get("TEXTSTRING", "")).strip().upper()
        if txt:
            out.add(txt)
    return sorted(out)


def _write_future_csv(path: Path, station_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["station_name", "planned_line", "earliest_opening_year", "notes"],
        )
        w.writeheader()
        for name in station_names:
            w.writerow(
                {
                    "station_name": name,
                    "planned_line": "",
                    "earliest_opening_year": "",
                    "notes": (
                        "Official URA Master Plan 2003 MRT Name annotation dataset "
                        f"({DATASET_MP03_MRT_NAME})"
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MRT reference datasets from data.gov.sg")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "reference",
        help="Output directory for MRT reference files",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    symbol_geo = _download_json(DATASET_MRT_SYMBOL)
    mp03_geo = _download_json(DATASET_MP03_MRT_NAME)

    _write_geojson(out_dir / "mrt_station_symbol_ura.geojson", symbol_geo)
    _write_geojson(out_dir / "mrt_masterplan2003_name_ura.geojson", mp03_geo)

    names = _extract_station_names(mp03_geo)
    _write_future_csv(out_dir / "future_mrt_stations.csv", names)

    print(f"Wrote {out_dir / 'mrt_station_symbol_ura.geojson'}")
    print(f"Wrote {out_dir / 'mrt_masterplan2003_name_ura.geojson'}")
    print(f"Wrote {out_dir / 'future_mrt_stations.csv'} ({len(names)} stations)")


if __name__ == "__main__":
    main()

