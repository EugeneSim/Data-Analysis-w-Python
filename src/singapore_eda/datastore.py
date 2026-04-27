"""Shared paginated download from data.gov.sg CKAN `datastore_search`."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from singapore_eda.constants import HDB_DATASTORE_SEARCH, PAGE_SIZE
from singapore_eda.gov_http import env_flag, get_gov_client
from singapore_eda.gov_limits import skip_download_fresh_hours_default

log = logging.getLogger("singapore_eda.datastore")


def _headers() -> dict[str, str]:
    key = os.environ.get("DATA_GOV_SG_API_KEY", "").strip()
    if not key:
        return {}
    return {"X-API-Key": key}


def _count_csv_data_rows(path: Path) -> int:
    with path.open("rb") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)


def meta_path_for_csv(out_path: Path) -> Path:
    return out_path.parent / f"{out_path.name}.meta.json"


def read_download_meta(out_path: Path) -> dict[str, Any]:
    p = meta_path_for_csv(out_path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def write_download_meta(out_path: Path, payload: dict[str, Any]) -> None:
    p = meta_path_for_csv(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _env_check_new_data() -> bool:
    return env_flag("SINGAPORE_EDA_CHECK_NEW_DATA", False)


def fetch_datastore_total(resource_id: str) -> int | None:
    """
    One CKAN call: ``result['total']`` is the number of rows in the resource.
    """
    client = get_gov_client()
    payload = client.get_json(
        HDB_DATASTORE_SEARCH,
        params={"resource_id": resource_id, "limit": 1, "offset": 0},
        headers=_headers(),
        timeout=120,
        use_cache=False,
    )
    if not payload.get("success"):
        return None
    t = (payload.get("result") or {}).get("total")
    if t is None:
        return None
    try:
        return int(t)
    except (TypeError, ValueError):
        return None


def download_paginated_resource(
    resource_id: str,
    out_path: Path,
    *,
    max_rows: int | None = None,
    page_size: int = PAGE_SIZE,
    skip_if_fresh_hours: float | None = None,
    query_params: dict[str, Any] | None = None,
) -> int:
    """
    Download all rows (or up to max_rows) and write CSV. Returns count.

    If ``skip_if_fresh_hours`` (or env ``SINGAPORE_EDA_SKIP_DOWNLOAD_IF_FRESH_HOURS``) is
    set and the output file is newer than that many hours, **no API calls** are made and
    the existing row count (excluding header) is returned.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skip: float | None = skip_if_fresh_hours
    if skip is None:
        skip = skip_download_fresh_hours_default()
    if skip is not None and skip > 0 and out_path.is_file() and out_path.stat().st_size > 0:
        age_h = (time.time() - out_path.stat().st_mtime) / 3600.0
        if age_h < skip:
            if _env_check_new_data():
                old_meta = read_download_meta(out_path)
                try:
                    remote = fetch_datastore_total(resource_id)
                except OSError as e:  # noqa: BLE001
                    log.warning("could not read remote row total, keeping local file: %s", e)
                    remote = None
                prev = old_meta.get("api_total")
                if remote is not None and prev is not None and int(prev) < int(remote):
                    log.info("API row total %s > last known %s; re-downloading", remote, prev)
                else:
                    n = _count_csv_data_rows(out_path)
                    log.info(
                        "skip download: %s age %.2fh < %.2fh (rows=%s) — no API calls",
                        out_path,
                        age_h,
                        skip,
                        n,
                    )
                    return n
            else:
                n = _count_csv_data_rows(out_path)
                log.info(
                    "skip download: %s age %.2fh < %.2fh (rows=%s) — no API calls",
                    out_path,
                    age_h,
                    skip,
                    n,
                )
                return n
    offset = 0
    frames: list[pd.DataFrame] = []
    total = 0
    api_total: int | None = None
    # Per-page API cache: off by default (avoids large disk use on full imports); set
    # SINGAPORE_EDA_DATASTORE_PAGE_CACHE=1 to enable resumable dev / replay.
    use_page_cache = env_flag("SINGAPORE_EDA_DATASTORE_PAGE_CACHE", False)
    while True:
        params: dict[str, Any] = {
            "resource_id": resource_id,
            "limit": page_size,
            "offset": offset,
        }
        if query_params:
            params.update(query_params)
        client = get_gov_client()
        payload = client.get_json(
            HDB_DATASTORE_SEARCH,
            params=params,
            headers=_headers(),
            timeout=120,
            use_cache=use_page_cache,
        )
        if not payload.get("success"):
            raise RuntimeError(f"data.gov.sg error: {payload.get('error', payload)}")

        result = payload.get("result", {})
        if api_total is None and isinstance(result.get("total"), (int, float)):
            try:
                api_total = int(result["total"])
            except (TypeError, ValueError):
                pass
        recs: list[dict] = result.get("records", [])
        if not recs:
            break
        for row in recs:
            row.pop("_id", None)
        frames.append(pd.DataFrame.from_records(recs))
        n = len(recs)
        total += n
        offset += n
        if max_rows is not None and total >= max_rows:
            break
        if n < page_size:
            break
        # Min interval between pages is enforced inside GovClient (SINGAPORE_EDA_MIN_INTERVAL_SEC)

    if not frames:
        raise RuntimeError("No records returned from API.")
    df = pd.concat(frames, ignore_index=True)
    if max_rows is not None:
        df = df.head(max_rows)
    df.to_csv(out_path, index=False)
    n = int(len(df))
    incomplete = bool(api_total is not None and n < api_total)
    write_download_meta(
        out_path,
        {
            "resource_id": str(resource_id),
            "wrote_utc": datetime.now(UTC).isoformat(),
            "downloaded_rows": n,
            "api_total": api_total,
            "incomplete": incomplete,
        },
    )
    return n
