#!/usr/bin/env python3
"""Fetch official MRT travel-time features via OneMap routing API."""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]


def _load_local_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _must_env(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def _onemap_token() -> str:
    email = _must_env("ONEMAP_API_EMAIL")
    password = _must_env("ONEMAP_API_PASSWORD")
    url = "https://www.onemap.gov.sg/api/auth/post/getToken"
    resp = requests.post(url, json={"email": email, "password": password}, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    token = body.get("access_token") or body.get("token")
    if not token:
        raise RuntimeError(f"Unable to obtain OneMap token: {body}")
    return str(token)


def _load_mrt_points(path: Path) -> list[dict[str, Any]]:
    import json

    geo = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for f in geo.get("features", []):
        p = f.get("properties", {}) or {}
        g = f.get("geometry", {}) or {}
        if g.get("type") != "Point":
            continue
        coords = g.get("coordinates") or []
        if len(coords) != 2:
            continue
        name = str(p.get("NAME", "")).strip().upper()
        if not name:
            continue
        out.append({"station_name": name, "lon": float(coords[0]), "lat": float(coords[1])})
    # Deduplicate by station name, keep first.
    seen: set[str] = set()
    dedup: list[dict[str, Any]] = []
    for r in out:
        if r["station_name"] in seen:
            continue
        seen.add(r["station_name"])
        dedup.append(r)
    return dedup


def _route_duration_minutes(
    token: str,
    *,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    date_str: str,
    time_str: str,
) -> tuple[float | None, float | None]:
    url = "https://www.onemap.gov.sg/api/public/routingsvc/route"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "start": f"{start_lat:.7f},{start_lon:.7f}",
        "end": f"{end_lat:.7f},{end_lon:.7f}",
        "routeType": "pt",
        "date": date_str,
        "time": time_str,
        "mode": "TRANSIT",
    }
    resp = requests.get(url, headers=headers, params=params, timeout=60)
    if resp.status_code in (401, 403):
        raise RuntimeError("OneMap routing unauthorized. Check API credentials.")
    resp.raise_for_status()
    body = resp.json()

    # OneMap response shape may vary; parse defensively.
    duration_s = None
    distance_m = None
    plan = body.get("plan")
    if isinstance(plan, dict):
        its = plan.get("itineraries")
        if isinstance(its, list) and its:
            it0 = its[0] or {}
            duration_s = it0.get("duration")
            legs = it0.get("legs")
            if isinstance(legs, list):
                distance_m = sum(float((lg or {}).get("distance", 0.0)) for lg in legs)
    if duration_s is None:
        rs = body.get("route_summary") or {}
        duration_s = rs.get("total_time")
        distance_m = rs.get("total_distance")
    if duration_s is None:
        return None, None
    minutes = float(duration_s) / 60.0
    dist_km = float(distance_m) / 1000.0 if distance_m is not None else None
    return minutes, dist_km


def main() -> None:
    _load_local_env_file(ROOT / ".env")
    p = argparse.ArgumentParser(description="Fetch MRT travel-time features from OneMap.")
    p.add_argument(
        "--mrt-geojson",
        type=Path,
        default=ROOT / "data" / "reference" / "mrt_station_symbol_ura.geojson",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "data" / "reference" / "mrt_travel_time_to_cbd.csv",
    )
    p.add_argument("--date", type=str, default=datetime.now().strftime("%m-%d-%Y"))
    p.add_argument("--time", type=str, default="08:30:00")
    p.add_argument("--max-stations", type=int, default=200)
    args = p.parse_args()

    stations = _load_mrt_points(args.mrt_geojson)
    if not stations:
        raise RuntimeError(f"No MRT points found in {args.mrt_geojson}")
    stations = stations[: max(1, int(args.max_stations))]

    # CBD anchors to derive robust accessibility features.
    anchors = {"RAFFLES PLACE", "CITY HALL", "MARINA BAY"}
    station_map = {s["station_name"]: s for s in stations}
    anchor_pts = [station_map[n] for n in anchors if n in station_map]
    if not anchor_pts:
        raise RuntimeError("CBD anchor stations missing from MRT reference GeoJSON.")

    token = _onemap_token()
    rows: list[dict[str, Any]] = []
    for s in stations:
        vals: list[float] = []
        dists: list[float] = []
        for a in anchor_pts:
            if str(s["station_name"]) == str(a["station_name"]):
                vals.append(0.0)
                dists.append(0.0)
                continue
            mins, dist_km = _route_duration_minutes(
                token,
                start_lat=float(s["lat"]),
                start_lon=float(s["lon"]),
                end_lat=float(a["lat"]),
                end_lon=float(a["lon"]),
                date_str=args.date,
                time_str=args.time,
            )
            if mins is not None:
                vals.append(float(mins))
            if dist_km is not None:
                dists.append(float(dist_km))
        if not vals:
            continue
        rows.append(
            {
                "station_name": s["station_name"],
                "to_cbd_median_travel_min": round(sorted(vals)[len(vals) // 2], 2),
                "to_cbd_min_travel_min": round(min(vals), 2),
                "to_cbd_avg_route_km": round(sum(dists) / len(dists), 3) if dists else "",
                "route_eval_date": args.date,
                "route_eval_time": args.time,
                "source": "OneMap Routing API (routeType=pt, mode=TRANSIT)",
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "station_name",
                "to_cbd_median_travel_min",
                "to_cbd_min_travel_min",
                "to_cbd_avg_route_km",
                "route_eval_date",
                "route_eval_time",
                "source",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out_csv} ({len(rows)} stations)")


if __name__ == "__main__":
    main()

