"""Folium choropleth for aggregated town / planning-area metrics."""

from __future__ import annotations

import json
from pathlib import Path

import folium
import pandas as pd

# Match Singapore planning-area GeoJSONs (e.g. URA MP25 PLN_AREA_N) and small test fixtures
_NAME_PROPS: tuple[str, ...] = (
    "name",
    "Name",
    "PLN_AREA_N",
    "pln_area_n",
    "PLN_AREA_C",
    "Description",
    "Description_1",
)


def _feature_label(props: dict, name_prop: str | None) -> str:
    if name_prop and name_prop in props and str(props.get(name_prop, "")).strip():
        return str(props[name_prop]).strip()
    for k in _NAME_PROPS:
        if k in props and str(props.get(k, "")).strip():
            return str(props[k]).strip()
    return ""


def _match_value(name: str, value_by_name: dict[str, float] | None) -> float | None:
    if not name or not value_by_name:
        return None
    n = name.strip()
    if n in value_by_name:
        return value_by_name[n]
    for k, v in value_by_name.items():
        if str(k).strip().upper() == n.upper():
            return v
    return None


def geo_median_value_dict(df: pd.DataFrame) -> dict[str, float]:
    """
    Median resale by planning area (preferred) or HDB town.

    planning_area (from `town_to_planning_area.csv` join) matches GeoJSON plan-area names
    better than raw HDB `town` alone (e.g. Central / Central Area).
    """
    if "resale_price" not in df.columns:
        return {}
    d = df.dropna(subset=["resale_price"])
    if d.empty:
        return {}
    has_pa = "planning_area" in d.columns and d["planning_area"].notna().any()
    col = "planning_area" if has_pa else "town"
    g = d.groupby(d[col], observed=True)["resale_price"].median()
    return {str(i).strip(): float(v) for i, v in g.items() if str(i).strip()}


def block_street_table(df: pd.DataFrame, top_n: int = 200) -> pd.DataFrame:
    """
    Block-level (street + block) medians from transaction data — no lat/lon in HDB open CSV.
    For map pins, building coordinates are not in this dataset.
    """
    need = {"town", "street_name", "block", "resale_price"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    d = df.dropna(subset=list(need))
    if d.empty:
        return pd.DataFrame()
    g = (
        d.groupby(["town", "street_name", "block"], observed=True)
        .agg(
            n_trans=("resale_price", "size"),
            median_resale=("resale_price", "median"),
        )
        .reset_index()
    )
    g = g.sort_values("n_trans", ascending=False).head(top_n)
    return g


def folium_choropleth_by_name(
    geojson_path: Path,
    value_by_name: dict[str, float] | None,
    *,
    name_prop: str | None = "name",
    center: tuple[float, float] = (1.3521, 103.8198),
    zoom: int = 11,
    show_unmatched: bool = True,
) -> folium.Map:
    """
    Draw every polygon in the GeoJSON; fill by median where names match, grey otherwise.
    Tries `name_prop` first, then common property keys (PLN_AREA_N, etc.).
    """
    with Path(geojson_path).open(encoding="utf-8") as f:
        geo = json.load(f)
    m = folium.Map(location=[center[0], center[1]], zoom_start=zoom, tiles="cartodbpositron")
    if not value_by_name:
        vmin, vmax = 0.0, 1.0
    else:
        vals = [v for v in value_by_name.values() if v == v]  # exclude nan
        vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 1.0)
    for feat in geo.get("features", []):
        props = feat.get("properties", {}) or {}
        n = _feature_label(props, name_prop)
        v = _match_value(n, value_by_name) if n else None
        if v is not None and v == v:
            if vmax > vmin + 1e-9:
                col = _color(v, vmin, vmax)
            else:
                col = _color(v, v - 1.0, v + 1.0)
            fill = 0.6
        else:
            col = "#d9d9d9"
            fill = 0.35 if show_unmatched else 0.0
        style = {
            "fillColor": col,
            "color": "#555",
            "weight": 1,
            "fillOpacity": fill,
        }
        if not show_unmatched and v is None:
            continue
        if v is not None and v == v:
            tip = f"{n}: SGD {v:,.0f} (median)"
        else:
            tip = f"{n}: (no HDB match in this file)"
        layer = folium.GeoJson(data=feat, style_function=lambda _x, s=style: s)
        layer.add_child(folium.Tooltip(tip))
        layer.add_to(m)
    return m


def _color(x: float, lo: float, hi: float) -> str:
    if hi <= lo:
        return "#3182bd"
    t = (x - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    # simple blue to red
    r = int(13 + t * 200)
    b = int(200 - t * 180)
    g = 100
    return f"#{r:02x}{g:02x}{b:02x}"


def town_median_to_dict(df: pd.DataFrame) -> dict[str, float]:
    d = df.dropna(subset=["town", "resale_price"])
    g = d.groupby("town", observed=True)["resale_price"].median()
    return {str(i).upper(): float(v) for i, v in g.items()}
