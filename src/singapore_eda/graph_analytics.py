"""Town-level correlation graph and greedy modularity communities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def town_price_pivot(
    df: pd.DataFrame,
    *,
    min_months: int = 3,
) -> pd.DataFrame:
    d = df.dropna(subset=["month", "town", "resale_price"]).copy()
    d["town"] = d["town"].astype(str).str.upper()
    d["month"] = d["month"].dt.to_period("M").dt.to_timestamp()
    p = d.pivot_table(index="month", columns="town", values="resale_price", aggfunc="median")
    p = p.dropna(axis=1, thresh=min_months)
    return p


def correlation_graph(
    pivot: pd.DataFrame,
    *,
    min_corr: float = 0.5,
) -> nx.Graph:
    corr = pivot.corr()
    g = nx.Graph()
    towns = corr.columns.tolist()
    for t in towns:
        g.add_node(t)
    for i, a in enumerate(towns):
        for b in towns[i + 1 :]:
            v = corr.loc[a, b]
            if np.isfinite(v) and abs(v) >= min_corr:
                g.add_edge(a, b, weight=float(v))
    return g


def greedy_communities(g: nx.Graph) -> list[set[Any]]:
    return list(nx.community.greedy_modularity_communities(g))


def graph_summary(g: nx.Graph) -> dict[str, Any]:
    if g.number_of_nodes() == 0:
        return {"n_nodes": 0, "n_edges": 0, "n_communities": 0}
    comms = greedy_communities(g)
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "n_communities": len(comms),
        "density": float(nx.density(g)),
    }


def correlation_edges_dataframe(g: nx.Graph) -> pd.DataFrame:
    """Edges as a sortable table (|correlation| over threshold in graph)."""
    if g.number_of_nodes() == 0:
        return pd.DataFrame(columns=["town_a", "town_b", "correlation", "abs_correlation"])
    rows: list[dict[str, Any]] = []
    for u, v, data in g.edges(data=True):
        w = data.get("weight", float("nan"))
        w = float(w) if np.isfinite(w) else w
        rows.append(
            {
                "town_a": u,
                "town_b": v,
                "correlation": w,
                "abs_correlation": abs(w) if isinstance(w, (int, float)) and np.isfinite(w) else w,
            }
        )
    dfe = pd.DataFrame(rows)
    if dfe.empty:
        return dfe
    return dfe.sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def unweighted_spatial_edge_table(g: nx.Graph) -> pd.DataFrame:
    """Neighbouring planning-area pairs (unweighted, weight=1)."""
    if g.number_of_nodes() == 0:
        return pd.DataFrame(columns=["planning_area_a", "planning_area_b", "touches"])
    rows = [
        {
            "planning_area_a": u,
            "planning_area_b": v,
            "touches": 1,
        }
        for u, v in g.edges()
    ]
    return pd.DataFrame(rows).sort_values(["planning_area_a", "planning_area_b"])


def correlation_community_table(g: nx.Graph) -> pd.DataFrame:
    """Modularity-based groups of towns (for readable summaries, not st.json)."""
    if g.number_of_nodes() == 0:
        return pd.DataFrame(columns=["group_id", "towns", "n_towns"])
    comms = greedy_communities(g)
    return pd.DataFrame(
        {
            "group_id": list(range(1, len(comms) + 1)),
            "towns": [", ".join(sorted(str(x) for x in c)) for c in comms],
            "n_towns": [len(c) for c in comms],
        }
    )


def _node_name_from_row(row: pd.Series, name_prop: str) -> str:
    if name_prop in row.index and pd.notna(row[name_prop]):
        return str(row[name_prop]).strip()
    d = row.to_dict() if hasattr(row, "to_dict") else {}
    for k in ("name", "PLN_AREA_N", "pln_area_n", "Name"):
        if k in d and d[k] is not None and str(d[k]).strip():
            return str(d[k]).strip()
    return ""


def spatial_adjacency_graph(
    geojson_path: Path,
    *,
    name_prop: str = "name",
) -> tuple[nx.Graph, str]:
    """
    Planning-area (polygon) **touching** graph from a GeoJSON file.

    Requires the optional `geo` extra (geopandas + pyogrio or fiona). If geopandas is
    not installed, returns an empty graph and a message string.
    """
    p = Path(geojson_path)
    if not p.exists():
        return nx.Graph(), f"missing file: {p}"
    try:
        import geopandas as gpd
    except ImportError:
        return (
            nx.Graph(),
            "install geopandas (optional extra [geo]) for spatial adjacency",
        )
    gdf = gpd.read_file(p)
    g = nx.Graph()
    n = len(gdf)
    if n == 0:
        return g, "no features in GeoJSON"
    gdf = gdf.reset_index(drop=True)
    names: list[str] = []
    for i in range(n):
        row = gdf.iloc[i]
        nm = _node_name_from_row(row, name_prop) or f"feature_{i}"
        names.append(nm)
    for nm in names:
        g.add_node(nm)
    j = None
    try:
        j = gpd.sjoin(gdf, gdf, predicate="touches", how="inner")
    except TypeError:
        try:
            j = gpd.sjoin(gdf, gdf, op="touches", how="inner")  # type: ignore[call-arg]
        except Exception:  # noqa: BLE001
            j = None
    except Exception:  # noqa: BLE001
        j = None
    if j is not None and len(j) and "index_right" in j.columns:
        for li, r in j.iterrows():
            a, b = int(li), int(r["index_right"])
            if a == b or a > b:
                continue
            u, v = names[a], names[b]
            if u and v:
                g.add_edge(u, v, weight=1.0)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                gi, gj = gdf.geometry.iloc[i], gdf.geometry.iloc[j]
                if gi is None or gj is None or gi.is_empty or gj.is_empty:
                    continue
                if gi.touches(gj):
                    g.add_edge(names[i], names[j], weight=1.0)
    return g, "ok"
