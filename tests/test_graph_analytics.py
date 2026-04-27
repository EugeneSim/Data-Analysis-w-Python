from __future__ import annotations

from pathlib import Path

import pandas as pd

from singapore_eda.graph_analytics import (
    correlation_graph,
    graph_summary,
    spatial_adjacency_graph,
    town_price_pivot,
)

_FIX = Path(__file__).parent / "fixtures" / "hdb_sample.csv"


def test_graph_builds() -> None:
    raw = pd.read_csv(_FIX)
    raw["month"] = pd.to_datetime(raw["month"])
    p = town_price_pivot(raw, min_months=1)
    g = correlation_graph(p, min_corr=0.0)
    s = graph_summary(g)
    assert s["n_nodes"] >= 1


def test_spatial_adjacency_returns() -> None:
    geo = Path(__file__).parent / "fixtures" / "planning_areas_tiny.geojson"
    g, note = spatial_adjacency_graph(geo, name_prop="name")
    assert isinstance(note, str)
    assert g.number_of_nodes() >= 0
