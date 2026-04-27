from __future__ import annotations

import networkx as nx

from singapore_eda.graph_analytics import (
    correlation_community_table,
    correlation_edges_dataframe,
    unweighted_spatial_edge_table,
)


def test_empty_graph_tables() -> None:
    g = nx.Graph()
    assert correlation_edges_dataframe(g).empty
    assert correlation_community_table(g).empty
    assert unweighted_spatial_edge_table(g).empty


def test_correlation_graph_edges() -> None:
    g = nx.Graph()
    g.add_edge("A", "B", weight=0.5)
    d = correlation_edges_dataframe(g)
    assert len(d) == 1
    assert d["town_a"].iloc[0] in ("A", "B")
