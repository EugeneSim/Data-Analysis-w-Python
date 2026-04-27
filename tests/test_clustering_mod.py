from __future__ import annotations

from pathlib import Path

import pandas as pd

from singapore_eda.clean import clean_hdb
from singapore_eda.clustering import cluster_kmeans
from singapore_eda.features import add_features

_FIX = Path(__file__).parent / "fixtures" / "hdb_sample.csv"


def test_kmeans_labels() -> None:
    raw = pd.read_csv(_FIX)
    c = add_features(clean_hdb(raw), town_coverage=None, top_towns=15)
    lab, _km, _ = cluster_kmeans(
        c,
        ["resale_price", "floor_area_sqm", "remaining_lease_years"],
        n_clusters=2,
        random_state=0,
    )
    assert lab.notna().sum() > 0
