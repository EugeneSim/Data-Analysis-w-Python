"""Load raw CSV and return analysis-ready enriched frame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from singapore_eda.clean import clean_hdb
from singapore_eda.eip import join_eip
from singapore_eda.features import add_features
from singapore_eda.geo_join import enrich_with_reference
from singapore_eda.numerology import add_numerology_features
from singapore_eda.storey import add_storey_band
from singapore_eda.sun_exposure import add_sun_proxy_placeholder


def load_enriched(
    csv_path: Path | str,
    *,
    top_towns: int = 15,
    town_coverage: float | None = 0.8,
) -> pd.DataFrame:
    path = Path(csv_path)
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    c = clean_hdb(raw)
    f = add_features(
        c,
        top_towns=top_towns,
        town_coverage=town_coverage,
    )
    f = enrich_with_reference(f)
    f = add_storey_band(f)
    f = add_numerology_features(f)
    f = join_eip(f)
    f = add_sun_proxy_placeholder(f)
    return f
