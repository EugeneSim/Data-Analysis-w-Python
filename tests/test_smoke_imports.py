from __future__ import annotations

import importlib


def test_import_all_modules() -> None:
    modules = [
        "singapore_eda.clustering",
        "singapore_eda.connectivity",
        "singapore_eda.datastore",
        "singapore_eda.eip",
        "singapore_eda.forecasting",
        "singapore_eda.geo_join",
        "singapore_eda.gov_http",
        "singapore_eda.gov_limits",
        "singapore_eda.health",
        "singapore_eda.graph_analytics",
        "singapore_eda.history",
        "singapore_eda.mapviz",
        "singapore_eda.merge_resale",
        "singapore_eda.numerology",
        "singapore_eda.pipeline",
        "singapore_eda.rent_cache",
        "singapore_eda.rent_ingest",
        "singapore_eda.rental_yields",
        "singapore_eda.storey",
        "singapore_eda.sun_exposure",
    ]
    for module in modules:
        importlib.import_module(module)
