from __future__ import annotations

from pathlib import Path

from singapore_eda.rent_cache import age_hours, rent_csv_is_fresh


def test_rent_cache_helpers(tmp_path) -> None:
    p = tmp_path / "rent.csv"
    p.write_text("a\n1\n")
    assert age_hours(p) is not None
    assert rent_csv_is_fresh(p, ttl_hours=1e6) is True
    assert rent_csv_is_fresh(p, ttl_hours=0) is False


def test_rent_cache_missing() -> None:
    p = Path("/nonexistent/median_rent_hdb.csv")
    assert age_hours(p) is None
    assert rent_csv_is_fresh(p) is False
