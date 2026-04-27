from __future__ import annotations

from pathlib import Path

import pytest

from singapore_eda.datastore import _count_csv_data_rows, download_paginated_resource


def test_count_csv_rows(tmp_path: Path) -> None:
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    assert _count_csv_data_rows(p) == 2


def test_skip_fresh_avoids_name_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If output is fresh, we must not call get_gov_client (no network)."""
    out = tmp_path / "out.csv"
    out.write_text("h\n1\n", encoding="utf-8")
    err = "network should not be used when skip is active"

    def _boom() -> object:
        raise AssertionError(err)

    import singapore_eda.datastore as ds

    monkeypatch.setattr(ds, "get_gov_client", _boom)
    n = download_paginated_resource("dummy_id", out, skip_if_fresh_hours=24.0)
    assert n == 1
