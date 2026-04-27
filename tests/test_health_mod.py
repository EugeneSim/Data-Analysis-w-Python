from __future__ import annotations

import pytest

from singapore_eda import health as health_mod


def test_run_health_respects_http_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SINGAPORE_EDA_HEALTH_SKIP_HTTP", "1")
    rep = health_mod.run_health()
    assert rep.status in ("ok", "fail", "degraded")
    names = [c.name for c in rep.checks]
    assert "data_gov_reachable" in names
    dg = next(c for c in rep.checks if c.name == "data_gov_reachable")
    assert "skipped" in dg.detail
