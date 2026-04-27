from __future__ import annotations

import pytest

from singapore_eda import gov_limits


def test_resolve_tier_default_none_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SINGAPORE_EDA_API_TIER", raising=False)
    monkeypatch.delenv("DATA_GOV_SG_API_KEY", raising=False)
    assert gov_limits.resolve_api_tier() == "none"


def test_resolve_tier_dev_when_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SINGAPORE_EDA_API_TIER", raising=False)
    monkeypatch.setenv("DATA_GOV_SG_API_KEY", "k")
    assert gov_limits.resolve_api_tier() == "dev"


def test_resolve_tier_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SINGAPORE_EDA_API_TIER", "prod")
    assert gov_limits.resolve_api_tier() == "prod"


def test_datastore_pace_faster_than_file_tier_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SINGAPORE_EDA_API_TIER", "none")
    for k in (
        "SINGAPORE_EDA_MIN_INTERVAL_DATASTORE_SEC",
        "SINGAPORE_EDA_MIN_INTERVAL_FILE_SEC",
        "SINGAPORE_EDA_MIN_INTERVAL_SEC",
    ):
        monkeypatch.delenv(k, raising=False)
    d = gov_limits.min_interval_datastore_sec()
    f = gov_limits.min_interval_file_sec()
    assert d < f


def test_pace_config_has_doc_link() -> None:
    d = gov_limits.pace_config_public()
    assert "data.gov" in d["api_rate_limit_docs"]
    assert d["resolved_tier"] in ("none", "dev", "prod")
