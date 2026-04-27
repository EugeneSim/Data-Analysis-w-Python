from __future__ import annotations

from singapore_eda import gov_http


def test_is_allowed_gov_url_accepts() -> None:
    assert gov_http.is_allowed_gov_url("https://data.gov.sg/api/action/datastore_search")
    assert gov_http.is_allowed_gov_url("https://geo.data.gov.sg/gis/x.json")
    assert gov_http.is_allowed_gov_url(
        "https://api-open.data.gov.sg/v1/public/api/datasets/x/poll-download"
    )


def test_is_allowed_gov_url_rejects() -> None:
    assert not gov_http.is_allowed_gov_url("http://data.gov.sg/x")  # no https
    assert not gov_http.is_allowed_gov_url("https://evil.com/?u=data.gov.sg")
    assert not gov_http.is_allowed_gov_url("https://phishing-data.gov.sg.evil.com/x")


def test_env_flag() -> None:
    assert gov_http.env_flag("SINGAPORE_EDA_E2E_FAKE", False) is False


def test_admin_compare() -> None:
    assert gov_http.verify_admin_token("same-len-ok", "same-len-ok") is True
    assert gov_http.verify_admin_token("short", "longer!") is False
    assert gov_http.verify_admin_token("", "x") is False
