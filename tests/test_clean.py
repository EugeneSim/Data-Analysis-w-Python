from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from singapore_eda.clean import (
    _parse_remaining_lease_to_years,
    _parse_storey_mid,
    _remaining_lease_text_from_years,
    clean_hdb,
)
from singapore_eda.features import add_features, model_design_subset

_FIXTURE = Path(__file__).parent / "fixtures" / "hdb_sample.csv"


def test_parse_remaining_lease() -> None:
    expected = 61 + 4 / 12
    assert _parse_remaining_lease_to_years("61 years 04 months") == pytest.approx(
        expected, rel=1e-5
    )
    assert _parse_remaining_lease_to_years("94 years") == 94.0
    assert pd.isna(_parse_remaining_lease_to_years(""))
    assert pd.isna(_parse_remaining_lease_to_years(float("nan")))
    assert "y" in _remaining_lease_text_from_years(92.4167)
    assert "m" in _remaining_lease_text_from_years(92.4167)


def test_parse_storey() -> None:
    assert _parse_storey_mid("10 TO 12") == 11.0
    assert _parse_storey_mid("01 TO 03") == 2.0


def test_clean_hdb_fixture() -> None:
    raw = pd.read_csv(_FIXTURE)
    clean = clean_hdb(raw)
    assert len(clean) >= 15
    assert "remaining_lease_years" in clean.columns
    assert "remaining_lease_label" in clean.columns
    assert "storey_mid" in clean.columns
    assert clean["resale_price"].isna().sum() == 0


def test_features_model_subset() -> None:
    raw = pd.read_csv(_FIXTURE)
    clean = clean_hdb(raw)
    feat = add_features(clean, top_towns=5, town_coverage=None)
    m = model_design_subset(feat)
    assert "town_group" in m.columns
    assert "log_resale_price" in m.columns
    assert len(m) > 0
