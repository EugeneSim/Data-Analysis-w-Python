from __future__ import annotations

import pandas as pd

from singapore_eda.eip import eip_match_stats


def test_eip_stats_no_column() -> None:
    r = eip_match_stats(pd.DataFrame({"a": [1]}))
    assert r["eip_matched_rows"] == 0
    assert r["eip_status"] == "no_column"


def test_eip_stats_with_note() -> None:
    d = pd.DataFrame({"eip_status_note": ["x", None, ""]})
    r = eip_match_stats(d)
    assert r["eip_matched_rows"] == 1
    assert r["eip_row_rate"] > 0
