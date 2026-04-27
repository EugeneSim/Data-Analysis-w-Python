"""Sun / façade exposure: not available in the default HDB transaction CSV.

Set ``ENABLE_SUN_PROXY=1`` to attach placeholder columns for experiments only.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def add_sun_proxy_placeholder(df: pd.DataFrame) -> pd.DataFrame:
    """If env enabled, add NaN columns documenting intended future work."""
    if os.environ.get("ENABLE_SUN_PROXY", "").strip() not in ("1", "true", "True"):
        return df
    out = df.copy()
    out["sun_exposure_proxy"] = np.nan
    out["sun_exposure_note"] = "not_computed: requires façade/azimuth; see Quarto"
    return out
