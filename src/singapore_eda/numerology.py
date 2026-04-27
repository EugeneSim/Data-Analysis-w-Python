"""Cultural digit features from block / storey (descriptive; not causal claims)."""

from __future__ import annotations

import re

import pandas as pd

_DIG = re.compile(r"\d+")


def _digits_of_block(block: str | float) -> str:
    if pd.isna(block):
        return ""
    s = str(block).strip()
    m = _DIG.search(s)
    return m.group(0) if m else s


def add_numerology_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "block" in out.columns:
        bd = out["block"].map(_digits_of_block)
        out["block_has_4"] = bd.str.contains("4", na=False)
        out["block_ends_8"] = bd.str.endswith("8", na=False)
    if "storey_mid" in out.columns:
        sm = out["storey_mid"].round(0).astype("Int64")
        ss = sm.astype(str)
        out["storey_int_has_4"] = ss.str.contains("4", na=False)
        out["storey_int_ends_8"] = ss.str.endswith("8", na=False)
    return out
