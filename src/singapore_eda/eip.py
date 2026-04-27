"""Join optional EIP / ethnic-limit reference (block-level, illustrative stub by default)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from singapore_eda.paths import reference_dir


def load_eip_stub(path: Path | None = None) -> pd.DataFrame:
    p = path or (reference_dir() / "eip_block_stub.csv")
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for c in ("block", "street_name", "town"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.upper()
    return df


def join_eip(df: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    need = {"block", "street_name", "town"}
    eip = load_eip_stub(path)
    if eip.empty or not need.issubset(df.columns):
        return df
    d = df.copy()
    d["block"] = d["block"].astype(str).str.strip()
    d["street_name"] = d["street_name"].astype(str).str.strip().str.upper()
    d["town"] = d["town"].astype(str).str.strip().str.upper()
    return d.merge(
        eip,
        on=["block", "street_name", "town"],
        how="left",
    )


def eip_match_stats(df: pd.DataFrame) -> dict:
    """Share of rows with a non-empty EIP match (when eip_status_note is present)."""
    if "eip_status_note" not in df.columns:
        return {"eip_matched_rows": 0, "eip_row_rate": 0.0, "eip_status": "no_column"}
    s = df["eip_status_note"]
    m = s.notna() & (s.astype(str).str.strip() != "")
    m = m & (s.astype(str).str.upper() != "NAN")
    n = int(len(df))
    return {
        "eip_matched_rows": int(m.sum()),
        "eip_row_rate": float(m.mean()) if n else 0.0,
    }
