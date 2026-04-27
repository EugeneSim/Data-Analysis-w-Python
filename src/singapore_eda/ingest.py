"""Load raw HDB CSV into a DataFrame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from singapore_eda.constants import DEFAULT_RAW_CSV, HDB_CITATION_URL


@dataclass(frozen=True)
class IngestMeta:
    source_path: Path
    citation_url: str
    n_rows: int
    n_cols: int


def read_hdb_csv(
    path: Path | str | None = None,
) -> tuple[pd.DataFrame, IngestMeta]:
    """Read HDB resale CSV. Column names are normalized to snake_case strings."""
    path = Path(path) if path is not None else DEFAULT_RAW_CSV
    if not path.exists():
        msg = f"Raw data not found: {path}. Run: python -m singapore_eda or hdb-download"
        raise FileNotFoundError(msg)
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    meta = IngestMeta(
        source_path=path.resolve(),
        citation_url=HDB_CITATION_URL,
        n_rows=int(len(df)),
        n_cols=int(len(df.columns)),
    )
    return df, meta
