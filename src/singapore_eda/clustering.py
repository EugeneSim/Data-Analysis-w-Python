"""K-means segments on numeric features (transaction-level)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_kmeans(
    df: pd.DataFrame,
    features: list[str],
    *,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[pd.Series, KMeans | None, np.ndarray | None]:
    """Return cluster labels aligned to df index; NaN where features missing."""
    sub = df[features].copy()
    mask = sub.notna().all(axis=1)
    X = sub.loc[mask].values.astype(float)
    if len(X) == 0:
        return pd.Series(index=df.index, dtype="Int64"), None, None
    k = max(1, min(n_clusters, len(X)))
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    pred = km.fit_predict(Xs)
    out = pd.Series(index=df.index, dtype="Int64")
    out.loc[mask] = pred
    return out, km, Xs


def cluster_interpretation(
    df: pd.DataFrame,
    cluster: pd.Series,
    features: list[str],
) -> tuple[pd.DataFrame, dict[int, str]]:
    """
    Median feature profile per cluster + short text labels (vs sample-wide medians).
    """
    need = [c for c in features if c in df.columns]
    if not need or cluster.isna().all():
        return pd.DataFrame(), {}
    t = df[need].copy()
    t["_c"] = cluster
    t = t.dropna(subset=["_c"])
    med = t.groupby("_c", observed=True)[need].median()
    overall = t[need].median()
    labels: dict[int, str] = {}
    for c in med.index:
        if pd.isna(c):
            continue
        ci = int(c)
        parts: list[str] = []
        for f in need:
            mv, ov = med.loc[c, f], overall[f]
            if not (mv == mv) or not (ov == ov) or ov in (0, 0.0):
                continue
            r = mv / ov
            if f == "resale_price":
                if r < 0.92:
                    parts.append("lower median $")
                elif r > 1.08:
                    parts.append("higher median $")
            if f == "floor_area_sqm":
                if r < 0.95:
                    parts.append("smaller area")
                elif r > 1.05:
                    parts.append("larger area")
            if f == "remaining_lease_years":
                if r < 0.92:
                    parts.append("shorter remaining lease")
                elif r > 1.08:
                    parts.append("longer remaining lease")
        base = f"Cluster {ci}"
        short = f"{base} (similar to whole-sample medians)"
        labels[ci] = f"{base} — {', '.join(parts)}" if parts else short
    return med.reset_index().rename(columns={"_c": "cluster_id"}), labels
