"""Plotly figures for EDA and Streamlit."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def fig_median_price_by_town(
    df: pd.DataFrame, towns: Sequence[str] | None = None, top: int = 20
) -> go.Figure:
    d = df.copy()
    if towns is not None and "town" in d.columns:
        d = d[d["town"].isin(towns)]
    if d.empty or "resale_price" not in d.columns or "town" not in d.columns:
        return go.Figure()
    g = d.groupby("town", observed=True)["resale_price"].median().reset_index()
    g = g.sort_values("resale_price", ascending=False).head(top)
    return px.bar(
        g,
        x="town",
        y="resale_price",
        title=f"Median resale price by town (top {min(top, len(g))})",
    )


def fig_price_over_time(
    df: pd.DataFrame, resample: str = "M", towns: Sequence[str] | None = None
) -> go.Figure:
    d = df.copy()
    if "month" not in d.columns or "resale_price" not in d.columns:
        return go.Figure()
    if towns is not None and "town" in d.columns:
        d = d[d["town"].isin(towns)]
    d = d.dropna(subset=["month"]).sort_values("month")
    if d.empty:
        return go.Figure()
    # Map legacy "M" to month-end for pandas 2+ Grouper
    freq = "ME" if resample.upper() == "M" else resample
    s = d.groupby(pd.Grouper(key="month", freq=freq))["resale_price"].median()
    out = s.dropna().reset_index()
    fig = px.line(
        out,
        x="month",
        y="resale_price",
        title=f"Median resale price over time ({resample})",
    )
    return fig


def fig_flat_type_box(df: pd.DataFrame) -> go.Figure:
    if "flat_type" not in df.columns or "resale_price" not in df.columns:
        return go.Figure()
    return px.box(
        df,
        x="flat_type",
        y="resale_price",
        title="Resale price by flat type",
    )


def fig_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    if corr.empty:
        return go.Figure()
    return go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
        ),
        layout={"title": "Correlation matrix (numeric features)"},
    )


def fig_town_median_lines_with_forecast(
    df: pd.DataFrame,
    towns: list[str],
    *,
    horizon: int = 6,
) -> go.Figure:
    """Per-town monthly median (solid) and ETS forecast (dotted) — no misleading global mix."""
    from singapore_eda.forecasting import forecast_ets, monthly_median_price

    fig = go.Figure()
    for t in towns:
        s = monthly_median_price(df, town=t)
        if s.empty or len(s) < 1:
            continue
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                name=t,
                mode="lines",
            )
        )
        fc = forecast_ets(s, horizon=horizon)
        if not fc.empty:
            fig.add_trace(
                go.Scatter(
                    x=fc.index,
                    y=fc.values,
                    name=f"{t} (forecast, {horizon}m)",
                    mode="lines",
                    line=dict(dash="dot"),
                )
            )
    fig.update_layout(
        title="Monthly median resale: one series per selected town; dotted = ETS forecast",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
    )
    return fig
