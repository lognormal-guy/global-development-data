import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

#Figure 1. HDI trend line chart
def make_hdi_trend_chart(long_df: pd.DataFrame, 
                         region_filter=None, 
                         country_filter=None,
                         year_range=None) -> go.Figure:
    
    need = long_df[long_df["variable"].isin(["hdi", "pop_total"])].copy()

    #filters
    if region_filter is not None:
        if isinstance(region_filter, str):
            region_filter = [region_filter]
        need = need[need["region"].isin(region_filter)]

    if country_filter is not None:
        if isinstance(country_filter, str):
            country_filter = [country_filter]
        need = need[need["country"].isin(country_filter)]

    if year_range is not None:
        y0, y1 = year_range
        need = need[(need["year"] >= y0) & (need["year"] <= y1)]

    #pivot the data
    wide = (
        need.pivot_table(
            index=["iso3", "country", "region", "year"],
            columns="variable",
            values="value",
            aggfunc="first"
        )
        .reset_index()
    )

    #blank chart
    wide = wide.dropna(subset=["hdi"])
    if wide.empty:
        fig = px.line(title="No data for selected filters")
        fig.update_layout(template="plotly_white")
        return fig

    #population-weighted HDI calculation
    wide["hdipop"] = wide["hdi"] * wide["pop_total"]
    grouped = (
        wide.groupby(["region", "year"], as_index=False)
            .agg(pop_sum=("pop_total", "sum"),
                 hdipop_sum=("hdipop", "sum"),
                 n_countries=("iso3", "nunique"))
    )
    grouped["hdi_weighted"] = grouped["hdipop_sum"] / grouped["pop_sum"]

    #line chart
    fig = px.line(
        grouped,
        x="year",
        y="hdi_weighted",
        color="region",
        markers=True,
        labels={"year":"Year","hdi_weighted":"Population-Weighted HDI","region":"Region"},
        title="Rising Human Development Across Regions"
    )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        margin=dict(t=60, r=20, l=50, b=80),
        title_font=dict(size=18)
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=6))
    fig.update_traces(
        hovertemplate="Year: %{x}<br>Population-Weighted HDI: %{y:.3f}<br>Region: %{fullData.name}<br><extra></extra>"
    )
    return fig