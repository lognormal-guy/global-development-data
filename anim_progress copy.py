import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

#Figure 4. HDI vs Component animated scatter plot - my baby!
# TOTALLY inspired by Hans Rosling's book Factfullness and his ted talk
def make_hdi_vs_component_animated(panel_df: pd.DataFrame,
                                   region_filter=None,
                                   size_scale_px: float = 45.0,
                                   default_component: str = "education_index",
                                   frame_duration_ms: int = 600,
                                   transition_duration_ms: int = 350) -> go.Figure:
    #pretty names
    PRETTY = {
        "education_index": "Education Index",
        "health_index":    "Health Index",
        "income_index":    "Income Index",
        "hdi_calc":        "Human Development Index (HDI)"
    }
    components = ["education_index", "health_index", "income_index"]
    if default_component not in components:
        default_component = "education_index"

    #filter the data
    cols = ["country","iso3","region","year","pop_total","hdi_calc"] + components
    df = panel_df[[c for c in cols if c in panel_df.columns]].copy()

    #rregion filter
    if region_filter is not None:
        if isinstance(region_filter, str):
            region_filter = [region_filter]
        df = df[df["region"].isin(region_filter)]

    #blank chart
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="No data for selected filters")
        return fig

    #keep it all constant for the colors across the charts
    years = sorted(df["year"].unique().tolist())
    regions = sorted(df["region"].dropna().unique().tolist())
    color_map = {r: px.colors.qualitative.Plotly[i % 10] for i, r in enumerate(regions)}

    #build the axis ranges
    def rng(series, lo_floor=0.0, hi_cap=1.0, pad=0.02):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return (lo_floor, hi_cap)
        lo, hi = float(s.min()), float(s.max())
        if hi == lo:
            hi += 1e-6
        span = hi - lo
        lo2, hi2 = max(lo_floor, lo - span*pad), min(hi_cap, hi + span*pad)
        return (lo2, hi2)
    x_range = rng(df["hdi_calc"])
    y_ranges = {c: rng(df[c]) for c in components}

    #size scaling
    sizeref = 2.0 * float(df["pop_total"].max()) / (size_scale_px ** 2) if df["pop_total"].max() > 0 else 1.0
    #needed for the animation to not look wonky
    region_roster = {r: sorted(df.loc[df["region"] == r, "iso3"].unique().tolist()) for r in regions}

    #build the data arrays for a given year, region, component
    def arrays_for(yr: int, reg: str, comp: str):
        roster = region_roster[reg]
        sub = df[(df["year"] == yr) & (df["region"] == reg)].set_index("iso3")
        x = [float(sub.at[i,"hdi_calc"]) if i in sub.index else None for i in roster]
        y = [float(sub.at[i,comp]) if (i in sub.index and pd.notna(sub.at[i,comp])) else None for i in roster]
        size = [float(sub.at[i,"pop_total"]) if i in sub.index else 0.0 for i in roster]
        name_lookup = (df[df["region"] == reg].drop_duplicates("iso3")
                       .set_index("iso3")["country"].to_dict())
        cdata = [[name_lookup.get(i, i), reg, yr,
                  (float(sub.at[i,"pop_total"]) * 1_000_000
                   if i in sub.index and pd.notna(sub.at[i,"pop_total"]) else 0.0)]
                 for i in roster]
        ids = roster
        return x, y, size, cdata, ids

    trace_index = [(c, r) for c in components for r in regions]

    #global tooltip template function
    def hover_for(comp):
        return (
            rf"<b>%{{customdata[0]}}</b><br>"
            r"Region: %{customdata[1]}<br>"
            r"Year: %{customdata[2]:.0f}<br>"
            f"{PRETTY['hdi_calc']}: %{{x:.3f}}<br>"
            f"{PRETTY[comp]}: %{{y:.3f}}<br>"
            r"Population: %{customdata[3]:,.0f}<extra></extra>"
        )

    #initial traces for the first year; once again allowing all to be plotted
    #   but hiding any not selected
    first_year = years[0]
    fig = go.Figure()
    for comp, reg in trace_index:
        x, y, size, cdata, ids = arrays_for(first_year, reg, comp)
        visible = (comp == default_component)
        showlegend = visible
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            name=reg, legendgroup=reg, showlegend=showlegend, ids=ids,
            marker=dict(size=size, sizemode="area", sizeref=sizeref,
                        line=dict(width=0), color=color_map[reg]),
            customdata=cdata,
            hovertemplate=hover_for(default_component),
            visible=visible
        ))

    frames = []
    for yr in years:
        frame_data = []
        for comp, reg in trace_index:
            x, y, size, cdata, ids = arrays_for(yr, reg, comp)
            frame_data.append(dict(x=x, y=y, marker=dict(size=size),
                                   customdata=cdata, ids=ids))
        frames.append(go.Frame(name=str(yr), data=frame_data))
    fig.frames = frames

    #mask function for variants
    def visible_mask(target_comp): 
        return [(comp == target_comp) for comp, _ in trace_index]
    
    #build the selector
    buttons = []
    for comp in components:
        vis = visible_mask(comp)
        hovs = [hover_for(comp)] * len(trace_index)
        buttons.append(dict(
            label=PRETTY[comp],
            method="update",
            args=[{"visible": vis, "hovertemplate": hovs},
                  {"yaxis": {"title": PRETTY[comp], "range": y_ranges[comp]}}]
        ))

    #add tge layout together
    fig.update_layout(
        template="plotly_white",
        title=f"{PRETTY['hdi_calc']} vs Development Components",
        xaxis_title=PRETTY["hdi_calc"],
        yaxis_title=PRETTY[default_component],
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_ranges[default_component]),
        margin=dict(t=60, r=20, l=60, b=50),
        hovermode="closest",
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=1.0, xanchor="right",
            y=1.18, yanchor="top",
            showactive=True,
            buttons=buttons
        )],
        legend=dict(
            orientation="v",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=0.5,
            tracegroupgap=2
        )
    )


    fig.update_layout(
        #animation controls
        updatemenus=list(fig.layout.updatemenus) + [dict(
            type="buttons",
            showactive=False,
            y=0, x=0,
            xanchor="left", yanchor="top",
            pad={"t": 30, "r": 10},
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[[str(y) for y in years],
                           {"frame": {"duration": frame_duration_ms, "redraw": False},
                            "transition": {"duration": transition_duration_ms},
                            "fromcurrent": True}]),
                dict(label="⏸ Pause",
                     method="animate",
                     args=[[None],
                           {"frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}}])
            ]
        )],
        #year slider
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Year: "},
            pad={"t": 35},
            steps=[dict(label=str(y), method="animate",
                        args=[[str(y)],
                              {"frame": {"duration": 0, "redraw": False},
                               "mode": "immediate"}]) for y in years]
        )]
    )
    return fig