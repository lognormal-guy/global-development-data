import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

#Figure 2. Gini vs Life Expectancy scatter plot
def make_gini_vs_life_scatter(long_df: pd.DataFrame,
                              region_filter=None,
                              country_filter=None,
                              year: int | None = None,
                              life_variant: str = "le",
                              bubble_size_by_pop: bool = True) -> go.Figure:
    #pretty names
    valid_le = {
        "le":   "Life Expectancy at Birth (Total)",
        "le_f": "Life Expectancy at Birth (Female)",
        "le_m": "Life Expectancy at Birth (Male)"
    }
    variants = ["le", "le_f", "le_m"]
    if life_variant not in variants:
        life_variant = "le"

    need_vars = {"ineq_inc", "le", "le_f", "le_m"}
    if bubble_size_by_pop:
        need_vars.add("pop_total")

    df = long_df[long_df["variable"].isin(need_vars)].copy()

    #filters
    if region_filter is not None:
        if isinstance(region_filter, str):
            region_filter = [region_filter]
        df = df[df["region"].isin(region_filter)]

    if country_filter is not None:
        if isinstance(country_filter, str):
            country_filter = [country_filter]
        df = df[df["country"].isin(country_filter)]

    if year is None:
        year = int(df["year"].max())
    df = df[df["year"] == year]

    #pivot the data
    wide = (
        df.pivot_table(
            index=["iso3", "country", "region", "year"],
            columns="variable",
            values="value",
            aggfunc="first"
        )
        .reset_index()
    )

    #blank chart
    if wide.empty:
        fig = px.scatter(title="No data for selected filters")
        fig.update_layout(template="plotly_white")
        return fig

    #determine axis ranges - fixes this so they don't change when switching variants
    data_by_v = {v: wide.dropna(subset=["ineq_inc", v]).copy() for v in variants}
    x_min = min((d["ineq_inc"].min() for d in data_by_v.values() if not d.empty))
    x_max = max((d["ineq_inc"].max() for d in data_by_v.values() if not d.empty))
    y_min = min((d[v].min() for v, d in data_by_v.items() if not d.empty))
    y_max = max((d[v].max() for v, d in data_by_v.items() if not d.empty))
    x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
    x_range = [x_min - x_pad, x_max + x_pad]
    y_range = [y_min - y_pad, y_max + y_pad]

    region_order = sorted(wide["region"].dropna().unique().tolist())
    size_col = "pop_total" if (bubble_size_by_pop and "pop_total" in wide.columns) else None
    custom_cols = ["year"] + (["pop_total"] if size_col else [])

    #tooltip templates
    if size_col:
        hover = (
            "<b>%{hovertext}</b><br>"
            "Year: %{customdata[0]}<br>"
            "Region: %{fullData.name}<br>"
            "Gini: %{x:.1f}<br>"
            "%{meta}: %{y:.1f}<br>"
            "Population: %{customdata[1]:,}<extra></extra>"
        )
    else:
        hover = (
            "<b>%{hovertext}</b><br>"
            "Year: %{customdata[0]}<br>"
            "Region: %{fullData.name}<br>"
            "Gini: %{x:.1f}<br>"
            "%{meta}: %{y:.1f}<extra></extra>"
        )

    fig = go.Figure()
    trace_spans = {}

    #ok - seems like the oly way to make this look good is to add all the variants as separate traces
    #  and then 'hide' the ones other than the one we want to see via dropdown
    for v in variants:
        #filter for variant
        dv = data_by_v[v]
        start = len(fig.data)
        if not dv.empty:
            #scatterplot
            fig_v = px.scatter(
                dv,
                x="ineq_inc",
                y=v,
                color="region",
                size=size_col,
                size_max=36 if size_col else None,
                hover_name="country",
                custom_data=dv[custom_cols],
                labels={
                    "ineq_inc": "Income Inequality (Gini Index)",
                    v: valid_le[v],
                    "region": "Region",
                    "year": "Year"
                },
                category_orders={"region": region_order},
            )
            #styling - only show traces for the selected variant
            for tr in fig_v.data:
                tr.visible = (v == life_variant)
                tr.meta = valid_le[v]
                tr.hovertemplate = hover
                tr.update(marker=dict(line=dict(width=0.5, color="white"), opacity=0.85))
                fig.add_trace(tr)

            #add linear regression line
            if dv.shape[0] >= 2:
                x = dv["ineq_inc"].to_numpy(dtype=float)
                yv = dv[v].to_numpy(dtype=float)
                b, a = np.polyfit(x, yv, 1)  # slope, intercept
                xx0, xx1 = float(x_range[0]), float(x_range[1])
                y0, y1 = a + b * xx0, a + b * xx1
                fig.add_trace(go.Scatter(
                    x=[xx0, xx1], y=[y0, y1],
                    mode="lines",
                    line=dict(width=2, dash="dash"),
                    name="Linear fit",
                    hovertemplate="Fit: y = %{customdata[0]:.2f} + %{customdata[1]:.2f}·x<extra></extra>",
                    customdata=[[a, b], [a, b]],
                    showlegend=False,
                    visible=(v == life_variant)
                ))

        end = len(fig.data) - 1
        trace_spans[v] = (start, end)

    #layout and finish the tooltips
    fig.update_layout(
        template="plotly_white",
        title="Income Inequality vs Life Expectancy",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(t=60, r=20, l=50, b=80),
        title_font=dict(size=18),
        xaxis=dict(title="Income Inequality (Gini Index)", range=x_range),
        yaxis=dict(title=valid_le[life_variant], range=y_range),
        ## This part not needed on dashboard (since year set by filter)
        ## annotations=[dict(
        ##    x=0.98, y=0.98, xref="paper", yref="paper",
        ##    text=str(year), showarrow=False, align="right",
        ##    font=dict(size=28)
        ##)]
    )

    #visibility mask for traces; returns true for active
    def vis_mask(active_key: str):
        mask = [False] * len(fig.data)
        start, end = trace_spans[active_key]
        if start <= end:
            for i in range(start, end + 1):
                mask[i] = True
        return mask

    #add toggle for total/male/female
    buttons = [
        dict(label=valid_le[k], method="update",
             args=[{"visible": vis_mask(k)},
                   {"yaxis": {"title": valid_le[k]}}])
        for k in ["le", "le_f", "le_m"]
    ]
    fig.update_layout(
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=1.0, xanchor="right",
            y=1.18, yanchor="top",
            buttons=buttons,
            showactive=True
        )]
    )
    return fig