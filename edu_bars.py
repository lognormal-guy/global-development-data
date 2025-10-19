import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

#Figure 3. Education metrics by region bar chart
def make_edu_by_region_bars(long_df: pd.DataFrame,
                            region_filter=None,
                            country_filter=None,
                            year: int | list[int] | tuple[int, int] | None = None,
                            pop_weighted: bool = True,
                            wrap_width: int = 18) -> go.Figure:
    #pretty names
    valid_edu = {
        "eys":      "Expected Years of Schooling (years)",
        "mys":      "Mean Years of Schooling (years)",
        "ineq_edu": "Inequality in Education"
    }
    variants = list(valid_edu.keys())

    need_vars = set(variants)
    if pop_weighted:
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

    #overkill here - and not used on final dashboard, but allows both a year, a tuple of years, or a list of years
    def _normalize_years(dfi: pd.DataFrame, y):
        if y is None:
            return [int(dfi["year"].max())] if not dfi.empty else []
        if isinstance(y, int):
            return [y]
        if isinstance(y, (tuple, list)) and len(y) == 2 and all(isinstance(v, int) for v in y):
            a, b = y
            if a > b:
                a, b = b, a
            return list(range(a, b + 1))
        if isinstance(y, (list, tuple)) and all(isinstance(v, int) for v in y):
            return sorted(set(int(v) for v in y))
        try:
            return [int(y)]
        except Exception:
            return []

    #blank charts
    years = _normalize_years(df, year)
    if not years:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="No data for selected filters")
        return fig

    df = df[df["year"].isin(years)]
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="No data for selected filters")
        return fig

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

    #force region order (so colors match)
    region_order = sorted(wide["region"].dropna().unique().tolist())
    color_map = {r: px.colors.qualitative.Plotly[i % 10] for i, r in enumerate(region_order)}
    have_pop = pop_weighted and ("pop_total" in wide.columns)

    #calculate region averages, population weighted
    reg_rows = []
    for (reg, yr), g in wide.groupby(["region", "year"], dropna=True):
        w = g["pop_total"].astype(float) if have_pop else pd.Series(1.0, index=g.index)
        row = {"region_orig": reg, "year": int(yr)}
        for v in variants:
            if v in g.columns:
                mask = g[v].replace([np.inf, -np.inf], np.nan).notna() & w.notna()
                if mask.any():
                    yvals = g.loc[mask, v].astype(float)
                    ww = w.loc[mask].astype(float)
                    val = float((yvals * ww).sum() / ww.sum()) if have_pop else float(yvals.mean())
                else:
                    val = np.nan
            else:
                val = np.nan
            row[v] = val
        row["pop_used"] = int(w.sum()) if have_pop and np.isfinite(w.sum()) else None
        reg_rows.append(row)

    #final dataframe for plot
    reg_df = pd.DataFrame(reg_rows)
    if reg_df.empty or reg_df[variants].isna().all(axis=None):
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="No data for selected filters")
        return fig

    #needed this to make the region names not take up most of the chart
    def wrap_region(name: str, width: int = 18) -> str:
        if name is None:
            return ""
        s = str(name)
        if len(s) <= width:
            return s
        words = s.split()
        lines, line = [], []
        for w in words:
            if len(" ".join(line + [w])) > width:
                lines.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append(" ".join(line))
        return "<br>".join(lines)

    reg_df["region_wrapped"] = reg_df["region_orig"].apply(lambda x: wrap_region(x, wrap_width))
    reg_df["region_wrapped"] = pd.Categorical(
        reg_df["region_wrapped"],
        categories=[wrap_region(r, wrap_width) for r in region_order],
        ordered=True
    )

    years_sorted = sorted(set(reg_df["year"].astype(int).tolist()))
    multi_year = len(years_sorted) > 1

    #determine y-axis ranges per variant (so chartt is nice across variants)
    y_ranges = {}
    for v in variants:
        s = reg_df[v].replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            y_ranges[v] = [0, 1]
        else:
            ymin, ymax = float(s.min()), float(s.max())
            pad = (ymax - ymin) * 0.1 if ymax > ymin else max(abs(ymax), 1.0) * 0.1
            y_ranges[v] = [ymin - pad, ymax + pad]

    #tooltip templates
    if have_pop:
        hover_single = (
            "<b>%{hovertext}</b><br>"
            "Year: %{customdata[1]}<br>"
            "%{meta}: %{y:.2f}<br>"
            "Population used: %{customdata[0]:,}<extra></extra>"
        )
    else:
        hover_single = (
            "<b>%{hovertext}</b><br>"
            "Year: %{customdata[1]}<br>"
            "%{meta}: %{y:.2f}<extra></extra>"
        )

    fig = go.Figure()
    trace_spans = {}
    n_years = len(years_sorted)
    bar_width = 0.7 / n_years

    #build the bars - one set per variant, one bar per year within each variant
    #doing the same thing as the scatter - all traces added, only one variant visible at a time
    for i, v in enumerate(variants):
        start = len(fig.data)
        for yv in years_sorted:
            g_y = reg_df[reg_df["year"] == yv].sort_values("region_wrapped")
            x_vals = [wrap_region(r, wrap_width) for r in region_order]
            g_map = {rw: val for rw, val in zip(g_y["region_wrapped"], g_y[v])}
            pops  = {rw: pu  for rw, pu  in zip(g_y["region_wrapped"], g_y["pop_used"])}

            for rw, rname in zip(x_vals, region_order):
                y_val = g_map.get(rw, np.nan)
                pu = pops.get(rw, None)
                fig.add_trace(go.Bar(
                    x=[rw],
                    y=[y_val],
                    name=str(yv),
                    legendgroup=f"yr_{yv}",
                    hovertext=rname,
                    meta=valid_edu[v],
                    customdata=[[pu, yv]],
                    marker_color=color_map.get(rname),
                    marker_line_width=0,
                    width=bar_width,
                    hovertemplate=hover_single,
                    visible=(i == 0),
                    showlegend=False
                ))
        end = len(fig.data) - 1
        trace_spans[v] = (start, end)

    #visibility mask for traces; returns true for active
    def vis_mask(active_key: str):
        mask = [False] * len(fig.data)
        s, e = trace_spans[active_key]
        for k in range(s, e + 1):
            mask[k] = True
        return mask

    #add toggle for variants
    buttons = [
        dict(
            label=valid_edu[k],
            method="update",
            args=[
                {"visible": vis_mask(k)},
                {"yaxis": {"title": valid_edu[k], "range": y_ranges[k]}}
            ]
        )
        for k in variants
    ]

    if multi_year:
        title_suffix = f"{years_sorted[0]}–{years_sorted[-1]}"
        anno_text = f"{years_sorted[0]}–{years_sorted[-1]}"
    else:
        title_suffix = f"{years_sorted[0]}"
        anno_text = f"{years_sorted[0]}"

    #style and layout
    fig.update_layout(
        template="plotly_white",
        title=f"Education Metrics by Region",# ({title_suffix})",
        xaxis=dict(
            title="Region",
            categoryorder="array",
            categoryarray=[wrap_region(r, wrap_width) for r in region_order]
        ),
        yaxis=dict(title=valid_edu[variants[0]], range=y_ranges[variants[0]]),
        margin=dict(t=60, r=20, l=50, b=80),
        title_font=dict(size=18),
        barmode="group",
        bargap=0.05,
        bargroupgap=0.002,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=1.0, xanchor="right",
            y=1.18, yanchor="top",
            buttons=buttons,
            showactive=True
        )],
        #annotations=[dict(
        #    x=0.98, y=0.98, xref="paper", yref="paper",
        #    text=anno_text, showarrow=False, align="right",
        #    font=dict(size=28)
        #)]
    )
    return fig