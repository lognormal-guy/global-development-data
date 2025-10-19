# app.py — single-file Dash app (no local imports; pulls data from GitHub)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# -------------------------
# Data: load directly from your GitHub repo (raw links)
# -------------------------
DF_LONG_URL  = "https://raw.githubusercontent.com/lognormal-guy/global-development-data/main/HDR25_long_format.csv"
DF_PANEL_URL = "https://raw.githubusercontent.com/lognormal-guy/global-development-data/main/HDR25_panel_format.csv"

df_long  = pd.read_csv(DF_LONG_URL)
df_panel = pd.read_csv(DF_PANEL_URL)

# -------------------------
# Figure 1. HDI trend line chart
# (from hdi_trend.py, inlined)
# -------------------------
def make_hdi_trend_chart(long_df: pd.DataFrame, 
                         region_filter=None, 
                         country_filter=None,
                         year_range=None) -> go.Figure:
    need = long_df[long_df["variable"].isin(["hdi", "pop_total"])].copy()

    # filters
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

    # pivot the data
    wide = (
        need.pivot_table(
            index=["iso3", "country", "region", "year"],
            columns="variable",
            values="value",
            aggfunc="first"
        )
        .reset_index()
    )

    # blank chart
    wide = wide.dropna(subset=["hdi"])
    if wide.empty:
        fig = px.line(title="No data for selected filters")
        fig.update_layout(template="plotly_white")
        return fig

    # population-weighted HDI
    wide["hdipop"] = wide["hdi"] * wide["pop_total"]
    grouped = (
        wide.groupby(["region", "year"], as_index=False)
            .agg(pop_sum=("pop_total", "sum"),
                 hdipop_sum=("hdipop", "sum"),
                 n_countries=("iso3", "nunique"))
    )
    grouped["hdi_weighted"] = grouped["hdipop_sum"] / grouped["pop_sum"]

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

# -------------------------
# Figure 2. Gini vs Life Expectancy scatter
# (from gini_life.py, inlined)
# -------------------------
def make_gini_vs_life_scatter(long_df: pd.DataFrame,
                              region_filter=None,
                              country_filter=None,
                              year: int | None = None,
                              life_variant: str = "le",
                              bubble_size_by_pop: bool = True) -> go.Figure:
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

    # filters
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

    # pivot
    wide = (
        df.pivot_table(
            index=["iso3", "country", "region", "year"],
            columns="variable",
            values="value",
            aggfunc="first"
        )
        .reset_index()
    )

    # blank chart
    if wide.empty:
        fig = px.scatter(title="No data for selected filters")
        fig.update_layout(template="plotly_white")
        return fig

    # consistent axes across variants
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

    # tooltip
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

    # add all variants as separate traces; toggle via dropdown
    for v in variants:
        dv = data_by_v[v]
        start = len(fig.data)
        if not dv.empty:
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
            for tr in fig_v.data:
                tr.visible = (v == life_variant)
                tr.meta = valid_le[v]
                tr.hovertemplate = hover
                tr.update(marker=dict(line=dict(width=0.5, color="white"), opacity=0.85))
                fig.add_trace(tr)

            # regression line
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

    fig.update_layout(
        template="plotly_white",
        title="Income Inequality vs Life Expectancy",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(t=60, r=20, l=50, b=80),
        title_font=dict(size=18),
        xaxis=dict(title="Income Inequality (Gini Index)", range=x_range),
        yaxis=dict(title=valid_le[life_variant], range=y_range),
    )

    def vis_mask(active_key: str):
        mask = [False] * len(fig.data)
        start, end = trace_spans[active_key]
        if start <= end:
            for i in range(start, end + 1):
                mask[i] = True
        return mask

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

# -------------------------
# Figure 3. Education metrics by region (bars)
# (from edu_bars.py, inlined)
# -------------------------
def make_edu_by_region_bars(long_df: pd.DataFrame,
                            region_filter=None,
                            country_filter=None,
                            year: int | list[int] | tuple[int, int] | None = None,
                            pop_weighted: bool = True,
                            wrap_width: int = 18) -> go.Figure:
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

    # filters
    if region_filter is not None:
        if isinstance(region_filter, str):
            region_filter = [region_filter]
        df = df[df["region"].isin(region_filter)]

    if country_filter is not None:
        if isinstance(country_filter, str):
            country_filter = [country_filter]
        df = df[df["country"].isin(country_filter)]

    # normalize year input
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

    # pivot
    wide = (
        df.pivot_table(
            index=["iso3", "country", "region", "year"],
            columns="variable",
            values="value",
            aggfunc="first"
        )
        .reset_index()
    )

    region_order = sorted(wide["region"].dropna().unique().tolist())
    color_map = {r: px.colors.qualitative.Plotly[i % 10] for i, r in enumerate(region_order)}
    have_pop = pop_weighted and ("pop_total" in wide.columns)

    # region averages (optionally population-weighted)
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

    reg_df = pd.DataFrame(reg_rows)
    if reg_df.empty or reg_df[variants].isna().all(axis=None):
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="No data for selected filters")
        return fig

    # wrap long region labels for compact x-axis
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
    n_years = len(years_sorted)
    bar_width = 0.7 / n_years

    # y-axis ranges per variant
    y_ranges = {}
    for v in variants:
        s = reg_df[v].replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            y_ranges[v] = [0, 1]
        else:
            ymin, ymax = float(s.min()), float(s.max())
            pad = (ymax - ymin) * 0.1 if ymax > ymin else max(abs(ymax), 1.0) * 0.1
            y_ranges[v] = [ymin - pad, ymax + pad]

    # tooltip
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

    # add all variants; show only the first by default
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

    def vis_mask(active_key: str):
        mask = [False] * len(fig.data)
        s, e = trace_spans[active_key]
        for k in range(s, e + 1):
            mask[k] = True
        return mask

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

    fig.update_layout(
        template="plotly_white",
        title="Education Metrics by Region",
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
    )
    return fig

# -------------------------
# Figure 4. Animated HDI vs component
# (from anim_progress.py, inlined)
# -------------------------
def make_hdi_vs_component_animated(panel_df: pd.DataFrame,
                                   region_filter=None,
                                   country_filter=None,
                                   size_scale_px: float = 45.0,
                                   default_component: str = "education_index",
                                   frame_duration_ms: int = 600,
                                   transition_duration_ms: int = 350) -> go.Figure:
    PRETTY = {
        "education_index": "Education Index",
        "health_index":    "Health Index",
        "income_index":    "Income Index",
        "hdi_calc":        "Human Development Index (HDI)"
    }
    components = ["education_index", "health_index", "income_index"]
    if default_component not in components:
        default_component = "education_index"

    # select columns
    cols = ["country","iso3","region","year","pop_total","hdi_calc"] + components
    df = panel_df[[c for c in cols if c in panel_df.columns]].copy()

    # filters
    if region_filter is not None:
        if isinstance(region_filter, str):
            region_filter = [region_filter]
        df = df[df["region"].isin(region_filter)]

    if country_filter is not None:
        if isinstance(country_filter, str):
            country_filter = [country_filter]
        df = df[df["country"].isin(country_filter)]

    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="No data for selected filters")
        return fig

    years = sorted(df["year"].unique().tolist())
    regions = sorted(df["region"].dropna().unique().tolist())
    color_map = {r: px.colors.qualitative.Plotly[i % 10] for i, r in enumerate(regions)}

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

    sizeref = 2.0 * float(df["pop_total"].max()) / (size_scale_px ** 2) if df["pop_total"].max() > 0 else 1.0
    region_roster = {r: sorted(df.loc[df["region"] == r, "iso3"].unique().tolist()) for r in regions}

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

    def hover_for(comp):
        return (
            rf"<b>%{{customdata[0]}}</b><br>"
            r"Region: %{customdata[1]}<br>"
            r"Year: %{customdata[2]:.0f}<br>"
            f"{PRETTY['hdi_calc']}: %{{x:.3f}}<br>"
            f"{PRETTY[comp]}: %{{y:.3f}}<br>"
            r"Population: %{customdata[3]:,.0f}<extra></extra>"
        )

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

    def visible_mask(target_comp): 
        return [(comp == target_comp) for comp, _ in trace_index]
    
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

# -------------------------
# Dash app scaffolding
# -------------------------
# Selectors metadata
all_regions   = sorted(df_long["region"].dropna().unique().tolist())
all_countries = sorted(df_long["country"].dropna().unique().tolist())
year_min, year_max = int(df_long["year"].min()), int(df_long["year"].max())

DEFAULT_REGIONS     = all_regions
DEFAULT_YEAR_RANGE  = [year_min, year_max]
DEFAULT_COUNTRIES   = []  # blank = all countries
DEFAULT_LIFE        = "le"

# helper wrapper for cards
def control_card(children):
    return html.Div(
        children,
        style={
            "background":"#fff","border":"1px solid #e9ecef","borderRadius":"12px",
            "padding":"14px","boxShadow":"0 2px 8px rgba(0,0,0,0.04)"
        }
    )

app = Dash(__name__)
app.title = "Global Development according to UN Data"

app.layout = html.Div(
    [
        html.H2("Global Development", style={"margin":"10px 0 8px 0"}),
        html.Div("Tracing human development through time by looking at UN data."),

        # controls
        html.Div(
            [
                control_card([
                    html.Div("Year range"),
                    dcc.RangeSlider(
                        id="ctl-year-range",
                        min=year_min, max=year_max,
                        value=DEFAULT_YEAR_RANGE,
                        marks=None, allowCross=False,
                        tooltip={"placement":"bottom","always_visible":False}
                    ),
                    html.Div(id="year-readout",
                             style={"marginTop":"6px","textAlign":"right",
                                    "fontSize":"0.9rem","opacity":0.8})
                ]),
                control_card([
                    html.Div("Country(ies)"),
                    dcc.Dropdown(
                        id="ctl-country",
                        options=[{"label":c, "value":c} for c in all_countries],
                        value=DEFAULT_COUNTRIES, 
                        multi=True,
                        placeholder="All countries"
                    ),
                    html.Div(
                        "Tip: leave blank for all countries.",
                        style={"marginTop":"6px","fontSize":"0.85rem","opacity":0.7}
                    )
                ]),
                control_card([
                    html.Div("Region(s)"),
                    dcc.Dropdown(
                        id="ctl-region",
                        options=[{"label":r, "value":r} for r in all_regions],
                        value=DEFAULT_REGIONS,
                        multi=True,
                        placeholder="Select regions"
                    ),
                    html.Div(
                        [html.Button("Reset", id="ctl-reset", n_clicks=0)],
                        style={"display":"flex","justifyContent":"flex-end","marginTop":"8px"}
                    )
                ]),
            ],
            style={
                "display":"grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
                "gap":"14px",
                "margin":"14px 0 18px 0"
            }
        ),

        control_card([html.Div("Summary"), html.Div(id="summary-text")]),

        html.Div(
            [
                html.Div(
                    control_card([
                        dcc.Graph(
                            id="fig-animated",
                            config={"displayModeBar": True, "responsive": True},
                            style={"width":"100%","height":"60vh"}
                        )
                    ]),
                    className="card-anim"
                ),
                html.Div(
                    control_card([
                        dcc.Graph(
                            id="fig-hdi-trend",
                            config={"displayModeBar": False, "responsive": True},
                            style={"width":"100%","height":"36vh"}
                        )
                    ]),
                    className="card-small"
                ),
                html.Div(
                    control_card([
                        dcc.Graph(
                            id="fig-edu-bars",
                            config={"displayModeBar": False, "responsive": True},
                            style={"width":"100%","height":"36vh"}
                        )
                    ]),
                    className="card-small"
                ),
                html.Div(
                    control_card([
                        dcc.Graph(
                            id="fig-gini-life",
                            config={"displayModeBar": False, "responsive": True},
                            style={"width":"100%","height":"36vh"}
                        )
                    ]),
                    className="card-small"
                ),
            ],
            className="charts-grid"
        )
    ],
    style={"maxWidth":"1300px","margin":"0 auto","padding":"12px"}
)

# callbacks
@app.callback(Output("year-readout","children"), Input("ctl-year-range","value"))
def _readout(yr):
    if not yr:
        return ""
    return f"Selected: {int(yr[0])}–{int(yr[1])}"

@app.callback(
    Output("ctl-region","value"),
    Output("ctl-year-range","value"),
    Output("ctl-country","value"),
    Input("ctl-reset","n_clicks"),
    prevent_initial_call=True
)
def _reset(_n):
    return (DEFAULT_REGIONS, DEFAULT_YEAR_RANGE, DEFAULT_COUNTRIES)

@app.callback(
    Output("fig-hdi-trend","figure"),
    Output("fig-edu-bars","figure"),
    Output("fig-gini-life","figure"),
    Output("fig-animated","figure"),
    Output("summary-text","children"),
    Input("ctl-region","value"),
    Input("ctl-year-range","value"),
    Input("ctl-country","value"),
)
def _update_all(regions, year_range, countries):
    region_filter  = regions   if regions   else None
    country_filter = countries if countries else None

    yr0, yr1 = int(year_range[0]), int(year_range[1])
    scatter_year = yr1

    bubble_size_by_pop = True
    pop_weighted_bars  = True

    fig_trend = make_hdi_trend_chart(
        df_long,
        region_filter=region_filter,
        country_filter=country_filter,
        year_range=(yr0, yr1)
    )
    fig_bars = make_edu_by_region_bars(
        df_long,
        region_filter=region_filter,
        country_filter=country_filter,
        year=(yr0, yr1),
        pop_weighted=pop_weighted_bars,
        wrap_width=18
    )
    fig_scatter = make_gini_vs_life_scatter(
        df_long,
        region_filter=region_filter,
        country_filter=country_filter,
        year=scatter_year,
    )
    fig_anim = make_hdi_vs_component_animated(
        df_panel,
        region_filter=region_filter,
        country_filter=country_filter,
        size_scale_px=45.0,
        default_component="education_index",
        frame_duration_ms=600,
        transition_duration_ms=350
    )

    # summary
    def _fmt_list(names: list[str]) -> str:
        if not names:
            return ""
        head = ", ".join(names[:4]) + ("…" if len(names) > 4 else "")
        tail = f" ({len(names)} selected)" if len(names) > 4 else ""
        return head + tail

    is_all_regions = (not region_filter) or (set(region_filter) == set(all_regions))
    region_txt = "" if is_all_regions else _fmt_list(region_filter or [])
    country_txt = _fmt_list(country_filter or [])

    if not region_txt and not country_txt:
        lead = "All Regions, All Countries"
    elif country_txt:
        lead = country_txt
    else:
        lead = region_txt

    summary = (
        f"Showing {lead} for {yr0}–{yr1}. "
        f"Bars are {'population-weighted' if pop_weighted_bars else 'unweighted'}; "
        f"scatter bubbles are {'sized by population' if bubble_size_by_pop else 'uniform size'}."
    )

    return fig_trend, fig_bars, fig_scatter, fig_anim, summary

# expose server for hosts like Gunicorn
server = app.server

if __name__ == "__main__":
    app.run(debug=True)
