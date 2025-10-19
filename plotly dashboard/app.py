#imports
import pandas as pd
from hdi_trend import make_hdi_trend_chart
from edu_bars import make_edu_by_region_bars
from gini_life import make_gini_vs_life_scatter
from anim_progress import make_hdi_vs_component_animated

from dash import Dash, dcc, html, Input, Output

#data load
df_long  = pd.read_csv("HDR25_long_format.csv")
df_panel = pd.read_csv("HDR25_panel_format.csv")

#build the dash app
#get all the data - with sorting
all_regions   = sorted(df_long["region"].dropna().unique().tolist())
all_countries = sorted(df_long["country"].dropna().unique().tolist())
year_min, year_max = int(df_long["year"].min()), int(df_long["year"].max())

DEFAULT_REGIONS     = all_regions
DEFAULT_YEAR_RANGE  = [year_min, year_max]
DEFAULT_COUNTRIES   = []  # blank = all countries
DEFAULT_LIFE        = "le"

#start the app
app = Dash(__name__)
app.title = "Global Development according to UN Data"

#wrapper for control cards
def control_card(children):
    return html.Div(
        children,
        style={
            "background":"#fff","border":"1px solid #e9ecef","borderRadius":"12px",
            "padding":"14px","boxShadow":"0 2px 8px rgba(0,0,0,0.04)"
        }
    )

#app layout for top level
app.layout = html.Div(
    [
        html.H2("Global Development", style={"margin":"10px 0 8px 0"}),
        html.Div("Tracing human development through time by looking at UN data."),

        #controls grid
        html.Div(
            [
                #year range (first)
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

                #countries (second) — blank by default = all countries
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

                #regions (third)
                control_card([
                    html.Div("Region(s)"),
                    dcc.Dropdown(
                        id="ctl-region",
                        options=[{"label":r, "value":r} for r in all_regions],
                        value=DEFAULT_REGIONS,
                        multi=True,
                        placeholder="Select regions"
                    ),
                    #resets everything
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

        control_card([
            html.Div("Summary"),
            html.Div(id="summary-text")
        ]),

        html.Div(
            [
                #animated chart (big)
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

                #HDI trend (small)
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

                #education bars (small)
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

                #gini vs Life (small)
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

#display selected years
@app.callback(Output("year-readout","children"), Input("ctl-year-range","value"))
def _readout(yr):
    if not yr:
        return ""
    return f"Selected: {int(yr[0])}–{int(yr[1])}"

#reset button resets everything
@app.callback(
    Output("ctl-region","value"),
    Output("ctl-year-range","value"),
    Output("ctl-country","value"),
    Input("ctl-reset","n_clicks"),
    prevent_initial_call=True
)
def _reset(_n):
    return (DEFAULT_REGIONS, DEFAULT_YEAR_RANGE, DEFAULT_COUNTRIES)

#triggers on filter changes
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

    #overriding these to true - control removed for pop weighting, bubble sizes & life by gender
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
        # life_variant=DEFAULT_LIFE,
        # bubble_size_by_pop=bubble_size_by_pop
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

    #build the summary block
    #if a lot of countries selected, add an elipsis
    def _fmt_list(names: list[str]) -> str:
        if not names:
            return ""
        head = ", ".join(names[:4]) + ("…" if len(names) > 4 else "")
        tail = f" ({len(names)} selected)" if len(names) > 4 else ""
        return head + tail

    #if allregions/countries selected, don't show region names
    is_all_regions = (not region_filter) or (set(region_filter) == set(all_regions))
    region_txt = "" if is_all_regions else _fmt_list(region_filter or [])
    country_txt = _fmt_list(country_filter or [])

    #if both regions and countries anly show countries
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


server = app.server
app.run(debug=True)
