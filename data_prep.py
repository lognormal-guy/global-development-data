# %%
import pandas as pd
import numpy as np

# %%
#read the data
df = pd.read_csv('HDR25_Composite_indices_complete_time_series.csv',encoding='latin1')

#read the wb regions
wb_regions = pd.read_csv('world-regions-according-to-the-world-bank.csv')

# %%
#reshape the data
#define id columns; melt everything else
id_vars = ['iso3', 'country', 'hdicode']
drop_vars = ['hdi_rank_2023','gii_rank_2023', 'rankdiff_hdi_phdi_2023','gdi_group_2023','region'] #ranks + region which isn't well populated
val_cols = list(df.columns)
val_cols = [x for x in val_cols if x not in (id_vars + drop_vars)]

#melt apart
df_long = df.melt(
    id_vars=(id_vars + drop_vars),
    var_name='variable_year',
    value_name='value'
)

#split years, make year numeric, drop bad cols 
df_long['year'] = df_long['variable_year'].str[-4:]
df_long['variable'] = df_long['variable_year'].str[:-5]
df_long = df_long[id_vars + ['variable', 'year', 'value']]
df_long.year = pd.to_numeric(df_long.year, errors='coerce')

#drop aggregates (any iso3 code with length less than 3)
df_long = df_long[df_long['iso3'].str.len() == 3]

#pivot back
df_mid = (
    df_long
    .pivot_table(
        index=id_vars + ['year'],
        columns='variable',
        values='value',
        aggfunc='first'
    )
    .reset_index()
)
df_mid.columns.name = None

# %%
#imputing missing data
id_vars= id_vars + ['year']
val_cols = [c for c in df_mid.columns if c not in id_vars]
df_panel = df_mid.sort_values(["iso3", "year"]).copy()

#impute JUST in the middle
for c in [x for x in val_cols if x != "pop_total"]:
    df_panel[c] = (
        df_panel.groupby("iso3", group_keys=False)[c]
                .apply(lambda s: s.interpolate(method="linear", limit_area="inside"))
    )

# %%

df_panel["education_index"] = (np.minimum(df_panel["mys"], 15) / 15 + np.minimum(df_panel["eys"], 18) / 18) / 2
df_panel["health_index"] = (np.maximum(np.minimum(df_panel["le"], 85), 20) - 20) / (85 - 20)
df_panel["income_index"] = (np.log(np.maximum(np.minimum(df_panel["gnipc"], 75000), 100)) - np.log(100)) / (np.log(75000) - np.log(100))
df_panel["hdi_calc"] = np.round((df_panel["education_index"] * df_panel["health_index"] * df_panel["income_index"]) ** (1/3),3)

# %%
df_panel = df_panel[['iso3', 'country', 'year', 'mys', 'eys', 'le', 'gnipc', 'education_index', 'health_index', 'income_index', 'hdi_calc', 'pop_total']].copy()
df_panel = df_panel.dropna(subset=["education_index", "health_index", "income_index"])

# %%
#map region from wb_regions onto the df_long
region_map = dict(zip(wb_regions['code'], wb_regions['region']))
df_panel['region'] = df_panel['iso3'].map(region_map)
df_long['region'] = df_long['iso3'].map(region_map)

# %%
#output the data
df_long.to_csv('HDR25_long_format.csv', index=False)
df_panel.to_csv('HDR25_panel_format.csv', index=False)

# %%



