import requests
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import crash

st.set_page_config(
    page_title="Pathway to Improved Cities",
    layout="wide"
)

st.title("Pathway to Improved Cities Dashboard")

# ──────────────────────────────────────────────
# Shared: Community Area mapping + GeoJSON
# ──────────────────────────────────────────────

community_area_names = {
    1: "Rogers Park", 2: "West Ridge", 3: "Uptown", 4: "Lincoln Square",
    5: "North Center", 6: "Lake View", 7: "Lincoln Park", 8: "Near North Side",
    9: "Edison Park", 10: "Norwood Park", 11: "Jefferson Park", 12: "Forest Glen",
    13: "North Park", 14: "Albany Park", 15: "Portage Park", 16: "Irving Park",
    17: "Dunning", 18: "Montclare", 19: "Belmont Cragin", 20: "Hermosa",
    21: "Avondale", 22: "Logan Square", 23: "Humboldt Park", 24: "West Town",
    25: "Austin", 26: "West Garfield Park", 27: "East Garfield Park", 28: "Near West Side",
    29: "North Lawndale", 30: "South Lawndale", 31: "Lower West Side", 32: "Loop",
    33: "Near South Side", 34: "Armour Square", 35: "Douglas", 36: "Oakland",
    37: "Fuller Park", 38: "Grand Boulevard", 39: "Kenwood", 40: "Washington Park",
    41: "Hyde Park", 42: "Woodlawn", 43: "South Shore", 44: "Chatham",
    45: "Avalon Park", 46: "South Chicago", 47: "Burnside", 48: "Calumet Heights",
    49: "Roseland", 50: "Pullman", 51: "South Deering", 52: "East Side",
    53: "West Pullman", 54: "Riverdale", 55: "Hegewisch", 56: "Garfield Ridge",
    57: "Archer Heights", 58: "Brighton Park", 59: "McKinley Park", 60: "Bridgeport",
    61: "New City", 62: "West Elsdon", 63: "Gage Park", 64: "Clearing",
    65: "West Lawn", 66: "Chicago Lawn", 67: "West Englewood", 68: "Englewood",
    69: "Greater Grand Crossing", 70: "Ashburn", 71: "Auburn Gresham", 72: "Beverly",
    73: "Washington Heights", 74: "Mount Greenwood", 75: "Morgan Park",
    76: "O'Hare", 77: "Edgewater"
}

@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/RandomFractals/ChicagoCrimes/master/data/chicago-community-areas.geojson"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

chicago_geo = load_geojson()

area_map = {
    int(f['properties']['area_num_1']): f['properties']['community']
    for f in chicago_geo['features']
}

# ──────────────────────────────────────────────
# Tab layout
# ──────────────────────────────────────────────

tab_safety, tab_transport, tab_infra, tab_socio = st.tabs([
    "Public Safety",
    "Transportation",
    "Infrastructure",
    "Socioeconomics & Diversity"
])


# ══════════════════════════════════════════════
# TAB 1 — PUBLIC SAFETY (formerly Crime)
# ══════════════════════════════════════════════

with tab_safety:
    st.header("Public Safety Dashboard")
    st.markdown("Analyze and forecast crime trends across Chicago community areas.")

    @st.cache_data
    def load_crime_data():
        pivot = pd.read_csv("crime_monthly_pivot.csv")
        pivot['Community Area Name'] = pivot['Community Area'].map(area_map)
        lag_crime_cols = [c for c in pivot.columns if c not in ['Community Area', 'Year', 'Month', 'Community Area Name']]
        for crime in lag_crime_cols:
            pivot[f'{crime}_lag1'] = pivot.groupby('Community Area')[crime].shift(1)
            pivot[f'{crime}_lag3'] = pivot.groupby('Community Area')[crime].shift(3)
        return pivot

    try:
        pivot = load_crime_data()

        crime_cols = [
            c for c in pivot.columns
            if c not in ['Community Area', 'Year', 'Month', 'Community Area Name']
            and not c.endswith('_lag1')
            and not c.endswith('_lag3')
        ]
        community_areas = sorted(pivot['Community Area Name'].dropna().unique())

        col1, col2 = st.columns(2)
        with col1:
            selected_area = st.selectbox("Select Community Area", community_areas, key="safety_area")
        with col2:
            selected_crime = st.selectbox("Select Crime Type", crime_cols, key="safety_crime")

        area_data = pivot[pivot['Community Area Name'] == selected_area].sort_values(['Year', 'Month'])

        st.subheader(f"Historical {selected_crime} counts — {selected_area}")
        st.line_chart(area_data[selected_crime].values)

        # Prediction
        feature_cols = [
            c for c in pivot.columns
            if c not in ['Community Area', 'Year', 'Month', 'Community Area Name', selected_crime]
        ]
        model_data = area_data.dropna(subset=feature_cols + [selected_crime])

        if len(model_data) > 0:
            X = model_data[feature_cols]
            y = model_data[selected_crime]
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            prediction = model.predict(X.iloc[-1].values.reshape(1, -1))[0]
            st.subheader(f"Predicted {selected_crime} counts for next month")
            st.metric(label="Forecast", value=round(prediction))
        else:
            st.info("Not enough data to generate a prediction.")

        # Crime Map
        st.subheader("Crime Distribution Map")
        col_map1, col_map2 = st.columns(2)

        with col_map1:
            crime_map_type = st.selectbox("Crime type (historical map)", crime_cols, key="crime_map_select")
            map_data = pivot.groupby('Community Area Name')[crime_map_type].sum().reset_index()
            fig = px.choropleth_mapbox(
                map_data, geojson=chicago_geo,
                locations='Community Area Name', featureidkey="properties.community",
                color=crime_map_type, color_continuous_scale="Reds",
                mapbox_style="open-street-map", zoom=9,
                center={"lat": 41.8781, "lon": -87.6298}, opacity=0.5,
                labels={crime_map_type: "Crime Count"}
            )
            fig.update_coloraxes(colorbar_tickformat='.2f')
            st.plotly_chart(fig, use_container_width=True)

        with col_map2:
            crime_pred_type = st.selectbox("Crime type (predicted map)", crime_cols, key="crime_map_pred_select")
            crime_pred_type_upper = crime_pred_type.upper()
            lag_cols = [f'{crime_pred_type_upper}_lag1', f'{crime_pred_type_upper}_lag3']
            missing_lags = [col for col in lag_cols if col not in pivot.columns]

            if missing_lags:
                st.warning(f"No lag features found for '{crime_pred_type}'.")
            else:
                pred_data = pivot.dropna(subset=lag_cols + [crime_pred_type_upper])
                if pred_data.empty:
                    st.warning(f"Not enough data to predict for '{crime_pred_type}'.")
                else:
                    pred_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    pred_model.fit(pred_data[lag_cols], pred_data[crime_pred_type_upper])
                    latest_month = pivot.groupby('Community Area Name').tail(1).copy()
                    latest_month['Predicted'] = pred_model.predict(latest_month[lag_cols].fillna(0)).round(2)
                    fig_pred = px.choropleth_mapbox(
                        latest_month[['Community Area Name', 'Predicted']],
                        geojson=chicago_geo,
                        locations='Community Area Name', featureidkey="properties.community",
                        color='Predicted', color_continuous_scale="Reds",
                        mapbox_style="open-street-map", zoom=9,
                        center={"lat": 41.8781, "lon": -87.6298}, opacity=0.5,
                        labels={'Predicted': f'Predicted {crime_pred_type} Count'}
                    )
                    fig_pred.update_coloraxes(colorbar_tickformat='.2f')
                    st.plotly_chart(fig_pred, use_container_width=True)

    except FileNotFoundError:
        st.error("`crime_monthly_pivot.csv` not found. Please place it in the working directory.")


# ══════════════════════════════════════════════
# TAB 2 — TRANSPORTATION
# ══════════════════════════════════════════════

with tab_transport:
    crash.render(chicago_geo=chicago_geo)


# ══════════════════════════════════════════════
# TAB 3 — INFRASTRUCTURE
# ══════════════════════════════════════════════

with tab_infra:
    st.header("Infrastructure Dashboard")
    st.markdown(
        "Track infrastructure quality, 311 service requests, building permits, and public facility conditions."
    )
    st.info(
        "No infrastructure dataset loaded yet.\n\n"
        "**To connect data:** Place a file named `infrastructure_monthly.csv` in the working directory.\n\n"
        "**Suggested datasets (Chicago Data Portal):**\n"
        "- [311 Service Requests](https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy)\n"
        "- [Building Permits](https://data.cityofchicago.org/Buildings/Building-Permits/ydr8-5enu)\n"
        "- [Street Lights - All Out](https://data.cityofchicago.org/Service-Requests/311-Service-Requests-Street-Lights-All-Out/zuxi-7xem)\n"
        "- [Pothole Repairs](https://data.cityofchicago.org/Service-Requests/311-Service-Requests-Pot-Holes-Reported/7as2-ds3y)\n\n"
        "**Expected CSV columns:** `Community Area`, `Year`, `Month`, + metric columns"
    )


# ══════════════════════════════════════════════
# TAB 4 — SOCIOECONOMICS & DIVERSITY
# ══════════════════════════════════════════════

with tab_socio:
    st.header("Socioeconomics & Diversity Dashboard")
    st.markdown(
        "Explore income, poverty, racial demographics, educational attainment, and hardship indices across community areas."
    )

    @st.cache_data
    def load_socio_data():
        # Expected columns: Community Area, Year (or static), demographic/economic metrics
        return pd.read_csv("socioeconomics.csv")

    try:
        socio_df = load_socio_data()
        socio_df['Community Area Name'] = socio_df['Community Area'].map(area_map)

        socio_metric_cols = [
            c for c in socio_df.columns
            if c not in ['Community Area', 'Year', 'Community Area Name']
        ]

        s_metric = st.selectbox("Select Indicator", socio_metric_cols, key="socio_metric")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f"{s_metric} by Community Area")
            s_map_data = socio_df[['Community Area Name', s_metric]].dropna()
            fig_s = px.choropleth_mapbox(
                s_map_data, geojson=chicago_geo,
                locations='Community Area Name', featureidkey="properties.community",
                color=s_metric, color_continuous_scale="Purples",
                mapbox_style="open-street-map", zoom=9,
                center={"lat": 41.8781, "lon": -87.6298}, opacity=0.6,
                labels={s_metric: s_metric}
            )
            st.plotly_chart(fig_s, use_container_width=True)

        with col2:
            st.subheader(f"Top 10 — {s_metric}")
            top10 = (
                socio_df[['Community Area Name', s_metric]]
                .dropna()
                .sort_values(s_metric, ascending=False)
                .head(10)
            )
            fig_bar = px.bar(
                top10, x=s_metric, y='Community Area Name',
                orientation='h', color=s_metric,
                color_continuous_scale='Purples'
            )
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Correlation Explorer")
        if len(socio_metric_cols) >= 2:
            col_x, col_y = st.columns(2)
            with col_x:
                x_var = st.selectbox("X-axis variable", socio_metric_cols, key="socio_x")
            with col_y:
                y_var = st.selectbox("Y-axis variable", socio_metric_cols,
                                     index=min(1, len(socio_metric_cols) - 1), key="socio_y")

            fig_scatter = px.scatter(
                socio_df, x=x_var, y=y_var,
                hover_name='Community Area Name',
                trendline='ols',
                color=y_var, color_continuous_scale='Purples',
                labels={x_var: x_var, y_var: y_var}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Add more columns to enable correlation analysis.")

    except FileNotFoundError:
        st.info(
            "No socioeconomic dataset loaded yet.\n\n"
            "**To connect data:** Place a file named `socioeconomics.csv` in the working directory.\n\n"
            "**Suggested datasets:**\n"
            "- [Census Community Area Profiles](https://data.cityofchicago.org/Health-Human-Services/Census-Data-Selected-socioeconomic-indicators-in-C/kn9c-c2s2)\n"
            "- [Chicago Health Atlas](https://www.chicagohealthatlas.org/)\n"
            "- [ACS 5-Year Estimates via Census API](https://www.census.gov/data/developers/data-sets/acs-5year.html)\n\n"
            "**Expected CSV columns:** `Community Area`, + indicator columns (e.g. `Poverty Rate`, `Median Income`, `% Black`, `% Hispanic`, `Hardship Index`)"
        )