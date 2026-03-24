"""
public_safety.py
----------------
Renders the Public Safety tab for the Pathway to Improved Cities dashboard.
Loads crime_monthly_pivot.csv, applies lag features, and visualizes
historical trends, forecasts, and choropleth maps by community area.
"""

import os
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import file_loader

CRIME_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crime_monthly_pivot.csv")


@st.cache_data
def _load_crime_data(area_map):
    pivot = pd.read_csv(CRIME_CSV)
    pivot['Community Area Name'] = pivot['Community Area'].map(area_map)
    lag_crime_cols = [c for c in pivot.columns if c not in ['Community Area', 'Year', 'Month', 'Community Area Name']]
    for crime in lag_crime_cols:
        pivot[f'{crime}_lag1'] = pivot.groupby('Community Area')[crime].shift(1)
        pivot[f'{crime}_lag3'] = pivot.groupby('Community Area')[crime].shift(3)
    return pivot


def render(chicago_geo, area_map):
    st.header("Public Safety Dashboard")
    st.markdown("Analyze and forecast crime trends across Chicago community areas.")

    with st.expander("Upload a supplemental dataset"):
        file_loader.uploader(domain="public_safety", local_csv=None, label="Upload a public safety dataset")

    if not os.path.exists(CRIME_CSV):
        st.error(f"`crime_monthly_pivot.csv` not found. Place it in the `src/` folder.")
        return

    pivot = _load_crime_data(area_map)

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