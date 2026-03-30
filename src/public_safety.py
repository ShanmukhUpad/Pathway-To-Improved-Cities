import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import file_loader
import ml_predictor

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
        st.info("No local crime data found. Fetching the latest data from the Chicago Data Portal…")
        try:
            import data_fetcher
            data_fetcher.fetch_crimes(force=True)
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Auto-fetch failed: {exc}")
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

    # Prediction — use only the selected crime's own lag features + Month for seasonality
    crime_upper = selected_crime.upper()
    lag1_col = f'{crime_upper}_lag1'
    lag3_col = f'{crime_upper}_lag3'
    feature_cols = [c for c in [lag1_col, lag3_col, 'Month'] if c in area_data.columns]
    model_data = area_data.dropna(subset=feature_cols + [selected_crime])

    if len(model_data) >= 6:
        X = model_data[feature_cols].values
        y = model_data[selected_crime].values

        # Chronological split: train on earlier months, test on most recent 20%
        split = max(1, int(len(X) * 0.8))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        if len(X_test) >= 2:
            y_pred = model.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))
        else:
            rmse = r2 = None

        prediction = model.predict(X[-1].reshape(1, -1))[0]
        st.subheader(f"Predicted {selected_crime} counts for next month")
        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Forecast", round(prediction))
        if rmse is not None:
            col_f2.metric("RMSE", f"{rmse:.2f}")
            col_f3.metric("R²", f"{r2:.3f}")
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

    # ── Generic ML predictor on the full crime pivot ──────────────────────────
    base_crime_cols = [
        c for c in pivot.columns
        if c not in ['Community Area', 'Year', 'Month', 'Community Area Name']
        and not c.endswith('_lag1')
        and not c.endswith('_lag3')
    ]
    lag_cols = [c for c in pivot.columns if c.endswith('_lag1') or c.endswith('_lag3')]
    ml_predictor.render_predictor(
        pivot.dropna(subset=lag_cols[:2] if lag_cols else base_crime_cols[:1]),
        key_prefix="safety",
        default_target=selected_crime,
        default_features=lag_cols[:6] + ['Community Area', 'Year', 'Month'],
    )