import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import geopandas as gpd
import file_loader
import ml_predictor
import map_utils

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


@st.cache_data(show_spinner="Loading community area geometries...")
def _load_community_areas_gdf():
    """Load Chicago community area polygons as a GeoDataFrame (cached)."""
    geo_url = "https://raw.githubusercontent.com/RandomFractals/ChicagoCrimes/master/data/chicago-community-areas.geojson"
    gdf_ca = gpd.read_file(geo_url)
    gdf_ca["area_num_1"] = gdf_ca["area_num_1"].astype(int)
    return gdf_ca


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

    mapbox_style = map_utils.mapbox_style_picker(key_prefix="safety")

    area_data = pivot[pivot['Community Area Name'] == selected_area].sort_values(['Year', 'Month'])

    st.subheader(f"Historical {selected_crime} counts — {selected_area}")
    st.line_chart(area_data[selected_crime].values)

    # ── Historical trend summary ──────────────────────────────────────────────
    if not area_data[selected_crime].dropna().empty:
        series = area_data[selected_crime].dropna()
        latest_val = series.iloc[-1]
        mean_val   = series.mean()
        max_val    = series.max()
        pct_vs_mean = ((latest_val - mean_val) / mean_val * 100) if mean_val else 0
        direction  = "above" if pct_vs_mean > 0 else "below"
        trend_color = "🔴" if pct_vs_mean > 10 else ("🟡" if pct_vs_mean > 0 else "🟢")
        st.info(
            f"{trend_color} **{selected_area} - {selected_crime} trend:** "
            f"The most recent month recorded **{latest_val:.0f} incidents**, which is "
            f"**{abs(pct_vs_mean):.1f}% {direction} the historical average** of {mean_val:.1f}. "
            f"The all-time peak was **{max_val:.0f} incidents**. "
            + ("This elevated level suggests increased enforcement or community intervention may be warranted."
               if pct_vs_mean > 10
               else ("Counts are near the historical average. Monitor for seasonal changes."
                     if abs(pct_vs_mean) <= 10
                     else "Counts are below the historical average, suggesting a positive trend."))
        )

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

        # ── Forecast interpretation ───────────────────────────────────────────
        r2_label  = "strong" if r2 and r2 >= 0.1 else ("weak" if r2 and r2 > 0 else "no correlation")
        trend_vs_latest = prediction - latest_val if not area_data[selected_crime].dropna().empty else 0
        chg_dir   = "increase" if trend_vs_latest > 0 else "decrease"
        st.info(
            f"**Forecast interpretation:** The model predicts **{round(prediction)} {selected_crime} incidents** "
            f"next month in {selected_area}, a **{abs(trend_vs_latest):.0f}-incident {chg_dir}** from last month. "
            + (f"An RMSE of {rmse:.2f} means predictions are typically off by plus or minus {rmse:.1f} incidents. "
               f"An R² of {r2:.3f} indicates a **{r2_label}** fit: "
               + ("the model captures the crime pattern well and forecasts are reliable."
                  if r2 >= 0.1
                  else ("the model shows a weak relationship. Treat the forecast as directional only."
                        if r2 > 0
                        else "the model shows no explanatory power. The forecast should not be relied upon."))
               if rmse is not None else "")
        )
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
            mapbox_style=mapbox_style, zoom=9,
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
                    mapbox_style=mapbox_style, zoom=9,
                    center={"lat": 41.8781, "lon": -87.6298}, opacity=0.5,
                    labels={'Predicted': f'Predicted {crime_pred_type} Count'}
                )
                fig_pred.update_coloraxes(colorbar_tickformat='.2f')
                st.plotly_chart(fig_pred, use_container_width=True)

                # ── Predicted map summary ─────────────────────────────────────
                top_pred = latest_month.nlargest(3, "Predicted")[["Community Area Name", "Predicted"]]
                top_names = ", ".join(
                    f"{r['Community Area Name']} ({r['Predicted']:.0f})"
                    for _, r in top_pred.iterrows()
                )
                st.info(
                    f"**Predicted hotspots for {crime_pred_type} next month:** {top_names}. "
                    "These areas have the highest forecasted incident counts based on recent lag patterns. "
                    "Proactive resource allocation here may reduce impact."
                )

    # ── Crime Scatterplots ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Crime Scatterplots")

    # Aggregate total crime count per community area for the selected crime type
    scatter_agg = pivot.groupby("Community Area")[selected_crime].sum().reset_index()
    scatter_agg.columns = ["Community Area", "Total Crime Count"]
    scatter_agg["Community Area Name"] = scatter_agg["Community Area"].map(area_map)

    scatter_col1, scatter_col2 = st.columns(2)

    with scatter_col1:
        fig_sc1 = px.scatter(
            scatter_agg,
            x="Community Area",
            y="Total Crime Count",
            color="Community Area Name",
            hover_name="Community Area Name",
            hover_data={"Community Area": True, "Total Crime Count": True},
            labels={
                "Community Area": "Community Area Number",
                "Total Crime Count": f"Total {selected_crime} Count",
            },
            title=f"Community Area vs Total {selected_crime}",
        )
        fig_sc1.update_layout(
            showlegend=False,
            margin={"t": 40, "b": 0},
        )
        st.plotly_chart(fig_sc1, use_container_width=True)

    with scatter_col2:
        # Build predicted vs actual scatter if predictions are available
        crime_pred_upper = selected_crime.upper()
        sc_lag_cols = [f"{crime_pred_upper}_lag1", f"{crime_pred_upper}_lag3"]
        sc_missing = [c for c in sc_lag_cols if c not in pivot.columns]

        if not sc_missing:
            sc_pred_data = pivot.dropna(subset=sc_lag_cols + [crime_pred_upper])
            if len(sc_pred_data) >= 6:
                sc_model = RandomForestRegressor(n_estimators=100, random_state=42)
                sc_model.fit(sc_pred_data[sc_lag_cols], sc_pred_data[crime_pred_upper])
                sc_latest = pivot.groupby("Community Area Name").tail(1).copy()
                sc_latest["Predicted"] = sc_model.predict(
                    sc_latest[sc_lag_cols].fillna(0)
                ).round(2)
                sc_latest["Actual"] = sc_latest[selected_crime]

                fig_sc2 = px.scatter(
                    sc_latest,
                    x="Actual",
                    y="Predicted",
                    color="Community Area Name",
                    hover_name="Community Area Name",
                    hover_data={"Actual": True, "Predicted": True},
                    labels={
                        "Actual": f"Actual {selected_crime} Count",
                        "Predicted": f"Predicted {selected_crime} Count",
                    },
                    title=f"Actual vs Predicted {selected_crime}",
                )
                # Add a 45-degree reference line
                max_val_sc = max(
                    sc_latest["Actual"].max(),
                    sc_latest["Predicted"].max(),
                    1,
                )
                fig_sc2.add_shape(
                    type="line",
                    x0=0, y0=0,
                    x1=max_val_sc, y1=max_val_sc,
                    line=dict(color="gray", dash="dash"),
                )
                fig_sc2.update_layout(
                    showlegend=False,
                    margin={"t": 40, "b": 0},
                )
                st.plotly_chart(fig_sc2, use_container_width=True)
            else:
                st.info("Not enough data to produce an actual vs predicted scatterplot.")
        else:
            st.info("Lag features not available for the selected crime type; cannot produce predicted scatterplot.")

    # ── Moran's I Spatial Autocorrelation ────────────────────────────────────
    st.divider()
    try:
        gdf_ca = _load_community_areas_gdf()

        # Aggregate selected crime by community area
        crime_by_ca = pivot.groupby("Community Area")[selected_crime].sum().reset_index()
        crime_by_ca.columns = ["Community Area", "crime_total"]

        gdf_merged = gdf_ca.merge(
            crime_by_ca,
            left_on="area_num_1",
            right_on="Community Area",
            how="inner",
        )
        gdf_merged["area_num_str"] = gdf_merged["area_num_1"].astype(str)

        if len(gdf_merged) >= 10:
            map_utils.render_moran_analysis(
                gdf=gdf_merged,
                value_col="crime_total",
                name_col="community",
                id_col="area_num_str",
                geojson=chicago_geo,
                featureidkey="properties.area_num_1",
                key_prefix="safety_moran",
                mapbox_style=mapbox_style,
            )
        else:
            st.warning("Not enough community areas with data to compute Moran's I (need at least 10).")
    except Exception as exc:
        st.warning(f"Could not compute Moran's I: {exc}")

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