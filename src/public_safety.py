import os
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import geopandas as gpd
import ml_predictor
import map_utils

CRIME_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crime_monthly_pivot.csv")


@st.cache_resource(show_spinner="Training crime prediction model...")
def _train_crime_model(X_json: str, y_json: str):
    """Train a RandomForest once and cache the fitted model."""
    X = pd.read_json(X_json)
    y = pd.read_json(y_json, typ="series")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


@st.cache_data
def _load_crime_data(area_map):
    pivot = pd.read_csv(CRIME_CSV)
    pivot['Community Area Name'] = pivot['Community Area'].map(area_map)
    lag_crime_cols = [c for c in pivot.columns if c not in ['Community Area', 'Year', 'Month', 'Community Area Name']]
    for crime in lag_crime_cols:
        grp = pivot.groupby('Community Area')[crime]
        pivot[f'{crime}_lag1'] = grp.shift(1)
        pivot[f'{crime}_lag3'] = grp.shift(3)
        pivot[f'{crime}_lag12'] = grp.shift(12)
        pivot[f'{crime}_rolling3'] = grp.transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
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
        and not c.endswith('_lag12')
        and not c.endswith('_rolling3')
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

    # Prediction — lag features + annual seasonality + trend
    crime_upper = selected_crime.upper()
    lag1_col     = f'{crime_upper}_lag1'
    lag3_col     = f'{crime_upper}_lag3'
    lag12_col    = f'{crime_upper}_lag12'
    rolling3_col = f'{crime_upper}_rolling3'
    candidate_features = [lag1_col, lag3_col, lag12_col, rolling3_col, 'Month', 'Year']
    feature_cols = [c for c in candidate_features if c in area_data.columns]
    model_data = area_data.dropna(subset=feature_cols + [selected_crime])

    if len(model_data) >= 10:
        X = model_data[feature_cols].values
        y = model_data[selected_crime].values

        # ── Model selection via time-series CV ────────────────────────────────
        # Ridge can follow trends (extrapolate); RF cannot — try both, keep best.
        n_splits = min(5, max(2, len(X) // 6))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        candidates = {
            "Ridge Regression": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "Random Forest":    lambda: RandomForestRegressor(n_estimators=100, random_state=42),
        }

        best_name, best_r2, best_rmse = None, -np.inf, np.inf
        for name, make_model in candidates.items():
            fold_r2s, fold_rmses = [], []
            for tr_idx, te_idx in tscv.split(X):
                m = make_model()
                m.fit(X[tr_idx], y[tr_idx])
                p = m.predict(X[te_idx])
                fold_r2s.append(r2_score(y[te_idx], p))
                fold_rmses.append(np.sqrt(mean_squared_error(y[te_idx], p)))
            avg_r2 = float(np.mean(fold_r2s))
            if avg_r2 > best_r2:
                best_name = name
                best_r2   = avg_r2
                best_rmse = float(np.mean(fold_rmses))

        r2   = best_r2
        rmse = best_rmse

        # Train winning model on ALL available data for the actual forecast
        model = candidates[best_name]()
        model.fit(X, y)
        prediction = max(0, model.predict(X[-1].reshape(1, -1))[0])

        # Compute the actual next-month label (e.g., "May 2026")
        _today = datetime.now()
        _nm    = _today.replace(month=_today.month % 12 + 1,
                                year=_today.year + (_today.month // 12))
        next_month_label = _nm.strftime("%B %Y")

        trend_vs_latest = prediction - latest_val if not area_data[selected_crime].dropna().empty else 0
        arrow   = "▲" if trend_vs_latest >= 0 else "▼"
        chg_dir = "increase" if trend_vs_latest >= 0 else "decrease"

        # ── Prominent headline ────────────────────────────────────────────────
        st.markdown(f"""
<div style="background:rgba(224,80,80,0.1); border-left:4px solid #e05050;
            padding:18px 22px; border-radius:8px; margin:14px 0;">
  <p style="margin:0; font-size:11px; color:#9eaec4; text-transform:uppercase;
            letter-spacing:0.08em;">Crime Forecast — {next_month_label}</p>
  <p style="margin:6px 0 2px; font-size:2.4rem; font-weight:800;
            color:#ffffff; line-height:1.1;">
    {round(prediction):,}
    <span style="font-size:1.1rem; font-weight:500; color:#e08080;">
      &nbsp;{selected_crime}
    </span>
  </p>
  <p style="margin:2px 0 0; font-size:14px; color:#9eaec4;">
    in <strong style="color:#ffffff;">{selected_area}</strong>
    &nbsp;·&nbsp; {arrow} {abs(trend_vs_latest):.0f} incidents from last month
  </p>
</div>
""", unsafe_allow_html=True)

        # ── Supporting metrics ────────────────────────────────────────────────
        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Forecast", f"{round(prediction):,}")
        col_f2.metric("CV RMSE", f"±{rmse:.1f}")
        col_f3.metric("CV R²", f"{r2:.3f}")
        st.caption(f"Model: **{best_name}** · {n_splits}-fold time-series CV")

        # ── Interpretation ────────────────────────────────────────────────────
        r2_label = (
            "strong" if r2 >= 0.6 else
            "moderate" if r2 >= 0.3 else
            "weak" if r2 >= 0.0 else
            "poor (worse than baseline)"
        )
        st.info(
            f"**What this means:** In **{next_month_label}**, the model expects "
            f"**{round(prediction):,} {selected_crime} incidents** in {selected_area} — "
            f"a **{abs(trend_vs_latest):.0f}-incident {chg_dir}** from last month. "
            f"Typical prediction error: ±{rmse:.1f} incidents (CV RMSE). "
            f"Model fit (CV R²): **{r2:.3f}** — {r2_label}."
        )
    else:
        st.info("Not enough data to generate a prediction.")

    # Crime Map
    st.subheader("Crime Distribution Map")
    col_map1, col_map2 = st.columns(2)

    with col_map1:
        crime_map_type = st.selectbox("Crime type (historical map)", crime_cols, key="crime_map_select")
        map_data = pivot.groupby('Community Area Name')[crime_map_type].sum().reset_index()
        fig = px.choropleth_map(
            map_data, geojson=chicago_geo,
            locations='Community Area Name', featureidkey="properties.community",
            color=crime_map_type, color_continuous_scale="Reds",
            map_style=mapbox_style, zoom=9,
            center={"lat": 41.8781, "lon": -87.6298}, opacity=0.5,
            labels={crime_map_type: "Crime Count"}
        )
        fig.update_coloraxes(colorbar_tickformat='.2f')
        st.plotly_chart(fig, width="stretch")

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
                pred_model = _train_crime_model(
                    pred_data[lag_cols].to_json(),
                    pred_data[crime_pred_type_upper].to_json(),
                )
                latest_month = pivot.groupby('Community Area Name').tail(1).copy()
                latest_month['Predicted'] = pred_model.predict(latest_month[lag_cols].fillna(0)).round(2)
                fig_pred = px.choropleth_map(
                    latest_month[['Community Area Name', 'Predicted']],
                    geojson=chicago_geo,
                    locations='Community Area Name', featureidkey="properties.community",
                    color='Predicted', color_continuous_scale="Reds",
                    map_style=mapbox_style, zoom=9,
                    center={"lat": 41.8781, "lon": -87.6298}, opacity=0.5,
                    labels={'Predicted': f'Predicted {crime_pred_type} Count'}
                )
                fig_pred.update_coloraxes(colorbar_tickformat='.2f')
                st.plotly_chart(fig_pred, width="stretch")

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
        st.plotly_chart(fig_sc1, width="stretch")

    with scatter_col2:
        # Build predicted vs actual scatter if predictions are available
        crime_pred_upper = selected_crime.upper()
        sc_lag_cols = [f"{crime_pred_upper}_lag1", f"{crime_pred_upper}_lag3"]
        sc_missing = [c for c in sc_lag_cols if c not in pivot.columns]

        if not sc_missing:
            sc_pred_data = pivot.dropna(subset=sc_lag_cols + [crime_pred_upper])
            if len(sc_pred_data) >= 6:
                sc_model = _train_crime_model(
                    sc_pred_data[sc_lag_cols].to_json(),
                    sc_pred_data[crime_pred_upper].to_json(),
                )
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
                st.plotly_chart(fig_sc2, width="stretch")
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
                map_style=mapbox_style,
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
        and not c.endswith('_lag12')
        and not c.endswith('_rolling3')
    ]
    lag_cols = [c for c in pivot.columns if c.endswith(('_lag1', '_lag3', '_lag12', '_rolling3'))]
    ml_predictor.render_predictor(
        pivot.dropna(subset=lag_cols[:2] if lag_cols else base_crime_cols[:1]),
        key_prefix="safety",
        default_target=selected_crime,
        default_features=lag_cols[:6] + ['Community Area', 'Year', 'Month'],
    )