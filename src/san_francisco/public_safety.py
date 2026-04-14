import io
import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import geopandas as gpd
import file_loader
import ml_predictor
import map_utils

# ── City-specific constants ───────────────────────────────────────────────────
CITY_NAME      = "San Francisco"
GEO_URL        = "https://data.sfgov.org/api/geospatial/p5b7-5n3h?method=export&type=GeoJSON"
GEO_ID_FIELD   = "nhood"
GEO_NAME_FIELD = "nhood"
MAP_CENTER     = {"lat": 37.7749, "lon": -122.4194}
MAP_ZOOM       = 11
REGION_LABEL   = "Neighborhood"
# ─────────────────────────────────────────────────────────────────────────────

CRIME_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crime_monthly_pivot.csv")


@st.cache_resource(show_spinner="Training crime prediction model...")
def _train_crime_model(X_json: str, y_json: str):
    X = pd.read_json(io.StringIO(X_json))
    y = pd.read_json(io.StringIO(y_json), typ="series")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


@st.cache_data
def _load_crime_data(area_map):
    pivot = pd.read_csv(CRIME_CSV)
    pivot["Community Area"] = pivot["Community Area"].astype(str).str.strip()
    pivot["Community Area Name"] = pivot["Community Area"].map(area_map).fillna(pivot["Community Area"])
    lag_crime_cols = [c for c in pivot.columns if c not in ["Community Area", "Year", "Month", "Community Area Name"]]
    for crime in lag_crime_cols:
        grp = pivot.groupby("Community Area")[crime]
        pivot[f"{crime}_lag1"]    = grp.shift(1)
        pivot[f"{crime}_lag3"]    = grp.shift(3)
        pivot[f"{crime}_lag12"]   = grp.shift(12)
        pivot[f"{crime}_rolling3"] = grp.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    return pivot


@st.cache_data(show_spinner="Loading area geometries...")
def _load_area_gdf():
    return gpd.read_file(GEO_URL)


def render(chicago_geo=None, area_map=None):
    st.header(f"Public Safety Dashboard — {CITY_NAME}")
    st.markdown(f"Analyze and forecast crime trends across {CITY_NAME} {REGION_LABEL}s.")

    with st.expander("Upload a supplemental dataset"):
        file_loader.uploader(domain="public_safety", local_csv=None, label="Upload a public safety dataset")

    if not os.path.exists(CRIME_CSV):
        st.info(f"No local crime data found. Fetching the latest data from {CITY_NAME} Open Data…")
        try:
            import data_fetcher
            data_fetcher.fetch_crimes(force=True)
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Auto-fetch failed: {exc}")
        return

    if area_map is None:
        area_map = {}
    pivot = _load_crime_data(area_map)

    crime_cols = [
        c for c in pivot.columns
        if c not in ["Community Area", "Year", "Month", "Community Area Name"]
        and not c.endswith(("_lag1", "_lag3", "_lag12", "_rolling3"))
    ]
    areas = sorted(pivot["Community Area Name"].dropna().unique())

    col1, col2 = st.columns(2)
    with col1:
        selected_area = st.selectbox(f"Select {REGION_LABEL}", areas, key="safety_area")
    with col2:
        selected_crime = st.selectbox("Select Crime Type", crime_cols, key="safety_crime")

    mapbox_style = map_utils.mapbox_style_picker(key_prefix="safety")

    area_data = pivot[pivot["Community Area Name"] == selected_area].sort_values(["Year", "Month"])
    st.subheader(f"Historical {selected_crime} counts — {selected_area}")
    st.line_chart(area_data[selected_crime].values)

    if not area_data[selected_crime].dropna().empty:
        series     = area_data[selected_crime].dropna()
        latest_val = series.iloc[-1]
        mean_val   = series.mean()
        max_val    = series.max()
        pct_vs_mean = ((latest_val - mean_val) / mean_val * 100) if mean_val else 0
        direction   = "above" if pct_vs_mean > 0 else "below"
        trend_color = "🔴" if pct_vs_mean > 10 else ("🟡" if pct_vs_mean > 0 else "🟢")
        st.info(
            f"{trend_color} **{selected_area} – {selected_crime}:** "
            f"Most recent month: **{latest_val:.0f} incidents** — "
            f"**{abs(pct_vs_mean):.1f}% {direction} average** ({mean_val:.1f}). "
            f"Peak: **{max_val:.0f}**."
        )

    # ── Forecast ──────────────────────────────────────────────────────────────
    crime_upper  = selected_crime.upper()
    feature_cols = [c for c in [f"{crime_upper}_lag1", f"{crime_upper}_lag3",
                                 f"{crime_upper}_lag12", f"{crime_upper}_rolling3",
                                 "Month", "Year"] if c in area_data.columns]
    model_data = area_data.dropna(subset=feature_cols + [selected_crime])

    if len(model_data) >= 10:
        X, y     = model_data[feature_cols].values, model_data[selected_crime].values
        n_splits = min(5, max(2, len(X) // 6))
        tscv     = TimeSeriesSplit(n_splits=n_splits)
        candidates = {
            "Ridge Regression": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "Random Forest":    lambda: RandomForestRegressor(n_estimators=100, random_state=42),
        }
        best_name, best_r2, best_rmse = None, -np.inf, np.inf
        for name, make_model in candidates.items():
            fold_r2s, fold_rmses = [], []
            for tr_idx, te_idx in tscv.split(X):
                m = make_model(); m.fit(X[tr_idx], y[tr_idx]); p = m.predict(X[te_idx])
                fold_r2s.append(r2_score(y[te_idx], p))
                fold_rmses.append(np.sqrt(mean_squared_error(y[te_idx], p)))
            if np.mean(fold_r2s) > best_r2:
                best_name  = name
                best_r2    = float(np.mean(fold_r2s))
                best_rmse  = float(np.mean(fold_rmses))

        model      = candidates[best_name](); model.fit(X, y)
        prediction = max(0, model.predict(X[-1].reshape(1, -1))[0])

        st.subheader(f"Predicted {selected_crime} counts for next month")
        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast", round(prediction))
        c2.metric("CV RMSE",  f"{best_rmse:.2f}")
        c3.metric("CV R²",    f"{best_r2:.3f}")
        st.caption(f"Best model: **{best_name}** ({n_splits}-fold time-series CV)")
    else:
        st.info("Not enough data to generate a prediction.")
        latest_val = 0

    # ── Crime maps ────────────────────────────────────────────────────────────
    st.subheader(f"Crime Distribution by {REGION_LABEL}")
    col_map1, col_map2 = st.columns(2)

    with col_map1:
        crime_map_type = st.selectbox("Crime type (historical map)", crime_cols, key="crime_map_select")
        map_data = pivot.groupby("Community Area")[crime_map_type].sum().reset_index()
        try:
            gdf = _load_area_gdf()
            # Align types for merge
            map_data["Community Area"] = map_data["Community Area"].astype(str)
            gdf[GEO_ID_FIELD]          = gdf[GEO_ID_FIELD].astype(str)
            city_geo = gdf.__geo_interface__
            fig = px.choropleth_map(
                map_data, geojson=city_geo,
                locations="Community Area", featureidkey=f"properties.{GEO_ID_FIELD}",
                color=crime_map_type, color_continuous_scale="Reds",
                map_style=mapbox_style, zoom=MAP_ZOOM, center=MAP_CENTER, opacity=0.5,
                labels={crime_map_type: "Crime Count"},
            )
            fig.update_coloraxes(colorbar_tickformat=".2f")
            st.plotly_chart(fig, width="stretch")
        except Exception as exc:
            st.warning(f"Map unavailable: {exc}")
            st.bar_chart(map_data.set_index("Community Area")[crime_map_type])

    with col_map2:
        crime_pred_type = st.selectbox("Crime type (predicted map)", crime_cols, key="crime_map_pred_select")
        ptu = crime_pred_type.upper()
        lag_cols = [f"{ptu}_lag1", f"{ptu}_lag3"]
        if any(c not in pivot.columns for c in lag_cols):
            st.warning(f"No lag features found for '{crime_pred_type}'.")
        else:
            pred_data = pivot.dropna(subset=lag_cols + [ptu])
            if not pred_data.empty:
                pred_model = _train_crime_model(pred_data[lag_cols].to_json(), pred_data[ptu].to_json())
                latest_month = pivot.groupby("Community Area").tail(1).copy()
                latest_month["Predicted"] = pred_model.predict(latest_month[lag_cols].fillna(0)).round(2)
                try:
                    gdf = _load_area_gdf()
                    latest_month["Community Area"] = latest_month["Community Area"].astype(str)
                    gdf[GEO_ID_FIELD]              = gdf[GEO_ID_FIELD].astype(str)
                    city_geo = gdf.__geo_interface__
                    fig_pred = px.choropleth_map(
                        latest_month[["Community Area", "Predicted"]],
                        geojson=city_geo,
                        locations="Community Area", featureidkey=f"properties.{GEO_ID_FIELD}",
                        color="Predicted", color_continuous_scale="Reds",
                        map_style=mapbox_style, zoom=MAP_ZOOM, center=MAP_CENTER, opacity=0.5,
                        labels={"Predicted": f"Predicted {crime_pred_type}"},
                    )
                    fig_pred.update_coloraxes(colorbar_tickformat=".2f")
                    st.plotly_chart(fig_pred, width="stretch")
                except Exception as exc:
                    st.warning(f"Map unavailable: {exc}")
                    st.bar_chart(latest_month.set_index("Community Area")["Predicted"])

                top3 = latest_month.nlargest(3, "Predicted")[["Community Area Name", "Predicted"]]
                top_names = ", ".join(f"{r['Community Area Name']} ({r['Predicted']:.0f})" for _, r in top3.iterrows())
                st.info(f"**Predicted hotspots next month:** {top_names}")

    # ── Scatterplots ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Crime Scatterplots")
    scatter_agg = pivot.groupby("Community Area")[selected_crime].sum().reset_index()
    scatter_agg.columns = ["Community Area", "Total Crime Count"]
    scatter_agg["Community Area Name"] = scatter_agg["Community Area"].map(area_map).fillna(scatter_agg["Community Area"])

    sc1, sc2 = st.columns(2)
    with sc1:
        fig_sc1 = px.scatter(
            scatter_agg, x="Community Area", y="Total Crime Count",
            hover_name="Community Area Name",
            title=f"{REGION_LABEL} vs Total {selected_crime}",
        )
        fig_sc1.update_layout(showlegend=False, margin={"t": 40, "b": 0})
        st.plotly_chart(fig_sc1, width="stretch")

    with sc2:
        sc_lag_cols = [f"{crime_upper}_lag1", f"{crime_upper}_lag3"]
        if all(c in pivot.columns for c in sc_lag_cols):
            sc_pred_data = pivot.dropna(subset=sc_lag_cols + [crime_upper])
            if len(sc_pred_data) >= 6:
                sc_model  = _train_crime_model(sc_pred_data[sc_lag_cols].to_json(), sc_pred_data[crime_upper].to_json())
                sc_latest = pivot.groupby("Community Area Name").tail(1).copy()
                sc_latest["Predicted"] = sc_model.predict(sc_latest[sc_lag_cols].fillna(0)).round(2)
                sc_latest["Actual"]    = sc_latest[selected_crime]
                fig_sc2 = px.scatter(
                    sc_latest, x="Actual", y="Predicted",
                    hover_name="Community Area Name",
                    title=f"Actual vs Predicted {selected_crime}",
                )
                max_v = max(sc_latest["Actual"].max(), sc_latest["Predicted"].max(), 1)
                fig_sc2.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v, line=dict(color="gray", dash="dash"))
                fig_sc2.update_layout(showlegend=False, margin={"t": 40, "b": 0})
                st.plotly_chart(fig_sc2, width="stretch")

    # ── Moran's I ─────────────────────────────────────────────────────────────
    st.divider()
    try:
        gdf_area = _load_area_gdf()
        gdf_area[GEO_ID_FIELD] = gdf_area[GEO_ID_FIELD].astype(str)
        crime_by_area = pivot.groupby("Community Area")[selected_crime].sum().reset_index()
        crime_by_area.columns = ["Community Area", "crime_total"]
        crime_by_area["Community Area"] = crime_by_area["Community Area"].astype(str)
        gdf_merged = gdf_area.merge(crime_by_area, left_on=GEO_ID_FIELD, right_on="Community Area", how="inner")
        if len(gdf_merged) >= 10:
            city_geo_dict = _load_area_gdf().__geo_interface__
            map_utils.render_moran_analysis(
                gdf=gdf_merged,
                value_col="crime_total",
                name_col=GEO_NAME_FIELD,
                id_col=GEO_ID_FIELD,
                geojson=city_geo_dict,
                featureidkey=f"properties.{GEO_ID_FIELD}",
                key_prefix="safety_moran",
                map_style=mapbox_style,
            )
        else:
            st.warning("Not enough areas with data to compute Moran's I (need at least 10).")
    except Exception as exc:
        st.warning(f"Could not compute Moran's I: {exc}")

    # ── Generic ML predictor ──────────────────────────────────────────────────
    base_cols = [c for c in pivot.columns if c not in ["Community Area", "Year", "Month", "Community Area Name"]
                 and not c.endswith(("_lag1", "_lag3", "_lag12", "_rolling3"))]
    lag_all   = [c for c in pivot.columns if c.endswith(("_lag1", "_lag3", "_lag12", "_rolling3"))]
    ml_predictor.render_predictor(
        pivot.dropna(subset=lag_all[:2] if lag_all else base_cols[:1]),
        key_prefix="safety",
        default_target=selected_crime,
        default_features=lag_all[:6] + ["Community Area", "Year", "Month"],
    )
