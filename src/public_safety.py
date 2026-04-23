import io
import os
<<<<<<< Updated upstream
=======
import json
from datetime import datetime
>>>>>>> Stashed changes
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
from city_config import CityConfig


@st.cache_resource(show_spinner="Training crime prediction model...")
def _train_crime_model(X_json: str, y_json: str):
    X = pd.read_json(io.StringIO(X_json))
    y = pd.read_json(io.StringIO(y_json), typ="series")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


@st.cache_data
def _load_crime_data(city_key: str, crime_path: str, area_map: dict, area_col: str):
    from city_config import get_city
    city = get_city(city_key)
    pivot = pd.read_csv(crime_path)
    pivot['_area_key'] = pivot[area_col].map(city.normalize_area_key)
    pivot['Community Area Name'] = pivot['_area_key'].map(area_map)
    skip = {area_col, '_area_key', 'Year', 'Month', 'Community Area Name'}
    lag_crime_cols = [c for c in pivot.columns if c not in skip]
    for crime in lag_crime_cols:
        grp = pivot.groupby(area_col)[crime]
        pivot[f'{crime}_lag1'] = grp.shift(1)
        pivot[f'{crime}_lag3'] = grp.shift(3)
        pivot[f'{crime}_lag12'] = grp.shift(12)
        pivot[f'{crime}_rolling3'] = grp.transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
    return pivot


@st.cache_data(show_spinner="Loading community geometries...")
def _load_geo_gdf(city_key: str, geo_json_str: str, id_field: str, id_kind: str = "int"):
    """Build GeoDataFrame from a serialized geojson dict (cache-friendly)."""
    gdf = gpd.read_file(io.StringIO(geo_json_str), driver="GeoJSON")
    if id_field in gdf.columns:
        if id_kind == "int":
            gdf[id_field] = pd.to_numeric(gdf[id_field], errors="coerce").astype("Int64")
        elif id_kind == "upper_str":
            gdf[id_field] = gdf[id_field].astype(str).str.strip().str.upper()
        else:
            gdf[id_field] = gdf[id_field].astype(str).str.strip()
    return gdf


def render(city: CityConfig, geo: dict, area_map: dict):
    st.header(f"Public Safety Dashboard — {city.name}")
    st.markdown(f"Crime trends and forecasts across {city.name}.")

<<<<<<< Updated upstream
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
=======
    crime_path = city.crime_path
    if not os.path.exists(crime_path):
        st.warning(f"No crime CSV at `{crime_path}`. Run data refresh.")
>>>>>>> Stashed changes
        return

    area_col = city.crime_area_col
    pivot = _load_crime_data(city.key, crime_path, area_map, area_col)

    skip = {area_col, '_area_key', 'Year', 'Month', 'Community Area Name'}
    crime_cols = [
        c for c in pivot.columns
        if c not in skip
        and not c.endswith(('_lag1', '_lag3', '_lag12', '_rolling3'))
    ]
    community_areas = sorted(pivot['Community Area Name'].dropna().unique())
    if not community_areas:
        st.warning(
            f"Crime CSV's `{area_col}` values do not match boundary IDs "
            f"(`{city.boundary_id_field}`). Choropleth will be empty until "
            "the area mapping is wired up for this city."
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        selected_area = st.selectbox("Select Area", community_areas, key=f"safety_area_{city.key}")
    with col2:
        selected_crime = st.selectbox("Select Crime Type", crime_cols, key=f"safety_crime_{city.key}")

    mapbox_style = map_utils.mapbox_style_picker(key_prefix=f"safety_{city.key}")

    area_data = pivot[pivot['Community Area Name'] == selected_area].sort_values(['Year', 'Month'])

    st.subheader(f"Historical {selected_crime} counts — {selected_area}")
    st.line_chart(area_data[selected_crime].values)

    latest_val = mean_val = 0.0
    if not area_data[selected_crime].dropna().empty:
        series = area_data[selected_crime].dropna()
        latest_val = series.iloc[-1]
        mean_val = series.mean()
        max_val = series.max()
        pct_vs_mean = ((latest_val - mean_val) / mean_val * 100) if mean_val else 0
        direction = "above" if pct_vs_mean > 0 else "below"
        trend_color = "🔴" if pct_vs_mean > 10 else ("🟡" if pct_vs_mean > 0 else "🟢")
        st.info(
            f"{trend_color} **{selected_area} - {selected_crime} trend:** "
            f"Most recent month recorded **{latest_val:.0f} incidents** — "
            f"**{abs(pct_vs_mean):.1f}% {direction}** the historical average "
            f"({mean_val:.1f}). All-time peak: **{max_val:.0f}**."
        )

    crime_upper = selected_crime.upper()
    lag1, lag3, lag12, roll3 = (
        f'{crime_upper}_lag1', f'{crime_upper}_lag3',
        f'{crime_upper}_lag12', f'{crime_upper}_rolling3',
    )
    candidate_features = [lag1, lag3, lag12, roll3, 'Month', 'Year']
    feature_cols = [c for c in candidate_features if c in area_data.columns]
    model_data = area_data.dropna(subset=feature_cols + [selected_crime])

    if len(model_data) >= 10:
        X = model_data[feature_cols].values
        y = model_data[selected_crime].values

        n_splits = min(5, max(2, len(X) // 6))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        candidates = {
            "Ridge Regression": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "Random Forest":    lambda: RandomForestRegressor(n_estimators=100, random_state=42),
        }

        best_name, best_r2, best_rmse = None, -np.inf, np.inf
        for name, make_model in candidates.items():
            r2s, rmses = [], []
            for tr, te in tscv.split(X):
                m = make_model()
                m.fit(X[tr], y[tr])
                p = m.predict(X[te])
                r2s.append(r2_score(y[te], p))
                rmses.append(np.sqrt(mean_squared_error(y[te], p)))
            avg = float(np.mean(r2s))
            if avg > best_r2:
                best_name, best_r2, best_rmse = name, avg, float(np.mean(rmses))

        r2, rmse = best_r2, best_rmse
        model = candidates[best_name]()
        model.fit(X, y)
        prediction = max(0, model.predict(X[-1].reshape(1, -1))[0])

<<<<<<< Updated upstream
        st.subheader(f"Predicted {selected_crime} counts for next month")
        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Forecast", round(prediction))
        col_f2.metric("CV RMSE", f"{rmse:.2f}")
        col_f3.metric("CV R²", f"{r2:.3f}")
        st.caption(f"Best model: **{best_name}** (selected via {n_splits}-fold time-series CV)")

        # ── Forecast interpretation ───────────────────────────────────────────
=======
        _today = datetime.now()
        _nm = _today.replace(month=_today.month % 12 + 1,
                             year=_today.year + (_today.month // 12))
        next_month_label = _nm.strftime("%B %Y")
        trend_vs_latest = prediction - latest_val
        arrow = "▲" if trend_vs_latest >= 0 else "▼"
        chg_dir = "increase" if trend_vs_latest >= 0 else "decrease"

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

        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast", f"{round(prediction):,}")
        c2.metric("CV RMSE", f"±{rmse:.1f}")
        c3.metric("CV R²", f"{r2:.3f}")
        st.caption(f"Model: **{best_name}** · {n_splits}-fold time-series CV")

>>>>>>> Stashed changes
        r2_label = (
            "strong" if r2 >= 0.6 else
            "moderate" if r2 >= 0.3 else
            "weak" if r2 >= 0.0 else
            "poor (worse than baseline)"
        )
        trend_vs_latest = prediction - latest_val if not area_data[selected_crime].dropna().empty else 0
        chg_dir = "increase" if trend_vs_latest > 0 else "decrease"
        st.info(
<<<<<<< Updated upstream
            f"**Forecast interpretation:** The model predicts **{round(prediction)} {selected_crime} incidents** "
            f"next month in {selected_area}, a **{abs(trend_vs_latest):.0f}-incident {chg_dir}** from last month. "
            f"Cross-validated RMSE of {rmse:.2f} means predictions are typically off by ±{rmse:.1f} incidents. "
            f"CV R² of {r2:.3f} indicates a **{r2_label}** fit"
            + (" — the model captures temporal patterns well and forecasts are reliable."
               if r2 >= 0.6
               else (" — the model explains some variance; treat the forecast as a useful estimate."
                     if r2 >= 0.3
                     else (" — the model has limited explanatory power; treat the forecast as directional only."
                           if r2 >= 0.0
                           else " — the model performs worse than a simple average. Consider the forecast unreliable.")))
=======
            f"**Forecast for {next_month_label}:** "
            f"**{round(prediction):,} {selected_crime}** in {selected_area} — "
            f"**{abs(trend_vs_latest):.0f}-incident {chg_dir}** from last month. "
            f"Typical error ±{rmse:.1f}. CV R² **{r2:.3f}** ({r2_label})."
>>>>>>> Stashed changes
        )
    else:
        st.info("Not enough data to generate a prediction.")

    # ── Choropleth maps ──────────────────────────────────────────────────
    st.subheader("Crime Distribution Map")
    feature_id_key = f"properties.{city.boundary_id_field}"
    col_map1, col_map2 = st.columns(2)

    with col_map1:
        crime_map_type = st.selectbox(
            "Crime type (historical)", crime_cols,
            key=f"crime_map_select_{city.key}",
        )
        map_data = pivot.groupby(['_area_key', 'Community Area Name'])[crime_map_type].sum().reset_index()
        map_data['_area_key'] = map_data['_area_key'].astype(str)
        fig = px.choropleth_map(
            map_data, geojson=geo,
            locations='_area_key', featureidkey=feature_id_key,
            color=crime_map_type, color_continuous_scale="Reds",
            map_style=mapbox_style, zoom=city.zoom,
            center={"lat": city.center[0], "lon": city.center[1]}, opacity=0.5,
            labels={crime_map_type: "Crime Count"},
            hover_name='Community Area Name',
        )
        fig.update_coloraxes(colorbar_tickformat='.2f')
        st.plotly_chart(fig, width="stretch")

    with col_map2:
        crime_pred_type = st.selectbox(
            "Crime type (predicted)", crime_cols,
            key=f"crime_map_pred_select_{city.key}",
        )
        pred_upper = crime_pred_type.upper()
        lag_cols = [f'{pred_upper}_lag1', f'{pred_upper}_lag3']
        if any(c not in pivot.columns for c in lag_cols):
            st.warning(f"No lag features for '{crime_pred_type}'.")
        else:
            pred_data = pivot.dropna(subset=lag_cols + [pred_upper])
            if pred_data.empty:
                st.warning(f"Not enough data for '{crime_pred_type}'.")
            else:
                pred_model = _train_crime_model(
                    pred_data[lag_cols].to_json(),
                    pred_data[pred_upper].to_json(),
                )
                latest_month = pivot.groupby('_area_key').tail(1).copy()
                latest_month['Predicted'] = pred_model.predict(
                    latest_month[lag_cols].fillna(0)
                ).round(2)
                latest_month['_area_key'] = latest_month['_area_key'].astype(str)
                fig_pred = px.choropleth_map(
                    latest_month[['_area_key', 'Community Area Name', 'Predicted']],
                    geojson=geo,
                    locations='_area_key', featureidkey=feature_id_key,
                    color='Predicted', color_continuous_scale="Reds",
                    map_style=mapbox_style, zoom=city.zoom,
                    center={"lat": city.center[0], "lon": city.center[1]}, opacity=0.5,
                    labels={'Predicted': f'Predicted {crime_pred_type}'},
                    hover_name='Community Area Name',
                )
                fig_pred.update_coloraxes(colorbar_tickformat='.2f')
                st.plotly_chart(fig_pred, width="stretch")

                top_pred = latest_month.nlargest(3, "Predicted")[["Community Area Name", "Predicted"]]
                top_names = ", ".join(
                    f"{r['Community Area Name']} ({r['Predicted']:.0f})"
                    for _, r in top_pred.iterrows()
                )
                st.info(f"**Predicted hotspots next month:** {top_names}.")

    # ── Scatter ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Crime Scatterplots")

    scatter_agg = pivot.groupby(area_col)[selected_crime].sum().reset_index()
    scatter_agg.columns = [area_col, "Total Crime Count"]
    scatter_agg["_area_key"] = scatter_agg[area_col].map(city.normalize_area_key)
    scatter_agg["Community Area Name"] = scatter_agg["_area_key"].map(area_map)

    sc1, sc2 = st.columns(2)
    with sc1:
        fig_sc1 = px.scatter(
            scatter_agg, x=area_col, y="Total Crime Count",
            color="Community Area Name", hover_name="Community Area Name",
            labels={area_col: "Area ID", "Total Crime Count": f"Total {selected_crime}"},
            title=f"Area vs Total {selected_crime}",
        )
        fig_sc1.update_layout(showlegend=False, margin={"t": 40, "b": 0})
        st.plotly_chart(fig_sc1, width="stretch")

    with sc2:
        sc_lag = [f"{crime_upper}_lag1", f"{crime_upper}_lag3"]
        if not all(c in pivot.columns for c in sc_lag):
            st.info("Lag features unavailable.")
        else:
            sc_pred_data = pivot.dropna(subset=sc_lag + [crime_upper])
            if len(sc_pred_data) < 6:
                st.info("Not enough data.")
            else:
                sc_model = _train_crime_model(
                    sc_pred_data[sc_lag].to_json(),
                    sc_pred_data[crime_upper].to_json(),
                )
                sc_latest = pivot.groupby("Community Area Name").tail(1).copy()
                sc_latest["Predicted"] = sc_model.predict(sc_latest[sc_lag].fillna(0)).round(2)
                sc_latest["Actual"] = sc_latest[selected_crime]

                fig_sc2 = px.scatter(
                    sc_latest, x="Actual", y="Predicted",
                    color="Community Area Name", hover_name="Community Area Name",
                    labels={"Actual": f"Actual {selected_crime}",
                            "Predicted": f"Predicted {selected_crime}"},
                    title=f"Actual vs Predicted {selected_crime}",
                )
                max_v = max(sc_latest["Actual"].max(), sc_latest["Predicted"].max(), 1)
                fig_sc2.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v,
                                  line=dict(color="gray", dash="dash"))
                fig_sc2.update_layout(showlegend=False, margin={"t": 40, "b": 0})
                st.plotly_chart(fig_sc2, width="stretch")

    # ── Moran's I ───────────────────────────────────────────────────────
    st.divider()
    try:
        gdf_ca = _load_geo_gdf(city.key, json.dumps(geo), city.boundary_id_field, city.area_id_kind)
        crime_by_ca = pivot.groupby('_area_key')[selected_crime].sum().reset_index()
        crime_by_ca.columns = ['_area_key', "crime_total"]
        gdf_merged = gdf_ca.merge(
            crime_by_ca,
            left_on=city.boundary_id_field, right_on='_area_key',
            how="inner",
        )
        gdf_merged["_id_str"] = gdf_merged[city.boundary_id_field].astype(str)
        if len(gdf_merged) >= 10:
            map_utils.render_moran_analysis(
                gdf=gdf_merged,
                value_col="crime_total",
                name_col=city.boundary_name_field,
                id_col="_id_str",
                geojson=geo,
                featureidkey=f"properties.{city.boundary_id_field}",
                key_prefix=f"safety_moran_{city.key}",
                map_style=mapbox_style,
            )
        else:
            st.warning(f"Need ≥10 areas for Moran's I (have {len(gdf_merged)}).")
    except Exception as exc:
        st.warning(f"Moran's I unavailable: {exc}")

    # ── ML predictor ────────────────────────────────────────────────────
    base_cols = [c for c in pivot.columns if c not in skip
                 and not c.endswith(('_lag1', '_lag3', '_lag12', '_rolling3'))]
    lag_cols = [c for c in pivot.columns if c.endswith(('_lag1', '_lag3', '_lag12', '_rolling3'))]
    ml_predictor.render_predictor(
        pivot.dropna(subset=lag_cols[:2] if lag_cols else base_cols[:1]),
        key_prefix=f"safety_{city.key}",
        default_target=selected_crime,
        default_features=lag_cols[:6] + [area_col, 'Year', 'Month'],
    )
