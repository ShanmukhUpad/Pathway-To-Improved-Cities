import io
import json
import os
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import geopandas as gpd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import ml_predictor
import map_utils
from city_config import CityConfig


DAY_LABELS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

_DF1_COLS = [
    'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'ROADWAY_SURFACE_COND', 'ROAD_DEFECT',
    'ALIGNMENT', 'TRAFFICWAY_TYPE', 'LANE_CNT', 'POSTED_SPEED_LIMIT',
    'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION', 'INTERSECTION_RELATED_I',
    'CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'CRASH_MONTH', 'FIRST_CRASH_TYPE',
]

_DF2_COLS = [
    'FIRST_CRASH_TYPE', 'CRASH_TYPE', 'WEATHER_CONDITION', 'LIGHTING_CONDITION',
    'ROADWAY_SURFACE_COND', 'POSTED_SPEED_LIMIT', 'TRAFFICWAY_TYPE',
    'INTERSECTION_RELATED_I', 'CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'DAMAGE',
    'NUM_UNITS', 'HIT_AND_RUN_I',
]


def _clean_crash_df(df):
    df = df.copy()
    df['CRASH_DATE'] = pd.to_datetime(df['CRASH_DATE'], errors='coerce')
    df = df.dropna(subset=['CRASH_DATE'])
    df['CRASH_HOUR'] = df['CRASH_DATE'].dt.hour
    df['CRASH_DAY_OF_WEEK'] = df['CRASH_DATE'].dt.dayofweek
    df['CRASH_MONTH'] = df['CRASH_DATE'].dt.month

    df1_cols = [c for c in _DF1_COLS if c in df.columns]
    df1 = df[df1_cols].copy().dropna()
    if 'LANE_CNT' in df1.columns:
        df1['LANE_CNT'] = pd.to_numeric(df1['LANE_CNT'], errors='coerce')
        df1.dropna(subset=['LANE_CNT'], inplace=True)
        df1['LANE_CNT'] = df1['LANE_CNT'].astype(int)

    str_cols = [c for c in [
        'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'ROADWAY_SURFACE_COND',
        'ROAD_DEFECT', 'ALIGNMENT', 'TRAFFICWAY_TYPE',
        'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION', 'FIRST_CRASH_TYPE',
    ] if c in df1.columns]
    for col in str_cols:
        df1 = df1[df1[col].astype(str).str.upper().str.strip() != 'UNKNOWN']
        df1 = df1[df1[col].astype(str).str.strip() != '']

    if 'INTERSECTION_RELATED_I' in df1.columns:
        df1 = df1[df1['INTERSECTION_RELATED_I'].astype(str).str.upper().str.strip().isin(['Y', 'N'])]
    if 'POSTED_SPEED_LIMIT' in df1.columns:
        df1 = df1[(df1['POSTED_SPEED_LIMIT'] > 0) & (df1['POSTED_SPEED_LIMIT'] <= 100)]
    if 'LANE_CNT' in df1.columns:
        df1 = df1[(df1['LANE_CNT'] > 0) & (df1['LANE_CNT'] <= 20)]
    df1 = df1[df1['CRASH_HOUR'].between(0, 23)]
    df1 = df1[df1['CRASH_DAY_OF_WEEK'].between(0, 6)]
    df1 = df1[df1['CRASH_MONTH'].between(1, 12)]
    df1.reset_index(drop=True, inplace=True)

    df2_cols = [c for c in _DF2_COLS if c in df.columns]
    df2 = df[df2_cols].copy().dropna()
    str_cols2 = [c for c in [
        'FIRST_CRASH_TYPE', 'CRASH_TYPE', 'WEATHER_CONDITION',
        'LIGHTING_CONDITION', 'ROADWAY_SURFACE_COND', 'TRAFFICWAY_TYPE',
    ] if c in df2.columns]
    for col in str_cols2:
        df2 = df2[df2[col].astype(str).str.upper().str.strip() != 'UNKNOWN']
        df2 = df2[df2[col].astype(str).str.strip() != '']
    if 'INTERSECTION_RELATED_I' in df2.columns:
        df2 = df2[df2['INTERSECTION_RELATED_I'].astype(str).str.upper().str.strip().isin(['Y', 'N'])]
    if 'HIT_AND_RUN_I' in df2.columns:
        df2 = df2[df2['HIT_AND_RUN_I'].astype(str).str.upper().str.strip().isin(['Y', 'N'])]
    if 'DAMAGE' in df2.columns:
        df2 = df2[df2['DAMAGE'].astype(str).str.upper().str.strip().isin(
            ['$500 OR LESS', '$501 - $1,500', 'OVER $1,500'])]
    if 'POSTED_SPEED_LIMIT' in df2.columns:
        df2 = df2[(df2['POSTED_SPEED_LIMIT'] > 0) & (df2['POSTED_SPEED_LIMIT'] <= 100)]
    if 'NUM_UNITS' in df2.columns:
        df2 = df2[(df2['NUM_UNITS'] > 0) & (df2['NUM_UNITS'] <= 50)]
    df2 = df2[df2['CRASH_HOUR'].between(0, 23)]
    df2 = df2[df2['CRASH_DAY_OF_WEEK'].between(0, 6)]
    df2.reset_index(drop=True, inplace=True)

    return df1, df2


def _resolve_crash_path(city: CityConfig) -> str | None:
    """Primary crash CSV, fall back to legacy dataset if present."""
    if os.path.exists(city.crash_path):
        return city.crash_path
    if city.crash_legacy_path and os.path.exists(city.crash_legacy_path):
        return city.crash_legacy_path
    return None


@st.cache_data(show_spinner="Loading crash data...")
def _load_crash_data(city_key: str, path: str):
    df = pd.read_csv(path, low_memory=False)
    return _clean_crash_df(df)


@st.cache_data(show_spinner="Forecasting crash trend...")
def _crash_monthly_forecast(city_key: str, path: str):
    """Build city-wide monthly crash count time series + forecast next month."""
    raw = pd.read_csv(path, usecols=["CRASH_DATE"], low_memory=False)
    raw["CRASH_DATE"] = pd.to_datetime(raw["CRASH_DATE"], errors="coerce")
    raw = raw.dropna(subset=["CRASH_DATE"])
    raw["Period"] = raw["CRASH_DATE"].dt.to_period("M")
    monthly = raw.groupby("Period").size().reset_index(name="count")
    monthly = monthly.sort_values("Period").reset_index(drop=True)
    monthly["label"] = monthly["Period"].astype(str)
    if len(monthly) < 6:
        return None

    s = monthly["count"].values.astype(float)
    lag1 = np.concatenate([[np.nan], s[:-1]])
    lag2 = np.concatenate([[np.nan, np.nan], s[:-2]])
    lag3 = np.concatenate([[np.nan, np.nan, np.nan], s[:-3]])
    roll3 = pd.Series(s).shift(1).rolling(3, min_periods=1).mean().values
    df_f = pd.DataFrame({"count": s, "lag1": lag1, "lag2": lag2, "lag3": lag3, "roll3": roll3})
    df_f = df_f.dropna()
    if len(df_f) < 6:
        return None

    X = df_f[["lag1", "lag2", "lag3", "roll3"]].values
    y = df_f["count"].values

    n_splits = min(4, max(2, len(X) // 4))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    candidates = {
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    }
    best_name, best_r2, best_rmse = None, -np.inf, np.inf
    for name, mk in candidates.items():
        r2s, rmses = [], []
        for tr, te in tscv.split(X):
            m = mk(); m.fit(X[tr], y[tr]); p = m.predict(X[te])
            r2s.append(r2_score(y[te], p))
            rmses.append(np.sqrt(mean_squared_error(y[te], p)))
        avg = float(np.clip(np.mean(r2s), -1, 1))
        if avg > best_r2:
            best_name, best_r2, best_rmse = name, avg, float(np.mean(rmses))

    model = candidates[best_name]()
    model.fit(X, y)
    x_next = np.array([[s[-1], s[-2], s[-3], np.mean(s[-3:])]])
    prediction = max(0.0, float(model.predict(x_next)[0]))
    latest = float(s[-1])

    _today = datetime.now()
    _nm = _today.replace(month=_today.month % 12 + 1,
                         year=_today.year + (_today.month // 12))
    return {
        "prediction": prediction,
        "latest": latest,
        "next_label": _nm.strftime("%B %Y"),
        "r2": best_r2,
        "rmse": best_rmse,
        "model": best_name,
        "monthly_labels": monthly["label"].tolist(),
        "monthly_counts": monthly["count"].tolist(),
    }


@st.cache_data(show_spinner="Loading geometries for crash join...")
def _load_geo_gdf_crash(city_key: str, geo_json_str: str, id_field: str):
    gdf = gpd.read_file(io.StringIO(geo_json_str), driver="GeoJSON")
    if id_field in gdf.columns:
        try:
            gdf[id_field] = gdf[id_field].astype(int)
        except (TypeError, ValueError):
            gdf[id_field] = gdf[id_field].astype(str)
    return gdf


@st.cache_data(show_spinner="Loading crash coordinates...")
def _load_crash_coords(city_key: str, path: str,
                      lat_bounds: tuple, lon_bounds: tuple):
    raw = pd.read_csv(path, usecols=["LATITUDE", "LONGITUDE"], low_memory=False)
    raw["LATITUDE"] = pd.to_numeric(raw["LATITUDE"], errors="coerce")
    raw["LONGITUDE"] = pd.to_numeric(raw["LONGITUDE"], errors="coerce")
    coords = raw.dropna(subset=["LATITUDE", "LONGITUDE"])
    lat_lo, lat_hi = lat_bounds
    lon_lo, lon_hi = lon_bounds
    coords = coords[
        (coords["LATITUDE"] > lat_lo) & (coords["LATITUDE"] < lat_hi) &
        (coords["LONGITUDE"] > lon_lo) & (coords["LONGITUDE"] < lon_hi)
    ]
    return coords.reset_index(drop=True)


@st.cache_data(show_spinner="Aggregating crashes by area...")
def _crash_counts_by_area(city_key: str, path: str, geo_json_str: str,
                          id_field: str, name_field: str,
                          lat_bounds: tuple, lon_bounds: tuple):
    coords = _load_crash_coords(city_key, path, lat_bounds, lon_bounds)
    if coords.empty:
        return coords, pd.DataFrame()
    gdf_ca = _load_geo_gdf_crash(city_key, geo_json_str, id_field)
    crash_pts = gpd.GeoDataFrame(
        coords,
        geometry=gpd.points_from_xy(coords["LONGITUDE"], coords["LATITUDE"]),
        crs="EPSG:4326",
    )
    keep_cols = [c for c in [id_field, name_field, "geometry"] if c in gdf_ca.columns]
    joined = gpd.sjoin(crash_pts, gdf_ca[keep_cols], how="inner", predicate="within")
    grp_cols = [id_field]
    if name_field in joined.columns and name_field != id_field:
        grp_cols.append(name_field)
    crash_counts = joined.groupby(grp_cols).size().reset_index(name="Crash Count")
    crash_counts[id_field] = crash_counts[id_field].astype(str)
    return coords, crash_counts


def render(city: CityConfig, geo: dict | None = None):
    st.header(f"Crash Analysis — {city.name}")
    st.markdown(f"Traffic crash patterns across {city.name}.")

    mapbox_style = map_utils.mapbox_style_picker(key_prefix=f"crash_{city.key}")

    path = _resolve_crash_path(city)
    if path is None:
        st.warning(f"No crash CSV for {city.name}. Run data refresh.")
        return

    try:
        df1, df2 = _load_crash_data(city.key, path)
    except Exception as exc:
        st.error(f"Crash data load failed: {exc}")
        return

    if df1.empty:
        st.warning("Crash dataset cleaned to zero rows. Schema may differ for this city.")
        return

    # ── Crash forecast card ──────────────────────────────────────────────
    try:
        fc = _crash_monthly_forecast(city.key, path)
        if fc:
            pred   = fc["prediction"]
            latest = fc["latest"]
            delta  = pred - latest
            arrow  = "▲" if delta >= 0 else "▼"
            chg    = "increase" if delta >= 0 else "decrease"
            st.markdown(f"""
<div style="background:rgba(224,80,80,0.1);border-left:4px solid #e05050;
            padding:18px 22px;border-radius:8px;margin:14px 0 20px;">
  <p style="margin:0;font-size:11px;color:#9eaec4;text-transform:uppercase;
            letter-spacing:.08em;">Crash Forecast — {fc['next_label']}</p>
  <p style="margin:6px 0 2px;font-size:2.4rem;font-weight:800;
            color:#ffffff;line-height:1.1;">
    {round(pred):,}
    <span style="font-size:1.1rem;font-weight:500;color:#e08080;">
      &nbsp;CRASHES
    </span>
  </p>
  <p style="margin:2px 0 0;font-size:14px;color:#9eaec4;">
    in <strong style="color:#ffffff;">{city.name}</strong>
    &nbsp;·&nbsp; {arrow} {abs(delta):,.0f} from last month
  </p>
</div>""", unsafe_allow_html=True)
            cf1, cf2, cf3 = st.columns(3)
            cf1.metric("Forecast", f"{round(pred):,}")
            cf2.metric("CV RMSE", f"±{fc['rmse']:.0f}")
            cf3.metric("CV R²",   f"{fc['r2']:.3f}")
            st.caption(f"Model: **{fc['model']}** · time-series CV · city-wide monthly crashes")

            # Monthly trend line
            fig_trend = px.line(
                x=fc["monthly_labels"], y=fc["monthly_counts"],
                labels={"x": "Month", "y": "Crashes"},
                title="Monthly crash trend",
                markers=True,
            )
            fig_trend.update_xaxes(tickangle=-45, nticks=24)
            fig_trend.update_layout(margin={"t": 30, "b": 0}, height=280)
            st.plotly_chart(fig_trend, width="stretch")
            st.divider()
    except Exception as _exc:
        pass  # forecast optional — don't block rest of tab

    # ── Timing ──────────────────────────────────────────────────────────
    st.subheader("Crash Timing")
    col_h, col_d, col_m = st.columns(3)

    with col_h:
        hourly = df1.groupby('CRASH_HOUR').size().reset_index(name='Crashes')
        fig_h = px.bar(hourly, x='CRASH_HOUR', y='Crashes',
                       labels={'CRASH_HOUR': 'Hour', 'Crashes': 'Crashes'},
                       title='Crashes by Hour',
                       color='Crashes', color_continuous_scale='Oranges')
        fig_h.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_h, width="stretch")

    with col_d:
        daily = df1.groupby('CRASH_DAY_OF_WEEK').size().reset_index(name='Crashes')
        daily['Day'] = daily['CRASH_DAY_OF_WEEK'].map(DAY_LABELS)
        fig_d = px.bar(daily, x='Day', y='Crashes',
                       title='Crashes by Day',
                       color='Crashes', color_continuous_scale='Oranges',
                       category_orders={'Day': list(DAY_LABELS.values())})
        fig_d.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_d, width="stretch")

    with col_m:
        monthly = df1.groupby('CRASH_MONTH').size().reset_index(name='Crashes')
        monthly['Month'] = monthly['CRASH_MONTH'].map(MONTH_LABELS)
        fig_m = px.bar(monthly, x='Month', y='Crashes',
                       title='Crashes by Month',
                       color='Crashes', color_continuous_scale='Oranges',
                       category_orders={'Month': list(MONTH_LABELS.values())})
        fig_m.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_m, width="stretch")

    peak_hour = hourly.loc[hourly["Crashes"].idxmax(), "CRASH_HOUR"]
    peak_day = daily.loc[daily["Crashes"].idxmax(), "Day"]
    peak_month = monthly.loc[monthly["Crashes"].idxmax(), "Month"]
    st.info(f"Peak hour **{peak_hour}:00**, peak day **{peak_day}**, peak month **{peak_month}**.")

    # ── Density map ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Crash Location Density")
    if geo is not None:
        try:
            coords, crash_counts = _crash_counts_by_area(
                city.key, path, json.dumps(geo),
                city.boundary_id_field, city.boundary_name_field,
                city.lat_bounds, city.lon_bounds,
            )

            if not coords.empty and not crash_counts.empty:
                fig_density = px.choropleth_map(
                    crash_counts, geojson=geo,
                    locations=city.boundary_id_field,
                    featureidkey=f"properties.{city.boundary_id_field}",
                    color="Crash Count", color_continuous_scale="YlOrRd",
                    map_style=mapbox_style,
                    zoom=city.zoom, center={"lat": city.center[0], "lon": city.center[1]},
                    hover_name=city.boundary_name_field if city.boundary_name_field in crash_counts.columns else None,
                    title="Crashes by Area",
                    opacity=0.7,
                )
                fig_density.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=500)
                st.plotly_chart(fig_density, width="stretch")

                fig_kde = px.density_map(
                    coords, lat="LATITUDE", lon="LONGITUDE", radius=15,
                    zoom=city.zoom, center={"lat": city.center[0], "lon": city.center[1]},
                    map_style=mapbox_style,
                    title="Crash Kernel Density",
                    color_continuous_scale="YlOrRd", opacity=0.8,
                )
                fig_kde.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=500)
                st.plotly_chart(fig_kde, width="stretch")

                top3 = crash_counts.nlargest(3, "Crash Count")
                name_col = city.boundary_name_field if city.boundary_name_field in top3.columns else city.boundary_id_field
                top_names = ", ".join(
                    f"**{r[name_col]}** ({r['Crash Count']:,})" for _, r in top3.iterrows()
                )
                st.info(f"**Crash hotspots:** {top_names}.")
        except Exception as exc:
            st.warning(f"Could not load crash location data: {exc}")

    st.divider()

    # ── Conditions ─────────────────────────────────────────────────────
    st.subheader("Road & Environment Conditions")
    condition_options = {
        'Weather Condition': 'WEATHER_CONDITION',
        'Lighting Condition': 'LIGHTING_CONDITION',
        'Roadway Surface Condition': 'ROADWAY_SURFACE_COND',
        'Road Defect': 'ROAD_DEFECT',
        'Traffic Control Device': 'TRAFFIC_CONTROL_DEVICE',
        'Alignment': 'ALIGNMENT',
    }
    available = {k: v for k, v in condition_options.items() if v in df1.columns}
    if available:
        selected_condition = st.selectbox("Breakdown by condition", list(available.keys()),
                                          key=f"infra_condition_{city.key}")
        col = available[selected_condition]
        cond_counts = df1[col].value_counts().reset_index()
        cond_counts.columns = [selected_condition, 'Crashes']
        fig_cond = px.bar(cond_counts, x='Crashes', y=selected_condition,
                          orientation='h',
                          title=f'Crash Count by {selected_condition}',
                          color='Crashes', color_continuous_scale='Oranges')
        fig_cond.update_layout(yaxis={'categoryorder': 'total ascending'},
                               coloraxis_showscale=False)
        st.plotly_chart(fig_cond, width="stretch")

        if not cond_counts.empty:
            top_cond = cond_counts.iloc[0][selected_condition]
            top_pct = cond_counts.iloc[0]['Crashes'] / cond_counts['Crashes'].sum() * 100
            st.info(f"**{top_cond}** accounts for **{top_pct:.1f}%** of crashes.")

    st.divider()

    # ── Crash type ─────────────────────────────────────────────────────
    st.subheader("Crash Type Breakdown")
    col_ct, col_tw = st.columns(2)
    with col_ct:
        if 'FIRST_CRASH_TYPE' in df1.columns:
            ct = df1['FIRST_CRASH_TYPE'].value_counts().head(12).reset_index()
            ct.columns = ['Crash Type', 'Count']
            fig_ct = px.bar(ct, x='Count', y='Crash Type', orientation='h',
                            title='Top Crash Types',
                            color='Count', color_continuous_scale='Oranges')
            fig_ct.update_layout(yaxis={'categoryorder': 'total ascending'},
                                 coloraxis_showscale=False)
            st.plotly_chart(fig_ct, width="stretch")
    with col_tw:
        if 'TRAFFICWAY_TYPE' in df2.columns:
            tw = df2['TRAFFICWAY_TYPE'].value_counts().head(12).reset_index()
            tw.columns = ['Trafficway Type', 'Count']
            fig_tw = px.bar(tw, x='Count', y='Trafficway Type', orientation='h',
                            title='Crashes by Trafficway Type',
                            color='Count', color_continuous_scale='Oranges')
            fig_tw.update_layout(yaxis={'categoryorder': 'total ascending'},
                                 coloraxis_showscale=False)
            st.plotly_chart(fig_tw, width="stretch")

    st.divider()

    # ── Damage ─────────────────────────────────────────────────────────
    if 'DAMAGE' in df2.columns and 'HIT_AND_RUN_I' in df2.columns:
        st.subheader("Damage Severity")
        col_dmg, col_hr = st.columns(2)
        with col_dmg:
            order = ['$500 OR LESS', '$501 - $1,500', 'OVER $1,500']
            dmg = (df2['DAMAGE'].str.upper().str.strip()
                   .value_counts().reindex(order, fill_value=0).reset_index())
            dmg.columns = ['Damage Level', 'Crashes']
            fig_dmg = px.pie(dmg, names='Damage Level', values='Crashes',
                             title='Damage Distribution',
                             color_discrete_sequence=px.colors.sequential.Oranges[2:])
            st.plotly_chart(fig_dmg, width="stretch")
        with col_hr:
            hr = (df2['HIT_AND_RUN_I'].str.upper().str.strip()
                  .map({'Y': 'Hit and Run', 'N': 'Not Hit and Run'})
                  .value_counts().reset_index())
            hr.columns = ['Type', 'Crashes']
            fig_hr = px.pie(hr, names='Type', values='Crashes',
                            title='Hit and Run',
                            color_discrete_sequence=['#fd8d3c', '#fdbe85'])
            st.plotly_chart(fig_hr, width="stretch")

    st.divider()

    # ── Speed/lane ─────────────────────────────────────────────────────
    if 'POSTED_SPEED_LIMIT' in df1.columns and 'LANE_CNT' in df1.columns:
        st.subheader("Road Characteristics")
        col_sp, col_ln = st.columns(2)
        with col_sp:
            sp = df1['POSTED_SPEED_LIMIT'].value_counts().sort_index().reset_index()
            sp.columns = ['Speed Limit (mph)', 'Crashes']
            fig_sp = px.bar(sp, x='Speed Limit (mph)', y='Crashes',
                            title='Crashes by Speed Limit',
                            color='Crashes', color_continuous_scale='Oranges')
            fig_sp.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_sp, width="stretch")
        with col_ln:
            ln = df1['LANE_CNT'].value_counts().sort_index().reset_index()
            ln.columns = ['Lane Count', 'Crashes']
            fig_ln = px.bar(ln, x='Lane Count', y='Crashes',
                            title='Crashes by Lane Count',
                            color='Crashes', color_continuous_scale='Oranges')
            fig_ln.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_ln, width="stretch")

    # ── ML predictor ────────────────────────────────────────────────────
    st.divider()
    if 'DAMAGE' in df2.columns:
        ml_predictor.render_predictor(
            df2, key_prefix=f"crash_{city.key}",
            default_target="DAMAGE",
            default_features=[c for c in [
                'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'ROADWAY_SURFACE_COND',
                'POSTED_SPEED_LIMIT', 'TRAFFICWAY_TYPE', 'INTERSECTION_RELATED_I',
                'CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'NUM_UNITS', 'HIT_AND_RUN_I',
                'FIRST_CRASH_TYPE', 'CRASH_TYPE',
            ] if c in df2.columns],
        )
