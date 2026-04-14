import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import file_loader
import ml_predictor
import map_utils

# ── City-specific constants ───────────────────────────────────────────────────
CITY_NAME  = "Los Angeles"
MAP_CENTER = {"lat": 34.0522, "lon": -118.2437}
MAP_ZOOM   = 9
LAT_MIN, LAT_MAX = 33.70, 34.40
LON_MIN, LON_MAX = -118.70, -117.90
GEO_URL    = "https://opendata.arcgis.com/datasets/031d488e158144d0b3aecaa9c888b7b3_0.geojson"
GEO_ID_FIELD = "APREC"
# ─────────────────────────────────────────────────────────────────────────────

_SRC = os.path.dirname(os.path.abspath(__file__))
CRASH_CSV = os.path.join(_SRC, "traffic_crashes_latest.csv")

# "NOT AVAILABLE" is used for columns that don't exist in this city's dataset
_NA = "NOT AVAILABLE"

DAY_LABELS   = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
MONTH_LABELS = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


def _has_real_data(df: pd.DataFrame, col: str) -> bool:
    """True if the column exists and has at least one value that isn't NA/unknown."""
    if col not in df.columns:
        return False
    valid = df[col].dropna()
    valid = valid[~valid.astype(str).str.upper().str.strip().isin(["UNKNOWN", _NA, ""])]
    return len(valid) > 0


@st.cache_data
def _load_crash_data():
    df = pd.read_csv(CRASH_CSV, low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    df["CRASH_DATE"] = pd.to_datetime(df["CRASH_DATE"], errors="coerce")
    df["CRASH_HOUR"]        = df["CRASH_DATE"].dt.hour
    df["CRASH_DAY_OF_WEEK"] = df["CRASH_DATE"].dt.dayofweek
    df["CRASH_MONTH"]       = df["CRASH_DATE"].dt.month
    return df


def _clean_categorical(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col].astype(str).str.strip()
    return s[~s.str.upper().isin(["UNKNOWN", _NA, "NAN", ""])]


def render(chicago_geo=None, area_map=None):
    st.header(f"Traffic Crash Analysis — {CITY_NAME}")

    with st.expander("Upload a supplemental dataset"):
        file_loader.uploader(domain="crashes", local_csv=None, label="Upload a crash dataset")

    if not os.path.exists(CRASH_CSV):
        st.info(f"No local crash data found. Fetching from {CITY_NAME} Open Data…")
        try:
            import data_fetcher
            data_fetcher.fetch_crashes(force=True)
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Auto-fetch failed: {exc}")
        return

    df = _load_crash_data()
    mapbox_style = map_utils.mapbox_style_picker(key_prefix="crash")

    # ── Section 1: Temporal patterns ─────────────────────────────────────────
    st.subheader("Crashes by Hour and Day")
    col1, col2 = st.columns(2)
    with col1:
        hour_counts = df["CRASH_HOUR"].value_counts().sort_index()
        fig_h = px.bar(x=hour_counts.index, y=hour_counts.values,
                       labels={"x": "Hour of Day", "y": "Crashes"},
                       title="Crashes by Hour of Day", color=hour_counts.values,
                       color_continuous_scale="Reds")
        fig_h.update_layout(showlegend=False, margin={"t": 40, "b": 0})
        st.plotly_chart(fig_h, width="stretch")
    with col2:
        dow_counts = df["CRASH_DAY_OF_WEEK"].value_counts().sort_index()
        fig_d = px.bar(x=[DAY_LABELS.get(i, i) for i in dow_counts.index], y=dow_counts.values,
                       labels={"x": "Day", "y": "Crashes"},
                       title="Crashes by Day of Week", color=dow_counts.values,
                       color_continuous_scale="Blues")
        fig_d.update_layout(showlegend=False, margin={"t": 40, "b": 0})
        st.plotly_chart(fig_d, width="stretch")

    # ── Section 2: Monthly trend ──────────────────────────────────────────────
    st.subheader("Monthly Crash Trend")
    monthly = df.groupby(df["CRASH_DATE"].dt.to_period("M")).size()
    monthly.index = monthly.index.astype(str)
    st.line_chart(monthly)

    # ── Section 3: Condition breakdowns (only show cols with real data) ───────
    cat_cols = {
        "WEATHER_CONDITION":     "Weather Conditions",
        "LIGHTING_CONDITION":    "Lighting Conditions",
        "ROADWAY_SURFACE_COND":  "Road Surface Conditions",
        "FIRST_CRASH_TYPE":      "Crash Type",
    }
    available_cats = {k: v for k, v in cat_cols.items() if _has_real_data(df, k)}

    if available_cats:
        st.subheader("Crash Conditions Breakdown")
        cols = st.columns(min(2, len(available_cats)))
        for i, (col_name, label) in enumerate(available_cats.items()):
            with cols[i % len(cols)]:
                counts = _clean_categorical(df, col_name).value_counts().head(10)
                fig = px.bar(x=counts.values, y=counts.index, orientation="h",
                             title=label, labels={"x": "Count", "y": ""},
                             color=counts.values, color_continuous_scale="Oranges")
                fig.update_layout(showlegend=False, height=350, margin={"t": 40, "b": 0})
                st.plotly_chart(fig, width="stretch")

    # ── Section 4: Speed limit distribution ───────────────────────────────────
    if _has_real_data(df, "POSTED_SPEED_LIMIT"):
        st.subheader("Posted Speed Limit Distribution")
        speeds = pd.to_numeric(df["POSTED_SPEED_LIMIT"], errors="coerce").dropna()
        speeds = speeds[(speeds > 0) & (speeds <= 100)]
        fig_sp = px.histogram(speeds, nbins=20, title="Speed Limit Distribution",
                              labels={"value": "Speed Limit (mph)", "count": "Crashes"})
        fig_sp.update_layout(margin={"t": 40, "b": 0})
        st.plotly_chart(fig_sp, width="stretch")

    # ── Section 5: Density map (only if lat/lon available) ────────────────────
    if _has_real_data(df, "LATITUDE") and _has_real_data(df, "LONGITUDE"):
        st.subheader("Crash Density Map")
        coords = df[["LATITUDE", "LONGITUDE"]].copy()
        coords["LATITUDE"]  = pd.to_numeric(coords["LATITUDE"],  errors="coerce")
        coords["LONGITUDE"] = pd.to_numeric(coords["LONGITUDE"], errors="coerce")
        coords = coords.dropna()
        coords = coords[
            (coords["LATITUDE"]  > LAT_MIN) & (coords["LATITUDE"]  < LAT_MAX) &
            (coords["LONGITUDE"] > LON_MIN) & (coords["LONGITUDE"] < LON_MAX)
        ]
        if not coords.empty:
            fig_kde = px.density_map(
                coords, lat="LATITUDE", lon="LONGITUDE",
                radius=8, zoom=MAP_ZOOM - 1, center=MAP_CENTER,
                map_style=mapbox_style,
                title=f"Crash Density — {CITY_NAME}",
            )
            fig_kde.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=500)
            st.plotly_chart(fig_kde, width="stretch")

    # ── Section 6: Moran's I ──────────────────────────────────────────────────
    if _has_real_data(df, "LATITUDE") and _has_real_data(df, "LONGITUDE"):
        st.divider()
        try:
            import geopandas as gpd
            coords2 = df[["LATITUDE", "LONGITUDE"]].copy()
            coords2["LATITUDE"]  = pd.to_numeric(coords2["LATITUDE"],  errors="coerce")
            coords2["LONGITUDE"] = pd.to_numeric(coords2["LONGITUDE"], errors="coerce")
            coords2 = coords2.dropna()
            coords2 = coords2[
                (coords2["LATITUDE"] > LAT_MIN)  & (coords2["LATITUDE"]  < LAT_MAX) &
                (coords2["LONGITUDE"] > LON_MIN) & (coords2["LONGITUDE"] < LON_MAX)
            ]
            gdf_area = gpd.read_file(GEO_URL)
            gdf_area[GEO_ID_FIELD] = gdf_area[GEO_ID_FIELD].astype(str)
            gdf_pts  = gpd.GeoDataFrame(coords2,
                                         geometry=gpd.points_from_xy(coords2["LONGITUDE"], coords2["LATITUDE"]),
                                         crs="EPSG:4326")
            joined   = gpd.sjoin(gdf_pts, gdf_area[[GEO_ID_FIELD, "geometry"]], how="left", predicate="within")
            crash_counts = joined.groupby(GEO_ID_FIELD).size().reset_index(name="crash_count")
            gdf_merged   = gdf_area.merge(crash_counts, on=GEO_ID_FIELD, how="left").fillna({"crash_count": 0})
            gdf_merged["_id_str"] = gdf_merged[GEO_ID_FIELD].astype(str)
            if len(gdf_merged) >= 10:
                map_utils.render_moran_analysis(
                    gdf=gdf_merged, value_col="crash_count",
                    name_col=GEO_ID_FIELD, id_col="_id_str",
                    geojson=gdf_area.__geo_interface__,
                    featureidkey=f"properties.{GEO_ID_FIELD}",
                    key_prefix="crash_moran", map_style=mapbox_style,
                )
        except Exception as exc:
            st.warning(f"Spatial autocorrelation unavailable: {exc}")

    # ── Section 7: ML predictor ───────────────────────────────────────────────
    st.divider()
    ml_cols = ["CRASH_HOUR", "CRASH_DAY_OF_WEEK", "CRASH_MONTH"]
    if _has_real_data(df, "POSTED_SPEED_LIMIT"):
        df["POSTED_SPEED_LIMIT"] = pd.to_numeric(df["POSTED_SPEED_LIMIT"], errors="coerce")
        ml_cols.append("POSTED_SPEED_LIMIT")
    if _has_real_data(df, "NUM_UNITS"):
        df["NUM_UNITS"] = pd.to_numeric(df["NUM_UNITS"], errors="coerce")
        ml_cols.append("NUM_UNITS")

    ml_df = df[ml_cols].dropna()
    if len(ml_df) >= 20:
        ml_predictor.render_predictor(
            ml_df, key_prefix="crash",
            default_target="CRASH_HOUR",
            default_features=[c for c in ml_cols if c != "CRASH_HOUR"],
        )
