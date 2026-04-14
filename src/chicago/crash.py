import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import os
import file_loader
import ml_predictor
import map_utils

_SRC = os.path.dirname(os.path.abspath(__file__))
# Prefer the auto-fetched file; fall back to the bundled snapshot
CRASH_CSV_LATEST = os.path.join(_SRC, "traffic_crashes_latest.csv")
CRASH_CSV_LEGACY = os.path.join(_SRC, "Traffic_Crashes_-_Crashes_20260309.csv")


def _resolve_crash_csv() -> str | None:
    """Return the best available crash CSV path, or None if neither exists."""
    if os.path.exists(CRASH_CSV_LATEST):
        return CRASH_CSV_LATEST
    if os.path.exists(CRASH_CSV_LEGACY):
        return CRASH_CSV_LEGACY
    return None

DAY_LABELS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

# ──────────────────────────────────────────────
# Data loading & cleaning  (notebook-faithful)
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# Shared crash cleaning helpers
# ──────────────────────────────────────────────

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
    """
    Parse dates, build df1 (road/conditions) and df2 (severity/damage)
    from a raw crash DataFrame.  Used by both load_crash_data() and the
    uploaded-file path.
    """
    df = df.copy()
    df['CRASH_DATE'] = pd.to_datetime(df['CRASH_DATE'])
    df['CRASH_HOUR']        = df['CRASH_DATE'].dt.hour
    df['CRASH_DAY_OF_WEEK'] = df['CRASH_DATE'].dt.dayofweek
    df['CRASH_MONTH']       = df['CRASH_DATE'].dt.month

    # ── df1: road / environment conditions ──────────────────────────────
    df1 = df[_DF1_COLS].copy().dropna()
    df1['LANE_CNT'] = pd.to_numeric(df1['LANE_CNT'], errors='coerce')
    df1.dropna(subset=['LANE_CNT'], inplace=True)
    df1['LANE_CNT'] = df1['LANE_CNT'].astype(int)

    for col in ['WEATHER_CONDITION', 'LIGHTING_CONDITION', 'ROADWAY_SURFACE_COND',
                'ROAD_DEFECT', 'ALIGNMENT', 'TRAFFICWAY_TYPE',
                'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION', 'FIRST_CRASH_TYPE']:
        df1 = df1[df1[col].str.upper().str.strip() != 'UNKNOWN']
        df1 = df1[df1[col].str.strip() != '']

    df1 = df1[df1['INTERSECTION_RELATED_I'].str.upper().str.strip().isin(['Y', 'N'])]
    df1 = df1[(df1['POSTED_SPEED_LIMIT'] > 0) & (df1['POSTED_SPEED_LIMIT'] <= 100)]
    df1 = df1[(df1['LANE_CNT'] > 0) & (df1['LANE_CNT'] <= 20)]
    df1 = df1[df1['CRASH_HOUR'].between(0, 23)]
    df1 = df1[df1['CRASH_DAY_OF_WEEK'].between(0, 6)]
    df1 = df1[df1['CRASH_MONTH'].between(1, 12)]
    df1.reset_index(drop=True, inplace=True)

    # ── df2: severity / damage ──────────────────────────────────────────
    df2 = df[_DF2_COLS].copy().dropna()

    for col in ['FIRST_CRASH_TYPE', 'CRASH_TYPE', 'WEATHER_CONDITION',
                'LIGHTING_CONDITION', 'ROADWAY_SURFACE_COND', 'TRAFFICWAY_TYPE']:
        df2 = df2[df2[col].str.upper().str.strip() != 'UNKNOWN']
        df2 = df2[df2[col].str.strip() != '']

    df2 = df2[df2['INTERSECTION_RELATED_I'].str.upper().str.strip().isin(['Y', 'N'])]
    df2 = df2[df2['HIT_AND_RUN_I'].str.upper().str.strip().isin(['Y', 'N'])]
    df2 = df2[df2['DAMAGE'].str.upper().str.strip().isin(['$500 OR LESS', '$501 - $1,500', 'OVER $1,500'])]
    df2 = df2[(df2['POSTED_SPEED_LIMIT'] > 0) & (df2['POSTED_SPEED_LIMIT'] <= 100)]
    df2 = df2[(df2['NUM_UNITS'] > 0) & (df2['NUM_UNITS'] <= 50)]
    df2 = df2[df2['CRASH_HOUR'].between(0, 23)]
    df2 = df2[df2['CRASH_DAY_OF_WEEK'].between(0, 6)]
    df2.reset_index(drop=True, inplace=True)

    return df1, df2


@st.cache_data(show_spinner="Loading crash data...")
def load_crash_data():
    path = _resolve_crash_csv()
    if path is None:
        raise FileNotFoundError("No crash CSV found.")
    df = pd.read_csv(path, low_memory=False)
    return _clean_crash_df(df)


def render(chicago_geo=None):
    st.header("Transportation Dashboard")
    st.markdown(
        "Traffic crash patterns across Chicago — road conditions, timing, "
        "crash types, and damage severity."
    )

    mapbox_style = map_utils.mapbox_style_picker(key_prefix="crash")

    with st.expander("Upload a supplemental dataset"):
        file_loader.uploader(
            domain="transportation",
            local_csv=None,
            label="Upload a crash dataset"
        )

    try:
        df1, df2 = load_crash_data()
    except FileNotFoundError:
        st.info("No local crash data found. Fetching the latest data from the Chicago Data Portal…")
        try:
            import data_fetcher
            data_fetcher.fetch_crashes(force=True)
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(
                f"Auto-fetch failed: {exc}\n\n"
                "You can also download the file manually from the "
                "[Chicago Data Portal — Traffic Crashes]"
                "(https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if) "
                "and place it in the `src/` folder."
            )
        return

    # ── Section 1: Temporal patterns ────────────────────────────────────
    st.subheader("Crash Timing")
    col_h, col_d, col_m = st.columns(3)

    with col_h:
        hourly = df1.groupby('CRASH_HOUR').size().reset_index(name='Crashes')
        fig_h = px.bar(
            hourly, x='CRASH_HOUR', y='Crashes',
            labels={'CRASH_HOUR': 'Hour of Day', 'Crashes': 'Number of Crashes'},
            title='Crashes by Hour of Day',
            color='Crashes', color_continuous_scale='Oranges'
        )
        fig_h.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_h, width="stretch")

    with col_d:
        daily = df1.groupby('CRASH_DAY_OF_WEEK').size().reset_index(name='Crashes')
        daily['Day'] = daily['CRASH_DAY_OF_WEEK'].map(DAY_LABELS)
        fig_d = px.bar(
            daily, x='Day', y='Crashes',
            labels={'Day': 'Day of Week', 'Crashes': 'Number of Crashes'},
            title='Crashes by Day of Week',
            color='Crashes', color_continuous_scale='Oranges',
            category_orders={'Day': list(DAY_LABELS.values())}
        )
        fig_d.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_d, width="stretch")

    with col_m:
        monthly = df1.groupby('CRASH_MONTH').size().reset_index(name='Crashes')
        monthly['Month'] = monthly['CRASH_MONTH'].map(MONTH_LABELS)
        fig_m = px.bar(
            monthly, x='Month', y='Crashes',
            labels={'Month': 'Month', 'Crashes': 'Number of Crashes'},
            title='Crashes by Month',
            color='Crashes', color_continuous_scale='Oranges',
            category_orders={'Month': list(MONTH_LABELS.values())}
        )
        fig_m.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_m, width="stretch")

    # ── Timing summary ───────────────────────────────────────────────────────
    peak_hour = hourly.loc[hourly["Crashes"].idxmax(), "CRASH_HOUR"]
    peak_day  = daily.loc[daily["Crashes"].idxmax(), "Day"]
    peak_month= monthly.loc[monthly["Crashes"].idxmax(), "Month"]
    st.info(
        f"**When crashes happen most:** Peak hour is **{peak_hour}:00** "
        f"({'evening rush' if 15 <= peak_hour <= 19 else 'morning rush' if 6 <= peak_hour <= 9 else 'overnight' if peak_hour < 6 else 'midday'}), "
        f"peak day is **{peak_day}**, and peak month is **{peak_month}**. "
        "Targeted enforcement and road safety campaigns during these windows could meaningfully reduce crash frequency."
    )

    # ── Crash Density Heatmap ───────────────────────────────────────────────
    st.divider()
    st.subheader("Crash Location Density")
    path = _resolve_crash_csv()
    if path and chicago_geo:
        try:
            raw_coords = pd.read_csv(path, usecols=["LATITUDE", "LONGITUDE"], low_memory=False)
            raw_coords["LATITUDE"] = pd.to_numeric(raw_coords["LATITUDE"], errors="coerce")
            raw_coords["LONGITUDE"] = pd.to_numeric(raw_coords["LONGITUDE"], errors="coerce")
            coords = raw_coords.dropna(subset=["LATITUDE", "LONGITUDE"])
            coords = coords[(coords["LATITUDE"] > 41.6) & (coords["LATITUDE"] < 42.1)]

            if not coords.empty:
                # Spatial join to count crashes per community area
                geo_url = "https://raw.githubusercontent.com/RandomFractals/ChicagoCrimes/master/data/chicago-community-areas.geojson"
                gdf_ca = gpd.read_file(geo_url)
                gdf_ca["area_num_1"] = gdf_ca["area_num_1"].astype(int)

                crash_pts = gpd.GeoDataFrame(
                    coords,
                    geometry=gpd.points_from_xy(coords["LONGITUDE"], coords["LATITUDE"]),
                    crs="EPSG:4326",
                )
                joined = gpd.sjoin(crash_pts, gdf_ca[["area_num_1", "community", "geometry"]],
                                   how="inner", predicate="within")
                crash_counts = joined.groupby(["area_num_1", "community"]).size().reset_index(name="Crash Count")
                crash_counts["area_num_1"] = crash_counts["area_num_1"].astype(str)

                fig_density = px.choropleth_map(
                    crash_counts, geojson=chicago_geo,
                    locations="area_num_1", featureidkey="properties.area_num_1",
                    color="Crash Count", color_continuous_scale="YlOrRd",
                    map_style=mapbox_style,
                    zoom=9.5, center={"lat": 41.8358, "lon": -87.6877},
                    hover_name="community",
                    hover_data={"Crash Count": True, "area_num_1": False},
                    title="Crashes by Community Area",
                    opacity=0.7,
                )
                fig_density.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=500)
                st.plotly_chart(fig_density, width="stretch")

                fig_kde = px.density_map(
                    coords,
                    lat="LATITUDE",
                    lon="LONGITUDE",
                    radius=15,
                    zoom=9.5,
                    center={"lat": 41.8358, "lon": -87.6877},
                    map_style=mapbox_style,
                    title="Crash Kernel Density Estimation",
                    color_continuous_scale="YlOrRd",
                    opacity=0.8,
                )
                fig_kde.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=500)
                st.plotly_chart(fig_kde, width="stretch")

                top3 = crash_counts.nlargest(3, "Crash Count")
                top_names = ", ".join(
                    f"**{r['community']}** ({r['Crash Count']:,})"
                    for _, r in top3.iterrows()
                )
                st.info(
                    f"**Crash hotspots by community area:** The highest crash counts are in {top_names}. "
                    "These areas should be prioritized for safety interventions such as traffic calming, "
                    "signal improvements, or targeted enforcement."
                )
        except Exception as exc:
            st.warning(f"Could not load crash location data: {exc}")

    st.divider()

    # ── Section 2: Road & environment conditions ─────────────────────────
    st.subheader("Road & Environment Conditions")

    condition_options = {
        'Weather Condition':        'WEATHER_CONDITION',
        'Lighting Condition':       'LIGHTING_CONDITION',
        'Roadway Surface Condition':'ROADWAY_SURFACE_COND',
        'Road Defect':              'ROAD_DEFECT',
        'Traffic Control Device':   'TRAFFIC_CONTROL_DEVICE',
        'Alignment':                'ALIGNMENT',
    }
    selected_condition = st.selectbox(
        "Breakdown by condition",
        list(condition_options.keys()),
        key="infra_condition"
    )
    col = condition_options[selected_condition]
    cond_counts = df1[col].value_counts().reset_index()
    cond_counts.columns = [selected_condition, 'Crashes']

    fig_cond = px.bar(
        cond_counts, x='Crashes', y=selected_condition,
        orientation='h',
        title=f'Crash Count by {selected_condition}',
        color='Crashes', color_continuous_scale='Oranges'
    )
    fig_cond.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_cond, width="stretch")

    top_cond       = cond_counts.iloc[0][selected_condition]
    top_cond_count = cond_counts.iloc[0]["Crashes"]
    total_crashes  = cond_counts["Crashes"].sum()
    top_cond_pct   = top_cond_count / total_crashes * 100
    st.info(
        f"**{selected_condition} insight:** **{top_cond}** accounts for "
        f"**{top_cond_pct:.1f}% of all crashes** ({top_cond_count:,} incidents). "
        "This is the highest-risk condition category. Infrastructure improvements or signage "
        "targeting this condition would have the greatest safety impact."
    )

    st.divider()

    # ── Section 3: Crash type breakdown ─────────────────────────────────
    st.subheader("Crash Type Breakdown")
    col_ct, col_tw = st.columns(2)

    with col_ct:
        ct_counts = df1['FIRST_CRASH_TYPE'].value_counts().head(12).reset_index()
        ct_counts.columns = ['Crash Type', 'Count']
        fig_ct = px.bar(
            ct_counts, x='Count', y='Crash Type', orientation='h',
            title='Top Crash Types',
            color='Count', color_continuous_scale='Oranges'
        )
        fig_ct.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_ct, width="stretch")

    with col_tw:
        tw_counts = df2['TRAFFICWAY_TYPE'].value_counts().head(12).reset_index()
        tw_counts.columns = ['Trafficway Type', 'Count']
        fig_tw = px.bar(
            tw_counts, x='Count', y='Trafficway Type', orientation='h',
            title='Crashes by Trafficway Type',
            color='Count', color_continuous_scale='Oranges'
        )
        fig_tw.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_tw, width="stretch")

    top_crash_type = ct_counts.iloc[0]["Crash Type"]
    top_crash_pct  = ct_counts.iloc[0]["Count"] / ct_counts["Count"].sum() * 100
    st.info(
        f"**Most common crash type: {top_crash_type}** ({top_crash_pct:.1f}% of crashes). "
        "Understanding dominant crash types informs the design of intersections, signage, and driver education programs."
    )

    st.divider()

    # ── Section 4: Damage severity ───────────────────────────────────────
    st.subheader("Damage Severity")
    col_dmg, col_hr = st.columns(2)

    with col_dmg:
        damage_order = ['$500 OR LESS', '$501 - $1,500', 'OVER $1,500']
        dmg_counts = (
            df2['DAMAGE']
            .str.upper().str.strip()
            .value_counts()
            .reindex(damage_order, fill_value=0)
            .reset_index()
        )
        dmg_counts.columns = ['Damage Level', 'Crashes']
        fig_dmg = px.pie(
            dmg_counts, names='Damage Level', values='Crashes',
            title='Crash Distribution by Damage Level',
            color_discrete_sequence=px.colors.sequential.Oranges[2:]
        )
        st.plotly_chart(fig_dmg, width="stretch")

    with col_hr:
        hr_counts = (
            df2['HIT_AND_RUN_I']
            .str.upper().str.strip()
            .map({'Y': 'Hit and Run', 'N': 'Not Hit and Run'})
            .value_counts()
            .reset_index()
        )
        hr_counts.columns = ['Type', 'Crashes']
        fig_hr = px.pie(
            hr_counts, names='Type', values='Crashes',
            title='Hit and Run vs. Not Hit and Run',
            color_discrete_sequence=['#fd8d3c', '#fdbe85']
        )
        st.plotly_chart(fig_hr, width="stretch")

    over_1500_pct = dmg_counts.loc[dmg_counts["Damage Level"] == "OVER $1,500", "Crashes"].sum() / dmg_counts["Crashes"].sum() * 100
    hr_pct        = df2["HIT_AND_RUN_I"].str.upper().str.strip().eq("Y").mean() * 100
    st.info(
        f"**Damage & hit-and-run:** {over_1500_pct:.1f}% of crashes result in damage over $1,500, "
        f"and **{hr_pct:.1f}% are hit-and-run** incidents. "
        + ("A high hit-and-run rate points to enforcement gaps. Increased camera coverage or penalties may deter this behavior."
           if hr_pct > 15 else
           "Hit-and-run rates are within a typical range, but continued monitoring is recommended.")
    )

    st.divider()

    # ── Section 5: Speed limit & lane count distributions ────────────────
    st.subheader("Road Characteristics")
    col_sp, col_ln = st.columns(2)

    with col_sp:
        speed_counts = df1['POSTED_SPEED_LIMIT'].value_counts().sort_index().reset_index()
        speed_counts.columns = ['Speed Limit (mph)', 'Crashes']
        fig_sp = px.bar(
            speed_counts, x='Speed Limit (mph)', y='Crashes',
            title='Crashes by Posted Speed Limit',
            color='Crashes', color_continuous_scale='Oranges'
        )
        fig_sp.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_sp, width="stretch")

    with col_ln:
        lane_counts = df1['LANE_CNT'].value_counts().sort_index().reset_index()
        lane_counts.columns = ['Lane Count', 'Crashes']
        fig_ln = px.bar(
            lane_counts, x='Lane Count', y='Crashes',
            title='Crashes by Number of Lanes',
            color='Crashes', color_continuous_scale='Oranges'
        )
        fig_ln.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_ln, width="stretch")

    st.divider()

    # ── Section 6: Intersection vs. non-intersection ─────────────────────
    st.subheader("Intersection-Related Crashes")
    col_int, col_units = st.columns(2)

    with col_int:
        int_counts = (
            df2['INTERSECTION_RELATED_I']
            .str.upper().str.strip()
            .map({'Y': 'Intersection-Related', 'N': 'Not Intersection-Related'})
            .value_counts()
            .reset_index()
        )
        int_counts.columns = ['Type', 'Crashes']
        fig_int = px.pie(
            int_counts, names='Type', values='Crashes',
            title='Intersection vs. Non-Intersection Crashes',
            color_discrete_sequence=['#e6550d', '#fdae6b']
        )
        st.plotly_chart(fig_int, width="stretch")

    with col_units:
        unit_counts = df2['NUM_UNITS'].value_counts().sort_index().reset_index()
        unit_counts.columns = ['Units Involved', 'Crashes']
        fig_units = px.bar(
            unit_counts.head(15), x='Units Involved', y='Crashes',
            title='Crashes by Number of Units Involved',
            color='Crashes', color_continuous_scale='Oranges'
        )
        fig_units.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_units, width="stretch")

    # ── Section 7: Scatterplots ────────────────────────────────────────────────
    st.divider()
    st.subheader("Speed and Lane Analysis Scatterplots")
    scatter_sample = df1.sample(min(5000, len(df1)), random_state=42) if len(df1) > 5000 else df1
    col_sc1, col_sc2 = st.columns(2)

    with col_sc1:
        fig_sc1 = px.scatter(
            scatter_sample, x="POSTED_SPEED_LIMIT", y="CRASH_HOUR",
            color="LIGHTING_CONDITION",
            title="Speed Limit vs Crash Hour by Lighting",
            labels={"POSTED_SPEED_LIMIT": "Posted Speed Limit (mph)", "CRASH_HOUR": "Hour of Day"},
            opacity=0.4,
        )
        fig_sc1.update_layout(margin={"t": 30}, legend=dict(orientation="h", yanchor="bottom", y=-0.4))
        st.plotly_chart(fig_sc1, width="stretch")

    with col_sc2:
        fig_sc2 = px.scatter(
            scatter_sample, x="LANE_CNT", y="POSTED_SPEED_LIMIT",
            color="ROADWAY_SURFACE_COND",
            title="Lane Count vs Speed Limit by Surface Condition",
            labels={"LANE_CNT": "Number of Lanes", "POSTED_SPEED_LIMIT": "Speed Limit (mph)"},
            opacity=0.4,
        )
        fig_sc2.update_layout(margin={"t": 30}, legend=dict(orientation="h", yanchor="bottom", y=-0.4))
        st.plotly_chart(fig_sc2, width="stretch")

    st.info(
        "**Scatter analysis:** These plots reveal relationships between road design and crash timing. "
        "Clusters at specific speed-hour combinations highlight when certain road types are most dangerous."
    )

    # ── Section 8: Moran's I Spatial Autocorrelation ─────────────────────────
    st.divider()
    try:
        if path:
            raw_for_moran = pd.read_csv(path, usecols=["LATITUDE", "LONGITUDE"], low_memory=False)
            raw_for_moran["LATITUDE"] = pd.to_numeric(raw_for_moran["LATITUDE"], errors="coerce")
            raw_for_moran["LONGITUDE"] = pd.to_numeric(raw_for_moran["LONGITUDE"], errors="coerce")
            raw_for_moran = raw_for_moran.dropna(subset=["LATITUDE", "LONGITUDE"])
            raw_for_moran = raw_for_moran[
                (raw_for_moran["LATITUDE"] > 41.6) & (raw_for_moran["LATITUDE"] < 42.1)
            ]

            if len(raw_for_moran) > 100:
                geo_url = "https://raw.githubusercontent.com/RandomFractals/ChicagoCrimes/master/data/chicago-community-areas.geojson"
                gdf_ca = gpd.read_file(geo_url)
                gdf_ca["area_num_1"] = gdf_ca["area_num_1"].astype(int)

                crash_points = gpd.GeoDataFrame(
                    raw_for_moran,
                    geometry=gpd.points_from_xy(raw_for_moran["LONGITUDE"], raw_for_moran["LATITUDE"]),
                    crs="EPSG:4326",
                )
                joined = gpd.sjoin(
                    crash_points,
                    gdf_ca[["area_num_1", "community", "geometry"]],
                    how="inner", predicate="within",
                )
                crash_by_ca = joined.groupby("area_num_1").size().reset_index(name="crash_count")
                gdf_merged = gdf_ca.merge(crash_by_ca, on="area_num_1", how="inner")
                gdf_merged["area_num_str"] = gdf_merged["area_num_1"].astype(str)

                if len(gdf_merged) >= 10 and chicago_geo:
                    map_utils.render_moran_analysis(
                        gdf=gdf_merged,
                        value_col="crash_count",
                        name_col="community",
                        id_col="area_num_str",
                        geojson=chicago_geo,
                        featureidkey="properties.area_num_1",
                        key_prefix="crash_moran",
                        map_style=mapbox_style,
                    )
    except Exception as exc:
        st.warning(f"Could not compute spatial autocorrelation: {exc}")

    # ── Section 9: ML Predictions ────────────────────────────────────────────
    st.divider()
    _CRASH_DEFAULT_FEATURES = [
        'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'ROADWAY_SURFACE_COND',
        'POSTED_SPEED_LIMIT', 'TRAFFICWAY_TYPE', 'INTERSECTION_RELATED_I',
        'CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'NUM_UNITS', 'HIT_AND_RUN_I',
        'FIRST_CRASH_TYPE', 'CRASH_TYPE',
    ]
    ml_predictor.render_predictor(
        df2,
        key_prefix="crash",
        default_target="DAMAGE",
        default_features=_CRASH_DEFAULT_FEATURES,
    )