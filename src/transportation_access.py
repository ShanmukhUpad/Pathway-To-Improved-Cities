import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import ml_predictor
import map_utils
from city_config import CityConfig

_DATA = os.path.dirname(os.path.abspath(__file__))


@st.cache_data(show_spinner="Loading transportation access data...")
def _load_and_process():
    bike  = pd.read_csv(os.path.join(_DATA, "bike_routes_clean.csv"))
    bus   = pd.read_csv(os.path.join(_DATA, "bus_stops_clean.csv"))
    divvy = pd.read_csv(os.path.join(_DATA, "divvy_bicycle_clean.csv"))

    bike.columns  = bike.columns.str.strip().str.lower()
    bus.columns   = bus.columns.str.strip().str.lower()
    divvy.columns = divvy.columns.str.strip().str.lower()

    # ── Bus stops ────────────────────────────────────────────────────────
    bus["num_routes"] = bus["routes"].apply(
        lambda x: len(str(x).split(",")) if pd.notnull(x) else 0
    )
    bus_per_ward = (
        bus.groupby("ward")
        .agg(num_stops=("stop_id", "count"), avg_routes_per_stop=("num_routes", "mean"))
        .reset_index()
    )

    # ── Divvy stations ───────────────────────────────────────────────────
    divvy = divvy[divvy["status"].str.strip().str.lower() == "in service"].copy()
    divvy["utilization_ratio"] = divvy["docks_in_service"] / divvy["total_docks"]

    # Assign each Divvy station to the ward of its nearest bus stop (vectorized)
    bus_lats  = bus["latitude"].values
    bus_lons  = bus["longitude"].values
    bus_wards = bus["ward"].values

    def _nearest_ward(lat, lon):
        dists = (bus_lats - lat) ** 2 + (bus_lons - lon) ** 2
        return bus_wards[np.argmin(dists)]

    divvy["ward"] = [
        _nearest_ward(r["latitude"], r["longitude"]) for _, r in divvy.iterrows()
    ]

    divvy_per_ward = (
        divvy.groupby("ward")
        .agg(num_stations=("id", "count"), avg_docks=("total_docks", "mean"))
        .reset_index()
    )

    # ── Bike routes ──────────────────────────────────────────────────────
    bike["contraflow_flag"] = bike["contraflow"].notnull().astype(int)

    # ── Ward-level merge ─────────────────────────────────────────────────
    merged = pd.merge(bus_per_ward, divvy_per_ward, on="ward", how="left").fillna(0)
    merged["accessibility_score"] = merged["num_stops"] + merged["avg_routes_per_stop"]
    merged["has_bike_infra"] = (
        merged["num_stations"] > merged["num_stations"].median()
    ).astype(int)

    return bus, divvy, bike, bus_per_ward, divvy_per_ward, merged


@st.cache_data(show_spinner="Loading ward boundaries...")
def _load_ward_geojson():
    resp = requests.get(
        "https://data.cityofchicago.org/resource/p293-wvbd.geojson",
        params={"$limit": 100}, timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def render(city: CityConfig):
    st.header(f"Transportation Access — {city.name}")

    if not city.has_transport_layer:
        st.info(
            f"Transit access layer (bus stops, bike-share, ward accessibility) is wired up "
            f"for Chicago only. {city.name} datasets not yet ingested."
        )
        return

    st.markdown(
        "Bus stop coverage, Divvy bike-share distribution, bike route infrastructure, "
        "and ward-level accessibility across Chicago."
    )

    mapbox_style = map_utils.mapbox_style_picker(key_prefix=f"transport_access_{city.key}")

    try:
        bus, divvy, bike, bus_per_ward, divvy_per_ward, merged = _load_and_process()
    except Exception as exc:
        st.error(f"Failed to load transportation access data: {exc}")
        return

    # ── 1. Bus Stops ─────────────────────────────────────────────────────
    st.subheader("Bus Stop Coverage")
    col_bmap, col_bbar = st.columns([3, 2])

    with col_bmap:
        if map_utils.MAPBOX_TOKEN:
            fig_bmap = go.Figure(go.Scattermap(
                lat=bus["latitude"],
                lon=bus["longitude"],
                mode="markers",
                marker=dict(size=5, color="#1f77b4"),
                text=bus["cta_stop_name"],
                hoverinfo="text",
                cluster=dict(enabled=True, maxzoom=12, step=50),
            ))
            fig_bmap.update_layout(
                map=dict(style=mapbox_style, center=dict(lat=41.8358, lon=-87.6877), zoom=9.5),
                margin={"r": 0, "t": 30, "l": 0, "b": 0},
                title="Bus Stops by Ward (Clustered)",
                showlegend=False, height=420,
            )
        else:
            fig_bmap = px.scatter_map(
                bus, lat="latitude", lon="longitude", color="ward",
                hover_name="cta_stop_name",
                hover_data={"routes": True, "ward": True},
                title="Bus Stops by Ward",
                map_style=mapbox_style, zoom=9.5,
                center={"lat": 41.8358, "lon": -87.6877}, height=420,
            )
            fig_bmap.update_traces(marker_size=4)
            fig_bmap.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, showlegend=False)
        st.plotly_chart(fig_bmap, width="stretch")

    with col_bbar:
        top_wards = bus_per_ward.sort_values("num_stops", ascending=False).head(20)
        fig_bbar = px.bar(
            top_wards, x="num_stops", y=top_wards["ward"].astype(str),
            orientation="h",
            title="Bus Stops per Ward (Top 20)",
            labels={"num_stops": "# Stops", "y": "Ward"},
            color="num_stops", color_continuous_scale="Blues",
            height=420,
        )
        fig_bbar.update_layout(
            yaxis={"categoryorder": "total ascending", "title": "Ward"},
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bbar, width="stretch")

    fig_routes = px.bar(
        bus_per_ward.sort_values("avg_routes_per_stop", ascending=False).head(20),
        x=bus_per_ward.sort_values("avg_routes_per_stop", ascending=False)
          .head(20)["ward"].astype(str),
        y="avg_routes_per_stop",
        title="Average Routes per Bus Stop by Ward (Top 20)",
        labels={"x": "Ward", "avg_routes_per_stop": "Avg Routes / Stop"},
        color="avg_routes_per_stop", color_continuous_scale="Blues",
    )
    fig_routes.update_layout(coloraxis_showscale=False, xaxis_title="Ward")
    st.plotly_chart(fig_routes, width="stretch")

    fig_bus_kde = px.density_map(
        bus,
        lat="latitude",
        lon="longitude",
        radius=15,
        zoom=9.5,
        center={"lat": 41.8358, "lon": -87.6877},
        map_style=mapbox_style,
        title="Bus Stop Kernel Density",
        color_continuous_scale="Blues",
        opacity=0.8,
    )
    fig_bus_kde.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=500)
    st.plotly_chart(fig_bus_kde, width="stretch")

    top_bus_ward    = bus_per_ward.loc[bus_per_ward["num_stops"].idxmax()]
    bottom_bus_ward = bus_per_ward.loc[bus_per_ward["num_stops"].idxmin()]
    st.info(
        f"**Bus stop coverage insight:** Ward **{int(top_bus_ward['ward'])}** has the most bus stops "
        f"({int(top_bus_ward['num_stops'])}), while Ward **{int(bottom_bus_ward['ward'])}** has the fewest "
        f"({int(bottom_bus_ward['num_stops'])}). "
        "Large gaps in stop counts between wards signal areas where residents may face longer walks to transit, "
        "disproportionately affecting low-income and elderly populations."
    )

    st.divider()

    # ── 2. Divvy Stations ────────────────────────────────────────────────
    st.subheader("Divvy Bike-Share Stations")
    col_dmap, col_dhist = st.columns([3, 2])

    with col_dmap:
        fig_dmap = px.scatter_map(
            divvy,
            lat="latitude", lon="longitude",
            color="utilization_ratio",
            color_continuous_scale="Greens",
            hover_name="station_name",
            hover_data={
                "total_docks": True,
                "docks_in_service": True,
                "utilization_ratio": ":.2f",
            },
            title="Divvy Stations (In Service) - Utilization Ratio",
            map_style=mapbox_style,
            zoom=9.5,
            center={"lat": 41.8358, "lon": -87.6877},
            height=420,
        )
        fig_dmap.update_traces(marker_size=6)
        fig_dmap.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})
        st.plotly_chart(fig_dmap, width="stretch")

    with col_dhist:
        fig_util = px.histogram(
            divvy, x="utilization_ratio",
            title="Utilization Ratio Distribution",
            labels={"utilization_ratio": "Utilization Ratio"},
            color_discrete_sequence=["#31a354"],
            nbins=20,
            height=195,
        )
        fig_util.update_layout(margin={"t": 30})
        st.plotly_chart(fig_util, width="stretch")

        fig_docks = px.histogram(
            divvy, x="total_docks",
            title="Total Docks per Station",
            labels={"total_docks": "Total Docks"},
            color_discrete_sequence=["#74c476"],
            nbins=15,
            height=195,
        )
        fig_docks.update_layout(margin={"t": 30})
        st.plotly_chart(fig_docks, width="stretch")

    avg_util         = divvy["utilization_ratio"].mean()
    near_capacity    = (divvy["utilization_ratio"] >= 0.9).sum()
    near_cap_pct     = near_capacity / len(divvy) * 100
    st.info(
        f"**Divvy utilization insight:** The average station utilization is **{avg_util:.1%}** "
        f"({near_capacity} stations, {near_cap_pct:.1f}% of the network, are at or above 90% capacity). "
        + ("Many stations are near capacity, suggesting dock shortages during peak times. "
           "Expanding docks or adding new stations in high-utilization areas would improve reliability."
           if near_cap_pct > 20 else
           "Most stations have available capacity, though targeted monitoring of near-full stations is still valuable.")
    )

    fig_divvy_kde = px.density_map(
        divvy,
        lat="latitude",
        lon="longitude",
        radius=20,
        zoom=9.5,
        center={"lat": 41.8358, "lon": -87.6877},
        map_style=mapbox_style,
        title="Divvy Station Kernel Density",
        color_continuous_scale="Greens",
        opacity=0.8,
    )
    fig_divvy_kde.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=500)
    st.plotly_chart(fig_divvy_kde, width="stretch")

    st.divider()

    # ── 3. Bike Route Infrastructure ─────────────────────────────────────
    st.subheader("Bike Route Infrastructure")
    col_rt, col_cfl = st.columns(2)

    with col_rt:
        route_counts = bike["displayrou"].value_counts().reset_index()
        route_counts.columns = ["Route Type", "Count"]
        fig_rt = px.bar(
            route_counts, x="Count", y="Route Type",
            orientation="h",
            title="Bike Routes by Type",
            color="Count", color_continuous_scale="Greens",
        )
        fig_rt.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_rt, width="stretch")

    with col_cfl:
        cfl_counts = (
            bike["contraflow_flag"]
            .map({1: "Has Contraflow", 0: "No Contraflow"})
            .value_counts()
            .reset_index()
        )
        cfl_counts.columns = ["Type", "Count"]
        fig_cfl = px.pie(
            cfl_counts, names="Type", values="Count",
            title="Contraflow Route Presence",
            color_discrete_sequence=["#31a354", "#a1d99b"],
        )
        st.plotly_chart(fig_cfl, width="stretch")

    top_route_type  = route_counts.iloc[0]["Route Type"]
    top_route_pct   = route_counts.iloc[0]["Count"] / route_counts["Count"].sum() * 100
    contraflow_pct  = bike["contraflow_flag"].mean() * 100
    st.info(
        f"**Bike infrastructure insight:** **{top_route_type}** is the dominant route type "
        f"({top_route_pct:.1f}% of all routes). "
        f"Only **{contraflow_pct:.1f}% of routes have contraflow lanes**, which allow cyclists to travel against "
        "one-way traffic. Expanding contraflow coverage improves safety and connectivity in dense urban areas."
    )

    # ── Transit Relationship Scatterplots ──────────────────────────────────
    st.divider()
    st.subheader("Transit Relationship Scatterplots")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        fig_s1 = px.scatter(
            merged, x="num_stops", y="num_stations",
            size="accessibility_score", color="accessibility_score",
            color_continuous_scale="RdYlGn",
            hover_data={"ward": True},
            title="Bus Stops vs Divvy Stations by Ward",
            labels={"num_stops": "Number of Bus Stops", "num_stations": "Number of Divvy Stations"},
        )
        fig_s1.update_layout(coloraxis_showscale=False, margin={"t": 30})
        st.plotly_chart(fig_s1, width="stretch")

    with col_s2:
        fig_s2 = px.scatter(
            merged, x="avg_routes_per_stop", y="avg_docks",
            color="has_bike_infra",
            hover_data={"ward": True},
            title="Avg Routes/Stop vs Avg Docks by Bike Infrastructure",
            labels={"avg_routes_per_stop": "Avg Routes per Stop", "avg_docks": "Avg Docks per Station"},
        )
        fig_s2.update_layout(margin={"t": 30})
        st.plotly_chart(fig_s2, width="stretch")

    st.info(
        "**Transit relationship insight:** These scatterplots reveal whether bus and bike-share infrastructure "
        "are co-located or complementary. Wards with high bus stops but few Divvy stations represent "
        "opportunities for bike-share expansion to create multimodal transit hubs."
    )

    st.divider()

    # ── 4. Ward-Level Accessibility ──────────────────────────────────────
    st.subheader("Ward-Level Accessibility")

    median_stops    = merged["num_stops"].median()
    median_stations = merged["num_stations"].median()
    underserved = merged[
        (merged["num_stops"] < median_stops) &
        (merged["num_stations"] < median_stations)
    ]

    col_acc, col_us = st.columns([3, 2])

    with col_acc:
        fig_acc = px.bar(
            merged.sort_values("accessibility_score", ascending=False),
            x=merged.sort_values("accessibility_score", ascending=False)["ward"].astype(str),
            y="accessibility_score",
            title="Accessibility Score by Ward",
            labels={"x": "Ward", "accessibility_score": "Accessibility Score"},
            color="accessibility_score",
            color_continuous_scale="RdYlGn",
        )
        fig_acc.update_layout(coloraxis_showscale=False, xaxis_title="Ward")
        st.plotly_chart(fig_acc, width="stretch")

    with col_us:
        st.metric(
            "Underserved Wards",
            len(underserved),
            help="Wards below median in both bus stops and Divvy stations",
        )
        if not underserved.empty:
            st.dataframe(
                underserved[["ward", "num_stops", "num_stations", "accessibility_score"]]
                .sort_values("accessibility_score")
                .reset_index(drop=True),
                width="stretch",
                height=300,
            )

    top_ward    = merged.loc[merged["accessibility_score"].idxmax()]
    bottom_ward = merged.loc[merged["accessibility_score"].idxmin()]
    underserved_list = ", ".join(f"Ward {int(r['ward'])}" for _, r in underserved.head(5).iterrows())
    st.info(
        "**What is a ward?** Chicago is divided into 50 wards, each represented by an elected alderperson. "
        "Wards are the primary unit of local government for infrastructure and transit decisions. "
        f"\n\n**Accessibility insight:** Ward **{int(top_ward['ward'])}** has the highest accessibility score "
        f"({top_ward['accessibility_score']:.1f}), while Ward **{int(bottom_ward['ward'])}** scores lowest "
        f"({bottom_ward['accessibility_score']:.1f}). "
        f"**{len(underserved)} wards are underserved** (below median in both bus stops and Divvy stations)"
        + (f": {underserved_list}. " if not underserved.empty else ". ")
        + "These wards should be prioritized for transit investment to reduce mobility inequality across Chicago."
    )

    # ── Moran's I Spatial Autocorrelation ────────────────────────────────
    st.divider()
    try:
        ward_geojson = _load_ward_geojson()
        gdf_wards = gpd.GeoDataFrame.from_features(ward_geojson["features"], crs="EPSG:4326")
        gdf_wards["ward"] = gdf_wards["ward"].astype(int)

        gdf_merged = gdf_wards.merge(merged, on="ward", how="inner")
        gdf_merged["ward_name"] = "Ward " + gdf_merged["ward"].astype(str)
        gdf_merged["ward_str"] = gdf_merged["ward"].astype(str)

        if len(gdf_merged) >= 10:
            map_utils.render_moran_analysis(
                gdf=gdf_merged,
                value_col="accessibility_score",
                name_col="ward_name",
                id_col="ward_str",
                geojson=ward_geojson,
                featureidkey="properties.ward",
                key_prefix="transport_moran",
                map_style=mapbox_style,
            )
    except Exception as exc:
        st.warning(f"Could not compute spatial autocorrelation: {exc}")

    st.divider()

    # ── 6. ML Predictions ────────────────────────────────────────────────
    ml_predictor.render_predictor(
        merged,
        key_prefix="transport_access",
        default_target="accessibility_score",
        default_features=["num_stations", "avg_docks", "avg_routes_per_stop"],
    )
