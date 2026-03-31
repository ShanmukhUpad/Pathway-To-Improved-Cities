import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import ml_predictor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "transport_access_clean_data")


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


def render():
    st.header("Transportation Access Analysis")
    st.markdown(
        "Bus stop coverage, Divvy bike-share distribution, bike route infrastructure, "
        "and ward-level accessibility across Chicago."
    )

    try:
        bus, divvy, bike, bus_per_ward, divvy_per_ward, merged = _load_and_process()
    except Exception as exc:
        st.error(f"Failed to load transportation access data: {exc}")
        return

    # ── 1. Bus Stops ─────────────────────────────────────────────────────
    st.subheader("Bus Stop Coverage")
    col_bmap, col_bbar = st.columns([3, 2])

    with col_bmap:
        fig_bmap = px.scatter_mapbox(
            bus,
            lat="latitude", lon="longitude",
            color="ward",
            hover_name="cta_stop_name",
            hover_data={"routes": True, "ward": True},
            title="Bus Stops by Ward",
            mapbox_style="carto-positron",
            zoom=9.5,
            center={"lat": 41.8358, "lon": -87.6877},
            height=420,
        )
        fig_bmap.update_traces(marker_size=4)
        fig_bmap.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, showlegend=False)
        st.plotly_chart(fig_bmap, use_container_width=True)

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
        st.plotly_chart(fig_bbar, use_container_width=True)

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
    st.plotly_chart(fig_routes, use_container_width=True)

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
        fig_dmap = px.scatter_mapbox(
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
            title="Divvy Stations (In Service) — Utilization Ratio",
            mapbox_style="carto-positron",
            zoom=9.5,
            center={"lat": 41.8358, "lon": -87.6877},
            height=420,
        )
        fig_dmap.update_traces(marker_size=6)
        fig_dmap.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})
        st.plotly_chart(fig_dmap, use_container_width=True)

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
        st.plotly_chart(fig_util, use_container_width=True)

        fig_docks = px.histogram(
            divvy, x="total_docks",
            title="Total Docks per Station",
            labels={"total_docks": "Total Docks"},
            color_discrete_sequence=["#74c476"],
            nbins=15,
            height=195,
        )
        fig_docks.update_layout(margin={"t": 30})
        st.plotly_chart(fig_docks, use_container_width=True)

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
        st.plotly_chart(fig_rt, use_container_width=True)

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
        st.plotly_chart(fig_cfl, use_container_width=True)

    top_route_type  = route_counts.iloc[0]["Route Type"]
    top_route_pct   = route_counts.iloc[0]["Count"] / route_counts["Count"].sum() * 100
    contraflow_pct  = bike["contraflow_flag"].mean() * 100
    st.info(
        f"**Bike infrastructure insight:** **{top_route_type}** is the dominant route type "
        f"({top_route_pct:.1f}% of all routes). "
        f"Only **{contraflow_pct:.1f}% of routes have contraflow lanes**, which allow cyclists to travel against "
        "one-way traffic. Expanding contraflow coverage improves safety and connectivity in dense urban areas."
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
        st.plotly_chart(fig_acc, use_container_width=True)

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
                use_container_width=True,
                height=300,
            )

    top_ward    = merged.loc[merged["accessibility_score"].idxmax()]
    bottom_ward = merged.loc[merged["accessibility_score"].idxmin()]
    underserved_list = ", ".join(f"Ward {int(r['ward'])}" for _, r in underserved.head(5).iterrows())
    st.info(
        f"**Accessibility insight:** Ward **{int(top_ward['ward'])}** has the highest accessibility score "
        f"({top_ward['accessibility_score']:.1f}), while Ward **{int(bottom_ward['ward'])}** scores lowest "
        f"({bottom_ward['accessibility_score']:.1f}). "
        f"**{len(underserved)} wards are underserved** (below median in both bus stops and Divvy stations)"
        + (f": {underserved_list}. " if not underserved.empty else ". ")
        + "These wards should be prioritized for transit investment to reduce mobility inequality across Chicago."
    )

    st.divider()

    # ── 5. ML Predictions ────────────────────────────────────────────────
    ml_predictor.render_predictor(
        merged,
        key_prefix="transport_access",
        default_target="accessibility_score",
        default_features=["num_stations", "avg_docks", "avg_routes_per_stop"],
    )
