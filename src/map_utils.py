"""
map_utils.py
------------
Shared utilities for Mapbox map styling and Moran's I spatial autocorrelation
analysis. Imported by all dashboard tab modules.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import warnings
from esda.moran import Moran, Moran_Local
from esda.getisord import G_Local
from libpysal.weights import Queen, KNN

# Load .env from src/ directory so the token is available at import time
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── Mapbox token ─────────────────────────────────────────────────────────────

_raw_token = os.environ.get("PUBLIC_MAPBOX_TOKEN", "")
# Plotly 6+ uses MapLibre by default. A public Mapbox token (pk.*) unlocks
# premium Mapbox-hosted styles (streets, satellite, etc.).
MAPBOX_TOKEN = _raw_token if _raw_token.startswith("pk.") else ""

MAP_STYLES = {
    "Open Street Map": "open-street-map",
    "Streets":         "streets",
    "Light":           "light",
    "Dark":            "dark",
    "Satellite":       "satellite",
}


def init_mapbox():
    """Store the Mapbox token so Plotly can use premium map styles."""
    if MAPBOX_TOKEN:
        import plotly.io as pio
        pio.templates.default = "plotly"


def mapbox_style_picker(key_prefix: str = "map") -> str:
    """
    Render a selectbox for map styles and return the chosen style string.
    Shows all styles when a Mapbox token is set; otherwise only open-street-map.
    """
    if not MAPBOX_TOKEN:
        return "open-street-map"

    choice = st.selectbox(
        "Map style",
        list(MAP_STYLES.keys()),
        index=1,  # default to Streets
        key=f"{key_prefix}_map_style",
    )
    return MAP_STYLES[choice]


# ── Moran's I ────────────────────────────────────────────────────────────────

_LISA_LABELS = {
    1: "HH (Hot Spot)",
    2: "LH",
    3: "LL (Cold Spot)",
    4: "HL",
}
_LISA_COLORS = {
    "HH (Hot Spot)":   "#d7191c",
    "HL":              "#fdae61",
    "LH":              "#abd9e9",
    "LL (Cold Spot)":  "#2c7bb6",
    "Not Significant": "#d3d3d3",
}

_HOTSPOT_COLORS = {
    "Hot Spot (99% confidence)":  "#d7191c",
    "Hot Spot (95% confidence)":  "#fdae61",
    "Hot Spot (90% confidence)":  "#fee08b",
    "Not Significant":            "#d3d3d3",
    "Cold Spot (90% confidence)": "#d1ecf1",
    "Cold Spot (95% confidence)": "#abd9e9",
    "Cold Spot (99% confidence)": "#2c7bb6",
}


def _build_weights(gdf: gpd.GeoDataFrame):
    """Build Queen contiguity weights; fall back to KNN(k=5) if islands exist."""
    try:
        w = Queen.from_dataframe(gdf)
        if w.n != len(gdf) or len(w.islands) > 0:
            raise ValueError("islands detected")
    except Exception:
        w = KNN.from_dataframe(gdf, k=5)
    w.transform = "r"
    return w


@st.cache_data(show_spinner="Computing Moran's I...")
def _compute_moran(gdf_json: str, value_col: str):
    """
    Cached computation of global + local Moran's I.
    Accepts GeoJSON string (hashable for caching) instead of GeoDataFrame.
    Returns dict with all needed results.
    """
    gdf = gpd.read_file(gdf_json, driver="GeoJSON")
    y = gdf[value_col].values.astype(float)
    w = _build_weights(gdf)

    # Global
    moran_global = Moran(y, w, permutations=99)

    # Local (LISA)
    moran_local = Moran_Local(y, w, permutations=99)

    # Classify quadrants
    sig = moran_local.p_sim < 0.05
    quadrant = moran_local.q.copy()
    labels = []
    for i in range(len(gdf)):
        if sig[i]:
            labels.append(_LISA_LABELS.get(quadrant[i], "Not Significant"))
        else:
            labels.append("Not Significant")

    # Z-scores for scatterplot
    z = (y - y.mean()) / y.std()
    lag = w.sparse.dot(z)

    # Getis-Ord Gi* hot spot analysis
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g_local = G_Local(y, w, transform="R", star=True, permutations=99)

    gi_labels = []
    for i in range(len(y)):
        p = g_local.p_sim[i]
        zg = g_local.Zs[i]
        if p < 0.01 and zg > 0:
            gi_labels.append("Hot Spot (99% confidence)")
        elif p < 0.05 and zg > 0:
            gi_labels.append("Hot Spot (95% confidence)")
        elif p < 0.10 and zg > 0:
            gi_labels.append("Hot Spot (90% confidence)")
        elif p < 0.01 and zg < 0:
            gi_labels.append("Cold Spot (99% confidence)")
        elif p < 0.05 and zg < 0:
            gi_labels.append("Cold Spot (95% confidence)")
        elif p < 0.10 and zg < 0:
            gi_labels.append("Cold Spot (90% confidence)")
        else:
            gi_labels.append("Not Significant")

    return {
        "I": round(moran_global.I, 4),
        "p_value": round(moran_global.p_sim, 4),
        "z_score": round(moran_global.z_sim, 4),
        "lisa_labels": labels,
        "z": z.tolist(),
        "lag": lag.tolist(),
        "local_Is": [round(v, 4) for v in moran_local.Is.tolist()],
        "local_p_sim": [round(v, 4) for v in moran_local.p_sim.tolist()],
        "quadrants": quadrant.tolist(),
        "gi_labels": gi_labels,
        "gi_z_scores": [round(v, 4) for v in g_local.Zs.tolist()],
        "gi_p_values": [round(v, 4) for v in g_local.p_sim.tolist()],
    }


def render_moran_analysis(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    name_col: str,
    id_col: str,
    geojson: dict,
    featureidkey: str,
    key_prefix: str,
    map_style: str = "streets",
):
    """
    Render a complete Moran's I spatial autocorrelation section:
      1. Global Moran's I metrics
      2. Moran scatterplot (z-score vs spatial lag)
      3. LISA cluster map (choropleth)
    """
    st.subheader("Spatial Autocorrelation (Moran's I)")

    valid = gdf[[value_col, name_col, id_col, "geometry"]].dropna(subset=[value_col]).copy()
    if len(valid) < 10:
        st.warning("Not enough areas with valid data to compute Moran's I.")
        return

    # Serialize GeoDataFrame to GeoJSON string for caching
    gdf_json = valid.to_json()
    result = _compute_moran(gdf_json, value_col)

    # ── 1. Global Moran's I metrics ──────────────────────────────────────
    autocorr = (
        "positive (similar values cluster together)"
        if result["I"] > 0 else
        "negative (dissimilar values are neighbors)"
    )
    sig_label = "statistically significant" if result["p_value"] < 0.05 else "not statistically significant"

    c1, c2, c3 = st.columns(3)
    c1.metric("Moran's I", f"{result['I']:.4f}")
    c2.metric("p-value", f"{result['p_value']:.4f}")
    c3.metric("z-score", f"{result['z_score']:.4f}")

    st.info(
        f"**Global Moran's I = {result['I']:.4f}** indicates {autocorr} spatial autocorrelation. "
        f"This result is **{sig_label}** (p = {result['p_value']:.4f}). "
        + ("High-value areas tend to be near other high-value areas, and low near low. "
           "Targeted, geographically-focused interventions are likely more effective than city-wide programs."
           if result["I"] > 0 and result["p_value"] < 0.05 else
           "The spatial pattern does not show strong clustering. "
           "City-wide approaches may be as appropriate as geographically-targeted ones."
           if result["p_value"] >= 0.05 else
           "Neighboring areas tend to have contrasting values, suggesting a checkerboard pattern.")
    )

    # ── 2. Moran scatterplot ─────────────────────────────────────────────
    col_scatter, col_map = st.columns(2)

    with col_scatter:
        z_vals = result["z"]
        lag_vals = result["lag"]
        names = valid[name_col].tolist()

        quad_colors = []
        for zv, lv in zip(z_vals, lag_vals):
            if zv > 0 and lv > 0:
                quad_colors.append("HH (Hot Spot)")
            elif zv < 0 and lv < 0:
                quad_colors.append("LL (Cold Spot)")
            elif zv > 0 and lv < 0:
                quad_colors.append("HL")
            else:
                quad_colors.append("LH")

        scatter_df = pd.DataFrame({
            "z_score": z_vals,
            "spatial_lag": lag_vals,
            "name": names,
            "quadrant": quad_colors,
        })

        fig_scatter = px.scatter(
            scatter_df, x="z_score", y="spatial_lag",
            color="quadrant",
            color_discrete_map=_LISA_COLORS,
            hover_name="name",
            labels={"z_score": "Z-score", "spatial_lag": "Spatial Lag of Z-score"},
            title="Moran Scatterplot",
        )
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
        # Add OLS fit line
        z_arr = np.array(z_vals)
        lag_arr = np.array(lag_vals)
        slope = np.polyfit(z_arr, lag_arr, 1)
        x_range = np.linspace(z_arr.min(), z_arr.max(), 50)
        fig_scatter.add_trace(go.Scatter(
            x=x_range, y=np.polyval(slope, x_range),
            mode="lines", line=dict(color="black", width=1.5, dash="dot"),
            name=f"Slope = {slope[0]:.3f}",
        ))
        fig_scatter.update_layout(
            margin={"t": 30, "b": 0},
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig_scatter, width="stretch")

    # ── 3. LISA cluster map ──────────────────────────────────────────────
    with col_map:
        lisa_df = pd.DataFrame({
            id_col: valid[id_col].values,
            "LISA Cluster": result["lisa_labels"],
            name_col: valid[name_col].values,
            value_col: valid[value_col].values,
        })

        fig_lisa = px.choropleth_map(
            lisa_df, geojson=geojson,
            locations=id_col, featureidkey=featureidkey,
            color="LISA Cluster",
            color_discrete_map=_LISA_COLORS,
            category_orders={"LISA Cluster": [
                "HH (Hot Spot)", "HL", "LH", "LL (Cold Spot)", "Not Significant"
            ]},
            map_style=map_style,
            zoom=9, center={"lat": 41.8358, "lon": -87.6877},
            opacity=0.7,
            hover_name=name_col,
            hover_data={value_col: True, "LISA Cluster": True},
            title="LISA Cluster Map",
        )
        fig_lisa.update_layout(
            margin={"r": 0, "t": 30, "l": 0, "b": 0},
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        )
        st.plotly_chart(fig_lisa, width="stretch")

    # ── 4. Cluster Summary Statistics ────────────────────────────────────
    st.markdown("---")
    st.subheader("LISA Cluster Summary")

    value_label = value_col.replace("_", " ").title()
    lisa_labels_list = result["lisa_labels"]
    n_total = len(lisa_labels_list)
    hh_count = lisa_labels_list.count("HH (Hot Spot)")
    hl_count = lisa_labels_list.count("HL")
    lh_count = lisa_labels_list.count("LH")
    ll_count = lisa_labels_list.count("LL (Cold Spot)")
    ns_count = lisa_labels_list.count("Not Significant")
    n_sig = n_total - ns_count

    cs1, cs2, cs3, cs4, cs5 = st.columns(5)
    cs1.metric("HH (Hot Spot)", hh_count)
    cs2.metric("HL (High-Low)", hl_count)
    cs3.metric("LH (Low-High)", lh_count)
    cs4.metric("LL (Cold Spot)", ll_count)
    cs5.metric("Not Significant", ns_count)

    sig_pct = n_sig / n_total * 100 if n_total else 0
    st.metric("Total Significant Areas", f"{n_sig} of {n_total} ({sig_pct:.1f}%)")

    if ns_count > 0.8 * n_total:
        cluster_insight = (
            f"Most areas ({ns_count} of {n_total}) show no significant local clustering. "
            f"Variations in {value_label} appear spatially random rather than concentrated. "
            "City-wide policies are likely as effective as geographically-targeted ones."
        )
    elif hh_count > 0 and ll_count > 0:
        cluster_insight = (
            f"Both high-value clusters ({hh_count} HH) and low-value clusters ({ll_count} LL) exist, "
            f"indicating a spatially polarized {value_label} landscape. "
            "Policy should address both ends: high-value clusters may need intervention to reduce concentration, "
            "while low-value clusters may benefit from targeted investment."
        )
    elif hh_count > ll_count:
        cluster_insight = (
            f"High-High clusters dominate ({hh_count} areas), suggesting concentrated zones of elevated {value_label}. "
            "These areas are candidates for place-based intervention such as targeted infrastructure upgrades, "
            "increased service deployment, or policy changes to address concentration."
        )
    else:
        cluster_insight = (
            f"Low-Low clusters dominate ({ll_count} areas), meaning parts of the city show uniformly low {value_label}. "
            "This may indicate systemic underinvestment. Consider equity audits to ensure resources reach these areas."
        )
    st.info(f"**Cluster pattern:** {cluster_insight}")

    # ── 5. LISA Statistics Table ─────────────────────────────────────────
    st.subheader("Local Indicator Details")

    names = valid[name_col].tolist()
    values = valid[value_col].tolist()
    local_is = result["local_Is"]
    local_ps = result["local_p_sim"]
    quads = result["quadrants"]
    quad_labels = [_LISA_LABELS.get(q, "N/A") for q in quads]
    sig_flags = ["Yes" if p < 0.05 else "No" for p in local_ps]

    lisa_table = pd.DataFrame({
        "Area": names,
        value_label: values,
        "Local Moran's I": local_is,
        "p-value": local_ps,
        "Quadrant": quad_labels,
        "Cluster": lisa_labels_list,
        "Significant": sig_flags,
    }).sort_values("p-value").reset_index(drop=True)

    st.dataframe(lisa_table, width=2000, height=400)

    # Identify top HH and LL areas for insight
    top_hh = lisa_table[(lisa_table["Cluster"] == "HH (Hot Spot)")].head(3)
    top_ll = lisa_table[(lisa_table["Cluster"] == "LL (Cold Spot)")].head(3)

    lisa_insight_parts = []
    if not top_hh.empty:
        hh_names = ", ".join(f"**{r['Area']}**" for _, r in top_hh.iterrows())
        lisa_insight_parts.append(
            f"**Most significant hot spots:** {hh_names} show the strongest High-High clustering "
            f"(lowest p-values). These areas and their neighbors all have elevated {value_label}. "
            "Prioritize these zones for concentrated intervention."
        )
    if not top_ll.empty:
        ll_names = ", ".join(f"**{r['Area']}**" for _, r in top_ll.iterrows())
        lisa_insight_parts.append(
            f"**Most significant cold spots:** {ll_names} show the strongest Low-Low clustering. "
            f"These areas form zones of depressed {value_label}. "
            "Investigate whether structural barriers explain the pattern."
        )
    if lisa_insight_parts:
        st.info(" ".join(lisa_insight_parts))

    # ── 6. Significance Maps ────────────────────────────────────────────
    st.subheader("Local Significance Maps")
    col_pmap, col_imap = st.columns(2)

    sig_df = pd.DataFrame({
        id_col: valid[id_col].values,
        name_col: valid[name_col].values,
        "p-value": local_ps,
        "Local Moran's I": local_is,
    })

    with col_pmap:
        fig_pval = px.choropleth_map(
            sig_df, geojson=geojson,
            locations=id_col, featureidkey=featureidkey,
            color="p-value",
            color_continuous_scale="YlOrRd_r",
            range_color=[0, 0.1],
            map_style=map_style,
            zoom=9, center={"lat": 41.8358, "lon": -87.6877},
            opacity=0.7,
            hover_name=name_col,
            hover_data={"p-value": ":.4f"},
            title="Local p-values (Moran's I)",
        )
        fig_pval.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=500)
        st.plotly_chart(fig_pval, width="stretch")

    with col_imap:
        local_is_arr = np.array(local_is)
        max_abs = max(abs(local_is_arr.min()), abs(local_is_arr.max()), 0.01)
        fig_local_i = px.choropleth_map(
            sig_df, geojson=geojson,
            locations=id_col, featureidkey=featureidkey,
            color="Local Moran's I",
            color_continuous_scale="RdBu_r",
            range_color=[-max_abs, max_abs],
            map_style=map_style,
            zoom=9, center={"lat": 41.8358, "lon": -87.6877},
            opacity=0.7,
            hover_name=name_col,
            hover_data={"Local Moran's I": ":.4f", "p-value": ":.4f"},
            title="Local Moran's I Values",
        )
        fig_local_i.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=500)
        st.plotly_chart(fig_local_i, width="stretch")

    n_sig_05 = sum(1 for p in local_ps if p < 0.05)
    n_sig_01 = sum(1 for p in local_ps if p < 0.01)
    st.info(
        f"**Significance overview:** {n_sig_05} of {n_total} areas ({n_sig_05/n_total*100:.1f}%) "
        f"show statistically significant local spatial autocorrelation at p < 0.05, "
        f"and {n_sig_01} ({n_sig_01/n_total*100:.1f}%) meet the stricter p < 0.01 threshold. "
        "Areas with low p-values are where the spatial clustering pattern is most robust "
        "and where geographically-targeted policy interventions have the strongest statistical justification."
    )

    # ── 7. Getis-Ord Gi* Hot/Cold Spot Analysis ─────────────────────────
    st.subheader("Getis-Ord Gi* Hot/Cold Spot Analysis")
    st.caption(
        "Gi* identifies statistically significant spatial clusters of high values (hot spots) "
        "and low values (cold spots). Unlike LISA, Gi* focuses purely on value concentration "
        "rather than spatial outliers (HL/LH)."
    )

    gi_labels = result["gi_labels"]
    gi_df = pd.DataFrame({
        id_col: valid[id_col].values,
        name_col: valid[name_col].values,
        "Gi* Classification": gi_labels,
        value_col: valid[value_col].values,
        "Gi* z-score": result["gi_z_scores"],
        "Gi* p-value": result["gi_p_values"],
    })

    fig_gi = px.choropleth_map(
        gi_df, geojson=geojson,
        locations=id_col, featureidkey=featureidkey,
        color="Gi* Classification",
        color_discrete_map=_HOTSPOT_COLORS,
        category_orders={"Gi* Classification": [
            "Hot Spot (99% confidence)", "Hot Spot (95% confidence)",
            "Hot Spot (90% confidence)", "Not Significant",
            "Cold Spot (90% confidence)", "Cold Spot (95% confidence)",
            "Cold Spot (99% confidence)",
        ]},
        map_style=map_style,
        zoom=9, center={"lat": 41.8358, "lon": -87.6877},
        opacity=0.7,
        hover_name=name_col,
        hover_data={value_col: True, "Gi* z-score": ":.4f", "Gi* p-value": ":.4f"},
        title="Gi* Hot/Cold Spot Map",
    )
    fig_gi.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=550,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    st.plotly_chart(fig_gi, width="stretch")

    n_hot = sum(1 for g in gi_labels if g.startswith("Hot Spot"))
    n_cold = sum(1 for g in gi_labels if g.startswith("Cold Spot"))
    n_gi_ns = sum(1 for g in gi_labels if g == "Not Significant")

    gc1, gc2, gc3 = st.columns(3)
    gc1.metric("Hot Spots", n_hot)
    gc2.metric("Cold Spots", n_cold)
    gc3.metric("Not Significant", n_gi_ns)

    hot_areas = [names[i] for i in range(n_total) if gi_labels[i].startswith("Hot Spot")]
    cold_areas = [names[i] for i in range(n_total) if gi_labels[i].startswith("Cold Spot")]

    gi_insight_parts = []
    if hot_areas:
        hot_list = ", ".join(f"**{a}**" for a in hot_areas[:5])
        suffix = f" and {len(hot_areas) - 5} more" if len(hot_areas) > 5 else ""
        gi_insight_parts.append(
            f"**Hot Spot Clusters ({n_hot} areas):** {hot_list}{suffix}. "
            f"These form statistically significant clusters of high {value_label}. "
            "Prioritize these areas for immediate resource allocation. "
            "The clustering pattern means interventions in one area will likely produce "
            "spillover benefits in adjacent areas."
        )
    else:
        gi_insight_parts.append(
            f"No statistically significant hot spots were detected. Elevated {value_label} values "
            "do not form spatial clusters, suggesting city-wide rather than place-based strategies."
        )

    if cold_areas:
        cold_list = ", ".join(f"**{a}**" for a in cold_areas[:5])
        suffix = f" and {len(cold_areas) - 5} more" if len(cold_areas) > 5 else ""
        gi_insight_parts.append(
            f"**Cold Spot Clusters ({n_cold} areas):** {cold_list}{suffix}. "
            f"These form clusters of consistently low {value_label}. "
            "Investigate whether structural barriers, historical disinvestment, or administrative "
            "boundaries explain the pattern."
        )

    if hot_areas and cold_areas:
        gi_insight_parts.append(
            f"**Spatial Strategy:** The presence of both hot and cold spots suggests a spatially bifurcated "
            f"{value_label} pattern. A corridor approach connecting adjacent hot and cold spot zones through "
            "shared infrastructure may help diffuse concentration and lift cold spot areas."
        )
    elif not hot_areas and not cold_areas:
        gi_insight_parts.append(
            f"The absence of significant spatial clustering suggests that {value_label} is relatively "
            "evenly distributed. Uniform city-wide policies are well-justified."
        )

    st.info(" ".join(gi_insight_parts))

    # ── 8. LISA vs Gi* Concordance ──────────────────────────────────────
    n_both_hot = sum(
        1 for i in range(n_total)
        if lisa_labels_list[i] == "HH (Hot Spot)" and gi_labels[i].startswith("Hot Spot")
    )
    n_both_cold = sum(
        1 for i in range(n_total)
        if lisa_labels_list[i] == "LL (Cold Spot)" and gi_labels[i].startswith("Cold Spot")
    )
    n_agree = n_both_hot + n_both_cold + sum(
        1 for i in range(n_total)
        if lisa_labels_list[i] == "Not Significant" and gi_labels[i] == "Not Significant"
    )
    concordance_pct = n_agree / n_total * 100 if n_total else 0

    concordance_areas = []
    for i in range(n_total):
        if lisa_labels_list[i] == "HH (Hot Spot)" and gi_labels[i].startswith("Hot Spot"):
            concordance_areas.append(f"{names[i]} (hot)")
        elif lisa_labels_list[i] == "LL (Cold Spot)" and gi_labels[i].startswith("Cold Spot"):
            concordance_areas.append(f"{names[i]} (cold)")

    concordance_text = (
        f"**LISA vs Gi* concordance: {concordance_pct:.1f}%** of areas receive the same classification "
        "from both methods. "
    )
    if concordance_areas:
        areas_str = ", ".join(f"**{a}**" for a in concordance_areas[:8])
        suffix = f" and {len(concordance_areas) - 8} more" if len(concordance_areas) > 8 else ""
        concordance_text += (
            f"Areas where both methods agree ({areas_str}{suffix}) "
            "are the highest-confidence targets for intervention."
        )
    else:
        concordance_text += "No areas are flagged as significant clusters by both methods."

    st.caption(concordance_text)
