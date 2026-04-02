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
from esda.moran import Moran, Moran_Local
from libpysal.weights import Queen, KNN

# ── Mapbox token ─────────────────────────────────────────────────────────────

MAPBOX_TOKEN = os.environ.get("PUBLIC_MAPBOX_TOKEN", "")

MAPBOX_STYLES = {
    "Streets":   "streets-v12",
    "Light":     "light-v11",
    "Dark":      "dark-v11",
    "Satellite": "satellite-streets-v12",
}


def init_mapbox():
    """Set the Plotly global Mapbox access token. Call once at app startup."""
    if MAPBOX_TOKEN:
        px.set_mapbox_access_token(MAPBOX_TOKEN)


def mapbox_style_picker(key_prefix: str = "map") -> str:
    """
    Render a selectbox for Mapbox styles and return the chosen style string.
    Falls back to 'open-street-map' if no token is set.
    """
    if not MAPBOX_TOKEN:
        return "open-street-map"

    choice = st.selectbox(
        "Map style",
        list(MAPBOX_STYLES.keys()),
        index=0,
        key=f"{key_prefix}_mapbox_style",
    )
    return MAPBOX_STYLES[choice]


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
def _compute_moran(_gdf_json: str, value_col: str):
    """
    Cached computation of global + local Moran's I.
    Accepts GeoJSON string (hashable for caching) instead of GeoDataFrame.
    Returns dict with all needed results.
    """
    gdf = gpd.read_file(_gdf_json, driver="GeoJSON")
    y = gdf[value_col].values.astype(float)
    w = _build_weights(gdf)

    # Global
    moran_global = Moran(y, w, permutations=999)

    # Local (LISA)
    moran_local = Moran_Local(y, w, permutations=999)

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

    return {
        "I": round(moran_global.I, 4),
        "p_value": round(moran_global.p_sim, 4),
        "z_score": round(moran_global.z_sim, 4),
        "lisa_labels": labels,
        "z": z.tolist(),
        "lag": lag.tolist(),
    }


def render_moran_analysis(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    name_col: str,
    id_col: str,
    geojson: dict,
    featureidkey: str,
    key_prefix: str,
    mapbox_style: str = "streets-v12",
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
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── 3. LISA cluster map ──────────────────────────────────────────────
    with col_map:
        lisa_df = pd.DataFrame({
            id_col: valid[id_col].values,
            "LISA Cluster": result["lisa_labels"],
            name_col: valid[name_col].values,
            value_col: valid[value_col].values,
        })

        fig_lisa = px.choropleth_mapbox(
            lisa_df, geojson=geojson,
            locations=id_col, featureidkey=featureidkey,
            color="LISA Cluster",
            color_discrete_map=_LISA_COLORS,
            category_orders={"LISA Cluster": [
                "HH (Hot Spot)", "HL", "LH", "LL (Cold Spot)", "Not Significant"
            ]},
            mapbox_style=mapbox_style,
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
        st.plotly_chart(fig_lisa, use_container_width=True)
