"""
folium_utils.py
---------------
Zoom-switching choropleth maps using Folium/Leaflet.
At low zoom: community area polygons (77). At high zoom: census tract polygons (801).
Layer switching happens client-side via JavaScript — no Streamlit round-trips.
"""

import os
import json
import requests
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import branca.colormap as cm
from branca.element import MacroElement, Template
from streamlit_folium import st_folium
from map_utils import MAPBOX_TOKEN

# ── Paths & constants ────────────────────────────────────────────────────────

_DIR = os.path.dirname(os.path.abspath(__file__))
CENSUS_TRACT_PATH = os.path.join(_DIR, "ChicagoCensusTracts.geojson")
COMMUNITY_AREA_URL = (
    "https://raw.githubusercontent.com/RandomFractals/ChicagoCrimes/"
    "master/data/chicago-community-areas.geojson"
)

CHICAGO_CENTER = [41.8358, -87.6877]
ZOOM_START = 10
ZOOM_THRESHOLD = 11  # below → community areas; at/above → census tracts


# ── Tile layer mapping ───────────────────────────────────────────────────────

_FOLIUM_TILE_MAP = {
    "open-street-map": "OpenStreetMap",
    "streets":         "CartoDB positron",
    "light":           "CartoDB positron",
    "dark":            "CartoDB dark_matter",
    "satellite":       "OpenStreetMap",
}


def folium_tiles(mapbox_style: str) -> str:
    """Convert a Plotly-style map style name to a Folium tiles parameter."""
    if MAPBOX_TOKEN and mapbox_style == "streets":
        return (
            f"https://api.mapbox.com/styles/v1/mapbox/streets-v12/"
            f"tiles/{{z}}/{{x}}/{{y}}@2x?access_token={MAPBOX_TOKEN}"
        )
    if MAPBOX_TOKEN and mapbox_style == "satellite":
        return (
            f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/"
            f"tiles/{{z}}/{{x}}/{{y}}@2x?access_token={MAPBOX_TOKEN}"
        )
    return _FOLIUM_TILE_MAP.get(mapbox_style, "OpenStreetMap")


# ── GeoJSON loading ──────────────────────────────────────────────────────────

def _round_coords(coords, precision=6):
    """Recursively round coordinates in a GeoJSON geometry."""
    for i, item in enumerate(coords):
        if isinstance(item, (int, float)):
            coords[i] = round(item, precision)
        elif isinstance(item, list):
            _round_coords(item, precision)


@st.cache_data(show_spinner="Loading census tract boundaries...")
def load_tract_geojson():
    """Load census tract GeoJSON with coordinates rounded to 6 decimals."""
    with open(CENSUS_TRACT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for feature in data["features"]:
        _round_coords(feature["geometry"]["coordinates"])
    return data


@st.cache_resource(show_spinner="Loading census tract geometries...")
def load_tract_gdf():
    """Load census tracts as a GeoDataFrame for spatial joins."""
    gdf = gpd.read_file(CENSUS_TRACT_PATH)
    gdf["commarea"] = gdf["commarea"].astype(str)
    # Convert Timestamp columns to strings to avoid JSON serialization errors
    for col in gdf.columns:
        if pd.api.types.is_datetime64_any_dtype(gdf[col]):
            gdf[col] = gdf[col].astype(str)
    return gdf


@st.cache_data(show_spinner="Loading community area boundaries...")
def load_community_geojson():
    """Fetch community area GeoJSON from GitHub."""
    resp = requests.get(COMMUNITY_AREA_URL)
    resp.raise_for_status()
    return resp.json()


# ── Data helpers ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Aggregating to census tracts...")
def aggregate_points_to_tracts(
    lat: list, lon: list, tract_gdf_json: str
) -> dict:
    """
    Spatial-join lat/lon points to census tracts. Returns {geoid10: count}.
    Accepts tract GDF as GeoJSON string for caching.
    """
    tract_gdf = gpd.read_file(tract_gdf_json, driver="GeoJSON")
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        pts, tract_gdf[["geoid10", "geometry"]],
        how="inner", predicate="within",
    )
    counts = joined.groupby("geoid10").size()
    return counts.to_dict()


def disaggregate_to_tracts(community_data: dict, tract_geojson: dict) -> dict:
    """
    Map community-area values to child census tracts.
    community_data: {community_area_number_str: value}
    Returns: {geoid10: value}
    """
    tract_data = {}
    for feature in tract_geojson["features"]:
        props = feature["properties"]
        commarea = str(props.get("commarea", ""))
        geoid = props.get("geoid10", "")
        if commarea in community_data:
            tract_data[geoid] = community_data[commarea]
    return tract_data


# ── Zoom-switching choropleth ────────────────────────────────────────────────

def _build_colormap(values, color_scale="YlOrRd", caption=""):
    """Build a branca LinearColormap from a list of values."""
    if not values:
        return cm.LinearColormap(["#fee5d9", "#de2d26"], vmin=0, vmax=1, caption=caption)
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        vmax = vmin + 1

    scale_colors = {
        "YlOrRd":  ["#ffffb2", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"],
        "Reds":    ["#fee5d9", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"],
        "RdBu_r":  ["#2166ac", "#67a9cf", "#d1e5f0", "#fddbc7", "#ef8a62", "#b2182b"],
        "Blues":   ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"],
        "Greens":  ["#edf8e9", "#bae4b3", "#74c476", "#31a354", "#006d2c"],
    }
    colors = scale_colors.get(color_scale, scale_colors["YlOrRd"])
    return cm.LinearColormap(colors, vmin=vmin, vmax=vmax, caption=caption)


def _make_style_fn(data_dict, key_prop, colormap):
    """Return a style_function for folium.GeoJson."""
    def style_fn(feature):
        key = str(feature["properties"].get(key_prop, ""))
        val = data_dict.get(key)
        if val is not None:
            color = colormap(val)
        else:
            color = "#cccccc"
        return {
            "fillColor": color,
            "color": "#333333",
            "weight": 1,
            "fillOpacity": 0.7,
        }
    return style_fn


def _make_highlight_fn():
    """Return a highlight_function for hover effects."""
    def highlight_fn(feature):
        return {
            "weight": 3,
            "color": "#000000",
            "fillOpacity": 0.85,
        }
    return highlight_fn


class _ZoomLayerSwitch(MacroElement):
    """Folium MacroElement that toggles two FeatureGroups based on zoom level.

    References the actual layer objects so streamlit-folium's variable
    renaming resolves correctly via Jinja2's {{ this.comm_layer.get_name() }}.
    """

    _template = Template("""
        {% macro script(this, kwargs) %}
            (function () {
                var zMap   = {{ this._parent.get_name() }};
                var zComm  = {{ this.comm_layer.get_name() }};
                var zTract = {{ this.tract_layer.get_name() }};
                var zThresh = {{ this.threshold }};

                function zUpdate() {
                    var z = zMap.getZoom();
                    if (z < zThresh) {
                        if (!zMap.hasLayer(zComm))  zMap.addLayer(zComm);
                        if ( zMap.hasLayer(zTract)) zMap.removeLayer(zTract);
                    } else {
                        if (!zMap.hasLayer(zTract)) zMap.addLayer(zTract);
                        if ( zMap.hasLayer(zComm))  zMap.removeLayer(zComm);
                    }
                }

                zMap.on('zoomend', zUpdate);
                zMap.whenReady(zUpdate);
            })();
        {% endmacro %}
    """)

    def __init__(self, community_layer, tract_layer, threshold):
        super().__init__()
        self.comm_layer = community_layer
        self.tract_layer = tract_layer
        self.threshold = threshold


def create_zoom_choropleth(
    community_geojson: dict,
    tract_geojson: dict,
    community_data: dict,
    tract_data: dict,
    community_name_key: str = "community",
    community_id_key: str = "area_num_1",
    tract_name_key: str = "namelsad10",
    tract_id_key: str = "geoid10",
    value_label: str = "Value",
    color_scale: str = "YlOrRd",
    map_style: str = "open-street-map",
    height: int = 500,
) -> folium.Map:
    """
    Build a Folium map with community-area and census-tract layers
    that switch at ZOOM_THRESHOLD via client-side JavaScript.
    """
    tiles = folium_tiles(map_style)
    tile_is_url = tiles.startswith("http")

    m = folium.Map(
        location=CHICAGO_CENTER,
        zoom_start=ZOOM_START,
        tiles=tiles if not tile_is_url else None,
    )
    if tile_is_url:
        folium.TileLayer(
            tiles=tiles,
            attr="Mapbox",
            name="Mapbox",
        ).add_to(m)

    # Shared colormap across both layers
    all_vals = list(community_data.values()) + list(tract_data.values())
    all_vals = [v for v in all_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    colormap = _build_colormap(all_vals, color_scale, caption=value_label)

    # ── Community area layer ─────────────────────────────────────────────
    community_layer = folium.FeatureGroup(name="Community Areas")
    folium.GeoJson(
        community_geojson,
        style_function=_make_style_fn(community_data, community_id_key, colormap),
        highlight_function=_make_highlight_fn(),
        tooltip=folium.GeoJsonTooltip(
            fields=[community_name_key, community_id_key],
            aliases=["Area", "ID"],
            localize=True,
        ),
    ).add_to(community_layer)
    community_layer.add_to(m)

    # ── Census tract layer (hidden until zoom ≥ threshold) ──────────────
    tract_layer = folium.FeatureGroup(name="Census Tracts", show=False)
    folium.GeoJson(
        tract_geojson,
        style_function=_make_style_fn(tract_data, tract_id_key, colormap),
        highlight_function=_make_highlight_fn(),
        tooltip=folium.GeoJsonTooltip(
            fields=[tract_name_key, "commarea"],
            aliases=["Tract", "Community Area #"],
            localize=True,
        ),
    ).add_to(tract_layer)
    tract_layer.add_to(m)

    # ── Colormap legend ──────────────────────────────────────────────────
    colormap.add_to(m)

    # ── JavaScript: swap layers on zoom ──────────────────────────────────
    # Use MacroElement so the JS renders inside Folium's own script context
    # where map/layer variables are in scope.
    zoom_switch = _ZoomLayerSwitch(
        community_layer=community_layer,
        tract_layer=tract_layer,
        threshold=ZOOM_THRESHOLD,
    )
    m.add_child(zoom_switch)

    return m


def build_community_map(
    community_geojson: dict,
    community_data: dict,
    value_label: str = "Value",
    color_scale: str = "YlOrRd",
    map_style: str = "open-street-map",
    community_name_key: str = "community",
    community_id_key: str = "area_num_1",
) -> folium.Map:
    """Build a simple community-area choropleth Folium map (no JS zoom switching)."""
    tiles = folium_tiles(map_style)
    tile_is_url = tiles.startswith("http")
    m = folium.Map(
        location=CHICAGO_CENTER,
        zoom_start=ZOOM_START,
        tiles=tiles if not tile_is_url else None,
    )
    if tile_is_url:
        folium.TileLayer(tiles=tiles, attr="Mapbox", name="Mapbox").add_to(m)

    vals = [v for v in community_data.values() if v is not None and not (isinstance(v, float) and np.isnan(v))]
    colormap = _build_colormap(vals, color_scale, caption=value_label)
    folium.GeoJson(
        community_geojson,
        style_function=_make_style_fn(community_data, community_id_key, colormap),
        highlight_function=_make_highlight_fn(),
        tooltip=folium.GeoJsonTooltip(
            fields=[community_name_key, community_id_key],
            aliases=["Area", "ID"],
            localize=True,
        ),
    ).add_to(m)
    colormap.add_to(m)
    return m


def build_tract_map(
    tract_geojson: dict,
    tract_data: dict,
    value_label: str = "Value",
    color_scale: str = "YlOrRd",
    map_style: str = "open-street-map",
    tract_name_key: str = "namelsad10",
    tract_id_key: str = "geoid10",
) -> folium.Map:
    """Build a census-tract choropleth Folium map."""
    tiles = folium_tiles(map_style)
    tile_is_url = tiles.startswith("http")
    m = folium.Map(
        location=CHICAGO_CENTER,
        zoom_start=12,   # start more zoomed in for tract-level detail
        tiles=tiles if not tile_is_url else None,
    )
    if tile_is_url:
        folium.TileLayer(tiles=tiles, attr="Mapbox", name="Mapbox").add_to(m)

    vals = [v for v in tract_data.values() if v is not None and not (isinstance(v, float) and np.isnan(v))]
    colormap = _build_colormap(vals, color_scale, caption=value_label)
    folium.GeoJson(
        tract_geojson,
        style_function=_make_style_fn(tract_data, tract_id_key, colormap),
        highlight_function=_make_highlight_fn(),
        tooltip=folium.GeoJsonTooltip(
            fields=[tract_name_key, "commarea"],
            aliases=["Tract", "Community Area #"],
            localize=True,
        ),
    ).add_to(m)
    colormap.add_to(m)
    return m


def render_community_map(
    community_geojson: dict,
    community_data: dict,
    value_label: str = "Value",
    color_scale: str = "YlOrRd",
    map_style: str = "open-street-map",
    community_name_key: str = "community",
    community_id_key: str = "area_num_1",
    key: str = "folium_comm_map",
    height: int = 500,
):
    """Render community-area choropleth map in Streamlit."""
    m = build_community_map(
        community_geojson=community_geojson,
        community_data=community_data,
        value_label=value_label,
        color_scale=color_scale,
        map_style=map_style,
        community_name_key=community_name_key,
        community_id_key=community_id_key,
    )
    st_folium(m, use_container_width=True, height=height, key=key, returned_objects=[])


def render_tract_map(
    tract_geojson: dict,
    tract_data: dict,
    value_label: str = "Value",
    color_scale: str = "YlOrRd",
    map_style: str = "open-street-map",
    tract_name_key: str = "namelsad10",
    tract_id_key: str = "geoid10",
    key: str = "folium_tract_map",
    height: int = 500,
):
    """Render census-tract choropleth map in Streamlit."""
    m = build_tract_map(
        tract_geojson=tract_geojson,
        tract_data=tract_data,
        value_label=value_label,
        color_scale=color_scale,
        map_style=map_style,
        tract_name_key=tract_name_key,
        tract_id_key=tract_id_key,
    )
    st_folium(m, use_container_width=True, height=height, key=key, returned_objects=[])


def render_zoom_map(
    community_geojson: dict,
    tract_geojson: dict,
    community_data: dict,
    tract_data: dict,
    value_label: str = "Value",
    color_scale: str = "YlOrRd",
    map_style: str = "open-street-map",
    community_name_key: str = "community",
    community_id_key: str = "area_num_1",
    key: str = "folium_zoom_map",
    height: int = 500,
):
    """Render a zoom-switching choropleth in Streamlit."""
    m = create_zoom_choropleth(
        community_geojson=community_geojson,
        tract_geojson=tract_geojson,
        community_data=community_data,
        tract_data=tract_data,
        community_name_key=community_name_key,
        community_id_key=community_id_key,
        value_label=value_label,
        color_scale=color_scale,
        map_style=map_style,
        height=height,
    )
    st_folium(
        m,
        use_container_width=True,
        height=height,
        key=key,
        returned_objects=[],
    )
