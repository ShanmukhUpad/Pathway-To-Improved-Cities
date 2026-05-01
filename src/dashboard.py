import streamlit as st

import crash
import file_loader
import public_safety
import socieoeconomic
import data_fetcher
import transportation_access
import green_infrastructure
import environment
import map_utils
from city_config import CITIES, DEFAULT_CITY_KEY, get_city, load_boundary, list_cities

st.set_page_config(
    page_title="Pathway to Improved Cities",
    layout="wide",
)

map_utils.init_mapbox()


@st.cache_resource
def _get_scheduler():
    return data_fetcher.start_scheduler()


# ── Auto-refresh on startup (once per session) ────────────────────────────────
if not st.session_state.get("_refresh_kicked_off"):
    st.session_state["_refresh_kicked_off"] = True
    data_fetcher.start_background_refresh()
    _get_scheduler()


st.title("Pathway to Improved Cities Dashboard")


# ──────────────────────────────────────────────
# City selector + per-city boundary cache
# ──────────────────────────────────────────────

@st.cache_data(show_spinner="Loading boundary…")
def _cached_boundary(city_key: str):
    return load_boundary(get_city(city_key))


# ── City + basemap selectors ──────────────────────────────────────────────────
city_keys = [k for k, _ in list_cities()]
city_labels = {k: name for k, name in list_cities()}

if "active_city" not in st.session_state:
    st.session_state["active_city"] = DEFAULT_CITY_KEY

_BASEMAPS = {
    "Streets (OpenStreetMap)": "open-street-map",
    "Light (Carto Positron)":  "carto-positron",
    "Dark (Carto Dark Matter)": "carto-darkmatter",
    "Terrain (Stamen)":        "stamen-terrain",
    "Blank white":             "white-bg",
}

_sel_col, _map_col = st.columns([1, 1])
with _sel_col:
    active_key = st.selectbox(
        "City",
        city_keys,
        format_func=lambda k: city_labels[k],
        key="active_city",
    )
with _map_col:
    _bm_label = st.selectbox(
        "Basemap",
        list(_BASEMAPS.keys()),
        index=0,
        key="basemap_label",
    )

# Store selected style in session_state so all modules can read it
st.session_state["basemap_style"] = _BASEMAPS[_bm_label]
city = get_city(active_key)

# Load this city's boundary (may be empty for cities without wired geometry)
geo, area_map = None, {}
try:
    geo, area_map = _cached_boundary(active_key)
except Exception as exc:
    st.warning(f"Boundary load failed for {city.name}: {exc}")


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("Data")
    st.markdown(f"**City:** {city.name}")
    st.markdown(
        f"**Crime CSV** — `{data_fetcher.last_updated(city.crime_path)}`\n\n"
        f"**Crash CSV** — `{data_fetcher.last_updated(city.crash_path)}`"
    )
    if data_fetcher._refresh_lock.locked():
        st.caption("Refreshing data in background…")

    st.caption(
        "Data sources: Chicago Data Portal · NYC Open Data · DataSF · LA City Data"
    )


# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────

# ── Tab layout ────────────────────────────────────────────────────────────────

tab_safety, tab_transport, tab_infra, tab_env, tab_socio, tab_upload = st.tabs([
    "Public Safety",
    "Transportation",
    "Infrastructure",
    "Environment",
    "Socioeconomics & Diversity",
    "Data Upload",
])

with tab_safety:
    public_safety.render(city=city, geo=geo, area_map=area_map)

with tab_transport:
    crash.render(city=city, geo=geo)
    st.divider()
    transportation_access.render(city=city)

with tab_infra:
    green_infrastructure.render(city=city)

with tab_env:
    environment.render(city=city, geo=geo)

with tab_socio:
    socieoeconomic.render(city=city, geo=geo)

with tab_upload:
    st.header("Data Upload & Analysis")
    st.markdown(
        "Upload any dataset for automated analysis and forecasting."
    )
    file_loader.uploader(
        domain="upload",
        local_csv=None,
        label="Upload a dataset (CSV, Parquet, GeoJSON, or Shapefile)",
    )
