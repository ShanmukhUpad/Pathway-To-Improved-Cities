import importlib.util
import os
import sys
import requests
import streamlit as st
import pandas as pd
import plotly.express as px

import map_utils
import file_loader

st.set_page_config(
    page_title="Pathway to Improved Cities",
    layout="wide",
)

map_utils.init_mapbox()

# ── City registry ─────────────────────────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

CITIES = {
    "Chicago":        "chicago",
    "New York City":  "new_york",
    "Los Angeles":    "los_angeles",
    "San Francisco":  "san_francisco",
    "Austin":         "austin",
    "Seattle":        "seattle",
}

# Module names that live inside each city's subdirectory
_CITY_MODULES = [
    "data_fetcher",
    "public_safety",
    "crash",
    "socieoeconomic",
    "transportation_access",
]


def _load_city_modules(city_key: str) -> None:
    """
    Load city-specific modules and register them under their short names in
    sys.modules.  Loading data_fetcher first ensures that when public_safety
    does `import data_fetcher` it resolves to the correct city's copy.
    """
    city_dir = os.path.join(SRC_DIR, city_key)

    # Remove previously loaded city modules
    for name in _CITY_MODULES:
        sys.modules.pop(name, None)

    # Ensure only the current city's dir is prepended to sys.path
    for c in CITIES.values():
        d = os.path.join(SRC_DIR, c)
        while d in sys.path:
            sys.path.remove(d)
    sys.path.insert(0, city_dir)

    # Load in dependency order: data_fetcher before the render modules
    for name in _CITY_MODULES:
        fp = os.path.join(city_dir, f"{name}.py")
        if not os.path.exists(fp):
            continue
        spec = importlib.util.spec_from_file_location(name, fp)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod          # pre-register so intra-city imports resolve
        spec.loader.exec_module(mod)
        mod._CITY_KEY = city_key         # tag for cache-hit detection


# ── City selector ─────────────────────────────────────────────────────────────

st.title("Pathway to Improved Cities Dashboard")

selected_label = st.selectbox(
    "Select City",
    list(CITIES.keys()),
    index=0,
    key="city_selector",
)
city_key = CITIES[selected_label]

# Reload modules when city changes (or on first load)
_prev_city = st.session_state.get("loaded_city")
if _prev_city != city_key:
    if _prev_city is not None:          # actual city change — clear stale caches
        st.cache_data.clear()
        st.cache_resource.clear()
    _load_city_modules(city_key)
    st.session_state.loaded_city = city_key
    if _prev_city is not None:
        st.rerun()

# By this point the city modules are in sys.modules under their short names
import data_fetcher
import public_safety
import crash
import socieoeconomic
import transportation_access

# ── Sidebar: data refresh ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("Data")

    crime_ts = data_fetcher.last_updated(data_fetcher.CRIME_OUT)
    crash_ts = data_fetcher.last_updated(data_fetcher.CRASH_OUT)
    st.markdown(
        f"**Crime data** — last updated: `{crime_ts}`\n\n"
        f"**Crash data** — last updated: `{crash_ts}`"
    )

    portal_label = {
        "chicago":       "Chicago Data Portal",
        "new_york":      "NYC Open Data",
        "los_angeles":   "LA Open Data",
        "san_francisco": "SF Open Data",
        "austin":        "Austin Open Data",
        "seattle":       "Seattle Open Data",
    }.get(city_key, "Open Data Portal")

    if st.button(f"Refresh from {portal_label}", width="stretch"):
        with st.spinner("Fetching latest data…"):
            try:
                data_fetcher.refresh_all(force=True)
                st.cache_data.clear()
                st.success("Data refreshed!")
                st.rerun()
            except Exception as exc:
                st.error(f"Refresh failed: {exc}")

    st.caption(f"Selected city: **{selected_label}**")


# ── Chicago-specific shared data (GeoJSON + area map) ─────────────────────────

chicago_geo = None
area_map    = {}

if city_key == "chicago":
    community_area_names = {
        1: "Rogers Park", 2: "West Ridge", 3: "Uptown", 4: "Lincoln Square",
        5: "North Center", 6: "Lake View", 7: "Lincoln Park", 8: "Near North Side",
        9: "Edison Park", 10: "Norwood Park", 11: "Jefferson Park", 12: "Forest Glen",
        13: "North Park", 14: "Albany Park", 15: "Portage Park", 16: "Irving Park",
        17: "Dunning", 18: "Montclare", 19: "Belmont Cragin", 20: "Hermosa",
        21: "Avondale", 22: "Logan Square", 23: "Humboldt Park", 24: "West Town",
        25: "Austin", 26: "West Garfield Park", 27: "East Garfield Park", 28: "Near West Side",
        29: "North Lawndale", 30: "South Lawndale", 31: "Lower West Side", 32: "Loop",
        33: "Near South Side", 34: "Armour Square", 35: "Douglas", 36: "Oakland",
        37: "Fuller Park", 38: "Grand Boulevard", 39: "Kenwood", 40: "Washington Park",
        41: "Hyde Park", 42: "Woodlawn", 43: "South Shore", 44: "Chatham",
        45: "Avalon Park", 46: "South Chicago", 47: "Burnside", 48: "Calumet Heights",
        49: "Roseland", 50: "Pullman", 51: "South Deering", 52: "East Side",
        53: "West Pullman", 54: "Riverdale", 55: "Hegewisch", 56: "Garfield Ridge",
        57: "Archer Heights", 58: "Brighton Park", 59: "McKinley Park", 60: "Bridgeport",
        61: "New City", 62: "West Elsdon", 63: "Gage Park", 64: "Clearing",
        65: "West Lawn", 66: "Chicago Lawn", 67: "West Englewood", 68: "Englewood",
        69: "Greater Grand Crossing", 70: "Ashburn", 71: "Auburn Gresham", 72: "Beverly",
        73: "Washington Heights", 74: "Mount Greenwood", 75: "Morgan Park",
        76: "O'Hare", 77: "Edgewater",
    }

    @st.cache_data
    def _load_chicago_geojson():
        url = "https://raw.githubusercontent.com/RandomFractals/ChicagoCrimes/master/data/chicago-community-areas.geojson"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()

    chicago_geo = _load_chicago_geojson()
    area_map    = {
        int(f["properties"]["area_num_1"]): f["properties"]["community"]
        for f in chicago_geo["features"]
    }


# ── Tab layout ────────────────────────────────────────────────────────────────

tab_safety, tab_transport, tab_infra, tab_socio = st.tabs([
    "Public Safety",
    "Transportation",
    "Infrastructure",
    "Socioeconomics & Diversity",
])


# ══════════════════════════════════════════════
# TAB 1 — PUBLIC SAFETY
# ══════════════════════════════════════════════

with tab_safety:
    if city_key == "chicago":
        public_safety.render(chicago_geo=chicago_geo, area_map=area_map)
    else:
        public_safety.render()


# ══════════════════════════════════════════════
# TAB 2 — TRANSPORTATION
# ══════════════════════════════════════════════

with tab_transport:
    if city_key == "chicago":
        crash.render(chicago_geo=chicago_geo)
    else:
        crash.render()
    st.divider()
    transportation_access.render()


# ══════════════════════════════════════════════
# TAB 3 — INFRASTRUCTURE
# ══════════════════════════════════════════════

with tab_infra:
    st.header("Infrastructure Dashboard")
    st.markdown(
        f"Track infrastructure quality, 311 service requests, building permits, "
        f"and public facility conditions for **{selected_label}**."
    )
    with st.expander("Upload a supplemental dataset"):
        file_loader.uploader(domain="infrastructure", local_csv=None, label="Upload an infrastructure dataset")
    st.info(
        "No infrastructure dataset loaded yet.\n\n"
        "**To connect data:** Place a file named `infrastructure_monthly.csv` in the city's directory.\n\n"
        "**Expected CSV columns:** `Community Area`, `Year`, `Month`, + metric columns"
    )


# ══════════════════════════════════════════════
# TAB 4 — SOCIOECONOMICS & DIVERSITY
# ══════════════════════════════════════════════

with tab_socio:
    socieoeconomic.render()
