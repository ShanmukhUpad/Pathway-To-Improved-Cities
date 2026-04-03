import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import crash
import file_loader
import public_safety
import socieoeconomic
import data_fetcher
import transportation_access
import map_utils

st.set_page_config(
    page_title="Pathway to Improved Cities",
    layout="wide"
)

map_utils.init_mapbox()

st.title("Pathway to Improved Cities Dashboard")

# ──────────────────────────────────────────────
# Sidebar: live data refresh
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("Data")

    st.markdown(
        f"**Crime data** — last updated: `{data_fetcher.last_updated(data_fetcher.CRIME_OUT)}`\n\n"
        f"**Crash data** — last updated: `{data_fetcher.last_updated(data_fetcher.CRASH_OUT)}`"
    )

    if st.button("Refresh from Chicago Data Portal", width="stretch"):
        with st.spinner("Fetching latest data…"):
            try:
                data_fetcher.refresh_all(force=True)
                st.cache_data.clear()
                st.success("Data refreshed!")
                st.rerun()
            except Exception as exc:
                st.error(f"Refresh failed: {exc}")

    st.caption(
        "Data source: [Chicago Data Portal](https://data.cityofchicago.org)  \n"
        "Set `CHICAGO_DATA_PORTAL_TOKEN` env var for higher rate limits."
    )


# ──────────────────────────────────────────────
# Shared: Community Area mapping + GeoJSON
# ──────────────────────────────────────────────

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
    76: "O'Hare", 77: "Edgewater"
}

@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/RandomFractals/ChicagoCrimes/master/data/chicago-community-areas.geojson"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

chicago_geo = load_geojson()

area_map = {
    int(f['properties']['area_num_1']): f['properties']['community']
    for f in chicago_geo['features']
}

# ──────────────────────────────────────────────
# Tab layout
# ──────────────────────────────────────────────

tab_safety, tab_transport, tab_infra, tab_socio = st.tabs([
    "Public Safety",
    "Transportation",
    "Infrastructure",
    "Socioeconomics & Diversity"
])


# ══════════════════════════════════════════════
# TAB 1 — PUBLIC SAFETY
# ══════════════════════════════════════════════

with tab_safety:
    public_safety.render(chicago_geo=chicago_geo, area_map=area_map)


# ══════════════════════════════════════════════
# TAB 2 — TRANSPORTATION
# ══════════════════════════════════════════════

with tab_transport:
    crash.render(chicago_geo=chicago_geo)
    st.divider()
    transportation_access.render()


# ══════════════════════════════════════════════
# TAB 3 — INFRASTRUCTURE
# ══════════════════════════════════════════════

with tab_infra:
    st.header("Infrastructure Dashboard")
    st.markdown(
        "Track infrastructure quality, 311 service requests, building permits, and public facility conditions."
    )

    with st.expander("Upload a supplemental dataset"):
        file_loader.uploader(domain="infrastructure", local_csv=None, label="Upload an infrastructure dataset")

    st.info(
        "No infrastructure dataset loaded yet.\n\n"
        "**To connect data:** Place a file named `infrastructure_monthly.csv` in the working directory.\n\n"
        "**Suggested datasets (Chicago Data Portal):**\n"
        "- [311 Service Requests](https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy)\n"
        "- [Building Permits](https://data.cityofchicago.org/Buildings/Building-Permits/ydr8-5enu)\n"
        "- [Street Lights - All Out](https://data.cityofchicago.org/Service-Requests/311-Service-Requests-Street-Lights-All-Out/zuxi-7xem)\n"
        "- [Pothole Repairs](https://data.cityofchicago.org/Service-Requests/311-Service-Requests-Pot-Holes-Reported/7as2-ds3y)\n\n"
        "**Expected CSV columns:** `Community Area`, `Year`, `Month`, + metric columns"
    )


# ══════════════════════════════════════════════
# TAB 4 — SOCIOECONOMICS & DIVERSITY
# ══════════════════════════════════════════════

with tab_socio:
    socieoeconomic.render()
    