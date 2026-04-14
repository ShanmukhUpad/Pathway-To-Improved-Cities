import streamlit as st

CITY_NAME = "New York City"


def render():
    st.subheader(f"Transportation Access — {CITY_NAME}")
    st.info(
        f"Transportation access data for **{CITY_NAME}** is not yet available in this dashboard.  \n\n"
        "To add transit data, place CSV files in this city's directory with the following format:  \n"
        "- `bus_stops_clean.csv` — columns: `stop_id`, `latitude`, `longitude`, `routes`, `ward`  \n"
        "- `divvy_bicycle_clean.csv` — columns: `id`, `latitude`, `longitude`, `total_docks`, `docks_in_service`, `status`  \n"
        "- `bike_routes_clean.csv` — columns: `length`, `contraflow`, `bikeway_type`  \n\n"
        f"**Suggested public data sources for {CITY_NAME}:**  \n"
        "- NYC MTA bus stops: https://data.cityofnewyork.us  \n"
        "- Citi Bike station data: https://citibikenyc.com/system-data  \n"
        "- NYC bike infrastructure: https://data.cityofnewyork.us"
    )
