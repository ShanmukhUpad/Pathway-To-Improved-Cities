import io
import tempfile
import os
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import ml_predictor
import map_utils

MIN_MATCH_COUNT = 4  # require at least 4 domain columns + lat/lon

# ── Domain column signatures ─────────────────────────────────────────────────
# Each tab registers the columns it knows about. Uploaded files must match
# at least MIN_MATCH_COUNT of these AND contain latitude/longitude.

DOMAIN_COLUMNS = {
    "transportation": {
        # Crash data
        "CRASH_DATE", "CRASH_HOUR", "CRASH_DAY_OF_WEEK", "CRASH_MONTH",
        "WEATHER_CONDITION", "LIGHTING_CONDITION", "ROADWAY_SURFACE_COND",
        "ROAD_DEFECT", "ALIGNMENT", "TRAFFICWAY_TYPE", "LANE_CNT",
        "POSTED_SPEED_LIMIT", "TRAFFIC_CONTROL_DEVICE", "DEVICE_CONDITION",
        "INTERSECTION_RELATED_I", "FIRST_CRASH_TYPE", "CRASH_TYPE",
        "DAMAGE", "NUM_UNITS", "HIT_AND_RUN_I", "CRASH_RECORD_ID", "CRASH_DATE_EST_I",
        # Transit
        "ROUTE", "ROUTE_ID", "STOP_ID", "STOP_NAME", "DIRECTION",
        "RIDERSHIP", "BOARDINGS", "ALIGHTINGS", "HEADWAY", "DELAY",
        "ON_TIME", "TRIP_ID", "SERVICE_DATE", "DAY_TYPE",
        "BUS_ROUTE", "RAIL_LINE", "STATION", "PLATFORM",
        "DEPARTURE_TIME", "ARRIVAL_TIME", "TRANSIT_TYPE",
        # Traffic
        "SPEED", "VOLUME", "VEHICLE_COUNT", "TRAFFIC_COUNT",
        "VEHICLE_TYPE", "MODE", "DISTANCE",
        # Parking violations
        "VIOLATION_TYPE", "LICENSE_PLATE", "FINE_AMOUNT", "ISSUE_DATE",
        "METER_TYPE", "PARKING_ZONE", "TICKET_NUMBER",
        # Pedestrian & bike
        "BIKE_ROUTE", "PEDESTRIAN", "SIDEWALK_CONDITION",
        # Shared
        "LATITUDE", "LONGITUDE", "COMMUNITY_AREA", "WARD", "STREET_NAME",
        "STREET_NUMBER", "DIRECTION_OF_TRAVEL", "DATE",
    },
    "public_safety": {
        # Crime (Chicago Data Portal standard columns)
        "Community Area", "Year", "Month", "IUCR", "Primary Type",
        "Description", "Location Description", "Arrest", "Domestic",
        "Beat", "District", "Ward", "FBI Code", "Case Number",
        "Date", "Block", "Updated On", "X Coordinate", "Y Coordinate",
        "Latitude", "Longitude",
        # General safety / incidents
        "OFFENSE_TYPE", "INCIDENT_DATE", "REPORTING_AREA",
        "VICTIM_AGE", "VICTIM_SEX", "VICTIM_RACE",
        "WEAPON_TYPE", "DISPOSITION", "PRECINCT",
        "CALL_TYPE", "RESPONSE_TIME", "PRIORITY",
        "FIRE_INCIDENT", "EMS_INCIDENT", "UNIT",
        "OFFENSE_CODE", "REPORTED_DATE", "CLEARED",
    },
    "infrastructure": {
        "Community Area", "Year", "Month",
        # 311 service requests
        "SR_NUMBER", "SR_TYPE", "SR_SHORT_CODE", "OWNER_DEPARTMENT",
        "STATUS", "ORIGIN", "CREATED_DATE", "LAST_MODIFIED_DATE",
        "CLOSED_DATE", "STREET_ADDRESS", "CITY", "STATE", "ZIP_CODE",
        "WARD", "POLICE_DISTRICT", "LATITUDE", "LONGITUDE",
        # Building permits
        "PERMIT_NUMBER", "PERMIT_TYPE", "APPLICATION_START_DATE",
        "ISSUE_DATE", "WORK_TYPE", "TOTAL_FEE", "CONTRACTOR",
        "BUILDING_TYPE", "PROPERTY_USE",
        # Assets / inspections
        "INSPECTION_STATUS", "REPORTED_DATE", "COMPLETION_DATE",
        "REPAIR_TYPE", "ASSET_ID", "ASSET_TYPE", "CONDITION",
        "PRIORITY", "DEPARTMENT",
    },
    "socioeconomics": {
        "community_area", "community_name",
        "percent_poverty", "PERCENT AGED 16+ UNEMPLOYED",
        "PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA",
        "PERCENT AGED UNDER 18 OR OVER 64", "PER CAPITA INCOME",
        "HARDSHIP INDEX", "Median Income", "Poverty Rate", "BELOW_POVERTY_LINE",
        # Demographics
        "Population", "White", "Black", "Hispanic", "Asian",
        "MEDIAN_AGE", "FOREIGN_BORN", "ENGLISH_ONLY", "SPEAKS_SPANISH",
        # Housing
        "TOTAL_HOUSEHOLDS", "OWNER_OCCUPIED", "RENTER_OCCUPIED",
        "VACANT_UNITS", "MEDIAN_RENT", "MEDIAN_HOME_VALUE",
        # Education / employment
        "UNEMPLOYMENT_RATE", "EDUCATION_LEVEL",
        "HIGH_SCHOOL_GRAD", "BACHELORS_DEGREE",
        # Health / benefits
        "HEALTH_INSURANCE", "SNAP_BENEFITS", "GINI_INDEX",
        # Coordinates (optional but allowed)
        "LATITUDE", "LONGITUDE",
    },
}

# ── Lat/lon detection ─────────────────────────────────────────────────────────

_LAT_ALIASES = {"LATITUDE", "LAT", "Y_COORD", "Y_COORDINATE"}
_LON_ALIASES = {"LONGITUDE", "LON", "LNG", "LONG", "X_COORD", "X_COORDINATE"}

# Bounding box column names (case-insensitive)
_BBOX_NORTH = {"NORTH", "NORTH_BOUND", "NORTH_LAT", "MAX_LAT"}
_BBOX_SOUTH = {"SOUTH", "SOUTH_BOUND", "SOUTH_LAT", "MIN_LAT"}
_BBOX_EAST  = {"EAST",  "EAST_BOUND",  "EAST_LON",  "MAX_LON", "MAX_LNG"}
_BBOX_WEST  = {"WEST",  "WEST_BOUND",  "WEST_LON",  "MIN_LON", "MIN_LNG"}


def _find_latlon(df):
    """
    Return (df, lat_col, lon_col).
    First checks for explicit lat/lon columns. If not found, checks for
    bounding box columns (NORTH/SOUTH/EAST/WEST) and derives centroid columns.
    """
    upper_map = {c.upper().strip(): c for c in df.columns}

    lat_col = next((upper_map[a] for a in _LAT_ALIASES if a in upper_map), None)
    lon_col = next((upper_map[a] for a in _LON_ALIASES if a in upper_map), None)

    if lat_col and lon_col:
        return df, lat_col, lon_col

    # Fall back to bounding box centroid
    north = next((upper_map[a] for a in _BBOX_NORTH if a in upper_map), None)
    south = next((upper_map[a] for a in _BBOX_SOUTH if a in upper_map), None)
    east  = next((upper_map[a] for a in _BBOX_EAST  if a in upper_map), None)
    west  = next((upper_map[a] for a in _BBOX_WEST  if a in upper_map), None)

    if north and south and east and west:
        df = df.copy()
        df["_latitude"]  = (pd.to_numeric(df[north], errors="coerce") +
                            pd.to_numeric(df[south], errors="coerce")) / 2
        df["_longitude"] = (pd.to_numeric(df[east],  errors="coerce") +
                            pd.to_numeric(df[west],  errors="coerce")) / 2
        return df, "_latitude", "_longitude"

    return df, None, None


# ── File readers ──────────────────────────────────────────────────────────────

def _read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, low_memory=False)
        elif name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        elif name.endswith(".geojson"):
            raw = uploaded_file.read()
            gdf = gpd.read_file(io.BytesIO(raw))
            df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        else:
            return None, f"Unsupported file type: `{uploaded_file.name}`"
        return df, None
    except Exception as e:
        return None, f"Could not read `{uploaded_file.name}`: {e}"


def _read_shapefile(uploaded_files):
    required_exts = {".shp", ".shx", ".dbf"}
    names_by_ext = {os.path.splitext(f.name)[1].lower(): f for f in uploaded_files}
    missing = required_exts - set(names_by_ext.keys())
    if missing:
        return None, (
            f"Shapefile upload is missing required components: "
            f"{', '.join(sorted(missing))}. "
            f"Please upload .shp, .shx, and .dbf together (plus .prj, .cpg if available)."
        )
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for ext, f in names_by_ext.items():
                dest = os.path.join(tmpdir, f"upload{ext}")
                with open(dest, "wb") as out:
                    out.write(f.read())
            shp_path = os.path.join(tmpdir, "upload.shp")
            gdf = gpd.read_file(shp_path)
            df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        return df, None
    except Exception as e:
        return None, f"Could not read shapefile: {e}"


# ── Validation ────────────────────────────────────────────────────────────────

def _validate(df, domain):
    """
    Returns (df, is_valid, matched_cols, domain_cols, lat_col, lon_col, missing_latlon).
    Requirements: lat+lon present AND >= MIN_MATCH_COUNT domain columns matched.
    df may be augmented with derived centroid columns.
    """
    domain_cols = DOMAIN_COLUMNS.get(domain, set())
    uploaded_upper = {c.upper().strip() for c in df.columns}
    domain_upper   = {c.upper().strip() for c in domain_cols}
    matched = uploaded_upper & domain_upper
    df, lat_col, lon_col = _find_latlon(df)
    has_latlon = lat_col is not None and lon_col is not None
    is_valid = has_latlon and len(matched) >= MIN_MATCH_COUNT
    return df, is_valid, matched, domain_cols, lat_col, lon_col, not has_latlon


# ── Community-area choropleth helpers ─────────────────────────────────────────

# Column names (upper-cased) that identify Chicago community area numbers
_CA_NUMBER_ALIASES = {
    "COMMUNITY_AREA_NUMBER", "COMMUNITY AREA NUMBER",
    "COMMUNITY_AREA", "COMAREA", "AREA_NUMBE",
}

def _find_ca_number_col(df):
    """Return the first column whose upper-stripped name is a known CA-number alias."""
    upper_map = {c.upper().strip(): c for c in df.columns}
    for alias in _CA_NUMBER_ALIASES:
        if alias in upper_map:
            return upper_map[alias]
    return None


@st.cache_data(show_spinner="Loading Chicago community area boundaries…")
def _load_community_areas_geojson():
    """
    Fetch Chicago community area polygon boundaries from the Chicago Data Portal.
    Result is cached for the Streamlit session so it is only downloaded once.
    """
    url = "https://data.cityofchicago.org/resource/igwz-8jzy.geojson"
    resp = requests.get(url, params={"$limit": 100}, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Name-column detection ─────────────────────────────────────────────────────

_NAME_ALIASES = {
    "NAME", "COMMUNITY", "COMMUNITY AREA NAME", "COMMUNITY_AREA_NAME",
    "NEIGHBORHOOD", "AREA_NAME", "WARD_NAME", "LOCATION", "STREET_NAME",
    "ADDRESS", "LABEL", "TITLE", "DESCRIPTION",
}

def _find_name_col(df):
    """Return the first likely label/name column, or None."""
    upper_map = {c.upper().strip(): c for c in df.columns}
    for alias in _NAME_ALIASES:
        if alias in upper_map:
            return upper_map[alias]
    # Fall back to the first object column
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return None


# ── Choropleth / scatter map ──────────────────────────────────────────────────

def _render_map(df, lat_col, lon_col, attr, domain, mapbox_style="open-street-map"):
    """Render a choropleth (if CA column found) or scatter map."""
    ca_col = _find_ca_number_col(df)
    if ca_col:
        try:
            geojson = _load_community_areas_geojson()
            plot_df = df[[ca_col, attr]].dropna().copy()
            plot_df[ca_col] = (
                pd.to_numeric(plot_df[ca_col], errors="coerce")
                .dropna().astype(int).astype(str)
            )
            plot_df = plot_df.groupby(ca_col, as_index=False)[attr].mean()
            fig = px.choropleth_mapbox(
                plot_df, geojson=geojson, locations=ca_col,
                featureidkey="properties.area_numbe",
                color=attr, color_continuous_scale="Viridis",
                mapbox_style=mapbox_style, zoom=8.5,
                center={"lat": 41.8358, "lon": -87.6877}, opacity=0.7,
                labels={attr: attr}, title=f"{attr} - uploaded dataset",
            )
            fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
            st.plotly_chart(fig, use_container_width=True)
            return
        except Exception as exc:
            st.warning(f"Choropleth failed ({exc}). Falling back to scatter map.")

    if not lat_col or not lon_col:
        st.info("No lat/lon columns found. Cannot render a map.")
        return

    plot_df = df[[lat_col, lon_col, attr]].dropna()
    if plot_df.empty:
        st.warning("No rows with valid lat/lon and attribute values.")
        return

    fig = px.scatter_mapbox(
        plot_df, lat=lat_col, lon=lon_col, color=attr,
        color_continuous_scale="Viridis", mapbox_style=mapbox_style,
        zoom=10, center={"lat": plot_df[lat_col].median(), "lon": plot_df[lon_col].median()},
        opacity=0.6, labels={attr: attr}, title=f"{attr} - uploaded dataset",
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)


# ── Full upload analysis pipeline ─────────────────────────────────────────────

def _render_upload_analysis(df, lat_col, lon_col, domain):
    """
    Unified analysis suite rendered for any uploaded dataset regardless of tab:
      1. Map (choropleth or scatter)
      2. Distribution chart + statistics
      3. Dynamic summary insight
      4. Top / bottom 5 rows by selected attribute
      5. ML predictor
    """
    st.markdown("---")
    st.subheader("Uploaded Dataset Analysis")
    st.caption("Session-only and will be cleared on page refresh.")

    latlon_upper = {lat_col.upper(), lon_col.upper()} if lat_col and lon_col else set()
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c.upper() not in latlon_upper
    ]
    if not numeric_cols:
        st.info("No numeric attributes found to visualize.")
        return

    attr = st.selectbox(
        "Select attribute to visualize on map",
        numeric_cols,
        key=f"upload_attr_{domain}",
    )

    upload_mapbox_style = map_utils.mapbox_style_picker(key_prefix=f"upload_{domain}")

    # ── 1. Map ────────────────────────────────────────────────────────────────
    _render_map(df, lat_col, lon_col, attr, domain, mapbox_style=upload_mapbox_style)

    st.divider()

    # ── 2. Distribution + statistics ─────────────────────────────────────────
    st.subheader(f"Distribution of {attr}")
    series = df[attr].dropna()
    mean_val   = series.mean()
    median_val = series.median()
    std_val    = series.std()
    min_val    = series.min()
    max_val    = series.max()
    skew_val   = series.skew()

    col_hist, col_stats = st.columns([3, 1])
    with col_hist:
        fig_hist = px.histogram(
            df, x=attr, nbins=30,
            title=f"Distribution of {attr}",
            color_discrete_sequence=["#4f8ef7"],
        )
        fig_hist.update_layout(margin={"t": 30})
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_stats:
        st.metric("Mean",    f"{mean_val:.2f}")
        st.metric("Median",  f"{median_val:.2f}")
        st.metric("Std Dev", f"{std_val:.2f}")
        c1, c2 = st.columns(2)
        c1.metric("Min", f"{min_val:.2f}")
        c2.metric("Max", f"{max_val:.2f}")

    # ── 3. Summary insight ────────────────────────────────────────────────────
    skew_label = (
        "right-skewed (a long tail of high values)"
        if skew_val > 0.5 else
        ("left-skewed (a long tail of low values)" if skew_val < -0.5 else "approximately symmetric")
    )
    cv = std_val / abs(mean_val) if mean_val != 0 else 0
    variability = (
        "High variability across records suggests significant inequality or heterogeneity in this attribute."
        if cv > 0.5 else
        "Values are relatively consistent across records."
    )
    mean_vs_median = ""
    if abs(mean_val - median_val) / (std_val if std_val > 0 else 1) > 0.3:
        mean_vs_median = (
            f" The mean ({mean_val:.2f}) is pulled above the median ({median_val:.2f}) by high outliers."
            if mean_val > median_val else
            f" The mean ({mean_val:.2f}) is pulled below the median ({median_val:.2f}) by low outliers."
        )
    st.info(
        f"**{attr} summary:** The average value is **{mean_val:.2f}** and the median is **{median_val:.2f}**."
        f"{mean_vs_median} "
        f"The distribution is {skew_label}. {variability}"
    )

    # ── 4. Top / bottom 5 ─────────────────────────────────────────────────────
    name_col = _find_name_col(df)
    display_cols = [name_col, attr] if name_col else [attr]
    col_top, col_bot = st.columns(2)
    with col_top:
        st.markdown(f"**Top 5 by {attr}**")
        st.dataframe(
            df.nlargest(5, attr)[display_cols].reset_index(drop=True),
            use_container_width=True,
        )
    with col_bot:
        st.markdown(f"**Bottom 5 by {attr}**")
        st.dataframe(
            df.nsmallest(5, attr)[display_cols].reset_index(drop=True),
            use_container_width=True,
        )

    # ── 5. ML predictor ───────────────────────────────────────────────────────
    ml_predictor.render_predictor(
        df,
        key_prefix=f"upload_{domain}",
        default_target=attr,
        default_features=numeric_cols[:min(8, len(numeric_cols))],
    )


# ── Public uploader ───────────────────────────────────────────────────────────

def uploader(domain: str, local_csv: str = None, label: str = "Upload a dataset"):
    """
    Render a file uploader widget for the given domain tab.

    Parameters
    ----------
    domain      : one of "transportation", "public_safety", "infrastructure", "socioeconomics"
    local_csv   : optional fallback path to a local CSV file
    label       : uploader widget label

    Returns
    -------
    df          : pd.DataFrame if data is available, else None
    source      : "upload", "local", or None
    """
    ACCEPTED_TYPES = ["csv", "parquet", "geojson", "shp", "shx", "dbf", "prj", "cpg"]

    st.markdown(f"**{label}**")
    st.caption(
        "Accepted formats: CSV, Parquet, GeoJSON, Shapefile (.shp + .shx + .dbf + companions). "
        "Uploaded data is session-only and cleared on page refresh. "
        "Dataset must include latitude & longitude columns and at least "
        f"{MIN_MATCH_COUNT} recognized domain attributes."
    )

    uploaded = st.file_uploader(
        label,
        type=ACCEPTED_TYPES,
        accept_multiple_files=True,
        key=f"uploader_{domain}",
        label_visibility="collapsed",
    )

    df = None
    source = None

    if uploaded:
        exts = {os.path.splitext(f.name)[1].lower() for f in uploaded}
        is_shapefile_upload = ".shp" in exts

        if is_shapefile_upload:
            df, err = _read_shapefile(uploaded)
        elif len(uploaded) == 1:
            df, err = _read_uploaded_file(uploaded[0])
        else:
            df, err = _read_uploaded_file(uploaded[0])
            st.warning(
                f"Multiple files detected. Only `{uploaded[0].name}` was loaded. "
                "For shapefiles, ensure you include the .shp file along with companions."
            )

        if err:
            st.error(err)
            return None, None

        df, is_valid, matched, domain_cols, lat_col, lon_col, missing_latlon = _validate(df, domain)

        if not is_valid:
            if missing_latlon:
                st.error(
                    "This dataset is missing **latitude and longitude** columns. "
                    "Both are required for upload.\n\n"
                    "Expected column names include: `latitude`, `longitude`, `lat`, `lon`, `lng`."
                )
            else:
                st.error(
                    f"This file does not appear to match the **{domain.replace('_', ' ').title()}** domain. "
                    f"Only {len(matched)} recognized column(s) were found — at least {MIN_MATCH_COUNT} are required.\n\n"
                    f"**Matched columns:** {', '.join(sorted(matched)) if matched else 'none'}\n\n"
                    f"**Expected columns include:** {', '.join(sorted(list(domain_cols))[:10])}{'...' if len(domain_cols) > 10 else ''}"
                )
            return None, None

        derived = lat_col == "_latitude"
        latlon_note = (
            "Lat/lon derived from bounding box (NORTH/SOUTH/EAST/WEST centroids)."
            if derived else
            f"Lat/lon: `{lat_col}` / `{lon_col}`."
        )
        st.success(
            f"File loaded — {len(df):,} rows, {len(df.columns)} columns. "
            f"Matched {len(matched)} domain column(s). {latlon_note}"
        )
        source = "upload"

        _render_upload_analysis(df, lat_col, lon_col, domain)

        return df, source

    # No upload — fall back to local file if available
    if local_csv:
        try:
            df = pd.read_csv(local_csv, low_memory=False)
            source = "local"
            return df, source
        except FileNotFoundError:
            pass

    # Nothing available
    domain_cols = DOMAIN_COLUMNS.get(domain, set())
    st.info(
        "No data loaded yet. Upload a file above.\n\n"
        f"**Requirements:** latitude & longitude columns + at least {MIN_MATCH_COUNT} of these domain columns: "
        f"{', '.join(sorted(list(domain_cols))[:12])}{'...' if len(domain_cols) > 12 else ''}"
    )
    return None, None
