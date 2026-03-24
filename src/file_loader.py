import io
import tempfile
import os
import streamlit as st
import pandas as pd
import geopandas as gpd

MIN_MATCH_COUNT = 3  # moderate threshold

# ── Domain column signatures ─────────────────────────────────────────────────
# Each tab registers the columns it knows about. Uploaded files must match
# at least MIN_MATCH_COUNT of these to be accepted for that tab.

DOMAIN_COLUMNS = {
    "transportation": {
        "CRASH_DATE", "CRASH_HOUR", "CRASH_DAY_OF_WEEK", "CRASH_MONTH",
        "WEATHER_CONDITION", "LIGHTING_CONDITION", "ROADWAY_SURFACE_COND",
        "ROAD_DEFECT", "ALIGNMENT", "TRAFFICWAY_TYPE", "LANE_CNT",
        "POSTED_SPEED_LIMIT", "TRAFFIC_CONTROL_DEVICE", "DEVICE_CONDITION",
        "INTERSECTION_RELATED_I", "FIRST_CRASH_TYPE", "CRASH_TYPE",
        "DAMAGE", "NUM_UNITS", "HIT_AND_RUN_I", "LATITUDE", "LONGITUDE",
        "CRASH_RECORD_ID", "CRASH_DATE_EST_I",
    },
    "public_safety": {
        "Community Area", "Year", "Month", "IUCR", "Primary Type",
        "Description", "Location Description", "Arrest", "Domestic",
        "Beat", "District", "Ward", "Community Area", "FBI Code",
        "Latitude", "Longitude",
    },
    "infrastructure": {
        "Community Area", "Year", "Month", "SR_NUMBER", "SR_TYPE",
        "SR_SHORT_CODE", "OWNER_DEPARTMENT", "STATUS", "ORIGIN",
        "CREATED_DATE", "LAST_MODIFIED_DATE", "CLOSED_DATE",
        "STREET_ADDRESS", "CITY", "STATE", "ZIP_CODE",
        "WARD", "POLICE_DISTRICT", "LATITUDE", "LONGITUDE",
    },
    "socioeconomics": {
        "Community Area", "PERCENT OF HOUSING CROWDED",
        "PERCENT HOUSEHOLDS BELOW POVERTY", "PERCENT AGED 16+ UNEMPLOYED",
        "PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA",
        "PERCENT AGED UNDER 18 OR OVER 64", "PER CAPITA INCOME",
        "HARDSHIP INDEX", "Median Income", "Poverty Rate",
        "Population", "White", "Black", "Hispanic", "Asian",
    },
}


def _read_uploaded_file(uploaded_file):
    """
    Read a single uploaded file into a DataFrame.
    Supports CSV, Parquet, GeoJSON.
    Returns (df, error_message). On success error_message is None.
    """
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
    """
    Assemble a shapefile from its component parts uploaded together.
    Writes them to a temp directory, then reads with geopandas.
    Returns (df, error_message).
    """
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
            # Write all components with a shared stem
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


def _validate(df, domain):
    """
    Check that df has at least MIN_MATCH_COUNT columns from the domain signature.
    Returns (is_valid, matched_columns, domain_columns).
    """
    domain_cols = DOMAIN_COLUMNS.get(domain, set())
    # Case-insensitive comparison
    uploaded_upper = {c.upper().strip() for c in df.columns}
    domain_upper   = {c.upper().strip() for c in domain_cols}
    matched = uploaded_upper & domain_upper
    return len(matched) >= MIN_MATCH_COUNT, matched, domain_cols


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
        "Uploaded data is session-only and cleared on page refresh."
    )

    is_shapefile_upload = False
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
            # Multiple non-shapefile files — only accept first, warn about rest
            df, err = _read_uploaded_file(uploaded[0])
            st.warning(
                f"Multiple files detected. Only `{uploaded[0].name}` was loaded. "
                "For shapefiles, ensure you include the .shp file along with companions."
            )

        if err:
            st.error(err)
            return None, None

        valid, matched, domain_cols = _validate(df, domain)
        if not valid:
            st.error(
                f"This file does not appear to match the **{domain.replace('_', ' ').title()}** domain. "
                f"Only {len(matched)} recognized column(s) were found — at least {MIN_MATCH_COUNT} are required.\n\n"
                f"**Matched columns:** {', '.join(sorted(matched)) if matched else 'none'}\n\n"
                f"**Expected columns include:** {', '.join(sorted(list(domain_cols))[:10])}{'...' if len(domain_cols) > 10 else ''}"
            )
            return None, None

        st.success(
            f"File loaded successfully — {len(df):,} rows, {len(df.columns)} columns. "
            f"Matched {len(matched)} domain column(s)."
        )
        source = "upload"
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
        "No data loaded yet. Upload a file above, or place the default dataset in the working directory.\n\n"
        f"**This tab recognizes columns such as:** {', '.join(sorted(list(domain_cols))[:12])}{'...' if len(domain_cols) > 12 else ''}"
    )
    return None, None