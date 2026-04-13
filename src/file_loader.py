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


def _extract_latlon_from_gdf(gdf):
    """
    Reproject a GeoDataFrame to WGS 84 and extract centroid lat/lon columns.
    Returns (DataFrame with lat/lon, original GeoDataFrame reprojected to 4326).
    """
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif not gdf.crs.equals("EPSG:4326"):
        gdf = gdf.to_crs(epsg=4326)
    centroids = gdf.geometry.centroid
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    upper_cols = {c.upper().strip() for c in df.columns}
    if not upper_cols & _LAT_ALIASES:
        df["LATITUDE"] = centroids.y
    if not upper_cols & _LON_ALIASES:
        df["LONGITUDE"] = centroids.x
    return df, gdf


# ── File readers ──────────────────────────────────────────────────────────────

def _read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, low_memory=False)
            return df, None, None
        elif name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
            return df, None, None
        elif name.endswith(".geojson"):
            raw = uploaded_file.read()
            gdf = gpd.read_file(io.BytesIO(raw))
            df, gdf = _extract_latlon_from_gdf(gdf)
            return df, gdf, None
        else:
            return None, None, f"Unsupported file type: `{uploaded_file.name}`"
    except Exception as e:
        return None, None, f"Could not read `{uploaded_file.name}`: {e}"


def _read_shapefile(uploaded_files):
    required_exts = {".shp", ".shx", ".dbf"}
    names_by_ext = {os.path.splitext(f.name)[1].lower(): f for f in uploaded_files}
    missing = required_exts - set(names_by_ext.keys())
    if missing:
        return None, None, (
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
            df, gdf = _extract_latlon_from_gdf(gdf)
        return df, gdf, None
    except Exception as e:
        return None, None, f"Could not read shapefile: {e}"


# ── Validation ────────────────────────────────────────────────────────────────

def _validate(df):
    """
    Returns (df, is_valid, lat_col, lon_col, has_latlon).
    Requirements: at least one numeric column.
    Lat/lon is detected but not required (spatial files provide it via geometry).
    """
    df, lat_col, lon_col = _find_latlon(df)
    has_latlon = lat_col is not None and lon_col is not None
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    is_valid = len(numeric_cols) > 0
    return df, is_valid, lat_col, lon_col, has_latlon


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

def _render_map(df, lat_col, lon_col, attr, domain, map_style="open-street-map", gdf=None):
    """Render a choropleth (preferred) or scatter map as fallback."""

    # 1. Choropleth from polygon geometry (GeoJSON / Shapefile uploads)
    has_polygons = (
        gdf is not None
        and hasattr(gdf, "geom_type")
        and gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all()
    )
    if has_polygons:
        try:
            name_col = _find_name_col(df)
            gdf_plot = gdf.copy()
            gdf_plot[attr] = df[attr].values
            gdf_plot["_id"] = range(len(gdf_plot))
            gdf_plot["_id_str"] = gdf_plot["_id"].astype(str)
            if name_col:
                gdf_plot["_name"] = df[name_col].values
            else:
                gdf_plot["_name"] = [f"Area {i+1}" for i in range(len(gdf_plot))]

            gdf_plot = gdf_plot.dropna(subset=[attr])
            geojson_dict = gdf_plot.__geo_interface__

            # Auto-center on the data
            bounds = gdf_plot.total_bounds  # [minx, miny, maxx, maxy]
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2

            fig = px.choropleth_map(
                gdf_plot, geojson=geojson_dict,
                locations="_id_str", featureidkey="properties._id",
                color=attr, color_continuous_scale="Viridis",
                map_style=map_style, zoom=9,
                center={"lat": center_lat, "lon": center_lon},
                opacity=0.7, hover_name="_name",
                hover_data={attr: True, "_id_str": False, "_id": False},
                labels={attr: attr}, title=f"{attr} - uploaded dataset",
            )
            fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=550)
            st.plotly_chart(fig, width="stretch")
            return
        except Exception as exc:
            st.warning(f"Choropleth from geometry failed ({exc}). Trying fallback.")

    # 2. Choropleth via Chicago community area column
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
            fig = px.choropleth_map(
                plot_df, geojson=geojson, locations=ca_col,
                featureidkey="properties.area_numbe",
                color=attr, color_continuous_scale="Viridis",
                map_style=map_style, zoom=8.5,
                center={"lat": 41.8358, "lon": -87.6877}, opacity=0.7,
                labels={attr: attr}, title=f"{attr} - uploaded dataset",
            )
            fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
            st.plotly_chart(fig, width="stretch")
            return
        except Exception as exc:
            st.warning(f"Choropleth failed ({exc}). Falling back to scatter map.")

    # 3. Scatter map from lat/lon (last resort)
    if not lat_col or not lon_col:
        st.info("No lat/lon columns found. Cannot render a map.")
        return

    plot_df = df[[lat_col, lon_col, attr]].dropna()
    if plot_df.empty:
        st.warning("No rows with valid lat/lon and attribute values.")
        return

    fig = px.scatter_map(
        plot_df, lat=lat_col, lon=lon_col, color=attr,
        color_continuous_scale="Viridis", map_style=map_style,
        zoom=10, center={"lat": plot_df[lat_col].median(), "lon": plot_df[lon_col].median()},
        opacity=0.6, labels={attr: attr}, title=f"{attr} - uploaded dataset",
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig, width="stretch")


# ── Full upload analysis pipeline ─────────────────────────────────────────────

def _render_upload_analysis(df, lat_col, lon_col, domain, gdf=None):
    """
    Unified analysis suite rendered for any uploaded dataset regardless of tab:
      1. Map (choropleth or scatter)
      2. Distribution chart + statistics
      3. Dynamic summary insight
      4. Top / bottom 5 rows by selected attribute
      5. ML predictor
      6. Spatial autocorrelation (if polygon geometry available)
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
    _render_map(df, lat_col, lon_col, attr, domain, map_style=upload_mapbox_style, gdf=gdf)

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
        st.plotly_chart(fig_hist, width="stretch")

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
            width="stretch",
        )
    with col_bot:
        st.markdown(f"**Bottom 5 by {attr}**")
        st.dataframe(
            df.nsmallest(5, attr)[display_cols].reset_index(drop=True),
            width="stretch",
        )

    # ── 5. ML predictor ───────────────────────────────────────────────────────
    ml_predictor.render_predictor(
        df,
        key_prefix=f"upload_{domain}",
        default_target=attr,
        default_features=numeric_cols[:min(8, len(numeric_cols))],
    )

    # ── 6. Spatial Autocorrelation ──────────────────────────────────────────
    st.divider()
    has_polygons = (
        gdf is not None
        and hasattr(gdf, "geom_type")
        and gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all()
    )
    if has_polygons:
        name_col = _find_name_col(df)
        gdf_for_moran = gdf.copy()
        gdf_for_moran[attr] = df[attr].values
        if name_col:
            gdf_for_moran["_name"] = df[name_col].values
        else:
            gdf_for_moran["_name"] = [f"Area {i+1}" for i in range(len(gdf))]
        gdf_for_moran["_id"] = range(len(gdf_for_moran))
        gdf_for_moran["_id_str"] = gdf_for_moran["_id"].astype(str)

        valid_moran = gdf_for_moran.dropna(subset=[attr])
        if len(valid_moran) >= 10:
            geojson_dict = valid_moran.__geo_interface__
            map_utils.render_moran_analysis(
                gdf=valid_moran,
                value_col=attr,
                name_col="_name",
                id_col="_id_str",
                geojson=geojson_dict,
                featureidkey="properties._id",
                key_prefix=f"upload_{domain}_moran",
                map_style=upload_mapbox_style,
            )
        else:
            st.info("Need at least 10 areas with valid data to run spatial autocorrelation.")
    else:
        st.info(
            "**Spatial autocorrelation not available for this dataset.** "
            "Upload a GeoJSON or Shapefile with polygon geometries to enable "
            "Global Moran's I, Local Moran's I (LISA), and Getis-Ord Gi* analysis."
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
        "CSVs require latitude & longitude columns for mapping. "
        "GeoJSON and Shapefiles carry their own geometry and CRS (any projection is auto-converted)."
    )

    uploaded = st.file_uploader(
        label,
        type=ACCEPTED_TYPES,
        accept_multiple_files=True,
        key=f"uploader_{domain}",
        label_visibility="collapsed",
    )

    df = None
    gdf = None
    source = None

    if uploaded:
        exts = {os.path.splitext(f.name)[1].lower() for f in uploaded}
        is_shapefile_upload = ".shp" in exts

        if is_shapefile_upload:
            df, gdf, err = _read_shapefile(uploaded)
        elif len(uploaded) == 1:
            df, gdf, err = _read_uploaded_file(uploaded[0])
        else:
            df, gdf, err = _read_uploaded_file(uploaded[0])
            st.warning(
                f"Multiple files detected. Only `{uploaded[0].name}` was loaded. "
                "For shapefiles, ensure you include the .shp file along with companions."
            )

        if err:
            st.error(err)
            return None, None

        df, is_valid, lat_col, lon_col, has_latlon = _validate(df)

        if not is_valid:
            st.error(
                "This dataset has **no numeric columns** to analyze. "
                "Upload a file with at least one numeric attribute."
            )
            return None, None

        has_polygons = (
            gdf is not None
            and hasattr(gdf, "geom_type")
            and gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all()
        )

        if has_latlon:
            derived = lat_col == "_latitude"
            latlon_note = (
                "Lat/lon derived from bounding box centroids."
                if derived else
                (f"Lat/lon derived from geometry."
                 if gdf is not None else
                 f"Lat/lon: `{lat_col}` / `{lon_col}`.")
            )
        else:
            latlon_note = "No lat/lon columns — map will use community area choropleth if available."

        spatial_note = (
            " Spatial autocorrelation (Moran's I, Gi*) enabled."
            if has_polygons else ""
        )

        st.success(
            f"File loaded — {len(df):,} rows, {len(df.columns)} columns. "
            f"{latlon_note}{spatial_note}"
        )
        source = "upload"

        _render_upload_analysis(df, lat_col, lon_col, domain, gdf=gdf)

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
    st.info(
        "No data loaded yet. Upload a file above.\n\n"
        "**CSV / Parquet:** need latitude & longitude columns for mapping.\n\n"
        "**GeoJSON / Shapefile:** geometry and CRS are read automatically (any projection supported). "
        "Polygon data will also enable spatial autocorrelation analysis."
    )
    return None, None
